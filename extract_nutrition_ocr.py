import cv2
import re
import os
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

# -------------------------
# PERSPECTIVE HELPERS
# -------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# -------------------------
# OCR-BASED CONTOUR DETECTION (FIXED)
# -------------------------
def detect_document_contour(image):
    results = reader.readtext(image)

    points = []

    for (bbox, text, conf) in results:
        if not bbox or len(bbox) < 4:
            continue

        for pt in bbox:
            points.append(pt)

    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.float32)

    # Convex hull of all text
    hull = cv2.convexHull(pts)

    # Fit minimum area rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)

    return np.array(box, dtype=np.float32)


# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_image(frame):
    img = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    blur = cv2.GaussianBlur(denoised, (0, 0), 15)
    normalized = cv2.divide(denoised, blur, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(normalized)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)

    return sharpened


# -------------------------
# PANEL DETECTION
# -------------------------
def detect_panel_with_ocr(image):
    results = reader.readtext(image)

    rects = []

    for (bbox, text, conf) in results:
        if not bbox or len(bbox) < 4:
            continue

        # Filter low-confidence text
        if conf < 0.1:
            continue

        pts = np.array(bbox, dtype=np.int32)

        if pts.shape != (4, 2):
            continue

        x, y, w, h = cv2.boundingRect(pts)

        # Filter tiny noise
        if w < 15 or h < 15:
            continue

        rects.append((x, y, w, h))

    if not rects:
        return image

    # --- Remove outliers (important) ---
    centers = np.array([(x + w/2, y + h/2) for (x, y, w, h) in rects])

    mean = np.mean(centers, axis=0)
    distances = np.linalg.norm(centers - mean, axis=1)

    # Keep boxes near cluster center
    threshold = np.percentile(distances, 90)
    filtered = [rects[i] for i in range(len(rects)) if distances[i] < threshold]

    if not filtered:
        filtered = rects

    # --- Compute tight bounding box ---
    xs = [r[0] for r in filtered]
    ys = [r[1] for r in filtered]
    ws = [r[0] + r[2] for r in filtered]
    hs = [r[1] + r[3] for r in filtered]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(ws)
    y_max = max(hs)

    # --- Dynamic padding (smaller) ---
    pad_x = int((x_max - x_min) * 0.03)
    pad_y = int((y_max - y_min) * 0.03)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image.shape[1], x_max + pad_x)
    y_max = min(image.shape[0], y_max + pad_y)

    return image[y_min:y_max, x_min:x_max]
# -------------------------
# ROW GROUPING
# -------------------------
def group_into_rows(results, y_threshold=15):
    rows = []
    items = []

    for (bbox, text, conf) in results:
        if not bbox or len(bbox) < 4:
            continue

        pts = np.array(bbox, dtype=np.float32)

        if pts.shape != (4, 2):
            continue

        y_center = np.mean(pts[:, 1])
        items.append((y_center, text))

    items.sort(key=lambda x: x[0])

    current_row = []
    current_y = None

    for y, text in items:
        if current_y is None:
            current_row = [text]
            current_y = y
        elif abs(y - current_y) < y_threshold:
            current_row.append(text)
        else:
            rows.append(" ".join(current_row))
            current_row = [text]
            current_y = y

    if current_row:
        rows.append(" ".join(current_row))

    return rows


# -------------------------
# NUTRIENT EXTRACTION
# -------------------------
def normalize_ocr_text(text):
    """
    Normalize OCR text to handle common character confusions.
    
    Handles OCR misrecognitions:
    - Letter 'o' confused with '0' (zero)
    - Letter 'l' or 'i' confused with '1' (one)
    - Letter 'z' confused with '2' (two)
    - Letter 'q' confused with 'g' in units
    - Digit '9' confused with 'g' in units
    - Missing spaces between value and unit
    
    Examples:
    - "1og" → "1g" (1 + letter O + g)
    - "i2g" → "12g" (letter i + 2 + g) 
    - "z50" → "250" (letter z + 50)
    - "1oo" → "100" (1 + letter O + letter O)
    - "5q" → "5g" (5 + q → 5g)
    """
    text = text.lower().replace("'", "'")
    text = re.sub(r"\s+", " ", text).strip()
    
    # 1. Fix 'o' between digit and 'g' or digit and other digits
    # "1og" → "1g" (remove middle 'o' between digit and letter)
    text = re.sub(r'(\d)o([g0-9])', r'\1\2', text)
    
    # 2. Fix other standalone 'o' → '0'
    text = re.sub(r'\bo\b', '0', text)
    
    # 3. Fix 'z' → '2' in numbers (word boundary)
    text = re.sub(r'\bz(\d)', r'2\1', text)  # "z50" → "250"
    
    # 4. Fix letter 'i' or 'l' at start of numbers → '1'
    text = re.sub(r'\b[il](\d)', r'1\1', text)  # "i2g" → "12g", "l0" → "10"
    
    # 5. Fix gram unit confusions (q/9 → g after digit)
    text = re.sub(r'(\d)\s*[q9]\b', r'\1g', text)  # "5q" → "5g", "35q" → "35g"
    
    # 6. Fix double 'o' in numeric context → '00'
    text = re.sub(r'(\d)oo(\D|$)', r'\g<1>00\2', text)  # "1oo" → "100"
    
    return text





def extract_value_candidates(line, expect_gram=False):
    """
    Extract numeric value candidates from a line with OCR confusion handling.
    
    Args:
        line: Text line to extract from
        expect_gram: True if expecting gram unit (e.g., "5g"), False for bare numbers
        
    Returns:
        List of tuples: (value_str, has_unit_indicator)
    """
    line = normalize_ocr_text(line)
    candidates = []
    
    if expect_gram:
        # Accept OCR variants where g is confused as q/9/o/0
        # Match: digit(s).digit(s) followed by optional space and g/q/9/o/0
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([gq9o0])\b", line):
            candidates.append((m.group(1), True))
        
        # If no gram-unit matches, try bare numbers
        if not candidates:
            for m in re.finditer(r"(\d+(?:\.\d+)?)", line):
                candidates.append((m.group(1), False))
    else:
        # For non-gram (e.g., calories), match bare numbers only
        for m in re.finditer(r"\b(\d{1,4})\b", line):
            candidates.append((m.group(1), False))
    
    return candidates


def is_plausible_value(value, nutrient_key):
    """Check if a numeric value is plausible for a given nutrient."""
    try:
        v = float(value)
    except ValueError:
        return False
    
    if nutrient_key == "calories":
        return 0 <= v <= 2000
    elif nutrient_key in ("fat", "carbohydrates", "protein"):
        return 0 <= v <= 200
    return True


def extract_nutrients_from_rows(rows):
    """
    Extract nutrients from OCR rows with robust OCR confusion handling.
    
    Handles common OCR character confusions and uses context-aware scoring
    to select the best value for each nutrient.
    """
    nutrients = {
        "calories": None,
        "fat": None,
        "carbohydrates": None,
        "protein": None,
    }
    
    # Normalize all rows for case-insensitive matching
    normalized_rows = [normalize_ocr_text(row) for row in rows]
    
    # Define nutrient patterns with aliases and extraction rules
    nutrient_specs = {
        "calories": {
            "keywords": ["calories", "calorie", "energy"],
            "expect_gram": False,
            "exclude": ["saturated", "trans", "fat"],
        },
        "fat": {
            "keywords": ["total fat", "fat", "totalfat"],
            "expect_gram": True,
            "exclude": ["saturated", "trans", "polyunsaturated", "monounsaturated"],
        },
        "carbohydrates": {
            "keywords": ["carbohydrate", "carbohydrates", "carb", "carbs", "totalcarb"],
            "expect_gram": True,
            "exclude": ["fiber", "sugar", "added"],
        },
        "protein": {
            "keywords": ["protein"],
            "expect_gram": True,
            "exclude": [],
        },
    }
    
    def matches_nutrient_keywords(line, keywords):
        """Check if line contains any nutrient keywords."""
        for keyword in keywords:
            if keyword in line:
                return True
        return False
    
    def has_excluded_words(line, exclude_list):
        """Check if line contains excluded words."""
        for word in exclude_list:
            if word in line:
                return True
        return False
    
    # Extract each nutrient
    for nutrient_key, spec in nutrient_specs.items():
        best_value = None
        best_score = -1e9
        
        for idx, line in enumerate(normalized_rows):
            # Check if this line mentions the nutrient
            if not matches_nutrient_keywords(line, spec["keywords"]):
                continue
            
            # Skip if line contains excluded terms
            if has_excluded_words(line, spec["exclude"]):
                continue
            
            # Try extracting from this line and nearby lines
            search_contexts = [
                (line, 0),  # Same line
            ]
            
            # Add neighboring lines for context
            if idx > 0:
                search_contexts.append((normalized_rows[idx - 1] + " " + line, 1))
            if idx + 1 < len(normalized_rows):
                search_contexts.append((line + " " + normalized_rows[idx + 1], 1))
            
            # For calories, check further away lines (common layout)
            if nutrient_key == "calories":
                for offset in (-2, 2, -3, 3):
                    j = idx + offset
                    if 0 <= j < len(normalized_rows):
                        search_contexts.append((normalized_rows[j], abs(offset)))
            
            # Extract candidates from all contexts
            for context, distance in search_contexts:
                candidates = extract_value_candidates(context, spec["expect_gram"])
                
                for value_str, has_unit in candidates:
                    # Skip implausible values
                    if not is_plausible_value(value_str, nutrient_key):
                        continue
                    
                    # For gram-based nutrients, distrust unit-less values from other lines
                    if spec["expect_gram"] and not has_unit and distance > 0:
                        continue
                    
                    # Scoring: preference for same line, unit match, plausibility
                    score = 10.0  # Base score for finding value
                    if distance == 0:
                        score += 5.0  # Same line bonus
                    if has_unit and spec["expect_gram"]:
                        score += 3.0  # Unit indicator bonus
                    score -= distance * 0.5  # Distance penalty
                    
                    # Nutrient-specific penalties
                    if nutrient_key == "calories":
                        if "%" in context or "daily" in context:
                            score -= 4.0  # Likely a percentage, not actual calories
                        if "serving" in context or "per" in context:
                            score -= 2.0
                        if "." in value_str:
                            score -= 1.0  # Calories rarely have decimals
                    
                    if score > best_score:
                        best_score = score
                        best_value = value_str
        
        nutrients[nutrient_key] = best_value
    
    return nutrients




# -------------------------
# PIPELINE
# -------------------------
def extract_text_pipeline(frame):
    contour = detect_document_contour(frame)

    if contour is not None:
        warped = perspective_transform(frame, contour)

        # DEBUG: show detected contour
        debug = frame.copy()
        cv2.polylines(debug, [contour.astype(int)], True, (0, 255, 0), 3)
        cv2.imshow("Detected Contour", debug)

    else:
        warped = detect_panel_with_ocr(frame)  # fallback

    cropped = detect_panel_with_ocr(warped)
    processed = preprocess_image(cropped)

    results = reader.readtext(processed)

    rows = group_into_rows(results)
    nutrients = extract_nutrients_from_rows(rows)

    return rows, nutrients, cropped, results, warped


# -------------------------
# MAIN
# -------------------------
def extract_nutrition(image_path):
    """
    Extract nutrition information from an image file using EasyOCR.
    
    Args:
        image_path: Path to the nutrition label image
        
    Returns:
        Dictionary with extracted nutrition data (calories, fat, carbohydrates, protein)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image from file
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from: {image_path}")
    
    try:
        # Run extraction pipeline
        rows, nutrients, cropped, results, warped = extract_text_pipeline(image)
        
        return nutrients
    except Exception as e:
        raise ValueError(f"Error extracting nutrition from image: {str(e)}")