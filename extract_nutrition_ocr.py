from ultralytics import YOLO
from PIL import Image
import re
import os
import cv2
import numpy as np
from rapidfuzz import fuzz

# -----------------------
# Paddle setup
# -----------------------
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

from paddleocr import PaddleOCR
import paddle

SAVE_PREPROCESSED = os.environ.get("SAVE_PREPROCESSED", "1") == "1"

try:
    paddle.set_flags(
        {
            "FLAGS_use_mkldnn": 0,
            "FLAGS_enable_pir_api": 0,
            "FLAGS_enable_pir_in_executor": 0,
        }
    )
except Exception:
    pass

def select_paddle_device():
    if hasattr(paddle, "device") and paddle.device.is_compiled_with_cuda():
        try:
            paddle.set_device("gpu:0")
            return "gpu:0"
        except Exception:
            pass
    paddle.set_device("cpu")
    return "cpu"

PADDLE_DEVICE = select_paddle_device()

# -----------------------
# camera helper
# -----------------------
def open_available_camera(indices=(0, 1, 2, 3)):
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()
    return None, None


def capture_image_from_webcam(image_path="debug_meal_image.jpg"):
    print("Opening webcam for capture...")

    cap, camera_index = open_available_camera()

    if cap is None:
        print("Error: Could not open webcam (tried indices 0, 1, 2, 3)")
        exit(1)

    # Request 1080p capture and a codec/fps combination many webcams support.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    # Let autofocus and auto-exposure settle before user capture.
    for _ in range(20):
        cap.read()

    print(f"Webcam opened on index {camera_index}. Press SPACE to capture or Q to quit.")

    def sharpness_score(f):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    captured = False
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            print("Error: Could not read frame")
            exit(1)

        score = sharpness_score(frame)
        score_text = f"Sharpness: {score:.0f}"
        quality_text = "Good" if score >= 220 else "Adjust focus"

        cv2.putText(frame, score_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, quality_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Display the frame with sharpness feedback.
        cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE key to capture
            # Capture a short burst and save the sharpest frame to reduce blur.
            burst_frames = []
            for _ in range(6):
                ok, burst = cap.read()
                if ok:
                    burst_frames.append(burst)

            if burst_frames:
                frame_to_save = max(burst_frames, key=sharpness_score)
            else:
                frame_to_save = frame

            cv2.imwrite(image_path, frame_to_save)
            print(f"Image saved to {image_path}")
            captured = True
            break
        elif key == ord('q'):  # Q key to quit
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

    if not captured:
        exit(1)

    return image_path

def create_ocr_engine():
    kwargs = {"lang": "en", "use_textline_orientation": True}
    if PADDLE_DEVICE.startswith("gpu"):
        try:
            return PaddleOCR(device=PADDLE_DEVICE, **kwargs)
        except TypeError:
            return PaddleOCR(use_gpu=True, **kwargs)
    return PaddleOCR(**kwargs)

# -----------------------
# load detector
# -----------------------
detector = YOLO("nutrition_label_detector/nutrition_label_detector_baseline.pt")

# -----------------------
# OCR engine
# -----------------------
ocr_engine = create_ocr_engine()

# -----------------------
# detect nutrition label
# -----------------------
def detect_label(image_path):
    results = detector(image_path)[0]

    if len(results.boxes) == 0:
        raise ValueError("No nutrition label detected")

    box = results.boxes.xyxy[0].tolist()
    img = Image.open(image_path).convert("RGB")
    crop = img.crop(box)
    return crop

# -----------------------
# preprocessing
# -----------------------
def preprocess(pil_img):
    img = np.array(pil_img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale only when the source is not already large.
    h, w = gray.shape[:2]
    if max(h, w) < 2000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # mild denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    # Keep OCR input under model-side max limits to avoid internal resize paths.
    h, w = gray.shape[:2]
    max_side = max(h, w)
    max_side_limit = 3800
    if max_side > max_side_limit:
        scale = max_side_limit / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return gray

def build_ocr_variants(pil_img):
    base = preprocess(pil_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    boosted = clahe.apply(base)

    binary = cv2.adaptiveThreshold(
        boosted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )

    return [base, binary]


def save_preprocessed_variants(variants, out_dir="debug_preprocessed"):
    os.makedirs(out_dir, exist_ok=True)
    output_paths = []

    names = ["01_gray", "02_binary"]
    for idx, img in enumerate(variants):
        name = names[idx] if idx < len(names) else f"{idx+1:02d}"
        path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(path, img)
        output_paths.append(path)

    return output_paths

# -----------------------
# OCR helpers
# -----------------------
def extract_lines(result, min_conf=0.3):
    lines = []

    for item in result:
        if isinstance(item, dict):
            texts = item.get("rec_texts", [])
            scores = item.get("rec_scores", [])
            for i, t in enumerate(texts):
                if t and (i >= len(scores) or scores[i] >= min_conf):
                    lines.append(str(t))
        elif isinstance(item, list):
            for w in item:
                if isinstance(w, (list, tuple)) and len(w) > 1:
                    t = w[1][0]
                    s = w[1][1] if len(w[1]) > 1 else 1.0
                    if t and s >= min_conf:
                        lines.append(str(t))

    return lines

def run_ocr(image):
    variants = build_ocr_variants(image)
    if SAVE_PREPROCESSED:
        saved_paths = save_preprocessed_variants(variants)
        print("Saved preprocessed images:")
        for path in saved_paths:
            print(f"- {path}")

    # Prefer the first OCR pass; only use second pass when the first is too sparse.
    primary_img = cv2.cvtColor(variants[0], cv2.COLOR_GRAY2BGR)
    if hasattr(ocr_engine, "predict"):
        primary_result = ocr_engine.predict(primary_img)
    else:
        primary_result = ocr_engine.ocr(primary_img, cls=True)

    primary_lines = extract_lines(primary_result)
    primary_lines = [l for l in primary_lines if l.strip()]

    # If first pass is strong enough, avoid mixing in noisy second-pass text.
    if len(primary_lines) >= 8:
        merged = primary_lines
    else:
        seen = set()
        merged = []

        for l in primary_lines:
            key = l.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(l)

        secondary_img = cv2.cvtColor(variants[1], cv2.COLOR_GRAY2BGR)
        if hasattr(ocr_engine, "predict"):
            secondary_result = ocr_engine.predict(secondary_img)
        else:
            secondary_result = ocr_engine.ocr(secondary_img, cls=True)

        secondary_lines = extract_lines(secondary_result)
        for l in secondary_lines:
            key = l.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(l)

    full_text = "\n".join(merged)
    print("\n--- OCR OUTPUT ---")
    print(full_text)
    print("------------------\n")

    return full_text.lower()

# -----------------------
# RAPIDFUZZ + CONTEXT PARSER
# -----------------------
NUTRIENTS = {
    "calories": ["calories", "calorie", "energy"],
    "fat": ["total fat", "fat"],
    "carbohydrates": ["total carbohydrate", "carbohydrate", "carb"],
    "protein": ["protein"],
}

def find_best_index(lines, keywords):
    best_score = 0
    best_idx = None

    for i, line in enumerate(lines):
        for kw in keywords:
            score = fuzz.partial_ratio(kw, line)
            if score > best_score:
                best_score = score
                best_idx = i

    return best_idx if best_score > 65 else None

def extract_value_from_context(lines, idx, expect_g):
    if idx is None:
        return None

    def normalize_units(line):
        line = line.lower()
        line = re.sub(r'(\d)\s*[q9o]\b', r'\1g', line)
        line = re.sub(r'\b[il]\s*g\b', '1g', line)
        line = re.sub(r'\bo\s*g\b', '0g', line)
        line = re.sub(r'\bog\b', '0g', line)
        return line

    # Prefer same line, then nearest neighbors.
    offsets = [0, -1, 1, -2, 2] if expect_g else [0, 1, -1, 2, -2, 3, -3]
    for offset in offsets:
        j = idx + offset
        if not (0 <= j < len(lines)):
            continue

        line = normalize_units(lines[j])

        if expect_g:
            # For gram nutrients, only accept values tied to a g-like unit.
            m = re.search(r'(\d+(?:\.\d+)?)\s*g\b', line)
            if m:
                return m.group(1)
            # If OCR dropped the unit on the matched nutrient line, use same-line number.
            if offset == 0:
                m = re.search(r'(\d+(?:\.\d+)?)', line)
                if m:
                    return m.group(1)
        else:
            # For calories, avoid percentage/DV lines.
            if '%' in line or 'daily value' in line or 'dv' in line:
                continue
            # Avoid serving-size/context numbers near calories label.
            if any(tok in line for tok in ("serving", "size", "pieces", "container", "per serving")):
                continue
            if re.search(r'\b\d+(?:\.\d+)?\s*(g|mg)\b', line):
                continue
            m = re.search(r'\b(\d{1,4})\b', line)
            if m:
                return m.group(1)

    return None

def parse_nutrients(text):
    lines = [l.strip().lower() for l in text.splitlines() if l.strip()]

    def normalize_units(line):
        line = line.lower()
        line = re.sub(r'(\d)\s*[q9o]\b', r'\1g', line)
        line = re.sub(r'\b[il]\s*g\b', '1g', line)
        line = re.sub(r'\bo\s*g\b', '0g', line)
        line = re.sub(r'\bog\b', '0g', line)
        return line

    def extract_from_labeled_line(include_roots, exclude_roots=None, expect_g=False):
        exclude_roots = exclude_roots or []
        for i, line in enumerate(lines):
            if not any(root in line for root in include_roots):
                continue
            if any(root in line for root in exclude_roots):
                continue

            norm = normalize_units(line)
            if expect_g:
                m = re.search(r'(\d+(?:\.\d+)?)\s*g\b', norm)
                if m:
                    return m.group(1)
                # Unit may be dropped by OCR; use same-line numeric fallback.
                m = re.search(r'(\d+(?:\.\d+)?)', norm)
                if m:
                    return m.group(1)
            else:
                if '%' in norm or 'daily value' in norm or 'dv' in norm:
                    continue
                m = re.search(r'\b(\d{1,4})\b', norm)
                if m:
                    return m.group(1)
                # Calories are commonly on the next line after the label.
                for offset in [1, 2, -1, -2, 3, -3]:
                    j = i + offset
                    if not (0 <= j < len(lines)):
                        continue
                    candidate = normalize_units(lines[j])
                    if '%' in candidate or 'daily value' in candidate or 'dv' in candidate:
                        continue
                    if any(tok in candidate for tok in ("serving", "size", "pieces", "container", "per serving")):
                        continue
                    if re.search(r'\b\d+(?:\.\d+)?\s*(g|mg)\b', candidate):
                        continue
                    m = re.search(r'\b(\d{1,4})\b', candidate)
                    if m:
                        return m.group(1)
        return None

    def extract_protein_from_labeled_line():
        for line in lines:
            if "prot" not in line:
                continue

            norm = normalize_units(line)

            # Normal case: explicit gram unit.
            m = re.search(r'(\d+(?:\.\d+)?)\s*g\b', norm)
            if m:
                return m.group(1)

            # OCR often reads "1g" as "ig", "lg", "io", "lo", or "10".
            if re.search(r'\b[il1]\s*[g0o]\b', norm) or re.search(r'\b10\b', norm):
                return "1"

            # Last-resort same-line numeric fallback.
            m = re.search(r'(\d+(?:\.\d+)?)', norm)
            if m:
                return m.group(1)

        return None

    calories = extract_from_labeled_line(["calor", "energy"], ["fat cal"], expect_g=False)
    fat = extract_from_labeled_line(["fat"], ["saturated", "trans", "poly", "mono"], expect_g=True)
    carbohydrates = extract_from_labeled_line(["carb"], ["fiber", "sugar", "added"], expect_g=True)
    protein = extract_protein_from_labeled_line()

    # Fallback to fuzzy-context parser only for missing nutrients.
    if calories is None:
        calories = extract_value_from_context(lines, find_best_index(lines, NUTRIENTS["calories"]), False)
    if fat is None:
        fat = extract_value_from_context(lines, find_best_index(lines, NUTRIENTS["fat"]), True)
    if carbohydrates is None:
        carbohydrates = extract_value_from_context(lines, find_best_index(lines, NUTRIENTS["carbohydrates"]), True)
    if protein is None:
        protein = extract_value_from_context(lines, find_best_index(lines, NUTRIENTS["protein"]), True)

    return {
        "calories": calories,
        "fat": fat,
        "carbohydrates": carbohydrates,
        "protein": protein,
    }

# -----------------------
# full pipeline
# -----------------------
def extract_nutrition(image_path):
    crop = detect_label(image_path)
    text = run_ocr(crop)
    return parse_nutrients(text)

# -----------------------
# test
# -----------------------
if __name__ == "__main__":
    image_path = capture_image_from_webcam("debug_meal_image.jpg")

    print(f"\nTesting OCR on: {image_path}\n")

    try:
        result = extract_nutrition(image_path)

        print("\n--- FINAL EXTRACTED NUTRIENTS ---")
        for k, v in result.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"Error: {e}")