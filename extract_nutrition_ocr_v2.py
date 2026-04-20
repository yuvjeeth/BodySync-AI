from ultralytics import YOLO
from PIL import Image
import re
from difflib import SequenceMatcher
import os
import shutil
import cv2
import numpy as np

# Must be set before importing paddleocr/paddlex internals.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Work around OneDNN runtime conversion errors on some Paddle builds.
os.environ["FLAGS_use_mkldnn"] = "0"
# Work around PIR runtime attribute conversion crashes in some Paddle builds.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

from paddleocr import PaddleOCR
import paddle

SAVE_PREPROCESSED = os.environ.get("SAVE_PREPROCESSED", "1") == "1"

try:
    # Apply flags again at runtime in case env vars are ignored by this build.
    paddle.set_flags(
        {
            "FLAGS_use_mkldnn": 0,
            "FLAGS_enable_pir_api": 0,
            "FLAGS_enable_pir_in_executor": 0,
        }
    )
except Exception:
    pass

print(paddle.__version__)
print(hasattr(paddle, "device"))


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
print(f"Paddle execution device: {PADDLE_DEVICE}")

if PADDLE_DEVICE == "cpu" and paddle.__version__.startswith("3.3"):
    print(
        "Warning: Paddle 3.3.x on CPU may hit a known oneDNN/PIR runtime issue in PaddleOCR. "
        "If this fails, pin to a stable CPU stack: paddlepaddle==2.6.2 and paddleocr==2.7.3."
    )


def create_ocr_engine():
    # Support both newer (device=...) and older (use_gpu=...) PaddleOCR APIs.
    kwargs = {
        "lang": "en",
        "use_textline_orientation": True,
    }
    if PADDLE_DEVICE.startswith("gpu"):
        try:
            return PaddleOCR(device=PADDLE_DEVICE, **kwargs)
        except TypeError:
            try:
                return PaddleOCR(use_gpu=True, **kwargs)
            except TypeError:
                return PaddleOCR(**kwargs)
    return PaddleOCR(**kwargs)
# -----------------------
# load detector
# -----------------------
detector = YOLO("nutrition_label_detector/nutrition_label_detector_baseline.pt")


# -----------------------
# initialize PaddleOCR
# -----------------------
ocr_engine = create_ocr_engine()


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


# -----------------------
# detect nutrition label
# -----------------------
def detect_label(image_path):
    results = detector(image_path)[0]

    if len(results.boxes) == 0:
        raise ValueError("No nutrition label was detected in the image")

    box = results.boxes.xyxy[0].tolist()

    img = Image.open(image_path).convert("RGB")
    crop = img.crop(box)

    crop_output_path = f"{os.path.splitext(image_path)[0]}_crop.jpg"
    crop.save(crop_output_path)
    print(f"Saved cropped label to: {crop_output_path}")

    return crop


# -----------------------
# preprocessing (lighter for PaddleOCR)
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
    base_gray = preprocess(pil_img)

    # Variant A: clean grayscale.
    variant_a = base_gray

    # Variant B: contrast-enhanced binary for faint/low-contrast text.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    boosted = clahe.apply(base_gray)
    variant_b = cv2.adaptiveThreshold(
        boosted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    return [variant_a, variant_b]


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


def extract_lines_from_ocr_result(result, min_conf=0.30):
    lines = []
    if not result:
        return lines

    for item in result:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or []
            rec_scores = item.get("rec_scores") or []
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if idx < len(rec_scores) else 1.0
                if text and score >= min_conf:
                    lines.append(str(text))
        elif isinstance(item, list):
            # Older PaddleOCR output format: [box, [text, score]]
            for word_info in item:
                if (
                    isinstance(word_info, (list, tuple))
                    and len(word_info) > 1
                    and isinstance(word_info[1], (list, tuple))
                    and len(word_info[1]) > 0
                ):
                    text = word_info[1][0]
                    score = word_info[1][1] if len(word_info[1]) > 1 else 1.0
                    if text and score >= min_conf:
                        lines.append(str(text))
    return lines


# -----------------------
# OCR using PaddleOCR
# -----------------------
def run_ocr(image):
    variants = build_ocr_variants(image)
    if SAVE_PREPROCESSED:
        saved_paths = save_preprocessed_variants(variants)
        print("Saved preprocessed images:")
        for path in saved_paths:
            print(f"- {path}")

    merged_lines = []
    seen = set()

    for variant in variants:
        img = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)

        try:
            # PaddleOCR >= 3 uses predict(), while 2.x uses ocr().
            if hasattr(ocr_engine, "predict"):
                result = ocr_engine.predict(img)
            else:
                result = ocr_engine.ocr(img, cls=True)
        except Exception as e:
            error_text = str(e)
            known_paddle_runtime_error = (
                "ConvertPirAttribute2RuntimeAttribute" in error_text
                or "onednn_instruction.cc" in error_text
            )
            if not known_paddle_runtime_error:
                raise
            raise RuntimeError(
                "PaddleOCR runtime failed in oneDNN/PIR path. "
                f"Current device is {PADDLE_DEVICE}. "
                "This is a known issue with some CPU builds on Paddle 3.3.x. "
                "Use a stable CPU combo: pip uninstall -y paddleocr paddlepaddle; "
                "pip install paddlepaddle==2.6.2 paddleocr==2.7.3. "
                "If you need GPU, install a CUDA-enabled paddle build that matches your CUDA/cuDNN "
                "(often easiest via Linux/WSL2)."
            ) from e

        variant_lines = extract_lines_from_ocr_result(result)
        for line in variant_lines:
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged_lines.append(line)

    full_text = "\n".join(merged_lines)

    print("\n--- OCR OUTPUT ---")
    print(full_text)
    print("------------------\n")

    return full_text.lower()


# -----------------------
# parsing nutrients
# -----------------------
def parse_nutrients(text):
    def normalize_line(s):
        s = s.lower().replace("’", "'")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def alpha_words(s):
        return re.findall(r"[a-z]+", s)

    def fuzzy_ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def nutrient_label_score(line, spec):
        words = alpha_words(line)
        if not words:
            return 0.0

        joined = "".join(words)
        best = 0.0
        for alias in spec["aliases"]:
            if alias in joined:
                best = max(best, 1.0)
            for w in words:
                best = max(best, fuzzy_ratio(alias, w))
                if len(words) > 1:
                    best = max(best, fuzzy_ratio(alias, "".join(words[:2])))

        for blocked in spec.get("exclude_roots", []):
            if blocked in joined:
                best -= 0.4
        return best

    def extract_candidates(line, expect_gram):
        line = normalize_line(line)
        # OCR normalization for unit/value confusions.
        line = re.sub(r'(\d)\s*[q9o]\b', r'\1g', line)
        line = re.sub(r'\bog\b', '0g', line)
        # Common OCR confusion: leading 'z' instead of '2' in gram values (e.g., z1g -> 21g).
        line = re.sub(r'\bz(?=\d+\s*g\b)', '2', line)
        line = re.sub(r'\b[il]\s*g\b', '1g', line)
        line = re.sub(r'\b[il]o\b', '1g', line)
        candidates = []
        if expect_gram:
            # Accept OCR variants where g is confused as q/9/o/0.
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(g|q|9|o|0)\b", line):
                candidates.append((m.group(1), True))
            if not candidates:
                for m in re.finditer(r"(\d+(?:\.\d+)?)", line):
                    candidates.append((m.group(1), False))
        else:
            for m in re.finditer(r"\b(\d{1,4})\b", line):
                candidates.append((m.group(1), False))
        return candidates

    def plausible(value, key):
        try:
            v = float(value)
        except ValueError:
            return False
        if key == "calories":
            return 0 <= v <= 2000
        return 0 <= v <= 100

    nutrient_specs = {
        "calories": {
            "aliases": ["calories", "calorie", "energy"],
            "expect_gram": False,
        },
        "fat": {
            "aliases": ["totalfat", "fat"],
            "expect_gram": True,
            "exclude_roots": ["satur", "trans", "poly", "mono"],
        },
        "carbohydrates": {
            "aliases": ["totalcarbohydrate", "carbohydrate", "carb"],
            "expect_gram": True,
        },
        "protein": {
            "aliases": ["protein"],
            "expect_gram": True,
        },
    }

    lines = [normalize_line(ln) for ln in text.splitlines() if ln.strip()]

    def parse_one(nutrient_key, spec):
        best_score = -1e9
        best_value = None

        for idx, line in enumerate(lines):
            label_score = nutrient_label_score(line, spec)
            if label_score < 0.55:
                continue

            contexts = [(line, 0)]
            if idx + 1 < len(lines):
                contexts.append((line + " " + lines[idx + 1], 1))
            if idx > 0:
                contexts.append((lines[idx - 1] + " " + line, 1))

            # Calories are often split across nearby lines (e.g., value above label).
            if nutrient_key == "calories":
                for offset in (-3, -2, 2, 3):
                    j = idx + offset
                    if 0 <= j < len(lines):
                        contexts.append((lines[j], abs(offset)))

            for context, dist in contexts:
                for value, has_unit in extract_candidates(context, spec["expect_gram"]):
                    # For gram-based nutrients, do not trust unit-less values from neighboring lines.
                    if spec["expect_gram"] and not has_unit and dist > 0:
                        continue
                    if not plausible(value, nutrient_key):
                        continue
                    score = label_score * 10.0
                    if has_unit:
                        score += 2.0
                    score -= dist * 1.0
                    if nutrient_key == "protein" and any(
                        token in context for token in ("sugar", "added", "fiber")
                    ):
                        score -= 4.0
                    if nutrient_key == "calories" and "." in value:
                        score -= 1.0
                    if nutrient_key == "calories":
                        if "%" in context:
                            score -= 3.0
                        if any(
                            token in context
                            for token in (
                                "serving size",
                                "servings per",
                                "per container",
                                "daily value",
                            )
                        ):
                            score -= 4.0
                        if any(token in context for token in ("amount per serving", "calor")):
                            score += 1.5
                    if score > best_score:
                        best_score = score
                        best_value = value
        return best_value

    return {
        key: parse_one(key, spec)
        for key, spec in nutrient_specs.items()
    }


# -----------------------
# full pipeline
# -----------------------
def extract_nutrition(image_path):
    crop = detect_label(image_path)
    text = run_ocr(crop)
    nutrients = parse_nutrients(text)
    return nutrients


# -----------------------
# Debug mode - webcam test
# -----------------------
if __name__ == "__main__":
    print("Opening webcam for capture...")
    # image_path = "nutrition_label_detector/0.jpg"
    image_path = "debug_meal_image.jpg"

    cap, camera_index = open_available_camera()

    if cap is None:
        print("Error: Could not open webcam")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Webcam opened on index {camera_index}. Press SPACE to capture or Q to quit.")

    captured = False

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            print("Error: Could not read frame")
            exit(1)

        cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            captured = True
            break
        elif key == ord('q'):
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()
    cv2.destroyAllWindows()

    if not captured:
        exit(1)

    print(f"\nTesting OCR on: {image_path}\n")

    try:
        result = extract_nutrition(image_path)

        print("\n--- FINAL EXTRACTED NUTRIENTS ---")
        for nutrient, value in result.items():
            print(f"{nutrient}: {value}")

    except Exception as e:
        print(f"Error: {e}")