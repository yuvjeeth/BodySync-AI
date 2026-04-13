from ultralytics import YOLO
from PIL import Image
import pytesseract
import re
import os
import shutil


# -----------------------
# load detector
# -----------------------
detector = YOLO("nutrition_label_detector/nutrition_label_detector_baseline.pt")


def configure_tesseract():
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        pytesseract.pytesseract.tesseract_cmd = path_cmd
        return

    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in common_windows_paths:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


configure_tesseract()


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

    try:
        crop.show()
    except Exception as exc:
        print(f"Could not open crop preview window: {exc}")

    return crop


# -----------------------
# OCR
# -----------------------
def run_ocr(image):
    try:
        # Tesseract often performs better on larger images, so we can try resizing the crop.
        image = image.resize(
            (image.width * 2, image.height * 2)
            )

        text = pytesseract.image_to_string(image)

    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found in PATH. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and optionally set TESSERACT_CMD to the full tesseract.exe path."
        ) from exc

    print("\n--- OCR RAW OUTPUT START ---")
    print(text)
    print("--- OCR RAW OUTPUT END ---\n")

    return text


# -----------------------
# regex extraction
# -----------------------
def parse_nutrients(text):
    text = text.lower()

    nutrients = {}

    patterns = {
        "calories": r"calories[^0-9]*(\d+)",
        "fat": r"total\s*fat[^0-9]*(\d+\.?\d*\s*g)",
        "carbohydrates": r"(?:carb|carbohydrate[s]?)[^0-9]*(\d+\.?\d*\s*g)",
        "protein": r"protein[^0-9]*(\d+\.?\d*\s*g)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            nutrients[key] = match.group(1)

    return nutrients


# -----------------------
# full pipeline
# -----------------------
def extract_nutrition(image_path):

    crop = detect_label(image_path)

    text = run_ocr(crop)

    nutrients = parse_nutrients(text)

    return nutrients


# -----------------------
# run
# -----------------------
result = extract_nutrition("nutrition_label_detector/0.jpg")

print(result)