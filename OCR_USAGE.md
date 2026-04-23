# Nutrition Label OCR Extraction - Usage Guide

## Overview
This code extracts nutrition information from food nutrition labels using:
- **YOLO Detection**: Detects nutrition labels in images
- **PaddleOCR**: Extracts text from detected labels
- **RapidFuzz + Regex**: Parses nutrition values (calories, fat, carbs, protein)

## Quick Start

### 1. **Extract OCR from an Image File**
```python
from extract_nutrition_ocr import extract_nutrition

# Extract nutrition from a nutrition label image
result = extract_nutrition("path/to/nutrition_label.jpg")

print(result)
# Output: {'calories': '100', 'fat': '5', 'carbohydrates': '20', 'protein': '2'}
```

### 2. **Capture from Webcam and Extract**
```python
from extract_nutrition_ocr import extract_nutrition_from_webcam

# Captures image from webcam and extracts nutrition
result = extract_nutrition_from_webcam()
print(result)
```

### 3. **Command Line Usage**
```bash
# Extract from image file
python extract_nutrition_ocr.py path/to/image.jpg

# Capture from webcam (interactive)
python extract_nutrition_ocr.py
```

### 4. **In tools.py (Integration)**
```python
from tools import capture_and_extract_nutrition

result = capture_and_extract_nutrition()
print(result)
```

## Features

### Preprocessing
- **Grayscale Conversion**: Converts RGB to grayscale for better OCR
- **Upscaling**: 2x upscale for images < 2000px on largest dimension
- **Denoising**: Fast non-local means denoising
- **Adaptive Thresholding**: Two variants for robust extraction:
  - Grayscale with CLAHE contrast enhancement
  - Binary adaptive threshold image

### OCR Strategy
1. **Primary Pass**: Run OCR on enhanced grayscale image
2. **Fallback**: If < 8 text lines found, run secondary OCR on binary image
3. **Merge**: Combine results avoiding duplicates

### Nutrition Parsing
- **Labeled Lines**: Direct extraction from lines containing "calor", "fat", "carb", "prot"
- **Fuzzy Matching**: RapidFuzz matching for incomplete/OCR-garbled text
- **Context Extraction**: Smart lookups around detected nutrient keywords
- **Unit Handling**: Automatically corrects common OCR errors (0g, 1g, g units)

### Nutrient Extraction
Extracts four key nutrients:
- **Calories**: Without units (kcal)
- **Fat**: In grams (g)
- **Carbohydrates**: In grams (g)
- **Protein**: In grams (g)

## Output

The function returns a dictionary:
```python
{
    "calories": "100",      # or None if not found
    "fat": "5",
    "carbohydrates": "20",
    "protein": "2"
}
```

## Preprocessing Debug Output

By default, preprocessed images are saved to `debug_preprocessed/`:
- `01_gray.png`: Grayscale CLAHE-enhanced image
- `02_binary.png`: Binary adaptive threshold image

Disable with:
```bash
SAVE_PREPROCESSED=0 python extract_nutrition_ocr.py path/to/image.jpg
```

## Error Handling

```python
from extract_nutrition_ocr import extract_nutrition

try:
    result = extract_nutrition("image.jpg")
except FileNotFoundError as e:
    print(f"Image file not found: {e}")
except ValueError as e:
    print(f"No nutrition label detected: {e}")
except Exception as e:
    print(f"OCR extraction failed: {e}")
```

## Webcam Capture Tips

When capturing from webcam:
1. **Focus on Label**: Ensure the nutrition label is clearly visible
2. **Lighting**: Use good lighting for best OCR results
3. **Sharpness**: Real-time feedback shows sharpness score (aim for > 220)
4. **Alignment**: Try to capture the label straight-on, not at an angle

**Controls:**
- Press **SPACE** to capture image (will take 6-frame burst and save the sharpest)
- Press **Q** to quit capture mode

## Requirements

All packages listed in `requirements.txt`:
- `torch`, `ultralytics` - YOLO detection
- `paddleocr`, `paddle` - OCR engine
- `opencv-python` - Image preprocessing
- `Pillow` - Image handling
- `rapidfuzz` - Fuzzy string matching
- `pydantic`, `langchain_core`, `groq` - LLM integration (optional)

## Performance Notes

- **First Run**: PaddleOCR models (~500MB) download on first use to `~/.paddlex/`
- **GPU Support**: Automatically uses GPU if available (CUDA), falls back to CPU
- **Image Size**: Handles images up to 3800px on largest dimension
- **Processing Time**: ~5-10 seconds per image (depending on size and hardware)

## Troubleshooting

### No nutrition label detected
- Ensure the image clearly shows a nutrition label
- Try different lighting/angles
- Check that the YOLO model exists: `nutrition_label_detector/nutrition_label_detector_baseline.pt`

### Poor OCR results
- Ensure good lighting in the image
- Try straighter angles (not tilted)
- Make sure nutrition label text is clear and readable

### GPU not being used
- PaddleOCR automatically falls back to CPU if GPU unavailable
- Check CUDA installation if you want GPU acceleration

### Preprocessing debug images
- Check `debug_preprocessed/01_gray.png` and `debug_preprocessed/02_binary.png`
- These show what the OCR engine sees and help debug issues
