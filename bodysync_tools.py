import cv2
import os
from extract_nutrition_ocr import extract_nutrition
from body_analysis import analyze_body_from_image_bgr


def webcam_capture_image(output_path: str = "captured_image.jpg") -> str:
    """
    Opens the webcam, allows user to take a picture, and saves it.

    Args:
        output_path: Path where the captured image will be saved

    Returns:
        "Done" when the image is successfully captured
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "Error: Could not open webcam"

    print("Webcam opened. Press SPACE to capture or Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            return "Error: Could not read frame"

        cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # SPACE
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            break
        if key == ord("q"):
            print("Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Done"


def extract_nutrition_from_file(image_path: str) -> str:
    """
    Extracts nutrition information from a nutrition label image using OCR.
    
    Args:
        image_path: Path to the nutrition label image
        
    Returns:
        Formatted string with extracted nutrition data (calories, fat, carbs, protein)
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found: {image_path}"
    
    try:
        nutrition_data = extract_nutrition(image_path)
        
        if not nutrition_data:
            return "No nutrition information could be extracted from the label."
        
        formatted_output = "Extracted Nutrition Information:\n"
        for nutrient, value in nutrition_data.items():
            if value is not None:
                formatted_output += f"  • {nutrient.capitalize()}: {value}\n"
            else:
                formatted_output += f"  • {nutrient.capitalize()}: Could not extract\n"
        
        return formatted_output
    except FileNotFoundError as e:
        return f"Error: Image file not found. {str(e)}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error extracting nutrition data: {str(e)}"


def capture_and_extract_nutrition() -> str:
    """
    Captures an image from the webcam and extracts nutrition information from the label.
    
    Process:
    1. Opens webcam for user to capture nutrition label image
    2. Uses YOLO model to detect nutrition label
    3. Runs PaddleOCR to extract text
    4. Parses text to extract calories, fat, carbohydrates, and protein
    
    Returns:
        Formatted string with extracted nutrition information
    """
    image_path = "meal_image.jpg"
    print("Step 1: Capturing image from webcam...")
    
    capture_result = webcam_capture_image(image_path)

    if not capture_result.startswith("Done"):
        return f"Error capturing image: {capture_result}"

    print(f"✓ Image captured successfully: {image_path}")
    print("Step 2: Extracting nutrition from label...")
    
    return extract_nutrition_from_file(image_path)


def analyze_body_from_file(image_path: str) -> str:
    """
    Analyzes body measurements from an image using pose estimation.
    
    Args:
        image_path: Path to the full-body image
        
    Returns:
        Formatted string with body ratios and somatotype analysis
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found: {image_path}"
    
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image from disk"

    result = analyze_body_from_image_bgr(img)

    ratios = result.get("ratios") or {}
    somatotype = result.get("somatotype_heuristic")
    detected = result.get("pose_detected")
    notes = result.get("notes", [])

    lines = [
        "Body Analysis (Pose-based):",
        f"  - Pose detected: {detected}",
        f"  - Keypoints detected: {result.get('keypoints_detected')}",
        f"  - Somatotype (heuristic): {somatotype}",
        "  - Body Ratios (normalized to image size):",
        f"    - Shoulder width: {ratios.get('shoulder_width_n'):.3f if ratios.get('shoulder_width_n') else 'N/A'}",
        f"    - Hip width: {ratios.get('hip_width_n'):.3f if ratios.get('hip_width_n') else 'N/A'}",
        f"    - Torso length: {ratios.get('torso_length_n'):.3f if ratios.get('torso_length_n') else 'N/A'}",
        f"    - Shoulder/Hip ratio: {ratios.get('shoulder_to_hip_ratio'):.3f if ratios.get('shoulder_to_hip_ratio') else 'N/A'}",
    ]
    
    if notes:
        lines.append("  - Notes:")
        for note in notes:
            lines.append(f"    • {note}")
    
    return "\n".join(lines)


def capture_and_analyze_body() -> str:
    """
    Captures an image from the webcam and runs pose-driven body ratio analysis.
    
    Process:
    1. Opens webcam for user to capture full-body image
    2. Uses MediaPipe Pose Landmarker to detect body keypoints
    3. Computes normalized body ratios (shoulders, hips, torso)
    4. Estimates somatotype (ectomorph, mesomorph, endomorph) using heuristic
    
    Returns:
        Formatted string with pose detection results and body analysis
    """
    image_path = "body_image.jpg"
    print("Step 1: Capturing full-body image from webcam...")
    print("Tip: Stand straight and face the camera for best results.")
    
    capture_result = webcam_capture_image(image_path)

    if not capture_result.startswith("Done"):
        return f"Error capturing image: {capture_result}"

    print(f"✓ Image captured successfully: {image_path}")
    print("Step 2: Analyzing body measurements...")
    
    return analyze_body_from_file(image_path)


