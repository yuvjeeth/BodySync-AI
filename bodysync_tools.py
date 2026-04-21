import cv2
from extract_nutrition_ocr_v2 import extract_nutrition
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


def capture_and_extract_nutrition() -> str:
    """
    Captures an image from the webcam and extracts nutrition information from the label.
    """
    image_path = "meal_image.jpg"
    capture_result = webcam_capture_image(image_path)

    if not capture_result.startswith("Done"):
        return f"Error capturing image: {capture_result}"

    try:
        nutrition_data = extract_nutrition(image_path)

        if not nutrition_data:
            return (
                "No nutrition information could be extracted from the label. "
                "Make sure the image contains a clear nutrition label."
            )

        formatted_output = "Extracted Nutrition Information:\n"
        for nutrient, value in nutrition_data.items():
            formatted_output += f"  • {nutrient.capitalize()}: {value}\n"

        return formatted_output
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error extracting nutrition data: {str(e)}"


def capture_and_analyze_body() -> str:
    """
    Captures an image from the webcam and runs pose-driven body ratio analysis.
    """
    image_path = "body_image.jpg"
    capture_result = webcam_capture_image(image_path)

    if not capture_result.startswith("Done"):
        return f"Error capturing image: {capture_result}"

    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read captured image from disk"

    result = analyze_body_from_image_bgr(img)

    ratios = result.get("ratios") or {}
    somatotype = result.get("somatotype_heuristic")
    detected = result.get("pose_detected")

    lines = [
        "Body Analysis (Pose-based):",
        f"  - Pose detected: {detected}",
        f"  - Keypoints detected: {result.get('keypoints_detected')}",
        f"  - Somatotype (heuristic): {somatotype}",
        "  - Ratios (normalized):",
        f"    - Shoulder width: {ratios.get('shoulder_width_n')}",
        f"    - Hip width: {ratios.get('hip_width_n')}",
        f"    - Torso length: {ratios.get('torso_length_n')}",
        f"    - Shoulder/Hip ratio: {ratios.get('shoulder_to_hip_ratio')}",
    ]
    return "\n".join(lines)

