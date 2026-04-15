import cv2
import json
from extract_nutrition_ocr import extract_nutrition

def webcam_capture_image(output_path: str = "captured_image.jpg") -> str:
    """
    Opens the webcam, allows user to take a picture, and saves it.
    
    Args:
        output_path: Path where the captured image will be saved
        
    Returns:
        "Done" when the image is successfully captured
    """
    # Open the default webcam (0 is usually the built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return "Error: Could not open webcam"
    
    print("Webcam opened. Press SPACE to capture or Q to quit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            return "Error: Could not read frame"
        
        # Display the frame
        cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE key to capture
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            break
        elif key == ord('q'):  # Q key to quit
            print("Cancelled")
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    return "Done"


def capture_and_extract_nutrition() -> str:
    """
    Captures an image from the webcam and extracts nutrition information from the label.
    
    Returns:
        A formatted string containing the extracted nutrition information
    """
    # Step 1: Capture image from webcam
    image_path = "meal_image.jpg"
    capture_result = webcam_capture_image(image_path)
    
    if not capture_result.startswith("Done"):
        return f"Error capturing image: {capture_result}"
    
    # Step 2: Extract nutrition from the captured image
    try:
        nutrition_data = extract_nutrition(image_path)
        
        # Format the nutrition data nicely
        if not nutrition_data:
            return "No nutrition information could be extracted from the label. Make sure the image contains a clear nutrition label."
        
        formatted_output = "Extracted Nutrition Information:\n"
        for nutrient, value in nutrition_data.items():
            formatted_output += f"  • {nutrient.capitalize()}: {value}\n"
        
        return formatted_output
    
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error extracting nutrition data: {str(e)}"