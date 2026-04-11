import cv2

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