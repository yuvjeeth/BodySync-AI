import cv2

from body_analysis import analyze_body_from_image_bgr
from pose_estimation import estimate_pose_keypoints_bgr, safe_get


POSE_EDGES = [
    # torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def draw_pose_overlay(image_bgr, kp_map):
    h, w = image_bgr.shape[:2]

    def pt(name):
        kp = kp_map.get(name)
        if not kp:
            return None
        x = int(kp.x * w)
        y = int(kp.y * h)
        return (x, y), kp.visibility

    # draw edges first
    for a, b in POSE_EDGES:
        pa = pt(a)
        pb = pt(b)
        if not pa or not pb:
            continue
        (xa, ya), va = pa
        (xb, yb), vb = pb
        if min(va, vb) < 0.3:
            continue
        cv2.line(image_bgr, (xa, ya), (xb, yb), (0, 255, 0), 3)

    # draw keypoints
    for name, kp in kp_map.items():
        if kp.visibility < 0.3:
            continue
        x = int(kp.x * w)
        y = int(kp.y * h)
        cv2.circle(image_bgr, (x, y), 6, (0, 0, 255), -1)

    return image_bgr


def main():
    image_path = "body_image.jpg"
    annotated_path = "body_image_annotated.jpg"

    # Some webcams/drivers hang on CAP_PROP_* setters on Windows.
    # Prefer "open and read" first, and try multiple indices.
    cap = None
    camera_index = None
    for idx in (0, 1, 2, 3):
        test = cv2.VideoCapture(idx)
        if test.isOpened():
            cap = test
            camera_index = idx
            break
        test.release()

    if cap is None:
        print("Error: Could not open webcam (tried indices 0-3)")
        return

    stable_needed = 6  # consecutive frames
    stable = 0
    countdown_frames = 45  # ~3s at 15fps
    countdown = None

    print(f"Auto-capture mode (camera index {camera_index}). Stand in full-body view. Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read frame")
                return

            # Run pose on a downscaled frame for speed, but keep capture at full res.
            small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            _, kp_map = estimate_pose_keypoints_bgr(small)

            # Full-body heuristic: shoulders + hips + at least one knee visible.
            l_sh = safe_get(kp_map, "left_shoulder")
            r_sh = safe_get(kp_map, "right_shoulder")
            l_hip = safe_get(kp_map, "left_hip")
            r_hip = safe_get(kp_map, "right_hip")
            l_knee = safe_get(kp_map, "left_knee")
            r_knee = safe_get(kp_map, "right_knee")

            full_body_ready = bool(l_sh and r_sh and l_hip and r_hip and (l_knee or r_knee))

            if full_body_ready:
                stable += 1
            else:
                stable = 0
                countdown = None

            if stable >= stable_needed and countdown is None:
                countdown = countdown_frames

            status = "Move back / center body"
            if full_body_ready:
                status = "Hold still..."
            if countdown is not None:
                sec = max(1, int((countdown + 14) / 15))
                status = f"Capturing in {sec}..."

            cv2.putText(
                frame,
                status,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0) if full_body_ready else (0, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.imshow("BodySync Pose Auto Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cancelled")
                return

            if countdown is not None:
                countdown -= 1
                if countdown <= 0:
                    ok_write = cv2.imwrite(image_path, frame)
                    if not ok_write:
                        print(f"Error: failed to write {image_path}")
                        return
                    print(f"Image saved to {image_path}")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    img = cv2.imread(image_path)
    if img is None:
        print("Error: could not read captured image.")
        return

    # Save annotated overlay for demo/report
    _, kp_map_full = estimate_pose_keypoints_bgr(img)
    annotated = img.copy()
    annotated = draw_pose_overlay(annotated, kp_map_full)
    ok_ann = cv2.imwrite(annotated_path, annotated)
    if not ok_ann:
        print(f"Error: failed to write {annotated_path}")
    else:
        print(f"Annotated image saved to {annotated_path}")

    result = analyze_body_from_image_bgr(img)
    print(result)


if __name__ == "__main__":
    main()

