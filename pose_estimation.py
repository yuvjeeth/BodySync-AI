from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float  # normalized [0,1]
    y: float  # normalized [0,1]
    visibility: float


_POSE_LANDMARK_NAMES = [
    # MediaPipe Pose Landmarker (33 landmarks)
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def _default_model_path() -> Path:
    # keep model local to repo so it's portable for demos/submission
    return Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"


def _ensure_pose_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    # Official MediaPipe model hosting (public GCS bucket).
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    print(f"Downloading pose model to: {model_path}")
    urlretrieve(url, model_path)  # nosec - expected public model download
    return model_path


def _create_landmarker(model_path: Path):
    import mediapipe as mp  # tasks-only build on Windows
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options), mp


def estimate_pose_keypoints_bgr(
    image_bgr: np.ndarray,
    *,
    model_path: Optional[str] = None,
) -> Tuple[List[PoseKeypoint], Dict[str, PoseKeypoint]]:
    """
    Returns keypoints in normalized coordinates using MediaPipe Pose Landmarker (Tasks API).
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], {}

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model_file = _ensure_pose_model(Path(model_path) if model_path else _default_model_path())
    landmarker, mp = _create_landmarker(model_file)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return [], {}

    lms = result.pose_landmarks[0]
    kp_list: List[PoseKeypoint] = []
    kp_map: Dict[str, PoseKeypoint] = {}

    for idx, lm in enumerate(lms):
        name = _POSE_LANDMARK_NAMES[idx] if idx < len(_POSE_LANDMARK_NAMES) else f"landmark_{idx}"
        # Pose Landmarker returns `visibility` for each landmark (float, may be absent in some builds)
        vis = float(getattr(lm, "visibility", 1.0))
        kp = PoseKeypoint(name=name, x=float(lm.x), y=float(lm.y), visibility=vis)
        kp_list.append(kp)
        kp_map[name] = kp

    return kp_list, kp_map


def normalized_distance(a: PoseKeypoint, b: PoseKeypoint) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def safe_get(kp_map: Dict[str, PoseKeypoint], name: str, min_vis: float = 0.3) -> Optional[PoseKeypoint]:
    kp = kp_map.get(name)
    if not kp:
        return None
    if kp.visibility < min_vis:
        return None
    return kp

