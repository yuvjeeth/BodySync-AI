from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pose_estimation import (
    PoseKeypoint,
    estimate_pose_keypoints_bgr,
    normalized_distance,
    safe_get,
)


@dataclass(frozen=True)
class BodyRatios:
    shoulder_width_n: Optional[float]
    hip_width_n: Optional[float]
    torso_length_n: Optional[float]
    shoulder_to_hip_ratio: Optional[float]


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return float(a / b)


def compute_body_ratios_from_pose(kp_map: Dict[str, PoseKeypoint]) -> BodyRatios:
    l_sh = safe_get(kp_map, "left_shoulder")
    r_sh = safe_get(kp_map, "right_shoulder")
    l_hip = safe_get(kp_map, "left_hip")
    r_hip = safe_get(kp_map, "right_hip")

    shoulder_width = normalized_distance(l_sh, r_sh) if l_sh and r_sh else None
    hip_width = normalized_distance(l_hip, r_hip) if l_hip and r_hip else None

    # "Torso length" as shoulder midpoint to hip midpoint
    torso_len = None
    if l_sh and r_sh and l_hip and r_hip:
        sh_mid = PoseKeypoint("shoulder_mid", (l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2, 1.0)
        hip_mid = PoseKeypoint("hip_mid", (l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2, 1.0)
        torso_len = normalized_distance(sh_mid, hip_mid)

    return BodyRatios(
        shoulder_width_n=shoulder_width,
        hip_width_n=hip_width,
        torso_length_n=torso_len,
        shoulder_to_hip_ratio=_ratio(shoulder_width, hip_width),
    )


def estimate_somatotype_heuristic(ratios: BodyRatios) -> Optional[str]:
    """
    Baseline heuristic (demo-quality).
    Replace with trained classifier once you have NHANES-derived features + labels.
    """
    r = ratios.shoulder_to_hip_ratio
    t = ratios.torso_length_n
    if r is None or t is None:
        return None

    # Rough patterns:
    # - mesomorph: broader shoulders relative to hips + moderate torso
    # - endomorph: lower shoulder/hip ratio + shorter torso
    # - ectomorph: higher torso length relative to widths, less pronounced shoulder/hip
    if r >= 1.15 and t <= 0.38:
        return "mesomorph"
    if r <= 1.05 and t <= 0.36:
        return "endomorph"
    return "ectomorph"


def analyze_body_from_image_bgr(image_bgr: np.ndarray) -> Dict[str, Any]:
    kps, kp_map = estimate_pose_keypoints_bgr(image_bgr)
    ratios = compute_body_ratios_from_pose(kp_map)
    somatotype = estimate_somatotype_heuristic(ratios)

    return {
        "pose_detected": bool(kps),
        "keypoints_detected": len(kps),
        "ratios": {
            "shoulder_width_n": ratios.shoulder_width_n,
            "hip_width_n": ratios.hip_width_n,
            "torso_length_n": ratios.torso_length_n,
            "shoulder_to_hip_ratio": ratios.shoulder_to_hip_ratio,
        },
        "somatotype_heuristic": somatotype,
        "notes": [
            "Ratios are normalized to image size (not real-world cm/in).",
            "Somatotype is a baseline heuristic; replace with NHANES-trained classifier.",
        ],
    }

