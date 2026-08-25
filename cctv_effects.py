"""
Simulate degraded CCTV-style video for testing recognition under noisy feeds.
"""

import cv2
import numpy as np


def apply_cctv_effects(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Degrade a clean webcam frame to approximate noisy CCTV footage.

    Applies: resolution loss, JPEG compression, blur, sensor noise,
    low-light contrast crush, and desaturation.

    Args:
        frame: BGR image from OpenCV.
        strength: 0.0 = no change, 1.0 = default CCTV look, up to ~2.0 = harsher.

    Returns:
        Degraded BGR frame (same size as input).
    """
    strength = float(np.clip(strength, 0.0, 2.0))
    if strength == 0.0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Resolution loss — cheap cameras and long cable runs lose detail
    scale = max(0.25, 0.55 - 0.15 * (strength - 1.0))
    small_w = max(8, int(w * scale))
    small_h = max(8, int(h * scale))
    out = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Heavy JPEG compression artifacts
    jpeg_quality = int(np.clip(45 - 25 * strength, 8, 95))
    ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if ok:
        out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # 3. Lens blur / slight motion smear
    blur_k = 3 if strength < 1.5 else 5
    out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)

    # 4. Sensor noise (stronger in shadows, typical of cheap IR/night CCTV)
    sigma = 6.0 + 10.0 * strength
    noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5. Low-light: crushed contrast and slight green tint common on analog CCTV
    alpha = 0.75 - 0.12 * strength
    beta = -18.0 * strength
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= max(0.2, 0.65 - 0.15 * strength)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Slight green channel bias (cheap sensor / IR bleed)
    out[:, :, 1] = np.clip(out[:, :, 1].astype(np.int16) + int(4 * strength), 0, 255).astype(
        np.uint8
    )

    return out


def draw_cctv_badge(frame: np.ndarray, strength: float = 1.0) -> None:
    """Draw a small on-screen label indicating CCTV simulation is active."""
    label = f"CCTV SIM (strength={strength:.1f})"
    cv2.putText(
        frame,
        label,
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
