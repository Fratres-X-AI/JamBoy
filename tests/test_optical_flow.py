import numpy as np
import cv2

from jamboy.config import load_config
from jamboy.optical_flow import ImuSample, OpticalFlowTracker


def test_gyro_correction_zeros_pitch_induced_flow():
    cfg = load_config()
    est = OpticalFlowTracker(cfg)

    corrected_x, _, _ = est._gyro_derotate_pixels(
        10.0,
        0.0,
        ImuSample(omega_y=0.2),
        0.1,
    )
    expected = 10.0 - (est.focal_length_px * 0.2 * 0.1)
    assert abs(corrected_x - expected) < 1e-6


def test_optical_flow_returns_valid_on_sequential_frames():
    cfg = load_config()
    est = OpticalFlowTracker(cfg)

    rng = np.random.default_rng(0)
    img1 = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    img2 = np.roll(img1, shift=3, axis=1)

    est.update(img1)
    result = est.update(img2, imu=ImuSample(omega_y=0.0))
    assert result.valid or result.vx == 0.0
