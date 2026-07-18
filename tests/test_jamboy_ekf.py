"""JamBoyEKF production architecture tests."""

from __future__ import annotations

import numpy as np

from jamboy.config import load_config
from jamboy.ekf import JamBoyEKF, NavigationEKF


def test_jamboy_ekf_predict_with_imu_accel():
    cfg = load_config()
    ekf = JamBoyEKF(cfg.ekf, baro_config=cfg.barometer)
    ekf.predict(0.1, imu_accel=np.array([1.0, 0.0, 0.0]))
    state = ekf.get_state()
    assert state[3] > 0.0


def test_jamboy_ekf_rejects_geo_outlier():
    cfg = load_config()
    ekf = JamBoyEKF(cfg.ekf)
    ekf.x[0, 0] = 0.0
    ekf.x[1, 0] = 0.0
    assert ekf.update_geo_match(np.array([500.0, 500.0]), confidence=0.9) is False


def test_confidence_score_range():
    cfg = load_config()
    nav = NavigationEKF(cfg.ekf, baro_config=cfg.barometer)
    nav.set_position(0, 0)
    assert 0.0 <= nav.confidence_score <= 100.0
