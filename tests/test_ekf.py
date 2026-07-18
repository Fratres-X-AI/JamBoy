import numpy as np

from jamboy.config import load_config
from jamboy.ekf import NavigationEKF


def test_ekf_position_update_moves_state():
    cfg = load_config()
    ekf = NavigationEKF(cfg.ekf)
    ekf.set_position(0.0, 0.0)
    accepted = ekf.update_position(2.0, 1.0, confidence=0.9)
    assert accepted
    pos = ekf.position
    assert pos[0] > 0.0
    assert pos[1] > 0.0


def test_ekf_baro_updates_altitude():
    cfg = load_config()
    ekf = NavigationEKF(cfg.ekf)
    ekf.set_position(0.0, 0.0, z=100.0)
    assert ekf.update_altitude(101.5)
    assert ekf.position[2] > 100.0


def test_ekf_rejects_outlier():
    cfg = load_config()
    ekf = NavigationEKF(cfg.ekf)
    ekf.set_position(0.0, 0.0)
    for _ in range(5):
        ekf.update_position(1.0, 1.0, confidence=0.9)
    accepted = ekf.update_position(500.0, 500.0, confidence=0.9)
    assert not accepted
