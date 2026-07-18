"""Production 6-state EKF for GPS-denied navigation with Mahalanobis gating."""

from __future__ import annotations

import numpy as np


def _noise(config: dict, nested_key: str, flat_key: str, default: float) -> float:
    nested = config.get("measurement_noise", {})
    if nested_key in nested:
        return float(nested[nested_key])
    return float(config.get(flat_key, default))


def _process_noise(config: dict, axis: str, flat_key: str, default: float) -> float:
    nested = config.get("process_noise", {})
    if axis in nested:
        return float(nested[axis])
    return float(config.get(flat_key, default))


class JamBoyEKF:
    """
    6-State Extended Kalman Filter for GPS-denied navigation.
    States: [x, y, z, vx, vy, vz] in local NED meters.
    """

    def __init__(self, config: dict, baro_config: dict | None = None):
        baro_config = baro_config or {}
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0

        q_pos = _process_noise(config, "position", "process_noise_pos", 0.5)
        q_vel = _process_noise(config, "velocity", "process_noise_vel", 1.0)
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        self.r_baro = float(
            baro_config.get("noise_variance")
            or baro_config.get("meas_noise_m", 2.0) ** 2
            or config.get("meas_noise_baro", 4.0)
        )
        self.r_flow = _noise(config, "optical_flow", "meas_noise_flow", 2.0)
        self.r_geo = _noise(config, "geo_match", "meas_noise_geo_base", 5.0)
        self.gate_threshold = float(config.get("innovation_gate", 9.0))

        self._last_reject_reason: str | None = None

    @property
    def position_uncertainty_m(self) -> float:
        return float(np.sqrt(max(np.trace(self.P[:3, :3]), 0.0)))

    @property
    def confidence_score(self) -> float:
        """0–100% trust score derived from position covariance (lower P → higher score)."""
        trace = float(np.trace(self.P[:3, :3]))
        return float(np.clip(100.0 * (1.0 / (1.0 + trace / 10.0)), 0.0, 100.0))

    def predict(self, dt: float, imu_accel: np.ndarray | None = None) -> None:
        if dt <= 0:
            return
        A = np.eye(6)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        u = np.zeros((3, 1))
        if imu_accel is not None:
            u = np.asarray(imu_accel, dtype=float).reshape(3, 1)

        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) * dt

        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q

    def update_barometer(self, z_baro: float) -> bool:
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        R = np.array([[self.r_baro]])
        return self._update_state(np.array([[z_baro]]), H, R)

    def update_geo_match(self, z_geo: np.ndarray, confidence: float = 1.0) -> bool:
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        noise = self.r_geo / max(float(confidence), 0.1)
        R = np.eye(2) * noise
        z = np.asarray(z_geo, dtype=float).reshape(2, 1)
        return self._update_state(z, H, R)

    def update_optical_flow(self, vx: float, vy: float, vz: float = 0.0) -> bool:
        H = np.zeros((2, 6))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        R = np.eye(2) * self.r_flow
        return self._update_state(np.array([[vx], [vy]]), H, R)

    def _update_state(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
            mahal = float((y.T @ S_inv @ y).item())
        except np.linalg.LinAlgError:
            self._last_reject_reason = "singular_S"
            return False

        if mahal > self.gate_threshold:
            self._last_reject_reason = f"mahalanobis={mahal:.2f}"
            return False

        K = self.P @ H.T @ S_inv
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ H)) @ self.P
        self._last_reject_reason = None
        return True

    def get_state(self) -> np.ndarray:
        return self.x.flatten().copy()


class _FilterPyCompat:
    """Shim so legacy code/tests can access `.ekf.x` as a flat length-6 vector."""

    def __init__(self, core: JamBoyEKF):
        self._core = core

    @property
    def x(self) -> np.ndarray:
        return self._core.x.flatten()

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._core.x = np.asarray(value, dtype=float).reshape(6, 1)

    @property
    def P(self) -> np.ndarray:
        return self._core.P


class NavigationEKF:
    """Backward-compatible facade over JamBoyEKF."""

    def __init__(self, config: dict, baro_config: dict | None = None):
        self._core = JamBoyEKF(config, baro_config=baro_config)

    @property
    def ekf(self) -> _FilterPyCompat:
        return _FilterPyCompat(self._core)

    @property
    def position(self) -> np.ndarray:
        return self._core.get_state()[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self._core.get_state()[3:6]

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self._core.P))

    @property
    def confidence_score(self) -> float:
        return self._core.confidence_score

    def predict(self, dt: float, accel: np.ndarray | None = None) -> None:
        self._core.predict(dt, imu_accel=accel)

    def update_velocity(self, vx: float, vy: float, vz: float = 0.0) -> bool:
        return self._core.update_optical_flow(vx, vy, vz)

    def update_altitude(self, altitude_m: float) -> bool:
        return self._core.update_barometer(altitude_m)

    def update_position(self, x: float, y: float, confidence: float = 1.0) -> bool:
        return self._core.update_geo_match(np.array([x, y]), confidence=confidence)

    def set_position(self, x: float, y: float, z: float = 0.0) -> None:
        self._core.x[0, 0] = x
        self._core.x[1, 0] = y
        self._core.x[2, 0] = z
        self._core.P[0, 0] = 1.0
        self._core.P[1, 1] = 1.0
        self._core.P[2, 2] = 1.0

    def set_velocity(self, vx: float, vy: float, vz: float = 0.0) -> None:
        self._core.x[3, 0] = vx
        self._core.x[4, 0] = vy
        self._core.x[5, 0] = vz

    def zero_velocity(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)
        self._core.P[3, 3] = 1.0
        self._core.P[4, 4] = 1.0
        self._core.P[5, 5] = 1.0

    def get_state(self) -> np.ndarray:
        return self._core.get_state()
