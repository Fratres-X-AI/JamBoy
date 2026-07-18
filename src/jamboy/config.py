"""Load and expose typed configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return data.get(key, default)


@dataclass
class JamBoyConfig:
    raw: dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def camera(self) -> dict[str, Any]:
        return self.raw.get("camera", {})

    @property
    def geo_match(self) -> dict[str, Any]:
        return self.raw.get("geo_match", {})

    @property
    def optical_flow(self) -> dict[str, Any]:
        return self.raw.get("optical_flow", {})

    @property
    def barometer(self) -> dict[str, Any]:
        return self.raw.get("barometer", {})

    @property
    def navigator(self) -> dict[str, Any]:
        return self.raw.get("navigator", {})

    @property
    def ekf(self) -> dict[str, Any]:
        return self.raw.get("ekf", {})

    @property
    def mavlink(self) -> dict[str, Any]:
        return self.raw.get("mavlink", {})

    @property
    def maps(self) -> dict[str, Any]:
        return self.raw.get("maps", {})

    @property
    def sim(self) -> dict[str, Any]:
        return self.raw.get("sim", {})

    @property
    def compute(self) -> dict[str, Any]:
        return self.raw.get("compute", {})

    @property
    def vibration(self) -> dict[str, Any]:
        return self.raw.get("vibration", {})

    @property
    def realism(self) -> dict[str, Any]:
        return self.raw.get("realism", {})

    @property
    def extrinsics(self) -> dict[str, Any]:
        return self.raw.get("extrinsics", {})

    @property
    def hardware(self) -> dict[str, Any]:
        return self.raw.get("hardware", {})

    def resolve(self, relative: str) -> Path:
        return (self.project_root / relative).resolve()


def load_config(path: str | Path | None = None) -> JamBoyConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    path = Path(path)
    project_root = path.parent.parent

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return JamBoyConfig(raw=raw, project_root=project_root)
