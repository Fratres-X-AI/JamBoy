#!/usr/bin/env python3
"""Chessboard camera calibration — saves intrinsics to config/calibration.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def calibrate_from_images(image_paths: list[Path], pattern_size: tuple[int, int], square_m: float) -> dict:
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_m

    objpoints, imgpoints = [], []
    gray_shape = None
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 3:
        raise RuntimeError(f"Need >=3 valid chessboard images, got {len(objpoints)}")

    ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    return {
        "reprojection_error": float(ret),
        "intrinsics": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "distortion": [float(x) for x in dist.ravel()[:5]],
        },
        "image_size": list(gray_shape),
        "pattern_size": list(pattern_size),
        "square_m": square_m,
    }


def calibrate_live(device: str, pattern_size: tuple[int, int], square_m: float, samples: int) -> dict:
    cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device}")

    tmp = ROOT / "data" / "calibration" / "captures"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    print(f"Show chessboard — press SPACE to capture ({samples} needed), q to finish early")
    while len(paths) < samples:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, pattern_size, None)
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(disp, f"{len(paths)}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("calibrate", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            path = tmp / f"cap_{len(paths):03d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(path)
            print(f"Saved {path}")
        elif key == ord("q") and len(paths) >= 3:
            break
    cap.release()
    cv2.destroyAllWindows()
    return calibrate_from_images(paths, pattern_size, square_m)


def main() -> int:
    parser = argparse.ArgumentParser(description="JamBoy camera calibration")
    parser.add_argument("--images", type=Path, nargs="*", help="Chessboard image paths")
    parser.add_argument("--camera", default="0", help="Live camera device for capture mode")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-m", type=float, default=0.025)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "config" / "calibration.yaml")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    if args.images:
        result = calibrate_from_images(args.images, pattern, args.square_m)
    else:
        result = calibrate_live(args.camera, pattern, args.square_m, args.samples)

    out_doc = {
        "camera": {
            "intrinsics": result["intrinsics"],
            "frame_width": result["image_size"][0],
            "frame_height": result["image_size"][1],
        },
        "calibration_meta": {
            "reprojection_error": result["reprojection_error"],
            "pattern_size": result["pattern_size"],
            "square_m": result["square_m"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        yaml.dump(out_doc, fh, default_flow_style=False)
    print(f"Wrote {args.out} (reproj err={result['reprojection_error']:.4f})")
    print("Merge camera.intrinsics into config/default.yaml or pass --config config/calibration.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
