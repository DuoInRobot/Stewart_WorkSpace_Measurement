#!/usr/bin/env python3
"""RealSense D435i ArUco detector.

本脚本面向 Intel RealSense D435i 的彩色相机流，使用设备自身的彩色内参
进行 ArUco 位姿估计。当前仓库环境里即使尚未安装 ``pyrealsense2``，代码
也会给出明确提示，方便在 Docker 环境准备好依赖后直接运行。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # Docker 里当前可能还没装
    rs = None


MARKER_SIZE_DEFAULT = 0.036  # m


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int


class RealSenseColorCamera:
    """D435i 彩色流封装。"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
    ) -> None:
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required for RealSense capture; "
                "install requirements-collection.txt before using hardware."
            )

        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile: Optional[rs.pipeline_profile] = None

    def start(self) -> CameraIntrinsics:
        self.profile = self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get initial RealSense color frame.")

        video_profile = color_frame.profile.as_video_stream_profile()
        intr = video_profile.get_intrinsics()
        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        # Brown-Conrady 模型常见为 [k1, k2, p1, p2, k3]
        coeffs = list(intr.coeffs)
        dist_coeffs = np.array(coeffs[:5], dtype=np.float32).reshape(-1, 1)
        return CameraIntrinsics(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            width=intr.width,
            height=intr.height,
        )

    def read(self) -> np.ndarray:
        if self.profile is None:
            raise RuntimeError("RealSense pipeline has not been started.")
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get RealSense color frame.")
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None


class ArUcoDetector:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        marker_size: float = MARKER_SIZE_DEFAULT,
        dictionary_id: int = cv2.aruco.DICT_6X6_250,
    ) -> None:
        self.marker_size = marker_size
        self.camera_matrix = intrinsics.camera_matrix
        self.dist_coeffs = intrinsics.dist_coeffs

        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        try:
            self.parameters = cv2.aruco.DetectorParameters()
            self.use_new_api = True
        except AttributeError:
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minCornerDistanceRate = 0.05
        self.parameters.minMarkerDistanceRate = 0.05
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.1
        self.parameters.markerBorderBits = 1
        self.parameters.maxErroneousBitsInBorderRate = 0.35
        self.parameters.minOtsuStdDev = 5.0
        self.parameters.errorCorrectionRate = 0.6

        if self.use_new_api:
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.detector = None

        self.obj_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        self._prev_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _rvec_distance(r1: np.ndarray, r2: np.ndarray) -> float:
        r1m, _ = cv2.Rodrigues(r1)
        r2m, _ = cv2.Rodrigues(r2)
        diff = r1m.T @ r2m
        cos_val = np.clip((np.trace(diff) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cos_val))

    def detect(self, frame: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.dictionary, parameters=self.parameters)

        results: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
        if ids is None:
            return results

        for i in range(len(ids)):
            marker_id = int(ids[i][0]) if ids.ndim > 1 else int(ids[i])
            n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                self.obj_points,
                corners[i],
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if n_sol == 0:
                continue

            valid = [
                (rvecs[si], tvecs[si], float(reproj_errs[si][0]))
                for si in range(n_sol)
                if tvecs[si].flatten()[2] > 0
            ]
            if not valid:
                continue

            if marker_id in self._prev_poses and len(valid) > 1:
                prev_rvec, prev_tvec = self._prev_poses[marker_id]
                score, rvec, tvec, err = min(
                    (
                        self._rvec_distance(cand_rvec, prev_rvec)
                        + 2.0 * float(np.linalg.norm(cand_tvec - prev_tvec)),
                        cand_rvec,
                        cand_tvec,
                        cand_err,
                    )
                    for cand_rvec, cand_tvec, cand_err in valid
                )
                _ = score
            else:
                rvec, tvec, err = min(valid, key=lambda item: item[2])

            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corners[i],
                self.camera_matrix,
                self.dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or tvec.flatten()[2] <= 0:
                continue

            results[marker_id] = (rvec, tvec, err, corners[i])
            self._prev_poses[marker_id] = (rvec.copy(), tvec.copy())

        return results

    def draw(self, frame: np.ndarray, detections: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> np.ndarray:
        image = frame.copy()
        for marker_id, (rvec, tvec, reproj_error, corners) in detections.items():
            cv2.aruco.drawDetectedMarkers(image, [corners], np.array([[marker_id]], dtype=np.int32))
            cv2.drawFrameAxes(
                image,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.marker_size * 1.5,
                2,
            )

            tx, ty, tz = tvec.flatten() * 100.0
            rx, ry, rz = rvec.flatten()
            anchor = corners.reshape(-1, 2)[0].astype(int)
            x0, y0 = int(anchor[0]), int(anchor[1])

            cv2.putText(image, f"ID: {marker_id}", (x0, y0 - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(image, f"Pos(cm): {tx:.1f}, {ty:.1f}, {tz:.1f}", (x0, y0 - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            cv2.putText(image, f"Rot(rad): {rx:.2f}, {ry:.2f}, {rz:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            cv2.putText(image, f"Reproj(px): {reproj_error:.3f}", (x0, y0 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco detector for RealSense D435i color stream.")
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense device serial number.")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_DEFAULT, help="Marker size in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        camera = RealSenseColorCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.serial,
        )
        intrinsics = camera.start()
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")

    print("--- RealSense D435i ArUco Detector Running ---")
    print(f"Resolution: {intrinsics.width}x{intrinsics.height} @ {args.fps} FPS")
    print("Camera matrix:")
    print(intrinsics.camera_matrix)
    print("Distortion coefficients:")
    print(intrinsics.dist_coeffs.flatten())
    print("Press 'q' or ESC to exit.")

    detector = ArUcoDetector(intrinsics=intrinsics, marker_size=args.marker_size)

    try:
        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            vis = detector.draw(frame, detections)
            cv2.imshow("RealSense D435i ArUco Detector", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
