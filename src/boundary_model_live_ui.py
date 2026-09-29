#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from typing import Optional, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np


from train_boundary_radius_nn_model import BoundaryRadiusNN


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "boundary_radius_nn_model.npz"
DEFAULT_OUTSIDE_LOG = DEFAULT_RAW_DIR / "boundary_outside_candidates.csv"
DEFAULT_MANUAL_BOUNDARY_LOG = DEFAULT_RAW_DIR / "manual_boundary_marks.csv"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8093
DEFAULT_WARNING_MARGIN_MM = 3.0
DEFAULT_UPPER_RADIUS_MM = 60.0
DEFAULT_OUTSIDE_LOG_INTERVAL_S = 0.2
DEFAULT_MANUAL_BOUNDARY_WEIGHT = 8.0
DEFAULT_MANUAL_BOUNDARY_SAMPLE_INTERVAL_S = 0.2

OUTSIDE_CANDIDATE_FIELDS = [
    "event_index",
    "timestamp_s",
    "dataset_label",
    "status",
    "marker_id",
    "reproj_error_px",
    "camera_in_marker_x_m",
    "camera_in_marker_y_m",
    "camera_in_marker_z_m",
    "camera_in_marker_roll_rad",
    "camera_in_marker_pitch_rad",
    "camera_in_marker_yaw_rad",
    "zero_camera_in_marker_x_m",
    "zero_camera_in_marker_y_m",
    "zero_camera_in_marker_z_m",
    "zero_camera_in_marker_roll_rad",
    "zero_camera_in_marker_pitch_rad",
    "zero_camera_in_marker_yaw_rad",
    "relative_x_mm",
    "relative_y_mm",
    "relative_z_mm",
    "relative_rotvec_x_rad",
    "relative_rotvec_y_rad",
    "relative_rotvec_z_rad",
    "q1_mm",
    "q2_mm",
    "q3_mm",
    "q4_mm",
    "q5_mm",
    "q6_mm",
    "pose_radius_mm",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "predicted_boundary_radius_mm",
    "radius_margin_mm",
    "neighbor_count",
    "nearest_angle_deg",
    "fallback_used",
]

MANUAL_BOUNDARY_FIELDS = [
    "event_index",
    "timestamp_s",
    "dataset_label",
    "mark_source",
    "model_status",
    "sample_weight",
    *OUTSIDE_CANDIDATE_FIELDS[3:],
]


@dataclass(frozen=True)
class Pose6:
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


@dataclass(frozen=True)
class LiveMarkerPose:
    timestamp_s: float
    marker_id: int
    camera_in_marker: Pose6
    reproj_error_px: float


def rpy_to_rotation_matrix(rpy_rad: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rotation_matrix_to_rpy_rad(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    horizontal = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if horizontal > 1e-9:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(-float(matrix[2, 0]), horizontal)
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        pitch = math.atan2(-float(matrix[2, 0]), horizontal)
        yaw = 0.0
    return roll, pitch, yaw


def invert_marker_pose_to_camera_pose(rvec_marker_in_camera: np.ndarray, tvec_marker_in_camera: np.ndarray) -> Pose6:
    rvec = np.asarray(rvec_marker_in_camera, dtype=float).reshape(3, 1)
    tvec = np.asarray(tvec_marker_in_camera, dtype=float).reshape(3, 1)
    marker_rotation, _ = cv2.Rodrigues(rvec)
    camera_rotation = marker_rotation.T
    camera_translation = -camera_rotation @ tvec
    return Pose6(
        position_m=tuple(float(value) for value in camera_translation.flatten()),
        rpy_rad=rotation_matrix_to_rpy_rad(camera_rotation),
    )


def relative_feature(reference: Pose6, current: Pose6, upper_radius_mm: float = DEFAULT_UPPER_RADIUS_MM) -> dict:
    reference_rotation = rpy_to_rotation_matrix(reference.rpy_rad)
    current_rotation = rpy_to_rotation_matrix(current.rpy_rad)
    relative_rotation = reference_rotation.T @ current_rotation
    relative_translation_mm = reference_rotation.T @ (
        np.asarray(current.position_m, dtype=float) - np.asarray(reference.position_m, dtype=float)
    ) * 1000.0
    relative_rotvec, _ = cv2.Rodrigues(relative_rotation)
    relative_rotvec = relative_rotvec.reshape(3)
    q = np.concatenate([relative_translation_mm, relative_rotvec * float(upper_radius_mm)])
    radius = float(np.linalg.norm(q))
    direction = q / radius if radius > 1e-9 else np.full(6, math.nan)
    return {
        "relative_x_mm": float(relative_translation_mm[0]),
        "relative_y_mm": float(relative_translation_mm[1]),
        "relative_z_mm": float(relative_translation_mm[2]),
        "relative_rotvec_x_rad": float(relative_rotvec[0]),
        "relative_rotvec_y_rad": float(relative_rotvec[1]),
        "relative_rotvec_z_rad": float(relative_rotvec[2]),
        "q1_mm": float(q[0]),
        "q2_mm": float(q[1]),
        "q3_mm": float(q[2]),
        "q4_mm": float(q[3]),
        "q5_mm": float(q[4]),
        "q6_mm": float(q[5]),
        "pose_radius_mm": radius,
        "direction": [float(value) for value in direction],
    }


@dataclass(frozen=True)
class BoundaryPrediction:
    radius_mm: float
    neighbor_count: int = 0
    nearest_angle_deg: float = 0.0
    fallback_used: bool = False


class BoundaryModelAdapter:
    def __init__(self, model: BoundaryRadiusNN, *, model_type: str = "nn") -> None:
        self.model = model
        self.model_type = model_type

    def predict(self, direction_or_q: Sequence[float]) -> BoundaryPrediction:
        radius = float(np.asarray(self.model.predict(direction_or_q), dtype=float).reshape(-1)[0])
        return BoundaryPrediction(radius_mm=radius)


def load_boundary_model(path: Path) -> BoundaryModelAdapter:
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError("the public live UI supports only .npz neural-network models")
    return BoundaryModelAdapter(BoundaryRadiusNN.load(model_path), model_type="nn")


def classify_margin(margin_mm: Optional[float], warning_margin_mm: float) -> str:
    if margin_mm is None or not math.isfinite(float(margin_mm)):
        return "not_ready"
    if margin_mm <= float(warning_margin_mm):
        return "near_boundary"
    return "inside"


def _json_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _format_float(value: Optional[float]) -> str:
    value = _json_float(value)
    return "nan" if value is None else f"{value:.6f}"


def _bool_field(value) -> str:
    return "1" if bool(value) else "0"


def _payload_pose(payload: dict, prefix: str) -> dict[str, str]:
    pose = payload.get(prefix) or {}
    field_prefix = "zero_camera_in_marker" if prefix == "zero_camera_in_marker" else "camera_in_marker"
    return {
        f"{field_prefix}_x_m": _format_float(pose.get("x_m")),
        f"{field_prefix}_y_m": _format_float(pose.get("y_m")),
        f"{field_prefix}_z_m": _format_float(pose.get("z_m")),
        f"{field_prefix}_roll_rad": _format_float(pose.get("roll_rad")),
        f"{field_prefix}_pitch_rad": _format_float(pose.get("pitch_rad")),
        f"{field_prefix}_yaw_rad": _format_float(pose.get("yaw_rad")),
    }


def _payload_to_common_row(payload: dict, event_index: int, *, dataset_label: str, status: Optional[str] = None) -> dict[str, str]:
    row = {
        "event_index": str(event_index),
        "timestamp_s": _format_float(payload.get("timestamp_s")),
        "dataset_label": dataset_label,
        "status": status if status is not None else str(payload.get("status", "")),
        "marker_id": "" if payload.get("marker_id") is None else str(payload.get("marker_id")),
        "reproj_error_px": _format_float(payload.get("reproj_error_px")),
        "pose_radius_mm": _format_float(payload.get("pose_radius_mm")),
        "predicted_boundary_radius_mm": _format_float(payload.get("predicted_boundary_radius_mm")),
        "radius_margin_mm": _format_float(payload.get("radius_margin_mm")),
        "neighbor_count": "" if payload.get("neighbor_count") is None else str(payload.get("neighbor_count")),
        "nearest_angle_deg": _format_float(payload.get("nearest_angle_deg")),
        "fallback_used": _bool_field(payload.get("fallback_used")),
    }
    row.update(_payload_pose(payload, "camera_in_marker"))
    row.update(_payload_pose(payload, "zero_camera_in_marker"))
    relative = payload.get("relative") or {}
    row.update(
        {
            "relative_x_mm": _format_float(relative.get("x_mm")),
            "relative_y_mm": _format_float(relative.get("y_mm")),
            "relative_z_mm": _format_float(relative.get("z_mm")),
            "relative_rotvec_x_rad": _format_float(relative.get("rotvec_x_rad")),
            "relative_rotvec_y_rad": _format_float(relative.get("rotvec_y_rad")),
            "relative_rotvec_z_rad": _format_float(relative.get("rotvec_z_rad")),
        }
    )
    q = payload.get("q") or []
    direction = payload.get("direction") or []
    for index in range(6):
        row[f"q{index + 1}_mm"] = _format_float(q[index] if index < len(q) else None)
        row[f"d{index + 1}"] = _format_float(direction[index] if index < len(direction) else None)
    return row


class OutsideCandidateLogger:
    def __init__(self, path: Path, *, min_interval_s: float = DEFAULT_OUTSIDE_LOG_INTERVAL_S, enabled: bool = True) -> None:
        self.path = Path(path)
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._event_index = 0
        self._last_logged_timestamp_s: Optional[float] = None
        self._last_logged_margin_mm: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_header()

    def enable(self) -> None:
        with self._lock:
            self.enabled = True

    def disable(self) -> None:
        with self._lock:
            self.enabled = False

    def clear(self) -> None:
        with self._lock:
            self._event_index = 0
            self._last_logged_timestamp_s = None
            self._last_logged_margin_mm = None
            self._write_header()

    def status(self) -> dict:
        with self._lock:
            return {
                "outside_logging_enabled": self.enabled,
                "outside_logged_count": self._event_index,
                "last_outside_logged_timestamp_s": self._last_logged_timestamp_s,
                "last_outside_logged_margin_mm": self._last_logged_margin_mm,
                "outside_log_path": str(self.path),
                "outside_log_interval_s": self.min_interval_s,
            }

    def maybe_log(self, payload: dict) -> bool:
        if payload.get("status") != "outside":
            return False
        try:
            timestamp_s = float(payload["timestamp_s"])
        except (KeyError, TypeError, ValueError):
            return False
        with self._lock:
            if not self.enabled:
                return False
            if (
                self._last_logged_timestamp_s is not None
                and self.min_interval_s > 0.0
                and timestamp_s - self._last_logged_timestamp_s < self.min_interval_s
            ):
                return False
            self._event_index += 1
            row = self._payload_to_row(payload, self._event_index)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=OUTSIDE_CANDIDATE_FIELDS)
                writer.writerow(row)
            self._last_logged_timestamp_s = timestamp_s
            self._last_logged_margin_mm = _json_float(payload.get("radius_margin_mm"))
            return True

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTSIDE_CANDIDATE_FIELDS)
            writer.writeheader()

    def _payload_to_row(self, payload: dict, event_index: int) -> dict[str, str]:
        return _payload_to_common_row(payload, event_index, dataset_label="boundary_candidate", status="outside")


class ManualBoundaryLogger:
    def __init__(self, path: Path, *, sample_weight: float = DEFAULT_MANUAL_BOUNDARY_WEIGHT) -> None:
        self.path = Path(path)
        self.sample_weight = float(sample_weight)
        self._lock = threading.Lock()
        self._event_index = 0
        self._last_logged_timestamp_s: Optional[float] = None
        self._last_logged_margin_mm: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_header()

    def clear(self) -> None:
        with self._lock:
            self._event_index = 0
            self._last_logged_timestamp_s = None
            self._last_logged_margin_mm = None
            self._write_header()

    def status(self) -> dict:
        with self._lock:
            return {
                "manual_boundary_logged_count": self._event_index,
                "last_manual_boundary_logged_timestamp_s": self._last_logged_timestamp_s,
                "last_manual_boundary_logged_margin_mm": self._last_logged_margin_mm,
                "manual_boundary_log_path": str(self.path),
                "manual_boundary_sample_weight": self.sample_weight,
            }

    def log(self, payload: dict) -> bool:
        if not _payload_is_loggable_boundary(payload):
            return False
        with self._lock:
            self._event_index += 1
            row = self._payload_to_row(payload, self._event_index)
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANUAL_BOUNDARY_FIELDS)
                writer.writerow(row)
            try:
                self._last_logged_timestamp_s = float(payload["timestamp_s"])
            except (KeyError, TypeError, ValueError):
                self._last_logged_timestamp_s = None
            self._last_logged_margin_mm = _json_float(payload.get("radius_margin_mm"))
            return True

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANUAL_BOUNDARY_FIELDS)
            writer.writeheader()

    def _payload_to_row(self, payload: dict, event_index: int) -> dict[str, str]:
        model_status = str(payload.get("status", ""))
        row = _payload_to_common_row(payload, event_index, dataset_label="boundary", status=model_status)
        row["mark_source"] = "manual_confirmed_boundary"
        row["model_status"] = model_status
        row["sample_weight"] = _format_float(self.sample_weight)
        return row


def _payload_is_loggable_boundary(payload: dict) -> bool:
    if payload.get("marker_detected") is False:
        return False
    required = ["timestamp_s", "pose_radius_mm", "q", "direction", "camera_in_marker", "zero_camera_in_marker"]
    for key in required:
        if payload.get(key) is None:
            return False
    direction = payload.get("direction") or []
    q = payload.get("q") or []
    return len(direction) == 6 and len(q) == 6


class LiveBoundaryState:
    def __init__(
        self,
        *,
        model: BoundaryModelAdapter,
        warning_margin_mm: float = DEFAULT_WARNING_MARGIN_MM,
        upper_radius_mm: float = DEFAULT_UPPER_RADIUS_MM,
        outside_logger: Optional[OutsideCandidateLogger] = None,
        manual_boundary_logger: Optional[ManualBoundaryLogger] = None,
        manual_boundary_sample_interval_s: float = DEFAULT_MANUAL_BOUNDARY_SAMPLE_INTERVAL_S,
    ) -> None:
        self.model = model
        self.warning_margin_mm = float(warning_margin_mm)
        self.upper_radius_mm = float(upper_radius_mm)
        self.outside_logger = outside_logger
        self.manual_boundary_logger = manual_boundary_logger
        self.manual_boundary_sample_interval_s = max(0.0, float(manual_boundary_sample_interval_s))
        self._lock = threading.Lock()
        self._latest_marker: Optional[LiveMarkerPose] = None
        self._zero_pose: Optional[Pose6] = None
        self._latest_frame_jpeg: Optional[bytes] = None
        self._camera_error: Optional[str] = None
        self._stats = {"inside": 0, "near_boundary": 0, "not_ready": 0}
        self._manual_boundary_sampling_enabled = False
        self._last_manual_boundary_sample_time_s: Optional[float] = None

    def update_marker(self, marker: Optional[LiveMarkerPose]) -> None:
        with self._lock:
            self._latest_marker = marker

    def update_frame(self, frame_jpeg: Optional[bytes]) -> None:
        with self._lock:
            self._latest_frame_jpeg = frame_jpeg

    def update_camera_error(self, error: Optional[str]) -> None:
        with self._lock:
            self._camera_error = error

    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame_jpeg

    def set_zero(self) -> bool:
        with self._lock:
            if self._latest_marker is None:
                return False
            self._zero_pose = self._latest_marker.camera_in_marker
            return True

    def clear_zero(self) -> None:
        with self._lock:
            self._zero_pose = None

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = {"inside": 0, "near_boundary": 0, "not_ready": 0}

    def enable_outside_logging(self) -> bool:
        if self.outside_logger is None:
            return False
        self.outside_logger.enable()
        return True

    def disable_outside_logging(self) -> bool:
        if self.outside_logger is None:
            return False
        self.outside_logger.disable()
        return True

    def clear_outside_log(self) -> bool:
        if self.outside_logger is None:
            return False
        self.outside_logger.clear()
        return True

    def clear_manual_boundary_log(self) -> bool:
        if self.manual_boundary_logger is None:
            return False
        self.manual_boundary_logger.clear()
        return True

    def mark_manual_boundary(self) -> dict:
        payload = self.status_payload()
        if self.manual_boundary_logger is None:
            payload["ok"] = False
            payload["reason"] = "manual boundary logger is not configured"
            return payload
        ok = self.manual_boundary_logger.log(payload)
        payload["ok"] = ok
        if not ok:
            payload["reason"] = "current pose is not ready for boundary marking"
        payload["manual_boundary_logging"] = self._manual_boundary_logging_status()
        return payload

    def start_manual_boundary_sampling(self) -> dict:
        with self._lock:
            self._manual_boundary_sampling_enabled = True
            self._last_manual_boundary_sample_time_s = None
        return {"ok": True, "manual_boundary_logging": self._manual_boundary_logging_status()}

    def stop_manual_boundary_sampling(self) -> dict:
        with self._lock:
            self._manual_boundary_sampling_enabled = False
        return {"ok": True, "manual_boundary_logging": self._manual_boundary_logging_status()}

    def status_payload(self, *, now_s: Optional[float] = None) -> dict:
        timestamp_s = time.time() if now_s is None else float(now_s)
        with self._lock:
            marker = self._latest_marker
            zero_pose = self._zero_pose
            stats = dict(self._stats)
            camera_error = self._camera_error

        payload = {
            "timestamp_s": timestamp_s,
            "marker_detected": marker is not None,
            "marker_id": None if marker is None else marker.marker_id,
            "reproj_error_px": None if marker is None else marker.reproj_error_px,
            "zero_set": zero_pose is not None,
            "warning_margin_mm": self.warning_margin_mm,
            "upper_radius_mm": self.upper_radius_mm,
            "model_type": getattr(self.model, "model_type", "unknown"),
            "status": "not_ready",
            "camera_error": camera_error,
            "camera_in_marker": None,
            "relative": None,
            "q": None,
            "direction": None,
            "pose_radius_mm": None,
            "predicted_boundary_radius_mm": None,
            "radius_margin_mm": None,
            "neighbor_count": None,
            "nearest_angle_deg": None,
            "fallback_used": None,
            "stats": stats,
            "outside_logging": self._outside_logging_status(),
            "manual_boundary_logging": self._manual_boundary_logging_status(),
        }
        if marker is not None:
            payload["camera_in_marker"] = _pose_to_payload(marker.camera_in_marker)
        if marker is None or zero_pose is None:
            self._record_status("not_ready")
            payload["stats"] = self._stats_snapshot()
            payload["outside_logging"] = self._outside_logging_status()
            payload["manual_boundary_logging"] = self._manual_boundary_logging_status()
            return payload

        feature = relative_feature(zero_pose, marker.camera_in_marker, self.upper_radius_mm)
        radius = feature["pose_radius_mm"]
        direction = np.asarray(feature["direction"], dtype=float)
        try:
            prediction = self.model.predict(direction)
            predicted_radius = prediction.radius_mm
            margin = predicted_radius - radius
            status = classify_margin(margin, self.warning_margin_mm)
        except ValueError:
            prediction = None
            predicted_radius = None
            margin = None
            status = "not_ready"

        payload.update(
            {
                "status": status,
                "zero_camera_in_marker": _pose_to_payload(zero_pose),
                "relative": {
                    "x_mm": _json_float(feature["relative_x_mm"]),
                    "y_mm": _json_float(feature["relative_y_mm"]),
                    "z_mm": _json_float(feature["relative_z_mm"]),
                    "rotvec_x_rad": _json_float(feature["relative_rotvec_x_rad"]),
                    "rotvec_y_rad": _json_float(feature["relative_rotvec_y_rad"]),
                    "rotvec_z_rad": _json_float(feature["relative_rotvec_z_rad"]),
                },
                "q": [_json_float(feature[f"q{index}_mm"]) for index in range(1, 7)],
                "direction": [_json_float(value) for value in feature["direction"]],
                "pose_radius_mm": _json_float(radius),
                "predicted_boundary_radius_mm": _json_float(predicted_radius),
                "radius_margin_mm": _json_float(margin),
                "neighbor_count": None if prediction is None else prediction.neighbor_count,
                "nearest_angle_deg": None if prediction is None else prediction.nearest_angle_deg,
                "fallback_used": None if prediction is None else prediction.fallback_used,
            }
        )
        self._record_status(status)
        if self.outside_logger is not None:
            self.outside_logger.maybe_log(payload)
        self._maybe_log_continuous_manual_boundary(payload, timestamp_s)
        payload["stats"] = self._stats_snapshot()
        payload["outside_logging"] = self._outside_logging_status()
        payload["manual_boundary_logging"] = self._manual_boundary_logging_status()
        return payload

    def _maybe_log_continuous_manual_boundary(self, payload: dict, timestamp_s: float) -> bool:
        with self._lock:
            enabled = self._manual_boundary_sampling_enabled
            last_sample_time_s = self._last_manual_boundary_sample_time_s
            interval_s = self.manual_boundary_sample_interval_s
        if not enabled or self.manual_boundary_logger is None:
            return False
        if last_sample_time_s is not None and interval_s > 0.0 and timestamp_s - last_sample_time_s < interval_s:
            return False
        if not self.manual_boundary_logger.log(payload):
            return False
        with self._lock:
            self._last_manual_boundary_sample_time_s = timestamp_s
        return True

    def _record_status(self, status: str) -> None:
        with self._lock:
            self._stats[status] = self._stats.get(status, 0) + 1

    def _stats_snapshot(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _outside_logging_status(self) -> dict:
        if self.outside_logger is None:
            return {
                "outside_logging_enabled": False,
                "outside_logged_count": 0,
                "last_outside_logged_timestamp_s": None,
                "last_outside_logged_margin_mm": None,
                "outside_log_path": None,
                "outside_log_interval_s": None,
            }
        return self.outside_logger.status()

    def _manual_boundary_logging_status(self) -> dict:
        with self._lock:
            sampling_enabled = self._manual_boundary_sampling_enabled
            interval_s = self.manual_boundary_sample_interval_s
        if self.manual_boundary_logger is None:
            return {
                "manual_boundary_logged_count": 0,
                "last_manual_boundary_logged_timestamp_s": None,
                "last_manual_boundary_logged_margin_mm": None,
                "manual_boundary_log_path": None,
                "manual_boundary_sample_weight": None,
                "manual_boundary_sampling_enabled": sampling_enabled,
                "manual_boundary_sample_interval_s": interval_s,
            }
        status = self.manual_boundary_logger.status()
        status["manual_boundary_sampling_enabled"] = sampling_enabled
        status["manual_boundary_sample_interval_s"] = interval_s
        return status


def _pose_to_payload(pose: Pose6) -> dict:
    return {
        "x_m": float(pose.position_m[0]),
        "y_m": float(pose.position_m[1]),
        "z_m": float(pose.position_m[2]),
        "roll_rad": float(pose.rpy_rad[0]),
        "pitch_rad": float(pose.rpy_rad[1]),
        "yaw_rad": float(pose.rpy_rad[2]),
    }


def _select_detection(detections, marker_id: Optional[int]):
    if not detections:
        return None
    if marker_id is not None and marker_id in detections:
        return marker_id, detections[marker_id]
    selected_id = sorted(detections.keys())[0]
    return selected_id, detections[selected_id]


class CameraPoller:
    def __init__(self, *, state: LiveBoundaryState, marker_id: int, marker_size: float, width: int, height: int, fps: int, serial: Optional[str], dictionary: int) -> None:
        self.state = state
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.dictionary = dictionary
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._camera = None
        self._detector = None

    def start(self) -> None:
        from arudo_detector import ArUcoDetector, RealSenseColorCamera

        camera = RealSenseColorCamera(width=self.width, height=self.height, fps=self.fps, serial=self.serial)
        try:
            intrinsics = camera.start()
            detector = ArUcoDetector(
                intrinsics=intrinsics,
                marker_size=self.marker_size,
                dictionary_id=self.dictionary,
            )
        except Exception:
            camera.stop()
            raise
        self._camera = camera
        self._detector = detector
        self.state.update_camera_error(None)
        self._thread = threading.Thread(target=self._run, name="boundary-model-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._detector = None

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._camera.read()
                detections = self._detector.detect(frame)
                selected = _select_detection(detections, self.marker_id)
                marker_pose = None
                if selected is not None:
                    detected_id, (rvec, tvec, reproj_error_px, _corners) = selected
                    marker_pose = LiveMarkerPose(
                        timestamp_s=time.time(),
                        marker_id=int(detected_id),
                        camera_in_marker=invert_marker_pose_to_camera_pose(rvec, tvec),
                        reproj_error_px=float(reproj_error_px),
                    )
                self.state.update_marker(marker_pose)
                image = self._detector.draw(frame, detections)
                payload = self.state.status_payload()
                cv2.putText(
                    image,
                    f"{payload['status']} margin={payload['radius_margin_mm']}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if payload["status"] == "inside" else (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", image)
                if ok:
                    self.state.update_frame(bytes(encoded))
        except Exception as exc:
            self.state.update_marker(None)
            self.state.update_camera_error(str(exc))
            self._stop_event.set()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stewart Boundary Model Live</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f5f7fa; color: #16202a; }
    main { display: grid; grid-template-columns: minmax(360px, 1fr) 420px; gap: 16px; padding: 16px; }
    .panel { background: #fff; border: 1px solid #d9e1ea; border-radius: 6px; padding: 12px; }
    img { width: 100%; background: #101820; border-radius: 4px; aspect-ratio: 4 / 3; object-fit: contain; }
    .status { font-size: 28px; font-weight: 700; margin: 4px 0 12px; }
    .inside { color: #087443; }
    .near_boundary { color: #b76e00; }
    .not_ready { color: #667085; }
    button { margin: 4px 4px 8px 0; padding: 8px 10px; border: 1px solid #98a2b3; background: #fff; border-radius: 4px; cursor: pointer; }
    dl { display: grid; grid-template-columns: 190px 1fr; gap: 6px 10px; margin: 0; font-size: 14px; }
    dt { color: #526071; }
    dd { margin: 0; font-family: monospace; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <section class="panel"><img id="frame" src="/api/frame.jpg" alt="Live camera"></section>
  <section class="panel">
    <div id="status" class="status not_ready">not_ready</div>
    <button onclick="post('/api/set_zero')">Set Zero</button>
    <button onclick="post('/api/clear_zero')">Clear Zero</button>
    <button onclick="post('/api/reset_stats')">Reset Stats</button>
    <button onclick="post('/api/mark_manual_boundary')">Mark Current As Boundary</button>
    <button onclick="post('/api/start_manual_boundary_sampling')">Start Continuous Boundary Sampling</button>
    <button onclick="post('/api/stop_manual_boundary_sampling')">Stop Continuous Boundary Sampling</button>
    <button onclick="post('/api/clear_manual_boundary_log')">Clear Manual Boundary Log</button>
    <dl id="data"></dl>
  </section>
</main>
<script>
function fmt(v, n=3) { return v === null || v === undefined ? '-' : Number(v).toFixed(n); }
async function post(url) { await fetch(url, {method: 'POST'}); await refresh(); }
function item(k, v) { return `<dt>${k}</dt><dd>${v}</dd>`; }
async function refresh() {
  const res = await fetch('/api/status');
  const data = await res.json();
  const status = document.getElementById('status');
  status.className = 'status ' + data.status;
  status.textContent = data.status;
  const rel = data.relative || {};
  const cam = data.camera_in_marker || {};
  const manualLog = data.manual_boundary_logging || {};
  document.getElementById('data').innerHTML =
    item('marker_detected', data.marker_detected) +
    item('marker_id', data.marker_id ?? '-') +
    item('zero_set', data.zero_set) +
    item('camera x/y/z m', `${fmt(cam.x_m)} ${fmt(cam.y_m)} ${fmt(cam.z_m)}`) +
    item('camera r/p/y rad', `${fmt(cam.roll_rad)} ${fmt(cam.pitch_rad)} ${fmt(cam.yaw_rad)}`) +
    item('relative x/y/z mm', `${fmt(rel.x_mm)} ${fmt(rel.y_mm)} ${fmt(rel.z_mm)}`) +
    item('relative rotvec rad', `${fmt(rel.rotvec_x_rad)} ${fmt(rel.rotvec_y_rad)} ${fmt(rel.rotvec_z_rad)}`) +
    item('pose_radius_mm', fmt(data.pose_radius_mm)) +
    item('predicted boundary mm', fmt(data.predicted_boundary_radius_mm)) +
    item('radius_margin_mm', fmt(data.radius_margin_mm)) +
    item('neighbor_count', data.neighbor_count ?? '-') +
    item('nearest_angle_deg', fmt(data.nearest_angle_deg)) +
    item('fallback_used', data.fallback_used ?? '-') +
    item('manual boundary count', manualLog.manual_boundary_logged_count ?? '-') +
    item('manual continuous sampling', manualLog.manual_boundary_sampling_enabled ?? '-') +
    item('manual sample interval s', fmt(manualLog.manual_boundary_sample_interval_s)) +
    item('last manual margin', fmt(manualLog.last_manual_boundary_logged_margin_mm)) +
    item('manual sample weight', fmt(manualLog.manual_boundary_sample_weight)) +
    item('manual boundary log path', manualLog.manual_boundary_log_path ?? '-') +
    item('stats', JSON.stringify(data.stats));
  document.getElementById('frame').src = '/api/frame.jpg?t=' + Date.now();
}
setInterval(refresh, 250);
refresh();
</script>
</body>
</html>
"""


def make_handler(state: LiveBoundaryState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(state.status_payload())
            elif path == "/api/frame.jpg":
                frame = state.latest_frame()
                if frame is None:
                    frame = _blank_jpeg()
                self._send_bytes(frame, "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/set_zero":
                self._send_json({"ok": state.set_zero()})
            elif path == "/api/clear_zero":
                state.clear_zero()
                self._send_json({"ok": True})
            elif path == "/api/reset_stats":
                state.reset_stats()
                self._send_json({"ok": True})
            elif path == "/api/enable_outside_logging":
                self._send_json({"ok": state.enable_outside_logging()})
            elif path == "/api/disable_outside_logging":
                self._send_json({"ok": state.disable_outside_logging()})
            elif path == "/api/clear_outside_log":
                self._send_json({"ok": state.clear_outside_log()})
            elif path == "/api/mark_manual_boundary":
                self._send_json(state.mark_manual_boundary())
            elif path == "/api/start_manual_boundary_sampling":
                self._send_json(state.start_manual_boundary_sampling())
            elif path == "/api/stop_manual_boundary_sampling":
                self._send_json(state.stop_manual_boundary_sampling())
            elif path == "/api/clear_manual_boundary_log":
                self._send_json({"ok": state.clear_manual_boundary_log()})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_json(self, payload: dict) -> None:
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _blank_jpeg() -> bytes:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(image, "waiting for camera", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    return bytes(encoded) if ok else b""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Stewart boundary model UI using RealSense ArUco pose.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--warning-margin-mm", type=float, default=DEFAULT_WARNING_MARGIN_MM)
    parser.add_argument("--upper-radius-mm", type=float, default=DEFAULT_UPPER_RADIUS_MM)
    parser.add_argument("--outside-log", type=Path, default=DEFAULT_OUTSIDE_LOG)
    parser.add_argument("--outside-log-interval-s", type=float, default=DEFAULT_OUTSIDE_LOG_INTERVAL_S)
    parser.add_argument("--disable-outside-logging", action="store_true")
    parser.add_argument("--manual-boundary-log", type=Path, default=DEFAULT_MANUAL_BOUNDARY_LOG)
    parser.add_argument("--manual-boundary-weight", type=float, default=DEFAULT_MANUAL_BOUNDARY_WEIGHT)
    parser.add_argument("--manual-boundary-sample-interval-s", type=float, default=DEFAULT_MANUAL_BOUNDARY_SAMPLE_INTERVAL_S)
    parser.add_argument("--marker-id", type=int, default=2)
    parser.add_argument("--marker-size", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--dictionary", type=int, default=cv2.aruco.DICT_6X6_250)
    return parser


def run_live_ui(args: argparse.Namespace) -> int:
    model = load_boundary_model(args.model)
    outside_logger = OutsideCandidateLogger(
        args.outside_log,
        min_interval_s=args.outside_log_interval_s,
        enabled=not args.disable_outside_logging,
    )
    manual_boundary_logger = ManualBoundaryLogger(
        args.manual_boundary_log,
        sample_weight=args.manual_boundary_weight,
    )
    state = LiveBoundaryState(
        model=model,
        warning_margin_mm=args.warning_margin_mm,
        upper_radius_mm=args.upper_radius_mm,
        outside_logger=outside_logger,
        manual_boundary_logger=manual_boundary_logger,
        manual_boundary_sample_interval_s=args.manual_boundary_sample_interval_s,
    )
    poller = CameraPoller(
        state=state,
        marker_id=args.marker_id,
        marker_size=args.marker_size,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        dictionary=args.dictionary,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    try:
        poller.start()
        print(f"Boundary model live UI: http://{args.host}:{args.port}")
        print(f"Model: {args.model} ({model.model_type})")
        print("Place Stewart at zero pose, then click Set Zero in the UI.")
        print(f"Outside candidates: {args.outside_log}")
        print(f"Manual boundary marks: {args.manual_boundary_log}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()
        poller.stop()
    return 0


def main() -> int:
    return run_live_ui(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
