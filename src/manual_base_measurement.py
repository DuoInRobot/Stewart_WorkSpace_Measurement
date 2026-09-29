#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import select as select_module
import sys
import termios
import time
import tty
from typing import Optional, Sequence

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data" / "raw"

@dataclass(frozen=True)
class WrenchSample:
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]
    sequence: Optional[int] = None


@dataclass(frozen=True)
class CameraPose:
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


def norm3(value: Sequence[float]) -> float:
    return math.sqrt(float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2)


def rotation_matrix_to_rpy_rad(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rotation[0, 0]) ** 2 + float(rotation[1, 0]) ** 2)
    singular = sy < 1e-9
    if singular:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = 0.0
    else:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return (roll, pitch, yaw)


def camera_pose_from_marker_pose(rvec_marker_in_camera: np.ndarray, tvec_marker_in_camera: np.ndarray) -> CameraPose:
    rvec = np.asarray(rvec_marker_in_camera, dtype=float).reshape(3, 1)
    tvec = np.asarray(tvec_marker_in_camera, dtype=float).reshape(3, 1)
    marker_rotation, _ = cv2.Rodrigues(rvec)
    camera_rotation = marker_rotation.T
    camera_translation = -camera_rotation @ tvec
    return CameraPose(
        position_m=tuple(float(value) for value in camera_translation.flatten()),
        rpy_rad=rotation_matrix_to_rpy_rad(camera_rotation),
    )


def select_detection(detections, marker_id: Optional[int]):
    if not detections:
        return None
    if marker_id is not None and marker_id in detections:
        return marker_id, detections[marker_id]
    selected_id = sorted(detections.keys())[0]
    return selected_id, detections[selected_id]


CSV_FIELDS = [
    "sample_index",
    "timestamp_s",
    "elapsed_s",
    "camera_host_monotonic_ns",
    "force_host_monotonic_ns",
    "sync_delta_ms",
    "operator_mode",
    "trajectory_id",
    "segment_id",
    "event_code",
    "marker_detected",
    "pose_valid",
    "marker_id",
    "reproj_error_px",
    "marker_in_camera_x_m",
    "marker_in_camera_y_m",
    "marker_in_camera_z_m",
    "marker_in_camera_roll_rad",
    "marker_in_camera_pitch_rad",
    "marker_in_camera_yaw_rad",
    "marker_in_camera_rvec_x_rad",
    "marker_in_camera_rvec_y_rad",
    "marker_in_camera_rvec_z_rad",
    "camera_in_marker_x_m",
    "camera_in_marker_y_m",
    "camera_in_marker_z_m",
    "camera_in_marker_roll_rad",
    "camera_in_marker_pitch_rad",
    "camera_in_marker_yaw_rad",
    "camera_in_marker_rvec_x_rad",
    "camera_in_marker_rvec_y_rad",
    "camera_in_marker_rvec_z_rad",
    "force_fresh",
    "force_sequence",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "torque_x_nm",
    "torque_y_nm",
    "torque_z_nm",
    "force_norm_n",
    "torque_norm_nm",
]


@dataclass(frozen=True)
class ObservedPose:
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    rvec_rad: tuple[float, float, float]


@dataclass(frozen=True)
class PosePair:
    marker_id: int
    marker_in_camera: ObservedPose
    camera_in_marker: ObservedPose
    reproj_error_px: float
    pose_valid: bool


@dataclass(frozen=True)
class ManualSample:
    sample_index: int
    timestamp_s: float
    elapsed_s: float
    camera_host_monotonic_ns: int
    force_host_monotonic_ns: int
    operator_mode: str
    trajectory_id: int
    segment_id: int
    event_code: str
    pose_pair: Optional[PosePair]
    wrench: Optional[WrenchSample]


@dataclass
class OperatorState:
    operator_mode: str = "unknown"
    trajectory_id: int = 0
    segment_id: int = 0
    _pending_event_code: str = "start"

    def handle_key(self, key: int) -> bool:
        if is_stop_key(key):
            return True
        try:
            char = chr(key).lower()
        except ValueError:
            return False
        mode_by_key = {
            "r": "reference",
            "i": "interior",
            "b": "boundary",
            "u": "unknown",
        }
        if char in mode_by_key:
            self.operator_mode = mode_by_key[char]
            self.segment_id += 1
            self._pending_event_code = f"mode_{self.operator_mode}"
        elif char == "n":
            self.trajectory_id += 1
            self.segment_id += 1
            self._pending_event_code = "new_trajectory"
        return False

    def consume_event_code(self) -> str:
        event_code = self._pending_event_code
        self._pending_event_code = ""
        return event_code


class TerminalKeyReader:
    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self._fd: Optional[int] = None
        self._old_settings = None

    def __enter__(self):
        if self.stream is not None and self.stream.isatty():
            self._fd = self.stream.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> Optional[int]:
        if self._fd is None:
            return None
        readable, _, _ = select_module.select([self.stream], [], [], 0.0)
        if not readable:
            return None
        char = self.stream.read(1)
        if not char:
            return None
        return ord(char)


def force_sensor_module_paths(force_sensor_python_dir: Path) -> list[Path]:
    linux_py310_dir = force_sensor_python_dir.parent
    return [force_sensor_python_dir, linux_py310_dir / "build"]


def default_output_path(output_dir: Path = DEFAULT_RAW_DIR, *, now: Optional[datetime] = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"manual_base_samples_{timestamp}.csv"


class ForceSensorReader:
    def __init__(
        self,
        *,
        sensor_python_dir: Path,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        sensor_name: str = "D80",
        max_age_ms: float = 100.0,
        max_force_n: float = 240.0,
        max_torque_nm: float = 6.0,
        clear_zero: bool = True,
        enable_logging: bool = False,
    ) -> None:
        self.sensor_python_dir = sensor_python_dir
        self.port = port
        self.baud = baud
        self.sensor_name = sensor_name
        self.max_age_ms = max_age_ms
        self.max_force_n = max_force_n
        self.max_torque_nm = max_torque_nm
        self.clear_zero = clear_zero
        self.enable_logging = enable_logging
        self._sensor = None

    def start(self) -> None:
        for module_path in reversed(force_sensor_module_paths(self.sensor_python_dir)):
            if str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))

        import force_sensor

        options = force_sensor.ForceSensorOptions()
        options.limits = force_sensor.WrenchLimits(self.max_force_n, self.max_torque_nm)
        options.enable_logging = self.enable_logging
        sensor = force_sensor.ForceSensor(self.port, self.baud, self.sensor_name, options)
        if not sensor.connect(self.clear_zero):
            raise RuntimeError("failed to connect force sensor")
        if not sensor.start():
            sensor.disconnect()
            raise RuntimeError("failed to start force sensor")
        self._sensor = sensor

    def read(self) -> Optional[WrenchSample]:
        if self._sensor is None:
            return None
        fresh, sample = self._sensor.latest_fresh(self.max_age_ms)
        if not fresh:
            return None
        return WrenchSample(
            force_n=(float(sample.force.x), float(sample.force.y), float(sample.force.z)),
            torque_nm=(float(sample.torque.x), float(sample.torque.y), float(sample.torque.z)),
            sequence=int(sample.sequence),
        )

    def stop(self) -> None:
        if self._sensor is not None:
            self._sensor.disconnect()
            self._sensor = None


class NullForceSensorReader:
    """Pose-only reader used when no external force-sensor driver is configured."""

    def start(self) -> None:
        return None

    def read(self) -> None:
        return None

    def latest(self) -> None:
        return None

    def stop(self) -> None:
        return None


def pose_pair_from_detection(
    *,
    marker_id: int,
    rvec_marker_in_camera: np.ndarray,
    tvec_marker_in_camera: np.ndarray,
    reproj_error_px: float,
    max_reproj_error_px: float,
) -> PosePair:
    rvec = np.asarray(rvec_marker_in_camera, dtype=float).reshape(3, 1)
    tvec = np.asarray(tvec_marker_in_camera, dtype=float).reshape(3, 1)
    marker_rotation, _ = cv2.Rodrigues(rvec)
    camera_rotation = marker_rotation.T
    camera_translation = -camera_rotation @ tvec
    camera_rvec, _ = cv2.Rodrigues(camera_rotation)
    error = float(reproj_error_px)
    return PosePair(
        marker_id=int(marker_id),
        marker_in_camera=ObservedPose(
            position_m=tuple(float(value) for value in tvec.flatten()),
            rpy_rad=rotation_matrix_to_rpy_rad(marker_rotation),
            rvec_rad=tuple(float(value) for value in rvec.flatten()),
        ),
        camera_in_marker=ObservedPose(
            position_m=tuple(float(value) for value in camera_translation.flatten()),
            rpy_rad=rotation_matrix_to_rpy_rad(camera_rotation),
            rvec_rad=tuple(float(value) for value in camera_rvec.flatten()),
        ),
        reproj_error_px=error,
        pose_valid=math.isfinite(error) and error <= float(max_reproj_error_px),
    )


def _format_vector(prefix: str, values: Sequence[float]) -> dict[str, str]:
    return {
        f"{prefix}_x_m": f"{float(values[0]):.6f}",
        f"{prefix}_y_m": f"{float(values[1]):.6f}",
        f"{prefix}_z_m": f"{float(values[2]):.6f}",
    }


def _format_rpy(prefix: str, values: Sequence[float]) -> dict[str, str]:
    return {
        f"{prefix}_roll_rad": f"{float(values[0]):.6f}",
        f"{prefix}_pitch_rad": f"{float(values[1]):.6f}",
        f"{prefix}_yaw_rad": f"{float(values[2]):.6f}",
    }


def _format_rvec(prefix: str, values: Sequence[float]) -> dict[str, str]:
    return {
        f"{prefix}_rvec_x_rad": f"{float(values[0]):.6f}",
        f"{prefix}_rvec_y_rad": f"{float(values[1]):.6f}",
        f"{prefix}_rvec_z_rad": f"{float(values[2]):.6f}",
    }


def sample_to_row(sample: ManualSample) -> dict[str, str]:
    pair = sample.pose_pair
    wrench = sample.wrench
    row = {
        "sample_index": str(sample.sample_index),
        "timestamp_s": f"{sample.timestamp_s:.6f}",
        "elapsed_s": f"{sample.elapsed_s:.6f}",
        "camera_host_monotonic_ns": str(int(sample.camera_host_monotonic_ns)),
        "force_host_monotonic_ns": str(int(sample.force_host_monotonic_ns)),
        "sync_delta_ms": f"{(sample.force_host_monotonic_ns - sample.camera_host_monotonic_ns) / 1_000_000.0:.3f}",
        "operator_mode": sample.operator_mode,
        "trajectory_id": str(sample.trajectory_id),
        "segment_id": str(sample.segment_id),
        "event_code": sample.event_code,
    }

    if pair is None:
        missing_pose = (math.nan, math.nan, math.nan)
        row.update(
            {
                "marker_detected": "0",
                "pose_valid": "0",
                "marker_id": "",
                "reproj_error_px": "nan",
            }
        )
        row.update(_format_vector("marker_in_camera", missing_pose))
        row.update(_format_rpy("marker_in_camera", missing_pose))
        row.update(_format_rvec("marker_in_camera", missing_pose))
        row.update(_format_vector("camera_in_marker", missing_pose))
        row.update(_format_rpy("camera_in_marker", missing_pose))
        row.update(_format_rvec("camera_in_marker", missing_pose))
    else:
        row.update(
            {
                "marker_detected": "1",
                "pose_valid": "1" if pair.pose_valid else "0",
                "marker_id": str(pair.marker_id),
                "reproj_error_px": f"{pair.reproj_error_px:.6f}",
            }
        )
        row.update(_format_vector("marker_in_camera", pair.marker_in_camera.position_m))
        row.update(_format_rpy("marker_in_camera", pair.marker_in_camera.rpy_rad))
        row.update(_format_rvec("marker_in_camera", pair.marker_in_camera.rvec_rad))
        row.update(_format_vector("camera_in_marker", pair.camera_in_marker.position_m))
        row.update(_format_rpy("camera_in_marker", pair.camera_in_marker.rpy_rad))
        row.update(_format_rvec("camera_in_marker", pair.camera_in_marker.rvec_rad))

    if wrench is None:
        row.update(
            {
                "force_fresh": "0",
                "force_sequence": "",
                "force_x_n": "nan",
                "force_y_n": "nan",
                "force_z_n": "nan",
                "torque_x_nm": "nan",
                "torque_y_nm": "nan",
                "torque_z_nm": "nan",
                "force_norm_n": "nan",
                "torque_norm_nm": "nan",
            }
        )
    else:
        row.update(
            {
                "force_fresh": "1",
                "force_sequence": "" if wrench.sequence is None else str(wrench.sequence),
                "force_x_n": f"{wrench.force_n[0]:.6f}",
                "force_y_n": f"{wrench.force_n[1]:.6f}",
                "force_z_n": f"{wrench.force_n[2]:.6f}",
                "torque_x_nm": f"{wrench.torque_nm[0]:.6f}",
                "torque_y_nm": f"{wrench.torque_nm[1]:.6f}",
                "torque_z_nm": f"{wrench.torque_nm[2]:.6f}",
                "force_norm_n": f"{norm3(wrench.force_n):.6f}",
                "torque_norm_nm": f"{norm3(wrench.torque_nm):.6f}",
            }
        )
    return row


def is_stop_key(key: int) -> bool:
    return key in (27, ord("q"), ord("Q"))


def draw_preview(
    frame: np.ndarray,
    *,
    detector,
    detections,
    pose_pair: Optional[PosePair],
    wrench: Optional[WrenchSample],
    operator_state: OperatorState,
    sample_index: int,
    elapsed_s: float,
) -> np.ndarray:
    image = detector.draw(frame, detections)
    lines = [
        f"sample={sample_index}  elapsed={elapsed_s:.1f}s  "
        f"mode={operator_state.operator_mode} traj={operator_state.trajectory_id} seg={operator_state.segment_id}",
        "keys: r reference | i interior | b boundary | u unknown | n new trajectory | q/Esc stop",
    ]
    if pose_pair is None:
        lines.append("marker: not detected")
    else:
        x, y, z = pose_pair.camera_in_marker.position_m
        lines.append(
            f"camera in marker: x={x:.3f} y={y:.3f} z={z:.3f} m  "
            f"reproj={pose_pair.reproj_error_px:.3f}px valid={int(pose_pair.pose_valid)}"
        )
    if wrench is None:
        lines.append("force: no fresh sample")
    else:
        fx, fy, fz = wrench.force_n
        mx, my, mz = wrench.torque_nm
        lines.append(f"F(N): {fx:.2f}, {fy:.2f}, {fz:.2f}  |F|={norm3(wrench.force_n):.2f}")
        lines.append(f"M(Nm): {mx:.3f}, {my:.3f}, {mz:.3f}  |M|={norm3(wrench.torque_nm):.3f}")

    overlay_height = 12 + len(lines) * 24
    cv2.rectangle(image, (0, 0), (image.shape[1], overlay_height), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return image


def run_acquisition(
    *,
    camera,
    detector,
    force_reader,
    output_path: Path,
    marker_id: Optional[int],
    max_reproj_error_px: float,
    duration_s: float,
    preview: bool,
    status_interval_s: float = 1.0,
    max_samples: Optional[int] = None,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    deadline = None if duration_s <= 0.0 else started_monotonic + duration_s
    next_status = started_monotonic
    sample_count = 0
    operator_state = OperatorState()

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with TerminalKeyReader() as key_reader:
            while deadline is None or time.monotonic() < deadline:
                key = key_reader.read_key()
                if key is not None and operator_state.handle_key(key):
                    break

                if max_samples is not None and sample_count >= max_samples:
                    break

                try:
                    frame = camera.read()
                except KeyboardInterrupt:
                    break
                camera_host_monotonic_ns = time.monotonic_ns()
                detections = detector.detect(frame)
                selected = select_detection(detections, marker_id)
                pose_pair = None
                if selected is not None:
                    detected_id, (rvec, tvec, reproj_error_px, _corners) = selected
                    pose_pair = pose_pair_from_detection(
                        marker_id=detected_id,
                        rvec_marker_in_camera=rvec,
                        tvec_marker_in_camera=tvec,
                        reproj_error_px=reproj_error_px,
                        max_reproj_error_px=max_reproj_error_px,
                    )

                wrench = force_reader.read()
                force_host_monotonic_ns = time.monotonic_ns()
                timestamp_s = time.time()
                sample_count += 1
                elapsed_s = time.monotonic() - started_monotonic
                sample = ManualSample(
                    sample_index=sample_count,
                    timestamp_s=timestamp_s,
                    elapsed_s=elapsed_s,
                    camera_host_monotonic_ns=camera_host_monotonic_ns,
                    force_host_monotonic_ns=force_host_monotonic_ns,
                    operator_mode=operator_state.operator_mode,
                    trajectory_id=operator_state.trajectory_id,
                    segment_id=operator_state.segment_id,
                    event_code=operator_state.consume_event_code(),
                    pose_pair=pose_pair,
                    wrench=wrench,
                )
                writer.writerow(sample_to_row(sample))
                handle.flush()

                now_monotonic = time.monotonic()
                if status_interval_s > 0.0 and now_monotonic >= next_status:
                    marker_status = "missing" if pose_pair is None else (
                        f"id={pose_pair.marker_id} reproj={pose_pair.reproj_error_px:.3f}px"
                    )
                    force_status = "stale" if wrench is None else f"|F|={norm3(wrench.force_n):.2f}N"
                    print(
                        f"sample={sample_count} elapsed={elapsed_s:.1f}s "
                        f"mode={operator_state.operator_mode} traj={operator_state.trajectory_id} "
                        f"seg={operator_state.segment_id} marker={marker_status} force={force_status}"
                    )
                    next_status = now_monotonic + status_interval_s

                if preview:
                    image = draw_preview(
                        frame,
                        detector=detector,
                        detections=detections,
                        pose_pair=pose_pair,
                        wrench=wrench,
                        operator_state=operator_state,
                        sample_index=sample_count,
                        elapsed_s=elapsed_s,
                    )
                    cv2.imshow("Manual Stewart Base Measurement", image)
                    preview_key = cv2.waitKey(1) & 0xFF
                    if operator_state.handle_key(preview_key):
                        break

    return sample_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously record a hand-moved Stewart base marker pose and wrist force data."
    )
    parser.add_argument("--output", default=None, help="CSV path; default creates a timestamped file.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 runs until q, Esc, or Ctrl+C.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV preview window.")
    parser.add_argument("--status-interval-s", type=float, default=1.0)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--dictionary", type=int, default=cv2.aruco.DICT_6X6_250)
    parser.add_argument("--marker-id", type=int, default=2)
    parser.add_argument("--marker-size", type=float, default=0.03, help="ArUco marker side length in meters.")
    parser.add_argument("--max-reproj-error-px", type=float, default=1.0)

    parser.add_argument(
        "--force-sensor-python-dir",
        type=Path,
        default=None,
        help="Optional external force-sensor driver directory; omit for pose-only capture.",
    )
    parser.add_argument("--force-port", default="/dev/ttyUSB0")
    parser.add_argument("--force-baud", type=int, default=115200)
    parser.add_argument("--force-sensor-name", default="D80")
    parser.add_argument("--force-max-age-ms", type=float, default=100.0)
    parser.add_argument("--force-max-force", type=float, default=240.0)
    parser.add_argument("--force-max-torque", type=float, default=6.0)
    parser.add_argument("--force-logging", action="store_true")
    parser.add_argument("--no-clear-zero", action="store_true")
    return parser


def run_manual_base_measurement(args: argparse.Namespace) -> int:
    from arudo_detector import ArUcoDetector, RealSenseColorCamera

    output_path = Path(args.output) if args.output else default_output_path()
    camera = None
    force_reader = None
    sample_count = 0
    try:
        camera = RealSenseColorCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.serial,
        )
        intrinsics = camera.start()
        detector = ArUcoDetector(
            intrinsics=intrinsics,
            marker_size=args.marker_size,
            dictionary_id=args.dictionary,
        )
        force_reader = (
            NullForceSensorReader()
            if args.force_sensor_python_dir is None
            else ForceSensorReader(
                sensor_python_dir=args.force_sensor_python_dir,
                port=args.force_port,
                baud=args.force_baud,
                sensor_name=args.force_sensor_name,
                max_age_ms=args.force_max_age_ms,
                max_force_n=args.force_max_force,
                max_torque_nm=args.force_max_torque,
                clear_zero=not args.no_clear_zero,
                enable_logging=args.force_logging,
            )
        )
        force_reader.start()

        print("manual Stewart base measurement started")
        print(f"output: {output_path}")
        print("move the bottom platform by hand; press q, Esc, or Ctrl+C to stop")
        sample_count = run_acquisition(
            camera=camera,
            detector=detector,
            force_reader=force_reader,
            output_path=output_path,
            marker_id=args.marker_id,
            max_reproj_error_px=args.max_reproj_error_px,
            duration_s=args.duration_s,
            preview=not args.no_preview,
            status_interval_s=args.status_interval_s,
        )
    except KeyboardInterrupt:
        print("measurement interrupted")
    finally:
        if force_reader is not None:
            force_reader.stop()
        if camera is not None:
            camera.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"measurement finished, samples={sample_count}")
    return 0


def main() -> int:
    return run_manual_base_measurement(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
