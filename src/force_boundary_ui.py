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
import random
import sys
import threading
import time
from typing import Optional, Sequence
from urllib.parse import parse_qs

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "raw" / "force_boundary_marks.csv"

from manual_base_measurement import (
    camera_pose_from_marker_pose,
    select_detection,
)
from arudo_detector import ArUcoDetector, RealSenseColorCamera  # noqa: E402

CSV_FIELDS = [
    "mark_index",
    "timestamp_s",
    "marker_id",
    "camera_x_m",
    "camera_y_m",
    "camera_z_m",
    "camera_roll_rad",
    "camera_pitch_rad",
    "camera_yaw_rad",
    "camera_reproj_error_px",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "torque_x_nm",
    "torque_y_nm",
    "torque_z_nm",
    "force_norm_n",
    "torque_norm_nm",
    "tcp_x_mm",
    "tcp_y_mm",
    "tcp_z_mm",
    "tcp_roll_deg",
    "tcp_pitch_deg",
    "tcp_yaw_deg",
    "note",
]


@dataclass(frozen=True)
class Wrench:
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]


@dataclass(frozen=True)
class RobotPose:
    tcp_pose: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ObservedCameraPose:
    marker_id: int
    position_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    reproj_error_px: float


@dataclass(frozen=True)
class Snapshot:
    timestamp_s: float
    wrench: Wrench
    robot_pose: RobotPose
    camera_pose: Optional[ObservedCameraPose] = None


@dataclass(frozen=True)
class ToolJogCommand:
    delta: tuple[float, float, float, float, float, float]
    speed_mm_s: float


TOOL_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
LINEAR_STEP_RANGE_MM = (0.01, 2.0)
ANGULAR_STEP_RANGE_DEG = (0.01, 1.0)
JOG_SPEED_RANGE_MM_S = (0.1, 10.0)


def norm3(value: Sequence[float]) -> float:
    return math.sqrt(float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2)


def force_sensor_module_paths(force_sensor_python_dir: Path) -> list[Path]:
    linux_py310_dir = force_sensor_python_dir.parent
    return [
        force_sensor_python_dir,
        linux_py310_dir / "build",
    ]


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _bounded_float(value, *, name: str, bounds: tuple[float, float]) -> float:
    result = float(value)
    if not math.isfinite(result) or result < bounds[0] or result > bounds[1]:
        raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]}")
    return result


def build_tool_jog_command(
    *,
    axis: str,
    direction: int,
    linear_step_mm: float,
    angular_step_deg: float,
    speed_mm_s: float,
) -> ToolJogCommand:
    axis = str(axis).lower()
    if axis not in TOOL_AXIS_INDEX:
        raise ValueError(f"unknown tool axis: {axis}")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    linear_step = _bounded_float(linear_step_mm, name="linear_step_mm", bounds=LINEAR_STEP_RANGE_MM)
    angular_step = _bounded_float(
        angular_step_deg,
        name="angular_step_deg",
        bounds=ANGULAR_STEP_RANGE_DEG,
    )
    speed = _bounded_float(speed_mm_s, name="speed_mm_s", bounds=JOG_SPEED_RANGE_MM_S)
    step = linear_step if TOOL_AXIS_INDEX[axis] < 3 else angular_step
    delta = [0.0] * 6
    delta[TOOL_AXIS_INDEX[axis]] = float(direction) * step
    return ToolJogCommand(delta=tuple(delta), speed_mm_s=speed)


def parse_tool_jog_request(request: dict) -> ToolJogCommand:
    try:
        direction = int(request.get("direction"))
    except (TypeError, ValueError) as exc:
        raise ValueError("direction must be -1 or 1") from exc
    return build_tool_jog_command(
        axis=str(request.get("axis", "")),
        direction=direction,
        linear_step_mm=request.get("linear_step_mm"),
        angular_step_deg=request.get("angular_step_deg"),
        speed_mm_s=request.get("speed_mm_s"),
    )


def execute_tool_jog(arm, command: ToolJogCommand, *, acceleration: float, timeout_s: float) -> None:
    x, y, z, roll, pitch, yaw = command.delta
    code = arm.set_tool_position(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=command.speed_mm_s,
        mvacc=float(acceleration),
        is_radian=False,
        wait=True,
        timeout=float(timeout_s),
        radius=None,
    )
    if code != 0:
        raise RuntimeError(f"set_tool_position failed, code={code}")


def motion_interlock_error(
    snapshot: Optional[Snapshot],
    *,
    force_limit_n: float,
    torque_limit_nm: float,
) -> Optional[str]:
    if snapshot is None:
        return "no fresh sensor snapshot available"
    if snapshot.camera_pose is None:
        return "requested marker is not detected"
    if norm3(snapshot.wrench.force_n) >= float(force_limit_n):
        return f"force hard limit reached ({force_limit_n} N)"
    if norm3(snapshot.wrench.torque_nm) >= float(torque_limit_nm):
        return f"torque hard limit reached ({torque_limit_nm} N*m)"
    return None


def _require_sdk_success(operation: str, code: int) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed, code={code}")


def prepare_xarm_motion(arm) -> None:
    _require_sdk_success("motion_enable", arm.motion_enable(enable=True))
    _require_sdk_success("set_mode", arm.set_mode(0))
    _require_sdk_success("set_state", arm.set_state(0))


def stop_xarm_motion(arm) -> None:
    _require_sdk_success("set_state(stop)", arm.set_state(4))


def observed_camera_pose_from_detection(
    *,
    marker_id: int,
    rvec_marker_in_camera: np.ndarray,
    tvec_marker_in_camera: np.ndarray,
    reproj_error_px: float,
) -> ObservedCameraPose:
    pose = camera_pose_from_marker_pose(rvec_marker_in_camera, tvec_marker_in_camera)
    return ObservedCameraPose(
        marker_id=marker_id,
        position_m=pose.position_m,
        rpy_rad=pose.rpy_rad,
        reproj_error_px=float(reproj_error_px),
    )


def read_camera_observation(camera, detector, *, requested_marker_id: Optional[int]) -> Optional[ObservedCameraPose]:
    frame = camera.read()
    selected = select_detection(detector.detect(frame), requested_marker_id)
    if selected is None:
        return None

    marker_id, (rvec, tvec, reproj_error_px, _corners) = selected
    return observed_camera_pose_from_detection(
        marker_id=marker_id,
        rvec_marker_in_camera=rvec,
        tvec_marker_in_camera=tvec,
        reproj_error_px=reproj_error_px,
    )


def build_mark_record(*, mark_index: int, snapshot: Snapshot, note: str = "") -> dict[str, str]:
    force = snapshot.wrench.force_n
    torque = snapshot.wrench.torque_nm
    tcp = snapshot.robot_pose.tcp_pose
    camera = snapshot.camera_pose
    if camera is None:
        raise ValueError("camera pose is required to record a boundary mark")

    return {
        "mark_index": str(mark_index),
        "timestamp_s": format_float(snapshot.timestamp_s),
        "marker_id": str(camera.marker_id),
        "camera_x_m": format_float(camera.position_m[0]),
        "camera_y_m": format_float(camera.position_m[1]),
        "camera_z_m": format_float(camera.position_m[2]),
        "camera_roll_rad": format_float(camera.rpy_rad[0]),
        "camera_pitch_rad": format_float(camera.rpy_rad[1]),
        "camera_yaw_rad": format_float(camera.rpy_rad[2]),
        "camera_reproj_error_px": format_float(camera.reproj_error_px),
        "force_x_n": format_float(force[0]),
        "force_y_n": format_float(force[1]),
        "force_z_n": format_float(force[2]),
        "torque_x_nm": format_float(torque[0]),
        "torque_y_nm": format_float(torque[1]),
        "torque_z_nm": format_float(torque[2]),
        "force_norm_n": format_float(norm3(force)),
        "torque_norm_nm": format_float(norm3(torque)),
        "tcp_x_mm": format_float(tcp[0]),
        "tcp_y_mm": format_float(tcp[1]),
        "tcp_z_mm": format_float(tcp[2]),
        "tcp_roll_deg": format_float(tcp[3]),
        "tcp_pitch_deg": format_float(tcp[4]),
        "tcp_yaw_deg": format_float(tcp[5]),
        "note": note,
    }


def snapshot_to_json(
    snapshot: Optional[Snapshot],
    *,
    status: str,
    mark_count: int,
    motion_enabled: bool = False,
) -> dict:
    if snapshot is None:
        return {
            "ok": False,
            "status": status,
            "mark_count": mark_count,
            "timestamp_s": None,
            "force_n": [None, None, None],
            "torque_nm": [None, None, None],
            "force_norm_n": None,
            "torque_norm_nm": None,
            "tcp_pose": [None, None, None, None, None, None],
            "mark_allowed": False,
            "marker_id": None,
            "camera_position_m": [None, None, None],
            "camera_rpy_rad": [None, None, None],
            "camera_reproj_error_px": None,
            "motion_enabled": motion_enabled,
        }

    camera = snapshot.camera_pose
    return {
        "ok": True,
        "status": status,
        "mark_count": mark_count,
        "timestamp_s": snapshot.timestamp_s,
        "force_n": list(snapshot.wrench.force_n),
        "torque_nm": list(snapshot.wrench.torque_nm),
        "force_norm_n": norm3(snapshot.wrench.force_n),
        "torque_norm_nm": norm3(snapshot.wrench.torque_nm),
        "tcp_pose": list(snapshot.robot_pose.tcp_pose),
        "mark_allowed": camera is not None,
        "marker_id": camera.marker_id if camera is not None else None,
        "camera_position_m": list(camera.position_m) if camera is not None else [None, None, None],
        "camera_rpy_rad": list(camera.rpy_rad) if camera is not None else [None, None, None],
        "camera_reproj_error_px": camera.reproj_error_px if camera is not None else None,
        "motion_enabled": motion_enabled,
    }


class BoundaryRecorder:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._validate_existing_header()
        self._mark_count = self._existing_mark_count()

        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            with self.output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()

    @property
    def mark_count(self) -> int:
        with self._lock:
            return self._mark_count

    def _existing_mark_count(self) -> int:
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return 0

        with self.output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)

    def _validate_existing_header(self) -> None:
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return

        with self.output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_FIELDS:
                raise ValueError(
                    f"incompatible CSV header in {self.output_path}; "
                    "choose a new --output path for camera boundary marks"
                )

    def record(self, snapshot: Snapshot, note: str = "") -> dict[str, str]:
        if snapshot.camera_pose is None:
            raise ValueError("camera pose is required to record a boundary mark")

        with self._lock:
            self._mark_count += 1
            record = build_mark_record(
                mark_index=self._mark_count,
                snapshot=snapshot,
                note=note,
            )
            with self.output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writerow(record)
            return record


class SensorRobotSource:
    def __init__(
        self,
        *,
        force_sensor_python_dir: Path,
        force_port: str,
        force_baud: int,
        force_sensor_name: str,
        force_max_age_ms: float,
        force_max_force: float,
        force_max_torque: float,
        clear_zero: bool,
        robot_ip: str,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        camera_serial: Optional[str],
        camera_dictionary: int,
        marker_size: float,
        marker_id: Optional[int],
        jog_acceleration: float = 20.0,
        jog_timeout_s: float = 10.0,
    ) -> None:
        self.force_sensor_python_dir = force_sensor_python_dir
        self.force_port = force_port
        self.force_baud = force_baud
        self.force_sensor_name = force_sensor_name
        self.force_max_age_ms = force_max_age_ms
        self.force_max_force = force_max_force
        self.force_max_torque = force_max_torque
        self.clear_zero = clear_zero
        self.robot_ip = robot_ip
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.camera_serial = camera_serial
        self.camera_dictionary = camera_dictionary
        self.marker_size = marker_size
        self.marker_id = marker_id
        self.jog_acceleration = float(jog_acceleration)
        self.jog_timeout_s = float(jog_timeout_s)
        self._sensor = None
        self._arm = None
        self._camera = None
        self._detector = None
        self._motion_enabled = False

    def start(self) -> None:
        for module_path in reversed(force_sensor_module_paths(self.force_sensor_python_dir)):
            if str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))

        import force_sensor
        from xarm.wrapper import XArmAPI

        sensor = None
        arm = None
        camera = None
        try:
            options = force_sensor.ForceSensorOptions()
            options.limits = force_sensor.WrenchLimits(self.force_max_force, self.force_max_torque)
            options.enable_logging = False

            sensor = force_sensor.ForceSensor(
                self.force_port,
                self.force_baud,
                self.force_sensor_name,
                options,
            )
            if not sensor.connect(self.clear_zero):
                raise RuntimeError("failed to connect force sensor")
            if not sensor.start():
                raise RuntimeError("failed to start force sensor")

            arm = XArmAPI(self.robot_ip)
            if not arm.connected:
                raise RuntimeError("failed to connect xArm")

            camera = RealSenseColorCamera(
                width=self.camera_width,
                height=self.camera_height,
                fps=self.camera_fps,
                serial=self.camera_serial,
            )
            intrinsics = camera.start()
            detector = ArUcoDetector(
                intrinsics=intrinsics,
                marker_size=self.marker_size,
                dictionary_id=self.camera_dictionary,
            )
        except Exception:
            if camera is not None:
                camera.stop()
            if arm is not None:
                arm.disconnect()
            if sensor is not None:
                sensor.disconnect()
            raise

        self._sensor = sensor
        self._arm = arm
        self._camera = camera
        self._detector = detector

    def read(self) -> Optional[Snapshot]:
        if self._sensor is None or self._arm is None or self._camera is None or self._detector is None:
            return None

        fresh, sample = self._sensor.latest_fresh(self.force_max_age_ms)
        if not fresh:
            return None

        code, pose = self._arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError(f"get_position failed, code={code}")
        if len(pose) < 6:
            raise RuntimeError(f"get_position returned invalid pose: {pose}")

        camera_pose = read_camera_observation(
            self._camera,
            self._detector,
            requested_marker_id=self.marker_id,
        )

        return Snapshot(
            timestamp_s=time.time(),
            wrench=Wrench(
                force_n=(float(sample.force.x), float(sample.force.y), float(sample.force.z)),
                torque_nm=(float(sample.torque.x), float(sample.torque.y), float(sample.torque.z)),
            ),
            robot_pose=RobotPose(
                tcp_pose=tuple(float(value) for value in pose[:6]),
            ),
            camera_pose=camera_pose,
        )

    def stop(self) -> None:
        if self._sensor is not None:
            self._sensor.disconnect()
            self._sensor = None
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._detector = None

    @property
    def motion_enabled(self) -> bool:
        return self._motion_enabled

    def prepare_motion(self) -> None:
        if self._arm is None:
            raise RuntimeError("xArm is not connected")
        prepare_xarm_motion(self._arm)
        self._motion_enabled = True

    def jog_tool(self, command: ToolJogCommand) -> None:
        if self._arm is None:
            raise RuntimeError("xArm is not connected")
        if not self._motion_enabled:
            raise RuntimeError("tool jog is not enabled")
        try:
            execute_tool_jog(
                self._arm,
                command,
                acceleration=self.jog_acceleration,
                timeout_s=self.jog_timeout_s,
            )
        except Exception:
            self._motion_enabled = False
            raise

    def stop_motion(self) -> None:
        if self._arm is None:
            raise RuntimeError("xArm is not connected")
        try:
            stop_xarm_motion(self._arm)
        finally:
            self._motion_enabled = False


class MockSource:
    def __init__(self) -> None:
        self._start = time.monotonic()
        self._motion_enabled = False
        self._jog_offset = [0.0] * 6

    def start(self) -> None:
        return None

    def read(self) -> Snapshot:
        elapsed = time.monotonic() - self._start
        return Snapshot(
            timestamp_s=time.time(),
            wrench=Wrench(
                force_n=(
                    8.0 * math.sin(elapsed),
                    6.0 * math.cos(elapsed * 0.7),
                    4.0 * math.sin(elapsed * 1.3),
                ),
                torque_nm=(
                    0.35 * math.sin(elapsed * 0.8),
                    0.25 * math.cos(elapsed * 1.1),
                    0.20 * math.sin(elapsed * 1.7),
                ),
            ),
            robot_pose=RobotPose(
                tcp_pose=(
                    320.0 + 10.0 * math.sin(elapsed * 0.5) + self._jog_offset[0],
                    0.0 + 8.0 * math.cos(elapsed * 0.4) + self._jog_offset[1],
                    210.0 + 6.0 * math.sin(elapsed * 0.6) + self._jog_offset[2],
                    180.0 + self._jog_offset[3],
                    random.uniform(-0.5, 0.5) + self._jog_offset[4],
                    random.uniform(-0.5, 0.5) + self._jog_offset[5],
                )
            ),
            camera_pose=ObservedCameraPose(
                marker_id=2,
                position_m=(
                    0.08 + 0.01 * math.sin(elapsed * 0.4),
                    -0.03 + 0.008 * math.cos(elapsed * 0.5),
                    0.22 + 0.006 * math.sin(elapsed * 0.6),
                ),
                rpy_rad=(
                    0.04 * math.sin(elapsed * 0.5),
                    0.03 * math.cos(elapsed * 0.4),
                    0.02 * math.sin(elapsed * 0.7),
                ),
                reproj_error_px=0.18,
            ),
        )

    def stop(self) -> None:
        return None

    @property
    def motion_enabled(self) -> bool:
        return self._motion_enabled

    def prepare_motion(self) -> None:
        self._motion_enabled = True

    def jog_tool(self, command: ToolJogCommand) -> None:
        if not self._motion_enabled:
            raise RuntimeError("tool jog is not enabled")
        for index, value in enumerate(command.delta):
            self._jog_offset[index] += value

    def stop_motion(self) -> None:
        self._motion_enabled = False


class SamplePoller:
    def __init__(
        self,
        source,
        *,
        interval_s: float,
        jog_force_limit_n: float = 20.0,
        jog_torque_limit_nm: float = 2.0,
    ) -> None:
        self.source = source
        self.interval_s = interval_s
        self.jog_force_limit_n = float(jog_force_limit_n)
        self.jog_torque_limit_nm = float(jog_torque_limit_nm)
        self._lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._latest: Optional[Snapshot] = None
        self._status = "starting"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.source.start()
        self._status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = self.source.read()
                with self._lock:
                    if snapshot is None:
                        self._status = "waiting for fresh sensor sample"
                    else:
                        self._latest = snapshot
                        self._status = "running" if snapshot.camera_pose is not None else "marker not detected"
            except Exception as exc:
                with self._lock:
                    self._latest = None
                    self._status = f"error: {exc}"
            time.sleep(self.interval_s)

    def latest(self) -> tuple[Optional[Snapshot], str]:
        with self._lock:
            return self._latest, self._status

    @property
    def motion_enabled(self) -> bool:
        return bool(getattr(self.source, "motion_enabled", False))

    def _require_motion_interlock(self) -> None:
        snapshot, _status = self.latest()
        error = motion_interlock_error(
            snapshot,
            force_limit_n=self.jog_force_limit_n,
            torque_limit_nm=self.jog_torque_limit_nm,
        )
        if error is not None:
            raise RuntimeError(error)

    def prepare_motion(self) -> None:
        self._require_motion_interlock()
        with self._motion_lock:
            self.source.prepare_motion()

    def jog_tool(self, command: ToolJogCommand) -> None:
        if not self.motion_enabled:
            raise RuntimeError("tool jog is not enabled")
        self._require_motion_interlock()
        with self._motion_lock:
            self.source.jog_tool(command)

    def stop_motion(self) -> None:
        self.source.stop_motion()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.source.stop()


def parse_request_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b""
    content_type = handler.headers.get("Content-Type", "")

    if "application/json" in content_type and raw:
        return json.loads(raw.decode("utf-8"))

    if raw:
        parsed = parse_qs(raw.decode("utf-8"))
        return {key: values[-1] for key, values in parsed.items()}

    return {}


def execute_motion_request(poller: SamplePoller, path: str, request: dict) -> dict:
    if path == "/motion/enable":
        poller.prepare_motion()
    elif path == "/motion/jog":
        poller.jog_tool(parse_tool_jog_request(request))
    elif path == "/motion/stop":
        poller.stop_motion()
    else:
        raise ValueError(f"unknown motion path: {path}")
    return {"ok": True, "motion_enabled": poller.motion_enabled}


def make_handler(poller: SamplePoller, recorder: BoundaryRecorder):
    class ForceBoundaryHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return None

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(INDEX_HTML.encode("utf-8"))
                return

            if self.path.startswith("/events"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    snapshot, status = poller.latest()
                    payload = snapshot_to_json(
                        snapshot,
                        status=status,
                        mark_count=recorder.mark_count,
                        motion_enabled=poller.motion_enabled,
                    )
                    try:
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except BrokenPipeError:
                        return
                    time.sleep(0.2)

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path.startswith("/motion/"):
                request = parse_request_body(self)
                try:
                    result = execute_motion_request(poller, self.path, request)
                except ValueError as exc:
                    self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except RuntimeError as exc:
                    self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.CONFLICT)
                else:
                    self.send_json(result)
                return

            if self.path != "/mark":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            request = parse_request_body(self)
            note = str(request.get("note", "")).strip()
            snapshot, status = poller.latest()

            if snapshot is None:
                self.send_json(
                    {"ok": False, "error": "no fresh snapshot available", "status": status},
                    HTTPStatus.CONFLICT,
                )
                return
            if snapshot.camera_pose is None:
                self.send_json(
                    {"ok": False, "error": "requested marker is not detected", "status": status},
                    HTTPStatus.CONFLICT,
                )
                return

            record = recorder.record(snapshot, note=note)
            self.send_json({"ok": True, "record": record, "mark_count": recorder.mark_count})

    return ForceBoundaryHandler


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stewart 边界测量</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --grid: #e8ecf2;
      --accent: #2457a7;
      --danger: #b42318;
      --success: #16794b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 18px auto;
      display: grid;
      gap: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 15px; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 13px; text-align: right; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(360px, 0.75fr);
      gap: 14px;
      align-items: stretch;
    }
    .trend-stack { display: grid; gap: 14px; }
    .trend-panel + .trend-panel { border-top: 1px solid var(--line); padding-top: 14px; }
    .trend-title { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .trend-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 168px;
      gap: 14px;
      align-items: stretch;
    }
    .chart-wrap { min-width: 0; height: 190px; }
    canvas { display: block; width: 100%; height: 190px; }
    .readouts { display: grid; grid-template-rows: repeat(3, 1fr); border-left: 1px solid var(--line); padding-left: 14px; }
    .readout { display: grid; grid-template-columns: 10px 1fr; column-gap: 8px; align-content: center; }
    .swatch { width: 10px; height: 10px; margin-top: 4px; border-radius: 2px; }
    .readout-label { color: var(--muted); font-size: 12px; }
    .readout-value { display: block; margin-top: 2px; font-size: 21px; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .pose-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .pose-cell { border-bottom: 1px solid var(--line); padding: 8px 0; min-width: 0; }
    .pose-cell span { display: block; color: var(--muted); font-size: 12px; }
    .pose-cell strong { display: block; margin-top: 4px; font-size: 19px; font-variant-numeric: tabular-nums; letter-spacing: 0; overflow-wrap: anywhere; }
    .control-fields { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    input {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 9px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-variant-numeric: tabular-nums;
    }
    .jog-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px 14px; }
    .jog-row { display: grid; grid-template-columns: 40px 42px minmax(38px, 1fr) 42px; gap: 6px; align-items: center; }
    .axis-name { font-weight: 700; text-align: center; }
    .axis-unit { color: var(--muted); font-size: 12px; text-align: center; }
    button {
      height: 38px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    button:disabled { opacity: 0.48; cursor: not-allowed; }
    .jog-button { width: 42px; padding: 0; background: #fff; color: var(--ink); border-color: var(--line); font-size: 20px; line-height: 1; }
    .motion-actions { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
    .stop-button { background: var(--danger); }
    .motion-state { margin-left: auto; color: var(--muted); font-size: 12px; text-align: right; }
    .motion-message { min-height: 20px; margin-top: 8px; color: var(--muted); }
    .mark-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; }
    .last { margin-top: 10px; color: var(--muted); font-variant-numeric: tabular-nums; min-height: 20px; }
    @media (max-width: 920px) {
      .overview-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      main { width: min(100% - 20px, 1280px); margin: 10px auto; }
      header { align-items: flex-start; flex-direction: column; }
      .status { text-align: left; }
      .trend-layout { grid-template-columns: 1fr; }
      .readouts { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: none; border-left: 0; border-top: 1px solid var(--line); padding: 10px 0 0; gap: 8px; }
      .readout-value { font-size: 17px; }
      .pose-grid, .control-fields, .jog-grid, .mark-row { grid-template-columns: 1fr; }
      .jog-row { grid-template-columns: 42px 42px minmax(60px, 1fr) 42px; }
      .motion-actions { flex-wrap: wrap; }
      .motion-state { width: 100%; margin-left: 0; text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Stewart 边界测量</h1>
      <div class="status">
        <div>采集状态：<span id="status">starting</span></div>
        <div>已标记边界：<span id="markCount">0</span></div>
      </div>
    </header>

    <section>
      <h2>六维力传感器实时趋势</h2>
      <div class="trend-stack">
        <div class="trend-panel">
          <div class="trend-title"><span>力</span><span>最近 10 s</span></div>
          <div class="trend-layout">
            <div class="chart-wrap"><canvas id="forceChart" aria-label="力实时曲线"></canvas></div>
            <div class="readouts">
              <div class="readout"><span class="swatch" style="background:#c62828"></span><div><span class="readout-label">Fx · N</span><strong class="readout-value" id="fxCurrent">--</strong></div></div>
              <div class="readout"><span class="swatch" style="background:#1565c0"></span><div><span class="readout-label">Fy · N</span><strong class="readout-value" id="fyCurrent">--</strong></div></div>
              <div class="readout"><span class="swatch" style="background:#2e7d32"></span><div><span class="readout-label">Fz · N</span><strong class="readout-value" id="fzCurrent">--</strong></div></div>
            </div>
          </div>
        </div>
        <div class="trend-panel">
          <div class="trend-title"><span>力矩</span><span>最近 10 s</span></div>
          <div class="trend-layout">
            <div class="chart-wrap"><canvas id="torqueChart" aria-label="力矩实时曲线"></canvas></div>
            <div class="readouts">
              <div class="readout"><span class="swatch" style="background:#7b1fa2"></span><div><span class="readout-label">Mx · N·m</span><strong class="readout-value" id="mxCurrent">--</strong></div></div>
              <div class="readout"><span class="swatch" style="background:#00838f"></span><div><span class="readout-label">My · N·m</span><strong class="readout-value" id="myCurrent">--</strong></div></div>
              <div class="readout"><span class="swatch" style="background:#ef6c00"></span><div><span class="readout-label">Mz · N·m</span><strong class="readout-value" id="mzCurrent">--</strong></div></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div class="overview-grid">
      <section>
        <h2>机械臂 TCP · 基座坐标系</h2>
        <div class="pose-grid">
          <div class="pose-cell"><span>X mm</span><strong id="tcpX">--</strong></div>
          <div class="pose-cell"><span>Y mm</span><strong id="tcpY">--</strong></div>
          <div class="pose-cell"><span>Z mm</span><strong id="tcpZ">--</strong></div>
          <div class="pose-cell"><span>Roll deg</span><strong id="tcpR">--</strong></div>
          <div class="pose-cell"><span>Pitch deg</span><strong id="tcpP">--</strong></div>
          <div class="pose-cell"><span>Yaw deg</span><strong id="tcpYaw">--</strong></div>
        </div>
      </section>

      <section>
        <h2>工具坐标系固定步长点动</h2>
        <div class="control-fields">
          <label>平移步长 mm<input id="linearStep" type="number" min="0.01" max="2" step="0.01" value="0.2"></label>
          <label>旋转步长 deg<input id="angularStep" type="number" min="0.01" max="1" step="0.01" value="0.1"></label>
          <label>TCP 速度 mm/s<input id="jogSpeed" type="number" min="0.1" max="10" step="0.1" value="2"></label>
        </div>
        <div class="jog-grid">
          <div class="jog-row"><span class="axis-name">X</span><button class="jog-button" data-axis="x" data-direction="-1" title="工具坐标 -X">−</button><span class="axis-unit">平移</span><button class="jog-button" data-axis="x" data-direction="1" title="工具坐标 +X">+</button></div>
          <div class="jog-row"><span class="axis-name">Rx</span><button class="jog-button" data-axis="rx" data-direction="-1" title="工具坐标 -Rx">−</button><span class="axis-unit">旋转</span><button class="jog-button" data-axis="rx" data-direction="1" title="工具坐标 +Rx">+</button></div>
          <div class="jog-row"><span class="axis-name">Y</span><button class="jog-button" data-axis="y" data-direction="-1" title="工具坐标 -Y">−</button><span class="axis-unit">平移</span><button class="jog-button" data-axis="y" data-direction="1" title="工具坐标 +Y">+</button></div>
          <div class="jog-row"><span class="axis-name">Ry</span><button class="jog-button" data-axis="ry" data-direction="-1" title="工具坐标 -Ry">−</button><span class="axis-unit">旋转</span><button class="jog-button" data-axis="ry" data-direction="1" title="工具坐标 +Ry">+</button></div>
          <div class="jog-row"><span class="axis-name">Z</span><button class="jog-button" data-axis="z" data-direction="-1" title="工具坐标 -Z">−</button><span class="axis-unit">平移</span><button class="jog-button" data-axis="z" data-direction="1" title="工具坐标 +Z">+</button></div>
          <div class="jog-row"><span class="axis-name">Rz</span><button class="jog-button" data-axis="rz" data-direction="-1" title="工具坐标 -Rz">−</button><span class="axis-unit">旋转</span><button class="jog-button" data-axis="rz" data-direction="1" title="工具坐标 +Rz">+</button></div>
        </div>
        <div class="motion-actions">
          <button id="enableMotion">启用点动</button>
          <button id="stopMotion" class="stop-button">立即停止</button>
          <span class="motion-state">点动状态：<strong id="motionState">未启用</strong></span>
        </div>
        <div class="motion-message" id="motionMessage"></div>
      </section>
    </div>

    <section>
      <h2>相机相对 Stewart 底座标靶</h2>
      <div class="pose-grid">
        <div class="pose-cell"><span>X m</span><strong id="cameraX">--</strong></div>
        <div class="pose-cell"><span>Y m</span><strong id="cameraY">--</strong></div>
        <div class="pose-cell"><span>Z m</span><strong id="cameraZ">--</strong></div>
        <div class="pose-cell"><span>Roll deg</span><strong id="cameraR">--</strong></div>
        <div class="pose-cell"><span>Pitch deg</span><strong id="cameraP">--</strong></div>
        <div class="pose-cell"><span>Yaw deg</span><strong id="cameraYaw">--</strong></div>
      </div>
      <div class="last">标记 ID：<span id="markerId">--</span>，重投影误差：<span id="reprojError">--</span> px</div>
    </section>

    <section>
      <h2>边界记录</h2>
      <div class="mark-row">
        <input id="note" placeholder="备注，例如 +X 边界 / 最大俯仰 / 第3次接触">
        <button id="mark">标记边界点</button>
      </div>
      <div class="last" id="lastMark"></div>
    </section>
  </main>

  <script>
    const TREND_WINDOW_S = 10;
    const forceColors = ["#c62828", "#1565c0", "#2e7d32"];
    const torqueColors = ["#7b1fa2", "#00838f", "#ef6c00"];
    const poseIds = ["tcpX", "tcpY", "tcpZ", "tcpR", "tcpP", "tcpYaw"];
    const cameraIds = ["cameraX", "cameraY", "cameraZ", "cameraR", "cameraP", "cameraYaw"];
    const currentIds = ["fxCurrent", "fyCurrent", "fzCurrent", "mxCurrent", "myCurrent", "mzCurrent"];
    const history = [];
    let latestTime = Date.now() / 1000;
    let motionEnabled = false;
    let sampleReady = false;
    const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "--";

    function drawTrend(canvasId, valueKey, colors, minimumRange, unit) {
      const canvas = document.getElementById(canvasId);
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 20 || rect.height < 20) return;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.round(rect.width);
      const height = Math.round(rect.height);
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const pad = {left: 47, right: 10, top: 10, bottom: 23};
      const plotWidth = width - pad.left - pad.right;
      const plotHeight = height - pad.top - pad.bottom;
      const startTime = latestTime - TREND_WINDOW_S;
      const visible = history.filter((sample) => sample.t >= startTime);
      const maxAbs = visible.reduce(
        (maximum, sample) => Math.max(maximum, ...sample[valueKey].map((value) => Math.abs(value))),
        minimumRange,
      );
      const limit = maxAbs * 1.15;
      const xOf = (time) => pad.left + (time - startTime) / TREND_WINDOW_S * plotWidth;
      const yOf = (value) => pad.top + (limit - value) / (2 * limit) * plotHeight;

      ctx.strokeStyle = "#e8ecf2";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#667085";
      ctx.font = "11px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let index = 0; index <= 4; index += 1) {
        const y = pad.top + index / 4 * plotHeight;
        const value = limit - index / 4 * 2 * limit;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(limit >= 10 ? 0 : 2), pad.left - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let index = 0; index <= 5; index += 1) {
        const x = pad.left + index / 5 * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height - pad.bottom);
        ctx.stroke();
        ctx.fillText(`${-TREND_WINDOW_S + index * 2}s`, x, height - pad.bottom + 5);
      }
      ctx.strokeStyle = "#9aa4b2";
      ctx.beginPath();
      ctx.moveTo(pad.left, yOf(0));
      ctx.lineTo(width - pad.right, yOf(0));
      ctx.stroke();
      ctx.fillStyle = "#667085";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(unit, pad.left + 4, pad.top + 3);

      colors.forEach((color, seriesIndex) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.7;
        ctx.beginPath();
        visible.forEach((sample, index) => {
          const x = xOf(sample.t);
          const y = yOf(sample[valueKey][seriesIndex]);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    }

    function drawCharts() {
      drawTrend("forceChart", "force", forceColors, 5, "N");
      drawTrend("torqueChart", "torque", torqueColors, 0.2, "N·m");
    }

    function updateMotionControls() {
      document.getElementById("motionState").textContent = motionEnabled ? "已启用" : "未启用";
      document.querySelectorAll(".jog-button").forEach((button) => {
        button.disabled = !motionEnabled || !sampleReady;
      });
    }

    async function postJson(path, payload = {}) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      return result;
    }

    const source = new EventSource("/events");
    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      document.getElementById("status").textContent = data.status;
      document.getElementById("markCount").textContent = data.mark_count;
      document.getElementById("mark").disabled = !data.mark_allowed;
      motionEnabled = Boolean(data.motion_enabled);
      sampleReady = Boolean(data.mark_allowed);
      updateMotionControls();
      if (!data.ok) return;

      const wrenchValues = [...data.force_n, ...data.torque_nm];
      wrenchValues.forEach((value, index) => {
        document.getElementById(currentIds[index]).textContent = fmt(value);
      });
      latestTime = Number.isFinite(data.timestamp_s) ? data.timestamp_s : Date.now() / 1000;
      history.push({t: latestTime, force: [...data.force_n], torque: [...data.torque_nm]});
      while (history.length && history[0].t < latestTime - TREND_WINDOW_S) history.shift();
      drawCharts();

      data.tcp_pose.forEach((value, index) => {
        document.getElementById(poseIds[index]).textContent = fmt(value);
      });
      const cameraValues = [
        ...data.camera_position_m,
        ...data.camera_rpy_rad.map((value) => Number.isFinite(value) ? value * 180 / Math.PI : value),
      ];
      cameraValues.forEach((value, index) => {
        document.getElementById(cameraIds[index]).textContent = fmt(value);
      });
      document.getElementById("markerId").textContent = data.marker_id ?? "--";
      document.getElementById("reprojError").textContent = fmt(data.camera_reproj_error_px);
    };

    document.getElementById("enableMotion").addEventListener("click", async () => {
      const message = document.getElementById("motionMessage");
      try {
        const result = await postJson("/motion/enable");
        motionEnabled = result.motion_enabled;
        message.textContent = "工具坐标点动已启用";
      } catch (error) {
        motionEnabled = false;
        message.textContent = `启用失败：${error.message}`;
      }
      updateMotionControls();
    });

    document.getElementById("stopMotion").addEventListener("click", async () => {
      const message = document.getElementById("motionMessage");
      try {
        await postJson("/motion/stop");
        message.textContent = "已发送停止命令";
      } catch (error) {
        message.textContent = `停止失败：${error.message}`;
      }
      motionEnabled = false;
      updateMotionControls();
    });

    document.querySelectorAll(".jog-button").forEach((button) => {
      button.disabled = true;
      button.addEventListener("click", async () => {
        const message = document.getElementById("motionMessage");
        document.querySelectorAll(".jog-button").forEach((item) => { item.disabled = true; });
        try {
          await postJson("/motion/jog", {
            axis: button.dataset.axis,
            direction: Number(button.dataset.direction),
            linear_step_mm: Number(document.getElementById("linearStep").value),
            angular_step_deg: Number(document.getElementById("angularStep").value),
            speed_mm_s: Number(document.getElementById("jogSpeed").value),
          });
          message.textContent = `${button.dataset.axis.toUpperCase()} 单步完成`;
        } catch (error) {
          motionEnabled = false;
          message.textContent = `点动失败：${error.message}`;
        }
        updateMotionControls();
      });
    });

    document.getElementById("mark").addEventListener("click", async () => {
      const last = document.getElementById("lastMark");
      try {
        const result = await postJson("/mark", {note: document.getElementById("note").value});
        const record = result.record;
        last.textContent = `#${record.mark_index} 已记录：相机=(${record.camera_x_m}, ${record.camera_y_m}, ${record.camera_z_m}) F=(${record.force_x_n}, ${record.force_y_n}, ${record.force_z_n})`;
      } catch (error) {
        last.textContent = `标记失败：${error.message}`;
      }
    });

    window.addEventListener("resize", drawCharts);
    drawCharts();
  </script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual Stewart boundary marking UI using RealSense pose, force sensor, and xArm TCP pose."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--interval-s", type=float, default=0.02)
    parser.add_argument("--mock", action="store_true", help="Run UI with generated sample data instead of hardware.")

    parser.add_argument("--robot-ip", default=None, help="Required in real hardware mode.")
    parser.add_argument(
        "--force-sensor-python-dir",
        type=Path,
        default=None,
        help="Required external force-sensor driver directory in real hardware mode.",
    )
    parser.add_argument("--force-port", default="/dev/ttyUSB0")
    parser.add_argument("--force-baud", type=int, default=115200)
    parser.add_argument("--force-sensor-name", default="D80")
    parser.add_argument("--force-max-age-ms", type=float, default=100.0)
    parser.add_argument("--force-max-force", type=float, default=240.0)
    parser.add_argument("--force-max-torque", type=float, default=6.0)
    parser.add_argument("--no-clear-zero", action="store_true")
    parser.add_argument("--jog-force-limit", type=float, default=20.0)
    parser.add_argument("--jog-torque-limit", type=float, default=2.0)
    parser.add_argument("--jog-acceleration", type=float, default=20.0)
    parser.add_argument("--jog-timeout-s", type=float, default=10.0)

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--camera-dictionary", type=int, default=cv2.aruco.DICT_6X6_250)
    parser.add_argument("--marker-id", type=int, default=2)
    parser.add_argument("--marker-size", type=float, default=0.03, help="ArUco marker side length in meters.")
    return parser


def validate_hardware_args(args: argparse.Namespace) -> None:
    if args.mock:
        return
    if not args.robot_ip:
        raise ValueError("real mode requires --robot-ip")
    if args.force_sensor_python_dir is None:
        raise ValueError("real mode requires --force-sensor-python-dir")


def build_source(args: argparse.Namespace):
    if args.mock:
        return MockSource()
    validate_hardware_args(args)
    return SensorRobotSource(
        force_sensor_python_dir=args.force_sensor_python_dir,
        force_port=args.force_port,
        force_baud=args.force_baud,
        force_sensor_name=args.force_sensor_name,
        force_max_age_ms=args.force_max_age_ms,
        force_max_force=args.force_max_force,
        force_max_torque=args.force_max_torque,
        clear_zero=not args.no_clear_zero,
        robot_ip=args.robot_ip,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_serial=args.camera_serial,
        camera_dictionary=args.camera_dictionary,
        marker_size=args.marker_size,
        marker_id=args.marker_id,
        jog_acceleration=args.jog_acceleration,
        jog_timeout_s=args.jog_timeout_s,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    recorder = BoundaryRecorder(Path(args.output))
    source = build_source(args)
    poller = SamplePoller(
        source,
        interval_s=args.interval_s,
        jog_force_limit_n=args.jog_force_limit,
        jog_torque_limit_nm=args.jog_torque_limit,
    )
    poller.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(poller, recorder))
    url = f"http://{args.host}:{args.port}"
    print(f"force boundary UI running: {url}")
    print(f"manual marks CSV: {Path(args.output)}")
    print("press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.shutdown()
        server.server_close()
        poller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
