from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_source_module(name: str):
    path = SRC_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"public_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aruco_helper_imports_without_opening_hardware():
    module = load_source_module("arudo_detector")
    assert module.RealSenseColorCamera.__name__ == "RealSenseColorCamera"
    assert module.ArUcoDetector.__name__ == "ArUcoDetector"


def test_raw_data_layout_is_documented_and_ignored():
    assert (REPOSITORY_ROOT / "data" / "raw" / "README.md").is_file()
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/*.csv" in gitignore


def test_manual_collector_defaults_to_raw_pose_only_mode():
    module = load_source_module("manual_base_measurement")
    args = module.build_arg_parser().parse_args([])
    assert args.force_sensor_python_dir is None
    assert args.serial is None
    assert module.default_output_path().parent == REPOSITORY_ROOT / "data" / "raw"
    assert module.NullForceSensorReader().latest() is None


def test_manual_collector_keeps_training_compatible_labels_and_fields():
    module = load_source_module("manual_base_measurement")
    state = module.OperatorState()
    for key, label in ((ord("r"), "reference"), (ord("i"), "interior"), (ord("b"), "boundary")):
        state.handle_key(key)
        assert state.operator_mode == label
    for field in ("operator_mode", "camera_in_marker_x_m", "force_x_n", "torque_z_nm"):
        assert field in module.CSV_FIELDS


def test_live_boundary_ui_loads_published_npz_and_uses_raw_outputs():
    module = load_source_module("boundary_model_live_ui")
    assert module.DEFAULT_MODEL == REPOSITORY_ROOT / "models" / "boundary_radius_nn_model.npz"
    assert module.DEFAULT_OUTSIDE_LOG.parent == REPOSITORY_ROOT / "data" / "raw"
    assert module.DEFAULT_MANUAL_BOUNDARY_LOG.parent == REPOSITORY_ROOT / "data" / "raw"
    model = module.load_boundary_model(module.DEFAULT_MODEL)
    prediction = model.predict(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert prediction.radius_mm > 0.0
    assert model.model_type == "nn"


def test_force_ui_mock_mode_has_no_device_defaults():
    module = load_source_module("force_boundary_ui")
    args = module.build_arg_parser().parse_args(["--mock"])
    assert args.robot_ip is None
    assert args.force_sensor_python_dir is None
    source = module.build_source(args)
    assert isinstance(source, module.MockSource)
    sample = source.read()
    assert len(sample.wrench.force_n) == 3


def test_force_ui_real_mode_requires_explicit_hardware_configuration():
    module = load_source_module("force_boundary_ui")
    args = module.build_arg_parser().parse_args([])
    with pytest.raises(ValueError, match="--robot-ip"):
        module.validate_hardware_args(args)


def test_force_ui_mock_mode_does_not_attempt_hardware_imports():
    script = f"""
import sys

attempted = []

class BlockHardwareImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'pyrealsense2', 'xarm', 'force_sensor'}}:
            attempted.append(fullname)
            raise ImportError(f'blocked hardware import: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockHardwareImports())
sys.path.insert(0, {str(SRC_DIR)!r})
import force_boundary_ui

args = force_boundary_ui.build_arg_parser().parse_args(['--mock'])
source = force_boundary_ui.build_source(args)
source.start()
source.read()
source.stop()
if attempted:
    raise SystemExit('hardware imports attempted: ' + ', '.join(attempted))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_boundary_camera_startup_error_is_propagated(monkeypatch):
    module = load_source_module("boundary_model_live_ui")

    class FailingCamera:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("camera unavailable")

        def stop(self):
            pass

    fake_arudo = types.SimpleNamespace(
        RealSenseColorCamera=FailingCamera,
        ArUcoDetector=object,
    )
    monkeypatch.setitem(sys.modules, "arudo_detector", fake_arudo)
    state = module.LiveBoundaryState(model=types.SimpleNamespace(model_type="stub"))
    poller = module.CameraPoller(
        state=state,
        marker_id=2,
        marker_size=0.03,
        width=640,
        height=480,
        fps=30,
        serial=None,
        dictionary=0,
    )

    with pytest.raises(RuntimeError, match="camera unavailable"):
        poller.start()


def test_boundary_camera_capture_error_is_visible_in_status(monkeypatch):
    module = load_source_module("boundary_model_live_ui")

    class FailingCamera:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            return object()

        def read(self):
            raise RuntimeError("camera disconnected")

        def stop(self):
            self.stopped = True

    class FakeDetector:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "arudo_detector",
        types.SimpleNamespace(RealSenseColorCamera=FailingCamera, ArUcoDetector=FakeDetector),
    )
    state = module.LiveBoundaryState(model=types.SimpleNamespace(model_type="stub"))
    poller = module.CameraPoller(
        state=state,
        marker_id=2,
        marker_size=0.03,
        width=640,
        height=480,
        fps=30,
        serial=None,
        dictionary=0,
    )
    poller.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if state.status_payload().get("camera_error"):
            break
        time.sleep(0.01)
    poller.stop()

    assert state.status_payload()["camera_error"] == "camera disconnected"


def test_boundary_ui_binds_server_before_starting_camera(monkeypatch, tmp_path):
    module = load_source_module("boundary_model_live_ui")
    starts = []

    class FakePoller:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            starts.append(True)

        def stop(self):
            pass

    def fail_bind(*_args, **_kwargs):
        raise OSError("port unavailable")

    monkeypatch.setattr(module, "CameraPoller", FakePoller)
    monkeypatch.setattr(module, "ThreadingHTTPServer", fail_bind)
    args = module.build_arg_parser().parse_args(
        [
            "--outside-log", str(tmp_path / "outside.csv"),
            "--manual-boundary-log", str(tmp_path / "manual.csv"),
        ]
    )

    with pytest.raises(OSError, match="port unavailable"):
        module.run_live_ui(args)
    assert starts == []


def test_force_ui_binds_server_before_starting_source(monkeypatch, tmp_path):
    module = load_source_module("force_boundary_ui")
    starts = []

    class FakePoller:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            starts.append(True)

        def stop(self):
            pass

    real_build_arg_parser = module.build_arg_parser

    class FakeParser:
        def parse_args(self):
            return real_build_arg_parser().parse_args(
                ["--mock", "--output", str(tmp_path / "marks.csv")]
            )

    def fail_bind(*_args, **_kwargs):
        raise OSError("port unavailable")

    monkeypatch.setattr(module, "SamplePoller", FakePoller)
    monkeypatch.setattr(module, "ThreadingHTTPServer", fail_bind)
    monkeypatch.setattr(module, "build_arg_parser", lambda: FakeParser())

    with pytest.raises(OSError, match="port unavailable"):
        module.main()
    assert starts == []
