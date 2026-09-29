from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
