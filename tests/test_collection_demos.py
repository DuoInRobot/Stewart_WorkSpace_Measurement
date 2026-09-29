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
