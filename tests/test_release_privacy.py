from __future__ import annotations

import csv
import json
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FIELDS = [
    "dataset_label",
    "pose_radius_mm",
    *[f"q{i}_mm" for i in range(1, 7)],
    *[f"d{i}" for i in range(1, 7)],
    "dataset_split",
]
BANNED_CSV_FIELDS = {
    "timestamp_s",
    "elapsed_s",
    "camera_host_monotonic_ns",
    "force_host_monotonic_ns",
    "marker_id",
    "force_sequence",
    "source_file",
    "source_row_index",
    "source_sample_index",
    "camera_in_marker_x_m",
    "marker_in_camera_x_m",
    "force_x_n",
    "torque_x_nm",
    "sample_weight",
    "source_kind",
}


def test_release_metadata_and_documentation_exist():
    for relative_path in (
        "README.md",
        "LICENSE",
        "requirements.txt",
        ".gitignore",
        "models/boundary_radius_nn_model.npz",
        "reports/accuracy_report.json",
        "reports/accuracy_report.md",
        "src/arudo_detector.py",
        "src/manual_base_measurement.py",
        "src/boundary_model_live_ui.py",
        "src/force_boundary_ui.py",
        "requirements-collection.txt",
        "data/raw/README.md",
    ):
        assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path


def test_all_csv_files_use_only_public_fields():
    for path in sorted((REPOSITORY_ROOT / "data").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            fields = csv.DictReader(handle).fieldnames
        assert fields == PUBLIC_FIELDS
        assert not (set(fields or ()) & BANNED_CSV_FIELDS)


def test_report_paths_are_relative_and_release_has_no_local_absolute_paths():
    summary = json.loads((REPOSITORY_ROOT / "reports" / "accuracy_report.json").read_text(encoding="utf-8"))
    for group in (summary["input_files"], summary["output_files"]):
        for value in group.values():
            assert not Path(value).is_absolute(), value

    local_home_marker = "/" + "home" + "/"
    drive_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
    original_name_marker = "manual_base_samples_" + "2026"
    fixed_robot_ip = "192.168." + "1.241"
    workspace_import = "WorkSpace" + "Measurement."
    suffixes = {".csv", ".json", ".md", ".py", ".txt"}
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or {"__pycache__", ".pytest_cache"} & set(path.parts):
            continue
        if path.suffix not in suffixes and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert local_home_marker not in text, path
        assert not drive_path.search(text), path
        assert original_name_marker not in text, path
        assert fixed_robot_ip not in text, path
        assert workspace_import not in text, path


def test_readme_documents_reproduction_metrics_and_split_limitation():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "python3 src/train_boundary_radius_nn_model.py",
        "14,366",
        "4,788",
        "边界准确率",
        "整体准确率",
        "行级随机划分",
        "MIT",
        "数据采集 Demo",
        "python3 src/manual_base_measurement.py",
        "python3 src/boundary_model_live_ui.py",
        "python3 src/force_boundary_ui.py --mock",
        "data/raw",
    ):
        assert phrase in readme
    removed_zh = "保守" + "边界"
    removed_en = "conser" + "vative"
    assert f"{removed_zh}准确率" not in readme

    for relative_path in (
        "README.md",
        "reports/accuracy_report.json",
        "reports/accuracy_report.md",
    ):
        text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert removed_en not in text
        assert removed_zh not in text
        assert ("双侧 " + "5 mm 边界准确率") not in text
        assert ("误差不超过 " + "1、2、5 mm 的比例") not in text
        assert "边界绝对误差：" not in text
