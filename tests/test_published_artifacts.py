from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPOSITORY_ROOT / "src" / "train_boundary_radius_nn_model.py"
SPEC = importlib.util.spec_from_file_location("stewart_boundary_nn_artifact_verifier", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def test_published_model_reproduces_published_json_metrics():
    metrics = trainer.verify_published_artifacts(
        dataset_csv=REPOSITORY_ROOT / "data" / "boundary_radius_nn_dataset.csv",
        model_npz=REPOSITORY_ROOT / "models" / "boundary_radius_nn_model.npz",
        summary_json=REPOSITORY_ROOT / "reports" / "accuracy_report.json",
    )

    assert metrics["test"]["boundary"]["within_5mm_count"] == 2637
    assert metrics["test"]["boundary"]["within_5mm_ratio"] == pytest.approx(2637 / 2840)
    assert metrics["test"]["combined_accuracy"] == pytest.approx((2637 + 1931) / 4788)
    removed_prefix = "conser" + "vative"
    assert f"{removed_prefix}_within_5mm_count" not in metrics["test"]["boundary"]
    assert f"{removed_prefix}_within_5mm_ratio" not in metrics["test"]["boundary"]


def test_artifact_verification_rejects_changed_metrics(tmp_path):
    summary = json.loads(
        (REPOSITORY_ROOT / "reports" / "accuracy_report.json").read_text(encoding="utf-8")
    )
    summary["metrics"]["test"]["combined_accuracy"] = 0.0
    changed_summary = tmp_path / "changed.json"
    changed_summary.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="published metric mismatch"):
        trainer.verify_published_artifacts(
            dataset_csv=REPOSITORY_ROOT / "data" / "boundary_radius_nn_dataset.csv",
            model_npz=REPOSITORY_ROOT / "models" / "boundary_radius_nn_model.npz",
            summary_json=changed_summary,
        )


def test_output_paths_are_rejected_before_any_output_is_written(tmp_path):
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    dataset_csv = repository_root / "dataset.csv"
    fields = [
        "dataset_label",
        "pose_radius_mm",
        *[f"q{i}_mm" for i in range(1, 7)],
        *[f"d{i}" for i in range(1, 7)],
        "dataset_split",
    ]
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            for label, radius in (("boundary", 10.0), ("interior", 8.0)):
                writer.writerow(
                    {
                        "dataset_label": label,
                        "pose_radius_mm": radius,
                        **{f"q{i}_mm": radius if i == 1 else 0.0 for i in range(1, 7)},
                        **{f"d{i}": 1.0 if i == 1 else 0.0 for i in range(1, 7)},
                        "dataset_split": split_name,
                    }
                )

    outside_model = tmp_path / "outside-model.npz"
    with pytest.raises(ValueError, match="output path must be inside repository root"):
        trainer.train_evaluate(
            dataset_csv=dataset_csv,
            model_npz=outside_model,
            summary_json=repository_root / "summary.json",
            report_md=repository_root / "report.md",
            nn_config=trainer.BoundaryRadiusNNConfig(hidden_sizes=(2,)),
            train_config=trainer.TrainingConfig(epochs=0),
            repository_root=repository_root,
        )

    assert not outside_model.exists()
    assert not (repository_root / "summary.json").exists()
    assert not (repository_root / "report.md").exists()
