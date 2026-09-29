from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPOSITORY_ROOT / "src" / "train_boundary_radius_nn_model.py"
SPEC = importlib.util.spec_from_file_location("stewart_boundary_nn_release_trainer", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)

BoundaryRadiusNNConfig = trainer.BoundaryRadiusNNConfig
TrainingConfig = trainer.TrainingConfig
evaluate_partition = trainer.evaluate_partition
load_dataset = trainer.load_dataset
train_evaluate = trainer.train_evaluate


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, directions):
        assert len(directions) == len(self.predictions)
        return self.predictions.copy()


def test_metric_math_uses_bilateral_5mm_boundary_and_combined_accuracy():
    metrics = evaluate_partition(
        FixedPredictionModel([14.0, 6.0, 16.0, 9.0, 12.0]),
        np.eye(5, 6),
        np.array([10.0, 10.0, 10.0, 8.0, 13.0]),
        np.array([1, 1, 1, 0, 0]),
        safety_margin_mm=1.0,
    )

    assert metrics["counts"] == {"total": 5, "boundary": 3, "interior": 2}
    assert metrics["mean_loss"] == pytest.approx(7.2)
    assert metrics["boundary"]["mae_mm"] == pytest.approx(14 / 3)
    assert metrics["boundary"]["rmse_mm"] == pytest.approx((68 / 3) ** 0.5)
    assert metrics["boundary"]["within_1mm_ratio"] == pytest.approx(0.0)
    assert metrics["boundary"]["within_2mm_ratio"] == pytest.approx(0.0)
    assert metrics["boundary"]["within_5mm_count"] == 2
    assert metrics["boundary"]["within_5mm_ratio"] == pytest.approx(2 / 3)
    removed_prefix = "conser" + "vative"
    assert not any(removed_prefix in key for key in metrics["boundary"])
    assert metrics["interior"]["inside_count"] == 1
    assert metrics["interior"]["inside_ratio"] == pytest.approx(0.5)
    assert metrics["interior"]["safety_margin_satisfied_count"] == 1
    assert metrics["combined_accuracy"] == pytest.approx(3 / 5)


def test_load_dataset_uses_published_fixed_splits():
    partitions = load_dataset(REPOSITORY_ROOT / "data" / "boundary_radius_nn_dataset.csv")

    assert set(partitions) == {"train", "validation", "test"}
    assert partitions["train"][0].shape == (14366, 6)
    assert partitions["validation"][0].shape == (4788, 6)
    assert partitions["test"][0].shape == (4788, 6)
    assert int(np.sum(partitions["train"][2] == 1)) == 8521
    assert int(np.sum(partitions["train"][2] == 0)) == 5845


def test_train_evaluate_trains_only_train_partition_and_writes_outputs(tmp_path):
    summary = train_evaluate(
        dataset_csv=REPOSITORY_ROOT / "data" / "boundary_radius_nn_dataset.csv",
        model_npz=tmp_path / "model.npz",
        summary_json=tmp_path / "summary.json",
        report_md=tmp_path / "report.md",
        nn_config=BoundaryRadiusNNConfig(hidden_sizes=(4,), seed=7),
        train_config=TrainingConfig(
            epochs=1,
            batch_size=1024,
            learning_rate=0.001,
            safety_margin_mm=1.0,
        ),
        repository_root=tmp_path,
    )

    assert summary["training"]["sample_count"] == 14366
    assert summary["split_counts"]["validation"]["total"] == 4788
    assert summary["split_counts"]["test"]["total"] == 4788
    assert set(summary["metrics"]) == {"train", "validation", "test"}
    assert (tmp_path / "model.npz").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["output_files"] == {
        "model_npz": "model.npz",
        "summary_json": "summary.json",
        "report_md": "report.md",
    }
