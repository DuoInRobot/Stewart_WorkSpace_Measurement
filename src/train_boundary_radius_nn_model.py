#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPOSITORY_ROOT / "data" / "boundary_radius_nn_dataset.csv"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "boundary_radius_nn_model.npz"
DEFAULT_SUMMARY = REPOSITORY_ROOT / "reports" / "accuracy_report.json"
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "accuracy_report.md"
DIRECTION_FIELDS = [f"d{index}" for index in range(1, 7)]
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class BoundaryRadiusNNConfig:
    hidden_sizes: tuple[int, ...] = (64, 64, 32)
    seed: int = 7
    min_radius_mm: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 250
    batch_size: int = 1024
    learning_rate: float = 0.003
    safety_margin_mm: float = 1.0


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


class BoundaryRadiusNN:
    def __init__(
        self,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        config: BoundaryRadiusNNConfig,
    ):
        self.weights = [np.asarray(weight, dtype=float) for weight in weights]
        self.biases = [np.asarray(bias, dtype=float) for bias in biases]
        self.config = config

    @classmethod
    def initialize(cls, config: BoundaryRadiusNNConfig) -> "BoundaryRadiusNN":
        rng = np.random.default_rng(int(config.seed))
        sizes = [6, *config.hidden_sizes, 1]
        weights = []
        biases = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = math.sqrt(2.0 / float(fan_in + fan_out))
            weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out, dtype=float))
        return cls(weights, biases, config)

    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [np.asarray(x, dtype=float)]
        preactivations: list[np.ndarray] = []
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ weight + bias
            preactivations.append(z)
            if layer_index == len(self.weights) - 1:
                activations.append(_softplus(z) + float(self.config.min_radius_mm))
            else:
                activations.append(np.tanh(z))
        return activations, preactivations

    def predict(self, directions: np.ndarray) -> np.ndarray:
        values = np.asarray(directions, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        activations, _ = self._forward(values)
        return activations[-1].reshape(-1)

    def save(self, path: Path) -> None:
        payload = {f"W{index}": value for index, value in enumerate(self.weights)}
        payload.update({f"b{index}": value for index, value in enumerate(self.biases)})
        payload["hidden_sizes"] = np.asarray(self.config.hidden_sizes, dtype=int)
        payload["seed"] = np.asarray([self.config.seed], dtype=int)
        payload["min_radius_mm"] = np.asarray([self.config.min_radius_mm], dtype=float)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: Path) -> "BoundaryRadiusNN":
        payload = np.load(Path(path))
        config = BoundaryRadiusNNConfig(
            hidden_sizes=tuple(int(value) for value in payload["hidden_sizes"].tolist()),
            seed=int(payload["seed"][0]),
            min_radius_mm=float(payload["min_radius_mm"][0]),
        )
        weights = []
        biases = []
        index = 0
        while f"W{index}" in payload:
            weights.append(payload[f"W{index}"])
            biases.append(payload[f"b{index}"])
            index += 1
        return cls(weights, biases, config)


def _finite_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def load_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[list[float], float, int]]] = {name: [] for name in SPLIT_NAMES}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            split_name = row.get("dataset_split")
            label = row.get("dataset_label")
            if split_name not in grouped:
                raise ValueError(f"row {row_number}: invalid dataset_split")
            if label not in {"boundary", "interior"}:
                raise ValueError(f"row {row_number}: invalid dataset_label")
            radius = _finite_float(row, "pose_radius_mm")
            direction = [_finite_float(row, field) for field in DIRECTION_FIELDS]
            if radius <= 0.0 or float(np.linalg.norm(direction)) <= 1e-9:
                raise ValueError(f"row {row_number}: invalid radius or direction")
            grouped[split_name].append((direction, radius, 1 if label == "boundary" else 0))

    partitions = {}
    for split_name in SPLIT_NAMES:
        samples = grouped[split_name]
        if not samples:
            raise ValueError(f"dataset split {split_name!r} is empty")
        directions, radii, labels = zip(*samples)
        label_array = np.asarray(labels, dtype=int)
        if not np.any(label_array == 1) or not np.any(label_array == 0):
            raise ValueError(f"dataset split {split_name!r} must contain both labels")
        partitions[split_name] = (
            np.asarray(directions, dtype=float),
            np.asarray(radii, dtype=float),
            label_array,
        )
    return partitions


def _train_model(
    directions: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
    nn_config: BoundaryRadiusNNConfig,
    train_config: TrainingConfig,
) -> tuple[BoundaryRadiusNN, list[dict[str, float]]]:
    model = BoundaryRadiusNN.initialize(nn_config)
    rng = np.random.default_rng(int(nn_config.seed) + 1000)
    history: list[dict[str, float]] = []
    sample_count = directions.shape[0]
    batch_size = max(1, int(train_config.batch_size))
    for epoch in range(int(train_config.epochs)):
        order = rng.permutation(sample_count)
        epoch_loss = 0.0
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            x = directions[indices]
            y = radii[indices]
            is_boundary = labels[indices] == 1
            activations, preactivations = model._forward(x)
            predictions = activations[-1].reshape(-1)
            gradient_prediction = np.zeros_like(predictions)
            loss_terms = np.zeros_like(predictions)
            if np.any(is_boundary):
                residual = predictions[is_boundary] - y[is_boundary]
                loss_terms[is_boundary] = 0.5 * residual**2
                gradient_prediction[is_boundary] = residual
            if np.any(~is_boundary):
                violation = y[~is_boundary] + train_config.safety_margin_mm - predictions[~is_boundary]
                active = violation > 0.0
                interior_loss = np.zeros_like(violation)
                interior_gradient = np.zeros_like(violation)
                interior_loss[active] = 0.5 * violation[active] ** 2
                interior_gradient[active] = -violation[active]
                loss_terms[~is_boundary] = interior_loss
                gradient_prediction[~is_boundary] = interior_gradient
            epoch_loss += float(np.sum(loss_terms))
            gradient = (
                gradient_prediction.reshape(-1, 1) / float(indices.size)
            ) * _sigmoid(preactivations[-1])
            weight_gradients = []
            bias_gradients = []
            for layer_index in reversed(range(len(model.weights))):
                weight_gradients.append(activations[layer_index].T @ gradient)
                bias_gradients.append(np.sum(gradient, axis=0))
                if layer_index > 0:
                    gradient = (gradient @ model.weights[layer_index].T) * (
                        1.0 - activations[layer_index] ** 2
                    )
            weight_gradients.reverse()
            bias_gradients.reverse()
            for layer_index in range(len(model.weights)):
                model.weights[layer_index] -= train_config.learning_rate * weight_gradients[layer_index]
                model.biases[layer_index] -= train_config.learning_rate * bias_gradients[layer_index]
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == train_config.epochs - 1:
            history.append({"epoch": float(epoch + 1), "loss": epoch_loss / float(sample_count)})
    return model, history


def _percentile(values: np.ndarray, percent: float) -> float:
    return float(np.percentile(values, percent)) if values.size else math.nan


def evaluate_partition(
    model: BoundaryRadiusNN,
    directions: np.ndarray,
    radii: np.ndarray,
    labels: np.ndarray,
    safety_margin_mm: float,
) -> dict:
    predictions = np.asarray(model.predict(directions), dtype=float).reshape(-1)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if predictions.size != radii.size or radii.size != labels.size or radii.size == 0:
        raise ValueError("predictions, radii, and labels must have the same non-zero length")

    boundary_mask = labels == 1
    interior_mask = labels == 0
    boundary_residuals = predictions[boundary_mask] - radii[boundary_mask]
    boundary_absolute = np.abs(boundary_residuals)
    interior_margins = predictions[interior_mask] - radii[interior_mask]
    violations = np.maximum(0.0, safety_margin_mm - interior_margins)
    loss_terms = np.zeros(radii.size, dtype=float)
    loss_terms[boundary_mask] = 0.5 * boundary_residuals**2
    loss_terms[interior_mask] = 0.5 * violations**2

    boundary_count = int(np.sum(boundary_mask))
    interior_count = int(np.sum(interior_mask))
    inside_count = int(np.sum(interior_margins >= 0.0))
    safety_count = int(np.sum(interior_margins >= safety_margin_mm))
    within_5mm_count = int(np.sum(boundary_absolute <= 5.0))

    def ratio(count: int, total: int) -> float:
        return float(count / total) if total else math.nan

    return {
        "counts": {"total": int(radii.size), "boundary": boundary_count, "interior": interior_count},
        "mean_loss": float(np.mean(loss_terms)),
        "boundary": {
            "sample_count": boundary_count,
            "mae_mm": float(np.mean(boundary_absolute)),
            "rmse_mm": float(np.sqrt(np.mean(boundary_residuals**2))),
            "residual_median_mm": _percentile(boundary_residuals, 50),
            "residual_p95_abs_mm": _percentile(boundary_absolute, 95),
            "within_1mm_ratio": ratio(int(np.sum(boundary_absolute <= 1.0)), boundary_count),
            "within_2mm_ratio": ratio(int(np.sum(boundary_absolute <= 2.0)), boundary_count),
            "within_5mm_count": within_5mm_count,
            "within_5mm_ratio": ratio(within_5mm_count, boundary_count),
            "underestimated_more_than_5mm_count": int(np.sum(boundary_residuals < -5.0)),
            "overestimated_count": int(np.sum(boundary_residuals > 0.0)),
        },
        "interior": {
            "sample_count": interior_count,
            "inside_count": inside_count,
            "inside_ratio": ratio(inside_count, interior_count),
            "outside_count": interior_count - inside_count,
            "outside_ratio": ratio(interior_count - inside_count, interior_count),
            "safety_margin_satisfied_count": safety_count,
            "safety_margin_satisfied_ratio": ratio(safety_count, interior_count),
            "margin_min_mm": _percentile(interior_margins, 0),
            "margin_p05_mm": _percentile(interior_margins, 5),
            "margin_median_mm": _percentile(interior_margins, 50),
            "margin_p95_mm": _percentile(interior_margins, 95),
            "margin_max_mm": _percentile(interior_margins, 100),
        },
        "combined_accuracy": ratio(within_5mm_count + inside_count, int(radii.size)),
    }


def render_report(summary: dict) -> str:
    titles = {"train": "训练集", "validation": "验证集", "test": "测试集（主要泛化结果）"}
    lines = [
        "# Stewart 边界神经网络准确度报告",
        "",
        "## 固定数据划分",
        "",
        "| 集合 | 总数 | 边界点 | 内部点 |",
        "|---|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        counts = summary["split_counts"][name]
        lines.append(f"| {titles[name]} | {counts['total']:,} | {counts['boundary']:,} | {counts['interior']:,} |")
    lines.extend(
        [
            "",
            "> 局限性：采用行级随机划分，相邻或重复的特征/目标观测可能跨集合，因此指标可能偏乐观。",
            "",
            "## 指标定义",
            "",
            "- 边界点正确：按发布模型的边界判定规则计算。",
            "- 内部点正确：`预测边界 >= 内部点半径`。",
            "- 整体准确率：`(边界点正确数 + 内部点正确数) / 全部点数`。",
            "",
            "## 结果",
            "",
        ]
    )
    for name in SPLIT_NAMES:
        metrics = summary["metrics"][name]
        boundary = metrics["boundary"]
        interior = metrics["interior"]
        lines.extend(
            [
                f"### {titles[name]}",
                "",
                f"- 混合平均损失：`{metrics['mean_loss']:.6f}`",
                f"- 边界 MAE：`{boundary['mae_mm']:.4f} mm`；RMSE：`{boundary['rmse_mm']:.4f} mm`；残差中位数：`{boundary['residual_median_mm']:.4f} mm`；绝对残差 P95：`{boundary['residual_p95_abs_mm']:.4f} mm`",
                f"- 边界准确率：`{boundary['within_5mm_ratio']:.2%}`（{boundary['within_5mm_count']:,}/{boundary['sample_count']:,}）",
                f"- 内部点 Inside 准确率：`{interior['inside_ratio']:.2%}`（{interior['inside_count']:,}/{interior['sample_count']:,}）",
                f"- 内部点 Outside 误判：`{interior['outside_ratio']:.2%}`（{interior['outside_count']:,}/{interior['sample_count']:,}）",
                f"- 1 mm 安全边距满足率：`{interior['safety_margin_satisfied_ratio']:.2%}`",
                f"- 内部点 margin：最小 `{interior['margin_min_mm']:.4f} mm`，P05 `{interior['margin_p05_mm']:.4f} mm`，中位 `{interior['margin_median_mm']:.4f} mm`，P95 `{interior['margin_p95_mm']:.4f} mm`，最大 `{interior['margin_max_mm']:.4f} mm`",
                f"- 整体准确率：`{metrics['combined_accuracy']:.2%}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _relative_output(path: Path, repository_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"output path must be inside repository root: {path}") from error


def _relative_input(path: Path, repository_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _compare_published_metrics(expected: object, actual: object, path: str = "metrics") -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f"published metric mismatch at {path}: keys differ")
        for key in expected:
            _compare_published_metrics(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"published metric mismatch at {path}: expected {expected!r}, recomputed {actual!r}"
            )
        return
    if expected != actual:
        raise ValueError(f"published metric mismatch at {path}: expected {expected!r}, recomputed {actual!r}")


def verify_published_artifacts(
    *,
    dataset_csv: Path = DEFAULT_DATASET,
    model_npz: Path = DEFAULT_MODEL,
    summary_json: Path = DEFAULT_SUMMARY,
) -> dict:
    with Path(summary_json).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    safety_margin_mm = float(summary["train_config"]["safety_margin_mm"])
    partitions = load_dataset(dataset_csv)
    model = BoundaryRadiusNN.load(model_npz)
    recomputed = {
        name: evaluate_partition(
            model,
            *partitions[name],
            safety_margin_mm=safety_margin_mm,
        )
        for name in SPLIT_NAMES
    }
    _compare_published_metrics(summary["metrics"], recomputed)
    return recomputed


def train_evaluate(
    *,
    dataset_csv: Path = DEFAULT_DATASET,
    model_npz: Path = DEFAULT_MODEL,
    summary_json: Path = DEFAULT_SUMMARY,
    report_md: Path = DEFAULT_REPORT,
    nn_config: BoundaryRadiusNNConfig = BoundaryRadiusNNConfig(),
    train_config: TrainingConfig = TrainingConfig(),
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    output_files = {
        "model_npz": _relative_output(model_npz, repository_root),
        "summary_json": _relative_output(summary_json, repository_root),
        "report_md": _relative_output(report_md, repository_root),
    }
    partitions = load_dataset(dataset_csv)
    train_directions, train_radii, train_labels = partitions["train"]
    model, history = _train_model(
        train_directions,
        train_radii,
        train_labels,
        nn_config,
        train_config,
    )
    model.save(model_npz)
    metrics = {
        name: evaluate_partition(
            model,
            *partitions[name],
            safety_margin_mm=train_config.safety_margin_mm,
        )
        for name in SPLIT_NAMES
    }
    summary = {
        "input_files": {"dataset_csv": _relative_input(dataset_csv, repository_root)},
        "output_files": output_files,
        "nn_config": asdict(nn_config),
        "train_config": asdict(train_config),
        "training": {
            "sample_count": int(train_directions.shape[0]),
            "boundary_count": int(np.sum(train_labels == 1)),
            "interior_count": int(np.sum(train_labels == 0)),
            "loss_history": history,
        },
        "split_counts": {name: metrics[name]["counts"] for name in SPLIT_NAMES},
        "metrics": metrics,
        "limitation": "Row-level random splitting can place correlated or duplicate observations in different partitions.",
    }
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(summary_json).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text(render_report(summary), encoding="utf-8")
    return summary


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive comma-separated integers")
    return sizes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Stewart boundary-radius neural network.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-npz", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--safety-margin-mm", type=float, default=TrainingConfig.safety_margin_mm)
    parser.add_argument("--hidden-sizes", default="64,64,32")
    parser.add_argument("--seed", type=int, default=BoundaryRadiusNNConfig.seed)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute metrics from the published model and fail if the JSON report differs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.verify_only:
        metrics = verify_published_artifacts(
            dataset_csv=args.dataset_csv,
            model_npz=args.model_npz,
            summary_json=args.summary_json,
        )
        print(
            "published artifacts verified: "
            f"test_combined_accuracy={metrics['test']['combined_accuracy']:.4f}"
        )
        return 0
    summary = train_evaluate(
        dataset_csv=args.dataset_csv,
        model_npz=args.model_npz,
        summary_json=args.summary_json,
        report_md=args.report_md,
        nn_config=BoundaryRadiusNNConfig(hidden_sizes=_parse_hidden_sizes(args.hidden_sizes), seed=args.seed),
        train_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            safety_margin_mm=args.safety_margin_mm,
        ),
        repository_root=REPOSITORY_ROOT,
    )
    test_metrics = summary["metrics"]["test"]
    print(
        "training complete: "
        f"train={summary['split_counts']['train']['total']}, "
        f"validation={summary['split_counts']['validation']['total']}, "
        f"test={summary['split_counts']['test']['total']}, "
        f"test_boundary_within_5mm_accuracy={test_metrics['boundary']['within_5mm_ratio']:.4f}, "
        f"test_combined_accuracy={test_metrics['combined_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
