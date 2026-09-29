from __future__ import annotations

import csv
from collections import Counter
import math
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
PUBLIC_FIELDS = [
    "dataset_label",
    "pose_radius_mm",
    *[f"q{i}_mm" for i in range(1, 7)],
    *[f"d{i}" for i in range(1, 7)],
    "dataset_split",
]
EXPECTED_COUNTS = {
    "train": {"total": 14366, "boundary": 8521, "interior": 5845},
    "validation": {"total": 4788, "boundary": 2840, "interior": 1948},
    "test": {"total": 4788, "boundary": 2840, "interior": 1948},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PUBLIC_FIELDS
        return list(reader)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in PUBLIC_FIELDS)


def test_data_schema_counts_and_numeric_invariants():
    combined = read_csv(DATA_DIR / "boundary_radius_nn_dataset.csv")
    assert len(combined) == 23942

    for split_name, expected in EXPECTED_COUNTS.items():
        rows = read_csv(DATA_DIR / f"{split_name}.csv")
        assert len(rows) == expected["total"]
        assert sum(row["dataset_label"] == "boundary" for row in rows) == expected["boundary"]
        assert sum(row["dataset_label"] == "interior" for row in rows) == expected["interior"]
        assert all(row["dataset_split"] == split_name for row in rows)

    for row in combined:
        assert row["dataset_label"] in {"boundary", "interior"}
        assert row["dataset_split"] in EXPECTED_COUNTS
        radius = float(row["pose_radius_mm"])
        q = [float(row[f"q{i}_mm"]) for i in range(1, 7)]
        direction = [float(row[f"d{i}"]) for i in range(1, 7)]
        assert radius > 0.0 and math.isfinite(radius)
        assert all(math.isfinite(value) for value in q + direction)
        assert math.isclose(math.sqrt(sum(value * value for value in direction)), 1.0, abs_tol=2e-6)
        assert all(math.isclose(qv, radius * dv, abs_tol=1e-4) for qv, dv in zip(q, direction))


def test_data_combined_file_is_exact_multiset_union_of_partitions():
    combined = Counter(row_key(row) for row in read_csv(DATA_DIR / "boundary_radius_nn_dataset.csv"))
    partitioned = Counter()
    for split_name in EXPECTED_COUNTS:
        partitioned.update(row_key(row) for row in read_csv(DATA_DIR / f"{split_name}.csv"))
    assert combined == partitioned
