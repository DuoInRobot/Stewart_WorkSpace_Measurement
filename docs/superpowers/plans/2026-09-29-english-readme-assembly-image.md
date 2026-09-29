# English README and Assembly Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a fully English README with the assembly image embedded below the introduction, and add the existing image and all 45 STL assets under `CAD/` to Git.

**Architecture:** Treat the README as a faithful English localization: preserve its structure, commands, paths, numbers, formulas, warnings, and links while translating all prose and table labels. Add regression checks for English-only prose, the exact image reference, and CAD file presence; do not translate or regenerate any report artifact.

**Tech Stack:** GitHub-flavored Markdown, HTML image tag, Python 3.10, pytest, Git

## Global Constraints

- Insert only the assembly image; do not add a CAD-assets section or descriptions of individual STL files.
- Keep the image path exactly `CAD/装配体.jpg` and preserve all existing CAD filenames and binary bytes.
- Add all 45 existing `CAD/*.STL` files and the JPEG to Git using ordinary Git storage.
- Preserve every command, file path, URL, model parameter, dataset count, metric value, and license reference from the current README.
- Use “Boundary accuracy” without bilateral or millimetre-threshold qualifiers.
- Do not modify `reports/accuracy_report.md`, `reports/accuracy_report.json`, datasets, or the model.

---

### Task 1: Add English README and CAD regression requirements

**Files:**
- Modify: `tests/test_release_privacy.py`

**Interfaces:**
- Consumes: repository-root `README.md` and the existing `CAD/` directory.
- Produces: release tests that require English README content, the exact assembly-image reference, 45 STL files, and no Chinese prose outside the image filename.

- [ ] **Step 1: Replace the README expectations and add CAD assertions**

In `test_readme_documents_reproduction_metrics_and_split_limitation()`, require these phrases instead of the current Chinese README phrases:

```python
    for phrase in (
        "python3 src/train_boundary_radius_nn_model.py",
        "14,366",
        "4,788",
        "Boundary accuracy",
        "Overall accuracy",
        "row-level stratified random split",
        "MIT",
        "Data Collection Demos",
        "python3 src/manual_base_measurement.py",
        "python3 src/boundary_model_live_ui.py",
        "python3 src/force_boundary_ui.py --mock",
        "data/raw",
        'src="CAD/装配体.jpg"',
        'alt="Stewart platform assembly"',
    ):
        assert phrase in readme
```

After those assertions, ensure the image filename is the only Chinese text in the README and verify the CAD inventory:

```python
    readme_without_image_path = readme.replace("CAD/装配体.jpg", "CAD/assembly.jpg")
    assert re.search(r"[\u4e00-\u9fff]", readme_without_image_path) is None

    cad_dir = REPOSITORY_ROOT / "CAD"
    assert (cad_dir / "装配体.jpg").is_file()
    assert len(list(cad_dir.glob("*.STL"))) == 45
```

Keep the existing checks for removed conservative terminology and generic boundary-accuracy wording in the public reports.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_release_privacy.py::test_readme_documents_reproduction_metrics_and_split_limitation
```

Expected: FAIL because the README is still Chinese and does not yet reference the assembly image.

### Task 2: Translate README and embed the assembly image

**Files:**
- Replace: `README.md`
- Add: `CAD/装配体.jpg`
- Add: all existing `CAD/*.STL`
- Test: `tests/test_release_privacy.py`

**Interfaces:**
- Consumes: the current README's commands, metrics, warnings, and structure.
- Produces: a fully English `README.md` and tracked CAD binaries.

- [ ] **Step 1: Replace README with the approved English content**

Use this complete content for `README.md`:

````markdown
# Stewart Boundary Radius Neural Network

This repository predicts the workspace boundary radius of a Stewart platform from a six-dimensional equivalent-pose direction. It includes sanitized, fixed training, validation, and test datasets.

<p align="center">
  <img src="CAD/装配体.jpg" alt="Stewart platform assembly" width="720">
</p>

## Data Privacy and Scope

The public data was exported from model-ready six-dimensional feature data. It does not contain acquisition times, device or marker information, raw camera poses, force/torque data, local file paths, or original sample identifiers. Every public CSV contains only the 15 fields required to train, validate, and interpret the model.

## Data Schema

| Field | Meaning |
|---|---|
| `dataset_label` | `boundary` or `interior` |
| `pose_radius_mm` | Six-dimensional equivalent-pose radius |
| `q1_mm` … `q6_mm` | Six-dimensional equivalent-pose vector |
| `d1` … `d6` | Unit direction vector `q / ||q||` |
| `dataset_split` | `train`, `validation`, or `test` |

The six-dimensional equivalent pose is defined as:

```text
q = [dx, dy, dz, 60*rx, 60*ry, 60*rz]
```

Translations are measured in millimetres. Rotations are converted to equivalent displacements using `60 mm/rad`.

The fixed data split is:

| Split | Total | Boundary | Interior |
|---|---:|---:|---:|
| Training | 14,366 | 8,521 | 5,845 |
| Validation | 4,788 | 2,840 | 1,948 |
| Test | 4,788 | 2,840 | 1,948 |

## Model and Loss Functions

The network architecture is:

```text
6 -> 64 -> 64 -> 32 -> 1
```

The hidden layers use `tanh`. The output uses `softplus + 1e-6 mm` to ensure that the predicted radius is positive.

Boundary samples use squared error:

```text
0.5 * (predicted_radius - boundary_radius)^2
```

Interior samples are penalized only when the predicted boundary does not enclose the interior point plus the 1 mm safety margin:

```text
0.5 * max(0, interior_radius + 1 mm - predicted_radius)^2
```

## Installation and Reproduction

Python 3.10 or later is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data Collection Demos

The three public collection entry points are in `src/`. Generated raw files are written to `data/raw/` by default. Install the RealSense and ArUco collection dependencies with:

```bash
python3 -m pip install -r requirements-collection.txt
```

`opencv-contrib-python` provides the ArUco module, and `pyrealsense2` provides the Intel RealSense Python interface. The force-sensor driver is not distributed with this repository. For real robot mode, install the [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK) separately:

```bash
python3 -m pip install xarm-python-sdk
```

### 1. Manual Pose Collection

To collect only RealSense/ArUco poses:

```bash
python3 src/manual_base_measurement.py
```

To connect an external force-sensor driver:

```bash
python3 src/manual_base_measurement.py \
  --force-sensor-python-dir /path/to/force-sensor-driver
```

Collection-window keys: `r` marks the zero-pose reference, `i` marks an interior point, `b` marks a boundary point, `u` marks an unknown point, `n` starts a new trajectory, and `q` or `Esc` exits.

### 2. Live Boundary-Model Collection

Use the published NPZ model to display the current pose and predicted boundary. The web UI can set the zero pose, mark a boundary, or collect samples continuously:

```bash
python3 src/boundary_model_live_ui.py \
  --model models/boundary_radius_nn_model.npz
```

The default web address is `http://127.0.0.1:8093`.

### 3. Force-Feedback Boundary Collection

Run the hardware-free mock demo with:

```bash
python3 src/force_boundary_ui.py --mock
```

The default web address is `http://127.0.0.1:8765`. Real mode requires RealSense, xArm, and an external force-sensor driver, and all device parameters must be supplied explicitly:

```bash
python3 src/force_boundary_ui.py \
  --robot-ip ROBOT_IP \
  --force-sensor-python-dir /path/to/force-sensor-driver
```

Complete a robot-motion safety assessment before enabling real motion control. The software force and torque limits do not replace a hardware emergency stop, physical limits, or on-site risk controls.

### Data Flow and Privacy

```text
Collection demos -> data/raw -> sanitization and preprocessing -> fixed public dataset -> training/validation
```

CSV files under `data/raw/` may contain experiment times, device or marker information, raw camera poses, robot poses, and force/torque measurements. Generated files in this directory are ignored by Git by default. Review and sanitize every raw file before publication. The current `data/*.csv` files are sanitized, fixed public training, validation, and test datasets; do not overwrite them directly with newly collected files.

Retrain the model and generate reports using the fixed configuration:

```bash
python3 src/train_boundary_radius_nn_model.py
```

To specify the full configuration explicitly:

```bash
python3 src/train_boundary_radius_nn_model.py \
  --epochs 250 \
  --batch-size 1024 \
  --learning-rate 0.003 \
  --safety-margin-mm 1.0 \
  --seed 7
```

Run the tests:

```bash
python3 -m pytest -q tests
```

Verify all metrics in the published model and JSON report without retraining:

```bash
python3 src/train_boundary_radius_nn_model.py --verify-only
```

## Metric Definitions

Boundary regression is reported with MAE, RMSE, and the 95th percentile of the absolute residual. Boundary samples are classified according to the published model's boundary rule and reported as “Boundary accuracy.”

An interior point is correct when the predicted boundary encloses it:

```text
predicted_boundary >= interior_point_radius
```

Overall accuracy is defined as:

```text
(correct boundary points + correctly enclosed interior points) / all points
```

## Published Results

| Split | Boundary MAE | Boundary RMSE | Boundary accuracy | Interior-point Inside accuracy | Overall accuracy |
|---|---:|---:|---:|---:|---:|
| Training | 2.3332 mm | 3.1608 mm | 92.55% | 98.56% | 95.00% |
| Validation | 2.3739 mm | 3.0277 mm | 91.48% | 98.87% | 94.49% |
| Test | 2.3587 mm | 3.1621 mm | 92.85% | 99.13% | 95.41% |

Complete machine-readable metrics are available in `reports/accuracy_report.json`. The auditable Markdown report is available in `reports/accuracy_report.md`.

## Important Limitation

The current dataset uses a row-level stratified random split rather than grouping by acquisition trajectory or batch. Adjacent observations from the same continuous acquisition, as well as exact duplicate feature/target observations, may appear in different splits. Validation and test metrics may therefore be optimistic. Add trajectory- or batch-grouped evaluation when measuring generalization to entirely new experimental batches.

## License

This project is licensed under the [MIT License](LICENSE).
````

- [ ] **Step 2: Stage only the approved CAD assets and README**

```bash
git add README.md tests/test_release_privacy.py CAD/装配体.jpg CAD/*.STL
```

Verify exactly 45 STL files and one JPEG are staged under `CAD/`:

```bash
git diff --cached --name-only -- CAD
```

- [ ] **Step 3: Run the focused release test and verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_release_privacy.py
```

Expected: all release-privacy tests pass.

- [ ] **Step 4: Commit the English README and CAD assets**

```bash
git commit -m "docs: publish English README and CAD assets"
```

### Task 3: Verify translation, assets, and unchanged release artifacts

**Files:**
- Verify: `README.md`
- Verify: `CAD/装配体.jpg`
- Verify: `CAD/*.STL`
- Verify unchanged: `reports/accuracy_report.md`, `reports/accuracy_report.json`, `data/*.csv`, `models/boundary_radius_nn_model.npz`

**Interfaces:**
- Consumes: the committed README, CAD assets, tests, and published model artifacts.
- Produces: verification evidence only.

- [ ] **Step 1: Run the complete test suite**

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider tests
```

Expected: all tests pass.

- [ ] **Step 2: Verify the published model and JSON metrics**

```bash
/usr/bin/python3 -B src/train_boundary_radius_nn_model.py --verify-only
```

Expected:

```text
published artifacts verified: test_combined_accuracy=0.9541
```

- [ ] **Step 3: Verify CAD assets are tracked**

```bash
git ls-files 'CAD/*'
```

Expected: 46 tracked paths: one JPEG and 45 STL files.

- [ ] **Step 4: Verify protected artifacts are unchanged**

Compare the implementation base commit with `HEAD` for:

```text
data/
models/boundary_radius_nn_model.npz
reports/accuracy_report.json
reports/accuracy_report.md
```

Expected: no diff.

- [ ] **Step 5: Confirm final repository state**

```bash
git status --short --branch
```

Expected: clean feature branch with no cache or temporary files.
