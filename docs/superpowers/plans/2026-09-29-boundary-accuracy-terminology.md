# Boundary Accuracy Terminology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Public Markdown uses only the generic label “边界准确率” and no longer displays bilateral/5 mm wording or the ±1/±2/±5 mm boundary-ratio line, while all calculations and published numeric values remain unchanged.

**Architecture:** Keep the internal metric schema and 5 mm correctness calculation intact for compatibility. Change only the Markdown renderer and its two published consumers, then enforce the terminology and renderer/report equality with tests.

**Tech Stack:** Python 3.10, pytest, Markdown, NumPy model artifact verification

## Global Constraints

- The displayed boundary accuracy values remain `92.55%`, `91.48%`, and `92.85%` for train, validation, and test.
- The displayed combined accuracy values remain `95.00%`, `94.49%`, and `95.41%`.
- `reports/accuracy_report.json`, all CSV files, and `models/boundary_radius_nn_model.npz` must not change.
- Keep “1 mm 安全边距满足率” because it is an interior-point safety metric, not a boundary-accuracy threshold.
- Preserve the untracked `CAD/` directory and all unrelated user files.

---

### Task 1: Enforce and implement generic boundary-accuracy wording

**Files:**
- Modify: `tests/test_published_artifacts.py`
- Modify: `tests/test_release_privacy.py`
- Modify: `src/train_boundary_radius_nn_model.py:289-333`
- Modify: `reports/accuracy_report.md`
- Modify: `README.md:163-195`

**Interfaces:**
- Consumes: `render_report(summary: dict) -> str` and the existing JSON keys `within_5mm_count`, `within_5mm_ratio`, and `combined_accuracy`.
- Produces: renderer output and published Markdown that label the existing `within_5mm_ratio` value only as “边界准确率”.

- [ ] **Step 1: Write the failing terminology and renderer-consistency tests**

Add to `tests/test_published_artifacts.py`:

```python
def test_public_markdown_uses_generic_boundary_accuracy_wording():
    summary = json.loads(
        (REPOSITORY_ROOT / "reports" / "accuracy_report.json").read_text(encoding="utf-8")
    )
    rendered = trainer.render_report(summary)
    published = (REPOSITORY_ROOT / "reports" / "accuracy_report.md").read_text(encoding="utf-8")

    assert rendered == published
    assert "边界准确率" in rendered
    assert "边界绝对误差：" not in rendered
    assert ("双侧 " + "5 mm 边界准确率") not in rendered
    assert ("双侧 " + "5 mm 内的边界点数") not in rendered
```

In `tests/test_release_privacy.py`, replace the required README phrase `"双侧 5 mm 边界准确率"` with `"边界准确率"`, then add these assertions inside the loop that reads README and both reports:

```python
        assert ("双侧 " + "5 mm 边界准确率") not in text
        assert ("误差不超过 " + "1、2、5 mm 的比例") not in text
        assert "边界绝对误差：" not in text
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_published_artifacts.py::test_public_markdown_uses_generic_boundary_accuracy_wording \
  tests/test_release_privacy.py::test_readme_documents_reproduction_metrics_and_split_limitation
```

Expected: both tests fail because the renderer, published report, and README still contain the old wording.

- [ ] **Step 3: Change the Markdown renderer without changing calculations**

In `render_report()` within `src/train_boundary_radius_nn_model.py`, make the metric definition lines exactly:

```python
            "- 边界点正确：按发布模型的边界判定规则计算。",
            "- 内部点正确：`预测边界 >= 内部点半径`。",
            "- 整体准确率：`(边界点正确数 + 内部点正确数) / 全部点数`。",
```

Remove the generated `边界绝对误差` line and replace the generated boundary-accuracy line with:

```python
                f"- 边界准确率：`{boundary['within_5mm_ratio']:.2%}`（{boundary['within_5mm_count']:,}/{boundary['sample_count']:,}）",
```

Do not change `compute_metrics()`, JSON field names, or the combined-accuracy formula.

- [ ] **Step 4: Synchronize the published Markdown report**

In `reports/accuracy_report.md`:

- Replace the two metric-definition lines with the generic definitions from Step 3.
- Delete each `边界绝对误差：±1 mm ... ±2 mm ... ±5 mm ...` result line.
- Rename each `双侧 5 mm 边界准确率` result line to `边界准确率`, preserving every percentage and count.

- [ ] **Step 5: Simplify the README metric definition and table**

Replace the boundary metric introduction in `README.md` with:

```markdown
边界回归使用 MAE、RMSE 和绝对残差 P95。边界点是否正确按照发布模型的边界判定规则计算，对外统一报告为“边界准确率”。
```

Remove the displayed formula containing `<= 5 mm`. Change the combined formula to:

```text
(边界点正确数 + 被正确包住的内部点数) / 全部点数
```

Change the result-table header from `双侧 5 mm 边界准确率` to `边界准确率`; preserve all table values.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/test_published_artifacts.py \
  tests/test_release_privacy.py
```

Expected: all selected tests pass, including byte-for-byte equality between `render_report(summary)` and `reports/accuracy_report.md`.

- [ ] **Step 7: Commit the terminology change**

```bash
git add README.md reports/accuracy_report.md src/train_boundary_radius_nn_model.py \
  tests/test_published_artifacts.py tests/test_release_privacy.py
git commit -m "docs: simplify boundary accuracy terminology"
```

### Task 2: Verify published artifacts and unchanged numeric results

**Files:**
- Verify only: `data/*.csv`
- Verify only: `models/boundary_radius_nn_model.npz`
- Verify only: `reports/accuracy_report.json`

**Interfaces:**
- Consumes: the committed renderer, model, fixed split data, and JSON metrics.
- Produces: verification evidence only; no generated files.

- [ ] **Step 1: Run the complete test suite**

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q -p no:cacheprovider tests
```

Expected: all tests pass.

- [ ] **Step 2: Verify model and JSON metrics**

```bash
/usr/bin/python3 -B src/train_boundary_radius_nn_model.py --verify-only
```

Expected:

```text
published artifacts verified: test_combined_accuracy=0.9541
```

- [ ] **Step 3: Verify protected artifacts are unchanged**

Compare the implementation base commit with `HEAD` for:

```text
data/boundary_radius_nn_dataset.csv
data/train.csv
data/validation.csv
data/test.csv
models/boundary_radius_nn_model.npz
reports/accuracy_report.json
```

Expected: no diff.

- [ ] **Step 4: Scan public Markdown for removed wording**

```bash
rg -n '双侧 5 mm 边界准确率|边界绝对误差：|误差不超过 1、2、5 mm 的比例' \
  README.md reports/accuracy_report.md
```

Expected: no matches.

- [ ] **Step 5: Confirm final repository state**

```bash
git status --short --branch
```

Expected: only the pre-existing untracked `CAD/` remains; no cache or temporary files are introduced.
