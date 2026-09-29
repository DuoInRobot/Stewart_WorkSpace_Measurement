# Stewart Workspace Boundary Neural Network Accuracy Report

## Fixed Dataset Split

| Dataset | Total Samples | Boundary Samples | Interior Samples |
|---|---:|---:|---:|
| Training Set | 14,366 | 8,521 | 5,845 |
| Validation Set | 4,788 | 2,840 | 1,948 |
| Test Set (Primary Generalization Results) | 4,788 | 2,840 | 1,948 |

## Evaluation Metrics

- **Correct boundary sample:** Determined according to the boundary classification rule of the released model.
- **Correct interior sample:** `Predicted boundary radius >= Interior sample radius`.
- **Overall accuracy:** `(Number of correct boundary samples + Number of correct interior samples) / Total number of samples`.

## Results

### Training Set

- Mean combined loss: `3.006027`
- Boundary MAE: `2.3332 mm`
- Boundary RMSE: `3.1608 mm`
- Median signed residual: `-1.2164 mm`
- 95th percentile absolute residual (P95): `5.5223 mm`
- Boundary accuracy: `92.55%` (`7,886/8,521`)
- Interior sample Inside accuracy: `98.56%` (`5,761/5,845`)
- Interior sample Outside misclassification rate: `1.44%` (`84/5,845`)
- 1 mm safety margin satisfaction rate: `97.95%`
- Interior sample margin:
  - Minimum: `-10.7069 mm`
  - P05: `3.7315 mm`
  - Median: `17.2229 mm`
  - P95: `38.9438 mm`
  - Maximum: `58.5214 mm`
- Overall accuracy: `95.00%`

### Validation Set

- Mean combined loss: `2.749038`
- Boundary MAE: `2.3739 mm`
- Boundary RMSE: `3.0277 mm`
- Median signed residual: `-1.4066 mm`
- 95th percentile absolute residual (P95): `5.6203 mm`
- Boundary accuracy: `91.48%` (`2,598/2,840`)
- Interior sample Inside accuracy: `98.87%` (`1,926/1,948`)
- Interior sample Outside misclassification rate: `1.13%` (`22/1,948`)
- 1 mm safety margin satisfaction rate: `98.20%`
- Interior sample margin:
  - Minimum: `-8.4216 mm`
  - P05: `3.3598 mm`
  - Median: `16.8657 mm`
  - P95: `38.8369 mm`
  - Maximum: `58.7668 mm`
- Overall accuracy: `94.49%`

### Test Set (Primary Generalization Results)

- Mean combined loss: `3.000196`
- Boundary MAE: `2.3587 mm`
- Boundary RMSE: `3.1621 mm`
- Median signed residual: `-1.3360 mm`
- 95th percentile absolute residual (P95): `5.6132 mm`
- Boundary accuracy: `92.85%` (`2,637/2,840`)
- Interior sample Inside accuracy: `99.13%` (`1,931/1,948`)
- Interior sample Outside misclassification rate: `0.87%` (`17/1,948`)
- 1 mm safety margin satisfaction rate: `98.31%`
- Interior sample margin:
  - Minimum: `-12.0500 mm`
  - P05: `3.5639 mm`
  - Median: `16.7373 mm`
  - P95: `39.5061 mm`
  - Maximum: `58.2492 mm`
- Overall accuracy: `95.41%`

# Stewart 边界神经网络准确度报告

## 固定数据划分

| 集合 | 总数 | 边界点 | 内部点 |
|---|---:|---:|---:|
| 训练集 | 14,366 | 8,521 | 5,845 |
| 验证集 | 4,788 | 2,840 | 1,948 |
| 测试集（主要泛化结果） | 4,788 | 2,840 | 1,948 |


## 指标定义

- 边界点正确：按发布模型的边界判定规则计算。
- 内部点正确：`预测边界 >= 内部点半径`。
- 整体准确率：`(边界点正确数 + 内部点正确数) / 全部点数`。

## 结果

### 训练集

- 混合平均损失：`3.006027`
- 边界 MAE：`2.3332 mm`；RMSE：`3.1608 mm`；残差中位数：`-1.2164 mm`；绝对残差 P95：`5.5223 mm`
- 边界准确率：`92.55%`（7,886/8,521）
- 内部点 Inside 准确率：`98.56%`（5,761/5,845）
- 内部点 Outside 误判：`1.44%`（84/5,845）
- 1 mm 安全边距满足率：`97.95%`
- 内部点 margin：最小 `-10.7069 mm`，P05 `3.7315 mm`，中位 `17.2229 mm`，P95 `38.9438 mm`，最大 `58.5214 mm`
- 整体准确率：`95.00%`

### 验证集

- 混合平均损失：`2.749038`
- 边界 MAE：`2.3739 mm`；RMSE：`3.0277 mm`；残差中位数：`-1.4066 mm`；绝对残差 P95：`5.6203 mm`
- 边界准确率：`91.48%`（2,598/2,840）
- 内部点 Inside 准确率：`98.87%`（1,926/1,948）
- 内部点 Outside 误判：`1.13%`（22/1,948）
- 1 mm 安全边距满足率：`98.20%`
- 内部点 margin：最小 `-8.4216 mm`，P05 `3.3598 mm`，中位 `16.8657 mm`，P95 `38.8369 mm`，最大 `58.7668 mm`
- 整体准确率：`94.49%`

### 测试集（主要泛化结果）

- 混合平均损失：`3.000196`
- 边界 MAE：`2.3587 mm`；RMSE：`3.1621 mm`；残差中位数：`-1.3360 mm`；绝对残差 P95：`5.6132 mm`
- 边界准确率：`92.85%`（2,637/2,840）
- 内部点 Inside 准确率：`99.13%`（1,931/1,948）
- 内部点 Outside 误判：`0.87%`（17/1,948）
- 1 mm 安全边距满足率：`98.31%`
- 内部点 margin：最小 `-12.0500 mm`，P05 `3.5639 mm`，中位 `16.7373 mm`，P95 `39.5061 mm`，最大 `58.2492 mm`
- 整体准确率：`95.41%`
