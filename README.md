# Stewart Boundary Radius Neural Network

本仓库使用六维等效姿态方向预测 Stewart 机构的工作空间边界半径，并提供经过清洗的固定训练集、验证集和测试集。

## 数据隐私与范围

公开数据从模型就绪的六维特征数据导出，不包含采集时间、设备或标记信息、原始相机姿态、力/力矩数据、本机文件路径以及原始样本标识。所有公开 CSV 仅包含训练、验证和解释模型所需的 15 个字段。

## 数据结构

| 字段 | 含义 |
|---|---|
| `dataset_label` | `boundary` 或 `interior` |
| `pose_radius_mm` | 六维等效姿态半径 |
| `q1_mm` … `q6_mm` | 六维等效姿态向量 |
| `d1` … `d6` | 单位方向向量 `q / ||q||` |
| `dataset_split` | `train`、`validation` 或 `test` |

六维等效姿态定义为：

```text
q = [dx, dy, dz, 60*rx, 60*ry, 60*rz]
```

其中平移单位为 mm，旋转量通过 `60 mm/rad` 转换为等效位移。

固定数据划分为：

| 集合 | 总数 | 边界点 | 内部点 |
|---|---:|---:|---:|
| 训练集 | 14,366 | 8,521 | 5,845 |
| 验证集 | 4,788 | 2,840 | 1,948 |
| 测试集 | 4,788 | 2,840 | 1,948 |

## 模型与损失函数

网络结构为：

```text
6 -> 64 -> 64 -> 32 -> 1
```

隐藏层使用 `tanh`，输出使用 `softplus + 1e-6 mm`，从而保证预测半径为正。

边界点使用平方误差：

```text
0.5 * (predicted_radius - boundary_radius)^2
```

内部点只在预测边界没有包住内部点及 1 mm 安全边距时产生惩罚：

```text
0.5 * max(0, interior_radius + 1 mm - predicted_radius)^2
```

## 安装与复现

建议使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 数据采集 Demo

三个公开采集入口位于 `src/`，生成的原始文件默认写入 `data/raw/`。安装 RealSense 和 ArUco 采集依赖：

```bash
python3 -m pip install -r requirements-collection.txt
```

`opencv-contrib-python` 提供 ArUco 模块，`pyrealsense2` 提供 Intel RealSense Python 接口。力传感器驱动不随本仓库发布；需要真实机器人模式时，另行安装 [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)：

```bash
python3 -m pip install xarm-python-sdk
```

### 1. 手动姿态采集

只采集 RealSense/ArUco 姿态：

```bash
python3 src/manual_base_measurement.py
```

连接外部力传感器驱动时：

```bash
python3 src/manual_base_measurement.py \
  --force-sensor-python-dir /path/to/force-sensor-driver
```

采集窗口按键为：`r` 标记零位参考、`i` 标记内部点、`b` 标记边界点、`u` 标记未知、`n` 开始新轨迹、`q` 或 `Esc` 结束。

### 2. 边界模型实时采集

使用公开 NPZ 模型显示当前姿态与预测边界，并在网页中设置零位、标记边界或连续采样：

```bash
python3 src/boundary_model_live_ui.py \
  --model models/boundary_radius_nn_model.npz
```

默认网页地址为 `http://127.0.0.1:8093`。

### 3. 力反馈边界采集

无硬件 mock 演示：

```bash
python3 src/force_boundary_ui.py --mock
```

默认网页地址为 `http://127.0.0.1:8765`。真实模式同时需要 RealSense、xArm 和外部力传感器驱动，并要求显式提供设备参数：

```bash
python3 src/force_boundary_ui.py \
  --robot-ip ROBOT_IP \
  --force-sensor-python-dir /path/to/force-sensor-driver
```

请先完成机器人运动安全评估，再启用真实运动控制。程序中的力和力矩限制不能替代硬件急停、限位与现场风险控制。

### 数据流与隐私

```text
采集 Demo -> data/raw -> 清洗与预处理 -> 固定公开数据集 -> 训练/验证
```

`data/raw/` 中的 CSV 可能包含实验时间、设备或标记信息、原始相机姿态、机器人姿态以及力/力矩测量。该目录中的生成文件默认被 Git 忽略；公开任何原始文件前必须进行字段审查和脱敏。当前 `data/*.csv` 是已经清洗并固定划分的公开训练、验证和测试数据，不应直接用新采集文件覆盖。

使用固定配置重新训练并生成模型和报告：

```bash
python3 src/train_boundary_radius_nn_model.py
```

显式写出完整配置时：

```bash
python3 src/train_boundary_radius_nn_model.py \
  --epochs 250 \
  --batch-size 1024 \
  --learning-rate 0.003 \
  --safety-margin-mm 1.0 \
  --seed 7
```

运行测试：

```bash
python3 -m pytest -q tests
```

无需重新训练即可核验公开模型与 JSON 报告中的全部指标是否一致：

```bash
python3 src/train_boundary_radius_nn_model.py --verify-only
```

## 指标定义

边界回归使用 MAE、RMSE、绝对残差 P95，以及误差不超过 1、2、5 mm 的比例。

双侧 5 mm 边界准确率定义为：

```text
|预测半径 - 实际边界半径| <= 5 mm
```

内部点正确表示预测边界能够包住内部点：

```text
预测边界 >= 内部点半径
```

整体准确率定义为：

```text
(双侧 5 mm 内的边界点数 + 被正确包住的内部点数) / 全部点数
```

## 发布结果

| 集合 | 边界 MAE | 边界 RMSE | 双侧 5 mm 边界准确率 | 内部点 Inside 准确率 | 整体准确率 |
|---|---:|---:|---:|---:|---:|
| 训练集 | 2.3332 mm | 3.1608 mm | 92.55% | 98.56% | 95.00% |
| 验证集 | 2.3739 mm | 3.0277 mm | 91.48% | 98.87% | 94.49% |
| 测试集 | 2.3587 mm | 3.1621 mm | 92.85% | 99.13% | 95.41% |

机器可读的完整指标位于 `reports/accuracy_report.json`，便于审计的 Markdown 报告位于 `reports/accuracy_report.md`。

## 重要限制

当前采用按标签分层的行级随机划分，而不是按采集轨迹或批次分组。同一次连续采集中的相邻观测以及完全重复的特征/目标可能分布在不同集合，因此验证集和测试集结果可能偏乐观。评估对全新实验批次的泛化能力时，应增加按轨迹或批次分组的测试。

## 许可证

本项目采用 [MIT License](LICENSE)。
