# 批次分析报告 batch_20260621_000

> 批次ID: `1` | 任务ID: `1`

## 1. 实验参数


| 参数 | 数值 |
| :--- | ---: |
| 像素尺寸 (μm/px) | 0.1070 |
| 帧率 (Hz) | 30.0000 |
| 温度 (°C) | 25.0000 |
| 黏度 (Pa·s) | 0.0009 |
| 标称粒径 (nm) | 100.0000 |
| 通道宽度 (μm) | 200.0000 |
| 通道高度 (μm) | 50.0000 |

## 2. 质控摘要

| 质控项 | 数量 |
| :--- | ---: |
| 总轨迹数 | 111 |
| 有效轨迹数 | 99 |
| 短轨迹数 (不参与D分布) | 11 |
| 修复轨迹数 | 0 |
| 断裂轨迹数 | 14 |
| 缺失帧数 | 41 |

- **漂移估计**: vx = 0.0893 μm/s, vy = -0.1127 μm/s

| 异常类型 | 数量 |
| :--- | ---: |
| 跳点/断裂 | 55 |
| 强度骤降 | 8 |

## 3. 模型判别统计

| 模型类型 | 轨迹数 |
| :--- | ---: |
| 布朗扩散 (brownian) | 70 |
| 受限扩散 (confined) | 0 |
| 定向扩散 (directed) | 37 |
| 次扩散 (subdiffusive) | 2 |
| 超扩散 (superdiffusive) | 0 |
| 异常扩散 (anomalous) | 2 |
| 未知 (unknown) | 0 |

## 4. 批次统计结果

### 4.1 扩散系数 D 分布

| 统计量 | 数值 |
| :--- | ---: |
| 均值 ± 标准差 (μm²/s) | 52.3123 ± 280.7388 |
| 中位数 (μm²/s) | 4.9160 |
| P25 (μm²/s) | 3.5409 |
| P75 (μm²/s) | 6.1618 |

### 4.2 水力学半径

| 统计量 | 数值 |
| :--- | ---: |
| 均值 ± 标准差 (nm) | 59.2319 ± 45.6219 |
| 中位数 (nm) | 49.9127 |

### 4.3 通道间差异

| 通道 | mean_D (μm²/s) | median_D (μm²/s) | 有效数 | 总数 |
| :--- | ---: | ---: | ---: | ---: |
| 通道 channel_A | 70.6383 | 4.8920 | 33 | 35 |
| 通道 channel_B | 71.8239 | 5.0660 | 34 | 39 |
| 通道 channel_C | 12.6825 | 4.9282 | 32 | 37 |

## 5. 校准曲线数据

| 通道 | 标称 (nm) | 实测 (nm) | 偏差 (%) | D (μm²/s) |
| :--- | ---: | ---: | ---: | ---: |
| 通道 channel_A | 100.0000 | 52.0544 | -47.9456 | 3.3327 |
| 通道 channel_B | 100.0000 | 65.8740 | -34.1260 | 5.3922 |
| 通道 channel_C | 100.0000 | 59.5764 | -40.4236 | 2.6839 |

- 平均偏差: -40.8317%
- 整体拟合 R²: -0.0000

## 6. 最差轨迹 Top-10

| Particle ID | 通道 | 帧数 | D (μm²/s) | R² | 模型 | 排除原因/异常 |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| 23 | 通道 channel_A | 10 | 6.6554 | 0.4589 | directed | short_trajectory_10_frames_less_than_20; {'short_trajectory': True} |
| 39 | 通道 channel_B | 19 | 10.3993 | 0.9783 | directed | short_trajectory_19_frames_less_than_20; {'short_trajectory': True} |
| 84 | 通道 channel_C | 14 | 25.9538 | 0.9978 | directed | short_trajectory_14_frames_less_than_20; {'short_trajectory': True} |
| 75 | 通道 channel_C | 5 | N/A | N/A | anomalous | short_trajectory_5_frames_less_than_20; {'short_trajectory': True} |
| 27 | 通道 channel_A | 7 | 0.4270 | 0.4981 | subdiffusive | short_trajectory_7_frames_less_than_20; {'short_trajectory': True} |
| 82 | 通道 channel_C | 17 | 18.4199 | 0.9990 | directed | short_trajectory_17_frames_less_than_20; {'short_trajectory': True} |
| 40 | 通道 channel_B | 19 | 12.1242 | 0.9836 | directed | short_trajectory_19_frames_less_than_20; {'short_trajectory': True} |
| 81 | 通道 channel_C | 10 | 13.0372 | 0.9940 | directed | short_trajectory_10_frames_less_than_20; {'short_trajectory': True} |
| 98 | 通道 channel_C | 17 | 2.4467 | 0.9201 | brownian | short_trajectory_17_frames_less_than_20; {'short_trajectory': True} |
| 36 | 通道 channel_B | 18 | 4.9462 | 0.9975 | brownian | short_trajectory_18_frames_less_than_20; {'short_trajectory': True} |

## 7. 复测建议

1. 通道间 D 值差异过大 (CV=65.4% > 30%)，建议检查通道一致性

## 8. 建议与备注

- 短轨迹共 11 条未参与粒径分布统计，仅在质控报告中保留记录。
- 漂移扣除前后结果均已保留在轨迹结果字段中。
