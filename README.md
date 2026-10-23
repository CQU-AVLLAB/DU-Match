# DU-Match

基于双空间可靠性建模的半监督语义分割研究代码。

DU-Match 结合预测置信度与特征原型信息，对未标注像素进行可靠性划分，并通过软目标纠正和一致性学习利用未标注数据。项目基于 UniMatch V2 扩展，保留基线实现用于对照实验。

## 方法概览

- **双空间建模**：结合预测空间置信度与 DINOv2 特征空间的类原型信息。
- **原型统计**：使用标注像素更新 EMA 类原型和类内离散度。
- **四类像素状态**：可靠、模糊、可疑和不确定（R/A/S/U）。
- **软目标纠正**：根据原型信息纠正伪标签，并保持 CutMix 目标与掩码对齐。
- **教师–学生训练**：包含 EMA 教师、预热、断点恢复和诊断工具。

详细实现与参数说明见 [DU-Match 技术说明](README_DU_MATCH.md)。

## 安装与准备

```bash
git clone https://github.com/CQU-AVLLAB/DU-Match.git
cd DU-Match
pip install -r requirements.txt
```

根据所用配置准备 Pascal VOC 或 Cityscapes 数据，修改 `configs/*_du_match.yaml` 中的数据路径；标签划分位于 `splits/`。

自行准备 DINOv2 预训练编码器，并放到配置指定的位置（例如 `pretrained/dinov2_base.pth`）。本仓库不提供模型权重、数据集、训练输出或缓存。

## 训练 DU-Match

```bash
# Pascal VOC，366 张标注图像，单 GPU
DUMATCH_PYTHON=python bash scripts/train_du_match.sh pascal 366 1

# Cityscapes，1/8 标注划分，单 GPU
DUMATCH_PYTHON=python bash scripts/train_du_match.sh cityscapes 1_8 1
```

训练入口为 `du_match.py`，默认输出到 `exp/<dataset>/du_match/dinov2_base/<split>`。如需消融实验，可通过第 5 个参数指定新的输出目录：

```bash
DUMATCH_PYTHON=python bash scripts/train_du_match.sh pascal 366 1 29502 exp/pascal/du_match/ablation/366
```

## 代码导航

| 文件或目录 | 用途 |
| --- | --- |
| `du_match.py` | DU-Match 训练入口 |
| `util/du_match.py` | 原型、双空间状态与一致性损失 |
| `configs/*_du_match*.yaml` | 方法与对照实验配置 |
| `scripts/train_du_match.sh` | 分布式训练启动脚本 |
| `diagnose_du_match.py` | 伪标签可靠性与纠正效果诊断 |
| `analyze_*`、`visualize_*`、`make_*` | 分析、可视化与绘图工具 |
| `tests/` | 方法单元测试与回归检查 |
| `unimatch_v2.py`、`fixmatch.py`、`supervised.py` | 基线训练入口 |

## 验证与实验说明

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

本次发布的 10 项单元测试通过。单元测试不代表完整训练性能；本页暂不列出未经核实的 DU-Match 精度或性能提升。训练条件、诊断方式和已知限制见 [技术说明](README_DU_MATCH.md)、[方法审查](docs/DU_MATCH_REVIEW.md) 和 [Cityscapes 训练检查](docs/CITYSCAPES_TRAINING_CHECK.md)。

## 来源与许可

本项目基于 [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2) 开发，使用 DINOv2 编码器。感谢上游作者的工作。上游项目介绍与引用信息保留在 [README_UPSTREAM.md](README_UPSTREAM.md)，许可见 [LICENSE](LICENSE)。
