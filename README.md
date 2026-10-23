# 重庆工贸车辆学院 · UniMatch V2 复现与跨域分析

本仓库整理 UniMatch V2 半监督语义分割复现代码，以及 Cityscapes → ACDC 的跨域评估与可视化分析脚本，供学习和研究使用。

## 来源与贡献范围

基于 [LiheYoung/UniMatch-V2](https://github.com/LiheYoung/UniMatch-V2)，本仓库不是上游官方仓库。UniMatch V2 方法与基础实现归原作者所有；保留原始 MIT 许可证及文件版权声明。[上游完整说明](README_UPSTREAM.md)另行保留。

本仓库补充整理 ACDC/Cityscapes 的置信度、不确定性分析及可视化脚本。本次发布不包含 DU-Match 实现。“重庆工贸车辆学院”为仓库展示名称，不表示院校官方发布渠道。

## 包含内容

- `unimatch_v2.py`、`fixmatch.py`、`supervised.py`：训练入口。
- `model/`、`dataset/`、`configs/`、`splits/`：模型、数据读取、配置和数据划分文本。
- `analyze_acdc_uncertainty.py`：Cityscapes 模型在 ACDC 验证集上的分析。
- `analyze_cityscapes_uncertainty.py`、`analyze_from_csv.py`：源域分析与统计。
- `visualize_cityscapes.py`、`visualize_pascal.py`、`visualization_demo.html`：可视化工具。

不包含模型权重、检查点、数据集图像及标签、日志、缓存或生成的实验图片。划分文本仅记录文件列表，不包含数据本体。

## 运行准备

1. 按上游说明安装匹配 CUDA 环境的 PyTorch、torchvision，并安装 `requirements.txt` 中的依赖；绘图分析还需要 matplotlib。
2. 自行准备 Cityscapes、ACDC 或 Pascal VOC 数据，并遵守各数据集条款。
3. 按[上游说明](README_UPSTREAM.md)和[官方权重页面](https://huggingface.co/LiheYoung/UniMatch-V2/tree/main)下载所需权重。
4. 修改配置的数据路径，以及训练脚本的 Python 环境和 GPU 设置。分析脚本开头的 `DATA_ROOT`、`CHECKPOINT`、输出路径也须改为本机路径。

```bash
pip install -r requirements.txt
python analyze_acdc_uncertainty.py
```

分析脚本为研究用脚本，运行前需核对路径与模型配置。ACDC 跨域分析使用 Cityscapes 训练模型直接推理，不使用 ACDC 更新模型；共用 19 类 TrainIds，忽略标签为 255。全分辨率推理显存占用较高，改变推理尺度时请明确记录评估协议。仓库不附带本地实验输出。

## 引用与许可证

使用基础方法时请引用：Lihe Yang, Zhen Zhao, Hengshuang Zhao. *UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation*. IEEE TPAMI, 2025。

基础代码使用 [MIT License](LICENSE)，原版权声明保留；第三方文件中的许可证、数据集与下载权重的条款分别适用。
