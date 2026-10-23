# 基线动机图：运行与解释

脚本位于 `/home/chenyang/DU-Match/make_motivation_figure.py`。
只新增独立分析脚本，不修改训练流程或原始 UniMatch-V2。

## 运行

在 WSL Ubuntu-22.04 终端执行：

```bash
cd /home/chenyang/DU-Match
/home/chenyang/miniconda3/envs/unimatchv2/bin/python make_motivation_figure.py
```

默认读取：

- 权重：`checkpoints/cityscapes_unimatch_v2_1_16labels_dinov2_base.pth`
- 数据：`/home/chenyang/datasets/cityscapes`
- 标注训练划分：`splits/cityscapes/1_16/labeled.txt`（186 张）
- 验证划分：`splits/cityscapes/val.txt`（500 张）

该权重为普通 state_dict，无法仅凭参数判断是学生还是 EMA 教师导出。
权重文件名指向 1/16 标注比例，但具体训练划分、训练来源需要结合原始训练记录确认。
脚本严格检查架构参数，不会静默忽略加载不匹配。

## 原型怎么得到

冻结同一个基线编码器，仅用指定标注训练子集的真实标签，累计每类归一化像素特征的总和、平方和与数量。
原型是全局均值的归一化向量，离散度是围绕未归一化均值的平均平方距离。
特征先双线性对齐，再 L2 归一化。距离、方差下限与 DU-Match 核心实现一致。

这是**离线、同模型的事后原型诊断**，不是训练时学生—教师配合的在线 EMA 原型。
验证集 GT 只计算错误率、统计与选择示例，不更新原型。

## 图如何解读

- (a) 按置信度分箱统计实际错误率；0.95 处单独分箱，避免把 0.9–0.95 的像素混入高置信柱。
- (b) 只在置信度至少 0.95、且预测类别存在训练原型的像素中，按特征不相容性分箱统计错误率。下方展示像素支持量。
- (c) 自动选择高置信错误数量较多的局部案例；选图不使用特征不相容性分数，因此不预设其一定支持方法。

统计使用完整指定验证范围，图像案例只用于解释。曲线无需单调，不能为了预期趋势修改数据或挑选阈值。
错误率采用像素汇总；没有把相关像素当独立样本计算显著性或置信区间。

默认最长边 1022 像素，短边按 14 的倍数对齐；GT 最近邻缩放至推理网格。
因此结果是固定尺度诊断，**不是**原分辨率官方滑窗 mIoU，也不等同于训练时强弱增强的伪标签分布。

## 输出

默认输出目录 `analysis_motivation_baseline/`：

- `motivation_case_1.png/.pdf` 至 `motivation_case_3.png/.pdf`：三个示例，共享同一套总体统计。
- `confidence_bins.csv`、`feature_bins.csv`：每箱样本量、错误数量与错误率。
- `per_image.csv`：逐图统计及选图分数。
- `summary.json`：参数、划分哈希、样本清单、样本数、原型来源、选图位置和限制。
- `posthoc_prototypes.pth`：离线训练集原型。
- `case_*.npz`：重画案例图所需的图像与数值，不必重新推理。

排版使用 Windows 正版 Times New Roman 字体（普通、粗体、斜体），PDF 嵌入 TrueType 子集。
若字体文件不存在，脚本报错，不静默替换。配色与统一色标均不改变实际数值。

只修改版式后重新出图：

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python make_motivation_figure.py --plot-only
```

小样本检查流程（不能当论文完整统计）：

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python make_motivation_figure.py \
  --max-train 4 --max-val 4 --max-side 518 --output analysis_motivation_smoke
```

论文使用前，应确认权重与标签划分的对应关系，再判断总体结果是否支持“原型不相容性提供补充证据”。
若曲线不支持这一假设，应进一步分析类别、推理尺度及原型诊断协议，不能用漂亮图形替代证据。
