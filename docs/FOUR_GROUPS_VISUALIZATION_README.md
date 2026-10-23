# 四组像素的真实可视化

代码位于 `/home/chenyang/DU-Match/visualize_four_groups.py`，桌面论文目录保存了同名副本。

## 生成

在 WSL Ubuntu-22.04 的 DU-Match 仓库中运行：

```bash
cd /home/chenyang/DU-Match
/home/chenyang/miniconda3/envs/unimatchv2/bin/python visualize_four_groups.py \
  --checkpoint exp/cityscapes_integration_20261016/latest.pth \
  --output analysis_four_groups
```

默认评估完整 Cityscapes 验证划分（500 张），最长边 1022，输入尺寸对齐至 14 的倍数，GT 最近邻对齐。
这是固定尺度诊断，不等于官方原尺寸 798 滑窗评测。

脚本加载检查点中的 **EMA 教师和训练期间保存的在线类别原型**，不会重新构建原型，也不会用验证标签更新原型。
当前图来自 1100 步的诊断检查点，必须作为初步结果；正式训练后换成正式检查点重新生成。

## 输出

- `fig_four_groups.pdf/.png`：类似参考图的综合版，含散点、统计表、柱状图和四组案例。
- `fig_four_groups_statistics.pdf/.png`：统计版，适合正文放大展示。
- `fig_four_groups_examples.pdf/.png`：四行完整街景案例版，保留整幅场景及完整热图。
- `statistics.json`：来源、配置、全部图像 ID、分组统计、案例位置及随机种子。
- `group_statistics.csv`：各组像素数、错误数、比例、GT 错误率和平均置信度。
- `per_image.csv`：逐图分组统计。
- `scatter_samples.npz`：来自真实像素的均匀随机优先级抽样。
- `case_*.npz`：每个组的原图、预测、GT、掩码与热图数值，可脱离 GPU 重新排版。
- `full_case_*.npz`：上述案例对应的完整源图、预测、GT、掩码和热图缓存。

## 分组和统计口径

完全使用 `util/du_match.py` 的状态定义：

`G = (confidence >= 0.95) AND (prediction_entropy < 0.2)`；`F = (feature_incompatibility < 0.7)`。

R 为 G=1/F=1；A 为 G=0/F=1；S 为 G=1/F=0；U 为 G=0/F=0。
G 同时取决于置信度和熵，因此散点图不画熵竖线，不把状态颜色简单解释为四象限。
横轴仍为原方法定义的归一化熵 U_pred；保留并标注特征阈值横线 tau_f=0.7，
图内说明颜色按完整 CEM 规则确定。该修改仅涉及表达，真实散点、统计、阈值与分组均保留。

像素比例的分母是有效且具有预测类别原型的像素数；不可用原型像素数单独记录。
组内错误率和平均置信度的分母是该组像素数；空组显示缺失值，不伪造为零错误率。
R 和 S 的平均置信度不能低于配置的置信度阈值，代码对此有数值检查。

统计涵盖全部指定验证图像，散点仅为可读性抽样。案例窗口按目标组像素数选择，不按 GT 正误筛选。
综合图保留局部案例，其热图只显示相应组，其他像素为灰色。
单独案例版展示完整街景，热图覆盖整个有效场景；原图以淡色覆盖和边界线标出目标组，
行标题不表示整张图都属于该组。完整热图的灰色仅表示忽略像素或不可用的特征原型。
所有热图固定为 0～1，避免按各图单独拉伸。
“Reliable”和“Suspicious”是方法状态，不代表 GT 已确认正确或错误。

各图不再在图内显示训练步数和验证集数量的横幅；这些来源信息仍完整保留在统计文件和论文图注中。

## 只重新排版

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python visualize_four_groups.py \
  --plot-only --output analysis_four_groups
```

如果使用旧的局部案例缓存，第一次改成整图版时运行：

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python visualize_four_groups.py \
  --plot-only --refresh-full-examples --output analysis_four_groups
```

该命令只对已选的四张源图重新推理，补充整图缓存，不重算或更改 500 张图像的总体统计。

Times New Roman 字体默认读取 `/mnt/c/Windows/Fonts` 下的 times*.ttf。
其他机器可以使用 `--font-dir /字体目录`，或在服务器生成统计后回本机排图。
缺少字体时直接报错，不静默替换字体。

综合版适合屏幕查看；论文正文优先使用拆开的统计版和案例版，避免 24 个缩略图被压得过小。
