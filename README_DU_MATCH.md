# DU-Match 技术说明

DU-Match 基于 UniMatch V2 扩展，新增训练入口 `du_match.py`、双空间可靠性建模、原型统计与软目标纠正。本文记录实现细节和实验限制。文中的本地路径是开发环境示例，使用时请替换为自己的路径。

## 运行

在 WSL Ubuntu-22.04 终端运行：

```bash
cd /home/chenyang/DU-Match
bash scripts/train_du_match.sh pascal 366 1
```

使用现成环境 `/home/chenyang/miniconda3/envs/unimatchv2/bin/python`。
默认保存到 `exp/pascal/du_match/dinov2_base/366`，不会读取复制过来的基线训练目录。
目录中已有 `latest.pth` 时，恢复学生、教师、优化器、原型、方差、观测次数和预热进度。
改变 DU-Match 参数做消融时请使用一个新的保存目录，例如：

```bash
bash scripts/train_du_match.sh pascal 366 1 29502 exp/pascal/du_match/ablation_teacher/366
```

Cityscapes 配置也已提供：`configs/cityscapes_du_match.yaml`，运行方式是：

```bash
bash scripts/train_du_match.sh cityscapes 1_8 1
```

Cityscapes 完整尺寸 GPU 训练尚未验证，798 裁剪和 base 编码器的显存需求较大。
机器目前只有 `dinov2_base.pth`，因此新增配置采用 base；原始 Cityscapes 配置采用 small。
新增 Cityscapes 配置使用标准 CE（对应所贴论文），原始配置是 OHEM。
单 GPU 学习率沿用本地 Pascal 的 `3.125e-7`，不是经过调参后的最佳学习率。
这些条件不能直接与官方论文数字比较，必须让基线和消融使用相同骨干、优化设置和标签划分。

**batch_size=1 的限制：** 为延续本机已有 Pascal 配置，默认每卡 batch 为 1。
原始源码使用本卡 batch 的 `flip(0)` 作为 CutMix donor；此时 donor 就是自身，CutMix 不产生跨样本混合。
若研究完整的 CutMix 效果，应在显存允许时对基线和 DU-Match 同时使用每卡 batch >= 2。
两条强增强和互补通道扰动仍然运行。

## 实现

- `util/du_match.py`：标注像素 EMA 原型、类内离散度、双空间状态、软纠正、统一分母损失和 CutMix 对齐。
- `du_match.py`：沿用原始训练流程，加入原型更新、预热、四类监督、诊断日志和 checkpoint。
- `diagnose_du_match.py`：只读验证集诊断，不用验证标签更新原型或参与训练。
- `configs/*_du_match.yaml`：方法参数；当前阈值与预热长度是实验起点，未经性能验证。
- `tests/`：公式、梯度、忽略像素、空间对齐、稀有类、原型恢复、分布式统计和基线回归。

表示采用最后一层 encoder patch token，不插入可训练映射。
先双线性对齐到像素分辨率，再 L2 归一化；按行分块计算，避免一次构建整幅 D 维密集特征图。
原型更新用真实标签，忽略 255，跨进程汇总像素和、平方和、计数后再计算中心和方差。
缺席类不衰减，第一次观测直接初始化，原型及所有状态存入 checkpoint。

## 与论文需要明确对齐的实现细节

1. 默认 `prototype_source: student` 对应所贴论文，学生标注特征更新原型，教师弱特征查询。
   可改为 `teacher` 做同空间消融，会增加一次标注教师前向；这属于方法变体，论文应同步描述。
2. `variance_floor: 0.01` 给距离分母中的类内方差设下限，防止接近零时爆炸；原始 EMA 方差本身未被截断。
   实际公式为 `delta = distance / (2 * max(variance, variance_floor) + eps)`。
3. 原型尚未观测的预测类退回原始置信度筛选，不把“没有原型”当作“高置信错误”。
   缺失类不进入原型 softmax。原型置信度用 `log(已就绪类数)` 归一化；全部类就绪时就是论文的 `log(C)`。
   少于两个就绪类时关闭原型纠正，避免单类 softmax 的虚假置信度。
4. `gamma_min: 0`，避免原型分布不确定时仍然强制纠正。`gamma_max: 0.8` 是可调上限。
5. 预热前 1000 步保持原始 `(Lsup + Lu) / 2`；之后按论文用 `Lsup + lambda_u * Lu`。
   两阶段全局梯度尺度不同，公平对照需要明确控制这个因素。
6. 预热阶段沿用“所有非忽略像素”分母，完整方法阶段采用 R/A/S 总数分母，空集合返回可反传的零损失。
7. 原型在当前迭代学生前向后、教师弱视图状态估计前更新一次；教师在 optimizer.step 后更新。
   所贴论文最后一句又说教师和原型一起在末尾更新，建议改成上述明确顺序，避免重复更新。
8. 教师 EMA 沿用源码 `mu_t = min(1 - 1/(t+1), 0.996)`，不是从第一步起固定的 mu。
9. CutMix 同时混合硬/软标签、R/A/S/U、缺失类回退及忽略掩码，保持原始反向 batch donor。
10. `reliability` 只写诊断，不作为训练权重。训练日志的状态比例来自 rank 0 本卡弱视图；原型统计跨卡同步。

## 快速验证

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python -m unittest discover -s tests -v
PYTHONPATH=. /home/chenyang/miniconda3/envs/unimatchv2/bin/python tests/check_distributed.py
PYTHONPATH=. /home/chenyang/miniconda3/envs/unimatchv2/bin/python tests/check_baseline.py
/home/chenyang/miniconda3/envs/unimatchv2/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  du_match.py --config configs/du_match_smoke.yaml \
  --labeled-id-path splits/pascal/366/labeled.txt \
  --unlabeled-id-path splits/pascal/366/unlabeled.txt \
  --save-path exp/du_match_smoke_new --smoke-steps 2
```

Smoke 配置使用 98 像素裁剪、每卡 batch 2、预热 1 步，只验证运行。
`configs/du_match_smoke_full.yaml` 是 518 像素、batch 1 的两步测试配置。
Smoke 保存的是 `smoke.pth`，标明 `smoke_only=True`，不会被当作完整 epoch 的 `latest.pth` 恢复。
两步随机初始化训练没有研究结论；不能把这类输出当作论文实验。
完整训练的 mIoU、多随机种子、显存峰值、性能增益和真实多 GPU 训练尚未验证。

## 检验“高置信错误挖掘”是否成立

正式训练后：

```bash
/home/chenyang/miniconda3/envs/unimatchv2/bin/python diagnose_du_match.py \
  --config configs/pascal_du_match.yaml \
  --checkpoint exp/pascal/du_match/dinov2_base/366/best.pth \
  --max-images 100 --max-side 518 --output exp/pascal/du_match/diagnostics.json
```

输出 R/A/S/U 的真实错误率、S 的错误精度、高置信错误召回率、纠正后目标错误率、修复/误改数量与净收益。
分母为空输出 null，不把“没有检测到样本”记成完美准确率。
该诊断采用固定最大边推理再对齐回原标签尺寸，不是官方 mIoU 评测；比较方法必须统一诊断尺度。
纠正效果按软目标 argmax 诊断，不等同于 KL 训练后的最终分割收益。

方法审查与论文比较见 `docs/DU_MATCH_REVIEW.md`；复制核对见 `docs/copy_verification.json`。
