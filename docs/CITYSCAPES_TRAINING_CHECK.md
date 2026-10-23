# Cityscapes training check — 2026-10-16

本机训练、验证、保存和恢复检查已通过。

- 模型：DINOv2-Base，798×798，batch 1，标注训练图像 186 张。
- 初始化：DINOv2 预训练编码器＋新分割头，没有使用旧分割权重。
- 优化：原生 SDPA；训练前向 BF16，损失和原型统计 FP32。
- 训练：1100 步后保存，再从 latest.pth 恢复运行 2 步。
- 前 1000 步保留原型预热，后 102 步实际启用 DUE、CEM、UCCL。
- 19 类原型均就绪。可靠、模糊、可疑和高不确定像素四组均实际出现。
- 8 张验证图像分别完成学生与教师的原尺寸 798 滑窗验证。
- 优化器步数和原型更新进度从 1100 恢复到 1102；学生与教师参数继续更新。
- 所有记录的训练损失和保存的模型/原型张量均为有限值。

这是子集流程检查：无标注训练子集为 1100 张，验证子集 8 张，学习率日程采用诊断配置。
不是 180 轮正式训练，也不是完整 500 张验证集上的论文结果。权重已标记 diagnostic_only，不能恢复为正式实验。

WSL 输出目录：

`/home/chenyang/DU-Match/exp/cityscapes_integration_20261016`

目录中保留 config.yaml、unlabeled_1100.txt、val_8.txt、train.log、resume.log、integration_report.json、latest.pth、best.pth 和 smoke.pth。
其中 latest.pth 是完成诊断第一轮后的可恢复检查点；smoke.pth 是恢复后两步的检查快照，不能作为完整 epoch 检查点。

## 复现第一次诊断训练

在 DU-Match 仓库根目录执行，输出目录应使用新名字。配置及子集清单位于上述诊断目录：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 du_match.py \
  --config exp/cityscapes_integration_20261016/config.yaml \
  --labeled-id-path splits/cityscapes/1_16/labeled.txt \
  --unlabeled-id-path exp/cityscapes_integration_20261016/unlabeled_1100.txt \
  --validation-id-path exp/cityscapes_integration_20261016/val_8.txt \
  --save-path exp/cityscapes_integration_repeat \
  --stop-after-epoch 1 --amp-bf16 --torch-sdpa --cuda-memory-fraction 0.85
```

## 检查恢复

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 du_match.py \
  --config exp/cityscapes_integration_20261016/config.yaml \
  --labeled-id-path splits/cityscapes/1_16/labeled.txt \
  --unlabeled-id-path exp/cityscapes_integration_20261016/unlabeled_1100.txt \
  --validation-id-path exp/cityscapes_integration_20261016/val_8.txt \
  --save-path exp/cityscapes_integration_repeat \
  --smoke-steps 1102 --amp-bf16 --torch-sdpa --cuda-memory-fraction 0.85
```

服务器迁移后仍需先跑环境检查、短测和保存恢复验证；本机通过不代替服务器实测。
正式对照中的基线和 DU-Match 必须使用相同的精度与注意力实现；batch 1 的 CutMix donor 仍是自身。
