# Implementation verification

Verified in the existing WSL Ubuntu-22.04 unimatchv2 environment (PyTorch 2.4.1+cu121, RTX 4090 Laptop GPU).

- Seven formula/gradient/spatial-alignment/initialization/partial-bank tests passed.
- Two-process CPU Gloo test passed: differing class counts produce globally pixel-weighted prototypes and dispersion.
- Original sibling model versus copied model: identical logits and complementary feature perturbation under the same weights and RNG; encoder map has expected shape.
- Actual VOC labeled/unlabeled data, DINOv2 base, single GPU: two training steps completed at crop 518 / batch 1, including baseline warmup, active DU-Match, backward, optimizer update, teacher EMA, and smoke checkpoint save.
- Validation-only diagnostic loaded the final model and prototype checkpoint and produced JSON; missing subsets yield null instead of fabricated percentages.
- Compilation and whitespace checks passed.
- Copy audit: 815 original non-Git/non-bytecode files compared; all preserved, with only copied model/semseg/dpt.py intentionally modified. Baseline entry point and existing config/script modifications were preserved.

Final smoke artifacts: exp/du_match_smoke_full/smoke.pth and diagnostics.json.
The earlier exp/du_match_smoke and exp/du_match_smoke_verified outputs are intermediate debugging runs, not final verification or scientific results. Cleanup was rejected by the execution policy, so these artifacts and temporary implementation scripts in the desktop workspace were retained. No original outputs were removed.

Limits: only two optimizer steps; no full validation mIoU, full epoch resume, true multi-GPU CUDA training, or Cityscapes 798-crop training has been verified. Fresh model smoke tests may have zero selected pseudo-supervised pixels; synthetic four-state and gradient tests separately verify active hard/soft/corrected losses. These tests do not establish scientific novelty or performance improvement.