import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# 配置
# ============================================================

OUT_DIR = "expected_paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 固定随机种子，使示意图每次一致
np.random.seed(42)


# ============================================================
# Figure A
# Qualitative Ablation Figure
# 理想状态下的伪标签筛选示意
# ============================================================

def make_qualitative_ablation():

    H, W = 180, 320

    # --------------------------------------------------------
    # 构造一个简化的道路场景 GT
    # 这里只用于汇报时说明筛选机制，不是真实实验结果
    # --------------------------------------------------------

    gt = np.zeros((H, W), dtype=np.uint8)

    # sky
    gt[:65, :] = 0

    # vegetation/buildings
    gt[40:110, :75] = 1
    gt[35:105, 245:] = 1

    # road
    for y in range(70, H):
        center = W // 2
        width = int((y - 50) * 1.5)

        x1 = max(center - width, 0)
        x2 = min(center + width, W)

        gt[y, x1:x2] = 2

    # sidewalk
    gt[110:, :55] = 3
    gt[110:, 270:] = 3

    # car/object
    gt[100:135, 185:230] = 4

    # --------------------------------------------------------
    # Baseline pseudo label
    # 人为制造若干“虚假自信错误”
    # --------------------------------------------------------

    baseline = gt.copy()

    # 错误区域 1：右侧暗区
    baseline[65:115, 235:285] = 1

    # 错误区域 2：道路边界
    baseline[115:160, 60:95] = 2

    # 错误区域 3：车附近
    baseline[95:140, 175:205] = 1

    # --------------------------------------------------------
    # Only Entropy
    #
    # 255 表示 rejected pseudo-label
    # --------------------------------------------------------

    entropy_mask = np.ones((H, W), dtype=bool)

    # 熵能过滤掉明显不确定边界
    entropy_mask[115:160, 60:95] = False
    entropy_mask[95:115, 175:205] = False

    entropy_only = baseline.copy()
    entropy_only[~entropy_mask] = 255

    # --------------------------------------------------------
    # Full = entropy + consistency
    #
    # Consistency进一步发现：
    # 一些看似 confidence 很高，但扰动后不稳定的区域
    # --------------------------------------------------------

    full_mask = entropy_mask.copy()

    # 删除右侧假高置信错误
    full_mask[65:115, 235:285] = False

    # 再删除车附近剩余不稳定错误
    full_mask[115:140, 175:205] = False

    full = baseline.copy()
    full[~full_mask] = 255

    # Consistency额外删除的区域
    removed_consistency = (
        entropy_mask
        &
        (~full_mask)
    )

    # --------------------------------------------------------
    # 计算错误伪标签
    # --------------------------------------------------------

    baseline_wrong = (
        baseline != gt
    )

    entropy_wrong = (
        (entropy_only != 255)
        &
        (entropy_only != gt)
    )

    full_wrong = (
        (full != 255)
        &
        (full != gt)
    )

    # --------------------------------------------------------
    # palette
    # --------------------------------------------------------

    colors = [
        "#87CEEB",  # sky
        "#6B8E23",  # building / vegetation
        "#806080",  # road
        "#FF4DD2",  # sidewalk
        "#00008B",  # car
        "#D3D3D3",  # rejected
    ]

    cmap = ListedColormap(colors)

    def prepare_mask(mask):
        tmp = mask.copy()
        tmp[tmp == 255] = 5
        return tmp

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )

    axes[0, 0].imshow(
        prepare_mask(baseline),
        cmap=cmap,
        vmin=0,
        vmax=5
    )
    axes[0, 0].set_title(
        "(a) w/o Uncertainty",
        fontsize=14
    )

    axes[0, 1].imshow(
        prepare_mask(entropy_only),
        cmap=cmap,
        vmin=0,
        vmax=5
    )
    axes[0, 1].set_title(
        r"(b) w/ $U_{ent}$",
        fontsize=14
    )

    axes[0, 2].imshow(
        prepare_mask(full),
        cmap=cmap,
        vmin=0,
        vmax=5
    )
    axes[0, 2].set_title(
        r"(c) Full: $U_{ent}+U_{cons}$",
        fontsize=14
    )

    # Wrong baseline
    axes[1, 0].imshow(
        baseline_wrong,
        cmap="Reds",
        vmin=0,
        vmax=1
    )
    axes[1, 0].set_title(
        "Wrong Selected Pseudo-labels",
        fontsize=13
    )

    # Consistency removed
    axes[1, 1].imshow(
        removed_consistency,
        cmap="Reds",
        vmin=0,
        vmax=1
    )
    axes[1, 1].set_title(
        r"Removed by $U_{cons}$",
        fontsize=13
    )

    # Remaining wrong labels
    axes[1, 2].imshow(
        full_wrong,
        cmap="Reds",
        vmin=0,
        vmax=1
    )
    axes[1, 2].set_title(
        "Remaining Wrong Pseudo-labels",
        fontsize=13
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        "Illustrative Qualitative Ablation of Pseudo-label Reliability Filtering",
        fontsize=16
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "Fig_A_qualitative_ablation.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# Figure B
# Pseudo-label Quality Evolution
# ============================================================

def make_pseudo_label_quality_curve():

    # 假设训练 40k iterations
    iterations = np.array([
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40
    ])

    # --------------------------------------------------------
    # 预期/理想趋势：
    #
    # Baseline:
    # 前期上升，随后伪标签噪声使性能趋于饱和
    #
    # Entropy:
    # 更稳定
    #
    # Full:
    # 更快、更平滑、最终更高
    # --------------------------------------------------------

    baseline_acc = np.array([
        72.0,
        77.8,
        81.0,
        82.7,
        83.4,
        83.8,
        83.9,
        83.7,
        83.6
    ])

    entropy_acc = np.array([
        72.0,
        78.5,
        82.1,
        84.0,
        85.0,
        85.6,
        86.0,
        86.2,
        86.3
    ])

    full_acc = np.array([
        72.0,
        79.1,
        83.0,
        85.2,
        86.4,
        87.2,
        87.7,
        88.0,
        88.2
    ])

    # --------------------------------------------------------
    # Mask Ratio
    # 被认为可靠并被用于训练的伪标签比例
    #
    # 理想趋势：
    # 初期较严格，随着 teacher 变好逐渐增大
    # --------------------------------------------------------

    mask_ratio_baseline = np.array([
        42,
        56,
        67,
        74,
        79,
        82,
        84,
        85,
        86
    ])

    mask_ratio_entropy = np.array([
        30,
        43,
        55,
        64,
        70,
        74,
        77,
        79,
        80
    ])

    mask_ratio_full = np.array([
        24,
        36,
        49,
        59,
        66,
        71,
        75,
        78,
        80
    ])

    # ========================================================
    # 两个 panel
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    # ------------------------
    # Pseudo-label accuracy
    # ------------------------

    axes[0].plot(
        iterations,
        baseline_acc,
        marker="o",
        linewidth=2,
        label="w/o Uncertainty"
    )

    axes[0].plot(
        iterations,
        entropy_acc,
        marker="s",
        linewidth=2,
        label=r"w/ $U_{ent}$"
    )

    axes[0].plot(
        iterations,
        full_acc,
        marker="^",
        linewidth=2.5,
        label=r"Full ($U_{ent}+U_{cons}$)"
    )

    axes[0].set_xlabel(
        "Training Iteration (×1k)"
    )

    axes[0].set_ylabel(
        "Pseudo-label Accuracy (%)"
    )

    axes[0].set_title(
        "(a) Pseudo-label Quality"
    )

    axes[0].set_ylim(
        70,
        90
    )

    axes[0].grid(
        alpha=0.25
    )

    axes[0].legend()

    # ------------------------
    # Mask ratio
    # ------------------------

    axes[1].plot(
        iterations,
        mask_ratio_baseline,
        marker="o",
        linewidth=2,
        label="w/o Uncertainty"
    )

    axes[1].plot(
        iterations,
        mask_ratio_entropy,
        marker="s",
        linewidth=2,
        label=r"w/ $U_{ent}$"
    )

    axes[1].plot(
        iterations,
        mask_ratio_full,
        marker="^",
        linewidth=2.5,
        label=r"Full ($U_{ent}+U_{cons}$)"
    )

    axes[1].set_xlabel(
        "Training Iteration (×1k)"
    )

    axes[1].set_ylabel(
        "Reliable Pseudo-label Ratio (%)"
    )

    axes[1].set_title(
        "(b) Reliable Mask Ratio"
    )

    axes[1].set_ylim(
        20,
        90
    )

    axes[1].grid(
        alpha=0.25
    )

    axes[1].legend()

    fig.suptitle(
        "Expected Evolution of Pseudo-label Quality During Training",
        fontsize=15
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "Fig_B_pseudo_label_quality_curve.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# Figure C
# Sensitivity Analysis
# ============================================================

def make_sensitivity_analysis():

    # ========================================================
    # τ sensitivity
    # ========================================================

    tau_values = np.array([
        0.80,
        0.85,
        0.90,
        0.95,
        0.97,
        0.99
    ])

    # 理想情况：
    # 阈值太低 -> 噪声伪标签多
    # 阈值太高 -> 可用伪标签太少
    # 中间最优
    tau_miou = np.array([
        72.7,
        73.4,
        74.1,
        74.8,
        74.6,
        73.9
    ])

    # ========================================================
    # α / β sensitivity
    #
    # U = α U_ent + β U_cons
    # 这里 β = 1 - α
    # ========================================================

    alpha_values = np.array([
        0.0,
        0.2,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0
    ])

    beta_values = (
        1.0
        -
        alpha_values
    )

    alpha_miou = np.array([
        73.5,   # only consistency
        74.1,
        74.6,
        74.9,   # balanced best
        74.8,
        74.2,
        73.8    # only entropy
    ])

    # ========================================================
    # λ sensitivity
    #
    # Reliability:
    # R = C * exp(-λ U)
    #
    # λ 太小 -> uncertainty影响太弱
    # λ 太大 -> 筛选过度
    # ========================================================

    lambda_values = np.array([
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0
    ])

    lambda_miou = np.array([
        73.6,
        74.1,
        74.6,
        74.9,
        74.7,
        74.2,
        73.5
    ])

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8)
    )

    # τ
    axes[0].plot(
        tau_values,
        tau_miou,
        marker="o",
        linewidth=2.2
    )

    best_tau = np.argmax(
        tau_miou
    )

    axes[0].scatter(
        tau_values[best_tau],
        tau_miou[best_tau],
        s=90,
        zorder=5
    )

    axes[0].set_xlabel(
        r"Reliability Threshold $\tau$"
    )

    axes[0].set_ylabel(
        "mIoU (%)"
    )

    axes[0].set_title(
        r"(a) Sensitivity to $\tau$"
    )

    axes[0].grid(
        alpha=0.25
    )

    # α
    axes[1].plot(
        alpha_values,
        alpha_miou,
        marker="s",
        linewidth=2.2
    )

    best_alpha = np.argmax(
        alpha_miou
    )

    axes[1].scatter(
        alpha_values[best_alpha],
        alpha_miou[best_alpha],
        s=90,
        zorder=5
    )

    axes[1].set_xlabel(
        r"$\alpha$  ($\beta = 1-\alpha$)"
    )

    axes[1].set_ylabel(
        "mIoU (%)"
    )

    axes[1].set_title(
        r"(b) Balance of $U_{ent}$ and $U_{cons}$"
    )

    axes[1].grid(
        alpha=0.25
    )

    # λ
    axes[2].plot(
        lambda_values,
        lambda_miou,
        marker="^",
        linewidth=2.2
    )

    best_lambda = np.argmax(
        lambda_miou
    )

    axes[2].scatter(
        lambda_values[best_lambda],
        lambda_miou[best_lambda],
        s=90,
        zorder=5
    )

    axes[2].set_xlabel(
        r"Uncertainty Decay $\lambda$"
    )

    axes[2].set_ylabel(
        "mIoU (%)"
    )

    axes[2].set_title(
        r"(c) Sensitivity to $\lambda$"
    )

    axes[2].grid(
        alpha=0.25
    )

    # 让三张图 y 范围一致，更像论文
    for ax in axes:
        ax.set_ylim(
            72.5,
            75.3
        )

    fig.suptitle(
        "Expected Hyper-parameter Sensitivity of the Proposed Method",
        fontsize=15
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        "Fig_C_sensitivity_analysis.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    make_qualitative_ablation()

    make_pseudo_label_quality_curve()

    make_sensitivity_analysis()

    print("\n=====================================")
    print("All expected figures generated.")
    print("=====================================")
    print(
        os.path.abspath(
            OUT_DIR
        )
    )