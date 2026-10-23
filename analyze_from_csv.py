import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = "analysis_cityscapes_paper/statistics.csv"
OUTPUT_DIR = "analysis_cityscapes_paper/post_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))


# ============================================================
# 1. 基础汇总
# ============================================================

mean_pixel_error = df["pixel_error_rate"].mean()
mean_confidence = df["mean_confidence"].mean()
mean_uncertainty = df["mean_uncertainty"].mean()

mean_unc_correct = df["uncertainty_correct"].mean()
mean_unc_wrong = df["uncertainty_wrong"].mean()

mean_high95_error = df["high_conf_095_error_rate"].mean()
mean_high99_error = df["high_conf_099_error_rate"].mean()

mean_fraction_errors_conf095 = df[
    "fraction_errors_conf095"
].mean()

mean_fraction_errors_conf099 = df[
    "fraction_errors_conf099"
].mean()

uncertainty_ratio = (
    mean_unc_wrong
    / max(mean_unc_correct, 1e-12)
)


summary = f"""
=====================================
Cityscapes Post Analysis
=====================================

Images: {len(df)}

Mean pixel error rate:
{mean_pixel_error:.6f}

Mean confidence:
{mean_confidence:.6f}

Mean uncertainty:
{mean_uncertainty:.6f}

Mean uncertainty - correct pixels:
{mean_unc_correct:.6f}

Mean uncertainty - wrong pixels:
{mean_unc_wrong:.6f}

Wrong/Correct uncertainty ratio:
{uncertainty_ratio:.4f}x

Mean error rate among confidence >= 0.95:
{mean_high95_error:.6f}

Mean error rate among confidence >= 0.99:
{mean_high99_error:.6f}

Mean fraction of errors with confidence >= 0.95:
{mean_fraction_errors_conf095:.6f}

Mean fraction of errors with confidence >= 0.99:
{mean_fraction_errors_conf099:.6f}
"""

print(summary)

with open(
    os.path.join(
        OUTPUT_DIR,
        "summary_from_csv.txt"
    ),
    "w"
) as f:
    f.write(summary)


# ============================================================
# 2. Correct vs Wrong uncertainty
# ============================================================

plt.figure(figsize=(6, 5))

labels = [
    "Correct Pixels",
    "Wrong Pixels"
]

values = [
    mean_unc_correct,
    mean_unc_wrong
]

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Mean Predictive Uncertainty"
)

plt.title(
    "Predictive Uncertainty: Correct vs Wrong"
)

for bar, value in zip(
    bars,
    values
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_correct_vs_wrong.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. High-confidence error fraction
# ============================================================

plt.figure(figsize=(6, 5))

labels = [
    "Conf >= 0.95",
    "Conf >= 0.99"
]

values = [
    mean_fraction_errors_conf095,
    mean_fraction_errors_conf099
]

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Fraction of Wrong Pixels"
)

plt.title(
    "High-Confidence Errors"
)

plt.ylim(
    0,
    max(values) * 1.25
)

for bar, value in zip(
    bars,
    values
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "high_confidence_error_fraction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. Confidence vs error scatter
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    df["mean_confidence"],
    df["pixel_error_rate"],
    s=12,
    alpha=0.5
)

plt.xlabel(
    "Mean Confidence"
)

plt.ylabel(
    "Pixel Error Rate"
)

plt.title(
    "Confidence vs Prediction Error"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confidence_vs_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. Uncertainty vs error scatter
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    df["mean_uncertainty"],
    df["pixel_error_rate"],
    s=12,
    alpha=0.5
)

plt.xlabel(
    "Mean Predictive Uncertainty"
)

plt.ylabel(
    "Pixel Error Rate"
)

plt.title(
    "Uncertainty vs Prediction Error"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_vs_error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. Wrong vs correct uncertainty scatter
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(
    df["uncertainty_correct"],
    df["uncertainty_wrong"],
    s=12,
    alpha=0.5
)

max_val = max(
    df["uncertainty_correct"].max(),
    df["uncertainty_wrong"].max()
)

plt.plot(
    [0, max_val],
    [0, max_val],
    "--"
)

plt.xlabel(
    "Uncertainty of Correct Pixels"
)

plt.ylabel(
    "Uncertainty of Wrong Pixels"
)

plt.title(
    "Wrong vs Correct Uncertainty"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "wrong_vs_correct_uncertainty.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved to:")
print(os.path.abspath(OUTPUT_DIR))