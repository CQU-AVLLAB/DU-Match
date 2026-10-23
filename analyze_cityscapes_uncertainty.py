import os
import csv
import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from model.semseg.dpt import DPT


# ============================================================
# 配置
# ============================================================

DATA_ROOT = "/home/chenyang/datasets/cityscapes"

CHECKPOINT = (
    "checkpoints/"
    "cityscapes_unimatch_v2_1_16labels_dinov2_base.pth"
)

OUTPUT_DIR = "analysis_cityscapes_paper"

# None = 完整 500 张
NUM_IMAGES = None

# 只保存多少张论文可视化
# 所有图片都参与统计，但只为前 N 张画复杂 matplotlib 图
NUM_VISUALIZE = 20

HIGH_CONF_THRESH = 0.95
HIGH_CONF_THRESH_2 = 0.99

ECE_BINS = 15


# ============================================================
# Cityscapes
# ============================================================

CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


PALETTE = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)


NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 基础工具
# ============================================================

def colorize_mask(mask):

    h, w = mask.shape

    color = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for class_id in range(len(PALETTE)):
        color[mask == class_id] = PALETTE[class_id]

    color[mask == 255] = [255, 255, 255]

    return color


def make_overlay(image, color_mask, alpha=0.55):

    image_np = np.asarray(
        image
    ).astype(np.float32)

    color_mask = color_mask.astype(
        np.float32
    )

    overlay = (
        image_np * (1.0 - alpha)
        +
        color_mask * alpha
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


def make_high_conf_error_overlay(
    image,
    pred,
    gt,
    confidence,
    threshold=0.95,
    alpha=0.80
):

    image_np = np.asarray(
        image
    ).astype(np.float32)

    valid = gt != 255

    high_conf_wrong = (
        valid
        &
        (pred != gt)
        &
        (confidence >= threshold)
    )

    result = image_np.copy()

    red = np.zeros_like(
        image_np
    )

    red[..., 0] = 255

    result[high_conf_wrong] = (
        image_np[high_conf_wrong]
        * (1.0 - alpha)
        +
        red[high_conf_wrong]
        * alpha
    )

    return (
        np.clip(
            result,
            0,
            255
        ).astype(np.uint8),
        high_conf_wrong
    )


def calculate_iou(
    pred,
    target,
    nclass=19
):

    valid = target != 255

    pred = pred[valid]
    target = target[valid]

    intersection = np.zeros(
        nclass,
        dtype=np.float64
    )

    union = np.zeros(
        nclass,
        dtype=np.float64
    )

    for cls in range(nclass):

        pred_cls = pred == cls
        target_cls = target == cls

        intersection[cls] = np.logical_and(
            pred_cls,
            target_cls
        ).sum()

        union[cls] = np.logical_or(
            pred_cls,
            target_cls
        ).sum()

    return intersection, union


# ============================================================
# ECE
# ============================================================

def compute_ece(
    confidences,
    correct,
    n_bins=15
):

    confidences = np.asarray(
        confidences
    )

    correct = np.asarray(
        correct
    ).astype(np.float32)

    boundaries = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        left = boundaries[i]
        right = boundaries[i + 1]

        if i == n_bins - 1:

            mask = (
                (confidences >= left)
                &
                (confidences <= right)
            )

        else:

            mask = (
                (confidences >= left)
                &
                (confidences < right)
            )

        if mask.sum() == 0:
            continue

        accuracy = correct[
            mask
        ].mean()

        confidence = confidences[
            mask
        ].mean()

        weight = mask.mean()

        ece += (
            abs(
                accuracy
                -
                confidence
            )
            *
            weight
        )

    return float(ece)


# ============================================================
# 模型
# ============================================================

def build_model():

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Building DINOv2-Base + DPT..."
    )

    model = DPT(
        encoder_size="base",
        nclass=19,
        features=128,
        out_channels=[
            96,
            192,
            384,
            768
        ],
    )

    print(
        "Loading checkpoint:"
    )

    print(
        CHECKPOINT
    )

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    if isinstance(
        ckpt,
        dict
    ):

        if "model" in ckpt:
            state_dict = ckpt["model"]

        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

        else:
            state_dict = ckpt

    else:
        state_dict = ckpt

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.cuda()
    model.eval()

    print(
        "Model device:",
        next(
            model.parameters()
        ).device
    )

    print(
        "Checkpoint loaded successfully."
    )

    return model


# ============================================================
# 推理
# ============================================================

@torch.inference_mode()
def predict(
    model,
    image
):

    original_w, original_h = image.size

    tensor = NORMALIZE(
        image
    ).unsqueeze(
        0
    ).cuda()

    new_h = int(
        original_h / 14 + 0.5
    ) * 14

    new_w = int(
        original_w / 14 + 0.5
    ) * 14

    tensor = F.interpolate(
        tensor,
        size=(
            new_h,
            new_w
        ),
        mode="bilinear",
        align_corners=True,
    )

    logits = model(
        tensor
    )

    logits = F.interpolate(
        logits,
        size=(
            original_h,
            original_w
        ),
        mode="bilinear",
        align_corners=True,
    )

    prob = torch.softmax(
        logits,
        dim=1
    )

    confidence, pred = prob.max(
        dim=1
    )

    entropy = -(
        prob
        *
        torch.log(
            prob + 1e-8
        )
    ).sum(
        dim=1
    )

    entropy = (
        entropy
        /
        math.log(
            prob.shape[1]
        )
    )

    return (
        pred[0]
        .cpu()
        .numpy()
        .astype(np.uint8),

        confidence[0]
        .cpu()
        .numpy(),

        entropy[0]
        .cpu()
        .numpy()
    )


# ============================================================
# 数据读取
# ============================================================

def get_cityscapes_val_files():

    image_root = os.path.join(
        DATA_ROOT,
        "leftImg8bit",
        "val"
    )

    files = []

    for city in sorted(
        os.listdir(
            image_root
        )
    ):

        city_dir = os.path.join(
            image_root,
            city
        )

        if not os.path.isdir(
            city_dir
        ):
            continue

        for filename in sorted(
            os.listdir(
                city_dir
            )
        ):

            if not filename.endswith(
                "_leftImg8bit.png"
            ):
                continue

            image_path = os.path.join(
                city_dir,
                filename
            )

            gt_filename = filename.replace(
                "_leftImg8bit.png",
                "_gtFine_labelTrainIds.png"
            )

            gt_path = os.path.join(
                DATA_ROOT,
                "gtFine",
                "val",
                city,
                gt_filename
            )

            if os.path.exists(
                gt_path
            ):

                files.append(
                    (
                        image_path,
                        gt_path
                    )
                )

    return files


# ============================================================
# 输出目录
# ============================================================

def ensure_dirs():

    folders = [
        "paper_examples",
        "prediction",
        "confidence",
        "uncertainty",
        "high_conf_error",
        "plots",
    ]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for folder in folders:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                folder
            ),
            exist_ok=True
        )


# ============================================================
# 论文单样本图
# ============================================================

def save_paper_figure(
    image,
    gt,
    pred,
    confidence,
    entropy,
    filename
):

    gt_color = colorize_mask(
        gt
    )

    pred_color = colorize_mask(
        pred
    )

    high_conf_overlay, _ = (
        make_high_conf_error_overlay(
            image,
            pred,
            gt,
            confidence,
            HIGH_CONF_THRESH
        )
    )

    valid = gt != 255

    entropy_valid = entropy[
        valid
    ]

    if entropy_valid.size > 0:

        entropy_vmax = np.percentile(
            entropy_valid,
            99
        )

    else:
        entropy_vmax = 1.0

    entropy_vmax = max(
        entropy_vmax,
        0.01
    )

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(30, 5.5)
    )

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Image",
        fontsize=15
    )

    axes[1].imshow(
        gt_color
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=15
    )

    axes[2].imshow(
        pred_color
    )

    axes[2].set_title(
        "Prediction",
        fontsize=15
    )

    im_conf = axes[3].imshow(
        confidence,
        cmap="viridis",
        vmin=0.80,
        vmax=1.00
    )

    axes[3].set_title(
        "Confidence",
        fontsize=15
    )

    fig.colorbar(
        im_conf,
        ax=axes[3],
        fraction=0.046,
        pad=0.04
    )

    im_unc = axes[4].imshow(
        entropy,
        cmap="magma",
        vmin=0.0,
        vmax=entropy_vmax
    )

    axes[4].set_title(
        "Predictive Uncertainty",
        fontsize=15
    )

    fig.colorbar(
        im_unc,
        ax=axes[4],
        fraction=0.046,
        pad=0.04
    )

    axes[5].imshow(
        high_conf_overlay
    )

    axes[5].set_title(
        "High-Confidence Errors",
        fontsize=15
    )

    for ax in axes:

        ax.axis(
            "off"
        )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "paper_examples",
        f"{filename}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# Reliability Diagram
# ============================================================

def save_reliability_diagram(
    confidences,
    correct,
    n_bins=15
):

    boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    bin_conf = []
    bin_acc = []

    for i in range(
        n_bins
    ):

        left = boundaries[i]
        right = boundaries[i + 1]

        if i == n_bins - 1:

            mask = (
                (confidences >= left)
                &
                (confidences <= right)
            )

        else:

            mask = (
                (confidences >= left)
                &
                (confidences < right)
            )

        if mask.sum() == 0:
            continue

        bin_conf.append(
            confidences[
                mask
            ].mean()
        )

        bin_acc.append(
            correct[
                mask
            ].mean()
        )

    plt.figure(
        figsize=(6.5, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.5,
        label="Perfect Calibration"
    )

    plt.plot(
        bin_conf,
        bin_acc,
        marker="o",
        linewidth=2,
        label="UniMatch V2"
    )

    plt.xlabel(
        "Confidence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "plots",
            "reliability_diagram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Correct vs Wrong Uncertainty
# ============================================================

def save_uncertainty_bar(
    mean_correct,
    mean_wrong
):

    labels = [
        "Correct Pixels",
        "Wrong Pixels"
    ]

    values = [
        mean_correct,
        mean_wrong
    ]

    plt.figure(
        figsize=(6.5, 5.5)
    )

    bars = plt.bar(
        labels,
        values
    )

    plt.ylabel(
        "Mean Predictive Uncertainty"
    )

    plt.title(
        "Uncertainty of Correct vs. Wrong Predictions"
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "plots",
            "uncertainty_correct_vs_wrong.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Confidence distribution for errors
# ============================================================

def save_error_confidence_histogram(
    wrong_confidences
):

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        wrong_confidences,
        bins=40
    )

    plt.axvline(
        HIGH_CONF_THRESH,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {HIGH_CONF_THRESH}"
    )

    plt.xlabel(
        "Confidence of Wrong Predictions"
    )

    plt.ylabel(
        "Pixel Count"
    )

    plt.title(
        "Confidence Distribution of Wrong Predictions"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "plots",
            "wrong_prediction_confidence_histogram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    model = build_model()

    files = get_cityscapes_val_files()

    print(
        f"\nCityscapes val images found: "
        f"{len(files)}"
    )

    if NUM_IMAGES is not None:

        files = files[
            :NUM_IMAGES
        ]

    print(
        f"Processing: "
        f"{len(files)} images"
    )

    total_intersection = np.zeros(
        19,
        dtype=np.float64
    )

    total_union = np.zeros(
        19,
        dtype=np.float64
    )

    all_confidences = []
    all_correct = []

    entropy_correct_list = []
    entropy_wrong_list = []

    wrong_confidences_list = []

    total_valid = 0
    total_wrong = 0

    total_high95 = 0
    total_high95_wrong = 0

    total_high99 = 0
    total_high99_wrong = 0

    statistics_path = os.path.join(
        OUTPUT_DIR,
        "statistics.csv"
    )

    with open(
        statistics_path,
        "w",
        newline=""
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )

        writer.writerow([
            "filename",
            "pixel_error_rate",
            "mean_confidence",
            "mean_uncertainty",
            "uncertainty_correct",
            "uncertainty_wrong",
            "high_conf_095_error_rate",
            "high_conf_099_error_rate",
            "fraction_errors_conf095",
            "fraction_errors_conf099",
        ])

        for index, (
            image_path,
            gt_path
        ) in enumerate(
            files
        ):

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            gt = np.array(
                Image.open(
                    gt_path
                )
            )

            pred, confidence, entropy = predict(
                model,
                image
            )

            inter, union = calculate_iou(
                pred,
                gt
            )

            total_intersection += inter
            total_union += union

            valid = gt != 255

            correct_mask = (
                valid
                &
                (pred == gt)
            )

            wrong_mask = (
                valid
                &
                (pred != gt)
            )

            valid_count = int(
                valid.sum()
            )

            wrong_count = int(
                wrong_mask.sum()
            )

            total_valid += valid_count
            total_wrong += wrong_count

            conf_valid = confidence[
                valid
            ]

            correct_valid = correct_mask[
                valid
            ].astype(
                np.float32
            )

            all_confidences.append(
                conf_valid
            )

            all_correct.append(
                correct_valid
            )

            ent_correct = entropy[
                correct_mask
            ]

            ent_wrong = entropy[
                wrong_mask
            ]

            if ent_correct.size > 0:

                entropy_correct_list.append(
                    ent_correct
                )

            if ent_wrong.size > 0:

                entropy_wrong_list.append(
                    ent_wrong
                )

            wrong_conf = confidence[
                wrong_mask
            ]

            if wrong_conf.size > 0:

                wrong_confidences_list.append(
                    wrong_conf
                )

            high95 = (
                valid
                &
                (
                    confidence
                    >=
                    HIGH_CONF_THRESH
                )
            )

            high99 = (
                valid
                &
                (
                    confidence
                    >=
                    HIGH_CONF_THRESH_2
                )
            )

            high95_count = int(
                high95.sum()
            )

            high99_count = int(
                high99.sum()
            )

            high95_wrong = int(
                (
                    high95
                    &
                    wrong_mask
                ).sum()
            )

            high99_wrong = int(
                (
                    high99
                    &
                    wrong_mask
                ).sum()
            )

            total_high95 += (
                high95_count
            )

            total_high95_wrong += (
                high95_wrong
            )

            total_high99 += (
                high99_count
            )

            total_high99_wrong += (
                high99_wrong
            )

            error_rate = (
                wrong_count
                /
                max(
                    valid_count,
                    1
                )
            )

            high95_error_rate = (
                high95_wrong
                /
                max(
                    high95_count,
                    1
                )
            )

            high99_error_rate = (
                high99_wrong
                /
                max(
                    high99_count,
                    1
                )
            )

            # 所有错误中有多少是高置信错误
            frac_errors_high95 = (
                high95_wrong
                /
                max(
                    wrong_count,
                    1
                )
            )

            frac_errors_high99 = (
                high99_wrong
                /
                max(
                    wrong_count,
                    1
                )
            )

            mean_unc_correct = (
                float(
                    ent_correct.mean()
                )
                if ent_correct.size > 0
                else 0.0
            )

            mean_unc_wrong = (
                float(
                    ent_wrong.mean()
                )
                if ent_wrong.size > 0
                else 0.0
            )

            filename = os.path.splitext(
                os.path.basename(
                    image_path
                )
            )[0]

            # -------------------------
            # 前 NUM_VISUALIZE 张出论文图
            # -------------------------

            if index < NUM_VISUALIZE:

                save_paper_figure(
                    image,
                    gt,
                    pred,
                    confidence,
                    entropy,
                    filename
                )

            writer.writerow([
                filename,
                f"{error_rate:.6f}",
                f"{conf_valid.mean():.6f}",
                f"{entropy[valid].mean():.6f}",
                f"{mean_unc_correct:.6f}",
                f"{mean_unc_wrong:.6f}",
                f"{high95_error_rate:.6f}",
                f"{high99_error_rate:.6f}",
                f"{frac_errors_high95:.6f}",
                f"{frac_errors_high99:.6f}",
            ])

            print(
                f"[{index+1}/{len(files)}] "
                f"{filename} | "
                f"Err={error_rate:.4f} | "
                f"Ucorr={mean_unc_correct:.4f} | "
                f"Uwrong={mean_unc_wrong:.4f} | "
                f"ErrConf95={frac_errors_high95:.4f}"
            )

    # ========================================================
    # 汇总
    # ========================================================

    all_confidences = np.concatenate(
        all_confidences
    )

    all_correct = np.concatenate(
        all_correct
    )

    entropy_correct_all = np.concatenate(
        entropy_correct_list
    )

    entropy_wrong_all = np.concatenate(
        entropy_wrong_list
    )

    wrong_confidences_all = np.concatenate(
        wrong_confidences_list
    )

    valid_classes = (
        total_union > 0
    )

    ious = np.zeros(
        19,
        dtype=np.float64
    )

    ious[
        valid_classes
    ] = (
        total_intersection[
            valid_classes
        ]
        /
        total_union[
            valid_classes
        ]
        *
        100
    )

    miou = float(
        ious[
            valid_classes
        ].mean()
    )

    ece = compute_ece(
        all_confidences,
        all_correct,
        ECE_BINS
    )

    mean_unc_correct = float(
        entropy_correct_all.mean()
    )

    mean_unc_wrong = float(
        entropy_wrong_all.mean()
    )

    pixel_error_rate = (
        total_wrong
        /
        max(
            total_valid,
            1
        )
    )

    high95_error_rate = (
        total_high95_wrong
        /
        max(
            total_high95,
            1
        )
    )

    high99_error_rate = (
        total_high99_wrong
        /
        max(
            total_high99,
            1
        )
    )

    # 所有错误中有多少同时属于高置信错误
    frac_wrong_high95 = (
        total_high95_wrong
        /
        max(
            total_wrong,
            1
        )
    )

    frac_wrong_high99 = (
        total_high99_wrong
        /
        max(
            total_wrong,
            1
        )
    )

    uncertainty_ratio = (
        mean_unc_wrong
        /
        max(
            mean_unc_correct,
            1e-10
        )
    )

    # ========================================================
    # 论文统计图
    # ========================================================

    save_reliability_diagram(
        all_confidences,
        all_correct,
        ECE_BINS
    )

    save_uncertainty_bar(
        mean_unc_correct,
        mean_unc_wrong
    )

    save_error_confidence_histogram(
        wrong_confidences_all
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            "=====================================\n"
        )

        f.write(
            "Cityscapes UniMatch V2 Reliability Analysis\n"
        )

        f.write(
            "=====================================\n"
        )

        f.write(
            f"Checkpoint: {CHECKPOINT}\n"
        )

        f.write(
            f"Images processed: {len(files)}\n\n"
        )

        f.write(
            f"mIoU: {miou:.4f}\n"
        )

        f.write(
            f"ECE: {ece:.6f}\n"
        )

        f.write(
            f"Mean confidence: "
            f"{all_confidences.mean():.6f}\n"
        )

        f.write(
            f"Pixel error rate: "
            f"{pixel_error_rate:.6f}\n"
        )

        f.write(
            f"Mean uncertainty correct: "
            f"{mean_unc_correct:.6f}\n"
        )

        f.write(
            f"Mean uncertainty wrong: "
            f"{mean_unc_wrong:.6f}\n"
        )

        f.write(
            f"Wrong/Correct uncertainty ratio: "
            f"{uncertainty_ratio:.4f}x\n"
        )

        f.write(
            f"Error rate among confidence >= 0.95: "
            f"{high95_error_rate:.6f}\n"
        )

        f.write(
            f"Error rate among confidence >= 0.99: "
            f"{high99_error_rate:.6f}\n"
        )

        f.write(
            f"Fraction of ALL errors with confidence >= 0.95: "
            f"{frac_wrong_high95:.6f}\n"
        )

        f.write(
            f"Fraction of ALL errors with confidence >= 0.99: "
            f"{frac_wrong_high99:.6f}\n"
        )

        f.write(
            "\nPer-class IoU\n"
        )

        f.write(
            "-------------------------------------\n"
        )

        for i in range(19):

            if total_union[i] > 0:

                f.write(
                    f"{i:02d} "
                    f"{CLASSES[i]:15s}: "
                    f"{ious[i]:6.2f}\n"
                )

    # ========================================================
    # 终端
    # ========================================================

    print("\n=====================================")
    print("FINAL SUMMARY")
    print("=====================================")

    print(
        f"mIoU: {miou:.4f}"
    )

    print(
        f"ECE: {ece:.6f}"
    )

    print(
        f"Mean confidence: "
        f"{all_confidences.mean():.6f}"
    )

    print(
        f"Pixel error rate: "
        f"{pixel_error_rate:.6f}"
    )

    print(
        f"U(correct): "
        f"{mean_unc_correct:.6f}"
    )

    print(
        f"U(wrong): "
        f"{mean_unc_wrong:.6f}"
    )

    print(
        f"U(wrong) / U(correct): "
        f"{uncertainty_ratio:.2f}x"
    )

    print(
        f"Error rate | Conf>=0.95: "
        f"{high95_error_rate:.6f}"
    )

    print(
        f"Error rate | Conf>=0.99: "
        f"{high99_error_rate:.6f}"
    )

    print(
        f"Among ALL errors, Conf>=0.95: "
        f"{frac_wrong_high95:.6f}"
    )

    print(
        f"Among ALL errors, Conf>=0.99: "
        f"{frac_wrong_high99:.6f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()