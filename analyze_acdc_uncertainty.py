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

DATA_ROOT = "/home/chenyang/datasets/acdc"

CHECKPOINT = (
    "checkpoints/"
    "cityscapes_unimatch_v2_1_16labels_dinov2_base.pth"
)

OUTPUT_DIR = "analysis_acdc_paper"

WEATHERS = [
    "fog",
    "night",
    "rain",
    "snow",
]

HIGH_CONF_THRESH = 0.95
HIGH_CONF_THRESH_2 = 0.99

ECE_BINS = 15

# 每种天气只保存前几张论文可视化
NUM_VIS_PER_WEATHER = 5


# ============================================================
# Cityscapes / ACDC 19 类
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
# 工具函数
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
        & (pred != gt)
        & (confidence >= threshold)
    )

    result = image_np.copy()

    red = np.zeros_like(image_np)
    red[..., 0] = 255

    result[high_conf_wrong] = (
        image_np[high_conf_wrong] * (1.0 - alpha)
        + red[high_conf_wrong] * alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


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

    if isinstance(ckpt, dict):
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
        next(model.parameters()).device
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
    ).unsqueeze(0).cuda()

    new_h = int(
        original_h / 14 + 0.5
    ) * 14

    new_w = int(
        original_w / 14 + 0.5
    ) * 14

    tensor = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=True,
    )

    logits = model(
        tensor
    )

    logits = F.interpolate(
        logits,
        size=(original_h, original_w),
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
        pred[0].cpu().numpy().astype(np.uint8),
        confidence[0].cpu().numpy(),
        entropy[0].cpu().numpy()
    )


# ============================================================
# ACDC 文件匹配
# ============================================================

def get_acdc_val_files(
    weather
):
    rgb_root = os.path.join(
        DATA_ROOT,
        "rgb_anon",
        weather,
        "val"
    )

    gt_root = os.path.join(
        DATA_ROOT,
        "gt",
        weather,
        "val"
    )

    files = []

    for sequence in sorted(
        os.listdir(rgb_root)
    ):
        seq_rgb_dir = os.path.join(
            rgb_root,
            sequence
        )

        if not os.path.isdir(
            seq_rgb_dir
        ):
            continue

        for filename in sorted(
            os.listdir(seq_rgb_dir)
        ):
            if not filename.endswith(
                "_rgb_anon.png"
            ):
                continue

            image_path = os.path.join(
                seq_rgb_dir,
                filename
            )

            gt_filename = filename.replace(
                "_rgb_anon.png",
                "_gt_labelTrainIds.png"
            )

            gt_path = os.path.join(
                gt_root,
                sequence,
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

            else:
                print(
                    "[WARNING] GT not found:",
                    gt_path
                )

    return files


# ============================================================
# 输出目录
# ============================================================

def ensure_dirs():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "plots"
        ),
        exist_ok=True
    )

    for weather in WEATHERS:
        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                "paper_examples",
                weather
            ),
            exist_ok=True
        )


# ============================================================
# 单张论文图
# ============================================================

def save_paper_figure(
    image,
    gt,
    pred,
    confidence,
    entropy,
    weather,
    filename
):
    gt_color = colorize_mask(
        gt
    )

    pred_color = colorize_mask(
        pred
    )

    high_conf_error = (
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
        "Image"
    )

    axes[1].imshow(
        gt_color
    )
    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        pred_color
    )
    axes[2].set_title(
        "Prediction"
    )

    im1 = axes[3].imshow(
        confidence,
        cmap="viridis",
        vmin=0.80,
        vmax=1.00
    )

    axes[3].set_title(
        "Confidence"
    )

    fig.colorbar(
        im1,
        ax=axes[3],
        fraction=0.046,
        pad=0.04
    )

    im2 = axes[4].imshow(
        entropy,
        cmap="magma",
        vmin=0.0,
        vmax=entropy_vmax
    )

    axes[4].set_title(
        "Predictive Uncertainty"
    )

    fig.colorbar(
        im2,
        ax=axes[4],
        fraction=0.046,
        pad=0.04
    )

    axes[5].imshow(
        high_conf_error
    )

    axes[5].set_title(
        "High-Confidence Errors"
    )

    for ax in axes:
        ax.axis(
            "off"
        )

    fig.suptitle(
        f"ACDC - {weather.capitalize()}",
        fontsize=18
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "paper_examples",
        weather,
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
# Streaming accumulator
# ============================================================

def create_stats():
    return {
        "intersection": np.zeros(
            19,
            dtype=np.float64
        ),

        "union": np.zeros(
            19,
            dtype=np.float64
        ),

        "valid_pixels": 0,

        "wrong_pixels": 0,

        "confidence_sum": 0.0,

        "entropy_correct_sum": 0.0,

        "entropy_correct_count": 0,

        "entropy_wrong_sum": 0.0,

        "entropy_wrong_count": 0,

        "high95_pixels": 0,

        "high95_wrong": 0,

        "high99_pixels": 0,

        "high99_wrong": 0,
    }


def update_stats(
    stats,
    pred,
    gt,
    confidence,
    entropy
):
    valid = gt != 255

    correct = (
        valid
        &
        (pred == gt)
    )

    wrong = (
        valid
        &
        (pred != gt)
    )

    inter, union = calculate_iou(
        pred,
        gt
    )

    stats["intersection"] += inter
    stats["union"] += union

    stats["valid_pixels"] += int(
        valid.sum()
    )

    stats["wrong_pixels"] += int(
        wrong.sum()
    )

    stats["confidence_sum"] += float(
        confidence[valid].sum()
    )

    stats[
        "entropy_correct_sum"
    ] += float(
        entropy[correct].sum()
    )

    stats[
        "entropy_correct_count"
    ] += int(
        correct.sum()
    )

    stats[
        "entropy_wrong_sum"
    ] += float(
        entropy[wrong].sum()
    )

    stats[
        "entropy_wrong_count"
    ] += int(
        wrong.sum()
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

    stats[
        "high95_pixels"
    ] += int(
        high95.sum()
    )

    stats[
        "high95_wrong"
    ] += int(
        (
            high95
            &
            wrong
        ).sum()
    )

    stats[
        "high99_pixels"
    ] += int(
        high99.sum()
    )

    stats[
        "high99_wrong"
    ] += int(
        (
            high99
            &
            wrong
        ).sum()
    )


def summarize_stats(
    stats
):
    valid_classes = (
        stats["union"] > 0
    )

    ious = np.zeros(
        19,
        dtype=np.float64
    )

    ious[
        valid_classes
    ] = (
        stats["intersection"][
            valid_classes
        ]
        /
        stats["union"][
            valid_classes
        ]
        *
        100.0
    )

    miou = float(
        ious[
            valid_classes
        ].mean()
    )

    mean_confidence = (
        stats["confidence_sum"]
        /
        max(
            stats["valid_pixels"],
            1
        )
    )

    pixel_error_rate = (
        stats["wrong_pixels"]
        /
        max(
            stats["valid_pixels"],
            1
        )
    )

    mean_unc_correct = (
        stats[
            "entropy_correct_sum"
        ]
        /
        max(
            stats[
                "entropy_correct_count"
            ],
            1
        )
    )

    mean_unc_wrong = (
        stats[
            "entropy_wrong_sum"
        ]
        /
        max(
            stats[
                "entropy_wrong_count"
            ],
            1
        )
    )

    unc_ratio = (
        mean_unc_wrong
        /
        max(
            mean_unc_correct,
            1e-12
        )
    )

    high95_error_rate = (
        stats[
            "high95_wrong"
        ]
        /
        max(
            stats[
                "high95_pixels"
            ],
            1
        )
    )

    high99_error_rate = (
        stats[
            "high99_wrong"
        ]
        /
        max(
            stats[
                "high99_pixels"
            ],
            1
        )
    )

    fraction_errors_conf095 = (
        stats[
            "high95_wrong"
        ]
        /
        max(
            stats[
                "wrong_pixels"
            ],
            1
        )
    )

    fraction_errors_conf099 = (
        stats[
            "high99_wrong"
        ]
        /
        max(
            stats[
                "wrong_pixels"
            ],
            1
        )
    )

    return {
        "miou": miou,

        "mean_confidence":
            mean_confidence,

        "pixel_error_rate":
            pixel_error_rate,

        "uncertainty_correct":
            mean_unc_correct,

        "uncertainty_wrong":
            mean_unc_wrong,

        "uncertainty_ratio":
            unc_ratio,

        "high95_error_rate":
            high95_error_rate,

        "high99_error_rate":
            high99_error_rate,

        "fraction_errors_conf095":
            fraction_errors_conf095,

        "fraction_errors_conf099":
            fraction_errors_conf099,

        "ious":
            ious,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()

    model = build_model()

    weather_stats = {}
    overall_stats = create_stats()

    csv_path = os.path.join(
        OUTPUT_DIR,
        "statistics.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow([
            "weather",
            "filename",
            "pixel_error_rate",
            "mean_confidence",
            "uncertainty_correct",
            "uncertainty_wrong",
            "fraction_errors_conf095",
            "fraction_errors_conf099",
        ])

        for weather in WEATHERS:

            files = get_acdc_val_files(
                weather
            )

            print(
                f"\n====================================="
            )

            print(
                f"{weather.upper()} val images: "
                f"{len(files)}"
            )

            print(
                f"====================================="
            )

            stats = create_stats()

            for index, (
                image_path,
                gt_path
            ) in enumerate(files):

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

                valid = gt != 255

                correct = (
                    valid
                    &
                    (pred == gt)
                )

                wrong = (
                    valid
                    &
                    (pred != gt)
                )

                update_stats(
                    stats,
                    pred,
                    gt,
                    confidence,
                    entropy
                )

                update_stats(
                    overall_stats,
                    pred,
                    gt,
                    confidence,
                    entropy
                )

                valid_count = int(
                    valid.sum()
                )

                wrong_count = int(
                    wrong.sum()
                )

                error_rate = (
                    wrong_count
                    /
                    max(
                        valid_count,
                        1
                    )
                )

                mean_conf = float(
                    confidence[
                        valid
                    ].mean()
                )

                mean_unc_correct = (
                    float(
                        entropy[
                            correct
                        ].mean()
                    )
                    if correct.sum() > 0
                    else 0.0
                )

                mean_unc_wrong = (
                    float(
                        entropy[
                            wrong
                        ].mean()
                    )
                    if wrong.sum() > 0
                    else 0.0
                )

                high95_wrong = (
                    wrong
                    &
                    (
                        confidence
                        >=
                        HIGH_CONF_THRESH
                    )
                )

                high99_wrong = (
                    wrong
                    &
                    (
                        confidence
                        >=
                        HIGH_CONF_THRESH_2
                    )
                )

                frac95 = (
                    int(
                        high95_wrong.sum()
                    )
                    /
                    max(
                        wrong_count,
                        1
                    )
                )

                frac99 = (
                    int(
                        high99_wrong.sum()
                    )
                    /
                    max(
                        wrong_count,
                        1
                    )
                )

                filename = os.path.splitext(
                    os.path.basename(
                        image_path
                    )
                )[0]

                writer.writerow([
                    weather,
                    filename,
                    f"{error_rate:.6f}",
                    f"{mean_conf:.6f}",
                    f"{mean_unc_correct:.6f}",
                    f"{mean_unc_wrong:.6f}",
                    f"{frac95:.6f}",
                    f"{frac99:.6f}",
                ])

                if index < NUM_VIS_PER_WEATHER:
                    save_paper_figure(
                        image,
                        gt,
                        pred,
                        confidence,
                        entropy,
                        weather,
                        filename
                    )

                print(
                    f"[{index + 1}/{len(files)}] "
                    f"{weather} | "
                    f"Err={error_rate:.4f} | "
                    f"Conf={mean_conf:.4f} | "
                    f"Ucorr={mean_unc_correct:.4f} | "
                    f"Uwrong={mean_unc_wrong:.4f} | "
                    f"ErrConf95={frac95:.4f}"
                )

            weather_stats[
                weather
            ] = summarize_stats(
                stats
            )

    overall_summary = summarize_stats(
        overall_stats
    )

    # ========================================================
    # 输出 summary
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
            "ACDC UniMatch V2 Uncertainty Analysis\n"
        )

        f.write(
            "=====================================\n\n"
        )

        for weather in WEATHERS:
            s = weather_stats[
                weather
            ]

            f.write(
                f"[{weather.upper()}]\n"
            )

            f.write(
                f"mIoU: "
                f"{s['miou']:.4f}\n"
            )

            f.write(
                f"Mean confidence: "
                f"{s['mean_confidence']:.6f}\n"
            )

            f.write(
                f"Pixel error rate: "
                f"{s['pixel_error_rate']:.6f}\n"
            )

            f.write(
                f"U(correct): "
                f"{s['uncertainty_correct']:.6f}\n"
            )

            f.write(
                f"U(wrong): "
                f"{s['uncertainty_wrong']:.6f}\n"
            )

            f.write(
                f"U(wrong)/U(correct): "
                f"{s['uncertainty_ratio']:.4f}x\n"
            )

            f.write(
                f"Fraction errors Conf>=0.95: "
                f"{s['fraction_errors_conf095']:.6f}\n"
            )

            f.write(
                f"Fraction errors Conf>=0.99: "
                f"{s['fraction_errors_conf099']:.6f}\n\n"
            )

        f.write(
            "=====================================\n"
        )

        f.write(
            "[OVERALL]\n"
        )

        f.write(
            "=====================================\n"
        )

        f.write(
            f"mIoU: "
            f"{overall_summary['miou']:.4f}\n"
        )

        f.write(
            f"Mean confidence: "
            f"{overall_summary['mean_confidence']:.6f}\n"
        )

        f.write(
            f"Pixel error rate: "
            f"{overall_summary['pixel_error_rate']:.6f}\n"
        )

        f.write(
            f"U(correct): "
            f"{overall_summary['uncertainty_correct']:.6f}\n"
        )

        f.write(
            f"U(wrong): "
            f"{overall_summary['uncertainty_wrong']:.6f}\n"
        )

        f.write(
            f"U(wrong)/U(correct): "
            f"{overall_summary['uncertainty_ratio']:.4f}x\n"
        )

        f.write(
            f"Fraction errors Conf>=0.95: "
            f"{overall_summary['fraction_errors_conf095']:.6f}\n"
        )

        f.write(
            f"Fraction errors Conf>=0.99: "
            f"{overall_summary['fraction_errors_conf099']:.6f}\n"
        )

    # ========================================================
    # 打印结果
    # ========================================================

    print(
        "\n====================================="
    )

    print(
        "FINAL ACDC SUMMARY"
    )

    print(
        "====================================="
    )

    for weather in WEATHERS:
        s = weather_stats[
            weather
        ]

        print(
            f"\n[{weather.upper()}]"
        )

        print(
            f"mIoU: "
            f"{s['miou']:.2f}"
        )

        print(
            f"Mean confidence: "
            f"{s['mean_confidence']:.4f}"
        )

        print(
            f"Pixel error: "
            f"{s['pixel_error_rate']:.4f}"
        )

        print(
            f"U(correct): "
            f"{s['uncertainty_correct']:.4f}"
        )

        print(
            f"U(wrong): "
            f"{s['uncertainty_wrong']:.4f}"
        )

        print(
            f"U ratio: "
            f"{s['uncertainty_ratio']:.2f}x"
        )

        print(
            f"Errors Conf>=.95: "
            f"{s['fraction_errors_conf095']:.4f}"
        )

        print(
            f"Errors Conf>=.99: "
            f"{s['fraction_errors_conf099']:.4f}"
        )

    print(
        "\n[OVERALL]"
    )

    print(
        f"mIoU: "
        f"{overall_summary['miou']:.2f}"
    )

    print(
        f"Mean confidence: "
        f"{overall_summary['mean_confidence']:.4f}"
    )

    print(
        f"Pixel error: "
        f"{overall_summary['pixel_error_rate']:.4f}"
    )

    print(
        f"U(correct): "
        f"{overall_summary['uncertainty_correct']:.4f}"
    )

    print(
        f"U(wrong): "
        f"{overall_summary['uncertainty_wrong']:.4f}"
    )

    print(
        f"U ratio: "
        f"{overall_summary['uncertainty_ratio']:.2f}x"
    )

    print(
        f"Errors Conf>=.95: "
        f"{overall_summary['fraction_errors_conf095']:.4f}"
    )

    print(
        f"Errors Conf>=.99: "
        f"{overall_summary['fraction_errors_conf099']:.4f}"
    )

    print(
        "\nSaved to:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()