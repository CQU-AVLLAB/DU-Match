import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model.semseg.dpt import DPT


# =========================
# 配置
# =========================

DATA_ROOT = "/home/chenyang/datasets/pascal"

CHECKPOINT = (
    "checkpoints/"
    "pascal_unimatch_v2_1464labels_dinov2_base.pth"
)

VAL_LIST = "splits/pascal/val.txt"

OUTPUT_DIR = "visual_results"

# 先生成 20 张看看效果
# 如果想全部 1449 张，改成 None
NUM_IMAGES = 20


# =========================
# Pascal VOC 21 类
# =========================

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# Pascal VOC 官方颜色
PALETTE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


# DINOv2 / ImageNet normalization
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def colorize_mask(mask):
    """
    [H, W] class index -> RGB color image
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(PALETTE)):
        color[mask == class_id] = PALETTE[class_id]

    # VOC ignore label
    color[mask == 255] = [255, 255, 255]

    return color


def make_overlay(image, color_mask, alpha=0.55):
    """
    原图 + 彩色预测 mask 半透明叠加
    """
    image = np.asarray(image).astype(np.float32)
    color_mask = color_mask.astype(np.float32)

    overlay = (
        image * (1.0 - alpha)
        + color_mask * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def calculate_iou(pred, target, nclass=21):
    """
    单张图计算 intersection / union
    """
    valid = target != 255

    pred = pred[valid]
    target = target[valid]

    intersection = np.zeros(nclass, dtype=np.float64)
    union = np.zeros(nclass, dtype=np.float64)

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


def build_model():
    print("Building DINOv2-Base + DPT...")

    model = DPT(
        encoder_size="base",
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
    )

    print("Loading checkpoint:")
    print(CHECKPOINT)

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    # 官方下载的 checkpoint 是纯 state_dict
    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.cuda()
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


@torch.inference_mode()
def predict(model, image):
    """
    完全按照官方 Pascal evaluation 思路：
    1. normalize
    2. 调整到 14 倍数
    3. model inference
    4. resize 回原始尺寸
    """

    original_w, original_h = image.size

    tensor = NORMALIZE(image).unsqueeze(0).cuda()

    # 官方 evaluate:
    # int(size / 14 + 0.5) * 14
    new_h = int(original_h / 14 + 0.5) * 14
    new_w = int(original_w / 14 + 0.5) * 14

    tensor = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=True,
    )

    logits = model(tensor)

    logits = F.interpolate(
        logits,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=True,
    )

    pred = logits.argmax(dim=1)[0]

    return pred.cpu().numpy().astype(np.uint8)


def main():

    os.makedirs(
        os.path.join(OUTPUT_DIR, "comparison"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, "prediction"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, "overlay"),
        exist_ok=True
    )

    model = build_model()

    with open(VAL_LIST, "r") as f:
        ids = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    print(f"\nValidation images: {len(ids)}")

    if NUM_IMAGES is not None:
        ids_to_visualize = ids[:NUM_IMAGES]
    else:
        ids_to_visualize = ids

    total_intersection = np.zeros(
        21,
        dtype=np.float64
    )

    total_union = np.zeros(
        21,
        dtype=np.float64
    )

    for index, item in enumerate(ids_to_visualize):

        image_rel, mask_rel = item.split()

        image_path = os.path.join(
            DATA_ROOT,
            image_rel
        )

        mask_path = os.path.join(
            DATA_ROOT,
            mask_rel
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        gt = np.array(
            Image.open(mask_path)
        )

        pred = predict(
            model,
            image
        )

        intersection, union = calculate_iou(
            pred,
            gt,
            21
        )

        total_intersection += intersection
        total_union += union

        gt_color = colorize_mask(gt)
        pred_color = colorize_mask(pred)

        overlay = make_overlay(
            image,
            pred_color,
            alpha=0.55
        )

        filename = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        # -------------------------
        # 保存预测 mask
        # -------------------------

        Image.fromarray(pred_color).save(
            os.path.join(
                OUTPUT_DIR,
                "prediction",
                f"{filename}.png"
            )
        )

        # -------------------------
        # 保存 overlay
        # -------------------------

        Image.fromarray(overlay).save(
            os.path.join(
                OUTPUT_DIR,
                "overlay",
                f"{filename}.png"
            )
        )

        # -------------------------
        # 四联图
        # -------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 6)
        )

        axes[0].imshow(image)
        axes[0].set_title(
            "Original Image",
            fontsize=14
        )

        axes[1].imshow(gt_color)
        axes[1].set_title(
            "Ground Truth",
            fontsize=14
        )

        axes[2].imshow(pred_color)
        axes[2].set_title(
            "Ours Prediction",
            fontsize=14
        )

        axes[3].imshow(overlay)
        axes[3].set_title(
            "Ours Overlay",
            fontsize=14
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            OUTPUT_DIR,
            "comparison",
            f"{filename}.png"
        )

        plt.savefig(
            save_path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"[{index + 1}/{len(ids_to_visualize)}] "
            f"{filename} -> {save_path}"
        )

    # -------------------------
    # 当前可视化样本 mIoU
    # -------------------------

    valid_classes = total_union > 0

    ious = np.zeros(
        21,
        dtype=np.float64
    )

    ious[valid_classes] = (
        total_intersection[valid_classes]
        / total_union[valid_classes]
        * 100
    )

    print("\n==============================")
    print("IoU on visualized images")
    print("==============================")

    for i in range(21):
        if total_union[i] > 0:
            print(
                f"{i:02d} "
                f"{CLASSES[i]:12s}: "
                f"{ious[i]:6.2f}"
            )

    miou = ious[valid_classes].mean()

    print("------------------------------")
    print(
        f"mIoU ({len(ids_to_visualize)} images): "
        f"{miou:.2f}"
    )

    print("\nVisualization saved to:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )


if __name__ == "__main__":
    main()