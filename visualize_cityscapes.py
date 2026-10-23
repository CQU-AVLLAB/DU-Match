import os
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

DATA_ROOT = "/home/chenyang/datasets/cityscapes"

CHECKPOINT = (
    "checkpoints/"
    "cityscapes_unimatch_v2_1_16labels_dinov2_base.pth"
)

OUTPUT_DIR = "visual_results_cityscapes"

# 先看 20 张
# 全部 500 张改成 None
NUM_IMAGES = 20


# =========================
# Cityscapes 19 类
# trainId: 0~18
# =========================

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


# Cityscapes 官方 trainId 颜色
PALETTE = np.array([
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 30],    # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
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
    [H, W] trainId -> RGB
    """
    h, w = mask.shape

    color = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for class_id in range(len(PALETTE)):
        color[mask == class_id] = PALETTE[class_id]

    # ignore label
    color[mask == 255] = [255, 255, 255]

    return color


def make_overlay(image, color_mask, alpha=0.55):
    """
    原图 + 彩色预测 mask
    """
    image = np.asarray(image).astype(np.float32)
    color_mask = color_mask.astype(np.float32)

    overlay = (
        image * (1.0 - alpha)
        + color_mask * alpha
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


def calculate_iou(pred, target, nclass=19):
    """
    单张图累计 intersection / union
    """

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


def build_model():

    print("Building DINOv2-Base + DPT for Cityscapes...")

    model = DPT(
        encoder_size="base",
        nclass=19,
        features=128,
        out_channels=[96, 192, 384, 768],
    )

    print("Loading checkpoint:")
    print(CHECKPOINT)

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    # 兼容两种 checkpoint 格式
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

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

    original_w, original_h = image.size

    tensor = NORMALIZE(
        image
    ).unsqueeze(0).cuda()

    # DINOv2 patch size = 14
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

    logits = model(tensor)

    logits = F.interpolate(
        logits,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=True,
    )

    pred = logits.argmax(
        dim=1
    )[0]

    return pred.cpu().numpy().astype(
        np.uint8
    )


def get_cityscapes_val_files():

    image_root = os.path.join(
        DATA_ROOT,
        "leftImg8bit",
        "val"
    )

    files = []

    for city in sorted(
        os.listdir(image_root)
    ):

        city_dir = os.path.join(
            image_root,
            city
        )

        if not os.path.isdir(city_dir):
            continue

        for filename in sorted(
            os.listdir(city_dir)
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

            if os.path.exists(gt_path):

                files.append(
                    (
                        image_path,
                        gt_path
                    )
                )

            else:

                print(
                    f"[WARNING] GT not found: "
                    f"{gt_path}"
                )

    return files


def main():

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "comparison"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "prediction"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "overlay"
        ),
        exist_ok=True
    )

    model = build_model()

    files = get_cityscapes_val_files()

    print(
        f"\nCityscapes validation images: "
        f"{len(files)}"
    )

    if NUM_IMAGES is not None:
        files_to_visualize = files[
            :NUM_IMAGES
        ]
    else:
        files_to_visualize = files

    total_intersection = np.zeros(
        19,
        dtype=np.float64
    )

    total_union = np.zeros(
        19,
        dtype=np.float64
    )

    for index, (
        image_path,
        mask_path
    ) in enumerate(
        files_to_visualize
    ):

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
            19
        )

        total_intersection += intersection
        total_union += union

        gt_color = colorize_mask(
            gt
        )

        pred_color = colorize_mask(
            pred
        )

        overlay = make_overlay(
            image,
            pred_color,
            alpha=0.55
        )

        filename = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        # -------------------------
        # 预测 mask
        # -------------------------

        Image.fromarray(
            pred_color
        ).save(
            os.path.join(
                OUTPUT_DIR,
                "prediction",
                f"{filename}.png"
            )
        )

        # -------------------------
        # overlay
        # -------------------------

        Image.fromarray(
            overlay
        ).save(
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
            figsize=(22, 7)
        )

        axes[0].imshow(
            image
        )

        axes[0].set_title(
            "Original Image",
            fontsize=14
        )

        axes[1].imshow(
            gt_color
        )

        axes[1].set_title(
            "Ground Truth",
            fontsize=14
        )

        axes[2].imshow(
            pred_color
        )

        axes[2].set_title(
            "UniMatch V2 Prediction",
            fontsize=14
        )

        axes[3].imshow(
            overlay
        )

        axes[3].set_title(
            "Prediction Overlay",
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

        plt.close(
            fig
        )

        print(
            f"[{index + 1}/"
            f"{len(files_to_visualize)}] "
            f"{filename}"
        )

    # =========================
    # 当前样本 mIoU
    # =========================

    valid_classes = (
        total_union > 0
    )

    ious = np.zeros(
        19,
        dtype=np.float64
    )

    ious[valid_classes] = (
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

    print("\n==============================")
    print("IoU on visualized images")
    print("==============================")

    for i in range(19):

        if total_union[i] > 0:

            print(
                f"{i:02d} "
                f"{CLASSES[i]:15s}: "
                f"{ious[i]:6.2f}"
            )

    miou = ious[
        valid_classes
    ].mean()

    print("------------------------------")

    print(
        f"mIoU "
        f"({len(files_to_visualize)} images): "
        f"{miou:.2f}"
    )

    print("\nVisualization saved to:")

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )


if __name__ == "__main__":
    main()