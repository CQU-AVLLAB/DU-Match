"""Export real Cityscapes framework assets from one student/EMA checkpoint.

Run from the WSL DU-Match repository (see method_assets/README.md).
Full-scene views preserve context for a framework schematic; they are not
training-batch captures or quantitative evaluation results.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


PALETTE = np.array([
    [128,64,128], [244,35,232], [70,70,70], [102,102,156],
    [190,153,153], [153,153,153], [250,170,30], [220,220,0],
    [107,142,35], [152,251,152], [70,130,180], [220,20,60],
    [255,0,0], [0,0,142], [0,0,70], [0,60,100], [0,80,100],
    [0,0,230], [119,11,32],
], dtype=np.uint8)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def strong_view(image, seed):
    """Same photometric transform distributions/order as dataset/semi.py."""
    seed_all(seed)
    result = image.copy()
    jitter = random.random() < 0.8
    if jitter:
        result = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(result)
    gray = bool(torch.rand(1).item() < 0.2)
    if gray:
        result = transforms.functional.rgb_to_grayscale(result, num_output_channels=3)
    blurred = random.random() < 0.5
    sigma = float(np.random.uniform(0.1, 2.0)) if blurred else None
    if blurred:
        result = result.filter(ImageFilter.GaussianBlur(radius=sigma))
    return result, dict(seed=seed, color_jitter=jitter, grayscale=gray,
                        blur=blurred, blur_sigma=sigma)


def colorize(labels):
    labels = np.asarray(labels)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(PALETTE))
    rgb[valid] = PALETTE[labels[valid]]
    return Image.fromarray(rgb)


def source_row(repo, split, index):
    rows = (repo / 'splits/cityscapes/1_16' / (split + '.txt')).read_text().splitlines()
    return rows[index].split()


def overview(out, images):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager
    font_file = Path('/mnt/c/Windows/Fonts/times.ttf')
    if not font_file.is_file():
        raise FileNotFoundError('Times New Roman is required for the preview.')
    fontManager.addfont(str(font_file))
    plt.rcParams.update({'font.family': 'Times New Roman', 'svg.fonttype': 'none'})
    font = FontProperties(fname=str(font_file), size=19)
    fig = plt.figure(figsize=(15, 12), facecolor='white')
    grid = fig.add_gridspec(4, 6, hspace=0.28, wspace=0.16)
    panels = [
        (0, 0, 2, '01_labeled_input', 'Labeled input'),
        (0, 2, 4, '02_labeled_ground_truth', 'Ground truth'),
        (0, 4, 6, '03_labeled_student_prediction', 'Student prediction'),
        (1, 0, 2, '04_unlabeled_input', 'Unlabeled input'),
        (1, 2, 4, '05_unlabeled_weak_view', 'Weak view'),
        (1, 4, 6, '06_teacher_weak_prediction', 'EMA teacher prediction'),
        (2, 0, 3, '07_unlabeled_strong_view_1', 'Strong view 1'),
        (2, 3, 6, '08_unlabeled_strong_view_2', 'Strong view 2'),
        (3, 0, 3, '09_student_strong_prediction_1', 'Student prediction 1'),
        (3, 3, 6, '10_student_strong_prediction_2', 'Student prediction 2'),
    ]
    for row, left, right, key, title in panels:
        ax = fig.add_subplot(grid[row, left:right])
        ax.imshow(images[key], interpolation='nearest')
        ax.set_title(title, fontproperties=font, pad=7)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.025)
    fig.savefig(out / 'method_assets_overview.png', dpi=220, facecolor='white')
    fig.savefig(out / 'method_assets_overview.svg', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='/home/chenyang/DU-Match')
    parser.add_argument('--data-root', default='/home/chenyang/datasets/cityscapes')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--labeled-index', type=int, default=1)
    parser.add_argument('--unlabeled-index', type=int, default=1)
    parser.add_argument('--height', type=int, default=504)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    repo, root, out = Path(args.repo), Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_ids').mkdir(exist_ok=True)
    sys.path.insert(0, str(repo))
    from model.semseg.dpt import DPT
    from model.backbone.dinov2_layers.attention import Attention

    lrow = source_row(repo, 'labeled', args.labeled_index)
    urow = source_row(repo, 'unlabeled', args.unlabeled_index)
    assert lrow[0] != urow[0]
    original_l = Image.open(root / lrow[0]).convert('RGB')
    original_u = Image.open(root / urow[0]).convert('RGB')
    height = args.height
    if height % 14:
        raise ValueError('Height must be a multiple of the DINOv2 patch size, 14.')
    width = round(original_u.width / original_u.height * height / 14) * 14
    size = (width, height)
    labeled = original_l.resize(size, Image.Resampling.BILINEAR)
    weak = original_u.resize(size, Image.Resampling.BILINEAR)
    gt = Image.open(root / lrow[1]).resize(size, Image.Resampling.NEAREST)
    # The unlabeled GT file is deliberately never opened.
    s1, aug1 = strong_view(weak, 101)
    s2, aug2 = strong_view(weak, 202)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    device = torch.device(args.device)
    images = {'01_labeled_input': labeled, '02_labeled_ground_truth': colorize(gt),
              '04_unlabeled_input': weak.copy(), '05_unlabeled_weak_view': weak,
              '07_unlabeled_strong_view_1': s1, '08_unlabeled_strong_view_2': s2}
    checkpoint_path = Path(args.checkpoint)
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint.get('epoch', -1) + 1
    student_state = checkpoint['model']
    teacher_state = checkpoint['model_ema']
    checkpoint_keys = list(checkpoint)
    # Do not retain optimizer tensors during visualization.
    del checkpoint
    model = DPT(encoder_size='base', nclass=19, features=128,
                out_channels=[96, 192, 384, 768]).to(device).eval()
    for layer in model.modules():
        if isinstance(layer, Attention):
            layer.use_torch_sdpa = True

    def load_state(state):
        clean = {k.removeprefix('module.'): value for k, value in state.items()}
        model.load_state_dict(clean, strict=True)

    def save_prediction(key, logits):
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite visualization logits: ' + key)
        ids = logits.argmax(0).cpu().numpy().astype(np.uint8)
        images[key] = colorize(ids)
        Image.fromarray(ids).save(out / 'train_ids' / (key + '.png'))

    with torch.inference_mode():
        load_state(student_state)
        del student_state
        print('Student labeled forward', flush=True)
        save_prediction('03_labeled_student_prediction', model(normalize(labeled)[None].to(device))[0])
        seed_all(303)
        print('Student paired strong forwards with complementary dropout', flush=True)
        pair = torch.stack([normalize(s1), normalize(s2)]).to(device)
        outputs = model(pair, comp_drop=True)
        save_prediction('09_student_strong_prediction_1', outputs[0])
        save_prediction('10_student_strong_prediction_2', outputs[1])
        del outputs, pair
        load_state(teacher_state)
        del teacher_state
        print('EMA teacher weak forward', flush=True)
        save_prediction('06_teacher_weak_prediction', model(normalize(weak)[None].to(device))[0])

    for name, im in images.items():
        im.save(out / (name + '.png'))
    gt.save(out / 'train_ids/02_labeled_ground_truth.png')
    original_l.save(out / 'labeled_original_2048x1024.png')
    original_u.save(out / 'unlabeled_original_2048x1024.png')
    manifest = dict(
        checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=digest.hexdigest(),
        completed_epochs=epoch, checkpoint_keys=checkpoint_keys,
        backbone='DINOv2-Base + DPT, 19 classes', device=args.device, precision='FP32',
        labeled_image=lrow[0], labeled_ground_truth=lrow[1], unlabeled_image=urow[0],
        labeled_index=args.labeled_index, unlabeled_index=args.unlabeled_index,
        size=list(size), strong_view_1=aug1, strong_view_2=aug2, complementary_dropout_seed=303,
        palette='Official Cityscapes trainId colors, ignore pixels black',
        teacher='model_ema from the same checkpoint', student='model from the same checkpoint',
        view_protocol='Full scenes resized to a shared patch-aligned 2:1 view; no random spatial crop or flip.',
        augmentation_protocol='Independent training-distribution color jitter, grayscale and Gaussian blur. '
                              'No cross-image CutMix: the current training batch size is one, so the donor is itself.',
        prediction_protocol='Student strong pair uses comp_drop=True; all networks use eval mode; no optimizer updates.',
        interpretation='Framework illustration using real checkpoint predictions, not a captured training batch, '
                       'before/after learning comparison, benchmark evaluation, or final converged result.',
        unlabeled_gt_used=False,
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    overview(out, images)
    print(json.dumps(dict(output=str(out), completed_epochs=epoch, size=size, augmentation=[aug1, aug2])), flush=True)


if __name__ == '__main__':
    main()
