"""Validation-only diagnostics; ground truth NEVER updates the prototype bank."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import yaml
from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from util.du_match import PrototypeBank


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-images', type=int, default=100, help='0 means the complete validation set')
    parser.add_argument('--max-side', type=int, default=518, help='Diagnostic inference scale, not official mIoU evaluation')
    args = parser.parse_args()
    if args.max_images < 0 or args.max_side < 14:
        raise ValueError('Invalid image limit or inference size')
    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'prototype_bank' not in checkpoint:
        raise ValueError('Requires a DU-Match checkpoint containing prototype statistics')
    options = checkpoint['du_match_options']
    model_configs = dict(small=(64, [48, 96, 192, 384]), base=(128, [96, 192, 384, 768]),
                         large=(256, [256, 512, 1024, 1024]), giant=(384, [1536, 1536, 1536, 1536]))
    encoder = cfg['backbone'].split('_')[-1]
    features, channels = model_configs[encoder]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPT(encoder_size=encoder, nclass=cfg['nclass'], features=features, out_channels=channels)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model_ema'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    bank = PrototypeBank(cfg['nclass'], model.backbone.embed_dim, variance_floor=options['variance_floor'],
                         min_updates=options['min_updates'], rows=options['chunk_rows']).to(device)
    bank.load_state_dict(checkpoint['prototype_bank'])
    dataset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    states = {state: dict(pixels=0, errors=0) for state in ('reliable', 'ambiguous', 'suspicious', 'uncertain', 'fallback')}
    counts = dict(valid=0, high_confidence=0, high_confidence_errors=0,
                  certainty_gate_errors=0, suspicious_errors=0, corrected_errors=0, fixed=0, harmed=0)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    for index in range(limit):
        image, label, _ = dataset[index]
        image, label = image.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        height, width = image.shape[-2:]
        scale = min(1., args.max_side / max(height, width))
        size = (max(14, round(height * scale / 14) * 14), max(14, round(width * scale / 14) * 14))
        logits, representation = model(F.interpolate(image, size, mode='bilinear', align_corners=True), return_features=True)
        probabilities = F.interpolate(logits, (height, width), mode='bilinear', align_corners=True).softmax(1)
        ignore = torch.where((label >= 0) & (label < cfg['nclass']), 0, 255)
        targets = bank.targets(probabilities, representation, ignore, options)
        valid = targets['valid']
        errors = targets['hard'] != label
        corrected_errors = targets['soft'].argmax(1) != label
        confidence = probabilities.max(1).values
        high = valid & (confidence >= options['confidence_threshold'])
        gate = high & (targets['upred'] < options['prediction_threshold'])
        suspicious = targets['suspicious']
        for name in states:
            states[name]['pixels'] += targets[name].sum().item()
            states[name]['errors'] += (targets[name] & errors).sum().item()
        masks = dict(valid=valid, high_confidence=high, high_confidence_errors=high & errors,
                     certainty_gate_errors=gate & errors, suspicious_errors=suspicious & errors,
                     corrected_errors=suspicious & corrected_errors,
                     fixed=suspicious & errors & ~corrected_errors, harmed=suspicious & ~errors & corrected_errors)
        for name, mask in masks.items():
            counts[name] += mask.sum().item()
    for values in states.values():
        values['error_rate'] = ratio(values['errors'], values['pixels'])
    suspicious_count = states['suspicious']['pixels']
    output = dict(images=limit, checkpoint=args.checkpoint, smoke_only=checkpoint.get('smoke_only', False),
                  max_side=args.max_side, ready_classes=bank.ready.sum().item(), counts=counts, states=states,
                  suspicious_precision=ratio(counts['suspicious_errors'], suspicious_count),
                  high_confidence_error_recall=ratio(counts['suspicious_errors'], counts['high_confidence_errors']),
                  certainty_gate_error_recall=ratio(counts['suspicious_errors'], counts['certainty_gate_errors']),
                  suspicious_corrected_error_rate=ratio(counts['corrected_errors'], suspicious_count),
                  correction_net_gain=ratio(counts['fixed'] - counts['harmed'], suspicious_count),
                  options=options,
                  note='Validation-only diagnostic. Correction evaluated by target argmax. Not training results or official mIoU; null denotes empty denominator.')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()