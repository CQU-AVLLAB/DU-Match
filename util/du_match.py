"""Dual-space targets; prototype statistics contain labeled pixels only."""
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def aligned_strips(features, size, rows=16):
    """Bilinear align_corners=True, normalize AFTER alignment, bound peak memory."""
    height, width = size
    xs = torch.linspace(-1, 1, width, device=features.device) if width > 1 else features.new_zeros(1)
    ys = torch.linspace(-1, 1, height, device=features.device) if height > 1 else features.new_zeros(1)
    for start in range(0, height, rows):
        end = min(start + rows, height)
        gy, gx = torch.meshgrid(ys[start:end], xs, indexing='ij')
        grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(features.shape[0], -1, -1, -1)
        z = F.grid_sample(features.float(), grid, mode='bilinear', padding_mode='border', align_corners=True)
        yield start, end, F.normalize(z, dim=1, eps=1e-8)


def entropy(prob):
    return (-(prob * prob.clamp_min(1e-8).log()).sum(1) / math.log(prob.shape[1])).clamp(0, 1)


class PrototypeBank(nn.Module):
    def __init__(self, nclass, dimension, momentum=0.99, variance_momentum=0.99,
                 variance_floor=0.01, min_updates=1, rows=16):
        super().__init__()
        if nclass < 2 or dimension < 1 or variance_floor <= 0 or min_updates < 1 or rows < 1:
            raise ValueError('Invalid prototype bank parameters')
        if not 0 <= momentum < 1 or not 0 <= variance_momentum < 1:
            raise ValueError('EMA momenta must be in [0, 1)')
        self.nclass, self.rows = nclass, rows
        self.momentum, self.variance_momentum = momentum, variance_momentum
        self.variance_floor, self.min_updates = variance_floor, min_updates
        self.register_buffer('centers', torch.zeros(nclass, dimension))
        self.register_buffer('variance', torch.ones(nclass))
        self.register_buffer('updates', torch.zeros(nclass, dtype=torch.long))
        self.register_buffer('pixel_counts', torch.zeros(nclass, dtype=torch.long))

    @property
    def ready(self):
        return self.updates >= self.min_updates

    @torch.no_grad()
    def update(self, features, labels):
        sums = torch.zeros_like(self.centers)
        squared = torch.zeros_like(self.variance)
        counts = torch.zeros_like(self.pixel_counts)
        for start, end, z in aligned_strips(features.detach(), labels.shape[-2:], self.rows):
            lab = labels[:, start:end].reshape(-1)
            flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
            valid = (lab >= 0) & (lab < self.nclass)
            lab, flat = lab[valid], flat[valid]
            sums.index_add_(0, lab, flat)
            squared.index_add_(0, lab, flat.square().sum(1))
            counts.add_(torch.bincount(lab, minlength=self.nclass))
        if dist.is_available() and dist.is_initialized():
            # Aggregate sufficient statistics BEFORE computing global centers/dispersion.
            for value in (sums, squared, counts):
                dist.all_reduce(value)
        present = counts > 0
        mean = sums / counts.clamp_min(1).unsqueeze(1)
        batch_var = (squared / counts.clamp_min(1) - mean.square().sum(1)).clamp_min(0)
        first = present & (self.updates == 0)
        existing = present & ~first
        self.centers[first] = F.normalize(mean[first], dim=1, eps=1e-8)
        self.variance[first] = batch_var[first]
        self.centers[existing] = F.normalize(
            self.momentum * self.centers[existing] + (1 - self.momentum) * mean[existing], dim=1, eps=1e-8)
        self.variance[existing] = (self.variance_momentum * self.variance[existing]
                                   + (1 - self.variance_momentum) * batch_var[existing])
        self.updates[present] += 1
        self.pixel_counts += counts

    @torch.no_grad()
    def targets(self, probabilities, features, ignore_mask, options):
        p = probabilities.detach().float()
        confidence, hard = p.max(1)
        upred = entropy(p)
        ufeat = torch.ones_like(upred)
        qproto = torch.zeros_like(p)
        ready = self.ready
        if ready.any():
            for start, end, z in aligned_strips(features.detach(), p.shape[-2:], self.rows):
                # Squared distance remains correct even for a zero feature/center vector.
                distance = (z.square().sum(1, keepdim=True)
                            + self.centers.square().sum(1)[None, :, None, None]
                            - 2 * torch.einsum('bdhw,cd->bchw', z, self.centers)).clamp_min(0)
                delta = distance / (2 * self.variance.clamp_min(self.variance_floor)[None, :, None, None] + 1e-8)
                predicted_delta = delta.gather(1, hard[:, start:end].unsqueeze(1)).squeeze(1)
                ufeat[:, start:end] = -torch.expm1(-predicted_delta)
                logits = (-delta / options['prototype_temperature']).masked_fill(~ready[None, :, None, None], -torch.inf)
                qproto[:, :, start:end] = logits.softmax(1)
        valid = ignore_mask != 255
        available = ready[hard]
        g = (confidence >= options['confidence_threshold']) & (upred < options['prediction_threshold'])
        f = ufeat < options['feature_threshold']
        # Missing predicted class -> baseline filtering; never treat an absent class as an error.
        fallback = valid & ~available
        reliable = valid & available & g & f
        ambiguous = valid & available & ~g & f
        suspicious = valid & available & g & ~f
        uncertain = valid & available & ~g & ~f
        hard_mask = reliable | (fallback & (confidence >= options['confidence_threshold']))
        # Entropy confidence requires at least two observed classes to avoid a spurious one-class certainty.
        ready_count = int(ready.sum())
        proto_conf = (1 - entropy(qproto) * math.log(self.nclass) / math.log(ready_count)).clamp(0, 1) if ready_count >= 2 else torch.zeros_like(upred)
        gamma = (ufeat * proto_conf).clamp(options['gamma_min'], options['gamma_max'])
        gamma = gamma * available * (ready.sum() >= 2)
        soft = torch.where(suspicious.unsqueeze(1),
                           (1 - gamma.unsqueeze(1)) * p + gamma.unsqueeze(1) * qproto, p)
        return dict(hard=hard, soft=soft, hard_mask=hard_mask, ambiguous=ambiguous,
                    suspicious=suspicious, uncertain=uncertain, valid=valid,
                    reliable=reliable, fallback=fallback, upred=upred, ufeat=ufeat,
                    gamma=gamma, reliability=(1 - upred) ** options['alpha'] * (1 - ufeat) ** options['beta'])


def cutmix_targets(targets, box):
    """Use exactly the reverse-batch donor used by the baseline image CutMix."""
    return {key: torch.where(box.unsqueeze(1).bool() if value.ndim == 4 else box.bool(),
                             value.flip(0), value) for key, value in targets.items()}


def consistency_loss(logits, targets, lambda_ambiguous=1.0, lambda_suspicious=1.0):
    hard = F.cross_entropy(logits, targets['hard'], reduction='none')
    soft = F.kl_div(F.log_softmax(logits.float(), 1), targets['soft'], reduction='none').sum(1)
    selected = targets['hard_mask'] | targets['ambiguous'] | targets['suspicious']
    numerator = (hard * targets['hard_mask'] + lambda_ambiguous * soft * targets['ambiguous']
                 + lambda_suspicious * soft * targets['suspicious']).sum()
    return numerator / selected.sum().clamp_min(1)


def baseline_loss(logits, hard, confidence, ignore, threshold):
    valid = ignore != 255
    return (F.cross_entropy(logits, hard, reduction='none')
            * valid * (confidence >= threshold)).sum() / valid.sum().clamp_min(1)

def supervised_loss(logits, labels, criterion, ignore_index=255):
    """An all-ignore crop has zero supervision, rather than an undefined mean."""
    if not labels.ne(ignore_index).any():
        return logits.sum() * 0.0
    return criterion(logits, labels)
