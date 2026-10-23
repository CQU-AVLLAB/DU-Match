import argparse
from copy import deepcopy
import logging
import os
import pprint
import json
import time
import statistics

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.du_match import PrototypeBank, consistency_loss, cutmix_targets, baseline_loss, supervised_loss


parser = argparse.ArgumentParser(description='DU-Match on UniMatch V2')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--validation-id-path', default=None, help='Optional diagnostic validation list; checkpoints are marked diagnostic-only')
parser.add_argument('--stop-after-epoch', default=0, type=int, help='Stop after saving this absolute epoch count; 0 means no early stop')
parser.add_argument('--cuda-memory-fraction', default=0.0, type=float, help='Optional process allocator cap in (0,1]')
parser.add_argument('--torch-sdpa', action='store_true', help='Use native memory-efficient scaled dot product attention')
parser.add_argument('--benchmark-memory-fraction', default=0.0, type=float, help='Benchmark only: cap allocator fraction, fail early instead of using shared memory')
parser.add_argument('--amp-bf16', action='store_true', help='Opt-in BF16 autocast for forwards; prototype statistics and losses remain FP32')
parser.add_argument('--benchmark-steps', default=0, type=int, help='Timing only: stop without validation or saving model weights')
parser.add_argument('--benchmark-discard', default=5, type=int, help='Discard startup steps from timing')
parser.add_argument('--smoke-steps', default=0, type=int, help='Stop after N training steps, save smoke.pth, skip validation')


def main():
    args = parser.parse_args()

    if args.benchmark_steps and (args.smoke_steps or args.benchmark_steps <= args.benchmark_discard):
        raise ValueError('Benchmark must exceed discard steps and cannot combine with smoke mode')
    if args.benchmark_steps and os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        raise ValueError('Benchmark requires a new output directory')
    benchmark_records = []
    if args.cuda_memory_fraction:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError('cuda-memory-fraction must be in (0,1]')
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    if args.benchmark_memory_fraction:
        if not args.benchmark_steps or not 0 < args.benchmark_memory_fraction <= 1:
            raise ValueError('Allocator cap is benchmark-only and must be in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(args.benchmark_memory_fraction)
    if args.amp_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16')

    def forward_network(network, inputs, **kwargs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp_bf16):
            return network(inputs, **kwargs)

    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)
    options = dict(prototype_momentum=0.99, variance_momentum=0.99, variance_floor=0.01,
                   min_updates=1, chunk_rows=16, warmup_iters=1000,
                   confidence_threshold=cfg['conf_thresh'], prediction_threshold=0.2,
                   feature_threshold=0.7, prototype_temperature=0.5,
                   gamma_min=0.0, gamma_max=0.8, lambda_ambiguous=1.0,
                   lambda_suspicious=1.0, lambda_u=1.0, alpha=1.0, beta=1.0,
                   prototype_source='student')
    options.update(cfg.get('du_match', {}))
    if options['prototype_source'] not in ('student', 'teacher'):
        raise ValueError('prototype_source must be student or teacher')
    if not 0 <= options['gamma_min'] <= options['gamma_max'] <= 1:
        raise ValueError('Require 0 <= gamma_min <= gamma_max <= 1')
    for name in ('confidence_threshold', 'prediction_threshold', 'feature_threshold'):
        if not 0 <= options[name] <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    if options['prototype_temperature'] <= 0 or options['warmup_iters'] < 0:
        raise ValueError('Invalid temperature or warmup')
    if any(options[name] < 0 for name in ('lambda_ambiguous', 'lambda_suspicious', 'lambda_u', 'alpha', 'beta')):
        raise ValueError('Loss weights and reliability exponents must be nonnegative')

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'du_match': options, 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    if args.torch_sdpa:
        from model.backbone.dinov2_layers.attention import Attention
        for layer in model.modules():
            if isinstance(layer, Attention):
                layer.use_torch_sdpa = True
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    bank = PrototypeBank(cfg['nclass'], model.module.backbone.embed_dim,
                         momentum=options['prototype_momentum'], variance_momentum=options['variance_momentum'],
                         variance_floor=options['variance_floor'], min_updates=options['min_updates'],
                         rows=options['chunk_rows']).cuda(local_rank)
    prototype_steps = 0

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    if args.validation_id_path:
        with open(args.validation_id_path) as stream:
            valset.ids = [line.strip() for line in stream if line.strip()]
        if not valset.ids:
            raise ValueError('Empty diagnostic validation list')

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        if checkpoint.get('diagnostic_only', False) and not cfg.get('diagnostic_only', False):
            raise ValueError('Diagnostic checkpoint cannot resume as a formal experiment')
        expected_runtime = dict(torch_sdpa=args.torch_sdpa, amp_bf16=args.amp_bf16)
        if checkpoint.get('runtime_options', expected_runtime) != expected_runtime:
            raise ValueError('Resume with the same attention and precision flags')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        if 'prototype_bank' not in checkpoint or 'du_match_options' not in checkpoint:
            raise ValueError('Use a fresh DU-Match save directory; baseline checkpoints have no prototype bank')
        if checkpoint['du_match_options'] != options:
            raise ValueError('DU-Match options differ from the checkpoint; use a new save directory for an ablation')
        bank.load_state_dict(checkpoint['prototype_bank'])
        prototype_steps = checkpoint['prototype_steps']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        if args.benchmark_steps:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            benchmark_previous = time.perf_counter()
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            iters = epoch * len(trainloader_u) + i
            pred_x, feature_x = forward_network(model, img_x, return_features=True)
            with torch.no_grad():
                if options['prototype_source'] == 'teacher':
                    _, feature_l = forward_network(model_ema, img_x, return_features=True)
                else:
                    feature_l = feature_x.detach()
                bank.update(feature_l, mask_x)
                prototype_steps += 1
                pred_u_w, feature_u_w = forward_network(model_ema, img_u_w, return_features=True)
                prob_u_w = pred_u_w.float().softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)
                active = prototype_steps > options['warmup_iters']
                targets = bank.targets(prob_u_w, feature_u_w, ignore_mask, options) if active else None
            del feature_x, feature_l, feature_u_w, pred_u_w, prob_u_w

            img_u_s1 = torch.where(cutmix_box1.unsqueeze(1).bool(), img_u_s1.flip(0), img_u_s1)
            img_u_s2 = torch.where(cutmix_box2.unsqueeze(1).bool(), img_u_s2.flip(0), img_u_s2)
            pred_u_s1, pred_u_s2 = forward_network(model, torch.cat((img_u_s1, img_u_s2)), comp_drop=True).float().chunk(2)
            valid_labeled_pixels = mask_x.ne(cfg['criterion']['kwargs'].get('ignore_index', 255)).sum()
            loss_x = supervised_loss(pred_x.float(), mask_x, criterion_l,
                                     cfg['criterion']['kwargs'].get('ignore_index', 255))
            if rank == 0 and valid_labeled_pixels.item() == 0:
                logger.warning('EMPTY_LABELED_CROP iteration=%d: zero supervised loss; retain unlabeled training', iters)
            if active:
                target1, target2 = cutmix_targets(targets, cutmix_box1), cutmix_targets(targets, cutmix_box2)
                loss_u_s1 = consistency_loss(pred_u_s1, target1, options['lambda_ambiguous'], options['lambda_suspicious'])
                loss_u_s2 = consistency_loss(pred_u_s2, target2, options['lambda_ambiguous'], options['lambda_suspicious'])
                supervised_pixels = targets['hard_mask'] | targets['ambiguous'] | targets['suspicious']
                mask_ratio = supervised_pixels.sum().float() / (ignore_mask != 255).sum().clamp_min(1)
            else:
                baseline_targets = dict(hard=mask_u_w, confidence=conf_u_w, ignore=ignore_mask)
                target1, target2 = cutmix_targets(baseline_targets, cutmix_box1), cutmix_targets(baseline_targets, cutmix_box2)
                loss_u_s1 = baseline_loss(pred_u_s1, target1['hard'], target1['confidence'], target1['ignore'], cfg['conf_thresh'])
                loss_u_s2 = baseline_loss(pred_u_s2, target2['hard'], target2['confidence'], target2['ignore'], cfg['conf_thresh'])
                mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().float() / (ignore_mask != 255).sum().clamp_min(1)
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            # Warmup exactly follows the baseline loss scale; the active phase follows the paper.
            loss = loss_x + options['lambda_u'] * loss_u_s if active else (loss_x + loss_u_s) / 2.0

            optimizer.zero_grad()
            if not torch.isfinite(loss):
                diagnostic = dict(iteration=iters, epoch=epoch, epoch_step=i,
                    valid_labeled_pixels=int(valid_labeled_pixels),
                    loss_x=float(loss_x.detach()), loss_u_s1=float(loss_u_s1.detach()),
                    loss_u_s2=float(loss_u_s2.detach()),
                    logits_x_finite=bool(torch.isfinite(pred_x).all()),
                    logits_s1_finite=bool(torch.isfinite(pred_u_s1).all()),
                    logits_s2_finite=bool(torch.isfinite(pred_u_s2).all()))
                with open(os.path.join(args.save_path, 'nonfinite_diagnostic.json'), 'w') as stream:
                    json.dump(diagnostic, stream, indent=2)
                torch.save(dict(img_x=img_x.cpu(), mask_x=mask_x.cpu(),
                                img_u_s1=img_u_s1.cpu(), img_u_s2=img_u_s2.cpu(),
                                diagnostics=diagnostic), os.path.join(args.save_path, 'nonfinite_batch.pth'))
                logger.error('Nonfinite diagnostics: %s', diagnostic)
                raise FloatingPointError(f'Nonfinite loss at iteration {iters}')
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                if buffer_ema.is_floating_point():
                    buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                else:
                    buffer_ema.copy_(buffer)
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('du_match/active', int(active), iters)
                writer.add_scalar('du_match/ready_classes', bank.ready.sum().item(), iters)
                if active:
                    valid_count = targets['valid'].sum().clamp_min(1)
                    for state in ('reliable', 'ambiguous', 'suspicious', 'uncertain', 'fallback'):
                        writer.add_scalar('du_match/' + state + '_ratio', targets[state].sum() / valid_count, iters)
                    for name in ('upred', 'ufeat', 'gamma', 'reliability'):
                        writer.add_scalar('du_match/' + name, (targets[name] * targets['valid']).sum() / valid_count, iters)
            del targets, target1, target2

            if (i % (max(1, len(trainloader_u) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
            if args.benchmark_steps:
                torch.cuda.synchronize()
                benchmark_now = time.perf_counter()
                benchmark_records.append(dict(step=iters + 1, active=bool(active),
                    seconds=benchmark_now - benchmark_previous, loss=loss.item(),
                    selected_ratio=float(mask_ratio), ready_classes=int(bank.ready.sum())))
                if rank == 0 and (iters + 1) % 5 == 0:
                    logger.info('BENCHMARK step=%d active=%s seconds=%.3f peak_GiB=%.2f',
                        iters + 1, active, benchmark_records[-1]['seconds'],
                        torch.cuda.max_memory_allocated() / 2**30)
                if iters + 1 >= args.benchmark_steps:
                    if rank == 0:
                        measured = benchmark_records[args.benchmark_discard:]
                        phases = {}
                        for phase, phase_active in [('warmup_with_bank', False), ('du_match_active', True)]:
                            values = [r['seconds'] for r in measured if r['active'] == phase_active]
                            if values:
                                phases[phase] = dict(steps=len(values), mean_seconds=statistics.mean(values),
                                    median_seconds=statistics.median(values), min_seconds=min(values), max_seconds=max(values))
                        result = dict(timing_only=True, gpu=torch.cuda.get_device_name(),
                            torch_version=torch.__version__, cuda_version=torch.version.cuda,
                            precision='BF16 forwards; FP32 losses/statistics' if args.amp_bf16 else 'float32; no AMP', config=cfg, du_match_options=options,
                            torch_sdpa=args.torch_sdpa, allocator_memory_fraction=args.benchmark_memory_fraction,
                            steps=args.benchmark_steps, discarded=args.benchmark_discard,
                            steps_per_epoch=len(trainloader_u), planned_training_steps=total_iters,
                            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                            peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                            phases=phases, records=benchmark_records,
                            note='Includes data wait, transfers, forward/backward, optimizer, EMA and logging. No validation or model save. Short test is not steady-state thermal performance or scientific training.')
                        with open(os.path.join(args.save_path, 'benchmark.json'), 'w') as stream:
                            json.dump(result, stream, indent=2)
                        logger.info('BENCHMARK RESULT: %s', json.dumps({k:v for k,v in result.items() if k not in ('config','du_match_options','records')}, indent=2))
                        writer.close()
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                    return
                benchmark_previous = time.perf_counter()

            if args.smoke_steps > 0 and iters + 1 >= args.smoke_steps:
                if rank == 0:
                    torch.save(dict(model=model.state_dict(), model_ema=model_ema.state_dict(),
                                    optimizer=optimizer.state_dict(), prototype_bank=bank.state_dict(),
                                    prototype_steps=prototype_steps, du_match_options=options,
                                    smoke_only=True, iteration=iters), os.path.join(args.save_path, 'smoke.pth'))
                    logger.info('Smoke check complete: %s steps, active=%s, finite loss=%s',
                                iters + 1, active, torch.isfinite(loss).item())
                    writer.close()
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
                return

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = mIoU >= previous_best
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'prototype_bank': bank.state_dict(),
                'prototype_steps': prototype_steps,
                'du_match_options': options,
                'runtime_options': dict(torch_sdpa=args.torch_sdpa, amp_bf16=args.amp_bf16),
                'diagnostic_only': bool(cfg.get('diagnostic_only', False) or args.validation_id_path),
                'validation_id_path': args.validation_id_path
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            if rank == 0:
                logger.info('Requested stop after epoch %d: validation and checkpoint save complete', epoch + 1)
                writer.close()
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            return


if __name__ == '__main__':
    main()
