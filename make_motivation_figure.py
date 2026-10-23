"""Baseline-only Cityscapes diagnosis; run from the DU-Match repository.

Frozen-model, post-hoc prototypes use the supplied labeled TRAINING split.
Validation labels are used exclusively for evaluation and case selection.
This is a fixed-scale diagnosis, not official sliding-window mIoU evaluation.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from model.semseg.dpt import DPT
from util.du_match import PrototypeBank, aligned_strips

PALETTE = np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],
    [152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],
    [0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]], dtype=np.uint8)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/cityscapes_unimatch_v2_1_16labels_dinov2_base.pth')
    p.add_argument('--data-root', default='/home/chenyang/datasets/cityscapes')
    p.add_argument('--labeled-list', default='splits/cityscapes/1_16/labeled.txt')
    p.add_argument('--val-list', default='splits/cityscapes/val.txt')
    p.add_argument('--encoder', choices=['small','base','large'], default='base')
    p.add_argument('--state-key', default='model_ema', help='Used only for packaged checkpoints; strict, no silent fallback')
    p.add_argument('--output', default='analysis_motivation_baseline')
    p.add_argument('--max-side', type=int, default=1022)
    p.add_argument('--max-train', type=int, default=0, help='0 = entire supplied labeled split')
    p.add_argument('--max-val', type=int, default=0, help='0 = entire supplied validation split')
    p.add_argument('--confidence', type=float, default=.95)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--plot-only', action='store_true', help='Redraw saved statistics and cases without model inference')
    a = p.parse_args()
    if a.max_side < 28 or a.bins < 2 or not 0 < a.confidence < 1 or min(a.max_train, a.max_val) < 0:
        p.error('Invalid scale, bin count, threshold, or image limit')
    return a


def records(path, split):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(len(r) != 2 or split not in Path(r[0]).parts or split not in Path(r[1]).parts for r in rows):
        raise ValueError(f'{path}: expected image/label pairs from {split} only')
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate images in {path}')
    return rows


def load_pair(root, row, side, device):
    with Image.open(root / row[0]) as im:
        im = im.convert('RGB')
        w, h = im.size
        scale = min(1., side / max(w, h))
        size = (max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14))
        rgb = np.asarray(im.resize(size, Image.Resampling.BILINEAR)).copy()
    with Image.open(root / row[1]) as im:
        gt = np.asarray(im.resize(size, Image.Resampling.NEAREST), dtype=np.int64).copy()
    if not np.isin(gt, list(range(19)) + [255]).all():
        raise ValueError('Expected Cityscapes labelTrainIds (0..18 or 255)')
    x = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).to(device) / 255
    x = (x - x.new_tensor([.485,.456,.406])[None,:,None,None]) / x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x, torch.from_numpy(gt).unsqueeze(0).to(device), rgb


def model_from_checkpoint(a):
    settings = {'small': (64,[48,96,192,384]), 'base': (128,[96,192,384,768]),
                'large': (256,[256,512,1024,1024])}
    width, channels = settings[a.encoder]
    model = DPT(encoder_size=a.encoder, nclass=19, features=width, out_channels=channels)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('smoke_only', False):
        raise ValueError('Smoke checkpoints cannot support a paper motivation figure')
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state, source = ckpt, 'plain state_dict; student/teacher provenance unavailable'
    else:
        state, source = ckpt[a.state_key], a.state_key
    model.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()}, strict=True)
    return model.to(a.device).eval(), source


@torch.inference_mode()
def build_bank(model, rows, a):
    """Pixel-weighted global moments, NOT reconstructed training-time EMA."""
    bank = PrototypeBank(19, model.backbone.embed_dim).to(a.device)
    sums = torch.zeros_like(bank.centers, dtype=torch.float64)
    squares = torch.zeros(19, device=a.device, dtype=torch.float64)
    counts = torch.zeros(19, device=a.device, dtype=torch.long)
    for j, row in enumerate(rows):
        x, y, _ = load_pair(Path(a.data_root), row, a.max_side, a.device)
        _, feat = model(x, return_features=True)
        for start, end, z in aligned_strips(feat, y.shape[-2:]):
            lab = y[:,start:end].reshape(-1)
            flat = z.permute(0,2,3,1).reshape(-1,z.shape[1]).double()
            valid = (lab >= 0) & (lab < 19)
            lab, flat = lab[valid], flat[valid]
            sums.index_add_(0, lab, flat)
            squares.index_add_(0, lab, flat.square().sum(1))
            counts += torch.bincount(lab, minlength=19)
        if j % 20 == 0 or j + 1 == len(rows):
            print(f'Prototype training images: {j+1}/{len(rows)}', flush=True)
    mean = sums / counts.clamp_min(1)[:,None]
    bank.centers.copy_(F.normalize(mean,dim=1).float())
    bank.variance.copy_((squares/counts.clamp_min(1)-mean.square().sum(1)).clamp_min(0).float())
    bank.updates.copy_((counts > 0).long())
    bank.pixel_counts.copy_(counts)
    return bank


def histogram(values, errors, edges):
    return np.histogram(values,bins=edges)[0], np.histogram(values,bins=edges,weights=errors.astype(np.int64))[0]


def choose_crop(mask, valid):
    """Fixed-size window with most high-confidence errors; no feature-score selection."""
    h,w = mask.shape
    size = min(h, max(56, round(h*.38)))
    best = None
    for top in range(0,h-size+1,max(1,size//4)):
        for left in range(0,w-size+1,max(1,size//4)):
            area = np.s_[top:top+size,left:left+size]
            if valid[area].mean() < .8:
                continue
            score = int(mask[area].sum())
            if best is None or score > best[0]:
                best = (score,top,left,size)
    return best or (0,0,0,min(h,w))


def colorize(y):
    out = np.full((*y.shape,3),245,dtype=np.uint8)
    good = y < 19
    out[good] = PALETTE[y[good]]
    return out


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#9aa6b5')
    ax.grid(axis='y', color='#e8edf3', linewidth=.7)
    ax.set_axisbelow(True)


def draw(out, sample, stats, a, rank):
    font_dir = Path('/mnt/c/Windows/Fonts') if sys.platform != 'win32' else Path('C:/Windows/Fonts')
    for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):
        font_manager.fontManager.addfont(str(font_dir/name))
    resolved = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    print(f'Figure font: {resolved}',flush=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.titlesize':12,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.linewidth':.8,
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    plt.rcParams.update({'mathtext.cal':'Times New Roman','mathtext.sf':'Times New Roman',
        'mathtext.tt':'Times New Roman','mathtext.fallback':None})
    fig = plt.figure(figsize=(12.8,6.6),facecolor='white')
    grid = fig.add_gridspec(2,2,left=.065,right=.975,top=.94,bottom=.08,
        height_ratios=[2.7,1.35],wspace=.28,hspace=.67)
    _,top,left,size = sample['crop']
    area = np.s_[top:top+size,left:left+size]
    rgb,gt,pred,conf,uf = [sample[k][area] for k in ('rgb','gt','pred','confidence','ufeat')]
    wrong = (pred!=gt)&(gt!=255)&(conf>=a.confidence)
    overlay = (rgb*.5+255*.5).astype(np.uint8)
    overlay[wrong] = [226,67,66]
    local = grid[1,:].subgridspec(1,7,width_ratios=[1.8,1,1,1,1,1,1],wspace=.17)
    overview = fig.add_subplot(local[0])
    overview.imshow(sample['rgb'])
    overview.add_patch(plt.Rectangle((left,top),size,size,fill=False,edgecolor='#d53838',linewidth=1.2))
    overview.set_axis_off(); overview.set_title('Input image',pad=7,size=11)
    views = [(rgb,'Image crop',None),(colorize(gt),'Ground truth',None),
             (colorize(pred),'Prediction',None),(conf,'Confidence','viridis'),
             (uf,'Incompatibility','magma'),(overlay,'Confident errors',None)]
    for k,(im,title,cmap) in enumerate(views):
        ax = fig.add_subplot(local[k+1])
        shown = ax.imshow(im,cmap=cmap,**({'vmin':0,'vmax':1} if cmap else {}))
        ax.set_title(title,pad=4,size=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if cmap:
            cb = fig.colorbar(shown,ax=ax,location='bottom',fraction=.06,pad=.10,ticks=[0,.5,1],aspect=20)
            cb.ax.tick_params(labelsize=9,length=2,pad=2)
    n,e = stats['confidence']
    edges = np.asarray(stats['edges']['confidence'])
    centers = (edges[:-1]+edges[1:])/2
    ax = fig.add_subplot(grid[0,0]); style(ax)
    rate = np.divide(100*e,n,out=np.full(len(n),np.nan),where=n>0)
    colors = ['#e77d73' if lo>=a.confidence-1e-8 else '#8fb9e7' for lo in edges[:-1]]
    ax.bar(centers,rate,width=np.diff(edges)*.9,color=colors,edgecolor='white')
    ax.axvspan(a.confidence,1,color='#e65050',alpha=.12)
    ax.axvline(a.confidence,color='#d34c50',ls='--',lw=1.3)
    ax.set(xlim=(0,1),xlabel='Prediction confidence',ylabel='Error rate (%)',title='(a) Errors retained by confidence filtering')
    hc,he = stats['high_count'],stats['high_errors']
    text = f'$C \\geq {a.confidence:g}$: {100*he/hc:.2f}% error rate' if hc else 'No high-confidence pixels'
    ax.text(.98,.95,text,transform=ax.transAxes,va='top',ha='right',size=11)
    right = grid[0,1].subgridspec(2,1,height_ratios=[4,1],hspace=.12)
    ax = fig.add_subplot(right[0]); style(ax)
    centers = (np.arange(a.bins)+.5)/a.bins
    n,e = stats['feature']
    rate = np.divide(100*e,n,out=np.full(a.bins,np.nan),where=n>0)
    ax.plot(centers,rate,'o-',color='#dc7148',lw=2,ms=5)
    ax.set(xlim=(0,1),ylabel='Error rate (%)',title='(b) Feature incompatibility within confident pixels')
    ax.tick_params(labelbottom=False)
    ax.text(.03,.95,f'$C \\geq {a.confidence:g}$',transform=ax.transAxes,va='top',size=11)
    support = fig.add_subplot(right[1]); style(support)
    support.bar(centers,n/n.sum()*100 if n.sum() else n,width=.08,color='#d7dfe9')
    support.set(xlim=(0,1),xlabel=r'Feature incompatibility $U^{feat}$',ylabel='Pixels (%)')
    support.tick_params(labelsize=8)
    support.yaxis.label.set_size(8)
    fig.text(.5,.335,'(c) An illustrative high-confidence failure of the baseline',size=12,ha='center')
    for ext in ('png','pdf'):
        fig.savefig(out/f'motivation_case_{rank}.{ext}',dpi=240)
    plt.close(fig)


def redraw(a):
    out = Path(a.output)
    meta = json.loads((out/'summary.json').read_text())
    saved = argparse.Namespace(**meta['arguments'])
    stats = dict(meta['counts']); stats['edges'] = {}
    for key in ('confidence','feature'):
        with (out/f'{key}_bins.csv').open() as f:
            rows = list(csv.DictReader(f))
        n = np.array([int(r['pixels']) for r in rows],dtype=np.int64)
        e = np.array([int(r['errors']) for r in rows],dtype=np.int64)
        edges = np.array([float(rows[0]['lower'])]+[float(r['upper']) for r in rows])
        # Early diagnostic runs used uniform bins; split the last bin exactly
        # using the separately accumulated >= threshold counts.
        if key=='confidence' and not np.isclose(edges,saved.confidence).any():
            k = np.searchsorted(edges,saved.confidence)-1
            if k != len(n)-1:
                raise ValueError('Cannot recover threshold bin; rerun inference')
            n = np.r_[n[:k],n[k]-stats['high_count'],stats['high_count']]
            e = np.r_[e[:k],e[k]-stats['high_errors'],stats['high_errors']]
            edges = np.insert(edges,k+1,saved.confidence)
        stats[key] = [n,e]; stats['edges'][key] = edges.tolist()
    for rank in range(1,4):
        path = out/f'case_{rank}.npz'
        if path.exists():
            with np.load(path) as loaded:
                sample = dict(loaded)
            draw(out,sample,stats,saved,rank)


@torch.inference_mode()
def main():
    a = arguments()
    if a.plot_only:
        redraw(a)
        return
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    train, val = records(a.labeled_list,'train'), records(a.val_list,'val')
    if {r[0] for r in train} & {r[0] for r in val}:
        raise ValueError('Training/validation overlap')
    train = train[:a.max_train] if a.max_train else train
    val = val[:a.max_val] if a.max_val else val
    model, source = model_from_checkpoint(a)
    bank = build_bank(model,train,a)
    torch.save(bank.state_dict(),out/'posthoc_prototypes.pth')
    opts = dict(confidence_threshold=a.confidence,prediction_threshold=.2,feature_threshold=.7,
        prototype_temperature=.5,gamma_min=0.,gamma_max=.8,alpha=1.,beta=1.)
    edges = dict(confidence=np.unique(np.append(np.linspace(0,1,a.bins+1),a.confidence)),feature=np.linspace(0,1,a.bins+1))
    stats = {k:[np.zeros(len(edges[k])-1,dtype=np.int64),np.zeros(len(edges[k])-1,dtype=np.int64)] for k in edges}
    stats['edges'] = {k:v.tolist() for k,v in edges.items()}
    stats.update(images=len(val),valid=0,high_count=0,high_errors=0,high_unavailable=0)
    cases, image_rows = [], []
    for j,row in enumerate(val):
        x,y,rgb = load_pair(Path(a.data_root),row,a.max_side,a.device)
        logits,feat = model(x,return_features=True)
        prob = logits.softmax(1)
        target = bank.targets(prob,feat,torch.where(y==255,255,0),opts)
        gt = y[0].cpu().numpy()
        pred = target['hard'][0].cpu().numpy()
        conf = prob.max(1).values[0].cpu().numpy()
        uf = target['ufeat'][0].cpu().numpy()
        valid,err = gt!=255,pred!=gt
        high = valid & (conf>=a.confidence)
        available = bank.ready[target['hard']][0].cpu().numpy()
        selected = high & available
        for key,values,mask in [('confidence',conf,valid),('feature',uf,selected)]:
            n,e = histogram(values[mask],err[mask],edges[key])
            stats[key][0] += n; stats[key][1] += e
        stats['valid'] += int(valid.sum()); stats['high_count'] += int(high.sum())
        stats['high_errors'] += int((high&err).sum())
        stats['high_unavailable'] += int((high&~available).sum())
        crop = choose_crop(high&err,valid)
        image_rows.append(dict(image=row[0],valid=int(valid.sum()),high=int(high.sum()),high_errors=int((high&err).sum()),crop_errors=crop[0]))
        if crop[0] and (len(cases)<3 or crop[0]>cases[-1]['crop'][0]):
            cases.append(dict(rgb=rgb,gt=gt,pred=pred,confidence=conf,ufeat=uf,crop=crop,image=row[0]))
            cases = sorted(cases,key=lambda c:c['crop'][0],reverse=True)[:3]
        if j%20==0 or j+1==len(val): print(f'Validation images: {j+1}/{len(val)}',flush=True)
    assert stats['confidence'][0].sum() == stats['valid']
    assert stats['feature'][0].sum() == stats['high_count'] - stats['high_unavailable']
    for key in ('confidence','feature'):
        with (out/f'{key}_bins.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['lower','upper','pixels','errors','error_rate'])
            for i,(n,e) in enumerate(zip(*stats[key])):
                writer.writerow([edges[key][i],edges[key][i+1],int(n),int(e),float(e/n) if n else ''])
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    metadata = dict(arguments=vars(a),checkpoint_source=source,
        prototype_estimator='Post-hoc global labeled-training pixel moments; frozen same model; NOT online EMA',
        train_images=len(train),prototype_pixels=bank.pixel_counts.cpu().tolist(),
        labeled_list_sha256=hashlib.sha256(Path(a.labeled_list).read_bytes()).hexdigest(),
        validation_list_sha256=hashlib.sha256(Path(a.val_list).read_bytes()).hexdigest(),
        evaluated_validation_images=[r[0] for r in val],
        counts={k:v for k,v in stats.items() if isinstance(v,int)},bin_edges=stats['edges'],
        selected_cases=[dict(image=c['image'],crop=list(c['crop'])) for c in cases],
        diagnostic_only=True,partial_run=len(train)<len(records(a.labeled_list,'train')) or len(val)<len(records(a.val_list,'val')),
        note='GT resized with nearest neighbor to inference grid. No validation GT updates prototypes. Bins are descriptive pixel aggregates, not independent-sample inference. Checkpoint training-split provenance must be independently verified.')
    (out/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for rank,c in enumerate(cases,1):
        np.savez_compressed(out/f'case_{rank}.npz',**{k:v for k,v in c.items() if k!='image'},image=c['image'])
        draw(out,c,stats,a,rank)
    print(json.dumps(metadata['counts'],indent=2),flush=True)
    print(f'Figures and reproducible counts saved to {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
