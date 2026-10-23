"""Reproducible four-state diagnosis using the SAVED online prototype bank.

Run from DU-Match. Validation labels only evaluate/illustrate predictions.
No prototype updates, synthetic scatter clouds, or hand-entered statistics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import torch
from model.semseg.dpt import DPT
from util.du_match import PrototypeBank
from make_motivation_figure import load_pair, records, colorize

NAMES = ['Reliable', 'Ambiguous', 'Suspicious', 'Uncertain']
KEYS = ['reliable', 'ambiguous', 'suspicious', 'uncertain']
COLORS = ['#278c51', '#397fca', '#dc4b4b', '#e69b35']
SYMBOLS = ['R', 'A', 'S', 'U']


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='exp/cityscapes_integration_20261016/latest.pth')
    p.add_argument('--data-root', default='/home/chenyang/datasets/cityscapes')
    p.add_argument('--val-list', default='splits/cityscapes/val.txt')
    p.add_argument('--output', default='analysis_four_groups')
    p.add_argument('--max-images', type=int, default=0, help='0 = full validation list')
    p.add_argument('--max-side', type=int, default=1022)
    p.add_argument('--scatter-points', type=int, default=24000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--font-dir', default='/mnt/c/Windows/Fonts')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--refresh-full-examples', action='store_true',
        help='With --plot-only, recompute only the selected full scenes; preserve aggregate statistics')
    a = p.parse_args()
    if a.max_images < 0 or a.max_side < 28 or a.scatter_points < 1:
        p.error('Invalid image limit, scale, or scatter size')
    return a


def setup_font(folder):
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        font_manager.fontManager.addfont(str(Path(folder)/name))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':13,
        'axes.labelsize':13, 'axes.titlesize':14, 'xtick.labelsize':11,
        'ytick.labelsize':11, 'pdf.fonttype':42, 'ps.fonttype':42,
        'mathtext.fontset':'custom', 'mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic', 'mathtext.bf':'Times New Roman:bold',
        'mathtext.cal':'Times New Roman', 'mathtext.sf':'Times New Roman',
        'mathtext.tt':'Times New Roman', 'mathtext.fallback':None})


def pick_crop(mask, valid):
    h,w = mask.shape
    size = min(h, max(84, round(h*.55)))
    step = max(1, size//4)
    best = (0,0,0,size)
    for y in sorted(set(range(0,h-size+1,step))|{h-size}):
        for x in sorted(set(range(0,w-size+1,step))|{w-size}):
            area = np.s_[y:y+size,x:x+size]
            if valid[area].mean() < .8:
                continue
            count = int(mask[area].sum())
            if count > best[0]:
                best = count,y,x,size
    return best


@torch.inference_mode()
def collect(a, out):
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if 'prototype_bank' not in ckpt:
        raise ValueError('Use a DU-Match checkpoint containing its ONLINE prototype bank')
    if ckpt.get('smoke_only', False):
        raise ValueError('Use the epoch checkpoint latest.pth, not smoke.pth')
    options = ckpt['du_match_options']
    state = {k.removeprefix('module.'):v for k,v in ckpt['model_ema'].items()}
    nclass = state['head.scratch.output_conv.2.weight'].shape[0]
    if nclass != 19:
        raise ValueError('This script is for Cityscapes (19 train IDs)')
    model = DPT(encoder_size='base', nclass=19, features=128, out_channels=[96,192,384,768])
    model.load_state_dict(state, strict=True)
    from model.backbone.dinov2_layers.attention import Attention
    for layer in model.modules():
        if isinstance(layer, Attention): layer.use_torch_sdpa = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    bank = PrototypeBank(19,model.backbone.embed_dim,
        variance_floor=options['variance_floor'], min_updates=options['min_updates'],
        rows=options['chunk_rows']).to(device)
    bank.load_state_dict(ckpt['prototype_bank'])
    meta = dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_epoch=ckpt.get('epoch'),
        optimization_steps=ckpt['prototype_steps'], diagnostic_only=ckpt.get('diagnostic_only',False),
        prototype_source='Saved online bank; labeled student features only', query_model='EMA teacher',
        max_side=a.max_side, inference_precision='FP32', options=options,
        val_list_sha256=hashlib.sha256(Path(a.val_list).read_bytes()).hexdigest(),
        seed=a.seed, scatter_sampling='Uniform random-priority reservoir over all eligible pixels',
        case_selection='Window maximizing group pixel count; GT correctness is NOT a selection criterion',
        note='Fixed-scale weak-view diagnosis, not official sliding-window mIoU; nearest-neighbor GT alignment. No validation labels update the bank.')
    del ckpt,state
    rows = records(a.val_list,'val')
    if a.max_images: rows=rows[:a.max_images]
    meta['images']=len(rows)
    stats={k:dict(pixels=0,errors=0,confidence_sum=0.) for k in KEYS}
    rng=np.random.default_rng(a.seed)
    sample=np.empty((0,5),dtype=np.float64) # random priority, H, Ufeat, state, GT error
    cases={}
    valid_count=eligible_count=fallback_count=0
    per_image=[]
    for index,row in enumerate(rows):
        x,y,rgb=load_pair(Path(a.data_root),row,a.max_side,device)
        logits,feat=model(x,return_features=True)
        prob=logits.float().softmax(1)
        target=bank.targets(prob,feat,torch.where(y==255,255,0),options)
        gt=y[0].cpu().numpy()
        pred=target['hard'][0].cpu().numpy()
        confidence=prob.max(1).values[0].cpu().numpy()
        hp=target['upred'][0].cpu().numpy()
        hf=target['ufeat'][0].cpu().numpy()
        valid=gt!=255
        error=pred!=gt
        labels=np.full(gt.shape,-1,dtype=np.int8)
        item=dict(image=row[0],valid=int(valid.sum()))
        valid_count+=item['valid']
        for k,key in enumerate(KEYS):
            mask=target[key][0].cpu().numpy()
            labels[mask]=k
            n=int(mask.sum()); e=int((mask&error).sum())
            stats[key]['pixels']+=n; stats[key]['errors']+=e
            stats[key]['confidence_sum']+=float(confidence[mask].sum(dtype=np.float64))
            item[key+'_pixels']=n; item[key+'_errors']=e
            crop=pick_crop(mask,valid)
            if crop[0] and (key not in cases or crop[0]>cases[key]['count']):
                count,top,left,size=crop
                area=np.s_[top:top+size,left:left+size]
                cases[key]=dict(count=count,image=row[0],box=[top,left,size],
                    rgb=rgb[area].copy(),gt=gt[area].copy(),pred=pred[area].copy(),
                    confidence=confidence[area].copy(),upred=hp[area].copy(),ufeat=hf[area].copy(),mask=mask[area].copy())
                np.savez_compressed(out/f'full_case_{key}.npz',image=row[0],
                    rgb=rgb,gt=gt,pred=pred,confidence=confidence,upred=hp,ufeat=hf,mask=mask,
                    feature_valid=valid & ~target['fallback'][0].cpu().numpy())
        eligible=labels>=0
        fallback_count+=int((valid&~eligible).sum())
        eligible_count+=int(eligible.sum())
        n=int(eligible.sum())
        if n:
            priority=rng.random(n)
            keep=np.argpartition(priority,min(a.scatter_points,n)-1)[:a.scatter_points]
            local=np.column_stack([priority[keep],hp[eligible][keep],hf[eligible][keep],labels[eligible][keep],error[eligible][keep]])
            sample=np.concatenate([sample,local])
            if len(sample)>a.scatter_points:
                sample=sample[np.argpartition(sample[:,0],a.scatter_points-1)[:a.scatter_points]]
        per_image.append(item)
        if index%25==0 or index+1==len(rows):
            print(f'Four-state analysis: {index+1}/{len(rows)}',flush=True)
    if not eligible_count: raise ValueError('No pixels with available predicted-class prototypes')
    assert sum(v['pixels'] for v in stats.values())==eligible_count
    assert eligible_count+fallback_count==valid_count
    for key,v in stats.items():
        n=v['pixels']
        v.update(ratio_percent=100*n/eligible_count,error_percent=100*v['errors']/n if n else None,
            confidence_percent=100*v['confidence_sum']/n if n else None)
    for key in ('reliable','suspicious'):
        if stats[key]['pixels']:
            assert stats[key]['confidence_percent']>=100*options['confidence_threshold']-1e-4
    meta.update(valid_pixels=valid_count,eligible_pixels=eligible_count,unavailable_pixels=fallback_count,
        ready_classes=int(bank.ready.sum()),groups=stats,
        cases={k:dict(image=v['image'],box=v['box'],group_pixels=v['count']) for k,v in cases.items()},
        evaluated_images=[r[0] for r in rows])
    (out/'statistics.json').write_text(json.dumps(meta,indent=2))
    with (out/'group_statistics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['group','pixels','errors','pixel_ratio_percent','GT_error_percent','mean_confidence_percent'])
        for key,v in stats.items(): w.writerow([key,v['pixels'],v['errors'],v['ratio_percent'],v['error_percent'],v['confidence_percent']])
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_image[0]));w.writeheader();w.writerows(per_image)
    np.savez_compressed(out/'scatter_samples.npz',sample=sample)
    for key,c in cases.items(): np.savez_compressed(out/f'case_{key}.npz',**c)
    return meta,sample,cases


@torch.inference_mode()
def restore_full_examples(a,out,meta):
    """Re-evaluate only selected source images, without recomputing statistics."""
    checkpoint=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
    options=checkpoint['du_match_options']
    assert checkpoint['prototype_steps']==meta['optimization_steps']
    model=DPT(encoder_size='base',nclass=19,features=128,out_channels=[96,192,384,768])
    model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['model_ema'].items()},strict=True)
    from model.backbone.dinov2_layers.attention import Attention
    for layer in model.modules():
        if isinstance(layer,Attention):layer.use_torch_sdpa=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    bank=PrototypeBank(19,model.backbone.embed_dim,variance_floor=options['variance_floor'],
        min_updates=options['min_updates'],rows=options['chunk_rows']).to(device)
    bank.load_state_dict(checkpoint['prototype_bank'])
    del checkpoint
    rows={r[0]:r for r in records(a.val_list,'val')}
    for key,info in meta['cases'].items():
        x,y,rgb=load_pair(Path(a.data_root),rows[info['image']],meta['max_side'],device)
        logits,features=model(x,return_features=True)
        p=logits.float().softmax(1)
        t=bank.targets(p,features,torch.where(y==255,255,0),options)
        np.savez_compressed(out/f'full_case_{key}.npz',image=info['image'],rgb=rgb,
            gt=y[0].cpu().numpy(),pred=t['hard'][0].cpu().numpy(),
            confidence=p.max(1).values[0].cpu().numpy(),upred=t['upred'][0].cpu().numpy(),
            ufeat=t['ufeat'][0].cpu().numpy(),mask=t[key][0].cpu().numpy(),
            feature_valid=(t['valid']&~t['fallback'])[0].cpu().numpy())
        print(f'Full-scene case restored: {key}',flush=True)


def stat_panels(fig,grid,meta,sample):
    left=fig.add_subplot(grid[0])
    opt=meta['options']
    for k,name in enumerate(NAMES):
        points=sample[sample[:,3]==k]
        left.scatter(points[:,1],points[:,2],s=3,c=COLORS[k],alpha=.28,linewidths=0,rasterized=True)
    # Entropy alone is not the full prediction-side gate. A vertical line
    # would falsely suggest that all four states are rectangular quadrants.
    left.axhline(opt['feature_threshold'],color='#666666',ls='--',lw=1)
    left.set(xlim=(0,1),ylim=(0,1),xlabel=r'Prediction uncertainty $U^{pred}$',ylabel=r'Feature incompatibility $U^{feat}$')
    left.set_title('(a) Dual-space pixel distribution',pad=12,fontweight='bold')
    handles=[Line2D([],[],marker='o',ls='',color=c,label=n,markersize=6) for c,n in zip(COLORS,NAMES)]
    left.legend(handles=handles,loc='upper right',framealpha=.95,fontsize=11)
    left.text(.985,opt['feature_threshold']-.018,
        rf'$\tau_f = {opt["feature_threshold"]:g}$',ha='right',va='top',fontsize=11,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    gate_note=('Color: complete CEM assignment\n'
        +rf'$G=1$ if $C\geq {opt["confidence_threshold"]:g}$ and $U^{{pred}}<{opt["prediction_threshold"]:g}$')
    left.text(.02,.02,gate_note,
        transform=left.transAxes,fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    right=grid[1].subgridspec(2,1,height_ratios=[1,1.05],hspace=.6)
    ax=fig.add_subplot(right[0]);ax.set_axis_off()
    ax.set_title('(b) Statistics of the four pixel groups',pad=12,fontweight='bold')
    rows=[]
    for name,key in zip(NAMES,KEYS):
        v=meta['groups'][key]
        rows.append([name,f'{v["ratio_percent"]:.2f}',f'{v["error_percent"]:.2f}' if v['error_percent'] is not None else '--',
            f'{v["confidence_percent"]:.2f}' if v['confidence_percent'] is not None else '--'])
    table=ax.table(cellText=rows,colLabels=['Group','Pixel ratio\n(%)','GT error\n(%)','Mean conf.\n(%)'],
        colWidths=[.31,.23,.23,.23],cellLoc='center',bbox=[0,0,1,.96])
    table.auto_set_font_size(False);table.set_fontsize(12)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#b4bdc7');cell.set_linewidth(.6)
        if r==0:cell.set_facecolor('#e8eef6');cell.set_text_props(weight='bold')
        elif c==0:cell.set_facecolor(matplotlib.colors.to_rgba(COLORS[r-1],.14));cell.set_text_props(color=COLORS[r-1],weight='bold')
    ax=fig.add_subplot(right[1])
    x=np.arange(4)
    error=[meta['groups'][k]['error_percent'] if meta['groups'][k]['error_percent'] is not None else np.nan for k in KEYS]
    conf=[meta['groups'][k]['confidence_percent'] if meta['groups'][k]['confidence_percent'] is not None else np.nan for k in KEYS]
    bars1=ax.bar(x-.19,error,.36,color='#e16767',label='GT error rate')
    bars2=ax.bar(x+.19,conf,.36,color='#6397d5',label='Mean confidence')
    for bars in (bars1,bars2):
        for bar in bars:
            if np.isfinite(bar.get_height()): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{bar.get_height():.1f}',ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=NAMES,ylim=(0,140),yticks=[0,25,50,75,100],ylabel='Percentage (%)')
    ax.set_title('(c) Error rate and prediction confidence',pad=12,fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=10)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)


def maps(c):
    tint=c['rgb'].copy().astype(np.float32)
    tint[c['mask']]=.72*tint[c['mask']]+.28*np.array([255,255,255])
    return [(tint.astype(np.uint8),'Image + mask',None),(colorize(c['pred']),'Prediction',None),
        (colorize(c['gt']),'Ground truth',None),(c['confidence'],'Confidence','viridis'),
        (c['upred'],r'$U^{pred}$','viridis'),(c['ufeat'],r'$U^{feat}$','viridis')]


def full_maps(c,color):
    overlay=c['rgb'].copy().astype(np.float32)
    overlay[c['mask']]=.82*overlay[c['mask']]+.18*np.array(matplotlib.colors.to_rgb(color))*255
    return [(overlay.astype(np.uint8),'Image + group',None),(colorize(c['pred']),'Prediction',None),
        (colorize(c['gt']),'Ground truth',None),(c['confidence'],'Confidence','viridis'),
        (c['upred'],r'$U^{pred}$','viridis'),(c['ufeat'],r'$U^{feat}$','viridis')]


def examples_cards(fig,grid,cases):
    for j,(key,name,color) in enumerate(zip(KEYS,NAMES,COLORS)):
        card=grid[j].subgridspec(3,3,height_ratios=[.26,1,1],wspace=.08,hspace=.36)
        title=fig.add_subplot(card[0,:]);title.set_axis_off()
        title.text(.5,.5,name,ha='center',va='center',color=color,fontweight='bold',fontsize=15,
            bbox=dict(boxstyle='round,pad=.25',facecolor=matplotlib.colors.to_rgba(color,.12),edgecolor=color,linewidth=.6))
        if key not in cases:
            title.text(.5,-1,'No pixels in this group',ha='center');continue
        c=cases[key]
        for index,(im,label,cmap) in enumerate(maps(c)):
            ax=fig.add_subplot(card[1+index//3,index%3])
            if cmap:
                cm=plt.get_cmap(cmap).copy();cm.set_bad('#eeeeee')
                ax.imshow(np.ma.masked_where(~c['mask'],im),cmap=cm,vmin=0,vmax=1)
            else:
                ax.imshow(im)
                if index==0 and c['mask'].any() and not c['mask'].all():
                    ax.contour(c['mask'],levels=[.5],colors=[color],linewidths=.65)
            ax.set_title(label,fontsize=10,pad=4);ax.set_axis_off()


def export(fig,out,stem):
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(out/f'{stem}.png',dpi=240,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def draw(a,out,meta,sample,cases,full_cases):
    setup_font(a.font_dir)
    fig=plt.figure(figsize=(16,10.6))
    g=fig.add_gridspec(2,1,height_ratios=[1.5,1],hspace=.30,left=.06,right=.98,top=.88,bottom=.08)
    stat_panels(fig,g[0].subgridspec(1,2,width_ratios=[1,1.05],wspace=.24),meta,sample)
    examples_cards(fig,g[1].subgridspec(1,4,wspace=.16),cases)
    fig.text(.5,.98,'Four-state uncertainty analysis and qualitative examples',ha='center',fontsize=19,weight='bold')
    fig.text(.5,.035,'Heatmaps show the indicated group only (gray: other pixels); all heatmap values use a fixed scale from 0 to 1.',ha='center',fontsize=11)
    cbax=fig.add_axes([.39,.012,.22,.012]);fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='viridis'),cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    export(fig,out,'fig_four_groups')
    fig=plt.figure(figsize=(10,5.2))
    g=fig.add_gridspec(1,2,width_ratios=[1,1.05],left=.075,right=.98,bottom=.16,top=.87,wspace=.25)
    stat_panels(fig,g,meta,sample)
    export(fig,out,'fig_four_groups_statistics')
    fig=plt.figure(figsize=(14,6.4))
    grid=fig.add_gridspec(4,7,width_ratios=[.52,1,1,1,1,1,1],left=.015,right=.985,top=.87,bottom=.15,wspace=.09,hspace=.16)
    headers=['Group','Image + group','Prediction','Ground truth','Confidence',r'$U^{pred}$',r'$U^{feat}$']
    for j,(key,name,color) in enumerate(zip(KEYS,NAMES,COLORS)):
        ax=fig.add_subplot(grid[j,0]);ax.set_axis_off();ax.text(.5,.5,name,rotation=90,ha='center',va='center',color=color,fontsize=15,weight='bold')
        if key not in full_cases: continue
        c=full_cases[key]
        for index,(im,label,cmap) in enumerate(full_maps(c,color)):
            ax=fig.add_subplot(grid[j,index+1])
            if cmap:
                cm=plt.get_cmap(cmap).copy();cm.set_bad('#eeeeee')
                valid=c.get('feature_valid',c['gt']!=255) if index==5 else c['gt']!=255
                ax.imshow(np.ma.masked_where(~valid,im),cmap=cm,vmin=0,vmax=1)
            else:
                ax.imshow(im)
                if index==0 and c['mask'].any() and not c['mask'].all(): ax.contour(c['mask'],levels=[.5],colors=[color],linewidths=.4)
            if j==0: ax.set_title(headers[index+1],fontsize=13,pad=9)
            ax.set_axis_off()
    fig.text(.5,.98,'Full-scene examples of the four supervision states',ha='center',fontsize=17,weight='bold')
    cbax=fig.add_axes([.38,.065,.26,.018]);fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='viridis'),cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    fig.text(.5,.015,'The image overlay identifies the target group. Heatmaps show the full scene; gray denotes ignored or unavailable pixels. Shared scale: 0–1.',ha='center',fontsize=10)
    export(fig,out,'fig_four_groups_examples')


def main():
    a=args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.plot_only:
        meta=json.loads((out/'statistics.json').read_text())
        with np.load(out/'scatter_samples.npz') as z: sample=z['sample']
        cases={}
        for key in KEYS:
            if (out/f'case_{key}.npz').exists():
                with np.load(out/f'case_{key}.npz') as z:cases[key]=dict(z)
    else:
        meta,sample,cases=collect(a,out)
    if a.refresh_full_examples:
        if not a.plot_only: raise ValueError('--refresh-full-examples requires --plot-only')
        restore_full_examples(a,out,meta)
    full_cases={}
    for key in KEYS:
        path=out/f'full_case_{key}.npz'
        if path.exists():
            with np.load(path) as z:full_cases[key]=dict(z)
        elif key in cases:
            raise ValueError('Full-scene cache missing: use --plot-only --refresh-full-examples once')
    draw(a,out,meta,sample,cases,full_cases)
    print(json.dumps(meta['groups'],indent=2),flush=True)
    print(f'Figures saved to {out.resolve()}',flush=True)


if __name__=='__main__':main()
