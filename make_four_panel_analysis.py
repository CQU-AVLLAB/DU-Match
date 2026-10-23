"""Paper figure from visualize_four_groups.py outputs; no invented data.

Run in the training environment (Matplotlib + PyTorch):
python make_four_panel_analysis.py --analysis analysis_four_groups_epoch061_20261020 --output output/pdf
Run visualize_four_groups.py first to collect predictions from a saved checkpoint.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from visualize_four_groups import setup_font
from make_motivation_figure import colorize

KEYS = ['reliable', 'ambiguous', 'suspicious', 'uncertain']
NAMES = ['Reliable', 'Ambiguous', 'Suspicious', 'Uncertain']
SYMS = ['R', 'A', 'S', 'U']
COLORS = ['#07964d', '#206bd5', '#e23e46', '#eb8c16']
DARK = ['#07552f', '#153d87', '#ad1726', '#9c4805']


def draw(analysis, output):
    setup_font('/mnt/c/Windows/Fonts')
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    meta = json.loads((analysis/'statistics.json').read_text())
    sample = np.load(analysis/'scatter_samples.npz')['sample']
    groups = [meta['groups'][k] for k in KEYS]
    assert abs(sum(g['ratio_percent'] for g in groups)-100) < 1e-6
    fig = plt.figure(figsize=(18, 12), facecolor='white')

    def panel(x, y, w, h, title, fs=15):
        fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.003,rounding_size=0.007',
            transform=fig.transFigure, facecolor='white',edgecolor='#a7c8ed',lw=.85,zorder=-10))
        fig.add_artist(Rectangle((x,y+h-.034),w,.034,transform=fig.transFigure,
            facecolor='#d2e3f9',edgecolor='none',zorder=-9))
        fig.text(x+.009,y+h-.017,title,fontsize=fs,weight='bold',va='center')

    fig.add_artist(FancyBboxPatch((.01,.95),.98,.04,boxstyle='round,pad=0.004,rounding_size=0.009',
        transform=fig.transFigure,facecolor='#c9ddf5',edgecolor='#a7c8ed',lw=.8))
    fig.text(.5,.97,'Pixel-wise Uncertainty Analysis and Confident Error Mining (CEM)',
        ha='center',va='center',weight='bold',fontsize=21)
    panel(.01,.395,.505,.54,'(a) Pixel distribution in the dual-space uncertainty plane')
    panel(.526,.635,.464,.30,'(b) Statistics of each pixel group',14.8)
    panel(.526,.395,.464,.228,'(c) GT error rate and mean confidence',14.8)
    panel(.01,.027,.98,.355,'(d) Qualitative examples of the four groups')

    ax=fig.add_axes([.073,.462,.425,.415])
    for i in range(4):
        p=sample[sample[:,3]==i]
        ax.scatter(p[:,1],p[:,2],s=3.3,c=COLORS[i],alpha=.38,lw=0,rasterized=True)
    opt=meta['options']
    ax.axhline(opt['feature_threshold'],ls='--',lw=.9,c='#5c6570')
    ax.text(.985,opt['feature_threshold']-.015,rf'$\tau_f={opt["feature_threshold"]:g}$',
        ha='right',va='top',fontsize=11,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
    ax.set(xlim=(0,1),ylim=(0,1),xlabel=r'Prediction uncertainty $U^{pred}$',
        ylabel=r'Feature incompatibility $U^{feat}$')
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=n+rf' ($\Omega_{{{s}}}$)',markersize=6)
        for c,n,s in zip(COLORS,NAMES,SYMS)],loc='upper right',fontsize=10.5,framealpha=.97)
    fig.text(.074,.407,'Colors follow the full confidence and compatibility gates.',fontsize=10.8,color='#374151')

    ax=fig.add_axes([.532,.671,.451,.216]);ax.axis('off')
    conditions=['Confident\nCompatible','Not confident\nCompatible','Confident\nIncompatible','Not confident\nIncompatible']
    rows=[[n+rf' ($\Omega_{{{s}}}$)',cond,f'{g["ratio_percent"]:.2f}',f'{g["error_percent"]:.2f}',f'{g["confidence_percent"]:.2f}']
        for n,s,cond,g in zip(NAMES,SYMS,conditions,groups)]
    tb=ax.table(cellText=rows,colLabels=['Group','Condition','Pixel ratio\n(%)','GT error\n(%)','Mean conf.\n(%)'],
        cellLoc='center',colWidths=[.21,.25,.18,.18,.18],bbox=[0,0,1,1])
    tb.auto_set_font_size(False);tb.set_fontsize(11.5)
    for (r,c),cell in tb.get_celld().items():
        cell.set_edgecolor('#aeb8c3');cell.set_linewidth(.65)
        if r==0:cell.set_facecolor('#edf1f5');cell.set_text_props(weight='bold')
        elif c==0:
            cell.set_facecolor(matplotlib.colors.to_rgba(COLORS[r-1],.19));cell.set_text_props(color=DARK[r-1],weight='bold')
        else:cell.set_facecolor('#ffffff' if r%2 else '#fafbfd')
    fig.text(.537,.646,rf'Confident: $C\geq{opt["confidence_threshold"]:g}$ and $U^{{pred}}<{opt["prediction_threshold"]:g}$; compatible: $U^{{feat}}<{opt["feature_threshold"]:g}$.',fontsize=10.5)

    ax=fig.add_axes([.57,.445,.403,.120])
    x=np.arange(4)
    for dx,field,c,label in [(-.18,'error_percent','#e34c54','GT error rate'),(.18,'confidence_percent','#377dd1','Mean confidence')]:
        bars=ax.bar(x+dx,[g[field] for g in groups],.33,color=c,label=label,zorder=3)
        for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+2.4,f'{b.get_height():.1f}',ha='center',fontsize=10)
    ax.set(ylim=(0,112),yticks=[0,25,50,75,100],xticks=x,xticklabels=NAMES,ylabel='Percentage (%)')
    ax.tick_params(labelsize=10.5);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e4e8ee',lw=.6,zorder=0)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=10.5)

    descriptions=['Confident and feature-compatible.', 'Low confidence; feature-compatible.',
                  'Confident; feature-incompatible.\nCandidate errors, not verified errors.',
                  'Low confidence; feature-incompatible.']
    cardw=.236
    for i,key in enumerate(KEYS):
        x0=.018+i*.2455;y0=.068;h=.268
        fig.add_artist(FancyBboxPatch((x0,y0),cardw,h,boxstyle='round,pad=0.003,rounding_size=.006',
            transform=fig.transFigure,facecolor=matplotlib.colors.to_rgba(COLORS[i],.065),edgecolor=COLORS[i],lw=.85,zorder=-8))
        fig.add_artist(Rectangle((x0,y0+h-.035),cardw,.035,transform=fig.transFigure,
            facecolor=matplotlib.colors.to_rgba(COLORS[i],.17),edgecolor='none',zorder=-7))
        fig.text(x0+cardw/2,y0+h-.0175,NAMES[i]+rf' ($\Omega_{{{SYMS[i]}}}$)',
            ha='center',va='center',color=DARK[i],weight='bold',fontsize=15.5)
        c=dict(np.load(analysis/f'full_case_{key}.npz'))
        rgb=c['rgb'];mask=c['mask'].astype(bool)
        ims=[rgb,colorize(c['pred']),colorize(c['gt']),c['confidence'],c['upred'],c['ufeat']]
        labels=['Image + group','Prediction','GT','Confidence',r'$U^{pred}$',r'$U^{feat}$']
        tw=(cardw-.019)/3
        for j,(im,label) in enumerate(zip(ims,labels)):
            left=x0+.005+(j%3)*(tw+.0045)
            bottom=.229 if j<3 else .142
            ia=fig.add_axes([left,bottom,tw,tw*.75])
            if j<3:
                ia.imshow(im)
                if j==0 and mask.any() and not mask.all():ia.contour(mask,levels=[.5],colors=[COLORS[i]],linewidths=.5)
            else:ia.imshow(im,cmap='jet',vmin=0,vmax=1,interpolation='nearest')
            ia.set_title(label,fontsize=11.3,weight='bold',pad=3)
            ia.set_xticks([]);ia.set_yticks([])
            for sp in ia.spines.values():sp.set_edgecolor('#c4cbd3');sp.set_linewidth(.4)
        fig.text(x0+cardw/2,.084,descriptions[i],ha='center',va='center',fontsize=10.7,color='#1f2937')
    cbax=fig.add_axes([.421,.041,.17,.009])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='jet'),
        cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    cbax.tick_params(labelsize=9,pad=1,length=2)
    fig.text(.405,.045,'Shared heatmap scale',ha='right',va='center',fontsize=10)
    output.mkdir(parents=True,exist_ok=True)
    stem=output/'DU_Match_Four_Panel_Real_Epoch061'
    for ext in ['pdf','svg','png']:fig.savefig(str(stem)+'.'+ext,dpi=260,facecolor='white')
    plt.close(fig)
    audit={'source_statistics':str(analysis/'statistics.json'),'checkpoint':meta['checkpoint'],
        'completed_epochs':meta['checkpoint_epoch']+1,'images':len(meta['evaluated_images']),
        'groups':meta['groups'],'plot_data':'Unmodified real EMA predictions; reservoir scatter sample.',
        'examples':'Whole selected scenes; group contour on RGB; unmasked fixed-scale heatmaps.',
        'selection':meta['case_selection'],'scale_note':meta['note']}
    (output/'four_panel_epoch061_provenance.json').write_text(json.dumps(audit,indent=2))
    print(stem,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('output/pdf'))
    a=p.parse_args();draw(a.analysis,a.output)

