"""Compact 4 x 6 qualitative figure from archived real EMA predictions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from visualize_four_groups import setup_font
from make_motivation_figure import colorize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    setup_font('/mnt/c/Windows/Fonts')
    plt.rcParams.update({'svg.fonttype':'none'})
    meta=json.loads((a.analysis/'statistics.json').read_text())
    names=['Reliable','Ambiguous','Suspicious','Uncertain']
    colors=['#07964d','#206bd5','#e23e46','#eb8c16']
    titles=['Image + group','Prediction','Ground truth','Confidence',r'$U^{pred}$',r'$U^{feat}$']
    fig=plt.figure(figsize=(18,7.4),facecolor='white')
    left=.065; gap=.007; w=(.99-left-5*gap)/6
    h=w*18/7.4/2
    sources=[]
    for r,(name,color) in enumerate(zip(names,colors)):
        c=dict(np.load(a.analysis/f'full_case_{name.lower()}.npz'))
        rgb=c['rgb'];mask=c['mask'].astype(bool)
        assert mask.shape==c['confidence'].shape==c['pred'].shape
        sources.append({'group':name,'image':str(c['image'])})
        bottom=.918-h-r*(h+.023)
        fig.text(.026,bottom+h/2,name,rotation=90,ha='center',va='center',
            color=color,weight='bold',fontsize=16)
        ims=[rgb,colorize(c['pred']),colorize(c['gt']),c['confidence'],c['upred'],c['ufeat']]
        for col,im in enumerate(ims):
            x=left+col*(w+gap)
            ax=fig.add_axes([x,bottom,w,h])
            if col<3:
                ax.imshow(im,interpolation='nearest')
                if col==0 and mask.any() and not mask.all():
                    ax.contour(mask,levels=[.5],colors=[color],linewidths=.6)
            else:
                ax.imshow(im,cmap='jet',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values():sp.set_visible(False)
            if r==0:fig.text(x+w/2,.947,titles[col],ha='center',va='bottom',fontsize=16)
    cbax=fig.add_axes([.421,.065,.19,.012])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='jet'),
        cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    cbax.tick_params(labelsize=10,length=2,pad=2)
    fig.text(.408,.071,'Shared heatmap scale',ha='right',va='center',fontsize=11)
    fig.text(.52,.019,'Contours indicate the named group; heatmaps show the whole scene on a fixed 0–1 scale.',
        ha='center',fontsize=11,color='#374151')
    a.output.mkdir(parents=True,exist_ok=True)
    stem=a.output/'DU_Match_Four_Groups_Grid_Epoch061'
    for ext in ['pdf','png','svg']:fig.savefig(str(stem)+'.'+ext,dpi=300,facecolor='white')
    plt.close(fig)
    (a.output/'four_groups_grid_provenance.json').write_text(json.dumps({
        'checkpoint':meta['checkpoint'],'completed_epochs':meta['checkpoint_epoch']+1,
        'model':meta['query_model'],'examples':sources,'selection':meta['case_selection'],
        'inference_note':meta['note'],'data_changes':False,'heatmap_scale':[0,1]},indent=2))
    print(stem)


if __name__=='__main__':main()
