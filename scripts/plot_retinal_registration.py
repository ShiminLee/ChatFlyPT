"""Plot the saved geometric registration audit, without running a neural model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'research/retinal-registration-followup-20260924/candidate-receptor-map.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/unified-pathway-20260924/retinal-registration')
    args=parser.parse_args()
    source=json.loads(args.source.read_text(encoding='utf8'))
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.3),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,top=.80,bottom=.24,wspace=.08)
    for ax,side in zip(axes,('left','right')):
        rows=[r for r in source['mapped'] if r['side']==side]
        for key,label,color in [('old_angular_error_deg','Previous bounding-box map','#c47b31'),
                                ('angular_error_deg','Author optical axes + camera geometry','#197b7a')]:
            values=np.sort([r[key] for r in rows])
            ax.step(values,np.arange(1,len(values)+1)/len(values)*100,
                    where='post',label=f'{label}\nmedian {np.median(values):.2f}°, 95th {np.quantile(values,.95):.2f}°',color=color,lw=1.8)
        ax.set_xscale('log');ax.set_xlim(.1,180);ax.set_ylim(0,101)
        ax.set_xticks([.1,1,10,100],labels=['0.1','1','10','100'])
        ax.set_xlabel('Angular difference from author atlas (degrees; log scale)')
        ax.set_title(f'{side.capitalize()} eye: {len(rows):,} accepted photoreceptors',fontsize=11,pad=12)
        ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',frameon=True,framealpha=.94,edgecolor='none',fontsize=7.5)
    axes[0].set_ylabel('Fraction of accepted photoreceptors (%)')
    fig.suptitle('Optical-axis registration audit — matched cells only',fontsize=14,y=.96)
    fig.text(.075,.035,'Geometric resampling error, not biological accuracy. Rejected/out-of-view cells are excluded;\n'
             'cells sharing a column are correlated. This figure does not validate neural dynamics or behavior.',fontsize=9,color='#555555')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    print(args.output.with_suffix('.png'))


if __name__=='__main__':main()
