"""Post-completion descriptive operating-point figure; no method selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text());ART=RUN/'artifacts'


def main():
    assert json.loads((ART/'audit.json').read_text())['passed']
    current=json.loads((ART/'scores.json').read_text())
    old=json.loads((ROOT/C['parent_run']/'artifacts/scores.json').read_text())
    methods=[('Full head',current,'full_head1','#777777','o'),
             ('Full local route',current,'full_local_head1','#c88a2a','s'),
             ('OOF-selected route (v18)',old,'cv_halfpct','#3497b8','D'),
             ('Local + 2 fold votes',current,'local_head1_uncalibrated_vote2','#22834b','*'),
             ('Local + 3 fold votes',current,'local_head1_uncalibrated_vote3','#9061b4','v'),
             ('Calibrated agree + 2 votes (primary)',current,'agree_head1_shrink_zero_vote2','#c94b46','^')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True)
    for ax,role,labels in zip(axes,['day90','day270'],[496,504]):
        raw=[r for r in current if r['role']==role and r['cap']==5 and r['policy']=='full_head1']
        half=50*np.mean([r['Recovery'] for r in raw])
        ax.axvline(1,color='#333333',linestyle='--',linewidth=1,label='NAR = 1%')
        ax.axhline(half,color='#aaaaaa',linestyle=':',linewidth=1)
        for title,rows,name,color,marker in methods:
            rs=[r for r in rows if r['role']==role and r['cap']==5 and r['policy']==name]
            xx=100*np.array([r['NAR'] for r in rs]);yy=100*np.array([r['Recovery'] for r in rs])
            x,y=xx.mean(),yy.mean()
            ax.errorbar(x,y,xerr=[[x-xx.min()],[xx.max()-x]],fmt=marker,color=color,
                        markersize=11 if marker=='*' else 6,capsize=3,linewidth=1.3,label=title)
        ax.set_title(f'{role.replace("day","Day ")} — cap5, {labels} labels/trial')
        ax.set_xlabel('Actual NAR (%)');ax.set_xlim(-.08,5.3);ax.set_ylim(0,36)
        ax.grid(alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Recovery of frozen errors (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.045),fontsize=9)
    fig.text(.5,.015,'Points: means over 3 model seeds × 3 support draws; horizontal bars: min–max NAR (not confidence intervals).\nDotted horizontal line: 50% of full-head recovery. Observed development queries; no certified future-risk guarantee.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.25,1,1))
    for suffix in ['png','pdf']:fig.savefig(ART/f'cap5_risk_recovery.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(ART/'cap5_risk_recovery.png')


if __name__=='__main__':main()
