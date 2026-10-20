"""Post-completion descriptive plot/report; no fitting, selection or new routes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=Path(__file__).resolve().parent;A=RUN/'artifacts'
assert json.loads((A/'audit.json').read_text())['passed']
rows=json.loads((A/'scores.json').read_text());summary=json.loads((A/'summary.json').read_text())
names=[('old_local_vote2','v19 local 2/3','o'),('old_local_vote3','v19 local 3/3','s'),
       ('bag_local_vote3','Five bags, local 3/5','^'),('bag_local_vote4','Five bags, local 4/5 (primary)','D'),
       ('bag_local_vote5','Five bags, local 5/5','v'),('bag_heads_vote4_full_local','Five bags, vote 4/5 then local','P'),
       ('old_heads_vote2_full_local','Old heads, vote 2/3 then local','X'),('full_local','Full-support local control','*')]
colors=plt.get_cmap('tab10').colors
fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
for ax,role in zip(axes,['day90','day270']):
    for i,(name,label,marker) in enumerate(names):
        r=[a for a in rows if a['role']==role and a['policy']==name]
        x=np.array([100*a['NAR'] for a in r]);y=np.array([100*a['Recovery'] for a in r])
        ax.errorbar(x.mean(),y.mean(),xerr=np.array([[x.mean()-x.min()],[x.max()-x.mean()]]),
                    marker=marker,color=colors[i],markersize=8,capsize=3,linewidth=1.2,linestyle='none',label=label)
    ax.axvline(1,color='crimson',ls='--',lw=1.2)
    ax.set_title('Day '+('90' if role=='day90' else '270'))
    ax.set_xlabel('NAR (%)');ax.set_xlim(0,1.32);ax.set_ylim(5,26);ax.grid(alpha=.2)
axes[0].set_ylabel('Recovery of originally wrong samples (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,fontsize=8,frameon=False)
fig.suptitle('Support coverage and consensus: observed development trade-off',fontsize=13)
fig.text(.5,.005,'Points: means over 3 model seeds x 3 support draws; bars: observed NAR min-max, not confidence intervals.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.15,1,.96])
for suffix in ['png','pdf']:fig.savefig(A/f'coverage_risk_recovery.{suffix}',dpi=180)
plt.close(fig)

description={'qualification':'post-completion description only; original primary decision unchanged; no fitting/threshold selection','comparisons':[]}
for role in ['day90','day270']:
    baseline=next(r for r in summary if r['role']==role and r['policy']=='old_local_vote2')
    for name in ['bag_local_vote3','bag_local_vote4','bag_heads_vote4_full_local','old_heads_vote2_full_local']:
        s=next(r for r in summary if r['role']==role and r['policy']==name)
        cells=[r for r in rows if r['role']==role and r['policy']==name]
        description['comparisons'].append({'role':role,'policy':name,'NAR_gt_1pct_states':sum(r['NAR']>.01 for r in cells),
            'Recovery_gain_vs_v19_pp':100*(s['Recovery']-baseline['Recovery']),
            'NAR_change_vs_v19_pp':100*(s['NAR']-baseline['NAR']),
            'deltaF1_vs_v19_pp':s['delta_f1_pp']-baseline['delta_f1_pp']})
(A/'descriptive_comparisons.json').write_text(json.dumps(description,ensure_ascii=False,indent=2)+'\n')
text='''
## 完成后解释（不替换预设主方法）

主方法五袋局部4/5投票在Day90/270的平均NAR为0.607/0.710%、最差0.755/0.781%，但Recovery较v19下降0.991/1.140pp，ΔF1下降0.209/0.420pp；更保守不构成原定改进。

五袋局部3/5找回更多纠错：Recovery较v19增加1.762/2.275pp，ΔF1增加0.447/0.874pp，但Day90有3/9、Day270有7/9状态NAR>1%。最多1.087/1.129%；不能以平均值掩盖超限。

五袋先4/5投票、再全支持近邻核验，NAR均值0.724/0.835%，最差0.879/0.907%，Recovery16.072/19.596%，相较v19的16.169/19.597%并无提升；ΔF1变化−0.018/−0.017pp。支持一个略降风险、接近原纠错的次级操作点，但差异很小，尚无跨日期确认或稳定优势。其同实际D较full-local诊断前沿仍低0.735/1.252pp。

解除每折近邻门控并非无代价：保持旧三个头只改为末端近邻核验，Recovery增加1.022/1.316pp，但NAR均值升至0.895/0.998%，且2/9与5/9状态超过1%。重复路由确实拒绝了一部分有效纠错，同时也拦住了一部分损伤。

本轮缩小了问题范围：更多支持覆盖能产生新增纠错；主要剩余困难是区分新增正确改写与新增误改。固定票数主要移动风险—纠错操作点，尚未稳定突破强局部几何路由前沿。后续若继续，应检验有标签的成对改写风险判别或更具独立性的局部证据，而不是继续增加票数；这只是后续假设，本轮未实施。

描述图：[PNG](artifacts/coverage_risk_recovery.png)、[PDF](artifacts/coverage_risk_recovery.pdf)。点为9状态均值，横线为NAR最小—最大范围，不是置信区间。所有预设次级结果完整保留，未按结果追加训练/阈值。
'''
p=RUN/'RESULTS.md'
assert '## 完成后解释（不替换预设主方法）' not in p.read_text()
p.write_text(p.read_text()+text)
