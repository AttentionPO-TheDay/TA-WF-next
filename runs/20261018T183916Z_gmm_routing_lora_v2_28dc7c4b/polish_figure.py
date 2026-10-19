"""Posthoc export formatting only; unchanged audited summary values."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime import RUN,sha,save
s=json.loads((RUN/'artifacts/summary.json').read_text())
labels=[r['condition'].replace('single','1 proto').replace('gmm4','4 proto').replace('ungated','no gate').replace('routed','gate').replace('_',' / ') for r in s]
y=np.arange(len(s));colors=['#3278a7' if r['condition'].startswith('single') else '#d08535' for r in s]
fig,axs=plt.subplots(1,2,figsize=(11,5.5),sharey=True,gridspec_kw={'width_ratios':[1,1]})
for ax,metric,title,xlabel in zip(axs,['delta_f1_pp','nar'],['Net recovery','Negative adaptation'],['Delta Macro-F1 (pp)','NAR (%)']):
 values=[r[metric+'_mean'] for r in s];errors=[r[metric+'_sd'] for r in s]
 bars=ax.barh(y,values,xerr=errors,color=colors,alpha=.85,height=.62,error_kw={'capsize':3,'elinewidth':1})
 ax.axvline(0,c='grey',lw=.8)
 for i,(bar,value) in enumerate(zip(bars,values)):
  if '_lora' in s[i]['condition']:bar.set_hatch('//')
  ax.text(value+errors[i]+.015,i,f'{value:.3f}',va='center',fontsize=8)
 ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
 ax.set_xlim(-.03,max(v+e for v,e in zip(values,errors))*1.25+.05)
axs[0].set_yticks(y,labels,fontsize=9);axs[0].invert_yaxis()
fig.suptitle('All 8 combinations: equal-weight factor means over 3 seeds',fontsize=12)
fig.text(.5,.015,'Error bars: sample SD across seeds. Blue: single prototype; orange: GMM4; hatched: Q/V LoRA. Gate rows preserve all classifications.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.045,1,.95])
for ext in ['png','pdf','svg']:fig.savefig(RUN/'artifacts/figures'/f'combination_f1_nar.{ext}',dpi=180)
plt.close(fig)
p=RUN/'artifacts/delivery_audit.json';a=json.loads(p.read_text())
for file in (RUN/'artifacts/figures').glob('combination_f1_nar.*'):a['outputs'][str(file.relative_to(RUN))]=sha(file)
a['figure_formatting_revision']={'reason':'Original scatter annotations overlapped and legend clipped; same audited values exported as horizontal bars with seed SD. No score or method change.','script_sha256':sha(RUN/'polish_figure.py')}
save('delivery_audit.json',a)
