import numpy as np
from sklearn.metrics import roc_curve,f1_score
from runtime import C

def partial_trap(x,y,a):
    stop=np.searchsorted(x,a,side='right');xx=x[:stop];yy=y[:stop]
    if len(xx)==0:xx=np.array([0.]);yy=np.array([0.])
    if xx[-1]<a:
        end=np.interp(a,x,y);xx=np.r_[xx,a];yy=np.r_[yy,end]
    return float(np.sum(np.diff(xx)*(yy[:-1]+yy[1:])*.5))
def curve(y,b,c,s):
    change=b!=c;bc=b==y;bad=~bc;corr=bad&(c==y);other=bad&(c!=y)&change;ix=np.flatnonzero(change);ix=ix[np.argsort(-s[ix],kind='stable')];ends=np.flatnonzero(np.r_[s[ix][1:]!=s[ix][:-1],True]) if len(ix) else np.array([],dtype=int)
    fp=np.r_[0,np.cumsum(bc[ix])[ends]];tp=np.r_[0,np.cumsum(corr[ix])[ends]];op=np.r_[0,np.cumsum(other[ix])[ends]];x=fp/max(bc.sum(),1);rec=tp/max(bad.sum(),1);a=C['metrics']['alpha'];allowed=np.flatnonzero(x<=a);j=int(allowed[np.argmax(rec[allowed])]);th=float(s[ix][ends[j-1]]) if j else None;mask=change&(s>=th) if th is not None else np.zeros(len(y),bool);p=np.where(mask,c,b)
    assert int((mask&bc).sum())==int(fp[j]) and int((mask&corr).sum())==int(tp[j]) and int((mask&other).sum())==int(op[j])
    # Unique risk budgets retain the maximal recovery attainable by a deterministic score threshold.
    unique_last=np.flatnonzero(np.r_[x[1:]!=x[:-1],True]);ux=x[unique_last];ur=rec[unique_last];keep=ux<=a;ux=ux[keep];ur=ur[keep];step=float(np.sum(np.diff(np.r_[ux,a])*ur)/a)
    total=int(corr.sum());raw=partial_trap(x,tp/total,a) if total else None
    nc=int(bc.sum());nw=int(bad.sum());damage=int(fp[j]);correct=int(tp[j]);invalid=int(op[j]);f1=float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0));m=np.zeros((102,102),dtype=np.int64);np.add.at(m,(y,p),1);den=m.sum(0)+m.sum(1);ff=np.divide(2*np.diag(m),den,out=np.zeros(102),where=den>0);assert abs(ff.mean()-f1)<1e-12
    met=dict(pauc_raw=raw,pauc_norm=raw/a if raw is not None else None,step_mean_recovery=step,rec_at_nar1=float(rec[j]),attained_nar1=float(x[j]),threshold=th,diag_f1=f1,accuracy=float((p==y).mean()),corrected=correct,damaged=damage,other_wrong_accepted=invalid,other_wrong_total=int(other.sum()),other_wrong_retention=invalid/max(int(other.sum()),1),other_per_correction=invalid/correct if correct else None,accepted_changes=int(mask.sum()),precision=correct/max(int(mask.sum()),1),frozen_correct=nc,frozen_wrong=nw,correctable_candidates=total,oracle_recovery=total/max(nw,1),raw_candidate_nar=float((bc&change).sum()/max(nc,1)),changed_count=len(ix))
    # An independent full-negative-population ROC confirms FPR=NAR and the local integral.
    if total and nc:
        floor=float(s[ix].min()-max(1.,abs(s[ix].min()))*2) if len(ix) else -1.;positive_scores=s[corr];negative_scores=np.where(change[bc],s[bc],floor);yy=np.r_[np.ones(total),np.zeros(nc)];ss=np.r_[positive_scores,negative_scores];rx,ry,_=roc_curve(yy,ss,drop_intermediate=False);check=partial_trap(rx,ry,a);assert abs(check-raw)<1e-12,(check,raw)
    # Verify the NAR/Recovery area identity for trapezoidal curves, not step interpolation.
    recovery_trap=partial_trap(x,rec,a)/a
    if raw is not None:assert abs(recovery_trap-(raw/a)*met['oracle_recovery'])<1e-12
    met['trapezoid_mean_recovery']=recovery_trap
    return met,np.stack((x,rec,tp/max(total,1)),axis=-1),p

def audit_direct(y,b,c,s,met):
    change=b!=c;bc=b==y;bad=~bc;corr=bad&(c==y);other=change&bad&(c!=y);a=C['metrics']['alpha'];thresholds=np.unique(s[change])[::-1];best_rec=0.;best=None
    # Exhaustively recheck a bounded synthetic case; large states validate the selected threshold directly.
    if len(y)<=64:
        for t in thresholds:
            m=change&(s>=t);nar=(m&bc).sum()/max(bc.sum(),1);r=(m&corr).sum()/max(bad.sum(),1)
            if nar<=a and r>best_rec:best_rec=r;best=t
        assert abs(best_rec-met['rec_at_nar1'])<1e-12
    mask=change&(s>=met['threshold']) if met['threshold'] is not None else np.zeros(len(y),bool);assert int((mask&bc).sum())==met['damaged'] and int((mask&corr).sum())==met['corrected'] and int((mask&other).sum())==met['other_wrong_accepted']
    assert abs((mask&bc).sum()/max(bc.sum(),1)-met['attained_nar1'])<1e-12
    return True

def preflight():
    # One correction, one damage, one different-wrong, and unchanged correct negatives.
    y=np.zeros(204,dtype=int);b=np.zeros(204,dtype=int);b[:3]=1;c=b.copy();c[0]=0;c[1]=2;c[3]=1;s=np.zeros(204);s[0]=3;s[1]=2;s[3]=1;m,_,_=curve(y,b,c,s);assert m['corrected']==1 and m['other_wrong_total']==1 and m['other_wrong_accepted']==0 and m['frozen_correct']==201 and m['damaged']==0;assert abs(m['pauc_raw']-.01)<1e-12 and abs(m['step_mean_recovery']-1/3)<1e-12;audit_direct(y,b,c,s,m)
    # Reverse the correction/damage ordering; exact low-risk partial integral is hand-checkable.
    s[0]=1;s[3]=3;s[1]=2;m,_,_=curve(y,b,c,s);assert abs(m['pauc_raw']-(.01-1/201))<1e-12
    # Tied correction/damage cannot be split to satisfy a 1% actual NAR budget.
    y=np.zeros(5,dtype=int);b=np.array([1,0,0,0,0]);c=np.array([0,1,0,0,0]);s=np.array([1.,1.,0.,0.,0.]);m,_,_=curve(y,b,c,s);assert m['corrected']==0 and m['attained_nar1']==0 and m['step_mean_recovery']==0;audit_direct(y,b,c,s,m)
    # No correctable candidates is undefined pAUC, not invented area=0.
    c=b.copy();m,_,_=curve(y,b,c,s);assert m['pauc_raw'] is None and m['rec_at_nar1']==0
    return dict(passed=True,full_negative_fpr_equals_nar=True,three_outcome_taxonomy=True,tied_groups_not_split=True,hand_integral=True,sklearn_partial_roc_agreement=True,undefined_no_positive=True)
