"""Source-only CE / local tail ranking, smooth group robustness and reference weighting."""
from runtime import *
from scipy.special import logsumexp,expit
from scipy.optimize import minimize,check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def evidence(z,b,c,cal):
 q=z.astype(np.float64)/np.array(cal['temperatures'])[:,:,None,None];w=np.array(cal['weights']);ix=np.arange(len(b));qc=q[:,:,ix,c].copy();qb=q[:,:,ix,b].copy();o=qc-qb;mu=np.sum(w[:,:,None]*o,(0,1));sd=np.sqrt(np.maximum(np.sum(w[:,:,None]*(o-mu)**2,(0,1)),0));lp=np.sum(w[:,:,None]*(qc-logsumexp(q,axis=-1)),(0,1));q[:,:,ix,c]=-np.inf;q[:,:,ix,b]=-np.inf;third=np.sum(w[:,:,None]*(qc-q.max(-1)),(0,1));return lp,third,mu-.25*sd

def reference_features(seed,sl):
 a=read(ROOT/C['v9']/'artifacts'/f'reference_{seed}_{sl}_smooth_balance.npz');b=a['base_logits'].argmax(-1);c=a['logits'].argmax(-1);zs=[]
 for s in C['seeds']:
  views=[]
  for v in C['views']:
   p=ROOT/C['v10']/'artifacts'/f'reference_{s}_{sl}_direction_iat.npz' if v=='original' else ROOT/C['v11']/'artifacts'/f'logits_{s}_direction_iat_{sl}_reference_{v}.npz'
   d=read(p);assert 'labels' not in d and np.array_equal(a['rows'],d['rows']);views.append(d['logits'])
  zs.append(np.stack(views))
 cal=json.loads((ROOT/C['v12']/'artifacts/source_calibration.json').read_text());lp,third,soft=evidence(np.stack(zs),b,c,cal)
 z=a['logits'].astype(np.float32);bz=a['base_logits'].astype(np.float32)
 def m(x):
  xx=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+np.float32(1e-12));cc=xx[np.arange(len(c)),c].copy();xx[np.arange(len(c)),c]=-np.inf;return cc-xx.max(-1)
 # Authoritative margin_gain already audited in v9; NumPy recomputation allows float reduction drift only.
 assert np.allclose(m(z)-m(bz),a['margin_gain'],atol=1e-5,rtol=1e-5)
 return dict(X=np.stack([a['margin_gain'],lp,third],-1),rows=a['rows'],changed=b!=c)

def subset(d,mask):return {k:v[mask] for k,v in d.items()}
def density(train,ref):
 mask=train['types']>0;s=train['X'][mask];rr=ref['X'][ref['changed']];cfg=C['density'];mean=s.mean(0);sd=np.maximum(s.std(0),.001)
 if len(rr)<cfg['min_reference_changed']:return None,dict(feasible=False,reason='too_few_reference_changed',reference_changed=len(rr))
 x=(np.concatenate([s,rr])-mean)/sd;y=np.r_[np.zeros(len(s),int),np.ones(len(rr),int)];groups=np.r_[np.char.add('source:',train['trace_groups'][mask]),np.array(['reference:'+str(int(row)) for row in ref['rows'][ref['changed']]])];p=np.zeros(len(x));folds=[]
 for tr,te in GroupKFold(cfg['folds']).split(x,y,groups):
  assert set(groups[tr]).isdisjoint(groups[te]);m=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-8).fit(x[tr],y[tr]);p[te]=m.predict_proba(x[te])[:,1];folds.append(dict(coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),heldout=te.tolist()))
 raw=np.exp(np.clip(np.log(np.maximum(p[:len(s)],1e-12))-np.log(np.maximum(1-p[:len(s)],1e-12)),-50,50));clipped=np.clip(raw,*cfg['clip']);clipped/=clipped.mean()
 ess=lambda w:float(w.sum()**2/(w@w));rawess=ess(raw)/len(s);clipess=ess(clipped)/len(s);support=float((p[len(s):]<cfg['reference_domain_p_max']).mean());feasible=rawess>=cfg['raw_ess_fraction_min'] and clipess>=cfg['clipped_ess_fraction_min'] and support>=cfg['reference_support_fraction_min']
 m=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-8).fit(x,y)
 model=dict(mean=mean.tolist(),sd=sd.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),folds=folds,source_changed_weights=clipped.tolist())
 info=dict(feasible=bool(feasible),source_changed=len(s),reference_changed=len(rr),raw_ess_fraction=rawess,clipped_ess_fraction=clipess,reference_support_fraction=support,domain_oof_auc=float(roc_auc_score(y,p)),clip_low_fraction=float((raw<.1).mean()),clip_high_fraction=float((raw>10).mean()),models=6)
 return model,info

def mix_losses(losses,gradients,mass,robust):
 pi=np.asarray(mass)/np.sum(mass);losses=np.asarray(losses);q=pi
 if robust:
  a=np.log(pi)+losses/C['fit']['robust_temperature'];value=C['fit']['robust_temperature']*logsumexp(a);q=np.exp(a-logsumexp(a))
 else:value=float(q@losses)
 return value,np.tensordot(q,np.asarray(gradients),axes=1)

def fitmodel(d,arm,weights=None,seed=42,check=False):
 mask=d['types']>0;xx=d['X'][mask];mean=xx.mean(0);sd=np.maximum(xx.std(0),.001);x=(xx-mean)/sd;y=d['types'][mask]-1;g=d['case_id'][mask];base=d['v12_primary'][mask];n=len(y);robust='B' in arm;pair='A' in arm
 w=np.ones(n) if weights is None else np.array(weights);groups=np.unique(g);common=dict(mean=mean.tolist(),sd=sd.tolist(),columns=[0,1,2],pair=pair,arm=arm)
 if not pair and not robust:
  lr=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-8).fit(x,y,sample_weight=w)
  common.update(coef=lr.coef_.tolist(),intercept=lr.intercept_.tolist(),iterations=int(lr.n_iter_[0]),success=True)
 else:
  data=[]
  if pair:
   rng=np.random.default_rng(seed);tail=[]
   for group in groups:
    rs=np.flatnonzero((g==group)&(y==0));nc=int(((d['case_id']==group)&(d['base']==d['labels'])).sum());budget=int(np.ceil(.01*nc));ds=np.flatnonzero((g==group)&(y==1));ds=ds[np.argsort(-base[ds],kind='stable')][:max(1,budget)];ws=np.flatnonzero((g==group)&(y==2));parts=[]
    for neg,cost in [(ds,1.),(ws,.25)]:
     if len(rs)==0 or len(neg)==0:continue
     total=len(rs)*len(neg);flat=rng.choice(total,min(total,C['fit']['pair_cap_per_group_type']),replace=False);a=rs[flat//len(neg)];b=neg[flat%len(neg)];pw=w[a]*w[b]*np.exp(-np.abs(base[a]-base[b])/C['fit']['local_distance_temperature']);pw/=pw.sum();parts.append((x[a]-x[b],pw,cost))
    if parts:data.append((group,parts));tail.append(dict(group=int(group),frozen_correct=nc,full_population_budget=budget,selected_D=len(ds)))
   if not data:raise RuntimeError('no source pairs')
   def objective(theta):
    losses=[];grads=[]
    for group,parts in data:
     loss=0.;grad=np.zeros(3)
     for delta,pw,cost in parts:
      z=-delta@theta;loss+=cost*float(pw@np.logaddexp(0,z));grad-=cost*(pw*expit(z))@delta
     losses.append(loss);grads.append(grad)
    val,grad=mix_losses(losses,grads,np.ones(len(losses)),robust);reg=C['fit']['rank_l2'];return val+.5*reg*(theta@theta),grad+reg*theta
   init=np.zeros(3)
  else:
   cw=n/(3*np.bincount(y,minlength=3));sw=w*cw[y];design=np.c_[x,np.ones(n)];one=np.eye(3)[y]
   for group in groups:
    ix=g==group;data.append((design[ix],one[ix],sw[ix]))
   def objective(theta):
    coef=theta.reshape(3,4);losses=[];grads=[];mass=[]
    for xx,yy,ww in data:
     z=xx@coef.T;lp=z-logsumexp(z,axis=1,keepdims=True);losses.append(float(ww@(-np.sum(yy*lp,1))/ww.sum()));grads.append(((ww[:,None]*(np.exp(lp)-yy)).T@xx/ww.sum()).ravel());mass.append(ww.sum())
    val,grad=mix_losses(losses,grads,mass,robust);reg=1/n;pen=coef.copy();pen[:,-1]=0;return val+.5*reg*np.sum(pen**2),grad+reg*pen.ravel()
   init=np.zeros(12)
  if check:
   test=np.arange(len(init))*.003;err=float(check_grad(lambda z:objective(z)[0],lambda z:objective(z)[1],test));assert err<2e-5,err;common['gradient_check_error']=err
  result=minimize(objective,init,jac=True,method='L-BFGS-B',options=dict(maxiter=C['fit']['optimizer_maxiter'],ftol=C['fit']['optimizer_ftol'],gtol=C['fit']['optimizer_gtol']))
  assert result.success and np.isfinite(result.x).all(),result.message
  if pair:common.update(coef=result.x.tolist(),intercept=0.,tail=tail)
  else:common.update(coef=result.x.reshape(3,4)[:,:3].tolist(),intercept=result.x.reshape(3,4)[:,3].tolist())
  common.update(iterations=int(result.nit),success=True,final_objective=float(result.fun))
 u=project(d['X'][mask],common);common.update(u_mean=float(u.mean()),u_sd=max(float(u.std()),.001));return common

def project(X,m):
 x=(X-np.array(m['mean']))/np.array(m['sd'])
 if m.get('pair',False):return x@np.array(m['coef'])
 g=x@np.array(m['coef']).T+np.array(m['intercept']);return g[:,0]-logsumexp(g[:,1:],axis=1)

def values(d,models,old):
 out=dict(v9_margin=d['v9_margin'],v12_primary=d['v12_primary']);out['v14_primary']=d['v12_primary']+.25*(project(d['X'],old)-old['u_mean'])/old['u_sd']
 for arm,m in models.items():
  if m is not None:out[arm]=d['v12_primary']+.25*(project(d['X'],m)-m['u_mean'])/m['u_sd']
 return out

def prefix(t,s):
 ix=np.flatnonzero(t>0)
 if not len(ix):return np.array([[np.nextafter(float(np.max(s)),np.inf),0,0,0]])
 ix=ix[np.argsort(-s[ix],kind='stable')];ends=np.flatnonzero(np.r_[s[ix][1:]!=s[ix][:-1],True]);return np.stack([np.r_[np.nextafter(s[ix].max(),np.inf),s[ix][ends]]]+[np.r_[0,np.cumsum(t[ix]==k)[ends]] for k in [1,2,3]],-1)

def calibrate(d,s):
 p=prefix(d['types'],s);nc=int((d['base']==d['labels']).sum());u=p[:,1]-p[:,2]-.25*p[:,3];ix=np.flatnonzero((p[:,2]/nc<=.01)&(p[:,1]>=10)&(u>0))
 if not len(ix):return dict(threshold=None,R=0,D=0,W=0,utility=0.)
 j=max(ix,key=lambda j:(u[j],-p[j,2],p[j,0]));return dict(threshold=float(p[j,0]),R=int(p[j,1]),D=int(p[j,2]),W=int(p[j,3]),utility=float(u[j]))

def policy(y,b,c,s,th):
 m=(b!=c)&(s>=th) if th is not None else np.zeros(len(y),bool);p=np.where(m,c,b);r=int((m&(b!=y)&(c==y)).sum());dd=int((m&(b==y)).sum());w=int((m&(b!=y)&(c!=y)).sum());mat=np.zeros((102,102));np.add.at(mat,(y,p),1);den=mat.sum(0)+mat.sum(1);return dict(R=r,D=dd,W=w,NAR=dd/max(int((b==y).sum()),1),Recovery=r/max(int((b!=y).sum()),1),accuracy=float((p==y).mean()),macro_f1=float(np.divide(2*mat.diagonal(),den,out=np.zeros(102),where=den>0).mean()),threshold=th)
