from runtime import *
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from metrics import curve,audit_direct,preflight
def kind(y, b, c):
    t = np.zeros(len(y), dtype=np.int8)
    t[(b != c) & (b != y) & (c == y)] = 1
    t[(b != c) & (b == y)] = 2
    t[(b != c) & (b != y) & (c != y)] = 3
    assert np.array_equal(t > 0, b != c)
    return t


def evidence(z, b, c, cal):
    temp, weight = np.array(cal['temperatures']), np.array(cal['weights'])
    q = z.astype(np.float64)/temp[:, :, None, None]
    n = len(b)
    ix = np.arange(n)
    qc, qb = q[:, :, ix, c].copy(), q[:, :, ix, b].copy()
    odds = qc-qb
    mean = np.sum(weight[:, :, None]*odds, axis=(0, 1))
    sd = np.sqrt(np.maximum(np.sum(weight[:, :, None]*(odds-mean)**2, axis=(0, 1)), 0))
    logpc = qc-logsumexp(q, axis=-1)
    logpc = np.sum(weight[:, :, None]*logpc, axis=(0, 1))
    q[:, :, ix, c] = -np.inf
    q[:, :, ix, b] = -np.inf
    third = np.sum(weight[:, :, None]*(qc-q.max(-1)), axis=(0, 1))
    return logpc, third, mean-.25*sd


def prefix(t, s):
    ix = np.flatnonzero(t > 0)
    if not len(ix):
        return np.array([[np.nextafter(float(np.max(s)), np.inf), 0., 0., 0.]])
    ix = ix[np.argsort(-s[ix], kind='stable')]
    ends = np.flatnonzero(np.r_[s[ix][1:] != s[ix][:-1], True])
    thresholds = np.r_[np.nextafter(s[ix].max(), np.inf), s[ix][ends]]
    return np.stack([thresholds]+[np.r_[0, np.cumsum(t[ix] == k)[ends]] for k in [1, 2, 3]], axis=-1)


def calibrate(d, values):
    t, b, y = d['types'], d['base'], d['labels']
    cr = prefix(t, values)
    nc = int((b == y).sum())
    utility = cr[:, 1]-cr[:, 2]-C['calibration']['utility_W_cost']*cr[:, 3]
    allowed = np.flatnonzero((cr[:, 2]/nc <= .01) & (cr[:, 1] >= C['calibration']['min_R']) & (utility > 0))
    if not len(allowed):
        return dict(threshold=None, R=0, D=0, W=0, utility=0., reason='no_feasible_positive_utility_source_rule')
    j = max(allowed, key=lambda i: (utility[i], -cr[i, 2], cr[i, 0]))
    return dict(threshold=float(cr[j, 0]), R=int(cr[j, 1]), D=int(cr[j, 2]), W=int(cr[j, 3]), utility=float(utility[j]), source_NAR=float(cr[j, 2]/nc), reason='fixed_source_utility_rule')


def policy_metrics(y, b, c, score, threshold):
    mask = (b != c) & (score >= threshold) if threshold is not None else np.zeros(len(y), bool)
    p = np.where(mask, c, b)
    nc, nw = int((b == y).sum()), int((b != y).sum())
    r = int((mask & (b != y) & (c == y)).sum())
    d = int((mask & (b == y)).sum())
    w = int((mask & (b != y) & (c != y)).sum())
    mat = np.zeros((102, 102), dtype=np.int64)
    np.add.at(mat, (y, p), 1)
    den = mat.sum(0)+mat.sum(1)
    f1 = np.divide(2*mat.diagonal(), den, out=np.zeros(102), where=den > 0).mean()
    return dict(R=r, D=d, W=w, NAR=d/nc, Recovery=r/nw, accuracy=float((p == y).mean()), macro_f1=float(f1), accepted=int(mask.sum()), threshold=threshold)



def fitmodel(d):
 mask=d['types']>0;x=d['X'][mask];mean=x.mean(0);sd=np.maximum(x.std(0),.001);xx=(x-mean)/sd
 m=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-8).fit(xx,d['types'][mask]);assert np.array_equal(m.classes_,[1,2,3])
 model=dict(mean=mean.tolist(),sd=sd.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),iterations=int(m.n_iter_[0]))
 u=projection(d['X'][mask],model);model.update(u_mean=float(u.mean()),u_sd=max(float(u.std()),.001));return model

def projection(X,m):
 x=(X-np.array(m['mean']))/np.array(m['sd']);g=x@np.array(m['coef']).T+np.array(m['intercept']);return g[:,0]-logsumexp(g[:,1:],axis=1)

def values(d,models,old):
 out=dict(v9_margin=d['v9_margin'],v12_primary=d['v12_primary']);out['v14_primary']=d['v12_primary']+.25*(projection(d['X'],old)-old['u_mean'])/old['u_sd']
 for arm,m in models.items():
  if m is not None:out[arm]=d['v12_primary']+.25*(projection(d['X'],m)-m['u_mean'])/m['u_sd']
 return out
