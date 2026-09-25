from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from ta_wf_next.models import VarCNNDirection
from ta_wf_next.traffic_views import generate_views
RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
ROLES=("source","valid","day14","day30","day90","day150","day270"); CONDITIONS=("varcnn_only","varcnn_constant","varcnn_window")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
class Fusion(nn.Module):
    def __init__(self):
        super().__init__(); self.var=VarCNNDirection(102)
        self.fine=nn.Sequential(nn.Conv1d(2,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.coarse=nn.Sequential(nn.Conv1d(2,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(640,512),nn.ReLU(),nn.Dropout(.5)); self.mlp=nn.Linear(512,102)
    def forward(self,p,w):
        fine=w[:,:200].reshape(-1,100,2).transpose(1,2); coarse=w[:,200:].reshape(-1,20,2).transpose(1,2); f=self.var.forward_features(p[:,None,:]); t=torch.cat((self.fine(fine),self.coarse(coarse)),1); return self.mlp(self.head(torch.cat((f,t),1)))
def metric(y,p):
    a=np.bincount(y*102+p,minlength=102**2).reshape(102,102); d=a.sum(0)+a.sum(1); return float(np.mean(y==p)),float(np.divide(2*np.diag(a),d,out=np.zeros(102),where=d!=0).mean())
def main():
    torch.set_num_threads(4); cfg=json.loads((RUN/"config.json").read_text()); man=json.loads((RUN/"artifacts/manifest.json").read_text()); seal=json.loads((RUN/"artifacts/input_seal.json").read_text()); errors=[]
    for rel,h in seal.items():
        if sha(ROOT/rel)!=h: errors.append("input seal "+rel)
    sampling_path=ROOT/man["sampling_manifest"]
    if sha(sampling_path)!=man["sampling_sha256"]: errors.append("sampling hash")
    sampling=json.loads(sampling_path.read_text())["sampling"]; data_root=Path(json.loads((ROOT/"configs/datasets.json").read_text())["data_root"]); data={}; hashes={}; source_full=None
    for role in ROLES:
        it=sampling[role]
        with np.load(data_root/it["path"],allow_pickle=False) as ar: idx=np.asarray(it["rows"],np.int64); y=ar["y"][idx].astype(np.int64); traces=ar["X"][idx]
        if len(idx)!=it["count"] or len(np.unique(idx))!=len(idx) or len(np.unique(y))!=102: errors.append(role+" rows/classes")
        p=np.sign(traces[:,:5000]).astype(np.float32); full=np.zeros((len(p),240),np.float32); neutral=np.zeros_like(full)
        for i,row in enumerate(traces):
            views=generate_views(row[:5000],input_kind="signed_timestamp",budget=5000,window_sizes=(50,250)); off=0
            for width,ws in views.direction_windows:
                for j,w in enumerate(ws): c=off+2*j; full[i,c:c+2]=w.positive_fraction,w.transition_fraction; neutral[i,c:c+2]=(0.5,0.) if w.partial else (w.positive_fraction,w.transition_fraction)
                off+=2*((5000+width-1)//width)
        source_full=full if role=="source" else source_full; data[role]=(p,neutral,y); hashes[role]={hashlib.sha256(x.astype(np.int8).tobytes()).hexdigest() for x in p}
    overlap=len(hashes["source"]&hashes["valid"])
    if overlap: errors.append("source valid overlap")
    with np.load(RUN/"artifacts/window_statistics.npz",allow_pickle=False) as ar: mean,std=ar["mean"],ar["std"]
    em,es=source_full.mean(0),source_full.std(0); es[es<1e-6]=1.
    if not np.array_equal(mean,em) or not np.array_equal(std,es): errors.append("source stats")
    for role in ROLES: p,w,y=data[role]; data[role]=(p,((w-mean)/std).astype(np.float32),y)
    path=RUN/"artifacts/predictions.npz"
    if sha(path)!=man["prediction_sha256"]: errors.append("predictions hash")
    with np.load(path,allow_pickle=False) as ar: preds={k:ar[k] for k in ar.files}
    with (RUN/"artifacts/metrics.csv").open(newline="") as f: rows={(r["role"],r["condition"],int(r["seed"])):r for r in csv.DictReader(f)}
    hist=json.loads((RUN/"artifacts/history.json").read_text())
    if len(preds)!=63 or len(rows)!=63 or len(hist)!=9: errors.append("counts")
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); replay_count=0
    for seed in cfg["training_seeds"]:
      bases=set(); fusions=set()
      for cond in CONDITIONS:
        key=f"{cond}_{seed}"; h=hist[key]; best=max(h,key=lambda x:x["macro_f1"]); info=man["models"][key]; cp=torch.load(RUN/"checkpoints"/f"{key}.pt",map_location="cpu",weights_only=True)
        if len(h)!=45 or [x["epoch"] for x in h]!=list(range(1,46)) or cp["epoch"]!=best["epoch"] or info["best_epoch"]!=best["epoch"] or abs(info["valid_macro_f1"]-best["macro_f1"])>1e-12: errors.append(key+" selection")
        bases.add(info["initial_base_hash"])
        if cond!="varcnn_only": fusions.add(info["initial_fusion_hash"])
        model=(VarCNNDirection(102) if cond=="varcnn_only" else Fusion()).to(dev); model.load_state_dict(cp["state_dict"]); model.eval()
        if sum(x.numel() for x in model.parameters())!=info["parameters"]: errors.append(key+" parameters")
        for role in ROLES:
          p,w,y=data[role]; w=np.zeros_like(w) if cond=="varcnn_constant" else w; out=[]
          with torch.inference_mode():
            for st in range(0,len(p),64):
              x=torch.from_numpy(p[st:st+64]).to(dev); z=model(x[:,None,:])[0] if cond=="varcnn_only" else model(x,torch.from_numpy(w[st:st+64]).to(dev)); out.append(z.argmax(1).cpu().numpy())
          k=f"{role}_{key}"; pr=np.concatenate(out); replay_count+=1
          if not np.array_equal(pr,preds[k]): errors.append(k+" replay")
          acc,f1=metric(y,preds[k]); row=rows[(role,cond,seed)]
          if abs(float(row["accuracy"])-acc)>1e-12 or abs(float(row["macro_f1"])-f1)>1e-12 or int(row["best_epoch"])!=best["epoch"]: errors.append(k+" metric")
      if len(bases)!=1 or len(fusions)!=1: errors.append(str(seed)+" init")
    result={"source_valid_direction_overlap":overlap,"checkpoint_predictions_replayed":replay_count,"metric_rows":len(rows),"errors":errors}; (RUN/"artifacts/integrity.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if errors: raise SystemExit(1)
if __name__=="__main__": main()
