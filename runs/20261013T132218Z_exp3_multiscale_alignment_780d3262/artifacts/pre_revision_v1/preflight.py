import json,os,torch
from pathlib import Path
import port_base as B
R=B.R; C=B.C
assert C["status"]=="frozen" and os.environ.get("CUDA_VISIBLE_DEVICES")==str(C["physical_gpu"]) and torch.cuda.is_available()
model,_=B.load_model(C["seeds"][0]); source=B.load_prepared(C["prepared"]["source_valid"],"source")
with torch.no_grad():
 x=torch.from_numpy(source["timestamps"][:4]).to(B.DEVICE); t=torch.from_numpy(source["tam"][:4]).to(B.DEVICE); z,_=B.forward_with_features(model,x,t); assert z.shape==(4,102) and torch.isfinite(z).all()
B.save_json("preflight.json",{"passed":True,"gpu":torch.cuda.get_device_name(0),"pseudo_label_calls":0,"gmm_files":0,"mechanism_kind":C["mechanism_kind"],"smoke_rows":4})
print("preflight passed",torch.cuda.get_device_name(0))
