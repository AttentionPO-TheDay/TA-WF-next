import json,os
from pathlib import Path
import torch
import port_base as B
from mechanism import control,keep_mask
R=B.R;C=B.C
assert C['status']=='frozen' and C['id']==R.name
assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(C['physical_gpu']) and torch.cuda.is_available()
assert control(1.,1.,C['adaptation']['base_learning_rate'])['scale']==1.0
assert control(2.,1.,C['adaptation']['base_learning_rate'])['scale']==.5
assert bool(keep_mask(torch.tensor([.01,.02,.5,.51]),.02,.5).tolist()==[False,True,True,False])
assert not list((R/'artifacts').glob('gmm_*.npz'))
model,_=B.load_model(C['seeds'][0]); source=B.load_prepared(C['prepared']['source_valid'],'source')
x=torch.from_numpy(source['timestamps'][:4]).to(B.DEVICE);t=torch.from_numpy(source['tam'][:4]).to(B.DEVICE)
with torch.no_grad():
 z,_=B.forward_with_features(model,x,t); assert z.shape==(4,102) and torch.isfinite(z).all()
B.save_json('preflight.json',{'passed':True,'gpu':torch.cuda.get_device_name(0),'pseudo_label_calls':0,'gmm_files':0,'dynamic_control_checks':True,'smoke_rows':4})
print('preflight passed',torch.cuda.get_device_name(0))
