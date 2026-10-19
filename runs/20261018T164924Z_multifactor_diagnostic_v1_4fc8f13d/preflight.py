"""Source-only parity, topology and adaptation scope checks before target access."""
import json
import torch
from runtime import RUN,C,setup,load_model,source,reference,save,configure
from mechanisms import adapt,from_tokens,get_tokens,topology
from token_alignment import aligned_generator

def main():
    setup();s=source();small={k:s[k][:1024] for k in ('timestamps','tam','rows')};del s
    m=load_model(42);t=get_tokens(m,small)
    with torch.no_grad():
        z=m(small['timestamps'][:128].cuda(),small['tam'][:128].cuda());cached=from_tokens(m,t[:128])
    assert torch.equal(z,cached),float((z-cached).abs().max())
    a=torch.eye(4,device='cuda');assert topology(a,a)==0 and topology(a,a.roll(1,1)+.3)>0
    ref=reference(42);checks={}
    for condition in C['conditions']:
        model=load_model(42);aligner,history,trainable=adapt(model,small,condition,ref,small,steps=2)
        checks[condition]={'steps':len(history),'trainable_names':trainable,'aligner_parameters':sum(p.numel() for p in aligner.parameters()) if aligner is not None else 0}
        if condition.startswith('prototype_'):
            assert history[-1]['topology_classes']>=2
        del model,aligner
    save('source_smoke.json',{'passed':True,'source_only':True,'target_inference':False,'cached_token_forward_bitwise_equal':True,'checks':checks})
    print('Source-only scopes, gradients, token parity and topology checks passed.',flush=True)

if __name__=='__main__':main()
