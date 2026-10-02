"""Train-only short local occlusion; preserves observation mask and packet indices."""
import torch
def span_mask(directions,observed,generator):
    n=observed.sum(1);width=torch.floor(n*.02).long().clamp(min=1,max=32);width=torch.minimum(width,(n-1).clamp(min=0))
    apply=(torch.rand(len(n),device=directions.device,generator=generator)<.5)&(n>1)
    start=torch.floor(torch.rand(len(n),device=directions.device,generator=generator)*(n-width+1)).long()
    position=torch.arange(directions.shape[1],device=directions.device)[None]
    erased=apply[:,None]&(position>=start[:,None])&(position<(start+width)[:,None])&observed
    return directions.masked_fill(erased,0)
