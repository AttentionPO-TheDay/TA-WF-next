"""Incremental ordered/counts path; original generator remains unchanged."""
import torch
from torch import nn
from ta_wf_next.learnable_generator import LocalPacketGenerator

class ResidualGenerator(LocalPacketGenerator):
    def __init__(self,mode):
        super().__init__(pool='attention',packet_budget=5000,d_model=52)
        assert mode in ('counts','ordered')
        self.residual_mode=mode
        self.order_residual=nn.Linear(100,52,bias=False)
        nn.init.zeros_(self.order_residual.weight)

    def residual_input(self,directions,observed,dtype):
        assert directions.ndim==2 and directions.shape[1]==5000
        assert observed.dtype==torch.bool and observed.shape==directions.shape
        d=directions.to(dtype).masked_fill(~observed,0).reshape(-1,100,5,10)
        if self.residual_mode=='counts':d=d.mean(-1,keepdim=True).expand_as(d)
        flags=observed.to(dtype).reshape(-1,100,50)
        return torch.cat((d.reshape(-1,100,50),flags),dim=-1)

    def residual_tokens(self,directions,observed,dtype):
        output=self.order_residual(self.residual_input(directions,observed,dtype))
        return output*observed.reshape(-1,100,50).any(-1)[...,None]

    def forward(self,directions,observed,summaries):
        original=super().forward(directions,observed,summaries)
        return original+self.residual_tokens(directions,observed,summaries.dtype)

def attach_residual(model,mode):
    # Constructor allocations cannot alter shared parameters or training RNG.
    with torch.random.fork_rng(devices=[]):
        generator=ResidualGenerator(mode)
    missing,unexpected=generator.load_state_dict(model.generator.state_dict(),strict=False)
    assert missing==['order_residual.weight'] and not unexpected
    model.generator=generator
