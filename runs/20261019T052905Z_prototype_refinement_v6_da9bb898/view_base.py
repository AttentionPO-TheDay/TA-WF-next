"""View ablation of the completed multiview system, preserving initialization.

base_model.py is an exact, explicit copy of the preceding run's model.py.
No archived project imports, pretrained weights, or new encoder are used.
"""
import torch
from torch import nn
from base_model import MultiViewModel as BaseModel, parameter_counts

VIEWS = ('packet_direction', 'packet_native', 'tam_only', 'fusion')


class ViewGenerator(nn.Module):
    def __init__(self, previous_generator, view):
        super().__init__()
        self.view = view
        self.packet_conv = previous_generator.packet_conv
        self.time_conv = previous_generator.time_conv
        self.act = previous_generator.act
        self.mode = 'learned'
        if view == 'tam_only':
            self.packet_conv.requires_grad_(False)
        elif view != 'fusion':
            self.time_conv.requires_grad_(False)

    @property
    def active_token_keys(self):
        return ('packet','time') if self.view == 'fusion' else (('time',) if self.view == 'tam_only' else ('packet',))

    def forward(self, signed_timestamps, tam):
        if signed_timestamps.ndim != 2 or signed_timestamps.shape[1] != 5000:
            raise ValueError('signed_timestamps must be B x 5000')
        if tam.shape != (signed_timestamps.shape[0],2,1800):
            raise ValueError('tam must be B x 2 x 1800')
        b = signed_timestamps.shape[0]
        x = signed_timestamps.float()
        if self.view != 'tam_only':
            valid = x.ne(0)
            weight = valid.to(x.dtype)
            direction = x.sign()
            time = torch.zeros_like(x) if self.view == 'packet_direction' else x.abs().clamp(max=80)/80
            w = weight.reshape(b,100,5,10)
            d = direction.reshape(b,100,5,10)
            t = time.reshape(b,100,5,10)
            n = w.sum(-1).clamp_min(1)
            adjacent = w[..., :-1]*w[..., 1:]
            switches = (d[..., :-1] != d[..., 1:]).to(x.dtype)*adjacent
            stats = torch.stack([(d*w).sum(-1)/n, w.mean(-1), (t*w).sum(-1)/n,
                                 switches.sum(-1)/adjacent.sum(-1).clamp_min(1)],-1)
            p = torch.stack([direction,weight,time],1)
            for layer in self.packet_conv:
                p = self.act(layer(p))*weight[:,None,:]
            p = p.reshape(b,16,100,5,10)
            p = (p*w[:,None,...]).sum(-1)/n[:,None,...]
            learned = p.permute(0,2,3,1).reshape(b,100,80)
            packet = torch.cat([stats.reshape(b,100,20),learned],-1)
            packet_mask = valid.reshape(b,100,50).any(-1)
        else:
            packet = x.new_zeros(b,100,100)
            packet_mask = torch.zeros(b,100,dtype=torch.bool,device=x.device)
        if self.view in ('tam_only','fusion'):
            z = tam.float().log1p()
            stats = z.reshape(b,2,120,15).permute(0,2,1,3).reshape(b,120,30)
            for layer in self.time_conv:
                z = self.act(layer(z))
            learned = z.reshape(b,16,120,5,3).mean(-1).permute(0,2,3,1).reshape(b,120,80)
            temporal = torch.cat([stats,learned],-1)
            time_mask = torch.ones(b,120,dtype=torch.bool,device=x.device)
        else:
            temporal = x.new_zeros(b,120,110)
            time_mask = torch.zeros(b,120,dtype=torch.bool,device=x.device)
        return {'packet':packet,'time':temporal,'packet_mask':packet_mask,'time_mask':time_mask}


class MultiViewModel(BaseModel):
    def __init__(self,view,num_classes=102,dropout=.1):
        if view not in VIEWS:
            raise ValueError(view)
        # Original constructor exactly preserves the RNG stream and state_dict.
        super().__init__('learned','transformer',num_classes,dropout)
        self.view = view
        self.generator = ViewGenerator(self.generator,view)
        if view == 'tam_only':
            self.packet_projection.requires_grad_(False)
            self.packet_positions.requires_grad_(False)
        elif view != 'fusion':
            self.time_projection.requires_grad_(False)
            self.time_positions.requires_grad_(False)

    def forward_tokens(self,signed_timestamps,tam):
        generated = self.generator(signed_timestamps,tam)
        packet,temporal = generated['packet'],generated['time']
        p_valid,t_valid = generated['packet_mask'],generated['time_mask']
        p = self.packet_projection(packet)+self.packet_positions+self.view_types[0]
        t = self.time_projection(temporal)+self.time_positions+self.view_types[1]
        valid = torch.cat([p_valid,t_valid],1)
        padding = ~valid
        z = torch.cat([p,t],1).masked_fill(padding[...,None],0)
        # Empty packet-only input: a constant zero attention key avoids all-mask
        # NaNs. It never enters the readout, whose true masks remain all false.
        attention_padding = padding.clone()
        attention_padding[:,0] &= valid.any(1)
        for layer in self.head:
            z = layer(z,src_key_padding_mask=attention_padding)
            z = z.masked_fill(padding[...,None],0)
        z = self.final_norm(z).masked_fill(padding[...,None],0)
        return z[:,:100],z[:,100:],p_valid,t_valid
