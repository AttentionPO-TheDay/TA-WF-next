import torch

from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
from ta_wf_next.transformer_proto import GeneratorTokenBatch


def make_batch() -> GeneratorTokenBatch:
    packet = torch.tensor([[[1.0, 0.0], [-1.0, 1.0], [0.0, 0.0]]])
    runs = torch.tensor([[[1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    windows = torch.tensor([[[0.5, 0.1, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    return GeneratorTokenBatch(
        packet, torch.tensor([[True, True, False]]),
        runs, torch.tensor([[True, False]]),
        windows, torch.tensor([[True, False]]),
        {"packet": torch.zeros(1, 3, 2, dtype=torch.long),
         "runs": torch.zeros(1, 2, 2, dtype=torch.long),
         "windows": torch.zeros(1, 2, 2, dtype=torch.long)},
    )


def test_shape_gradients_and_padding_isolation():
    torch.manual_seed(7)
    model = HierarchicalViewTransformer(max_lengths=(3, 2, 2), dropout=0.0).eval()
    batch = make_batch()
    output = model(batch)
    assert output.shape == (1, 102)
    output.sum().backward()
    assert model.local_encoders["packet"].layers[0].self_attn.in_proj_weight.grad is not None
    altered = GeneratorTokenBatch(
        batch.packet.clone(), batch.packet_mask, batch.runs.clone(), batch.runs_mask,
        batch.windows.clone(), batch.windows_mask, batch.spans,
    )
    altered.packet[0, 2] = 999
    altered.runs[0, 1] = 999
    altered.windows[0, 1] = 999
    assert torch.allclose(output.detach(), model(altered).detach(), atol=1e-5, rtol=1e-5)


def test_missing_views_do_not_affect_logits():
    model = HierarchicalViewTransformer(max_lengths=(3, 2, 2), dropout=0.0).eval()
    batch = make_batch()
    missing = GeneratorTokenBatch(
        batch.packet, batch.packet_mask, batch.runs, torch.zeros_like(batch.runs_mask),
        batch.windows, torch.zeros_like(batch.windows_mask), batch.spans,
    )
    changed = GeneratorTokenBatch(
        batch.packet, batch.packet_mask, batch.runs + 50, missing.runs_mask,
        batch.windows - 50, missing.windows_mask, batch.spans,
    )
    assert torch.allclose(model(missing), model(changed), atol=1e-5, rtol=1e-5)
