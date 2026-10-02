import torch

from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
from ta_wf_next.packet_patches import packet_patch_inputs
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views


def test_summary_collision_is_distinguished_and_views_unchanged():
    # Same mean and three transitions, but different within-patch ordering.
    directions = torch.tensor([[1, 1, -1, 1, -1, -1], [1, -1, -1, 1, 1, -1]], dtype=torch.float32)
    views = [generate_views(row.tolist(), input_kind="direction", budget=6, window_sizes=(6,))
             for row in directions]
    original = batch_from_views(views, packet_patch=6, max_runs=6)
    summary = packet_patch_inputs(original, directions, condition="summary", patch_size=6)
    ordered = packet_patch_inputs(original, directions, condition="ordered", patch_size=6)
    assert torch.equal(summary.packet[0], summary.packet[1])
    assert not torch.equal(ordered.packet[0], ordered.packet[1])
    assert torch.equal(ordered.packet[..., -6:], directions[:, None])
    assert ordered.runs is original.runs and ordered.windows is original.windows
    torch.manual_seed(7)
    a = HierarchicalViewTransformer(packet_dim=14, max_lengths=(1, 6, 1), dropout=0)
    torch.manual_seed(7)
    b = HierarchicalViewTransformer(packet_dim=14, max_lengths=(1, 6, 1), dropout=0)
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())
    logits = b(ordered)
    logits.square().sum().backward()
    assert torch.isfinite(logits).all()
    assert b.projections["packet"].weight.grad[:, -6:].abs().sum() > 0


def test_partial_patch_mask_and_batch_independence():
    directions = torch.tensor([[1, -1, 1, 0, 0, 0], [-1, 1, -1, 1, -1, 1]], dtype=torch.float32)
    views = [generate_views(row.tolist(), input_kind="direction", budget=6, window_sizes=(3,))
             for row in directions]
    batch = batch_from_views(views, packet_patch=3, max_runs=6)
    result = packet_patch_inputs(batch, directions, condition="ordered", patch_size=3)
    alone = packet_patch_inputs(batch_from_views(views[:1], packet_patch=3, max_runs=6),
                                directions[:1], condition="ordered", patch_size=3)
    assert torch.equal(result.packet[:1], alone.packet)
    assert torch.equal(result.packet[0, 1], torch.zeros(8))
    assert result.packet_mask[0].tolist() == [True, False]
    changed = directions.clone()
    changed[0, 4] = 1
    try:
        packet_patch_inputs(batch, changed, condition="ordered", patch_size=3)
    except ValueError as error:
        assert "internal zero" in str(error)
    else:
        raise AssertionError("internal padding should be rejected")
