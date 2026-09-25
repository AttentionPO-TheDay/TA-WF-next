import unittest

import torch

from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import (
    GeneratorTokenTransformer,
    batch_from_views,
    confident_teacher_kl,
    freeze_except_adapter,
    mask_observation_spans,
    pretrain_step,
    finetune_step,
    tta_adapter_step,
    view_consistency_loss,
)


class TransformerPrototypeTest(unittest.TestCase):
    def setUp(self):
        rows = [
            [1, 1, -1, -1, 1, -1, 1, 1, 1, -1] * 20,
            [-1, -1, 1, 1, -1, 1, -1, -1, 1, 1] * 20,
        ]
        views = tuple(generate_views(row, input_kind="direction", budget=len(row), window_sizes=(5, 10)) for row in rows)
        self.batch = batch_from_views(views, packet_patch=5, max_runs=32)

    def test_batch_and_forward(self):
        model = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        output = model(self.batch)
        self.assertEqual(tuple(output["logits"].shape), (2, 3))
        self.assertEqual(output["hidden"].shape[1], 1 + self.batch.packet.shape[1] + self.batch.runs.shape[1] + self.batch.windows.shape[1])

    def test_full_observation_budget_fits(self):
        row = [1, -1] * 2500
        view = generate_views(row, input_kind="direction", budget=5000, window_sizes=(50, 250))
        batch = batch_from_views((view,), packet_patch=50, max_runs=128)
        model = GeneratorTokenTransformer(num_classes=3)
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(tuple(out["logits"].shape), (1, 3))
        self.assertLessEqual(out["hidden"].shape[1], model.max_tokens)

    def test_encoding_is_batch_independent(self):
        short = [1, 1, -1, -1] * 10 + [0] * 160
        long = [1, -1] * 100
        views = tuple(generate_views(row, input_kind="direction", budget=200, window_sizes=(5, 10)) for row in (short, long))
        one = batch_from_views(views[:1], packet_patch=5, max_runs=32)
        both = batch_from_views(views, packet_patch=5, max_runs=32)
        model = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3).eval()
        with torch.no_grad():
            first = model(one)["logits"]
            second = model(both)["logits"][:1]
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_masked_loss_and_gradient(self):
        model = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        masked_batch, masked = mask_observation_spans(self.batch, torch.tensor([2, 30]), torch.tensor([3, 4]))
        output = model(masked_batch)
        loss = model.masked_reconstruction_loss(output, self.batch, masked)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.projections["packet"].weight.grad)

    def test_pretrain_and_finetune_steps(self):
        model = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        record = pretrain_step(model, self.batch, opt, span_length=3)
        self.assertGreater(record["masked_tokens"], 0)
        record = finetune_step(model, self.batch, torch.tensor([0, 1]), opt)
        self.assertTrue(torch.isfinite(torch.tensor(record["loss"])))

    def test_span_masking(self):
        starts = torch.tensor([2, 30])
        lengths = torch.tensor([3, 4])
        masked_batch, selected = mask_observation_spans(self.batch, starts, lengths)
        self.assertTrue(selected["packet"].any())
        self.assertTrue(torch.equal(masked_batch.packet_mask, self.batch.packet_mask))
        for name in ("packet", "runs", "windows"):
            bounds = self.batch.spans[name]
            overlap = (bounds[..., 0] < (starts + lengths)[:, None]) & (bounds[..., 1] > starts[:, None])
            torch.testing.assert_close(selected[name], overlap & getattr(self.batch, f"{name}_mask"))
            self.assertTrue(torch.all(getattr(masked_batch, name)[selected[name]] == 0))

    def test_adapter_and_tta_helpers(self):
        model = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        freeze_except_adapter(model)
        self.assertTrue(all(p.requires_grad == ("adapter" in n) for n, p in model.named_parameters()))
        value = torch.randn(2, 48)
        torch.testing.assert_close(model.adapter(value), value, rtol=0, atol=0)
        student = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        teacher = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        loss, selected = confident_teacher_kl(student, teacher, threshold=0.8)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(selected[0])
        self.assertFalse(selected[1])
        self.assertTrue(torch.isfinite(view_consistency_loss(torch.ones(2, 4), torch.ones(2, 4))))
        teacher = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        student = GeneratorTokenTransformer(max_tokens=256, d_model=48, nhead=4, num_classes=3)
        student.load_state_dict(teacher.state_dict())
        freeze_except_adapter(student)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
        student_before = {k: v.detach().clone() for k, v in student.state_dict().items()}
        perturbed, _ = mask_observation_spans(self.batch, torch.tensor([0, 8]), torch.tensor([5, 5]))
        tta_opt = torch.optim.AdamW(student.adapter.parameters(), lr=1e-3)
        record = tta_adapter_step(student, teacher, self.batch, perturbed, tta_opt, confidence_threshold=0.3)
        self.assertIn("teacher_selected", record)
        for k, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[k], rtol=0, atol=0)
        for k, value in student.state_dict().items():
            if not k.startswith("adapter."):
                torch.testing.assert_close(value, student_before[k], rtol=0, atol=0)
        self.assertTrue(any(
            not torch.equal(value, student_before[k])
            for k, value in student.state_dict().items() if k.startswith("adapter.")
        ))


if __name__ == "__main__":
    unittest.main()
