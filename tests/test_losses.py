import torch

from aethercell.losses import AetherCellLoss, SpecificityReference, topk_directional_loss


def test_directional_loss_is_zero_for_identical_changes():
    delta = torch.tensor([[3.0, -2.0, 1.0, 0.0], [-1.0, 4.0, 2.0, -3.0]])
    assert torch.allclose(topk_directional_loss(delta, delta, k=4), torch.tensor(0.0), atol=1e-6)


def test_specificity_reference_uses_leave_one_out_and_global_fallback():
    deltas = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0]])
    reference = SpecificityReference.from_batches([(deltas, ["A", "A", "B"])])
    background = reference.background(deltas, ["A", "A", "B"], leave_one_out=True)
    assert torch.allclose(background[0], deltas[1])
    assert torch.allclose(background[1], deltas[0])
    assert torch.allclose(background[2], deltas.mean(0))


def test_complete_objective_backpropagates():
    true_z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference = SpecificityReference.from_batches([(true_z, ["A", "A"])])
    objective = AetherCellLoss(reference, top_k=3)
    prediction = torch.randn(2, 4, requires_grad=True)
    pred_z = torch.randn(2, 2, requires_grad=True)
    total, parts = objective(
        prediction=prediction,
        target=torch.randn(2, 4),
        control=torch.randn(2, 4),
        delta_z_pred=pred_z,
        delta_z_true=true_z,
        cell_ids=["A", "A"],
    )
    total.backward()
    assert prediction.grad is not None
    assert pred_z.grad is not None
    assert set(parts) == {"reconstruction", "directional", "weighted_mse", "latent_alignment", "specificity"}
