"""Local FP64 E/F/stress loss sums and label counts; no distributed communication."""

import torch


def stress_label_mask(data):
    return data.stress_mask.reshape(-1).bool() & data.pbc.reshape(-1, 3).bool().all(-1)


def efs_loss(predictions, data, criterion):
    """Compute all three losses in FP64, independently of the network dtype.

    Cast gradients return to the predictions' original dtype automatically.
    Return differentiable sums and label counts, ordered as energy/force/stress.
    The training loop normalizes the sums using global label counts.

    criterion must use reduction='none'. Mask BEFORE evaluating the stress
    criterion: NaN * 0 is not a valid missing-label implementation.
    """
    if criterion.reduction != "none":
        raise ValueError("efs_loss requires criterion with reduction='none'.")
    energy, force, stress = predictions
    energy = energy.double()
    num_graphs = energy.numel()
    num_atoms = torch.bincount(data.batch, minlength=num_graphs).to(energy.dtype)
    e = criterion(energy.reshape(-1) / num_atoms, data.energy.double().reshape(-1) / num_atoms)
    f = criterion(force.double(), data.force.double())
    mask = stress_label_mask(data)[:, None, None]
    safe_pred = torch.where(mask, stress, torch.zeros_like(stress))
    safe_true = torch.where(mask, data.stress, torch.zeros_like(data.stress))
    s = criterion(safe_pred.double(), safe_true.double())
    sums = torch.stack((e.sum(), f.sum(), s.sum()))
    counts = torch.stack((energy.new_tensor(e.numel()), energy.new_tensor(f.numel()),
                          mask.sum().to(energy.dtype) * 9))
    return sums, counts


def weighted_loss_score(loss_sums, label_counts, factors):
    """Compute a weighted score after both sums and counts have been reduced."""
    means = loss_sums / label_counts.clamp_min(1)
    return float((means * means.new_tensor(factors)).sum().item())
