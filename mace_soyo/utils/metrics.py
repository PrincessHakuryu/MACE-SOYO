"""Epoch metrics: named FP64 sums per head, reduced once across ranks."""

import math

import torch
import torch.distributed as dist

from mace_soyo.utils.loss import stress_label_mask


class EpochMetrics:
    def __init__(self, device, head_names):
        self.head_names = list(head_names)
        names = (
            "energy_squared_error", "energy_absolute_error",
            "energy_per_atom_squared_error", "energy_per_atom_absolute_error",
            "force_squared_error", "force_absolute_error",
            "stress_squared_error", "stress_absolute_error",
            "stress_per_atom_squared_error", "stress_per_atom_absolute_error",
            "structures", "force_components", "stress_components",
        )
        self.sums = {
            name: torch.zeros(len(head_names), dtype=torch.float64, device=device)
            for name in names
        }

    @torch.no_grad()
    def update(self, data, predictions):
        energy, force, stress = (value.detach().double() for value in predictions)
        energy_error = energy.reshape(-1) - data.energy.double().reshape(-1)
        atom_counts = torch.bincount(data.batch, minlength=energy.numel())
        energy_per_atom_error = energy_error / atom_counts
        force_error = force - data.force.double()
        # Missing labels may contain NaNs; select valid structures before arithmetic.
        stress_valid = stress_label_mask(data)
        graph_heads = data.head_id.reshape(-1)
        atom_heads = graph_heads[data.batch]
        for head_id in range(len(self.head_names)):
            graphs = graph_heads == head_id
            atoms = atom_heads == head_id
            stress_graphs = graphs & stress_valid
            stress_error = (stress[stress_graphs] - data.stress.double()[stress_graphs]) * 1000
            stress_per_atom_error = stress_error / atom_counts[stress_graphs, None, None]
            errors = {
                "energy": energy_error[graphs],
                "energy_per_atom": energy_per_atom_error[graphs],
                "force": force_error[atoms],
                "stress": stress_error,
                "stress_per_atom": stress_per_atom_error,
            }
            for name, error in errors.items():
                self.sums[name + "_squared_error"][head_id] += error.square().sum()
                self.sums[name + "_absolute_error"][head_id] += error.abs().sum()
            self.sums["structures"][head_id] += graphs.sum()
            self.sums["force_components"][head_id] += atoms.sum() * 3
            self.sums["stress_components"][head_id] += stress_graphs.sum() * 9

    def reduce(self):
        # Tensor packing is only a transport detail at this communication boundary.
        names = list(self.sums)
        packed = torch.stack([self.sums[name] for name in names])
        dist.all_reduce(packed)
        self.sums = dict(zip(names, packed.unbind()))

    def compute(self):
        values = {name: tensor.cpu().tolist() for name, tensor in self.sums.items()}
        overall = {name: sum(per_head) for name, per_head in values.items()}
        heads = {
            head: _metric_report({name: per_head[h] for name, per_head in values.items()})
            for h, head in enumerate(self.head_names)
        }
        return {"overall": _metric_report(overall), "heads": heads}


def _metric_report(sums):
    report = {
        "structures": int(sums["structures"]),
        "stress_structures": int(sums["stress_components"] / 9),
    }
    for name, count_name in (
        ("energy", "structures"), ("energy_per_atom", "structures"),
        ("force", "force_components"), ("stress", "stress_components"),
        ("stress_per_atom", "stress_components"),
    ):
        count = max(sums[count_name], 1)
        report[name + "_rmse"] = math.sqrt(sums[name + "_squared_error"] / count)
        report[name + "_mae"] = sums[name + "_absolute_error"] / count
    return report


def log_epoch_metrics(log, report, *, phase, epoch, score, lr, elapsed, is_best):
    """Format reports only; checkpoint decisions belong to the training loop."""
    for head_id, (name, values) in enumerate(report["heads"].items()):
        energy = f"{values['energy_per_atom_mae']:.6f}" if values["structures"] else "N/A"
        force = f"{values['force_mae']:.6f}" if values["structures"] else "N/A"
        stress = f"{values['stress_mae']:.6f}" if values["stress_structures"] else "N/A"
        log.info(
            f"[{phase}] [HEAD {head_id}:{name}] [epoch] {epoch} "
            f"[structures] {values['structures']} [MAE E eV/atom] {energy} "
            f"[F eV/A] {force} [stress meV/A3] {stress} "
            f"[stress structures] {values['stress_structures']}"
        )
    values = report["overall"]
    best_marker = " [BEST SAVE]" if is_best else ""
    for metric, label in (("rmse", "RMSE"), ("mae", "MAE ")):
        message = (
            f"[{phase}] [EPOCH] {epoch} [{label}] [energy] [mol] {values['energy_' + metric]:.6f} "
            f"[atom] {values['energy_per_atom_' + metric]:.6f} [force] {values['force_' + metric]:.5f} "
            f"[stress_meV_A3_atom&meV_A3] {values['stress_per_atom_' + metric]:.6f} "
            f"{values['stress_' + metric]:.6f}"
        )
        if metric == "rmse":
            message += f" [time] {elapsed}" + best_marker
        log.info(message)
    log.info(
        f"[{phase}] [EPOCH] {epoch} [SAVE SCORE] [weighted_{phase.lower()}_loss] "
        f"{score:.8f} [lr] {lr:.8e}" + best_marker
    )
