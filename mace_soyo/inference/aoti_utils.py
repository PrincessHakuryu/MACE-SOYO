"""Runtime-only AOTI head/geometry contract; never imports the training model."""

from __future__ import annotations

import json

import numpy as np
import torch

LEGACY_INPUTS = ("z", "frac_pos", "cell", "batch", "edge_index", "image")
MULTIHEAD_INPUTS = LEGACY_INPUTS + ("head_id", "pbc")
SPIN_CHARGE_INPUTS = MULTIHEAD_INPUTS + ("charge", "spin", "condition_mask")


def spin_charge_from_atoms(atoms_list, *, device, dtype):
    """Read graph labels from ASE info; None is missing, charge=0 is present."""
    values, masks = [], []
    for name, default in (("charge", 0), ("spin", 1)):
        column, mask = [], []
        for atoms in atoms_list:
            value = atoms.info.get(name)
            present = value is not None
            if present:
                value = float(value)
                if not np.isfinite(value) or not value.is_integer() or (name == "spin" and value < 1):
                    raise ValueError(f"Invalid molecular {name}: {value}")
                lower = -100 if name == "charge" else 1
                assert lower <= value <= 100, f"atoms.info[{name!r}] must be in [{lower}, 100], got {value}."
            column.append(value if present else default)
            mask.append(present)
        values.append(torch.tensor(column, device=device, dtype=dtype))
        masks.append(torch.tensor(mask, device=device, dtype=torch.bool))
    return (*values, masks[0] & masks[1])


def select_head(metadata, head=None):
    """Resolve one head name for legacy, multihead, or conditioned packages."""
    inputs = tuple(metadata.get("mace_soyo_aoti_inputs", " ".join(LEGACY_INPUTS)).split())
    if inputs not in (LEGACY_INPUTS, MULTIHEAD_INPUTS, SPIN_CHARGE_INPUTS):
        raise ValueError(f"Unsupported AOTI inputs: {inputs}. Re-export this model.")
    dynamic = inputs != LEGACY_INPUTS
    if dynamic:
        names = tuple(json.loads(metadata["mace_soyo_aoti_head_names"]))
        count = int(metadata["mace_soyo_aoti_num_heads"])
    else:
        names = tuple(json.loads(metadata.get("mace_soyo_aoti_head_names", '["head_0"]')))
        count = 1
    if (len(names) != count or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("AOTI head names/count are inconsistent.")
    if head is None and len(names) == 1:
        head = names[0]
    if head not in names:
        raise ValueError(f"Specify a valid head; got {head!r}. Available heads: {list(names)}")
    return names, names.index(head), dynamic


def register_custom_ops():
    """Importing the extension registers the operators referenced by .pt2."""
    import cuequivariance_torch  # noqa: F401


def resolve_torch_dtype(dtype):
    if dtype in (torch.float32, "float32"):
        return torch.float32
    if dtype in (torch.float64, "float64"):
        return torch.float64
    raise ValueError(f"dtype must be float32 or float64, got {dtype!r}")


def load_metadata(model):
    """Current PyTorch API; errors are not swallowed."""
    metadata = dict(model.get_metadata())
    # Translate metadata from previously exported packages in one place.
    for key, value in list(metadata.items()):
        if key.startswith("nagasaki_aoti_"):
            metadata.setdefault(key.replace("nagasaki_aoti_", "mace_soyo_aoti_", 1), value)
            del metadata[key]
    return metadata


def normalize_pbc(pbc, num_graphs):
    if pbc.shape == (3,) or pbc.shape == (1, 3):
        return pbc.reshape(1, 3).expand(num_graphs, 3).contiguous()
    if pbc.shape != (num_graphs, 3):
        raise ValueError(f"PBC must be [3] or [{num_graphs}, 3], got {tuple(pbc.shape)}.")
    return pbc.contiguous()


def complete_runtime_cells(cell, pbc):
    """Fast normal-cell path; complete exceptional missing nonperiodic vectors."""
    missing = torch.linalg.vector_norm(cell, dim=-1) < 1e-10
    if bool((missing & pbc).any()):
        raise ValueError("A periodic axis has a missing/zero cell vector.")
    # The usual cell-less molecular case is completed entirely on the device.
    empty = missing.all(dim=-1) & ~pbc.any(dim=-1)
    cell = torch.where(empty[:, None, None], torch.eye(3, device=cell.device, dtype=cell.dtype), cell)
    if bool((missing & ~empty[:, None]).any()):
        from ase.cell import Cell
        arrays = cell.detach().cpu().numpy()
        cell = torch.as_tensor(np.stack([Cell(c).complete().array for c in arrays]),
                               device=cell.device, dtype=cell.dtype)
    return cell.contiguous()
