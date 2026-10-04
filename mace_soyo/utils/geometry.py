"""CPU geometry shared by ASE input and dataset workers (no GPU transfers)."""
import numpy as np


def atoms_geometry(atoms, *, pbc=None):
    """Return float64 fractional positions, a complete row cell and PBC.

    Only periodic axes are wrapped. Missing nonperiodic vectors are completed
    as a coordinate basis, not as a physical periodic simulation box.
    TorchSim already has device tensors and keeps its separate Torch path.
    """
    positions = np.asarray(atoms.positions, dtype=np.float64)
    original = np.asarray(atoms.cell, dtype=np.float64)
    axes = np.asarray(atoms.pbc if pbc is None else pbc, dtype=bool)
    if not np.isfinite(positions).all() or not np.isfinite(original).all():
        raise ValueError("Positions and cell must be finite.")
    if np.any(np.linalg.norm(original, axis=1)[axes] < 1e-10):
        raise ValueError("A periodic axis has a missing/zero cell vector.")
    cell = np.asarray(atoms.cell.complete(), dtype=np.float64)
    if abs(np.linalg.det(cell)) < 1e-10:
        raise ValueError("Cell vectors are linearly dependent.")
    frac = np.linalg.solve(cell.T, positions.T).T
    frac[:, axes] %= 1.0
    return frac, cell, axes
