# -*- coding: utf-8 -*-
"""Shared ordinary D3(BJ) and LASP-D3 backends for ASE and TorchSim."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DEFAULT_D3_PARAMETERS_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "dftd3_parameters.pt"
)
_D3_PARAMETER_KEYS = ("rcov", "r4r2", "c6ab", "cn_ref")
_LASP_BOHR_TO_ANGSTROM = 0.52917726
# Expose LASP's original 46.4758-Bohr default in Angstrom for both adapters.
DEFAULT_D3_CUTOFF_RADIUS = 46.4758 * _LASP_BOHR_TO_ANGSTROM


class LaspD3Correction:
    """Single-system CUDA D3; cutoff is in Angstrom, as in ordinary D3."""

    def __init__(self, *, device, dtype, cutoff, use_bj, functional_type, max_cache_size):
        self.device = device
        self.dtype = dtype
        self.cutoff = float(cutoff)
        self.use_bj = bool(use_bj)
        self.functional_type = int(functional_type)
        self.max_cache_size = int(max_cache_size)
        if self.max_cache_size < 1:
            raise ValueError("max_d3_cache_size must be at least 1.")
        if self.functional_type not in range(6):
            raise ValueError("LASP-D3 functional_type must be in [0, 5]: PBE, PBE0, B3LYP, BLYP, BP86, revPBE.")

        from mace_soyo.utils.d3_cffi import D3Calculator

        self._calculator_cls = D3Calculator
        self._cache = OrderedDict()

    def _get_calculator(self, elements):
        # A handle belongs to one element sequence, not one set of coordinates.
        key = (tuple(int(z) for z in elements), self.cutoff, self.use_bj, self.functional_type)
        calc = self._cache.get(key)
        if calc is not None:
            self._cache.move_to_end(key)
            return calc

        # LASP converts coordinates/cells internally, but expects cutoffs in Bohr.
        # Match the conversion constant used by its native library.
        cutoff_bohr = self.cutoff / _LASP_BOHR_TO_ANGSTROM
        calc = self._calculator_cls(
            elements=list(key[0]),
            max_length=len(key[0]),
            cutoff_radius=cutoff_bohr,
            cn_cutoff_radius=cutoff_bohr,
            damping_type=int(self.use_bj),
            functional_type=self.functional_type,
        )
        self._cache[key] = calc
        while len(self._cache) > self.max_cache_size:
            _, oldest = self._cache.popitem(last=False)
            oldest.close()
        return calc

    def compute(self, z, positions, cell):
        """Return E [1], F [N,3], stress [1,3,3] for a TTT row-vector cell."""
        with torch.cuda.device(self.device):
            calc = self._get_calculator(z.detach().cpu().tolist())
            # The native LASP-D3 ABI uses FP32 geometry and int64 elements.
            energy, forces, stress = calc.compute_torch(
                positions.to(dtype=torch.float32).contiguous(),
                z.to(dtype=torch.int64).contiguous(),
                cell.to(dtype=torch.float32).contiguous(),
            )
        return (energy.to(dtype=self.dtype), forces.to(dtype=self.dtype),
                stress.unsqueeze(0).to(dtype=self.dtype))

    def close(self):
        """Release cached handles; safe to call more than once."""
        with torch.cuda.device(self.device):
            for calc in self._cache.values():
                calc.close()
            self._cache.clear()


def resolve_d3_parameters_path(path: str | Path | None = None) -> Path:
    """Resolve the D3 reference-data file independently of the current cwd."""
    resolved = (
        DEFAULT_D3_PARAMETERS_PATH
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"D3 parameter file was not found: {resolved}. "
            "Pass d3_params_path=... explicitly if it is stored elsewhere."
        )
    return resolved


def load_d3_parameters(
    path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Load project D3 reference data as nvalchemiops ``D3Parameters``."""
    try:
        from nvalchemiops.torch.interactions.dispersion import D3Parameters
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TorchSim D3 requires nvalchemiops. Install TorchSim with its "
            "dispersion dependencies before setting use_d3=True."
        ) from exc

    parameter_path = resolve_d3_parameters_path(path)
    payload = torch.load(parameter_path, map_location="cpu", weights_only=True)

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected a mapping in D3 parameter file {parameter_path}, "
            f"got {type(payload).__name__}."
        )

    missing = [key for key in _D3_PARAMETER_KEYS if key not in payload]
    if missing:
        raise KeyError(
            f"D3 parameter file {parameter_path} is missing keys: {missing}."
        )

    params = D3Parameters(**{key: payload[key] for key in _D3_PARAMETER_KEYS})
    return params.to(device=device, dtype=dtype)


def make_torchsim_d3_model(
    *,
    a1: float,
    a2: float,
    s8: float,
    s6: float,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    d3_params_path: str | Path | None = None,
) -> Any:
    """Create TorchSim 0.6.x's D3(BJ) model with matching device/dtype."""
    try:
        from torch_sim.models.dispersion import D3DispersionModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "use_d3=True requires a TorchSim version that provides "
            "torch_sim.models.dispersion.D3DispersionModel."
        ) from exc

    params = load_d3_parameters(
        d3_params_path,
        device=device,
        dtype=dtype,
    )
    return D3DispersionModel(
        a1=float(a1),
        a2=float(a2),
        s8=float(s8),
        s6=float(s6),
        d3_params=params,
        cutoff=float(cutoff),
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )
