# -*- coding: utf-8 -*-
"""Shared helpers for the TorchSim D3(BJ) dispersion backend."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DEFAULT_D3_PARAMETERS_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "dftd3_parameters.pt"
)
_D3_PARAMETER_KEYS = ("rcov", "r4r2", "c6ab", "cn_ref")


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
