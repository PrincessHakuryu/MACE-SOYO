# -*- coding: utf-8 -*-
"""
AOTI-only ASE calculator for no-q MACESoyo packages.

Runtime boundary:
    ASE/Python side:
        atoms -> z, frac_pos, cell
        PBC neighbor list -> edge_index, image
        Single-structure ASE calls provide batch = zeros([N]) and cell = [1, 3, 3].

    AOTI .pt2 package:
        New: (z, frac_pos, cell, batch, edge_index, image, head_id, pbc) -> E/F/S
        Conditioned packages append charge/spin and masks read from atoms.info.
        Legacy six-input single-head packages remain supported.

This runtime calculator intentionally does NOT load a training checkpoint and does
NOT construct MACESoyo. The checkpoint is only used by
mace_soyo/export/AOTI_export.py when
building/checking the .pt2 package.

Runtime metadata such as cutoff/ABI is embedded inside the .pt2 package via
the AOTInductor config key ``aot_inductor.metadata``.  No sidecar JSON file is
required. The neighbor cutoff is read from the package and cannot be overridden.
Packages without embedded cutoff metadata must be re-exported.

This runtime intentionally supports only the new batched AOTI ABI. Old 5-input
packages should be re-exported with the batched exporter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace_soyo.utils.neighbors import build_batched_neighbor_list

from mace_soyo.inference.dispersion import (
    DEFAULT_D3_CUTOFF_RADIUS, LaspD3Correction, make_torchsim_d3_model,
)
from mace_soyo.inference.aoti_utils import (
    select_head, load_metadata, register_custom_ops, resolve_torch_dtype,
    SPIN_CHARGE_INPUTS, spin_charge_from_atoms,
)


from mace_soyo.utils.geometry import atoms_geometry
from mace_soyo.utils.device import resolve_cuda_device


def _maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_neighbor_list_gpu_pbc(
    frac_pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    pbc: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build PyG-style PBC neighbor list via nvalchemiops."""
    pos = torch.bmm(frac_pos.unsqueeze(0), cell.unsqueeze(0)).squeeze(0)
    batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    pbc_tensor = torch.tensor([list(bool(x) for x in pbc)], dtype=torch.bool, device=pos.device)
    lattice = cell.unsqueeze(0)
    
    return build_batched_neighbor_list(pos, lattice, batch, pbc_tensor, cutoff)


class MACESoyoCalculator(Calculator):
    """ASE calculator backed only by a precompiled no-q MACESoyo AOTI .pt2 package."""

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]
    _neighbor_list_fn = staticmethod(get_neighbor_list_gpu_pbc)

    def __init__(
        self,
        package_path: str,
        device: str = "cuda:0",
        profile: bool = False,
        use_d3: bool = False,
        a1: float = 0.4289,
        a2: float = 4.4407,
        s8: float = 0.7875,
        s6: float = 1.0,
        d3_cutoff_radius: float = DEFAULT_D3_CUTOFF_RADIUS,
        d3_params_path: str | Path | None = None,
        head: str | None = None,
        use_laspd3: bool = False,
        use_laspd3_BJ: bool = False,
        functional_type: int = 0,
        max_d3_cache_size: int = 16,
    ):
        super().__init__()
        self.package_path = str(package_path)
        self.device = resolve_cuda_device(device)
        self.profile = bool(profile)
        self.use_d3 = bool(use_d3)
        self.use_laspd3 = bool(use_laspd3)
        if self.use_d3 and self.use_laspd3:
            raise ValueError("use_d3 and use_laspd3 are mutually exclusive.")
        if use_laspd3_BJ and not self.use_laspd3:
            raise ValueError("use_laspd3_BJ=True requires use_laspd3=True.")

        register_custom_ops()
        with torch.cuda.device(self.device):
            self.aoti_model = torch._inductor.aoti_load_package(
                self.package_path, device_index=self.device.index,
            )
        self.metadata = load_metadata(self.aoti_model)
        # AOTI fixes the input precision at export time; it is not a runtime option.
        self.dtype = resolve_torch_dtype(self.metadata["mace_soyo_aoti_dtype"])

        self.laspd3_model = None
        if self.use_laspd3:
            self.laspd3_model = LaspD3Correction(
                device=self.device,
                dtype=self.dtype,
                cutoff=d3_cutoff_radius,
                use_bj=use_laspd3_BJ,
                functional_type=functional_type,
                max_cache_size=max_d3_cache_size,
            )
        if self.use_d3:
            self.d3model = make_torchsim_d3_model(
                a1=a1,
                a2=a2,
                s8=s8,
                s6=s6,
                cutoff=d3_cutoff_radius,
                device=self.device,
                dtype=self.dtype,
                d3_params_path=d3_params_path,
            )

        self.head_names, self._head_id, self._dynamic_head = select_head(self.metadata, head)
        self._head_tensor = torch.tensor([self._head_id], device=self.device, dtype=torch.long)
        self._conditioned = tuple(self.metadata.get("mace_soyo_aoti_inputs", "").split()) == SPIN_CHARGE_INPUTS

        cutoff = self.metadata.get("mace_soyo_aoti_cutoff")
        if cutoff is None:
            raise ValueError(
                "The .pt2 package is missing embedded cutoff metadata. "
                "Re-export with mace_soyo.export.AOTI_export."
            )
        self.cutoff = float(cutoff)

        print(f"[MACESoyoCalculator/AOTI] loaded package = {self.package_path}")
        print(f"[MACESoyoCalculator/AOTI] device = {self.device}")
        print(f"[MACESoyoCalculator/AOTI] dtype = {self.dtype}")
        print(f"[MACESoyoCalculator/AOTI] cutoff = {self.cutoff}")
        if self.use_d3:
            print("[MACESoyoCalculator/D3] backend = TorchSim D3(BJ)")
        if self.use_laspd3:
            print(f"[MACESoyoCalculator/D3] backend = LASP-D3; BJ = {bool(use_laspd3_BJ)}")
        print(f"[MACESoyoCalculator/AOTI] heads = {list(self.head_names)}; selected = {self.head!r}")
        print(f"[MACESoyoCalculator/AOTI] ABI = {self.metadata.get('mace_soyo_aoti_inputs', 'legacy six-input')} -> (energy, forces, stress)")

    @property
    def head(self) -> str:
        """Head selected at construction; create a new calculator to change it."""
        return self.head_names[self._head_id]

    def close(self):
        """Release cached LASP-D3 handles after optimization or MD."""
        if self.laspd3_model is not None:
            self.laspd3_model.close()

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        # ASE compares geometry/arrays, but not atoms.info. Reuse its cached
        # Atoms snapshot to detect electronic-state changes as well.
        if self.atoms is not None:
            for key in ("charge", "spin"):
                if atoms.info.get(key) != self.atoms.info.get(key):
                    changes.append(key)
        return changes

    def _atoms_to_inputs(self, atoms):
        frac, complete_cell, axes = atoms_geometry(atoms)

        z = torch.as_tensor(
            np.asarray(atoms.numbers, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        frac_pos = torch.as_tensor(
            frac,
            dtype=self.dtype,
            device=self.device,
        )
        cell = torch.as_tensor(
            complete_cell,
            dtype=self.dtype,
            device=self.device,
        )

        pbc = tuple(bool(x) for x in axes)

        with torch.cuda.device(self.device):
            edge_index, image = self._neighbor_list_fn(
                frac_pos=frac_pos,
                cell=cell,
                cutoff=self.cutoff,
                pbc=pbc,
            )
        # Zero edges are valid for isolated atoms or atoms beyond the cutoff.
        # Keep every atom and pass the empty neighbor list to the AOTI model.

        # ASE is a single-structure runtime path. The new AOTI ABI is batched,
        # so every atom belongs to graph 0.
        batch = torch.zeros(z.shape[0], dtype=torch.long, device=self.device)

        return z, frac_pos, cell, batch, edge_index.long(), image.long()

    @staticmethod
    def _stress_3x3_to_voigt6(stress33):
        return np.array(
            [
                stress33[0, 0],
                stress33[1, 1],
                stress33[2, 2],
                stress33[1, 2],
                stress33[0, 2],
                stress33[0, 1],
            ],
            dtype=np.float64,
        )

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if not self._conditioned and all(self.atoms.info.get(k) is not None for k in ("charge", "spin")):
            raise ValueError("This package has no charge/spin inputs. Export a use_spin_charge=True model.")
        if self.use_laspd3 and not self.atoms.pbc.all():
            raise ValueError("LASP-D3 only supports fully periodic (TTT) systems; use ordinary D3 for other PBC.")

        if self.profile:
            _maybe_sync(self.device)
            t0 = time.perf_counter()

        z, frac_pos, cell, batch, edge_index, image = self._atoms_to_inputs(self.atoms)
        ecount = int(edge_index.shape[1])

        # New batched AOTI ABI expects cell as [B, 3, 3]. ASE has B=1.
        cell_batched = cell.unsqueeze(0) if cell.dim() == 2 else cell

        args = (
            z.contiguous(),
            frac_pos.contiguous(),
            cell_batched.contiguous(),
            batch.contiguous(),
            edge_index.contiguous(),
            image.contiguous(),
        )
        pbc = torch.tensor([self.atoms.pbc.tolist()], dtype=torch.bool, device=self.device)
        if self._dynamic_head:
            args += (self._head_tensor, pbc)
        if self._conditioned:
            args += spin_charge_from_atoms([self.atoms], device=self.device, dtype=self.dtype)

        if self.profile:
            _maybe_sync(self.device)
            t1 = time.perf_counter()

        with torch.cuda.device(self.device), torch.inference_mode():
            energy, force, stress = self.aoti_model(*args)

        if self.use_laspd3:
            d3_energy, d3_forces, d3_stress = self.laspd3_model.compute(
                z, frac_pos @ cell, cell,
            )
            energy = energy.reshape(-1) + d3_energy.reshape(-1)
            force = force + d3_forces
            stress = stress.reshape(-1, 3, 3) + d3_stress
        elif self.use_d3:
            import torch_sim as ts

            d3_atoms = self.atoms.copy()
            d3_atoms.set_cell(cell.detach().cpu().numpy(), scale_atoms=False)
            d3_state = ts.initialize_state(d3_atoms, self.device, self.dtype)
            with torch.cuda.device(self.device), torch.inference_mode():
                d3_results = self.d3model(d3_state)
            energy = energy.reshape(-1) + d3_results["energy"].reshape(-1)
            force = force + d3_results["forces"]
            stress = stress.reshape(-1, 3, 3) + d3_results["stress"].reshape(-1, 3, 3)

        # No stress prediction is advertised for a non-TTT input, regardless of
        # the head's training dataset or optional dispersion backend.
        stress = torch.where(pbc.all(dim=-1)[:, None, None], stress.reshape(-1, 3, 3), 0.0)

        if self.profile:
            _maybe_sync(self.device)
            t2 = time.perf_counter()

        energy_scalar = float(energy.reshape(-1)[0].detach().cpu())
        forces_np = force.detach().cpu().numpy()
        stress33 = stress.reshape(-1, 3, 3)[0].detach().cpu().numpy()
        stress6 = self._stress_3x3_to_voigt6(stress33)

        self.results["energy"] = energy_scalar
        self.results["free_energy"] = energy_scalar
        self.results["forces"] = forces_np
        self.results["stress"] = stress6

        if self.profile:
            t3 = time.perf_counter()
            print(f"edges={ecount:7d}  prep={t1-t0:7.3f}s  model={t2-t1:7.3f}s  post={t3-t2:7.3f}s")
