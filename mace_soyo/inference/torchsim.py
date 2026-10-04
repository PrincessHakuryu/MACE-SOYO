# -*- coding: utf-8 -*-
"""TorchSim adapter for batched no-q MACESoyo AOTI packages.

AOTI ABI:
    New: (z, frac_pos, cell, batch, edge_index, image, head_id, pbc) -> E/F/S.
    Conditioned packages append charge, spin, condition_mask from state.
    Legacy six-input single-head packages remain supported.
    One named head per model instance applies to every system, including systems
    inserted/reordered by TorchSim autobatching.

TorchSim side:
    SimState.positions are Cartesian coordinates.
    SimState.cell uses column-vector convention internally.
    SimState.row_vector_cell is the row-vector cell used by ASE/MACESoyo.

MACESoyo side:
    z is natural 1-based atomic number: H=1, He=2, ...
    frac_pos uses row-vector convention: cart = frac @ cell.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch_sim as ts
from mace_soyo.utils.neighbors import build_batched_neighbor_list
from torch_sim.models.interface import ModelInterface
from mace_soyo.utils.device import resolve_cuda_device

from mace_soyo.inference.dispersion import make_torchsim_d3_model
from mace_soyo.inference.aoti_utils import (
    select_head, load_metadata, register_custom_ops, resolve_torch_dtype, complete_runtime_cells, normalize_pbc,
    SPIN_CHARGE_INPUTS, spin_charge_from_atoms,
)

ELECTRONIC_STATE_KEYS = ("charge", "spin", "condition_mask")


def atoms_to_state(atoms, device=None, dtype=None):
    """ASE -> TorchSim, retaining per-system charge/spin and missing-label masks.

    Use this for ALL structures, including unlabelled structures and later
    replacements. system_extras follow slicing, splitting, and concatenation.
    The standard ts.initialize_state(atoms) does not read these info fields.
    """
    from ase import Atoms

    atoms_list = [atoms] if isinstance(atoms, Atoms) else list(atoms)
    device = resolve_cuda_device(device)
    state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
    values = spin_charge_from_atoms(atoms_list, device=state.device, dtype=state.dtype)
    state.system_extras.update(zip(ELECTRONIC_STATE_KEYS, values))
    return state


def _state_spin_charge(state):
    """Read explicit system extras, never infer graph labels from atom arrays."""
    if any(key in state.atom_extras for key in ELECTRONIC_STATE_KEYS):
        raise ValueError("Molecular charge/spin and masks belong in system_extras, not atom_extras.")
    extras = state.system_extras
    present = [key in extras for key in ELECTRONIC_STATE_KEYS]
    if not any(present):
        count = state.cell.shape[0]
        return (state.positions.new_zeros(count), state.positions.new_ones(count),
                torch.zeros(count, device=state.device, dtype=torch.bool))
    if not all(present):
        raise ValueError("Provide charge, spin and condition_mask together in system_extras; "
                         "use mace_soyo.torchsim.atoms_to_state for ASE inputs.")
    result = []
    for key in ELECTRONIC_STATE_KEYS:
        value = extras[key]
        if value.shape != (state.cell.shape[0],):
            raise ValueError(f"system_extras[{key!r}] must have shape [n_systems].")
        dtype = torch.bool if key.endswith("_mask") else state.dtype
        result.append(value.to(device=state.device, dtype=dtype).contiguous())
    charge, spin, mask = result
    valid = (torch.isfinite(charge) & torch.isfinite(spin)
             & (charge == charge.round()) & (spin == spin.round())
             & (charge >= -100) & (charge <= 100) & (spin >= 1) & (spin <= 100))
    assert bool((~mask | valid).all()), "Masked-in charge/spin must be integers: charge in [-100, 100], spin in [1, 100]."
    return charge, spin, mask


class MACESoyoTorchSimAOTIModel(ModelInterface):
    """TorchSim ModelInterface wrapper around a batched MACESoyo AOTI .pt2."""

    def __init__(
        self,
        package_path: str,
        use_d3: bool = False,
        a1: float = 0.4289,
        a2: float = 4.4407,
        s8: float = 0.7875,
        s6: float = 1.0,
        use_laspd3: bool = False,
        use_laspd3_BJ: bool = False,
        functional_type:int = 0,
        device: str | torch.device | None = None,
        d3_cutoff_radius: float = 46.4758,
        compute_forces: bool = True,
        compute_stress: bool = True,
        d3_params_path: str | Path | None = None,
        max_d3_cache_size: int = 16,
        head: str | None = None,
    ) -> None:
        super().__init__()
        self.package_path = str(package_path)
        self._device = resolve_cuda_device(device)
        self._compute_forces = bool(compute_forces)
        self._compute_stress = bool(compute_stress)
        self._memory_scales_with = "n_atoms_x_density"

        register_custom_ops()
        with torch.cuda.device(self.device):
            self.aoti_model = torch._inductor.aoti_load_package(
                self.package_path, device_index=self.device.index,
            )
        self.metadata = load_metadata(self.aoti_model)
        # Input precision belongs to the compiled package, not to the caller.
        self._dtype = resolve_torch_dtype(self.metadata["mace_soyo_aoti_dtype"])

        self.use_d3 = bool(use_d3)
        self.use_laspd3 = bool(use_laspd3)
        self.use_laspd3_BJ = bool(use_laspd3_BJ)
        self.functional_type = int(functional_type)
        self.d3_cutoff_radius = float(d3_cutoff_radius)
        self.max_d3_cache_size = int(max_d3_cache_size)
        if self.max_d3_cache_size < 1:
            raise ValueError("max_d3_cache_size must be at least 1.")
        if self.use_d3 and self.use_laspd3:
            raise ValueError("use_d3 and use_laspd3 are mutually exclusive.")
        if self.use_laspd3_BJ and not self.use_laspd3:
            raise ValueError("use_laspd3_BJ=True requires use_laspd3=True.")

        self._d3_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._laspd3_calculator_cls: Any | None = None
        if self.use_laspd3:
            if self.device.type != "cuda":
                raise ValueError("The LASP-D3 TorchSim backend currently requires CUDA.")
            from mace_soyo.utils.d3_cffi import D3Calculator

            self._laspd3_calculator_cls = D3Calculator
            print("[MACESoyoTorchSim/D3] backend = LASP-D3")
        if self.use_d3:
            self.d3model = make_torchsim_d3_model(
                a1=a1,
                a2=a2,
                s8=s8,
                s6=s6,
                cutoff=self.d3_cutoff_radius,
                device=self.device,
                dtype=self.dtype,
                d3_params_path=d3_params_path,
            )
            print("[MACESoyoTorchSim/D3] backend = TorchSim D3(BJ)")
        self.head_names, self._head_id, self._dynamic_head = select_head(self.metadata, head)
        self._conditioned = tuple(self.metadata.get("mace_soyo_aoti_inputs", "").split()) == SPIN_CHARGE_INPUTS

        cutoff = self.metadata.get("mace_soyo_aoti_cutoff")
        if cutoff is None:
            raise ValueError("The .pt2 package must contain mace_soyo_aoti_cutoff metadata.")
        self.cutoff = float(cutoff)

        print(f"[MACESoyoTorchSim/AOTI] loaded package = {self.package_path}")
        print(f"[MACESoyoTorchSim/AOTI] device = {self.device}")
        print(f"[MACESoyoTorchSim/AOTI] dtype = {self.dtype}")
        print(f"[MACESoyoTorchSim/AOTI] cutoff = {self.cutoff}")
        print(f"[MACESoyoTorchSim/AOTI] heads = {list(self.head_names)}; selected = {self.head!r}")
        print(f"[MACESoyoTorchSim/AOTI] ABI = {self.metadata.get('mace_soyo_aoti_inputs', 'legacy six-input')} -> (energy, forces, stress)")

    @property
    def head(self) -> str:
        return self.head_names[self._head_id]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    def _get_d3(self, elements: list[int]) -> Any:
        # LASP-D3's calculator is initialized for a given element list/order.
        # Cache by element sequence so batched systems with different species/order
        # do not accidentally reuse an incompatible D3Calculator.
        key = (
            tuple(int(x) for x in elements),
            float(self.d3_cutoff_radius),
            int(self.use_laspd3_BJ),
            int(self.functional_type),
        )
        calc = self._d3_cache.get(key)
        if calc is not None:
            self._d3_cache.move_to_end(key)
            return calc

        if self._laspd3_calculator_cls is None:
            raise RuntimeError("LASP-D3 was not initialized.")
        calc = self._laspd3_calculator_cls(
            elements=list(key[0]),
            max_length=len(key[0]),
            cutoff_radius=self.d3_cutoff_radius,
            cn_cutoff_radius=self.d3_cutoff_radius,
            damping_type=int(self.use_laspd3_BJ),
            functional_type=self.functional_type,
        )
        self._d3_cache[key] = calc

        while len(self._d3_cache) > self.max_d3_cache_size:
            _, oldest = self._d3_cache.popitem(last=False)
            oldest.close()
        return calc

    def close(self) -> None:
        """Release every cached LASP-D3 handle."""
        for calc in self._d3_cache.values():
            calc.close()
        self._d3_cache.clear()

    def _compute_d3_single_gpu(
        self,
        z: torch.Tensor,
        wrapped_positions: torch.Tensor,
        cell_3x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LASP-D3 for one CUDA-resident structure."""
        d3 = self._get_d3(z.detach().cpu().tolist())
        pos32 = wrapped_positions.contiguous()
        if pos32.dtype != torch.float32:
            pos32 = pos32.to(dtype=torch.float32)
        z64 = z.contiguous()
        if z64.dtype != torch.int64:
            z64 = z64.to(dtype=torch.int64)
        cell32 = cell_3x3.contiguous()
        if cell32.dtype != torch.float32:
            cell32 = cell32.to(dtype=torch.float32)

        d3_energy, d3_forces, d3_stress = d3.compute_torch(pos32, z64, cell32)
        return (
            d3_energy.to(device=self.device, dtype=self.dtype),
            d3_forces.to(device=self.device, dtype=self.dtype),
            d3_stress.unsqueeze(0).to(device=self.device, dtype=self.dtype),
        )

    def forward(self, state: Any) -> dict[str, torch.Tensor]:
        if isinstance(state, ts.SimState):
            sim_state = state
        else:
            state_dict = dict(state)
            if "masses" not in state_dict:
                state_dict["masses"] = torch.ones_like(state_dict["positions"][..., 0])
            sim_state = ts.SimState(**state_dict)

        if sim_state.device.type != "cuda":
            raise ValueError(f"MACE-SOYO requires a CUDA SimState, got {sim_state.device}.")
        if sim_state.device != self.device or sim_state.dtype != self.dtype:
            sim_state = sim_state.clone().to(self.device, self.dtype)

        z = sim_state.atomic_numbers.to(device=self.device, dtype=torch.long).contiguous()
        batch = sim_state.system_idx.to(device=self.device, dtype=torch.long).contiguous()
        cell = sim_state.row_vector_cell.contiguous()
        pbc = normalize_pbc(sim_state.pbc.to(device=self.device, dtype=torch.bool), cell.shape[0])
        cell = complete_runtime_cells(cell, pbc)
        if self.use_laspd3 and not bool(pbc.all()):
            raise ValueError("LASP-D3 only supports fully periodic (TTT) systems; use TorchSim D3 for other PBC.")

        positions = sim_state.positions
        # MACE-SOYO convention is row-vector: cart = frac @ cell.
        frac_pos = torch.linalg.solve(
            cell[batch].transpose(-1, -2),
            positions.unsqueeze(-1),
        ).squeeze(-1).contiguous()
        frac_pos = torch.where(pbc[batch], torch.remainder(frac_pos, 1.0), frac_pos).contiguous()
        wrapped_positions = torch.bmm(frac_pos.unsqueeze(1), cell[batch]).squeeze(1).contiguous()

        with torch.cuda.device(self.device):
            edge_index, image = build_batched_neighbor_list(
                positions=wrapped_positions,
                cell=cell,
                batch=batch,
                pbc=pbc,
                cutoff=self.cutoff
            )

        # Empty edges are valid, including batches with isolated systems.
        # The AOTI model still receives every atom and its system index.
        args = (
            z,
            frac_pos,
            cell,
            batch,
            edge_index.contiguous(),
            image.contiguous(),
        )
        if self._dynamic_head:
            head_id = torch.full((cell.shape[0],), self._head_id, dtype=torch.long, device=self.device)
            args += (head_id, pbc)
        if self._conditioned:
            args += _state_spin_charge(sim_state)
        elif any(key in sim_state.system_extras for key in ELECTRONIC_STATE_KEYS):
            labels = _state_spin_charge(sim_state)
            if bool(labels[2].any()):
                raise ValueError("This package has no charge/spin inputs. Export a use_spin_charge=True model.")

        with torch.cuda.device(self.device), torch.inference_mode():
            energy, forces, stress = self.aoti_model(*args)

        # Keep the package's output dtypes (FP64 energy, model-dtype forces/stress).
        if self.use_laspd3:
            num_graphs = int(cell.shape[0])
            d3_energy_tensor = torch.zeros_like(energy)
            d3_forces_tensor = torch.zeros_like(forces)
            d3_stress_tensor = torch.zeros_like(stress)

            for system in range(num_graphs):
                selected = system == batch
                z_selected = z[selected]
                wrapped_positions_selected = wrapped_positions[selected]
                cell_3x3 = cell[system]
                with torch.cuda.device(self.device):
                    d3_energy, d3_forces, d3_stress = self._compute_d3_single_gpu(
                        z=z_selected,
                        wrapped_positions=wrapped_positions_selected,
                        cell_3x3=cell_3x3,
                    )
                d3_energy_tensor[system] = d3_energy.reshape(())
                d3_forces_tensor[selected] = d3_forces
                d3_stress_tensor[system] = d3_stress.reshape(3, 3)

            energy = energy + d3_energy_tensor
            forces = forces + d3_forces_tensor
            stress = stress + d3_stress_tensor
            
        elif self.use_d3:
            sim_state = sim_state.clone()  # Do not change the MD state.
            sim_state.cell = cell.transpose(-1, -2).contiguous()
            sim_state.positions = wrapped_positions
            sim_state.pbc = pbc
            with torch.cuda.device(self.device):
                d3results = self.d3model(sim_state)
            d3_energy = d3results["energy"]
            d3_forces = d3results["forces"]
            d3_stress = d3results["stress"] 
            energy = energy + d3_energy
            forces = forces + d3_forces
            stress = stress + d3_stress
        stress = torch.where(pbc.all(dim=-1)[:, None, None], stress.reshape(-1, 3, 3), 0.0)
        results: dict[str, torch.Tensor] = {"energy": energy.detach()}

        if self.compute_forces:
            results["forces"] = forces.detach()
        if self.compute_stress:
            results["stress"] = stress.detach()
        return results
