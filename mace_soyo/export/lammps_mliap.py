"""LAMMPS ML-IAP adapter for MACESoyo.

The serialized object stays small and maintainable: LAMMPS owns neighbor lists,
MPI domain decomposition, and force/virial accumulation; MACESoyo computes
local atomic energies. Hidden node features are exchanged between interaction
layers with LAMMPS' differentiable Kokkos communication hooks.
"""

from __future__ import annotations

from typing import Any

import torch
from ase.data import chemical_symbols

from mace_soyo.model import MACESoyo


class LAMMPSFeatureExchangeOp(torch.autograd.Function):
    """Differentiable local/ghost feature exchange supplied by ML-IAP/Kokkos."""

    @staticmethod
    def forward(ctx: Any, node_features: torch.Tensor, lmp_data: Any) -> torch.Tensor:
        original_shape = node_features.shape
        flat = node_features.flatten(start_dim=1).contiguous()
        output = torch.empty_like(flat)
        lmp_data.forward_exchange(flat, output, output.shape[-1])

        ctx.lmp_data = lmp_data
        ctx.original_shape = original_shape
        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        flat = grad_output.flatten(start_dim=1).contiguous()
        grad_input = torch.empty_like(flat)
        ctx.lmp_data.reverse_exchange(flat, grad_input, grad_input.shape[-1])
        return grad_input.reshape(ctx.original_shape), None


@torch.compiler.disable
def exchange_features(features: torch.Tensor, lmp_data: Any) -> torch.Tensor:
    """Keep MPI callbacks outside compiled graphs, including their backward."""
    nlocal = int(lmp_data.nlocal)
    prepared = torch.cat((features[:nlocal], torch.zeros_like(features[nlocal:])))
    return LAMMPSFeatureExchangeOp.apply(prepared, lmp_data)


class MACESoyoMLIAPEnergy(torch.nn.Module):
    """Tensor computation invoked by the Python ML-IAP wrapper."""

    def __init__(self, model: MACESoyo, atomic_numbers: list[int], head_index: int) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("head_id", torch.tensor([head_index], dtype=torch.long))
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.long),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.model.z_emb.weight.dtype

    def forward(
        self,
        edge_vectors: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        element_indices: torch.Tensor,
        lmp_data: Any,
    ) -> torch.Tensor:
        # LAMMPS defines rij = r_j - r_i. MACESoyo expects edge_index as
        # [sender=j, receiver=i], matching its native nvgraph path.
        edge_index = torch.stack((pair_j, pair_i), dim=0)
        z = self.atomic_numbers[element_indices]
        vectors = edge_vectors.to(dtype=self.dtype)
        if element_indices.numel() == 0:
            # An entirely empty MPI domain still participates in every exchange.
            # Avoid CUEQ/normalization kernels that reshape [0, C] with -1.
            features = vectors.new_zeros((0, self.model.irreps_middle_out.dim))
            features = features + vectors.sum() * 0.0
            for _ in range(1, self.model.num_layers):
                features = exchange_features(features, lmp_data)
            return features.sum().double().expand(0, 1)
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        batch = torch.zeros(
            element_indices.shape[0],
            dtype=torch.long,
            device=element_indices.device,
        )

        def exchange(features: torch.Tensor) -> torch.Tensor:
            return exchange_features(features, lmp_data)

        atomic_energy = self.model.compute_energy_from_inputs(
            z=z,
            v_r=vectors,
            dist=distances,
            edge_index=edge_index,
            batch=batch,
            num_graphs=1,
            head_id=self.head_id,
            # ML-IAP supplies no molecular labels: leave conditioning disabled.
            charge=vectors.new_zeros(1),
            spin=vectors.new_ones(1),
            condition_mask=torch.zeros(1, dtype=torch.bool, device=vectors.device),
            return_atomic=True,
            feature_exchange=exchange,
        )
        return atomic_energy


class MACESoyoMLIAP:
    """Serialized unified ML-IAP object loaded by LAMMPS."""

    def __init__(
        self,
        model: MACESoyo,
        *,
        head: str | None = None,
        compile_model: bool = True,
        compile_mode: str = "default",
    ) -> None:
        self.interface = None  # Populated by LAMMPS when loading the object.
        self.head_names = list(model.head_names)
        if head is None and len(self.head_names) == 1:
            head = self.head_names[0]
        if head not in self.head_names:
            raise ValueError(f"Choose --head from {self.head_names}; got {head!r}")
        self.head = head
        self.head_index = self.head_names.index(head)
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"Unsupported torch.compile mode: {compile_mode}")

        model = model.eval().cpu()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        atomic_numbers = list(range(1, int(model.num_elements) + 1))
        if atomic_numbers[-1] >= len(chemical_symbols):
            raise ValueError(
                f"num_elements={model.num_elements} exceeds ASE's element table"
            )

        self.element_types = [chemical_symbols[z] for z in atomic_numbers]
        self.rcutfac = 0.5 * float(model.cutoff)
        self.ninteractions = int(model.num_layers)
        self.ndescriptors = 1
        self.nparams = 1

        self.compile_model = bool(compile_model)
        self.compile_mode = compile_mode
        self.energy_model = MACESoyoMLIAPEnergy(model, atomic_numbers, self.head_index)
        self.runtime_model: torch.nn.Module | None = None
        self.device: torch.device | None = None

    def _initialize_runtime(self, lmp_data: Any) -> None:
        if lmp_data.ntotal:
            self.device = torch.as_tensor(lmp_data.elems).device
        else:
            import cupy
            self.device = torch.device("cuda", cupy.cuda.runtime.getDevice())
        using_kokkos = "kokkos" in lmp_data.__class__.__module__.lower()
        if self.device.type != "cuda" or not using_kokkos:
            raise RuntimeError(
                "MACESoyo ML-IAP multi-GPU path requires ML-IAP/Kokkos "
                f"CUDA tensors, got module={lmp_data.__class__.__module__!r}, "
                f"device={self.device}"
            )

        eager_model = self.energy_model.to(self.device).eval()
        if self.compile_model:
            # Same high-level strategy as NequIP ML-IAP: dynamic shapes for
            # changing neighbor counts and graph breaks around LAMMPS exchange.
            self.runtime_model = torch.compile(
                eager_model,
                dynamic=True,
                fullgraph=False,
                mode=self.compile_mode,
            )
        else:
            self.runtime_model = eager_model

    def compute_forces(self, lmp_data: Any) -> None:
        if self.runtime_model is None:
            self._initialize_runtime(lmp_data)
        assert self.runtime_model is not None
        assert self.device is not None

        with torch.enable_grad():
            edge_vectors = torch.as_tensor(
                lmp_data.rij if lmp_data.npairs else [],
                dtype=torch.float64,
                device=self.device,
            ).reshape(-1, 3)
            edge_vectors.requires_grad_(True)
            pair_i = torch.as_tensor(
                lmp_data.pair_i if lmp_data.npairs else [],
                dtype=torch.long,
                device=self.device,
            )
            pair_j = torch.as_tensor(
                lmp_data.pair_j if lmp_data.npairs else [],
                dtype=torch.long,
                device=self.device,
            )
            element_indices = torch.as_tensor(
                lmp_data.elems if lmp_data.ntotal else [],
                dtype=torch.long,
                device=self.device,
            )

            atomic_energy = self.runtime_model(
                edge_vectors,
                pair_i,
                pair_j,
                element_indices,
                lmp_data,
            ).reshape(-1)
            local_atomic_energy = torch.narrow(
                atomic_energy,
                0,
                0,
                int(lmp_data.nlocal),
            )
            total_energy = local_atomic_energy.sum() + edge_vectors.sum() * 0.0
            edge_forces = torch.autograd.grad(
                total_energy,
                edge_vectors,
                create_graph=False,
                retain_graph=False,
            )[0]

        if lmp_data.nlocal:
            lmp_eatoms = torch.as_tensor(lmp_data.eatoms)
            lmp_eatoms.copy_(
                local_atomic_energy.detach().to(
                    device=lmp_eatoms.device,
                    dtype=lmp_eatoms.dtype,
                )
            )
        lmp_data.energy = total_energy.detach()
        if lmp_data.npairs:
            lmp_data.update_pair_forces_gpu(
                edge_forces.detach().to(dtype=torch.float64).contiguous()
            )

    def compute_descriptors(self, lmp_data: Any) -> None:
        del lmp_data

    def compute_gradients(self, lmp_data: Any) -> None:
        del lmp_data
