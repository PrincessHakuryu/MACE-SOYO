"""Shared MACE backbone with per-dataset energy heads (no PME/BEC)."""

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from torch.autograd import grad

from mace_soyo.utils.neighbors import nvgraph
from .model_block import (
    BesselBasis,
    EquivariantInteractionBlock,
    PolynomialCutoff,
    ZBLBasis,
    AgnesiTransform,
    SpinChargeEmbedding,
)
from .readout import NonLinearReadout


class MACESoyo(torch.nn.Module):
    def __init__(
        self,
        cutoff=5.0,
        node_dim=192,
        num_layers=3,
        pair_dim=96,
        edge_weight_dim=64,
        num_rbf=16,
        max_l=2,
        max_ell=3,
        E_0=None,
        use_zbl=False,
        num_elements=105,
        max_correlation=3,
        num_heads=1,
        readout_hidden=64,
        head_names=None,
        use_spin_charge=False,
    ):
        super().__init__()
        for name, value in (("num_heads", num_heads), ("readout_hidden", readout_hidden),
                            ("num_layers", num_layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self.cutoff = cutoff
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.use_zbl = use_zbl
        self.num_heads = num_heads
        self.readout_hidden = readout_hidden
        self.head_names = list(head_names) if head_names is not None else [f"head_{h}" for h in range(num_heads)]
        if len(self.head_names) != num_heads or len(set(self.head_names)) != num_heads:
            raise ValueError("head_names must contain exactly num_heads unique names.")
        self.use_spin_charge = bool(use_spin_charge)
        self.max_l = int(max_l)
        self.max_ell = int(max_ell)
        if self.max_l < 0 or self.max_ell < 0:
            raise ValueError("max_l and max_ell must be non-negative.")

        self.irreps_input = cue.Irreps("O3", f"{node_dim}x0e")
        self.irreps_output = cue.Irreps("O3", f"{node_dim}x0e")
        node_parts, pair_parts = [], []
        edge_parts, hidden_parts, sh_parts = [], [], []
        for l in range(self.max_l + 1):
            parity = "e" if l % 2 == 0 else "o"
            pair_parts.append(f"{pair_dim}x{l}{parity}")
            node_parts.append(f"{node_dim}x{l}{parity}")
        for l in range(self.max_ell + 1):
            parity = "e" if l % 2 == 0 else "o"
            sh_parts.append(f"1x{l}{parity}")
            edge_parts.append(f"{pair_dim}x{l}{parity}")
            hidden_parts.append(f"{node_dim}x{l}{parity}")
        self.irreps_middle_out = cue.Irreps("O3", "+".join(node_parts))
        self.irreps_edge = cue.Irreps("O3", "+".join(pair_parts))
        self.irreps_hidden_edge = cue.Irreps("O3", "+".join(edge_parts))
        self.irreps_hidden_node = cue.Irreps("O3", "+".join(hidden_parts))
        self.irreps_shls = list(range(self.max_ell + 1))
        self.irreps_sh = cue.Irreps("O3", "+".join(sh_parts))
        if self.use_zbl:
            self.zbl = ZBLBasis(cutoff=cutoff)

        energy_irreps = cue.Irreps("O3", f"{num_heads}x0e")
        self.z_emb = torch.nn.Embedding(num_elements, node_dim)
        if self.use_spin_charge:
            self.spin_charge_embedding = SpinChargeEmbedding(node_dim)
        # Data loading prepares E0 in FP64; preserve its dtype here.
        if E_0 is None:
            e0 = torch.zeros(num_elements, num_heads, dtype=torch.float64)
        else:
            e0 = E_0.detach().clone()
            if num_heads == 1 and e0.shape == (num_elements,):
                e0 = e0[:, None]
            if e0.shape != (num_elements, num_heads):
                raise ValueError(f"E_0 must have shape {(num_elements, num_heads)}, got {tuple(e0.shape)}.")
            if not torch.isfinite(e0).all():
                raise ValueError("E_0 must be finite.")
        self.register_buffer("E0", e0)

        self.rbf_function = BesselBasis(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_factor = PolynomialCutoff(cutoff)
        self.sph_function = cuet.SphericalHarmonics(self.irreps_shls, normalize=True)
        self.layers = torch.nn.ModuleList()
        self.energy_readouts = torch.nn.ModuleList()
        self.distance_transform = AgnesiTransform()
        for layer_idx in range(num_layers):
            in_irreps = self.irreps_input if layer_idx == 0 else self.irreps_middle_out
            out_irreps = self.irreps_output if layer_idx == num_layers - 1 else self.irreps_middle_out
            self.layers.append(EquivariantInteractionBlock(
                irreps_in=in_irreps,
                irreps_hidden_edge=self.irreps_hidden_edge,
                irreps_hidden_node=self.irreps_hidden_node,
                irreps_edge=self.irreps_edge,
                irreps_out=out_irreps,
                irreps_sh=self.irreps_sh,
                edge_weight_dim=edge_weight_dim,
                num_rbf=num_rbf,
                max_correlation=max_correlation,
                first_layer=(layer_idx == 0),
            ))
            if layer_idx == num_layers - 1:
                readout = NonLinearReadout(out_irreps, num_heads, readout_hidden)
            else:
                readout = cuet.Linear(out_irreps, energy_irreps, layout_in=cue.ir_mul, layout_out=cue.ir_mul)
            self.energy_readouts.append(readout)

    def compute_energy_from_inputs(
        self,
        z: torch.Tensor,
        v_r: torch.Tensor,
        dist: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        head_id: torch.Tensor,
        charge=None, spin=None, condition_mask=None,
        *, return_atomic: bool = False, feature_exchange=None,
    ) -> torch.Tensor:
        """Compute energies from canonical tensors supplied by the input adapter."""
        node_heads = head_id[batch].unsqueeze(-1)
        edge_sh = self.sph_function(v_r)
        x = self.z_emb(z - 1)
        if self.use_spin_charge:
            x = self.spin_charge_embedding(x, batch, charge, spin, condition_mask)
        if feature_exchange is not None:
            # Keep communication in autograd even on a rank with no neighbor pairs.
            x = x + v_r.sum() * 0.0
        communication_anchor = x.new_zeros(())
        atomic_energy = x.new_zeros((x.shape[0], 1))
        edge_cutoff = self.cutoff_factor(dist).unsqueeze(-1)
        if self.use_zbl:
            atomic_energy = atomic_energy + self.zbl(dist, z, edge_index).unsqueeze(-1)
            dist_rbf = self.distance_transform(dist, edge_index, z)
        else:
            dist_rbf = dist
        edge_rbf = self.rbf_function(dist_rbf)
        for layer_idx, (layer, readout) in enumerate(zip(self.layers, self.energy_readouts)):
            if layer_idx > 0 and feature_exchange is not None:
                x = feature_exchange(x)
                # An empty rank must also execute the reverse exchanges.
                communication_anchor = communication_anchor + x.sum() * 0.0
            x = layer(x, edge_index, edge_sh, edge_rbf, edge_cutoff)
            # Compute all heads in a single graph, keeping DDP parameter usage static.
            outputs = readout(x, node_heads) if layer_idx == self.num_layers - 1 else readout(x)
            atomic_energy = atomic_energy + outputs.gather(1, node_heads)
        # Add E0 and sum atomic energies in FP64; the neural network keeps its dtype.
        atomic_e0 = self.E0[(z - 1).long()].gather(1, node_heads)
        atomic_energy = atomic_energy.double() + atomic_e0
        if feature_exchange is not None:
            atomic_energy = atomic_energy + communication_anchor
        if return_atomic:
            return atomic_energy
        total_energy = atomic_energy.new_zeros(num_graphs)
        return total_energy.scatter_add_(0, batch, atomic_energy.squeeze(-1))

    def forward(self, data):
        batch = data.batch
        cell = data.cell.to(dtype=self.z_emb.weight.dtype).reshape(-1, 3, 3)
        num_graphs = cell.shape[0]
        pbc = data.pbc.reshape(num_graphs, 3).to(device=cell.device, dtype=torch.bool)
        stress_enabled = pbc.all(dim=-1)

        # Only fully periodic structures receive a strain perturbation.
        # Missing stress labels are handled by the loss, not the prediction.
        strain = torch.zeros_like(cell, requires_grad=True)
        sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
        transform = torch.eye(3, device=cell.device, dtype=cell.dtype).unsqueeze(0)
        transform = transform + sym_strain * stress_enabled[:, None, None]
        lattice = torch.bmm(cell, transform)
        base_pos = torch.bmm(data.frac_pos.to(cell.dtype).unsqueeze(1), cell[batch]).squeeze(1)
        base_pos.requires_grad_(True)
        positions = torch.bmm(base_pos.unsqueeze(1), transform[batch]).squeeze(1)
        edge_index, shift = nvgraph(
            positions, self.cutoff, batch, lattice, pbc=pbc,
            batch_ptr=data.ptr,
        )
        j, i = edge_index # invert the edge_index given by alchemitoolkit in nvgraph
        vectors = positions[j] + shift - positions[i]
        volume = torch.linalg.det(cell).abs()
        safe_volume = torch.where(stress_enabled, volume, torch.ones_like(volume))
        energy = self.compute_energy_from_inputs(
            data.z, vectors, vectors.norm(dim=-1), edge_index, batch, num_graphs,
            head_id=data.head_id.reshape(-1),
            charge=data.charge, spin=data.spin, condition_mask=data.condition_mask,
        )
        # Keep coordinate derivatives defined for isolated atoms with no edges.
        energy = energy + 0.0 * (positions.sum() + strain.sum())
        position_grad, strain_grad = grad(
            energy, (positions, strain), grad_outputs=torch.ones_like(energy),
            create_graph=self.training, retain_graph=self.training,
        )
        stress = strain_grad / safe_volume[:, None, None]
        stress = torch.where(stress_enabled[:, None, None], stress, torch.zeros_like(stress))
        return energy, -position_grad, stress
