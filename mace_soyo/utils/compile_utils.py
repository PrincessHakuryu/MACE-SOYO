"""Shared tensor E/F/S tracing for AOTI export and optional compiled training.

Neighbors, losses, optimizers and DDP stay outside the compiled region.
Inference never applies the training-only saved-tensor graph repair.
"""
from __future__ import annotations

import contextlib
from typing import Sequence

import torch
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


@contextlib.contextmanager
def independent_symbolic_shapes():
    cfg = getattr(torch.fx.experimental, "_config", None)
    if cfg is None or not hasattr(cfg, "use_duck_shape"):
        yield
        return

    old_value = cfg.use_duck_shape
    cfg.use_duck_shape = False
    try:
        yield
    finally:
        cfg.use_duck_shape = old_value


class TensorEFS(torch.nn.Module):
    """Tensor-only E/F/S; choose one head per graph before differentiating."""

    def __init__(self, model: torch.nn.Module, *, create_graph: bool = False):
        super().__init__()
        self.model = model
        self.create_graph = create_graph

    @property
    def model_dtype(self) -> torch.dtype:
        return self.model.z_emb.weight.dtype

    def forward(
        self,
        z: torch.Tensor,           # [N], int64
        frac_pos: torch.Tensor,    # [N, 3]
        cell: torch.Tensor,        # [B, 3, 3]
        batch: torch.Tensor,       # [N], int64, atom -> graph index
        edge_index: torch.Tensor,  # [2, E], int64, j -> i after runtime convention
        image: torch.Tensor,       # [E, 3], int64 unit cell shifts
        head_id: torch.Tensor,     # [B], int64 runtime head selection
        pbc: torch.Tensor,         # [B, 3], bool (independent of head)
        charge=None, spin=None, condition_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.model_dtype
        device = z.device

        batch = batch.long()
        edge_index = edge_index.long()

        cell_tensor = cell.to(dtype=dtype)  # [B, 3, 3]
        num_graphs = cell_tensor.shape[0]

        strain = torch.zeros_like(cell_tensor, requires_grad=True)
        periodic = pbc.all(dim=-1)
        sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
        eyes = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(num_graphs, -1, -1)
            + sym_strain * periodic[:, None, None]
        )
        lattice = torch.bmm(cell_tensor, eyes)  # [B, 3, 3]

        volume = torch.linalg.det(cell_tensor).abs()
        volume = torch.where(periodic, volume, torch.ones_like(volume))
        vol_scale = 1.0 / volume.view(-1, 1, 1)

        # Row-vector convention: cart = frac @ cell.
        # For batched systems, each atom uses lattice[batch[n]].
        unfrac_pos = torch.bmm(
            frac_pos.to(dtype=dtype).unsqueeze(1),
            lattice[batch],
        ).squeeze(1)  # [N, 3]

        j, i = edge_index
        edge_batch = batch[i]
        shift = torch.bmm(
            image.to(dtype=dtype).unsqueeze(1),
            lattice[edge_batch],
        ).squeeze(1)  # [E, 3]

        v_r = unfrac_pos[j] + shift - unfrac_pos[i]
        dist = v_r.norm(dim=-1)

        energy = self.model.compute_energy_from_inputs(
            z=z,
            v_r=v_r,
            dist=dist,
            edge_index=edge_index,
            batch=batch,
            head_id=head_id,
            num_graphs=num_graphs,
            charge=charge, spin=spin, condition_mask=condition_mask,
        )
        energy = energy + 0.0 * (unfrac_pos.sum() + strain.sum())

        grads = torch.autograd.grad(
            outputs=energy,
            inputs=[unfrac_pos, strain],
            grad_outputs=torch.ones_like(energy),
            create_graph=self.create_graph,
            retain_graph=self.create_graph,
        )
        forces = -grads[0]
        stress = torch.where(periodic[:, None, None], vol_scale * grads[1], 0.0)
        return energy, forces, stress


def _repair_training_graph(fx):
    """Bypass observed saved-activation detach(alias(...)) from make_fx.

    Not a general 'remove detach' pass: intentional detaches are retained.
    Current energy forward does not intentionally detach these activations.
    Rebuild the graph rather than mutating its linked list during compilation.
    """
    graph = torch.fx.Graph()
    old_to_new = {}

    def translate_input_node(old_input):
        return old_to_new[old_input]

    saved_ops = {
        torch.ops.aten.sigmoid.default, torch.ops.aten.tanh.default,
        torch.ops.aten.rsqrt.default, torch.ops.aten.linalg_vector_norm.default,
    }
    repaired = 0
    for node in fx.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.detach.default:
            source = node.args[0]
            if (source.op == "call_function"
                    and source.target == torch.ops.aten.alias.default
                    and source.args[0].op == "call_function"
                    and source.args[0].target in saved_ops):
                old_to_new[node] = old_to_new[source]
                repaired += 1
                continue
        old_to_new[node] = graph.node_copy(node, translate_input_node)
    print(f"[compile] repaired {repaired} saved-activation detach nodes", flush=True)
    return torch.fx.GraphModule(fx, graph)


def make_fx_symbolic(
    model: torch.nn.Module,
    example_args: Sequence[torch.Tensor],
    *,
    training: bool = False,
) -> torch.fx.GraphModule:
    """Trace E/F/S, optionally retaining differentiability of force/stress."""
    decompositions = core_aten_decompositions()
    with torch.enable_grad(), independent_symbolic_shapes():
        fx = make_fx(
            model,
            decomposition_table=decompositions,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[x.clone() for x in example_args])
    if training:
        fx = _repair_training_graph(fx)
        original = {id(p) for p in model.parameters() if p.requires_grad}
        traced = {id(p) for p in fx.parameters() if p.requires_grad}
        if original != traced:
            raise RuntimeError("Training FX graph must share the original model parameters.")
    return fx


def training_inputs(model, data):
    """Build integer neighbor images eagerly; retain the AOTI tensor ABI."""
    from mace_soyo.utils.neighbors import nvgraph

    with torch.no_grad():
        cell = data.cell.to(dtype=model.z_emb.weight.dtype).reshape(-1, 3, 3)
        frac = data.frac_pos.to(dtype=cell.dtype)
        batch = data.batch.long()
        pbc = data.pbc.reshape(-1, 3).bool()
        positions = torch.bmm(frac.unsqueeze(1), cell[batch]).squeeze(1)
        edges, image = nvgraph(
            positions, model.cutoff, batch, cell, pbc=pbc,
            batch_ptr=data.ptr,
            return_image=True,
        )
    args = (data.z.long(), frac, cell, batch, edges, image.long(),
            data.head_id.long().reshape(-1), pbc)
    if model.use_spin_charge:
        args += (data.charge.to(cell.dtype), data.spin.to(cell.dtype),
                 data.condition_mask.bool())
    return args


class CompiledTrainingModel(torch.nn.Module):
    """DDP-compatible adapter; raw model owns all parameters/checkpoint keys.

    Separate lazy train/eval callables, outside the registered module tree.
    Evaluation traces E/F/S without a higher-order gradient graph.
    Create this adapter before DDP, but save/EMA/load the original raw model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._cache = {}
        # The compile boundary is the inner E/F/S core, not the DDP wrapper.
        torch._dynamo.config.optimize_ddp = False

    def forward(self, data):
        args = training_inputs(self.model, data)
        return self.forward_tensors(*args)

    def forward_tensors(self, *args):
        mode = "train" if self.training else "eval"
        if mode not in self._cache:
            print(f"[compile] tracing {mode} E/F/S (first batch)", flush=True)
            wrapper = TensorEFS(self.model, create_graph=self.training)
            fx = make_fx_symbolic(wrapper, args, training=self.training)
            self._cache[mode] = torch.compile(
                fx, backend="inductor", dynamic=True, fullgraph=True,
                options={"triton.cudagraphs": False},
            )
        # E/F/S derivatives are already explicit FX operations in eval mode.
        # Only training needs to record their parameter-gradient history.
        with torch.set_grad_enabled(self.training):
            return self._cache[mode](*args)
