#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Named multi-head, no-q AOTInductor exporter for MACESoyo.

AOTI ABI:
    inputs : (z, frac_pos, cell, batch, edge_index, image, head_id, pbc)
    outputs: (energy, forces, stress)

Design:
    - Neighbor-list construction stays outside the exported AOTI model.
    - The ASE runtime calculator does not load the checkpoint; it loads only the .pt2 package.
    - Runtime metadata such as cutoff/ABI is embedded in the .pt2 package via
      the AOTInductor config key ``aot_inductor.metadata``. No sidecar JSON is written.
    - AOTI runtime constant folding is enabled by default.
    - always_keep_tensor_constants=True is enabled by default to avoid the known
      cuet.Linear/AOTInductor lowering issue:
          aten.unsqueeze(Constant(1/sqrt(node_dim)), 0)
          AttributeError: 'Constant' object has no attribute 'data'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np
import torch
from ase.io import read

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from mace_soyo.utils.compile_utils import TensorEFS, make_fx_symbolic
from mace_soyo.inference.ase_calc import get_neighbor_list_gpu_pbc
from mace_soyo.inference.aoti_utils import MULTIHEAD_INPUTS, SPIN_CHARGE_INPUTS, resolve_torch_dtype
from mace_soyo.utils.geometry import atoms_geometry
from mace_soyo.utils.device import resolve_cuda_device
from mace_soyo.utils.model_checkpoint import load_model_from_checkpoint


# -----------------------------------------------------------------------------
# Data and export utilities
# -----------------------------------------------------------------------------


def atoms_to_export_inputs(
    structure_file: str,
    device: torch.device,
    dtype: torch.dtype,
    cutoff: float,
    *,
    head_id: int = 0,
) -> tuple[torch.Tensor, ...]:
    atoms = read(structure_file, index=0)
    frac, complete_cell, axes = atoms_geometry(atoms)

    z = torch.as_tensor(
        np.asarray(atoms.numbers, dtype=np.int64),
        dtype=torch.long,
        device=device,
    ).contiguous()
    frac_pos = torch.as_tensor(
        frac,
        dtype=dtype,
        device=device,
    ).contiguous()
    cell_single = torch.as_tensor(
        complete_cell,
        dtype=dtype,
        device=device,
    ).contiguous()

    pbc = tuple(bool(x) for x in atoms.pbc)

    edge_index, image = get_neighbor_list_gpu_pbc(
        frac_pos=frac_pos,
        cell=cell_single,
        cutoff=float(cutoff),
        pbc=pbc,
    )
    edge_index = edge_index.long().contiguous()
    image = image.long().contiguous()

    # The export example is still a single structure. The exported ABI is batched,
    # so we provide B=1 cell and a [N] all-zero atom-to-graph batch vector.
    cell = cell_single.unsqueeze(0).contiguous()  # [1, 3, 3]
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device).contiguous()

    if z.shape[0] < 2:
        raise RuntimeError(f"Need at least 2 atoms for dynamic export, got {z.shape[0]}.")
    if edge_index.shape[1] < 2:
        raise RuntimeError(f"Need at least 2 edges for dynamic export, got {edge_index.shape[1]}.")

    heads = torch.tensor([head_id], device=device, dtype=torch.long)
    pbc_tensor = torch.tensor([list(axes)], device=device, dtype=torch.bool)
    return z, frac_pos, cell, batch, edge_index, image, heads, pbc_tensor


def tile_single_structure_export_inputs(
    args: Sequence[torch.Tensor],
    num_graphs: int = 2,
) -> tuple[torch.Tensor, ...]:
    """Tile a single-structure export example into a real batched example.

    torch.export cannot prove that the cell batch dimension is dynamic if the
    example input has cell.shape[0] == 1; it specializes B to the constant 1 and
    rejects a dynamic ``num_graphs`` Dim.  For export only, we therefore build a
    small B=2 example by concatenating two disconnected copies of the same
    structure.  Neighbor-list construction still stays outside AOTI; we only
    offset the edge indices for the copied graph.
    """
    if num_graphs < 2:
        raise ValueError("num_graphs must be >= 2 for a dynamic batched export example.")

    z, frac_pos, cell, batch, edge_index, image, head_id, pbc = args
    if cell.shape[0] != 1:
        raise ValueError(
            f"tile_single_structure_export_inputs expects a single-graph cell [1,3,3], got {tuple(cell.shape)}"
        )
    if batch.numel() != z.numel() or bool((batch != 0).any().item()):
        raise ValueError("The input example must be a single graph with all-zero batch.")

    n_atoms = int(z.shape[0])

    z_parts = []
    frac_parts = []
    batch_parts = []
    edge_parts = []
    image_parts = []

    for graph_idx in range(num_graphs):
        atom_offset = graph_idx * n_atoms
        z_parts.append(z)
        frac_parts.append(frac_pos)
        batch_parts.append(torch.full_like(batch, graph_idx))
        edge_parts.append(edge_index + atom_offset)
        image_parts.append(image)

    z_b = torch.cat(z_parts, dim=0).contiguous()
    frac_pos_b = torch.cat(frac_parts, dim=0).contiguous()
    cell_b = cell.expand(num_graphs, -1, -1).clone().contiguous()
    batch_b = torch.cat(batch_parts, dim=0).contiguous()
    edge_index_b = torch.cat(edge_parts, dim=1).contiguous()
    image_b = torch.cat(image_parts, dim=0).contiguous()

    return (z_b, frac_pos_b, cell_b, batch_b, edge_index_b, image_b,
            head_id.repeat(num_graphs).contiguous(), pbc.repeat(num_graphs, 1).contiguous())


def dump_graphs(prefix: str, fx_model: Optional[torch.fx.GraphModule] = None, exported=None) -> None:
    if fx_model is not None:
        Path(prefix + ".fx.py").write_text(fx_model.code)
    if exported is not None:
        Path(prefix + ".exported.py").write_text(exported.graph_module.code)


def dynamic_shapes_for_efs(
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_graphs: int = 20000,
) -> tuple[dict[int, object], ...]:
    n_max = torch.inf if max_nodes is None else int(max_nodes)
    e_max = torch.inf if max_edges is None else int(max_edges)

    num_nodes = torch.export.Dim("num_nodes", min=2, max=n_max)
    num_edges = torch.export.Dim("num_edges", min=2, max=e_max)
    static = torch.export.Dim.STATIC

    if max_graphs < 2:
        raise ValueError("--max-graphs must be >= 2.")
    num_graphs = torch.export.Dim("num_graphs", min=1, max=int(max_graphs))
    cell_shape = {0: num_graphs, 1: static, 2: static}  # cell: [B, 3, 3]

    return (
        {0: num_nodes},             # z: [N]
        {0: num_nodes, 1: static},  # frac_pos: [N, 3]
        cell_shape,                 # cell: [B, 3, 3]
        {0: num_nodes},             # batch: [N]
        {0: static, 1: num_edges},  # edge_index: [2, E]
        {0: num_edges, 1: static},  # image: [E, 3]
        {0: cell_shape[0]},         # head_id: [B]
        {0: cell_shape[0], 1: static},  # pbc: [B, 3]
    )


def _parse_config_value(value: str) -> object:
    low = value.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_inductor_configs(items: list[str]) -> dict[str, object]:
    configs: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --inductor-config {item!r}; expected key=value")
        key, value = item.split("=", 1)
        configs[key] = _parse_config_value(value)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export clean no-q MACESoyo E/F/stress model with torch.export + AOTInductor"
    )
    parser.add_argument("--ckpt-path", required=True, help="Best train/valid .pth checkpoint")
    parser.add_argument("--structure-file", required=True, help="Realistic ASE-readable structure used as export example")
    parser.add_argument("--output-path", default="mace_soyo_multihead_efs.pt2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional dynamic upper bound for N")
    parser.add_argument("--max-edges", type=int, default=None, help="Optional dynamic upper bound for E")
    parser.add_argument("--max-graphs", type=int, default=20000, help="Dynamic upper bound for graph count B (default: 20000, minimum: 2). The tracing example always uses B=2; runtime B=1 remains supported.")
    parser.add_argument("--dump-graphs", action="store_true", help="Write <output-path>.fx.py and <output-path>.exported.py")
    parser.add_argument(
        "--inductor-config",
        action="append",
        default=[],
        help="AOTInductor config as key=value. Can be repeated. Bool values are parsed.",
    )
    args = parser.parse_args()

    device = resolve_cuda_device(args.device)
    torch.cuda.set_device(device)
    if args.max_graphs < 2:
        parser.error("--max-graphs must be >= 2.")
    dtype = resolve_torch_dtype(args.dtype)
    torch._dynamo.reset()

    base_model = load_model_from_checkpoint(args.ckpt_path, device=device, dtype=dtype)
    wrapper = TensorEFS(base_model).to(device=device).eval()

    example_args = atoms_to_export_inputs(
        structure_file=args.structure_file,
        device=device,
        dtype=dtype,
        cutoff=float(base_model.cutoff),
    )

    # Trace with B=2 so torch.export does not specialize the graph count to 1. This is intentional due to torch.export's current limitations with dynamic shapes.
    example_args = tile_single_structure_export_inputs(example_args, num_graphs=2)
    if base_model.num_heads > 1:
        example_args[6][1] = 1
    print(
        f"[export] tiled single structure into a B=2 batched example "
        f"for dynamic num_graphs export; max_graphs={args.max_graphs}"
    )

    input_names = MULTIHEAD_INPUTS
    if base_model.use_spin_charge:
        input_names = SPIN_CHARGE_INPUTS
        num_graphs = example_args[2].shape[0]
        example_args += (
            torch.zeros(num_graphs, device=device, dtype=dtype),
            torch.ones(num_graphs, device=device, dtype=dtype),
            torch.ones(num_graphs, device=device, dtype=torch.bool),
        )
    print("[export] input ABI:")
    for name, tensor in zip(input_names, example_args):
        print(
            f"  {name:10s} shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} device={tensor.device} stride={tensor.stride()}"
        )

    dynamic_shapes = dynamic_shapes_for_efs(
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_graphs=args.max_graphs,
    )
    if base_model.use_spin_charge:
        dynamic_shapes += tuple({0: dynamic_shapes[6][0]} for _ in range(3))

    print("[export] make_fx symbolic trace ...")
    fx_model = make_fx_symbolic(wrapper, example_args)

    if args.dump_graphs:
        dump_graphs(str(Path(args.output_path)), fx_model=fx_model)
        print(f"[debug] wrote {args.output_path}.fx.py")

    print("[export] torch.export.export ...")
    exported = torch.export.export(
        fx_model,
        example_args,
        dynamic_shapes=dynamic_shapes,
    )

    if args.dump_graphs:
        dump_graphs(str(Path(args.output_path)), exported=exported)
        print(f"[debug] wrote {args.output_path}.exported.py")

    metadata: dict[str, Any] = {
        "mace_soyo_aoti_inputs": " ".join(input_names),
        "mace_soyo_aoti_use_spin_charge": str(int(base_model.use_spin_charge)),
        "mace_soyo_aoti_outputs": "energy forces stress",
        "mace_soyo_aoti_cutoff": str(float(base_model.cutoff)),
        "mace_soyo_aoti_dtype": str(dtype).replace("torch.", ""),
        "mace_soyo_aoti_runtime_constant_folding": "1",
        "mace_soyo_aoti_keep_tensor_constants": "1",
        "mace_soyo_aoti_safe_norm": "0",
        "mace_soyo_aoti_abi": "batched_noq_multihead_efs_v3",
        "mace_soyo_aoti_metadata_schema": "3",
        "mace_soyo_aoti_num_heads": str(base_model.num_heads),
        "mace_soyo_aoti_head_names": json.dumps(base_model.head_names),
        "mace_soyo_aoti_max_nodes": "" if args.max_nodes is None else str(args.max_nodes),
        "mace_soyo_aoti_max_edges": "" if args.max_edges is None else str(args.max_edges),
        "mace_soyo_aoti_max_graphs": str(args.max_graphs),
    }
    print(f"[export] saving ALL heads: {base_model.head_names}")
    # The following is very important, "always_keep_tensor_constants", True" is necessary to avoid the cuet.Linear/AOTInductor lowering issue.
    inductor_configs = parse_inductor_configs(args.inductor_config)
    inductor_configs.setdefault("aot_inductor.use_runtime_constant_folding", True)
    inductor_configs.setdefault("always_keep_tensor_constants", True)
    inductor_configs.setdefault("constant_and_index_propagation", True)
    inductor_configs.setdefault("joint_graph_constant_folding", True)
    # Same idea as NeQuIP: store runtime ABI/cutoff information in the AOTI package itself.
    if "aot_inductor.metadata" in inductor_configs:
        raise ValueError("Head/ABI metadata is managed by the exporter and cannot be overridden.")
    inductor_configs["aot_inductor.metadata"] = metadata

    print("[export] inductor configs:")
    for k, v in inductor_configs.items():
        if k != "aot_inductor.metadata":
            print(f"  {k} = {v!r}")

    output_path = str(Path(args.output_path))
    print(f"[export] aoti_compile_and_package -> {output_path}")
    out_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
        inductor_configs=inductor_configs,
    )

    print("[metadata] embedded in .pt2 via aot_inductor.metadata")

    print("[check] loading package ...")
    loaded = torch._inductor.aoti_load_package(out_path, device_index=device.index)

    # One E/F/S consistency check per head.
    for head_id, head_name in enumerate(base_model.head_names):
        inputs = [x.clone() for x in example_args]
        inputs[6].fill_(head_id)
        with torch.inference_mode():
            got = tuple(x.clone() for x in loaded(*inputs))
        with torch.enable_grad():
            ref = wrapper(*[x.clone() for x in inputs])
        # Large E0 baselines must not loosen the energy comparison.
        atom_counts = torch.bincount(inputs[3], minlength=inputs[2].shape[0])
        energy_error = (got[0] - ref[0]) / atom_counts
        print(
            f"[check] {head_name}/energy: max_abs_per_atom={energy_error.abs().max().item():.3e} "
            "eV/atom (atol=2e-5, rtol=0)"
        )
        torch.testing.assert_close(
            energy_error, torch.zeros_like(energy_error), rtol=0, atol=2e-5,
        )
        for name, actual, expected, rtol, atol in zip(
            ("forces", "stress"), got[1:], ref[1:],
            (2e-3, 2e-3), (8e-4, 1e-4),
        ):
            print(f"[check] {head_name}/{name}: max_abs={(actual - expected).abs().max().item():.3e}")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
