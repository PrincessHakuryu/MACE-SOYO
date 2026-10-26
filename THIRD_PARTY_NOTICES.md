# Third-party notices

MACE-SOYO contains code adapted from the projects below. The original copyright
and permission notices are reproduced verbatim in `third-party-licenses/`.
MACE-SOYO's original code remains covered by the root [MIT License](LICENSE);
that notice does not replace the upstream notices.

The upstream revisions linked below are inspected reference snapshots, not a
claim that every adapted block originated at exactly that revision. This
document covers the identified core-model and ML-IAP source adaptations; it is
not a license inventory for every optional dependency or bundled component.

## MACE

- Project: [ACEsuit/mace](https://github.com/ACEsuit/mace).
- License: MIT; [original notice](third-party-licenses/MACE-LICENSE.txt).
- Local files: `mace_soyo/model/model_block.py` and
  `mace_soyo/model/readout.py`.
- Activation normalization and gating reference:
  [`mace/modules/gate.py`](https://github.com/ACEsuit/mace/blob/e37c8583a8b1b586e5888fa6dca86e8f591e2606/mace/modules/gate.py).
- Radial bases/cutoff, ZBL repulsion, and Agnesi transform reference:
  [`mace/modules/radial.py`](https://github.com/ACEsuit/mace/blob/e37c8583a8b1b586e5888fa6dca86e8f591e2606/mace/modules/radial.py).
- Tensor-product irreps helper and head-mask references:
  [`mace/modules/irreps_tools.py`](https://github.com/ACEsuit/mace/blob/e37c8583a8b1b586e5888fa6dca86e8f591e2606/mace/modules/irreps_tools.py).
- Readout and graph-label embedding references:
  [`mace/modules/blocks.py`](https://github.com/ACEsuit/mace/blob/e37c8583a8b1b586e5888fa6dca86e8f591e2606/mace/modules/blocks.py) and
  [`mace/modules/embeddings.py`](https://github.com/ACEsuit/mace/blob/e37c8583a8b1b586e5888fa6dca86e8f591e2606/mace/modules/embeddings.py).

These components have been adapted to MACE-SOYO's cuEquivariance interfaces,
feature layout, graph-label masking, neighbor convention, and readout structure.
They are not presented as unmodified upstream implementations.

## NequIP

- Project: [mir-group/nequip](https://github.com/mir-group/nequip).
- License: MIT; [original notice](third-party-licenses/NequIP-LICENSE.txt).
- Local files: `mace_soyo/model/model_block.py` and
  `mace_soyo/export/lammps_mliap.py`.
- Fused tensor-product convolution wrapper reference:
  [`nequip/nn/_tp_scatter_cueq.py`](https://github.com/mir-group/nequip/blob/27d9d2182da918ab7be0017d8300e53278f5e00e/nequip/nn/_tp_scatter_cueq.py).
- Differentiable forward/reverse local-ghost exchange reference:
  [`nequip/nn/_ghost_exchange_lmp_mliap.py`](https://github.com/mir-group/nequip/blob/27d9d2182da918ab7be0017d8300e53278f5e00e/nequip/nn/_ghost_exchange_lmp_mliap.py).

The wrappers have been adapted to MACE-SOYO's tensor layout, receiver-first
edges, runtime checks, and model interface. The tensor-product irreps helper
also follows the MACE implementation, which credits NequIP.

## Nequix and eSEN / FAIRChem

- Projects: [atomicarchitects/nequix](https://github.com/atomicarchitects/nequix)
  and [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem).
- Licenses: MIT;
  [Nequix notice](third-party-licenses/Nequix-LICENSE.txt) and
  [FAIRChem notice](third-party-licenses/FAIRChem-LICENSE.txt).
- Local component: `EquivariantRMSNorm` in
  `mace_soyo/model/model_block.py`.
- Primary reference:
  [Nequix `nequix/torch_impl/layer_norm.py`](https://github.com/atomicarchitects/nequix/blob/da0fb241f417dad1afafa4e723ad867667ee7445/nequix/torch_impl/layer_norm.py).
- That implementation identifies its eSEN / EquiformerV2 source chain:
  [eSEN `src/fairchem/core/models/esen/nn/layer_norm.py`](https://github.com/facebookresearch/fairchem/blob/977a803/src/fairchem/core/models/esen/nn/layer_norm.py).

MACE-SOYO adapts the degree-balanced normalization to `ir_mul` layout, compact
affine parameters, and its scalar-centering option. The source attribution is
Nequix/eSEN, not EquFlashV2 or DPA4.

## e3nn

- Project: [e3nn/e3nn](https://github.com/e3nn/e3nn).
- License: MIT; [original notice](third-party-licenses/e3nn-LICENSE.txt).
- Local component: `_normalize2mom_cst` and its use in the equivariant gate in
  `mace_soyo/model/model_block.py`.
- Underlying activation-normalization reference:
  [`e3nn/math/_normalize_activation.py`](https://github.com/e3nn/e3nn/blob/2aa7f58440a06b15352a2cbce01fa4c26f824969/e3nn/math/_normalize_activation.py).

The activation-normalization implementation is adapted through MACE's
`gate.py` and retains this upstream attribution chain. It is implemented using
PyTorch here; retaining the notice does not add an e3nn runtime dependency.
