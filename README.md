# MACE-SOYO

A multi-head interatomic potential with training, AOTInductor (AOTI) inference for
ASE/TorchSim, and a LAMMPS ML-IAP interface(in test). The current branch predicts energy,
forces, and stress; molecular charge/spin conditioning is optional.

This project aims to unlock the full potential of MACE-mh-1, which is my favourite model.
To me, Soyo evokes “a world of simplicity and peace,” capturing the spirit of MACE-mh-1: simple in design, 
reliable in practice, and versatile in application. 
Broadly speaking, MACE-SOYO offers an accuracy–speed trade-off between DPA4-Neo and DPA4-Mini.
Training, AOTI export and inference require an NVIDIA CUDA GPU. CPU and other
device types are rejected explicitly due to cuequivariance and nvalchemi-toolkit.

This is an independently developed project with assistance from GPT-6 Astra. Initial
benchmarks are encouraging; broader code validation is ongoing.

## Installation

From the repository root (with PyTorch already installed):
python>=3.12 is recommended.
```bash
pip install -r requirements.txt
```

## Checkpoints

Pretrained checkpoints and datasets are not bundled with this source release.
Only a two-head model is currently available.
A nine-head model trained on the full datasets may be released in the future:
OMat24, sAlex + MPTrj, OC20, OC22, OC25, OMol25 + OPoly26, ODAC25, OMC25, and
MatPES-r2SCAN.

## Training

### Recommended PyTorch version

We strongly recommend **PyTorch 2.12.0** for training. Install it before
the project requirements.

Other PyTorch versions may have incompatibilities with `make_fx`, compiled
force/stress training, or dynamic shapes. Disable compile_training in config.yml
to avoid any incompatibilities.

### Data and configuration

Training reads ASE LMDB (`.aselmdb`) files. Each entry in `datasets` defines one
named head; a directory may contain multiple shards. Copy the configuration
template before your first run:

```bash
cp config/config.example.yml config/config.yml
```

Edit the dataset paths and training settings in `config/config.yml`. This local
configuration is excluded from Git:

```yaml
num_heads: 2
readout_hidden: 64
max_correlations: 3
use_spin_charge: false
datasets:
  - name: omat_pbe
    train_path: /data/omat24/train
    valid_path: /data/omat24/val
    pbc: true
    e0_yaml_path: ./config/e0_omat.yaml
  - name: salexmp
    train_path: /data/salex-mptraj/train
    valid_path: /data/salex-mptraj/val
    pbc: true
    e0_yaml_path: ./config/e0_salex.yaml
```

`num_heads` must equal the number of dataset entries. Head names are stored in
checkpoints and AOTI packages. `pbc: true` means TTT; `false` means FFF and allows
molecules without a cell. Periodic structures require a valid three-dimensional
cell. An empty `valid_path` splits that dataset using `train_percent`,
`valid_percent`, and `split_seed`.

`max_correlations` sets the maximum symmetric-contraction degree in every
interaction layer (default: 3). It is stored in checkpoints and restored for
export. Keep it unchanged when resuming or fine-tuning an existing architecture.

Energy and forces are required, in eV and eV/Å. Stress uses ASE's eV/Å³ convention;
missing stress labels and nonperiodic structures are excluded from stress loss.
Floating-point training data, E0 references, total-energy accumulation, and
E/F/stress losses use FP64. Set `model_dtype: float32` (default) or `float64`
to select network precision. Geometry, predicted forces/stress, and parameter
gradients follow the network dtype.
Prepare an E0 YAML for each head before training:

```yaml
atom_energies:
  1: -13.6
  8: -400.0
```

These values illustrate the format only; use references appropriate to the actual
dataset and DFT settings. Every element in the dataset must be present. Training
does not fit E0 automatically. To fit E0 externally using only training data:

```bash
python -m mace_soyo.utils.sample_aselmdb --input /data/train --fit-e0 --e0-output /data/e0.yaml
```

Training and the E0/sampling tool accept one file or discover `.aselmdb` shards
recursively under a directory; the tool's JSON report records the input file list
and shard count.

To convert extxyz data into ASE LMDB:

```bash
python -m mace_soyo.utils.load_data \
  --input_extxyz /data/input.extxyz \
  --output_lmdb /data/train \
  --pbc True \
  --ignorewarn
```

Use `--input_extxyz /data/` to process a directory of extxyz files.
Use `--pbc False` for nonperiodic molecules. For periodic inputs, `--ignorewarn`
keeps frames without stress; without it, those frames are skipped.

### Launch

Run from the repository root. GPU and node counts are controlled by `torchrun`,
not by configuration fields:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run_ddp.py
```

Default checkpoints are saved in `model_pth/` beside `run_ddp.py`, regardless of
the launch directory. The directory is created automatically on the first save.

For a new run, set `resume: false` and `finetune: false`. For continuation, set
`resume: true` and `resume_path`, keeping the same architecture and heads.
New training checkpoints use format 4 and named readout layers; older training
checkpoints are not supported. Previously exported compatible `.pt2` files are unchanged.
For fine-tuning, set `finetune: true` and `finetune_path` with `resume: false`:

- More heads: keep the backbone and rebuild all readouts and E0 values.
- Fewer heads: retain heads selected by their existing names.
- The same head count: a single head may be renamed; multiple heads must retain
  the same name set. When retaining readouts, keep `readout_hidden` unchanged.
- `finetune_reset_e0: true` reloads E0 from the configured YAML files; otherwise
  retained heads use checkpoint E0. Fine-tuning starts a new optimizer/schedule.

For molecular conditioning, set `use_spin_charge: true`. The loader reads graph
labels `charge` and `spin` from ASE row data. Spin means multiplicity (singlet = 1).
Conditioning is enabled only when both labels are present; it does not predict
atomic charges or use atomic magnetic moments as molecular spin.
Charge must be an integer in [-100, 100] and spin in [1, 100].

## AOTI export

Export a trained checkpoint using a representative structure:

```bash
python -m mace_soyo.export.AOTI_export \
  --ckpt-path /path/best.pth \
  --structure-file /path/structure.extxyz \
  --output-path model.pt2 \
  --device cuda:0 \
  --dtype float32 \
  --max-graphs 20000
```

Use a structure with at least two atoms and a nonempty neighbor list for tracing.
The tracing example always uses B=2. Every new package supports variable batch
sizes, including B=1 for ASE. `--max-graphs` defaults to 20000 and must be at
least 2. Optional `--max-nodes` and `--max-edges` set upper bounds on atoms and
directed edges.

One `.pt2` contains all heads and their names. The exporter checks energy, forces,
and stress against the eager model for every head. Energy error must be at most
`2e-5 eV/atom` (0.02 meV/atom), with no relative tolerance. Charge/spin inputs are included
automatically when the checkpoint enables them. D3 is not embedded in the package.
Export `--dtype` is independent of training `model_dtype`; its default remains
`float32`, while E0 and output energies remain FP64.
Use compatible PyTorch/CUDA/CUEQ versions for export and inference. A package
exported for CUDA requires a compatible CUDA device.

## ASE

```python
from ase.io import read, write
from ase.optimize import FIRE
from mace_soyo import MACESoyoCalculator

atoms = read("input.extxyz")
atoms.calc = MACESoyoCalculator(
    package_path="/absolute/path/model.pt2",
    head="salexmp",
    device="cuda:0",
    use_d3=False,
)

print(atoms.get_potential_energy())
print(atoms.get_forces())
if atoms.pbc.all():
    print(atoms.get_stress())

FIRE(atoms).run(fmax=0.05, steps=1000)
write("relaxed.extxyz", atoms)
```

This optimizes atomic positions at fixed cell. For atomic and cell relaxation,
pass `ase.filters.FrechetCellFilter(atoms)` to FIRE instead (periodic structures
with a stress-capable head only).

Choose an exact saved head name; invalid names report the available heads. A
single-head package may omit `head`. `calc.head_names` lists the names. Each
calculator selects one head at construction. Boundary conditions come from
`atoms.pbc`, not from the head name. Non-TTT stress is a zero placeholder.
Outputs use eV, eV/Å, and eV/Å³.

For a charge/spin-conditioned molecular package, set labels on each structure
before calculation:

```python
atoms.info.update(charge=1, spin=1)
```

Both labels are needed to enable conditioning. Missing either disables it for
that structure; it does not imply a neutral singlet. Do not pass valid electronic
labels to an unconditioned package.

## TorchSim

```python
import torch_sim as ts
from ase.io import read, write
from mace_soyo import MACESoyoTorchSimAOTIModel
from mace_soyo.torchsim import atoms_to_state

model = MACESoyoTorchSimAOTIModel(
    package_path="/absolute/path/model.pt2",
    head="salexmp",
    device="cuda:0",
    use_d3=False,
)
atoms_list = read("input.extxyz", index=":")
state = atoms_to_state(atoms_list, device=model.device, dtype=model.dtype)
results = model(state)
print(results["energy"], results["forces"], results["stress"])

final_state = ts.optimize(
    state,
    model,
    optimizer=ts.Optimizer.fire,
    convergence_fn=ts.generate_force_convergence_fn(force_tol=0.05),
    max_steps=1000,
)
write("relaxed.extxyz", final_state.to_atoms())
```

Use `atoms_to_state` to retain `atoms.info['charge']` and `atoms.info['spin']` when
present. The batch can mix labeled and unlabeled systems; one model instance uses
one head throughout. The example relaxes positions at fixed cell. Match
`model.device` and `model.dtype` when constructing a state.

ASE and TorchSim support optional ordinary D3(BJ) with `use_d3=True`. Only enable
it when the selected head's labels DO NOT already include that dispersion
correction. Both interfaces also support `use_laspd3=True` after building
`other-modules/LASP-D3-torchsim` and installing its shared library in `mace_soyo/utils/`;
LASP-D3 requires TTT(CELL IS NEEDED). Do not enable both D3 backends together.

`d3_cutoff_radius` is in Å for both backends, independently of the neural-network
cutoff. The default is 24.59394 Å. LASP-D3 is up to 3x faster than D3 in nvalchemi-toolkit.
But it needs to be manually installed.

For LASP-D3(BJ) in ASE:

```python
calc = MACESoyoCalculator(
    package_path="/absolute/path/model.pt2",
    head="salexmp",
    use_laspd3=True,
    use_laspd3_BJ=True,
    functional_type=0,  # LASP-D3's PBE parameter set
)
atoms.calc = calc
FIRE(atoms).run(fmax=0.05, steps=1000)
calc.close()
```

LASP-D3 uses zero damping unless `use_laspd3_BJ=True`. Ordinary D3 parameters
`a1`, `a2`, `s8`, `s6`, and
`d3_params_path` apply only to `use_d3=True`; LASP-D3 selects its parameters with
`functional_type` and `use_laspd3_BJ`.

With CUDA (`nvcc`), a C++ compiler, a Fortran compiler and CMake >= 3.25.2:

```bash
cmake -S other-modules/LASP-D3-torchsim -B build/lasp-d3 -DBUILD_D3_TESTS=OFF
cmake --build build/lasp-d3 --target d3_shared -j
cp build/lasp-d3/libd3.so.1.0.0 mace_soyo/utils/
pip install --no-deps .
```

Build before installation, or reinstall after adding the shared library.

LASP-D3 source: [LipidL/LASP-D3](https://github.com/LipidL/LASP-D3).

## LAMMPS ML-IAP

This interface is experimental and completely written by GPT-6 Astra; 
its correctness has not yet been fully validated.

This exports a fixed-head Python/PyTorch object (`lammps.pt`), not AOTI or
TorchScript. Use a LAMMPS build with PYTHON, ML-IAP unified Python support, and
CUDA Kokkos. The embedded Python environment must contain MACE-SOYO, PyTorch,
CUEQ, and CuPy:

```bash
python -m pip install cupy-cuda12x
python -m mace_soyo.export.export_lammps_mliap \
  --ckpt-path /path/best.pth \
  --head salexmp \
  --output-path lammps.pt
```

The selected head is fixed at export. Compilation occurs on first use; add
`--no-compile` when exporting to use eager execution. This interface does not
add D3 or expose molecular charge/spin conditioning.

Use `units metal`, `newton on`, and element names matching the LAMMPS atom-type
order. For a C/H system with a 6 Å model cutoff and 1 Å neighbor skin:

```text
units metal
atom_style atomic
newton on
read_data structure.data
neighbor 1.0 bin
comm_modify cutoff 7.0
pair_style mliap unified lammps.pt 0
pair_coeff * * C H
```

Keep the final `0` in `pair_style`; the adapter exchanges local/ghost node
features between layers. Communication cutoff must cover the model cutoff plus
neighbor skin. See `example/in.lammps_mliap` for a complete input.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 lmp -k on g 1 -sf kk \
  -pk kokkos newton on neigh half comm device gpu/aware on -in in.lammps
```

Two GPUs, one MPI rank per GPU (CUDA-aware MPI required):

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 mpirun -np 2 lmp -k on g 2 -sf kk \
  -pk kokkos newton on neigh half comm device gpu/aware on -in in.lammps
```

## License

MACE-SOYO's original code is licensed under the [MIT License](LICENSE).
The vendored sources in `other-modules/LASP-D3-torchsim/` are adapted from
[LipidL/LASP-D3](https://github.com/LipidL/LASP-D3) for GPU-accelerated
dispersion calculations in ASE and TorchSim.
