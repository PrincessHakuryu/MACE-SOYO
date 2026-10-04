import multiprocessing
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import ase.db
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import torch
import yaml
from tqdm import tqdm

from mace_soyo.utils.dataset_config import parse_pbc


def _iter_extxyz_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.extxyz"))


def _ensure_empty_output_dir(output_path: Path) -> None:
    """Create output_path, but refuse to write into an existing non-empty path."""
    if output_path.exists():
        if output_path.is_file() or any(output_path.iterdir()):
            raise FileExistsError(f"{output_path} is not an empty directory; clear it or choose another output path.")
    output_path.mkdir(parents=True, exist_ok=True)


def normalize_labels(atoms):
    """Write canonical ASE energy/forces/stress fields, never free_energy as energy."""
    results = dict(atoms.calc.results) if atoms.calc is not None else {}
    for name in ("energy", "forces", "stress"):
        if name not in results:
            if name in atoms.info:
                results[name] = atoms.info[name]
            elif name in atoms.arrays:
                results[name] = atoms.arrays[name]
    if "energy" not in results or "forces" not in results:
        raise ValueError("Each frame requires energy and forces (standard ASE names).")
    atoms.calc = SinglePointCalculator(atoms, **results)
    for name in ("energy", "forces", "stress"):
        atoms.info.pop(name, None)
        atoms.arrays.pop(name, None)


def _save_atoms_to_ase_db(
    args: Tuple[Path, Sequence[Path], int, bool, Tuple[bool, bool, bool]],
) -> Tuple[str, int, int, int, int]:
    """Write a shard and return basic stats.

    Returns:
        (db_file, seen_frames, written_frames, skipped_missing_stress, missing_stress_frames)
        Missing-stress counts include only fully periodic (TTT) frames.
    """
    db_file, file_list, worker_id, ignorewarn, pbc_override = args
    seen_frames = 0
    written_frames = 0
    skipped_missing_stress = 0
    missing_stress_frames = 0

    with ase.db.connect(str(db_file)) as db:
        for file in tqdm(file_list, position=worker_id, desc=f"worker-{worker_id}"):
            file_seen = 0
            file_missing_stress = 0

            for atoms in ase.io.iread(str(file), index=":"):
                seen_frames += 1
                file_seen += 1

                if not np.array_equal(atoms.pbc, pbc_override):
                    # A PBC change invalidates ASE's SinglePointCalculator state.
                    # Reattach the existing labels; never recompute or drop E/F.
                    results = dict(atoms.calc.results) if atoms.calc is not None else None
                    atoms.set_pbc(pbc_override)
                    if results is not None:
                        atoms.calc = SinglePointCalculator(atoms, **results)

                normalize_labels(atoms)
                if atoms.pbc.all() and "stress" not in atoms.calc.results:
                    missing_stress_frames += 1
                    file_missing_stress += 1
                    if not ignorewarn:
                        skipped_missing_stress += 1
                        continue

                db.write(atoms, data=dict(atoms.info or {}))
                written_frames += 1

            if file_missing_stress:
                if ignorewarn:
                    warnings.warn(
                        f"{file}: {file_seen} frame(s) in total; {file_missing_stress} TTT frame(s) lack stress. "
                        f"ignorewarn=True: these structures were kept.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        f"{file}: {file_seen} frame(s) in total; {file_missing_stress} TTT frame(s) lack stress. "
                        f"ignorewarn=False: these structures were skipped.",
                        RuntimeWarning,
                    )

    return (str(db_file), seen_frames, written_frames, skipped_missing_stress, missing_stress_frames)


def preprocess_extxyz_to_aselmdb(
    input_path: str,
    output_path: str,
    n_workers: int = 8,
    ignorewarn: bool = False,
    *,
    pbc: Union[bool, str, Sequence[bool]],
) -> List[Path]:
    """Convert extxyz file(s) to one or more ASE LMDB-like DB files (.aselmdb).

    Args:
        input_path: extxyz file or directory containing *.extxyz files.
        output_path: output directory. It must not already contain files.
        n_workers: number of file-level workers.
        ignorewarn: controls TTT frames without stress. If False, these frames are
            skipped. If True, they are kept, but a warning is still emitted.
            Non-TTT frames never require stress and produce no missing-stress warning.
        pbc: required True (TTT) or False (FFF). Overrides every input frame and
            is saved in the output database. Neither input PBC nor training config
            is used as a default. Keep this consistent with the dataset config.
    """
    pbc_override = parse_pbc(pbc)
    src = Path(input_path)
    dst = Path(output_path)
    _ensure_empty_output_dir(dst)

    file_paths = _iter_extxyz_files(src)
    if not file_paths:
        raise FileNotFoundError(f"No .extxyz files found in: {src}")

    n_workers = max(1, min(int(n_workers), len(file_paths)))
    if n_workers == 1:
        out = dst / "data_0000.aselmdb"
        results = [_save_atoms_to_ase_db((out, file_paths, 0, bool(ignorewarn), pbc_override))]
    else:
        chunks = np.array_split(np.array(file_paths, dtype=object), n_workers)
        db_files = [dst / f"data_{i:04d}.aselmdb" for i in range(n_workers)]
        tasks = [
            (db_files[i], [Path(p) for p in chunks[i].tolist()], i, bool(ignorewarn), pbc_override)
            for i in range(n_workers)
            if len(chunks[i])
        ]
        with multiprocessing.Pool(len(tasks)) as pool:
            results = pool.map(_save_atoms_to_ase_db, tasks)

    total_seen = sum(item[1] for item in results)
    total_written = sum(item[2] for item in results)
    total_skipped = sum(item[3] for item in results)
    total_missing_stress = sum(item[4] for item in results)

    if total_missing_stress:
        warnings.warn(
            "preprocess summary: "
            f"seen={total_seen}, written={total_written}, "
            f"missing_stress_ttt={total_missing_stress}, skipped_missing_stress_ttt={total_skipped}",
            RuntimeWarning,
        )

    if total_written == 0:
        raise RuntimeError(
            "No structures were written. The input may be empty, or all frames may be TTT without stress "
            "and ignorewarn=False. The output directory has already been created; clear it or choose another path before retrying."
        )

    return [Path(item[0]) for item in results]


def load_atom_energies_from_yaml(yaml_path: str) -> Dict[int, float]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    atom_energies = cfg.get("atom_energies", {})
    return {int(k): float(v) for k, v in atom_energies.items()}


def build_e0_tensor_from_yaml(yaml_path: str, num_elements: int = 105) -> torch.Tensor:
    mapping = load_atom_energies_from_yaml(yaml_path)
    e0 = torch.zeros(num_elements, dtype=torch.float64)
    for z, value in mapping.items():
        if 1 <= z <= num_elements:
            e0[z - 1] = float(value)
    return e0
