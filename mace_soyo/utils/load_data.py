import argparse
import bisect
import gc
import os
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Direct script execution must import this project's utils, not an installed copy.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ase
import ase.db
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mace_soyo.utils.dataset_config import parse_pbc, resolve_data_path
from mace_soyo.utils.data_paths import discover_aselmdb_files
from mace_soyo.utils.load_extxyz import (
    build_e0_tensor_from_yaml,
    load_atom_energies_from_yaml,
    preprocess_extxyz_to_aselmdb,
)
from mace_soyo.utils.geometry import atoms_geometry

warnings.filterwarnings(
    "ignore",
    message="Length of split at index 1 is 0. This might result in an empty dataset.",
)


class PBCDataset(Dataset):
    def __init__(self, lmdb_path: str, pbc, head_id=0,
                 num_elements=105, e0_elements=None):
        self.lmdb_path = Path(lmdb_path)
        self.pbc = parse_pbc(pbc)
        self.head_id = int(head_id)
        self.num_elements = int(num_elements)
        self.e0_elements = None if e0_elements is None else frozenset(e0_elements)

        # Do NOT keep ASE/LMDB handles opened in __init__.
        # PyTorch DataLoader workers are forked after the dataset is constructed;
        # inherited LMDB environments can trigger:
        #     lmdb.Error: environment ... is already open in this process
        # So we only keep file paths + row ids here, and open DB handles lazily
        # inside the process that actually calls __getitem__.
        self.db_files, self.db_ids, self.id_cumulative = self._index_dbs(self.lmdb_path)
        self.dbs: List[Optional[Any]] = [None] * len(self.db_files)
        self._db_pid: Optional[int] = None

    @staticmethod
    def _discover_db_files(path: Path) -> List[Path]:
        return discover_aselmdb_files(path)

    @staticmethod
    def _close_one_db(db: Any) -> None:
        if db is not None:
            # Do not access db.env/close(): after fork that property reopens
            # the database before closing it. Close the inherited handle directly.
            db._env.close()
            # The backend destructor calls close() again. Keep the closed handle
            # and mark this PID so it cannot trigger another implicit reopen.
            db._env_pid = os.getpid()

    @staticmethod
    def _index_dbs(path: Path):
        db_files = PBCDataset._discover_db_files(path)
        db_ids: List[List[int]] = []
        lengths: List[int] = []

        for fp in db_files:
            db = ase.db.connect(str(fp), readonly=True, use_lock_file=False)
            try:
                ids = [int(x) for x in db.ids]
            finally:
                PBCDataset._close_one_db(db)
            db_ids.append(ids)
            lengths.append(len(ids))

        if not lengths or sum(lengths) == 0:
            raise RuntimeError(f"No rows found in .aselmdb files under: {path}")

        id_cumulative = np.cumsum(lengths)
        gc.collect()
        return db_files, db_ids, id_cumulative

    def _ensure_db_open(self, db_idx: int):
        pid = os.getpid()

        # If this Dataset object crossed a fork boundary, discard any inherited
        # handles before opening fresh handles in the worker/rank process.
        if self._db_pid != pid:
            self.close()
            self._db_pid = pid

        db = self.dbs[db_idx]
        if db is None:
            db = ase.db.connect(
                str(self.db_files[db_idx]),
                readonly=True,
                use_lock_file=False,
            )
            self.dbs[db_idx] = db
        return db

    def close(self) -> None:
        dbs = getattr(self, "dbs", None)
        if dbs is not None:
            for db in dbs:
                self._close_one_db(db)
            self.dbs = [None] * len(getattr(self, "db_files", []))
        self._db_pid = None
        gc.collect()

    def __getstate__(self):
        # For spawn/pickle-based workers, never serialize live DB handles.
        state = dict(self.__dict__)
        state["dbs"] = [None] * len(state.get("db_files", []))
        state["_db_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(self.id_cumulative[-1])

    def _global_to_db_local_index(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        db_idx = bisect.bisect_right(self.id_cumulative, idx)
        local_idx = idx
        if db_idx > 0:
            local_idx = idx - int(self.id_cumulative[db_idx - 1])
        return db_idx, int(local_idx)

    def _get_row_by_global_index(self, idx: int):
        db_idx, local_idx = self._global_to_db_local_index(idx)
        row_id = self.db_ids[db_idx][local_idx]
        db = self._ensure_db_open(db_idx)
        return db._get_row(int(row_id))

    @staticmethod
    def _parse_stress(stress_raw) -> torch.Tensor:
        s = np.asarray(stress_raw, dtype=np.float64).squeeze()
        if s.shape == (3, 3):
            return torch.tensor(s, dtype=torch.float64)
        if s.shape == (6,):
            xx, yy, zz, yz, xz, xy = s.tolist()
            return torch.tensor([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=torch.float64)
        if s.shape == (9,):
            return torch.tensor(s.reshape(3, 3), dtype=torch.float64)
        raise ValueError(f"Unsupported stress shape: {s.shape}")

    def _extract_targets(self, atoms: ase.Atoms, row) -> Tuple[float, np.ndarray]:
        """Training labels are standard ASE row fields, in eV and eV/Angstrom."""
        energy = float(row.energy)
        forces = np.asarray(row.forces, dtype=np.float64)
        if forces.shape != (len(atoms), 3):
            raise ValueError(f"Forces must have shape {(len(atoms), 3)}, got {forces.shape}")
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("Energy/force labels must be finite.")
        return energy, forces

    def __getitem__(self, idx: int):
        global_idx = int(idx)
        row = self._get_row_by_global_index(global_idx)
        atoms = row.toatoms()
        energy, force = self._extract_targets(atoms, row=row)

        numbers = atoms.get_atomic_numbers()
        if not len(numbers) or numbers.min() < 1 or numbers.max() > self.num_elements:
            raise ValueError(f"{self.lmdb_path}, row {global_idx}: invalid atomic numbers for num_elements={self.num_elements}.")
        if self.e0_elements is not None:
            missing = set(numbers.tolist()) - self.e0_elements
            if missing:
                raise ValueError(f"Head {self.head_id}: E0 YAML is missing atomic numbers {sorted(missing)}.")
        z = torch.tensor(numbers, dtype=torch.long)
        pos, frac_pos, cell = prepare_geometry(atoms, self.pbc)
        stress = torch.zeros((1, 3, 3), dtype=torch.float64)
        stress_valid = False
        if all(self.pbc):
            stress_raw = row.get("stress")
            if stress_raw is not None:
                parsed = self._parse_stress(stress_raw).unsqueeze(0)
                if not torch.isfinite(parsed).all():
                    raise ValueError(f"{self.lmdb_path}, row {global_idx}: non-finite stress label.")
                stress, stress_valid = parsed, True
        data = Data(
            z=z,
            pos=pos,
            energy=torch.tensor(energy, dtype=torch.float64),
            force=torch.tensor(force, dtype=torch.float64),
            stress=stress,
            stress_mask=torch.tensor([stress_valid], dtype=torch.bool),
            pbc=torch.tensor([self.pbc], dtype=torch.bool),
            head_id=torch.tensor([self.head_id], dtype=torch.long),
            cell=cell.unsqueeze(0),
            frac_pos=frac_pos,
            num_nodes=len(numbers),
            idx=global_idx,
            structure_id=global_idx,
        )

        # OMol stores the molecular labels in row.data. row.get('charge') can
        # instead return ASE's default zero charge, even for a molecular ion.
        condition_present = True
        for name, default in (("charge", 0), ("spin", 1)):
            value = row.data.get(name)
            present = value is not None
            if present:
                value = float(value)
                if not np.isfinite(value) or not value.is_integer() or (name == "spin" and value < 1):
                    raise ValueError(f"{self.lmdb_path}, row {global_idx}: invalid molecular {name}={value}.")
                lower = -100 if name == "charge" else 1
                assert lower <= value <= 100, f"{self.lmdb_path}, row {global_idx}: {name} must be in [{lower}, 100], got {value}."
            data[name] = torch.tensor([value if present else default], dtype=torch.float64)
            condition_present = condition_present and present
        data.condition_mask = torch.tensor([condition_present], dtype=torch.bool)

        return data


def prepare_geometry(atoms, pbc):
    """Wrap only periodic axes; complete missing NON-periodic cell vectors.

    FFF molecules may have no cell. The completed cell is only a coordinate
    basis, never a periodic image or a physical volume for stress training.
    """
    frac, cell, _ = atoms_geometry(atoms, pbc=parse_pbc(pbc))
    pos = frac @ cell
    return tuple(torch.tensor(x, dtype=torch.float64) for x in (pos, frac, cell))


def rank_generator(rank=0, seed=99):
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return generator


class DistributedEvalSampler(Sampler):
    """Shard validation without duplicating samples to pad the last rank."""

    def __init__(self, dataset, rank=None, num_replicas=None):
        self.dataset = dataset
        self.rank = torch.distributed.get_rank() if rank is None else rank
        self.num_replicas = torch.distributed.get_world_size() if num_replicas is None else num_replicas

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.num_replicas))

    def set_epoch(self, epoch):
        pass


def dataloader(batch_size, percent, config, ddp=False, rank=0, seed=42, load_e0=True):
    """Read prepared E0 YAMLs; resume/retained-E0 finetuning use checkpoint E0."""
    train_sets, valid_sets, e0_heads, head_info = [], [], [], []
    resolved_train_paths = [resolve_data_path(d["train_path"]) for d in config["datasets"]]
    if len(set(resolved_train_paths)) != len(resolved_train_paths):
        raise ValueError("The same resolved training folder is assigned to multiple heads.")

    for entry in config["datasets"]:
        e0_elements = None
        if load_e0:
            e0_elements = load_atom_energies_from_yaml(resolve_data_path(entry["e0_yaml_path"])).keys()
        options = dict(pbc=entry["pbc"], head_id=entry["head_id"],
                       num_elements=config.get("num_elements", 105), e0_elements=e0_elements)
        base = PBCDataset(resolve_data_path(entry["train_path"]), **options)
        if entry["valid_path"]:
            split_mode = "manual_valid_path"
            train_set = base
            valid_set = PBCDataset(resolve_data_path(entry["valid_path"]), **options)
            if {p.resolve() for p in base.db_files} & {p.resolve() for p in valid_set.db_files}:
                raise ValueError(f"Head {entry['name']}: train and validation contain the same .aselmdb file.")
        else:
            split_mode = "random_split"
            train_set, valid_set = random_split(base, percent, generator=rank_generator(seed=seed))
        if len(train_set) == 0:
            raise ValueError(f"Head {entry['name']} has no training structures after splitting.")
        if load_e0:
            e0_heads.append(build_e0_tensor_from_yaml(
                resolve_data_path(entry["e0_yaml_path"]), num_elements=config.get("num_elements", 105)))
        train_sets.append(train_set)
        valid_sets.append(valid_set)
        head_info.append({**entry, "train_count": len(train_set), "valid_count": len(valid_set),
                          "split_mode": split_mode})

    e0 = torch.stack(e0_heads, dim=1) if load_e0 else None

    train_dataset, valid_dataset = ConcatDataset(train_sets), ConcatDataset(valid_sets)
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset, seed=seed)
                     if ddp else None)
    valid_sampler = DistributedEvalSampler(valid_dataset) if ddp else None
    loader_options = dict(batch_size=batch_size, num_workers=int(config.get("num_workers", 2)), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler,
                              generator=rank_generator(rank, seed), **loader_options)
    valid_loader = DataLoader(valid_dataset, shuffle=False, sampler=valid_sampler,
                              generator=rank_generator(rank, seed), **loader_options)
    info = {
        "train_count": len(train_dataset), "valid_count": len(valid_dataset),
        "E_0": e0,  # None when checkpoint E0 will be retained
        "head_names": config["head_names"], "datasets": head_info,
        "split_mode": head_info[0]["split_mode"] if len(head_info) == 1 else "per_head",
    }
    return train_loader, valid_loader, info



def preprocess_command(
    input_path: str, output_path: str, n_workers: int, ignorewarn: bool = False, *, pbc,
):
    out_files = preprocess_extxyz_to_aselmdb(
        input_path,
        output_path,
        n_workers=n_workers,
        ignorewarn=ignorewarn,
        pbc=pbc,
    )
    print(f"done: wrote {len(out_files)} lmdb shard(s) into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_extxyz",required=True, help="Input extxyz file or directory to preprocess.")
    parser.add_argument("--output_lmdb", required=True, help="Output directory for *.aselmdb shards.")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--pbc",
        type=parse_pbc,
        required=True,
        metavar="True/False",
        help="Required: True=fully periodic TTT, False=nonperiodic FFF; override input PBC and save it in the database.",
    )
    parser.add_argument(
        "--ignorewarn",
        action="store_true",
        help="For TTT only: keep frames without stress and warn instead of skipping them. Non-TTT frames require no stress and produce no missing-stress warning.",
    )
    args = parser.parse_args()
    preprocess_command(
        args.input_extxyz, args.output_lmdb, args.n_workers,
        ignorewarn=args.ignorewarn, pbc=args.pbc,
    )
    
