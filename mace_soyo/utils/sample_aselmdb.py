#!/usr/bin/env python3
"""Randomly sample ASELMDB structures and optionally fit element E0 values.

Examples
--------
Sample 1M structures from a folder:
    python -m mace_soyo.utils.sample_aselmdb \
        --input /path/to/train \
        --output /path/to/replay-1M \
        --sample-size 1000000

Sample 1M and fit E0 on the full input dataset:
    python -m mace_soyo.utils.sample_aselmdb \
        --input /path/to/train \
        --output /path/to/replay-1M \
        --sample-size 1000000 \
        --fit-e0 \
        --e0-output /path/to/e0.yaml

Fit E0 only on the sampled structures instead:
    ... --fit-e0 --e0-scope sample --e0-output /path/to/e0.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path

import ase.db
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Folder containing *.aselmdb files")
    p.add_argument("--output", type=Path, help="Output folder for sampled *.aselmdb files")
    p.add_argument(
        "--sample-size",
        type=int,
        help="Number of structures to sample; omit this option for fit-only mode",
    )
    p.add_argument("--seed", type=int, default=20260919, help="Random seed (default: 20260919)")
    p.add_argument("--chunk-size", type=int, default=10000, help="Structures per output LMDB shard")
    p.add_argument("--fit-e0", action="store_true", help="Fit element E0 values")
    p.add_argument("--e0-output", type=Path, help="YAML path for fitted E0; required with --fit-e0")
    p.add_argument(
        "--e0-scope",
        choices=("full", "sample"),
        default="full",
        help="Fit E0 on the full input dataset (default) or the sampled subset",
    )
    p.add_argument("--zmax", type=int, default=118, help="Maximum atomic number (default: 118)")
    p.add_argument("--progress-every", type=int, default=100000, help="Progress interval in structures")
    return p.parse_args()


def list_databases(input_dir):
    files = sorted(input_dir.glob("*.aselmdb"))
    if not files:
        raise FileNotFoundError(f"No *.aselmdb files found in {input_dir}")
    return files


def count_databases(files):
    counts = []
    total = 0
    for path in files:
        db = ase.db.connect(str(path), readonly=True, use_lock_file=False)
        try:
            count = len(db)
        finally:
            db.close()
        counts.append(count)
        total += count
    return counts, total


def fit_add_row(gram, rhs, seen, row, zmax):
    energy = float(row.energy)
    numbers = np.asarray(row.numbers, dtype=np.int64)
    if numbers.size == 0 or not np.isfinite(energy):
        raise ValueError("Found an empty structure or non-finite energy")
    max_z = int(numbers.max())
    if numbers.min() < 1:
        raise ValueError("Atomic numbers must be positive")
    if max_z > zmax:
        raise ValueError(f"Atomic number {max_z} exceeds --zmax {zmax}")
    unique, multiplicity = np.unique(numbers, return_counts=True)
    gram[np.ix_(unique, unique)] += np.outer(multiplicity, multiplicity)
    rhs[unique] += multiplicity * energy
    seen[unique] += multiplicity
    return energy * energy, int(numbers.size)


def open_output_shard(output_dir, shard_no, start, end):
    path = output_dir / f"data.{shard_no:06d}_{start:09d}_{end:09d}.aselmdb"
    return path, ase.db.connect(str(path), use_lock_file=False)


def write_e0(path, active, fitted):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("atom_energies:\n")
        for z in active:
            f.write(f"  {int(z)}: {fitted[z]:.12f}\n")


def main():
    args = parse_args()
    if args.sample_size is None and not args.fit_e0:
        raise ValueError("Specify --sample-size for sampling or --fit-e0 for fit-only mode")
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.sample_size is not None and args.output is None:
        raise ValueError("--output is required when --sample-size is used")
    if args.sample_size is None and args.output is not None:
        raise ValueError("--output is only used together with --sample-size")
    if args.chunk_size <= 0 or args.progress_every <= 0:
        raise ValueError("--chunk-size and --progress-every must be positive")
    if args.fit_e0 and args.e0_output is None:
        raise ValueError("--e0-output is required when --fit-e0 is used")
    if not args.fit_e0 and args.e0_output is not None:
        raise ValueError("--e0-output requires --fit-e0")
    if args.e0_scope == "sample" and not args.fit_e0:
        raise ValueError("--e0-scope requires --fit-e0")
    if args.zmax < 1:
        raise ValueError("--zmax must be positive")
    if args.e0_scope == "sample" and args.sample_size is None:
        raise ValueError("--e0-scope sample requires --sample-size")

    files = list_databases(args.input)
    counts, total = count_databases(files)
    if total == 0:
        raise ValueError("Input dataset is empty")
    if args.sample_size is not None and args.sample_size > total:
        raise ValueError(f"--sample-size {args.sample_size} exceeds dataset size {total}")
    if args.sample_size is not None:
        if args.output.exists():
            if any(args.output.iterdir()):
                raise FileExistsError(f"Output folder exists and is not empty: {args.output}")
            args.output.rmdir()
        args.output.mkdir(parents=True)
    if args.fit_e0 and args.e0_output.exists():
        raise FileExistsError(f"E0 output already exists: {args.e0_output}")

    rng = random.Random(args.seed)
    selected = set(rng.sample(range(total), args.sample_size)) if args.sample_size is not None else set()
    gram = np.zeros((args.zmax + 1, args.zmax + 1), dtype=np.float64)
    rhs = np.zeros(args.zmax + 1, dtype=np.float64)
    seen = np.zeros(args.zmax + 1, dtype=np.int64)
    sum_energy_sq = 0.0
    nframes = 0
    sampled = 0
    sampled_sum_energy_sq = 0.0
    sampled_gram = np.zeros_like(gram)
    sampled_rhs = np.zeros_like(rhs)
    sampled_seen = np.zeros_like(seen)
    output_db = None
    output_shard = 0
    output_start = 0
    output_in_shard = 0
    started = time.monotonic()

    try:
        for file_no, (path, expected) in enumerate(zip(files, counts), start=1):
            db = ase.db.connect(str(path), readonly=True, use_lock_file=False)
            try:
                if len(db) != expected:
                    raise RuntimeError(f"Database changed while scanning: {path}")
                for row in db.select():
                    is_selected = nframes in selected
                    if args.fit_e0 and args.e0_scope == "full":
                        e2, _ = fit_add_row(gram, rhs, seen, row, args.zmax)
                        sum_energy_sq += e2
                    if is_selected:
                        if output_db is None or output_in_shard >= args.chunk_size:
                            if output_db is not None:
                                output_db.close()
                            end = output_start + args.chunk_size
                            _, output_db = open_output_shard(args.output, output_shard, output_start, end)
                            output_shard += 1
                            output_in_shard = 0
                            output_start = end
                        atoms = row.toatoms()
                        data = dict(row.data)
                        data["sample_source_shard"] = path.name
                        data["sample_source_row"] = int(row.id)
                        output_db.write(atoms, data=data, **dict(row.key_value_pairs))
                        if args.fit_e0 and args.e0_scope == "sample":
                            e2, _ = fit_add_row(sampled_gram, sampled_rhs, sampled_seen, row, args.zmax)
                            sampled_sum_energy_sq += e2
                        sampled += 1
                        output_in_shard += 1
                    nframes += 1
                    if nframes % args.progress_every == 0:
                        print(
                            f"PROGRESS {nframes}/{total} frames, sampled {sampled}, "
                            f"{nframes / max(time.monotonic() - started, 1e-9):.1f} frames/s",
                            flush=True,
                        )
            finally:
                db.close()
        if output_db is not None:
            output_db.close()
            output_db = None
    except BaseException:
        if output_db is not None:
            output_db.close()
        raise

    if nframes != total or (args.sample_size is not None and sampled != args.sample_size):
        raise RuntimeError(
            f"Scan/sample mismatch: frames={nframes}/{total}, "
            f"sampled={sampled}/{args.sample_size}"
        )

    report = {
        "status": "complete",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()) if args.output is not None else None,
        "files": len(files),
        "frames": total,
        "sample_size": sampled if args.sample_size is not None else None,
        "seed": args.seed,
        "seconds": time.monotonic() - started,
    }
    if args.fit_e0:
        if args.e0_scope == "full":
            fit_gram, fit_rhs, fit_seen, fit_energy_sq = gram, rhs, seen, sum_energy_sq
        else:
            fit_gram, fit_rhs, fit_seen, fit_energy_sq = sampled_gram, sampled_rhs, sampled_seen, sampled_sum_energy_sq
        active = np.flatnonzero(fit_seen > 0)
        solution, _, rank, singular_values = np.linalg.lstsq(
            fit_gram[np.ix_(active, active)], fit_rhs[active], rcond=None
        )
        fitted = np.zeros(args.zmax + 1, dtype=np.float64)
        fitted[active] = solution
        residual_sse = fit_energy_sq - 2.0 * float(np.dot(fitted, fit_rhs)) + float(fitted @ fit_gram @ fitted)
        residual_sse = max(residual_sse, 0.0)
        write_e0(args.e0_output, active, fitted)
        report["e0"] = {
            "output": str(args.e0_output.resolve()),
            "scope": args.e0_scope,
            "elements": [int(z) for z in active],
            "rank": int(rank),
            "rmse_eV": float(np.sqrt(residual_sse / (total if args.e0_scope == "full" else sampled))),
            "values": {str(int(z)): float(fitted[z]) for z in active},
            "singular_values": [float(x) for x in singular_values],
        }

    if args.output is not None:
        report_path = args.output / "sample_report.json"
    elif args.fit_e0:
        report_path = args.e0_output.with_name(args.e0_output.stem + "_report.json")
    else:
        report_path = None
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
