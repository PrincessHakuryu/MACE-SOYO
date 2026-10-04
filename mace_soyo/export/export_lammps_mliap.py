#!/usr/bin/env python3
"""Export a fixed-head ML-IAP object; compilation happens inside LAMMPS."""
import argparse
from pathlib import Path
import sys
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mace_soyo.utils.model_checkpoint import load_model_from_checkpoint
from mace_soyo.export.lammps_mliap import MACESoyoMLIAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-path", type=Path, default=Path("lammps.pt"))
    parser.add_argument("--head", help="Head name; required for multi-head models")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    args = parser.parse_args()
    model = load_model_from_checkpoint(
        args.ckpt_path, torch.device("cpu"), getattr(torch, args.dtype)
    )
    unified = MACESoyoMLIAP(
        model, head=args.head, compile_model=not args.no_compile,
        compile_mode=args.compile_mode,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unified, args.output_path)
    restored = torch.load(args.output_path, map_location="cpu", weights_only=False)
    assert restored.head == unified.head and restored.runtime_model is None
    print(f"Saved {args.output_path.resolve()}")
    print(f"head={restored.head}, cutoff={2 * restored.rcutfac:g} Angstrom")
    print(f"torch.compile={restored.compile_model}; compiled on first LAMMPS call")
    print("This is a Python ML-IAP object, not an AOTI/TorchScript package.")


if __name__ == "__main__":
    main()
