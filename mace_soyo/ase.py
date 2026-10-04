"""Stable public import path for the ASE AOTI calculator."""

from mace_soyo.inference.ase_calc import MACESoyoCalculator

# Explicit alias for codebases that prefer the backend in the class name.
MACESoyoASECalculator = MACESoyoCalculator

__all__ = [
    "MACESoyoCalculator",
    "MACESoyoASECalculator",
]
