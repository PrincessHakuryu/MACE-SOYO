"""Runtime backends used by the public :mod:`mace_soyo` package."""

from .ase_calc import MACESoyoCalculator

MACESoyoASECalculator = MACESoyoCalculator

__all__ = [
    "MACESoyoCalculator",
    "MACESoyoASECalculator",
]
