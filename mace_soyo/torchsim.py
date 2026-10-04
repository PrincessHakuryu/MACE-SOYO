"""Stable import path for the TorchSim MACESoyo runtime."""

from mace_soyo.inference.torchsim import MACESoyoTorchSimAOTIModel, atoms_to_state

# "Model" is TorchSim's precise term, while "Calculator" is a convenient
# user-facing alias consistent with other atomistic-potential packages.
MACESoyoTorchSimCalculator = MACESoyoTorchSimAOTIModel

__all__ = [
    "MACESoyoTorchSimAOTIModel",
    "MACESoyoTorchSimCalculator",
    "atoms_to_state",
]
