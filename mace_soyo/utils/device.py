"""CUDA-only device selection for training, export and inference."""

import torch


def resolve_cuda_device(device=None):
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError(f"MACE-SOYO requires an NVIDIA CUDA device, got {device}.")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("MACE-SOYO requires an available NVIDIA CUDA GPU and a CUDA-enabled PyTorch build.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device
