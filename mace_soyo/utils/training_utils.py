"""Random seeds and optimizer parameter assignment."""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_muon_param_groups(model):
    muon_params, adam_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Muon handles ordinary 2-D matrices; embeddings, readouts and norms use AdamW.
        use_adam = (
            parameter.ndim != 2 or min(parameter.shape) == 1
            or any(part in name.lower() for part in (
                "z_emb", "spin_charge_embedding", "energy_readouts",
                "norm", "bias", "symmetric_contraction",
            ))
        )
        if use_adam:
            adam_params.append(parameter)
        else:
            muon_params.append(parameter)
    return muon_params, adam_params
