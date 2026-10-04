"""MACE-style scalar readout: dense equivariant Linear + per-head masks.

Uses MACE's masked-head structure with CUEQ Linear and the original SiLU.
https://github.com/ACEsuit/mace/blob/develop/mace/modules/blocks.py
"""

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from torch import nn


def mask_head(x, heads, num_heads):
    """Keep each node's selected contiguous head block (MACE mask semantics)."""
    mask = heads.reshape(-1, 1) == torch.arange(num_heads, device=x.device)
    return (x.reshape(x.shape[0], num_heads, -1) * mask.unsqueeze(-1)).reshape_as(x)


class ScalarLinear(cuet.Linear):
    """CUEQ scalar Linear plus bias (CUEQ Linear has no built-in bias argument)."""

    def __init__(self, in_features, out_features):
        super().__init__(cue.Irreps("O3", f"{in_features}x0e"),
                         cue.Irreps("O3", f"{out_features}x0e"),
                         layout_in=cue.ir_mul, layout_out=cue.ir_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return super().forward(x) + self.bias


class NonLinearReadout(nn.Module):
    """Two hidden scalar layers with independent selected-head activations."""

    def __init__(self, irreps_in, num_heads, hidden):
        super().__init__()
        width = num_heads * hidden
        hidden_irreps = cue.Irreps("O3", f"{width}x0e")
        self.linear_in = cuet.Linear(
            irreps_in, hidden_irreps, layout_in=cue.ir_mul, layout_out=cue.ir_mul,
        )
        self.linear_hidden = ScalarLinear(width, width)
        self.linear_out = ScalarLinear(width, num_heads)
        self.activation = nn.SiLU()
        self.num_heads = num_heads

    def forward(self, x, heads):
        x = self.activation(self.linear_in(x))
        if self.num_heads > 1:
            x = mask_head(x, heads, self.num_heads)
        x = self.activation(self.linear_hidden(x))
        if self.num_heads > 1:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_out(x)
