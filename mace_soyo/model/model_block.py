import torch
import torch.nn as nn
import cuequivariance as cue
import cuequivariance_torch as cuet
import torch.nn.functional as F
import ase

_NORM_CACHE = {}


def _normalize2mom_cst(fn) -> float:
    """Same idea as e3nn.math.normalize2mom / MACE GatedEquivariantBlock."""
    if fn is None:
        return 1.0

    key = id(fn)
    cached = _NORM_CACHE.get(key)
    if cached is not None:
        return cached

    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
        second_moment = fn(z).pow(2).mean()
        result = second_moment.rsqrt().item()

    _NORM_CACHE[key] = result
    return result


def _get_single_0e_slice(irreps, name="irreps"):
    """Scalar channel range in CUEQ's ir_mul layout."""
    matches = [block for (_, ir), block in zip(irreps, irreps.slices())
               if ir.l == 0 and ir.p == 1]
    if len(matches) != 1:
        raise ValueError(f"{name} must contain exactly one 0e block, got {irreps}.")
    return matches[0]


def _assert_uniform_channel_multiplicity(irreps, name="irreps"):
    if not len(irreps) or len({mul for mul, _ in irreps}) != 1:
        raise ValueError(f"{name} requires uniform channels for the uniform_1d TP; got {irreps}.")


class SpinChargeEmbedding(nn.Module):
    """MACE-OMOL additive conditioning of scalar (0e) node features."""
    def __init__(self, node_channels: int):
        super().__init__()

        self.charge_embedding = nn.Embedding(201, node_channels)
        self.spin_embedding = nn.Embedding(101, node_channels)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2 * node_channels, node_channels, bias=False),
            nn.SiLU(),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        charge = charge.to(node_feats).reshape(-1, 1)
        spin = spin.to(node_feats).reshape(-1, 1)
        mask = condition_mask.to(device=node_feats.device, dtype=torch.bool).reshape(-1, 1)

        # Sanitize missing pairs before lookup; disable their full contribution.
        charge = torch.where(mask, charge, 0.0).reshape(-1).long() + 100
        spin = torch.where(mask, spin, 1.0).reshape(-1).long()
        condition = torch.cat((self.charge_embedding(charge), self.spin_embedding(spin)), dim=-1)
        condition = torch.where(mask, self.condition_mlp(condition), 0.0)
        return node_feats + condition[batch]


class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    def __init__(self, p=6, *, cutoff=None):
        super().__init__()

        self.cutoff = None if cutoff is None else float(cutoff)
        max_zbl_range = 2.0 * float(max(ase.data.covalent_radii[1:]))
        self._limit_zbl_range = (
            self.cutoff is not None and self.cutoff < max_zbl_range
        )

        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817],
                dtype=torch.get_default_dtype(),
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

        self.register_buffer("a_exp", torch.tensor(0.300))
        self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        dist: torch.Tensor,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        Z_u = z[sender].to(torch.int64)
        Z_v = z[receiver].to(torch.int64)
        Z_u_real = Z_u.to(dtype=dist.dtype)
        Z_v_real = Z_v.to(dtype=dist.dtype)

        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u_real, self.a_exp) + torch.pow(Z_v_real, self.a_exp))
        )

        r_over_a = dist / a

        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )

        v_edges = (14.3996 * Z_u_real * Z_v_real) / dist * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        if self._limit_zbl_range:
            r_max = r_max.clamp_max(self.cutoff)

        envelope = PolynomialCutoff.calculate_envelope(dist, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope

        atomic_energy = v_edges.new_zeros(z.shape[0])
        return atomic_energy.scatter_add_(0, receiver, v_edges)


# Radial basis embedding polynomial in DimeNet and Bessel basis
class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff: float, p: float = 6.0):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(dist, self.cutoff, self.p)

    @staticmethod
    def calculate_envelope(
        dist: torch.Tensor,
        cutoff: float | torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        x = dist / cutoff

        term1 = ((p + 1.0) * (p + 2.0) / 2.0) * x.pow(p)
        term2 = (p * (p + 2.0)) * x.pow(p + 1.0)
        term3 = ((p * (p + 1.0)) / 2.0) * x.pow(p + 2.0)

        out = 1.0 - term1 + term2 - term3
        return out * (dist < cutoff)


class BesselBasis(nn.Module):
    def __init__(self, num_rbf=8, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer(
            "freq",
            torch.arange(
                1,
                num_rbf + 1,
                dtype=torch.get_default_dtype(),
            )
            * torch.pi,
        )

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        d_scaled = dist / self.cutoff

        x = self.freq * d_scaled / torch.pi

        eps = 1.0e-8
        x_safe = torch.where(torch.abs(x) < eps, torch.ones_like(x), x)

        sinc_term = torch.sin(torch.pi * x_safe) / (torch.pi * x_safe)
        sinc_term = torch.where(torch.abs(x) < eps, torch.ones_like(x), sinc_term)

        norm_factor = (2.0 / self.cutoff) ** 0.5
        scale_factor = self.freq / self.cutoff

        rbf = norm_factor * scale_factor * sinc_term
        return rbf


class AgnesiTransform(torch.nn.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )

# cuEquivariance conv fusion for tensor-product edge message passing
# Based on mir-group/nequip


def tp_out_irreps(
    irreps1: cue.Irreps, irreps2: cue.Irreps, target_irreps: cue.Irreps
):

    irreps_out_list = []
    for mul, ir_in in irreps1:
        for _, ir_edge in irreps2:
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    irreps_out_list.append((mul, ir_out))
    irreps_out = cue.Irreps("O3", irreps_out_list)
    irreps_out, _, _ = irreps_out.sort()
    return irreps_out


class CueqConvFusionWrapper(nn.Module):
    def __init__(self, conv_tp: nn.Module):
        super().__init__()
        self.conv_tp = conv_tp

        num_segment = self.conv_tp.m.buffer_num_segments[0]
        segment_size = self.conv_tp.m.operand_extent

        self.weight_numel = num_segment * segment_size

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        out = self.conv_tp(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},      # operand 1 node_feats is indexed by sender
            {0: node_feats},  # output size is determined by node_feats.shape[0]
            {0: receiver},    # output is accumulated using receiver indices
        )

        return out[0]


# MACE-like symmetric contraction
class SimpleSymmetricContraction(nn.Module):
    def __init__(self, irreps_in, irreps_out, correlation):
        super().__init__()

        self.symmetric_contraction = cuet.SymmetricContraction(
            irreps_in,
            irreps_out,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
            contraction_degree=correlation,
            num_elements=1,
            original_mace=False,
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
        )

        self.linear = cuet.Linear(
            irreps_out,
            irreps_out,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
        )

    def forward(self, x):
        # One shared contraction bank, independent of element type.
        indices = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
        out = self.symmetric_contraction(x, indices)
        return self.linear(out)


# Equivariant gate nonlinearity
class CuePostGateIRMul(nn.Module):
    """Project node and message features, add them, then gate non-scalars."""
    def __init__(self, irreps_x, irreps_agg, irreps_out,
                 scalar_act=F.silu, gate_act=torch.sigmoid):
        super().__init__()
        self.scalar_act = scalar_act
        self.gate_act = gate_act
        scalars = cue.Irreps("O3", [(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        tensors = cue.Irreps("O3", [(mul, ir) for mul, ir in irreps_out if ir.l > 0])
        if not len(scalars) or any(ir.p != 1 for _, ir in scalars):
            raise ValueError("This SiLU gate requires even scalar output blocks.")
        self.num_scalars = scalars.dim
        self.num_gates = sum(mul for mul, _ in tensors)
        if self.num_gates:
            gates = cue.Irreps("O3", f"{self.num_gates}x0e")
            gate_irreps = scalars + gates + tensors
        else:
            gate_irreps = scalars
        self.register_buffer("scalar_cst", torch.tensor(_normalize2mom_cst(scalar_act)))
        self.register_buffer("gate_cst", torch.tensor(_normalize2mom_cst(gate_act)))
        self.pre_agg = cuet.Linear(irreps_agg, gate_irreps, layout_in=cue.ir_mul, layout_out=cue.ir_mul)
        # Node scalars can generate gates; CUEQ omits forbidden scalar-to-vector paths.
        self.pre_x = cuet.Linear(irreps_x, gate_irreps, layout_in=cue.ir_mul, layout_out=cue.ir_mul)
        self.tensor_blocks = []
        feature_offset = self.num_scalars + self.num_gates
        gate_offset = 0
        for mul, ir in tensors:
            feature_slice = slice(feature_offset, feature_offset + mul * ir.dim)
            gate_slice = slice(gate_offset, gate_offset + mul)
            self.tensor_blocks.append((feature_slice, gate_slice, ir.dim, mul))
            feature_offset += mul * ir.dim
            gate_offset += mul

    def forward(self, agg, x, denom):
        y = self.pre_x(x) + self.pre_agg(agg) / denom
        scalars = self.scalar_act(y[:, :self.num_scalars]) * self.scalar_cst
        if self.num_gates == 0:
            return scalars
        gates = self.gate_act(y[:, self.num_scalars:self.num_scalars + self.num_gates]) * self.gate_cst
        outputs = [scalars]
        for feature_slice, gate_slice, dim, mul in self.tensor_blocks:
            block = y[:, feature_slice].reshape(y.shape[0], dim, mul)
            gated = block * gates[:, gate_slice].unsqueeze(1)
            outputs.append(gated.reshape(y.shape[0], dim * mul))
        return torch.cat(outputs, dim=-1)


class EquivariantRMSNorm(nn.Module):
    """Equal weight per l block; optional scalar centering and channel-wise affine."""
    def __init__(self, irreps, eps=1.0e-8, affine=True, centering=True):
        super().__init__()
        _assert_uniform_channel_multiplicity(irreps)
        self.blocks = [(block, mul, ir.dim) for (mul, ir), block in zip(irreps, irreps.slices())]
        self.scalar_index = next(index for index, (_, ir) in enumerate(irreps)
                                 if ir.l == 0 and ir.p == 1)
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.gamma = nn.Parameter(torch.ones(sum(mul for mul, _ in irreps))) if affine else None
        self.beta = nn.Parameter(torch.zeros(irreps[self.scalar_index].mul)) if affine else None

    def forward(self, x):
        fields = []
        for index, (block, mul, dim) in enumerate(self.blocks):
            field = x[:, block].reshape(x.shape[0], dim, mul)
            if self.centering and index == self.scalar_index:
                field = field - field.mean(dim=-1, keepdim=True)
            fields.append(field)
        # Average over m/channels within each l, then equally over l blocks.
        norms = [field.square().mean(dim=(1, 2), keepdim=True) / len(fields) for field in fields]
        inv_rms = torch.rsqrt(torch.stack(norms).sum(dim=0) + self.eps)
        outputs = []
        channel_offset = 0
        for index, (field, (_, mul, dim)) in enumerate(zip(fields, self.blocks)):
            out = field * inv_rms
            if self.affine:
                scale = self.gamma[channel_offset:channel_offset + mul].reshape(1, 1, mul)
                out = out * scale
            if self.affine and index == self.scalar_index:
                out = out + self.beta.reshape(1, 1, mul)
            outputs.append(out.reshape(x.shape[0], dim * mul))
            channel_offset += mul
        return torch.cat(outputs, dim=-1)


# Main interaction block
class EquivariantInteractionBlock(nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_hidden_edge,
        irreps_hidden_node,
        irreps_edge,
        irreps_out,
        irreps_sh,
        edge_weight_dim: int,
        max_correlation: int = 3,
        num_rbf: int = 8,
        first_layer: bool = False,
    ):
        super().__init__()

        self.first_layer = bool(first_layer)
        # Irreps metadata
        self.irreps_in = cue.Irreps("O3", irreps_in)
        self.irreps_out = cue.Irreps("O3", irreps_out)
        self.irreps_sh = cue.Irreps("O3", irreps_sh)
        self.irreps_hidden_edge = cue.Irreps("O3", irreps_hidden_edge)
        self.irreps_hidden_node = cue.Irreps("O3", irreps_hidden_node)
        self.irreps_edge = cue.Irreps("O3", irreps_edge)

        _assert_uniform_channel_multiplicity(self.irreps_hidden_node, "irreps_hidden_node")
        _assert_uniform_channel_multiplicity(self.irreps_edge, "irreps_edge")
        _assert_uniform_channel_multiplicity(self.irreps_hidden_edge, "irreps_hidden")
        # TP input space
        edge_scalars = _get_single_0e_slice(self.irreps_edge, "irreps_edge")
        node_scalars = _get_single_0e_slice(self.irreps_in, "irreps_in")
        edge_scalar_channels = edge_scalars.stop - edge_scalars.start
        node_scalar_channels = node_scalars.stop - node_scalars.start

        if self.first_layer:
            self.irreps_tp_in = cue.Irreps("O3", f"{edge_scalar_channels}x0e")
        else:
            self.irreps_tp_in = self.irreps_edge

        self.scalar_slice = node_scalars
        # TP operation in cuEquivariance
        self.lin_node_to_edge = cuet.Linear(
            self.irreps_in,
            self.irreps_tp_in,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
        )

        irreps_mid = tp_out_irreps(
            self.irreps_tp_in,
            self.irreps_sh,
            self.irreps_hidden_edge,
        )

        self.tp_desc = cue.descriptors.channelwise_tensor_product(
            self.irreps_tp_in,
            self.irreps_sh,
            irreps_mid,
        )

        self.tp = CueqConvFusionWrapper(
            cuet.SegmentedPolynomial(
                self.tp_desc.flatten_coefficient_modes().squeeze_modes().polynomial,
                math_dtype=torch.get_default_dtype(),
                method="uniform_1d",
            )
        )
        # Scalar TP weights and environment density
        self.prepare_sender_tp = nn.Linear(node_scalar_channels, edge_weight_dim, bias=False)
        self.prepare_receiver_tp = nn.Linear(node_scalar_channels, edge_weight_dim, bias=False)
        self.prepare_rbf_tp = nn.Linear(num_rbf, edge_weight_dim, bias=True)
        self.pair_feat = nn.Sequential(
            nn.LayerNorm(edge_weight_dim),
            nn.SiLU(),
            nn.Linear(edge_weight_dim, edge_weight_dim),
            nn.LayerNorm(edge_weight_dim),
            nn.SiLU(),
            nn.Linear(edge_weight_dim, edge_weight_dim),
            nn.LayerNorm(edge_weight_dim),
            nn.SiLU(),
            nn.Linear(edge_weight_dim, self.tp.weight_numel),
        )

        self.prepare_sender_density = nn.Linear(node_scalar_channels, edge_weight_dim, bias=False)
        self.prepare_receiver_density = nn.Linear(node_scalar_channels, edge_weight_dim, bias=False)
        self.prepare_rbf_density = nn.Linear(num_rbf, edge_weight_dim, bias=True)

        self.density_fn = nn.Sequential(
            nn.LayerNorm(edge_weight_dim),
            nn.SiLU(),
            nn.Linear(edge_weight_dim, 1),
        )

        self.alpha = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # MACE-like many-body update
        self.gate = CuePostGateIRMul(
            self.irreps_tp_in,
            irreps_mid,
            self.irreps_hidden_node,
        )

        self.post_gate_linear = cuet.Linear(
            self.irreps_hidden_node,
            self.irreps_hidden_node,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
        )

        self.agg_update = SimpleSymmetricContraction(
            irreps_in=self.irreps_hidden_node,
            irreps_out=self.irreps_out,
            correlation=max_correlation,
        )
        # Residual branch
        self.sc_lin_x = cuet.Linear(
            self.irreps_in,
            self.irreps_out,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
        )
        # Node normalization
        self.node_norm = EquivariantRMSNorm(self.irreps_in)

    def forward(
        self,
        x,
        edge_index,
        edge_sh,
        edge_rbf,
        edge_cutoff,
    ):
        j, i = edge_index

        x_res = self.sc_lin_x(x)

        x = self.node_norm(x)

        x_lin = self.lin_node_to_edge(x)

        x_scalar = x[:, self.scalar_slice]

        x_sender_tp = self.prepare_sender_tp(x_scalar)
        x_receiver_tp = self.prepare_receiver_tp(x_scalar)
        edge_rbf_tp = self.prepare_rbf_tp(edge_rbf)

        raw_tp_weights = self.pair_feat(edge_rbf_tp + x_sender_tp[j] + x_receiver_tp[i])

        tp_weights = raw_tp_weights * edge_cutoff

        x_sender_density = self.prepare_sender_density(x_scalar)
        x_receiver_density = self.prepare_receiver_density(x_scalar)
        edge_rbf_density = self.prepare_rbf_density(edge_rbf)

        edge_density = torch.tanh(
            self.density_fn(edge_rbf_density + x_sender_density[j] + x_receiver_density[i]) ** 2
        ) * edge_cutoff

        density = edge_density.new_zeros((x.shape[0], 1))
        density.scatter_add_(0, i.unsqueeze(-1), edge_density)

        agg = self.tp(
            x_lin,
            edge_sh,
            tp_weights,
            edge_index,
        )
        denom = density * self.beta + self.alpha
        agg = self.gate(agg, x_lin, denom)

        agg = self.post_gate_linear(agg)
        x_update = self.agg_update(agg)

        return x_update + x_res
