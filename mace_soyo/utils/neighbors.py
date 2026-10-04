"""Shared nvalchemi neighbor search for training and inference."""
import torch
from nvalchemiops.torch.neighbors import neighbor_list


def build_batched_neighbor_list(positions, cell, batch, pbc, cutoff,
                                batch_ptr=None, max_neighbors=1024):
    """Return [sender, receiver] edges and sender-image integer shifts.

    Row-vector convention: r_ij = pos[sender] - pos[receiver] + image @ cell.
    nvalchemi emits [receiver, sender], so reverse ONLY the two index rows.
    The image already shifts the sender and must not be negated.
    Algorithm selection is automatic; max_neighbors is allocation capacity.
    """
    cell = cell.to(device=positions.device, dtype=positions.dtype).contiguous()
    pbc = pbc.to(device=positions.device, dtype=torch.bool).reshape(-1, 3)
    pbc = pbc.expand(cell.shape[0], -1).contiguous()
    edges, _, image = neighbor_list(
        positions=positions.detach().contiguous(),
        cutoff=float(cutoff),
        batch_idx=batch.to(device=positions.device, dtype=torch.int32).contiguous(),
        batch_ptr=None if batch_ptr is None else batch_ptr.to(device=positions.device, dtype=torch.int32).contiguous(),
        cell=cell.detach(), pbc=pbc,
        method=None, max_neighbors=max_neighbors, return_neighbor_list=True,
    )
    return edges.flip(0).long().contiguous(), image.long().contiguous()


def nvgraph(unfrac_pos, cutoff, batch, lattice, pbc,
            batch_ptr=None, return_image=False):
    """Return edges and integer images, or differentiable Cartesian shifts.

    PyG batches pack atoms contiguously by graph; PBC is specified per graph.
    Compiled E/F/S requests images and applies cell shifts inside its graph.
    """
    edge_index, image = build_batched_neighbor_list(
        unfrac_pos, lattice, batch, pbc, cutoff, batch_ptr=batch_ptr,
    )
    if return_image:
        return edge_index, image
    shift = torch.bmm(
        image.to(lattice.dtype).unsqueeze(1),
        lattice[batch[edge_index[1]]],
    ).squeeze(1)
    return edge_index, shift
