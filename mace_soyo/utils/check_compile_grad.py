"""Startup parity check for compiled E/F/S and force/stress training gradients."""

import math

import torch

from mace_soyo.utils.compile_utils import TensorEFS


def _example_inputs(model, variant):
    """Small, asymmetric structures covering every head and both conditioning masks."""
    device = model.z_emb.weight.device
    dtype = model.z_emb.weight.dtype
    num_graphs = max(2 + variant, model.num_heads + variant)
    template = torch.tensor(
        [[1.1, 1.2, 1.3], [2.2, 1.3, 1.5], [1.4, 2.4, 1.8],
         [1.5, 1.6, 2.5], [2.3, 2.1, 2.2]], device=device, dtype=dtype,
    ) * min(1.0, model.cutoff / 3.0)
    elements = torch.tensor([6, 1, 8, 1, 7], device=device).clamp_max(model.num_elements)
    numbers, positions, batches, pairs = [], [], [], []
    offset = 0
    for graph in range(num_graphs):
        count = 3 + (graph + variant) % 3
        numbers.append(elements[:count])
        positions.append(template[:count])
        batches.append(torch.full((count,), graph, dtype=torch.long, device=device))
        pairs.extend((offset + i, offset + j)
                     for i in range(count) for j in range(count) if i != j)
        offset += count
    cell = torch.eye(3, dtype=dtype, device=device).repeat(num_graphs, 1, 1) * 8.0
    pbc = torch.ones((num_graphs, 3), dtype=torch.bool, device=device)
    if variant:
        pbc[1::2] = False
    args = (
        torch.cat(numbers), torch.cat(positions) / 8.0, cell, torch.cat(batches),
        torch.tensor(pairs, dtype=torch.long, device=device).T.contiguous(),
        torch.zeros((len(pairs), 3), dtype=torch.long, device=device),
        torch.arange(num_graphs, device=device) % model.num_heads, pbc,
    )
    if model.use_spin_charge:
        graph_ids = torch.arange(num_graphs, device=device)
        charge = ((graph_ids + variant) % 3 - 1).to(dtype)
        spin = (graph_ids % 2 + 1).to(dtype)
        mask = torch.ones(num_graphs, dtype=torch.bool, device=device)
        if variant:
            mask[1::2] = False
        args += (charge, spin, mask)
    return args


def _check_outputs(actual, expected, atoms_per_graph, dtype):
    tolerances = (2e-5, 1e-4, 2e-6) if dtype == torch.float32 else (1e-8, 1e-8, 1e-9)
    for index, name in enumerate(("energy/atom", "forces", "stress")):
        a, b = actual[index].detach().double(), expected[index].detach().double()
        if index == 0:
            a, b = a.reshape(-1) / atoms_per_graph, b.reshape(-1) / atoms_per_graph
        # FP32 forces can differ slightly with compiled reduction order.
        rtol = 1e-5 if dtype == torch.float32 and index == 1 else 0
        torch.testing.assert_close(
            a, b, rtol=rtol, atol=tolerances[index], msg=lambda message: f"{name}: {message}",
        )


def _check_gradients(actual, expected, named_parameters, dtype):
    rtol, atol = (1e-3, 1e-6) if dtype == torch.float32 else (1e-7, 1e-10)
    errors, norms, failures = [], [], []
    for (name, parameter), a, b in zip(named_parameters, actual, expected):
        # A compiler can materialize zeros where eager returns an unused gradient.
        if a is None and b is None:
            continue
        a = torch.zeros_like(parameter) if a is None else a
        b = torch.zeros_like(parameter) if b is None else b
        difference = (a.detach().double() - b.detach().double()).norm()
        reference = b.detach().double().norm()
        errors.append(difference.square())
        norms.append(reference.square())
        failures.append(difference > rtol * reference + atol * math.sqrt(parameter.numel()))
        if not torch.isfinite(difference) or not torch.isfinite(reference):
            raise AssertionError(f"Non-finite parameter gradient: {name}")
    if not errors:
        raise AssertionError("No parameter gradients were produced; the check is inconclusive.")
    error = torch.stack(errors).sum().sqrt().item()
    reference = torch.stack(norms).sum().sqrt().item()
    if reference == 0:
        raise AssertionError("All reference gradients are zero; the check is inconclusive.")
    relative_error = error / reference
    if relative_error > rtol or torch.stack(failures).any():
        raise AssertionError(
            f"Parameter gradients differ: relative L2 error={relative_error:.3e} "
            f"(limit {rtol:.1e}); {int(torch.stack(failures).sum())} parameter tensors "
            "also exceed the per-tensor tolerance."
        )
    return relative_error


def check_compile_grad(compiled_model):
    """Validate the actual training adapter before DDP attaches gradient hooks.

    No optimizer step or .backward() is used; parameters and .grad are unchanged.
    The checked train callable stays in the adapter's cache for normal training.
    This is a startup regression check, not a guarantee for every future shape.
    """
    model = compiled_model.model #This is the original model, not the compiled wrapper.
    device, dtype = model.z_emb.weight.device, model.z_emb.weight.dtype
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named_parameters]
    eager = TensorEFS(model, create_graph=True)
    previous_mode = compiled_model.training
    compiled_model.train()
    try:
        with torch.random.fork_rng(devices=[device.index]), torch.enable_grad():
            for variant in (0, 1):
                args = _example_inputs(model, variant)
                atoms_per_graph = torch.bincount(args[3]).double()
                for index, name in enumerate(("energy", "force", "stress")):
                    reference_outputs = eager(*args)
                    reference = reference_outputs[index].double()
                    if index == 0:
                        reference = reference.reshape(-1) / atoms_per_graph
                    # Share backward weights so forward rounding cannot change them.
                    # Use nonuniform weights because the sum of forces can vanish.
                    grad_outputs = torch.linspace(
                        -0.7, 1.3, reference.numel(), device=device, dtype=reference.dtype,
                    ).reshape_as(reference)
                    reference_gradients = torch.autograd.grad(
                        reference, parameters, grad_outputs=grad_outputs, allow_unused=True,
                    )
                    reference_outputs = tuple(x.detach() for x in reference_outputs)
                    del reference #GPT-6 Astra prefer to del the useless matrix or variant, I thought it was OK.

                    outputs = compiled_model.forward_tensors(*args)
                    _check_outputs(outputs, reference_outputs, atoms_per_graph, dtype)
                    prediction = outputs[index].double()
                    if index == 0:
                        prediction = prediction.reshape(-1) / atoms_per_graph
                    gradients = torch.autograd.grad(
                        prediction, parameters, grad_outputs=grad_outputs, allow_unused=True,
                    )
                    try:
                        _check_gradients(gradients, reference_gradients, named_parameters, dtype)
                    except AssertionError as error:
                        raise AssertionError(f"{name} shared-grad_outputs check: {error}") from error
                    del outputs, prediction, grad_outputs, gradients, reference_gradients
    except Exception as error:
        raise RuntimeError(
            f"Compiled-training startup check failed with torch {torch.__version__} "
            f"on {device}. The compiler/dependency combination may be incompatible. "
            "Check the original error below (including possible CUDA/OOM errors). "
            "This phenomenon is likely possible with different pytorch versions, CUDA versions, or compiler backends. "
            "Try the recommended PyTorch 2.12.0 environment, or set "
            "compile_training: false. If it persists, submit an issue with this log "
            "and your dependency versions, or use an AI coding tool to investigate. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    finally:
        compiled_model.train(previous_mode)
