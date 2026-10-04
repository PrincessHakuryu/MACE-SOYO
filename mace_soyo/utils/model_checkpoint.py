"""Save, resume, export and transfer the current checkpoint format."""

import contextlib
from pathlib import Path

import torch

from mace_soyo.model import MACESoyo

CHECKPOINT_FORMAT_VERSION = 4


def read_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Expected a format-4 MACE-SOYO checkpoint. Older training checkpoints are not supported.")
    if "model_state_dict" not in checkpoint or "model_kwargs" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and model_kwargs.")
    return checkpoint


def _write_checkpoint(payload, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


@torch.no_grad()
def save_best_checkpoint(raw_model, save_path, model_kwargs, dataset_metadata):
    """The caller selects EMA parameters and decides whether this is the best model."""
    _write_checkpoint({
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": raw_model.state_dict(),
        "model_kwargs": model_kwargs,
        "datasets": dataset_metadata,
    }, save_path)


@torch.no_grad()
def save_checkpoint(epoch, raw_model, optimizers, schedulers, ema, best_train,
                    best_valid, save_path, model_kwargs, dataset_metadata,
                    steps_per_epoch, batch_size):
    _write_checkpoint({
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dicts": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scheduler_state_dicts": {name: sched.state_dict() for name, sched in schedulers.items()},
        "ema_state_dict": ema.state_dict(),
        "best_train": best_train,
        "best_valid": best_valid,
        "model_kwargs": model_kwargs,
        "datasets": dataset_metadata,
    }, save_path)


def align_resume_schedulers(schedulers, completed_epochs, steps_per_epoch, log):
    """Resume on the current config's LR curve, keeping loaded optimizer moments."""
    target_step = completed_epochs * steps_per_epoch
    for name, scheduler in schedulers.items():
        old_step = scheduler.last_epoch
        scheduler.last_epoch = target_step
        scheduler._step_count = target_step + 1
        rates = [base_lr * curve(target_step)
                 for base_lr, curve in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        for group, lr, base_lr in zip(scheduler.optimizer.param_groups, rates, scheduler.base_lrs):
            group["lr"] = lr
            group["initial_lr"] = base_lr
        scheduler._last_lr = rates
        log.info(f"resume scheduler {name}: step {old_step} -> {target_step}, "
                 f"completed_epochs={completed_epochs}, steps_per_epoch={steps_per_epoch}, "
                 f"lr={rates}; using current config LR curve")


def load_checkpoint(raw_model, optimizers, schedulers, ema, ckpt_path, device,
                    log, dataset_metadata, steps_per_epoch, batch_size):
    log.info(f"Loading checkpoint from {ckpt_path} ...")
    checkpoint = read_checkpoint(ckpt_path, device)
    saved_kwargs = checkpoint["model_kwargs"]
    if saved_kwargs["cutoff"] != raw_model.cutoff:
        log.warning(f"[resume WARNING] cutoff changed: saved={saved_kwargs['cutoff']}, "
                    f"current={raw_model.cutoff}; using the current neighbor environment.")
    if checkpoint["batch_size"] != batch_size:
        log.warning(f"[resume WARNING] batch_size changed: saved={checkpoint['batch_size']}, "
                    f"current={batch_size}; using completed epochs and current steps_per_epoch "
                    "for the LR schedule, without automatic LR scaling.")
    for key in ("num_heads", "head_names", "readout_hidden"):
        if saved_kwargs[key] != getattr(raw_model, key):
            raise ValueError(f"Resume requires unchanged {key}; use finetune for head changes.")
    saved_pbc = [(entry["name"], entry["pbc"]) for entry in checkpoint["datasets"]]
    current_pbc = [(entry["name"], entry["pbc"]) for entry in dataset_metadata]
    if saved_pbc != current_pbc:
        raise ValueError("Resume requires the same dataset/head PBC mapping.")
    if set(checkpoint["optimizer_state_dicts"]) != set(optimizers):
        raise ValueError("Checkpoint optimizer names differ from the current run.")
    if set(checkpoint["scheduler_state_dicts"]) != set(schedulers):
        raise ValueError("Checkpoint scheduler names differ from the current run.")

    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])
    for name, scheduler in schedulers.items():
        current_base_lrs = scheduler.base_lrs.copy()
        scheduler.load_state_dict(checkpoint["scheduler_state_dicts"][name])
        scheduler.base_lrs = current_base_lrs
    ema.load_state_dict(checkpoint["ema_state_dict"])
    ema.to(device)
    start_epoch = checkpoint["epoch"] + 1
    align_resume_schedulers(schedulers, start_epoch, steps_per_epoch, log)
    return start_epoch, checkpoint["best_train"], checkpoint["best_valid"]


def load_finetune_weights(raw_model, ckpt_path, device, reset_e0=False, new_e0=None):
    checkpoint = read_checkpoint(ckpt_path, device)
    state = checkpoint["model_state_dict"]
    source_kwargs = checkpoint["model_kwargs"]
    for key in ("cutoff", "node_dim", "num_layers", "num_elements", "max_l", "max_ell", "use_zbl", "use_spin_charge"):
        if source_kwargs[key] != getattr(raw_model, key):
            raise ValueError(f"Finetune changes backbone setting {key}: "
                             f"{source_kwargs[key]!r} -> {getattr(raw_model, key)!r}.")
    old_heads = source_kwargs["num_heads"]
    new_heads = raw_model.num_heads
    target_state = raw_model.state_dict()
    growing = new_heads > old_heads
    reset_e0 = reset_e0 or growing
    if reset_e0 and new_e0 is None:
        raise ValueError("Rebuilding E0 requires references from the current datasets.")

    if growing:
        # A larger head set starts with new readouts and E0; only the backbone transfers.
        state = {name: value for name, value in state.items()
                 if not name.startswith("energy_readouts.") and name != "E0"}
        state.update({name: value for name, value in target_state.items()
                      if name.startswith("energy_readouts.")})
        state["E0"] = new_e0.to(raw_model.E0).reshape_as(raw_model.E0)
        print(f"Finetune heads {old_heads} -> {new_heads}: reset all readouts and E0.")
        return raw_model.load_state_dict(state, strict=True)

    if source_kwargs["readout_hidden"] != raw_model.readout_hidden:
        raise ValueError("Retaining readouts requires unchanged readout_hidden.")
    source_names = source_kwargs["head_names"]
    if old_heads == 1:
        selected = [0]  # A single head may be renamed.
    else:
        missing = set(raw_model.head_names) - set(source_names)
        if missing:
            raise ValueError(f"Unknown finetune heads {sorted(missing)}; available: {source_names}.")
        selected = [source_names.index(name) for name in raw_model.head_names]
    indices = torch.tensor(selected, device=state["E0"].device, dtype=torch.long)
    state["E0"] = (new_e0.to(raw_model.E0).reshape_as(raw_model.E0) if reset_e0
                   else state["E0"].index_select(1, indices))

    for layer_index in range(raw_model.num_layers):
        final = layer_index == raw_model.num_layers - 1
        prefix = f"energy_readouts.{layer_index}."
        key = prefix + ("linear_in.weight" if final else "weight")
        output_width = raw_model.readout_hidden if final else 1
        # CUEQ stores scalar Linear weights in flattened [input, output] order.
        weight = state[key].reshape(raw_model.node_dim, old_heads, output_width)
        state[key] = weight.index_select(1, indices).reshape_as(target_state[key])

    prefix = f"energy_readouts.{raw_model.num_layers - 1}."
    width = raw_model.readout_hidden
    for layer_name, output_width in (("linear_hidden", width), ("linear_out", 1)):
        key = prefix + layer_name + "."
        weight = state[key + "weight"].reshape(old_heads, width, old_heads, output_width)
        weight = weight.index_select(0, indices).index_select(2, indices)
        # Preserve effective weights when CUEQ's input-width normalization changes.
        weight = weight * (new_heads / old_heads) ** 0.5
        state[key + "weight"] = weight.reshape_as(target_state[key + "weight"])
        bias = state[key + "bias"].reshape(old_heads, output_width)
        state[key + "bias"] = bias.index_select(0, indices).reshape_as(target_state[key + "bias"])
    for name, value in raw_model.named_buffers():
        if name.startswith("energy_readouts."):
            state[name] = value
    print(f"Finetune heads {old_heads} -> {new_heads}: {raw_model.head_names}; "
          f"E0={'rebuilt' if reset_e0 else 'retained'}.")
    return raw_model.load_state_dict(state, strict=True)


@contextlib.contextmanager
def _temporary_default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_model_from_checkpoint(ckpt_path, device, dtype):
    checkpoint = read_checkpoint(ckpt_path)
    if "optimizer_state_dicts" in checkpoint:
        raise ValueError("Export a best train/valid .pth (EMA weights), not checkpoint_last.pt.")
    with _temporary_default_dtype(dtype):
        model = MACESoyo(**checkpoint["model_kwargs"])
    model = model.to(device=device, dtype=dtype)
    # E0 keeps FP64 independently of the requested network precision.
    model.E0 = model.E0.double()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model
