# -*- coding: utf-8 -*-
import datetime
import math
import os
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage

from mace_soyo.utils.load_data import dataloader
from mace_soyo.model import MACESoyo
from mace_soyo.utils.dataset_config import read_config
from mace_soyo.utils.logger import GpuLogger
from mace_soyo.utils.device import resolve_cuda_device
from mace_soyo.utils.loss import efs_loss, weighted_loss_score
from mace_soyo.utils.metrics import EpochMetrics, log_epoch_metrics
from mace_soyo.utils.model_checkpoint import (
    read_checkpoint, load_checkpoint, load_finetune_weights,
    save_checkpoint, save_best_checkpoint,
)
from mace_soyo.utils.training_utils import build_muon_param_groups, set_seed

warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


def check_finite(name, x, idx):
    if not torch.isfinite(x).all():
        print(
            f"[NaN/Inf detected] batch={idx} {name} "
            f"shape={tuple(x.shape)} "
            f"min={torch.nan_to_num(x).min().item()} "
            f"max={torch.nan_to_num(x).max().item()}",
            flush=True,
        )
        raise FloatingPointError(f"{name} has NaN/Inf")


def build_warmup_stable_decay_lambda(
    total_steps: int,
    warmup_steps: int,
    warmup_factor: float,
    max_lr: float,
    min_lr: float,
    decay_phase_ratio: float,
):
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0 for WSD scheduler.")
    if max_lr <= 0:
        raise ValueError("max_lr must be > 0 for WSD scheduler.")
    if min_lr < 0:
        raise ValueError("min_lr must be >= 0 for WSD scheduler.")
    if min_lr > max_lr:
        raise ValueError("min_lr must be <= max_lr for WSD scheduler.")
    if not 0.0 < decay_phase_ratio <= 1.0:
        raise ValueError("decay_phase_ratio must be in (0, 1] for WSD scheduler.")

    warmup_steps = min(max(int(warmup_steps), 0), total_steps)
    warmup_factor = min(max(float(warmup_factor), 0.0), 1.0)
    warmup_start_ratio = warmup_factor
    min_ratio = min_lr / max_lr

    post_warmup_steps = total_steps - warmup_steps
    decay_steps = min(
        max(int(math.floor(decay_phase_ratio * total_steps)), 1),
        max(post_warmup_steps, 1),
    )
    stable_steps = max(post_warmup_steps - decay_steps, 0)
    decay_start = warmup_steps + stable_steps

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            if warmup_steps == 1:
                return 1.0
            warmup_progress = float(step_idx) / float(warmup_steps - 1)
            warmup_ratio = warmup_start_ratio + (1.0 - warmup_start_ratio) * warmup_progress
            return max(min_ratio, warmup_ratio)

        if step_idx < decay_start:
            return 1.0

        if decay_steps <= 1:
            progress = 1.0
        else:
            decay_step = min(max(step_idx - decay_start, 0), decay_steps - 1)
            progress = float(decay_step) / float(decay_steps - 1)
        cosine_ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_ratio

    return lr_lambda, warmup_steps, stable_steps, decay_steps


def main():
    config_data = read_config()
    model_dtype = torch.float64 if config_data["model_dtype"] == "float64" else torch.float32
    # Construct weights and CUEQ constants in the selected precision from the
    # start; E0 and floating-point training data remain independently FP64.
    torch.set_default_dtype(model_dtype)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = resolve_cuda_device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=1800),
    )

    rank = int(dist.get_rank())
    world_size = dist.get_world_size()

    log = GpuLogger(dist_id=rank, local_id=local_rank)
    log.info(f"[rank] {rank}   [local_rank] {local_rank}")
    log.info(f"device: {device}")
    log.info(f"model_dtype: {model_dtype}; data, E0 and E/F/stress losses: torch.float64")
    log.info(f"world_size: {world_size}")

    run_time = datetime.datetime.now()
    log.info(f"run_time: {run_time}")
    seed_id = int(config_data.get("seed", 721))
    set_seed(seed_id)

    log.info(f"seed_id: {seed_id}")
    batch_size = config_data["batch_size"]
    cutoff = config_data["cutoff"]
    log.info(f"batch_size: {batch_size}")
    log.info(f"cutoff: {cutoff}")

    resume = bool(config_data.get("resume", False))
    finetune = bool(config_data.get("finetune", False))
    finetune_reset_e0 = bool(config_data.get("finetune_reset_e0", False))
    if resume and finetune:
        raise ValueError("resume and finetune cannot both be True.")
    load_e0 = not resume
    if finetune:
        finetune_path = config_data["finetune_path"]
        source = read_checkpoint(finetune_path)
        old_heads = source["model_kwargs"]["num_heads"]
        load_e0 = finetune_reset_e0 or config_data["num_heads"] > old_heads
        del source
    log.info(f"E0 source: {'prepared YAML files' if load_e0 else 'checkpoint'}")

    start_time = datetime.datetime.now()
    percent = [config_data["train_percent"], config_data["valid_percent"]]
    split_seed = int(config_data.get("split_seed", 42))
    log.info(f"split_seed: {split_seed}")
    train_loader, valid_loader, info = dataloader(
        batch_size,
        percent,
        load_e0=load_e0,
        ddp=True,
        rank=rank,
        seed=split_seed,
        config=config_data,
    )
    end_time = datetime.datetime.now()

    log.info(f"load time: {end_time - start_time}")
    log.info(f"info [train]:{info['train_count']} [valid]:{info['valid_count']}")
    log.info(f"split_mode: {info.get('split_mode', 'random_split')}")
    for dataset_info in info["datasets"]:
        log.info(
            f"[DATASET {dataset_info['head_id']}:{dataset_info['name']}] "
            f"pbc={''.join('T' if p else 'F' for p in dataset_info['pbc'])} "
            f"train={dataset_info['train_count']} valid={dataset_info['valid_count']} "
            f"stress={'available TTT labels only' if all(dataset_info['pbc']) else 'disabled'} "
            f"E0={dataset_info['e0_yaml_path']} split={dataset_info['split_mode']}"
        )

    node_dim = config_data["node_dim"]
    edge_weight_dim = config_data["edge_weight_dim"]
    num_layers = config_data["num_layers"]
    pair_dim = config_data["pair_dim"]
    num_rbf = config_data["num_rbf"]
    max_l = config_data.get("max_l", 1)
    max_ell = config_data.get("max_ell", 3)
    num_heads = config_data["num_heads"]
    readout_hidden = config_data["readout_hidden"]
    use_zbl = config_data.get("use_zbl", False)
    E_factor = config_data["E_factor"]
    F_factor = config_data["F_factor"]
    S_factor = config_data["S_factor"]

    log.info(f"node_dim: {node_dim}")
    log.info(f"num_layers: {num_layers}")
    log.info(f"pair_dim: {pair_dim}")
    log.info(f"num_rbf: {num_rbf}")
    log.info(f"max_l: {max_l}")
    log.info(f"max_ell: {max_ell}")
    log.info(f"num_heads: {num_heads}; head_names: {info['head_names']}")
    log.info(f"readout_hidden per head: {readout_hidden}; total: {readout_hidden * num_heads}")
    log.info("neighbor search: nvalchemi automatic selection")
    log.info(f"use_zbl: {use_zbl}")
    log.info(f"edge_weight_dim: {edge_weight_dim}")
    log.info(f"Config E_factor: {E_factor}")
    log.info(f"Config F_factor: {F_factor}")
    log.info(f"Config S_factor: {S_factor}")
    model_kwargs = dict(
        cutoff=cutoff,
        node_dim=node_dim,
        edge_weight_dim=edge_weight_dim,
        num_layers=num_layers,
        pair_dim=pair_dim,
        num_rbf=num_rbf,
        max_l=max_l,
        max_ell=max_ell,
        num_elements=config_data.get("num_elements", 105),
        max_correlation=config_data.get("max_correlation", 3),
        num_heads=num_heads,
        readout_hidden=readout_hidden,
        head_names=info["head_names"],
        use_zbl=use_zbl,
        use_spin_charge=bool(config_data.get("use_spin_charge", False)),
    )

    raw_model = MACESoyo(E_0=info["E_0"], **model_kwargs)
    raw_model = raw_model.to(device)

    if finetune:
        load_finetune_weights(
            raw_model=raw_model, ckpt_path=finetune_path, device=device,
            reset_e0=finetune_reset_e0, new_e0=info["E_0"],
        )
        log.info("Finetune weights loaded; initializing optimizer and EMA from these weights.")

    efs_model = raw_model
    if config_data.get("compile_training", False):
        from mace_soyo.utils.compile_utils import CompiledTrainingModel
        efs_model = CompiledTrainingModel(raw_model)
        log.info("compile_training=True: startup gradient check, then separate train/eval compiled graphs.")

    ema_decay = config_data.get("ema_decay", 0.999)
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    ema = ExponentialMovingAverage(trainable_params, decay=ema_decay)
    log.info(f"EMA initialized with decay {ema_decay}")

    max_lr = float(config_data.get("max_lr", 1e-2))
    min_lr = float(config_data.get("min_lr", 1e-6))
    warmup_factor = float(config_data.get("warmup_factor", 0.2))
    warmup_epochs = float(config_data.get("warmup_epochs", 0.1))
    wsd_decay_ratio = float(config_data.get("wsd_decay_ratio", 0.65))
    log.info(f"max_lr: {max_lr}")
    log.info(f"min_lr: {min_lr}")
    log.info(f"warmup_factor: {warmup_factor}")
    log.info(f"warmup_epochs(config): {warmup_epochs}")
    log.info(f"wsd_decay_ratio: {wsd_decay_ratio}")

    muon_params, adam_params = build_muon_param_groups(raw_model)

    log.info(f"Muon params: {len(muon_params)} tensors / {sum(p.numel() for p in muon_params)} params")
    log.info(f"Adam params: {len(adam_params)} tensors / {sum(p.numel() for p in adam_params)} params")

    optimizers = {}

    if muon_params:
        optimizers["muon"] = torch.optim.Muon(
            muon_params,
            lr=max_lr,
            weight_decay=1e-3,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            # Reuse AdamW-style learning rates; choose "original" for standard Muon scaling.
            adjust_lr_fn="match_rms_adamw",
        )

    if adam_params:
        optimizers["adamw"] = torch.optim.AdamW(
            adam_params,
            lr=max_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-3,
            amsgrad=False,
        )

    if not optimizers:
        raise RuntimeError("No trainable parameters found.")

    lr_source_name = "muon" if "muon" in optimizers else "adamw"
    lr_source = optimizers[lr_source_name]
    interval = max(int(2560 / batch_size), 1)


    log.info(f"Total parameters: {sum(p.numel() for p in raw_model.parameters())}")
    log.info(f"Trainable parameters: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad)}")
    log.info(f"Learning-rate source: {lr_source_name}")

    criterion = torch.nn.HuberLoss(delta=0.01, reduction="none").to(device)
    log.info(f"criterion: {criterion}")
    loss_factors = (E_factor, F_factor, S_factor)
    loss_weights = torch.tensor(loss_factors, dtype=torch.float64, device=device)

    # Default checkpoints live next to this script, independent of the launch directory.
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pth")
    train_path = os.path.join(model_dir, f"{node_dim}_{num_layers}_{pair_dim}_train_{E_factor}E{F_factor}F{S_factor}S.pth")
    valid_path = os.path.join(model_dir, f"{node_dim}_{num_layers}_{pair_dim}_valid_{E_factor}E{F_factor}F{S_factor}S.pth")
    resume_path = os.path.join(model_dir, f"{node_dim}_{num_layers}_{pair_dim}_checkpoint_last.pt")
    if num_heads > 1:
        suffix = f"_H{num_heads}_R{readout_hidden}"
        train_path = train_path.replace(".pth", suffix + ".pth")
        valid_path = valid_path.replace(".pth", suffix + ".pth")
        resume_path = resume_path.replace(".pt", suffix + ".pt")
    log.info(f"train_path: {train_path}")
    log.info(f"valid_path: {valid_path}")
    max_clip = config_data["max_clip"]
    epoch_num = config_data["epoch_num"]
    # In DDP each rank executes len(train_loader) optimizer updates per epoch.
    # Scheduler stepping is also per-rank, so we must use the local loader length.
    # Using global train_count // batch_size would overestimate steps and stretch warmup.
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(epoch_num * steps_per_epoch, 1)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_lambda, warmup_steps, stable_steps, decay_steps = build_warmup_stable_decay_lambda(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_factor=warmup_factor,
        max_lr=max_lr,
        min_lr=min_lr,
        decay_phase_ratio=wsd_decay_ratio,
    )
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        for name, optimizer in optimizers.items()
    }


    log.info(f"max_clip: {max_clip}")
    log.info(f"epoch_num: {epoch_num}")
    log.info(f"steps_per_epoch: {steps_per_epoch}")
    log.info(f"total_steps: {total_steps}")
    log.info(f"warmup_steps: {warmup_steps}")
    log.info(f"stable_steps: {stable_steps}")
    log.info(f"decay_steps (WSD cosine): {decay_steps}")
    log.info(f"min_lr (WSD cosine floor): {min_lr}")

    best_train = float("inf")
    best_valid = float("inf")
    start_epoch = 0

    if resume:
        resume_path = config_data.get("resume_path") or resume_path
    log.info(f"resume: {resume}; finetune: {finetune}")
    log.info(f"save & resume_path: {resume_path}")

    if resume:
        log.info(f"resume checkpoint: {resume_path}")
        if not os.path.exists(resume_path):
            raise Exception("Checkpoint required for retrain does not exist.")

        start_epoch, best_train, best_valid = load_checkpoint(
            raw_model=raw_model,
            optimizers=optimizers,
            schedulers=schedulers,
            ema=ema,
            ckpt_path=resume_path,
            device=device,
            log=log,
            dataset_metadata=info["datasets"],
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
        )
        log.info(f"resumed from epoch {start_epoch}")
        dist.barrier(device_ids=[local_rank])

    if config_data.get("compile_training", False):
        from mace_soyo.utils.check_compile_grad import check_compile_grad
        check_compile_grad(efs_model)

    # Run the startup gradient check before DDP installs its backward hooks.
    ddp_model = DDP(
        efs_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for epoch in range(start_epoch, epoch_num):
        start_time = datetime.datetime.now()
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        train_metrics = EpochMetrics(device, info["head_names"])
        train_loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
        train_label_counts = torch.zeros(3, dtype=torch.float64, device=device)
        last_train_lr = lr_source.param_groups[0]["lr"]
        log.info("-" * 128)

        for batch_index, data in enumerate(train_loader):
            data = data.to(device)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            predictions = ddp_model(data)
            for name, value in zip(("energy", "force", "stress"), predictions):
                check_finite(name, value, batch_index)
            loss_sums, label_counts = efs_loss(predictions, data, criterion)
            # DDP averages gradients; normalize by global counts and compensate by world size.
            global_counts = label_counts.clone()
            dist.all_reduce(global_counts)
            loss_components = loss_sums * world_size / global_counts.clamp_min(1)
            loss = (loss_components * loss_weights).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_clip)
            last_train_lr = lr_source.param_groups[0]["lr"]
            for optimizer in optimizers.values():
                optimizer.step()
            for scheduler in schedulers.values():
                scheduler.step()
            ema.update()
            train_metrics.update(data, predictions)
            train_loss_sums += loss_sums.detach()
            train_label_counts += label_counts

            if batch_index % interval == 0:
                log.debug(
                    f"[GPU] {rank} [IDX]:{batch_index:0>4} [loss]:{loss.item():.5f} "
                    f"[energy]:{loss_components[0].item():.5f} "
                    f"[force]:{loss_components[1].item():.5f} "
                    f"[stress]:{loss_components[2].item():.6f} [lr]:{last_train_lr:.6f}"
                )

        train_metrics.reduce()
        train_report = train_metrics.compute()
        dist.all_reduce(train_loss_sums)
        dist.all_reduce(train_label_counts)
        train_score = weighted_loss_score(train_loss_sums, train_label_counts, loss_factors)
        is_best_train = train_score < best_train
        if is_best_train:
            best_train = train_score
            if rank == 0:
                with ema.average_parameters():
                    save_best_checkpoint(raw_model, train_path, model_kwargs, info["datasets"])
        log_epoch_metrics(
            log, train_report, phase="TRAIN", epoch=epoch, score=train_score,
            lr=last_train_lr, elapsed=datetime.datetime.now() - start_time, is_best=is_best_train,
        )
        dist.barrier(device_ids=[local_rank])

        if info["valid_count"] > 0:
            with ema.average_parameters():
                start_time = datetime.datetime.now()
                efs_model.eval()
                valid_metrics = EpochMetrics(device, info["head_names"])
                valid_loss_sums = torch.zeros(3, dtype=torch.float64, device=device)
                valid_label_counts = torch.zeros(3, dtype=torch.float64, device=device)
                # Unequal validation shards must not enter DDP forward or per-batch collectives.
                for batch_index, data in enumerate(valid_loader):
                    data = data.to(device)
                    predictions = efs_model(data)
                    for name, value in zip(("energy", "force", "stress"), predictions):
                        check_finite(name, value, batch_index)
                    loss_sums, label_counts = efs_loss(predictions, data, criterion)
                    valid_loss_sums += loss_sums.detach()
                    valid_label_counts += label_counts
                    valid_metrics.update(data, predictions)

                valid_metrics.reduce()
                valid_report = valid_metrics.compute()
                dist.all_reduce(valid_loss_sums)
                dist.all_reduce(valid_label_counts)
                valid_score = weighted_loss_score(valid_loss_sums, valid_label_counts, loss_factors)
                is_best_valid = valid_score < best_valid
                if is_best_valid:
                    best_valid = valid_score
                    if rank == 0:
                        save_best_checkpoint(raw_model, valid_path, model_kwargs, info["datasets"])
                log_epoch_metrics(
                    log, valid_report, phase="VALID", epoch=epoch, score=valid_score,
                    lr=last_train_lr, elapsed=datetime.datetime.now() - start_time, is_best=is_best_valid,
                )

        if rank == 0:
            save_checkpoint(
                epoch=epoch, raw_model=raw_model, optimizers=optimizers, schedulers=schedulers,
                ema=ema, best_train=best_train, best_valid=best_valid, save_path=resume_path,
                model_kwargs=model_kwargs, dataset_metadata=info["datasets"],
                steps_per_epoch=steps_per_epoch, batch_size=batch_size,
            )
        dist.barrier(device_ids=[local_rank])

    log.info(f"Training completed! Reached max epoch_num: {epoch_num}")
    log.info(f"Total training time: {datetime.datetime.now() - run_time}")
    dist.destroy_process_group()
    for handler in log.handlers:
        handler.close()


if __name__ == "__main__":
    main()
