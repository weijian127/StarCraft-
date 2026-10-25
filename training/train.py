"""Train the Baek model from schema-v2 arrays: python -m training.train."""

import argparse
import json
import math
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import PROCESSED_DIR, resolve_path
from data_processing.dataset import POV_MODES, SC2ReplayDataset
from models import BaekResNet3D


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install with uv sync --extra cuda or use --device cpu")
    return torch.device(requested)


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, amp=False, max_batches=None):
    """Run training or evaluation; weight loss by examples, including short batches."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = total = batches = 0
    confusion = torch.zeros(2, 2, dtype=torch.long)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = {name: value.to(device, non_blocking=True) for name, value in batch["inputs"].items()}
        targets = batch["targets"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; try --no-amp and inspect the data")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        predictions = logits.detach().argmax(dim=1)
        size = targets.numel()
        total_loss += loss.detach().item() * size
        correct += (predictions == targets).sum().item()
        total += size
        batches += 1
        confusion += torch.bincount((targets * 2 + predictions).cpu(), minlength=4).reshape(2, 2)
        if training:
            print(f"  batch {batches}/{min(len(loader), max_batches or len(loader))}: loss={loss.item():.4f}", flush=True)
    if not total:
        raise ValueError("No batches were processed")
    return {"loss": total_loss / total, "accuracy": correct / total,
            "samples": total, "batches": batches, "confusion_matrix": confusion.tolist()}


def evaluate(model, loader, device, *, seed, amp=False, max_batches=None):
    """Use identical validation frames each epoch without changing training RNG."""
    # Worker RNG seeds come from this generator; zero-worker collation uses CPU RNG.
    loader.generator.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return run_epoch(model, loader, device, amp=amp, max_batches=max_batches)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, model, optimizer, scaler, *, epoch, metrics, run_config):
    temp = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1, "model_name": "BaekResNet3D",
        "model_config": model.config, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch, "metrics": metrics, "run_config": run_config,
        "label_definition": "0 = Protoss wins, 1 = Terran wins",
    }, temp)
    temp.replace(path)


def load_model(checkpoint_path, device="cpu"):
    """Load this trainer's checkpoint safely for inference (no training restart)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "BaekResNet3D":
        raise ValueError("Not a supported BaekResNet3D checkpoint")
    model = BaekResNet3D(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=resolve_path, default=PROCESSED_DIR)
    parser.add_argument("--pov", choices=POV_MODES, default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-timesteps", type=int, default=50)
    parser.add_argument("--start-prob", type=float, default=0.1)
    parser.add_argument("--length-prob", type=float, default=0.8)
    parser.add_argument("--weighting", choices=("log", "exp", "uniform"), default="log")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Held-out validation fraction of whole matches")
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="0 avoids duplicating large arrays on Windows")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision on CUDA only")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--max-train-batches", type=int, help="Limit for smoke tests; default uses every sample")
    parser.add_argument("--max-val-batches", type=int, help="Limit for smoke tests; default uses every held-out sample")
    parser.add_argument("--output", type=resolve_path, help="New run directory; defaults to runs/<timestamp>")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, threads must be positive; num-workers must be nonnegative")
    if args.max_timesteps < 9 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("max-timesteps must be >=9 and learning-rate must be finite and positive")
    if any(value is not None and value < 1 for value in (args.max_train_batches, args.max_val_batches)):
        parser.error("Batch limits must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    amp = args.amp and device.type == "cuda"
    dataset_args = dict(root_dir=args.root, pov=args.pov, max_timesteps=args.max_timesteps,
                        weighting=args.weighting, start_prob=args.start_prob, length_prob=args.length_prob,
                        test_fraction=args.test_fraction, seed=args.seed, stratify=args.stratify)
    train_set = SC2ReplayDataset(split="train", **dataset_args)
    val_set = SC2ReplayDataset(split="test", **dataset_args)
    if not train_set.replay_ids.isdisjoint(val_set.replay_ids):
        raise ValueError("Training and validation contain overlapping matches")
    loaders = [DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + offset),
    ) for dataset, shuffle, offset in ((train_set, True, 0), (val_set, False, 1))]
    model = BaekResNet3D(freeze_embeddings=args.freeze_embeddings).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = args.output or resolve_path("runs/" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    run_config = {key: str(value) if hasattr(value, "__fspath__") else value for key, value in vars(args).items()}
    run_config.update({
        "output": str(output), "resolved_device": str(device), "amp_enabled": amp,
        "torch_version": str(torch.__version__), "model_config": model.config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "model_state_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
        "train_match_ids": sorted(train_set.replay_ids), "val_match_ids": sorted(val_set.replay_ids),
        "train_sample_ids": [s["sample_id"] for s in train_set.samples],
        "val_sample_ids": [s["sample_id"] for s in val_set.samples],
        "train_labels": dict(train_set.counts), "val_labels": dict(val_set.counts),
    })
    write_json(output / "config.json", run_config)
    print(f"Device: {device}; AMP: {amp}; parameters: {run_config['parameters']:,}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Train: {len(train_set)} POVs / {len(train_set.replay_ids)} matches {dict(train_set.counts)}; "
          f"validation: {len(val_set)} POVs / {len(val_set.replay_ids)} matches {dict(val_set.counts)}", flush=True)
    print(f"Output: {output}", flush=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(model, loaders[0], device, optimizer=optimizer, scaler=scaler,
                                  amp=amp, max_batches=args.max_train_batches)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_end = time.perf_counter()
        val_metrics = evaluate(model, loaders[1], device, seed=args.seed + 1,
                               amp=amp, max_batches=args.max_val_batches)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        metrics = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                   "seconds": end - start, "train_seconds": train_end - start,
                   "validation_seconds": end - train_end,
                   # PyTorch allocator peaks include model, optimizer, activations
                   # and workspaces; they exclude driver/context and other apps.
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                   if device.type == "cuda" else None,
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                   if device.type == "cuda" else None}
        history.append(metrics)
        write_json(output / "history.json", history)
        save_checkpoint(output / "last.pt", model, optimizer, scaler, epoch=epoch,
                        metrics=metrics, run_config=run_config)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(output / "best.pt", model, optimizer, scaler, epoch=epoch,
                            metrics=metrics, run_config=run_config)
        print(f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f}; "
              f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f}; "
              f"{metrics['seconds']:.1f}s", flush=True)
        if device.type == "cuda":
            print(f"  peak CUDA allocated={metrics['cuda_peak_allocated_bytes'] / 2**30:.3f} GiB; "
                  f"reserved={metrics['cuda_peak_reserved_bytes'] / 2**30:.3f} GiB", flush=True)
    print(f"Saved best.pt, last.pt, history.json and config.json in {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
