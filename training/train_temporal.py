"""Train elapsed-time prefixes, optionally for a wall-clock budget, with recovery."""

import argparse
import contextlib
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import PROCESSED_DIR, resolve_path
from data_processing.temporal import PrefixReplayDataset
from models import BaekResNet3D
from training.train import choose_device, write_json


def metrics_from_predictions(targets, probabilities, losses):
    """Binary probability metrics; zero/one targets always refer to Terran wins."""
    targets = torch.as_tensor(targets, dtype=torch.long)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
    losses = torch.as_tensor(losses, dtype=torch.float64)
    predicted = (probabilities >= 0.5).long()
    return {"samples": len(targets), "loss": losses.mean().item(),
            "accuracy": (predicted == targets).double().mean().item(),
            "brier": ((probabilities - targets) ** 2).mean().item(),
            "confusion_matrix": torch.bincount(targets * 2 + predicted, minlength=4).reshape(2, 2).tolist()}


def run_prefix_epoch(model, loader, device, *, optimizer=None, scaler=None, amp=False, log_every=20):
    training = optimizer is not None
    model.train(training)
    targets_all, probabilities_all, losses_all, cutoffs_all = [], [], [], []
    steps = skipped = 0
    for index, batch in enumerate(loader):
        inputs = {name: tensor.to(device, non_blocking=True) for name, tensor in batch["inputs"].items()}
        targets = batch["targets"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs)
                losses = F.cross_entropy(logits, targets, reduction="none")
                loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                if scaler is not None and scaler.is_enabled():
                    previous_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    overflow = scaler.get_scale() < previous_scale
                    skipped += int(overflow)
                    steps += int(not overflow)
                else:
                    loss.backward()
                    optimizer.step()
                    steps += 1
        targets_all.extend(targets.detach().cpu().tolist())
        probabilities_all.extend(logits.detach().float().softmax(-1)[:, 1].cpu().tolist())
        losses_all.extend(losses.detach().float().cpu().tolist())
        cutoffs_all.extend(batch["cutoff_seconds"].tolist())
        if training and log_every and (index + 1) % log_every == 0:
            print(f"  batch {index + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
    if not targets_all:
        raise ValueError("No temporal examples processed")
    result = metrics_from_predictions(targets_all, probabilities_all, losses_all)
    result.update(optimizer_steps=steps, amp_skipped_steps=skipped)
    result["by_cutoff_seconds"] = {}
    for cutoff in sorted(set(cutoffs_all)):
        selected = [i for i, value in enumerate(cutoffs_all) if value == cutoff]
        result["by_cutoff_seconds"][f"{cutoff:g}"] = metrics_from_predictions(
            [targets_all[i] for i in selected], [probabilities_all[i] for i in selected],
            [losses_all[i] for i in selected])
    return result


@contextlib.contextmanager
def keep_system_awake(enabled):
    """Prevent automatic Windows sleep only while this process is training."""
    if enabled and os.name == "nt":
        import ctypes
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Unable to request temporary Windows sleep prevention")
        try:
            yield
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    else:
        yield


def save_temporal_checkpoint(path, model, optimizer, scaler, generator, *, epoch, config, history, best_brier):
    checkpoint = {
        "format_version": 1, "model_name": "BaekResNet3D", "model_config": model.config,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "run_config": config,
        "history": history, "best_brier": best_brier,
        "label_definition": "0 = Protoss wins, 1 = Terran wins",
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_loader": generator.get_state(),
    }
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=resolve_path, default=PROCESSED_DIR)
    parser.add_argument("--pov", choices=("both", "single", "fogless"), default="both")
    parser.add_argument("--output", type=resolve_path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-hours", type=float, help="Stop at an epoch boundary after this wall-clock budget")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--min-seconds", type=float, default=60)
    parser.add_argument("--step-seconds", type=float, default=60)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--history-seconds", type=float, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-samples", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=resolve_path, help="Resume an epoch checkpoint into a new output directory")
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--review-on-finish", action="store_true")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads, args.checkpoint_every) < 1 or args.frames < 9:
        parser.error("Positive epochs/batch-size/threads/checkpoint-every and frames>=9 required")
    if args.log_every < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid logging interval or learning rate")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("max-hours must be finite and positive")
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    amp = args.amp and device.type == "cuda"
    data_settings = dict(root_dir=args.root, pov=args.pov, frames=args.frames, min_seconds=args.min_seconds,
                         step_seconds=args.step_seconds, max_seconds=args.max_seconds,
                         history_seconds=args.history_seconds, seed=args.seed,
                         test_fraction=args.test_fraction, cache_samples=args.cache_samples)
    train_set = PrefixReplayDataset(split="train", **data_settings)
    val_set = PrefixReplayDataset(split="test", **data_settings)
    if not train_set.replay_ids.isdisjoint(val_set.replay_ids):
        raise ValueError("Match leakage between training and validation")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 1), pin_memory=device.type == "cuda")
    model = BaekResNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    history, best_brier, first_epoch = [], float("inf"), 1
    checkpoint = None
    previous_best = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = checkpoint["run_config"]
        for field in ("root", "pov", "frames", "min_seconds", "step_seconds", "max_seconds", "history_seconds",
                      "seed", "test_fraction", "batch_size", "learning_rate", "amp"):
            current = str(getattr(args, field)) if field == "root" else getattr(args, field)
            if current != previous[field]:
                raise ValueError(f"Resume requires the same {field} setting")
        if previous["train_match_ids"] != sorted(train_set.replay_ids) or previous["val_match_ids"] != sorted(val_set.replay_ids):
            raise ValueError("Resume data split differs from the checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_brier, first_epoch = checkpoint["best_brier"], checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["rng_cpu"])
        if device.type == "cuda" and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        generator.set_state(checkpoint["rng_loader"])
        if args.epochs < first_epoch:
            raise ValueError("epochs must exceed the checkpoint epoch")
        # best.pt can be newer than the periodic recovery checkpoint. Preserve
        # its selection score so a worse resumed epoch cannot overwrite it.
        previous_best = args.resume.parent / "best.pt"
        if not previous_best.is_file():
            raise ValueError("Resume requires best.pt beside the source checkpoint")
        candidate = torch.load(previous_best, map_location="cpu", weights_only=True)
        compatible_fields = ("root", "pov", "frames", "min_seconds", "step_seconds", "max_seconds", "history_seconds",
                             "seed", "test_fraction", "batch_size", "learning_rate", "amp", "train_match_ids", "val_match_ids")
        if any(candidate["run_config"][key] != previous[key] for key in compatible_fields):
            raise ValueError("best.pt and the recovery checkpoint must have compatible training settings and splits")
        best_brier = min(best_brier, candidate["best_brier"])
        del candidate
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    config.update(pid=os.getpid(), git_commit=revision, torch_version=str(torch.__version__),
                  resolved_device=str(device), amp_enabled=amp,
                  train_match_ids=sorted(train_set.replay_ids), val_match_ids=sorted(val_set.replay_ids),
                  train_examples=len(train_set), val_examples=len(val_set),
                  validation_role="model selection, not an independent final test",
                  observation_setting="separate fog-limited POVs" if args.pov == "both" else args.pov,
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "config.json", config)
    # Keep the previously selected best checkpoint available when resuming last.pt.
    if args.resume:
        import shutil
        shutil.copy2(previous_best, args.output / "best.pt")
    start = time.monotonic()
    deadline = start + args.max_hours * 3600 if args.max_hours else float("inf")
    status = {"state": "running", "pid": os.getpid(), "epoch": first_epoch - 1,
              "started_at_utc": config["started_at_utc"], "max_hours": args.max_hours}
    write_json(args.output / "status.json", status)
    print(f"Device={device}, AMP={amp}; train={len(train_set)} prefixes/{len(train_set.replay_ids)} matches; "
          f"validation={len(val_set)} prefixes/{len(val_set.replay_ids)} matches", flush=True)
    print(f"Output={args.output}; budget={args.max_hours} hours; PID={os.getpid()}", flush=True)
    reason = "epoch_limit"
    epoch = first_epoch - 1
    try:
        with keep_system_awake(args.keep_awake):
            for epoch in range(first_epoch, args.epochs + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                epoch_start = time.monotonic()
                print(f"Epoch {epoch}/{args.epochs}", flush=True)
                train_metrics = run_prefix_epoch(model, train_loader, device, optimizer=optimizer,
                                                  scaler=scaler, amp=amp, log_every=args.log_every)
                val_metrics = run_prefix_epoch(model, val_loader, device, amp=amp, log_every=0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                          "seconds": time.monotonic() - epoch_start, "elapsed_seconds": time.monotonic() - start,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                          "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
                history.append(record)
                improved = val_metrics["brier"] < best_brier
                best_brier = min(best_brier, val_metrics["brier"])
                write_json(args.output / "history.json", history)
                save_args = dict(epoch=epoch, config=config, history=history, best_brier=best_brier)
                if improved:
                    save_temporal_checkpoint(args.output / "best.pt", model, optimizer, scaler, generator, **save_args)
                stop_requested = (args.output / "STOP").exists()
                budget_done = time.monotonic() >= deadline
                if epoch % args.checkpoint_every == 0 or epoch == first_epoch or epoch == args.epochs or stop_requested or budget_done:
                    save_temporal_checkpoint(args.output / "last.pt", model, optimizer, scaler, generator, **save_args)
                status.update(epoch=epoch, elapsed_seconds=time.monotonic() - start, best_brier=best_brier,
                              latest_validation=val_metrics, last_update_utc=datetime.now(timezone.utc).isoformat())
                write_json(args.output / "status.json", status)
                print(f"  train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f}; "
                      f"val acc={val_metrics['accuracy']:.3f} Brier={val_metrics['brier']:.4f}; "
                      f"{record['seconds']:.1f}s; AMP skipped steps={train_metrics['amp_skipped_steps']}", flush=True)
                if stop_requested or budget_done:
                    reason = "stop_file" if stop_requested else "time_budget"
                    break
            status.update(state="stopped" if reason == "stop_file" else "completed", reason=reason,
                          elapsed_seconds=time.monotonic() - start)
            if args.review_on_finish:
                status["review_state"] = "running"
                write_json(args.output / "status.json", status)
                from analysis.review import build_review
                build_review(args.output / "best.pt", args.output / "review", device=str(device))
                status["review_state"] = "completed"
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        status["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "status.json", status)
    print(f"Finished: {reason}; checkpoints and metrics in {args.output}", flush=True)
    return args.output


if __name__ == "__main__":
    main()
