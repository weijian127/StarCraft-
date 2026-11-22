"""Export an offline probability timeline, pivotal intervals and Grad-CAM evidence."""

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from analysis.attribution import grad_cam, pivotal_intervals
from config import resolve_path
from data_processing.temporal import PrefixReplayDataset
from training.train import choose_device, load_model, write_json


def masking_check(model, inputs, heatmap, *, target_class=1, fraction=0.1, seed=42):
    """Compare zero masking of top-attribution versus equal-sized random regions.

    The same spatiotemporal cells are masked across all three categorical maps.
    This is a sensitivity diagnostic; masking can create out-of-distribution input.
    """
    if not 0 < fraction < 1 or target_class not in (0, 1):
        raise ValueError("Invalid masking fraction or class")
    if heatmap.shape[0] != 1:
        raise ValueError("Masking check expects one replay clip")
    if heatmap.max().item() <= 0:
        return {"available": False, "reason": "No positive attribution to mask"}
    count = max(1, round(heatmap.numel() * fraction))
    top = heatmap.flatten().topk(count).indices.cpu()
    random = torch.randperm(heatmap.numel(), generator=torch.Generator().manual_seed(seed))[:count]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = model(inputs).softmax(-1)[0, target_class].item()
            masked_probabilities = []
            for selected in (top, random):
                masked = {key: tensor.contiguous().clone() for key, tensor in inputs.items()}
                for tensor in masked.values():
                    tensor.reshape(-1)[selected.to(tensor.device)] = 0
                masked_probabilities.append(model(masked).softmax(-1)[0, target_class].item())
    finally:
        model.train(was_training)
    return {"available": True, "masked_cells": count, "fraction": fraction,
            "target_class": target_class, "original_probability": before,
            "highlighted_probability_drop": before - masked_probabilities[0],
            "random_probability_drop": before - masked_probabilities[1], "random_seed": seed}


def time_grid_settings(config, *, min_seconds=None, step_seconds=None, max_seconds=None):
    """Return the training grid, the inference grid and whether they differ.

    An override changes only which cutoffs the exporter samples. The checkpoint and
    its recorded training settings are read, never rewritten.
    """
    training = {"min_seconds": float(config["min_seconds"]),
                "step_seconds": float(config["step_seconds"]),
                "max_seconds": float(config["max_seconds"])}
    inference = dict(training)
    for name, value in (("min_seconds", min_seconds), ("step_seconds", step_seconds),
                        ("max_seconds", max_seconds)):
        if value is not None:
            inference[name] = float(value)
    if (not all(math.isfinite(value) for value in inference.values())
            or inference["min_seconds"] < 0 or inference["step_seconds"] <= 0
            or inference["max_seconds"] < inference["min_seconds"]):
        raise ValueError("Invalid inference time grid")
    return training, inference, inference != training


def build_review(checkpoint_path, output, *, device="auto", sample_id=None, threshold=0.1,
                 merge_seconds=60, min_seconds=None, step_seconds=None, max_seconds=None):
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["run_config"]
    if "min_seconds" not in config:
        raise ValueError("A cutoff-trained temporal checkpoint is required")
    model = load_model(checkpoint_path, device)
    training_grid, inference_grid, grid_overridden = time_grid_settings(
        config, min_seconds=min_seconds, step_seconds=step_seconds, max_seconds=max_seconds)
    dataset_kwargs = dict(pov=config["pov"], split="test", frames=config["frames"],
                          history_seconds=config["history_seconds"], seed=config["seed"],
                          test_fraction=config["test_fraction"], cache_samples=1)
    # Rebuild the checkpoint's own grid first, so the validation-match check keeps
    # exactly its original meaning no matter what export grid is requested.
    dataset = PrefixReplayDataset(
        config["root"], min_seconds=training_grid["min_seconds"],
        step_seconds=training_grid["step_seconds"], max_seconds=training_grid["max_seconds"],
        **dataset_kwargs,
    )
    if sorted(dataset.replay_ids) != config["val_match_ids"]:
        raise ValueError("Review dataset does not match the checkpoint's validation matches")
    verification = dataset
    # Rebuild with the export grid when one was requested. Match grouping, the
    # held-out split and the terminal-observation exclusion are unchanged.
    if grid_overridden:
        dataset = PrefixReplayDataset(
            config["root"], min_seconds=inference_grid["min_seconds"],
            step_seconds=inference_grid["step_seconds"], max_seconds=inference_grid["max_seconds"],
            **dataset_kwargs,
        )
        if not set(dataset.replay_ids) <= set(config["val_match_ids"]):
            raise ValueError("Inference time grid changed the held-out match set")
    if sample_id is None:
        default_sample = verification.samples[0]["sample_id"]
        if default_sample in {row["sample_id"] for row in dataset.samples}:
            sample_id = default_sample
        else:
            sample_id = dataset.samples[0]["sample_id"]
            print(f"Default sample {default_sample} has no eligible cutoff in the inference grid; "
                  f"using {sample_id} instead.", flush=True)
    selected = [index for index, (base_index, _) in enumerate(dataset.examples)
                if dataset.base.samples[base_index]["sample_id"] == sample_id]
    if not selected:
        raise ValueError(
            "sample_id must identify an eligible held-out POV for the inference grid "
            f"{inference_grid['min_seconds']:g}-{inference_grid['max_seconds']:g}s "
            f"(step {inference_grid['step_seconds']:g}s)")
    output.mkdir(parents=True, exist_ok=False)
    points = []
    with torch.no_grad():
        for index in selected:
            item = dataset[index]
            inputs = {key: tensor.unsqueeze(0).to(device) for key, tensor in item["inputs"].items()}
            probability = model(inputs).softmax(-1)[0, 1].item()
            points.append({"cutoff_seconds": item["cutoff_seconds"], "terran_win_probability": probability,
                           "frame_indices": item["frame_indices"].tolist(), "frame_seconds": item["frame_seconds"].tolist()})
    intervals = pivotal_intervals([p["cutoff_seconds"] for p in points],
                                  [p["terran_win_probability"] for p in points],
                                  threshold=threshold, merge_seconds=merge_seconds)
    # Explain the two largest detected changes; always provide at least the last clip.
    cutoffs = [event["peak_cutoff_seconds"] for event in sorted(intervals, key=lambda e: abs(e["peak_change"]), reverse=True)[:2]]
    if not cutoffs:
        cutoffs = [points[-1]["cutoff_seconds"]]
    explanations = []
    palette = ListedColormap(["#172033", "#55b9a5", "#6495ed", "#c9cdd2", "#ef8c75"])
    for number, cutoff in enumerate(cutoffs, 1):
        position = next(i for i, point in enumerate(points) if point["cutoff_seconds"] == cutoff)
        item = dataset[selected[position]]
        inputs = {key: tensor.unsqueeze(0).to(device) for key, tensor in item["inputs"].items()}
        target = int(points[position]["terran_win_probability"] >= 0.5)
        attribution = grad_cam(model, inputs, target_class=target)
        heatmap = attribution["heatmap"][0].cpu().numpy()
        # Only selected observed frames are named; interpolation does not create new timestamps.
        frame = int(heatmap.mean(axis=(1, 2)).argmax())
        seconds = float(item["frame_seconds"][frame])
        spatial = item["inputs"]["player_relative"][frame].numpy()
        np.savez_compressed(output / f"gradcam_{number}.npz", heatmap=heatmap,
                            frame_seconds=item["frame_seconds"].numpy(), frame_indices=item["frame_indices"].numpy(),
                            target_class=target, cutoff_seconds=cutoff)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis in axes:
            axis.imshow(spatial, cmap=palette, vmin=0, vmax=4, interpolation="nearest")
            axis.set_xlabel("Minimap x (pixels)")
            axis.set_ylabel("Minimap y (pixels)")
        axes[0].set_title(f"Observed ownership at {seconds:.1f}s")
        overlay = axes[1].imshow(heatmap[frame], cmap="magma", vmin=0, vmax=1, alpha=0.65)
        axes[1].set_title(f"Evidence for {'Terran' if target else 'Protoss'} at cutoff {cutoff:g}s")
        fig.colorbar(overlay, ax=axes[1], label="Normalized Grad-CAM")
        filename = f"gradcam_{number}.png"
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        explanations.append({"cutoff_seconds": cutoff, "displayed_frame_seconds": seconds,
                             "target_class": target, "activation_shape": attribution["activation_shape"],
                             "image": filename, "arrays": f"gradcam_{number}.npz",
                             "masking_check": masking_check(model, inputs, attribution["heatmap"], target_class=target)})
    first = dataset[selected[0]]
    report = {"sample_id": sample_id, "replay_id": first["replay_id"], "focus_player_id": first["focus_player_id"],
              "label": int(first["targets"]), "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
              "observation_setting": config["observation_setting"], "threshold": threshold, "merge_seconds": merge_seconds,
              "time_grid": {"training": training_grid, "inference": inference_grid,
                            "override_applied": bool(grid_overridden), "frames": int(config["frames"])},
              "points": points, "pivotal_intervals": intervals, "explanations": explanations,
              "limitations": (
                  "Validation match used for model selection. Eleven-match development corpus; probabilities are uncalibrated. "
                  "Heatmaps show model evidence, not causal effects. Temporal/spatial heatmaps are interpolated from coarse encoder features. "
                  "Recorded actions are not present in these arrays. "
                  f"This checkpoint was trained for {checkpoint['epoch']} epoch(s) and cutoffs "
                  f"{training_grid['min_seconds']:g}-{training_grid['max_seconds']:g}s "
                  f"(step {training_grid['step_seconds']:g}s). "
                  + ("Other cutoffs are exported for pipeline testing only; predictive ability at those "
                     "times is not validated." if grid_overridden else "")
              )}
    write_json(output / "review.json", report)
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.plot([p["cutoff_seconds"] / 60 for p in points], [p["terran_win_probability"] for p in points], "o-", color="#287c8e")
    for interval in intervals:
        axis.axvspan(interval["start_seconds"] / 60, interval["end_seconds"] / 60, alpha=0.15, color="#df9244")
    axis.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    axis.set(xlabel="Elapsed game time (minutes)", ylabel="P(Terran wins)", ylim=(0, 1),
             title=f"Held-out replay · player {first['focus_player_id']} POV · epoch {checkpoint['epoch']}")
    fig.savefig(output / "timeline.png", dpi=140)
    plt.close(fig)
    encoded = json.dumps(points).replace("<", "\\u003c")
    cards = []
    for explanation in explanations:
        check = explanation["masking_check"]
        diagnostic = (f'Top-region probability drop: {check["highlighted_probability_drop"]:.3f}; '
                      f'random-region drop: {check["random_probability_drop"]:.3f}.' if check["available"]
                      else f'Masking diagnostic unavailable: {html.escape(check["reason"])}.')
        cards.append(f'<figure><img src="{explanation["image"]}" alt="Grad-CAM at {explanation["cutoff_seconds"]:g} seconds">'
                     f'<figcaption>Prediction cutoff {explanation["cutoff_seconds"]:g}s; '
                     f'shown observation {explanation["displayed_frame_seconds"]:.1f}s. {diagnostic}</figcaption></figure>')
    cards = "".join(cards)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC2 replay review</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#223047;background:#f5f7fa}}
img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption,.note{{color:#526070;font-size:14px;line-height:1.6}}input{{width:100%}}code{{overflow-wrap:anywhere}}
.readout{{font-size:22px;padding:12px 0}}a{{color:#175970}}</style>
<h1>SC2 replay review</h1><p>Player {first['focus_player_id']} POV · {html.escape(config['observation_setting'])}</p>
<p class="note">{html.escape(report['limitations'])}</p><img src="timeline.png" alt="Elapsed-time Terran win probability with pivotal intervals">
<label for="time">Inspect a prediction cutoff</label><input id="time" type="range" min="0" max="{len(points)-1}" step="1" value="0"><div class="readout" id="readout"></div>
<p>{len(intervals)} pivotal interval(s), using a {threshold:.0%} adjacent probability-change threshold.</p>
{cards}<p><a href="review.json">Full timeline, intervals and frame provenance</a></p>
<script>const points={encoded};const slider=document.getElementById('time');function update(){{const p=points[Number(slider.value)];document.getElementById('readout').textContent=`${{p.cutoff_seconds}}s · Terran ${{(100*p.terran_win_probability).toFixed(1)}}% · Protoss ${{(100*(1-p.terran_win_probability)).toFixed(1)}}%`;}}slider.addEventListener('input',update);update();</script></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    print(f"Review exported: {output / 'index.html'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=resolve_path, required=True)
    parser.add_argument("--output", type=resolve_path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--merge-seconds", type=float, default=60)
    parser.add_argument("--min-seconds", type=float,
                        help="Export-only cutoff grid: first cutoff (default: checkpoint value)")
    parser.add_argument("--step-seconds", type=float,
                        help="Export-only cutoff grid: spacing (default: checkpoint value)")
    parser.add_argument("--max-seconds", type=float,
                        help="Export-only cutoff grid: last cutoff (default: checkpoint value)")
    args = parser.parse_args()
    build_review(args.checkpoint, args.output, device=args.device, sample_id=args.sample_id,
                 threshold=args.threshold, merge_seconds=args.merge_seconds,
                 min_seconds=args.min_seconds, step_seconds=args.step_seconds,
                 max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
