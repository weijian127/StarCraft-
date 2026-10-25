"""Gradient-based attribution of the Terran/Protoss log-odds score."""

import torch
from torch.nn import functional as F


def grad_cam(model, inputs, *, target_class=1):
    """Return [B,T,H,W] positive evidence for one class, normalized per clip.

    Differentiate the raw logit margin through forward_features, so this also
    works with a future nonlinear prediction head. No gradient on integer IDs is
    needed. The model is evaluated with autograd enabled, in FP32. No parameter
    gradients or BN updates are accumulated. Zero evidence returns an all-zero map.
    """
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 (Protoss) or 1 (Terran)")
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad(), torch.autocast(device_type=next(model.parameters()).device.type, enabled=False):
            activations = model.forward_features(inputs)
            # Also permits attribution with all model parameters frozen.
            if not activations.requires_grad:
                activations.requires_grad_(True)
            pooled = model.drop_out(model.pool3(activations)).flatten(1)
            logits = model.linear(pooled)
            score = (logits[:, target_class] - logits[:, 1 - target_class]).sum()
            gradients, = torch.autograd.grad(score, activations)
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
            low_resolution = (weights * activations).sum(dim=1, keepdim=True).relu()
            shape = next(iter(inputs.values())).shape[1:]
            heatmap = F.interpolate(low_resolution, size=shape, mode="trilinear", align_corners=False)[:, 0]
            scale = heatmap.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
            heatmap = heatmap / scale.clamp_min(1e-12)
        return {"heatmap": heatmap.detach(), "logits": logits.detach(),
                "target_class": target_class, "activation_shape": tuple(activations.shape)}
    finally:
        model.train(was_training)


def pivotal_intervals(times, probabilities, *, threshold=0.1, merge_seconds=60):
    """Find adjacent changes >= threshold and merge events separated by a short gap.

    Inputs must be deterministic predictions from increasing elapsed cutoffs.
    Threshold is an absolute probability change (0.1 means ten percentage points).
    Reports model evidence, not a causal effect or an action recommendation.
    """
    import numpy as np
    times = np.asarray(times, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (times.ndim != 1 or times.shape != probabilities.shape or not np.isfinite(times).all()
            or not np.isfinite(probabilities).all() or not np.all(np.diff(times) > 0)
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.isfinite(threshold) or not 0 < threshold <= 1
            or not np.isfinite(merge_seconds) or merge_seconds < 0):
        raise ValueError("Invalid timeline or pivotal-interval settings")
    events = []
    for index in range(1, len(times)):
        delta = float(probabilities[index] - probabilities[index - 1])
        if abs(delta) < threshold:
            continue
        event = {"start_seconds": float(times[index - 1]), "end_seconds": float(times[index]),
                 "start_probability": float(probabilities[index - 1]),
                 "end_probability": float(probabilities[index]), "net_change": delta,
                 "peak_change": delta, "peak_cutoff_seconds": float(times[index]), "changes": 1}
        if events and event["start_seconds"] - events[-1]["end_seconds"] <= merge_seconds:
            previous = events[-1]
            previous.update(end_seconds=event["end_seconds"], end_probability=event["end_probability"],
                            net_change=event["end_probability"] - previous["start_probability"],
                            changes=previous["changes"] + 1)
            if abs(delta) > abs(previous["peak_change"]):
                previous.update(peak_change=delta, peak_cutoff_seconds=event["peak_cutoff_seconds"])
        else:
            events.append(event)
    return events
