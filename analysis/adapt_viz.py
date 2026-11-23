"""Convert viz.timeline / viz.gradcam outputs into the review.json the UI reads.

The team exports use focus-player labels: class 1 means the observed player wins,
not Terran. This adapter keeps the team files untouched and records the label
meaning explicitly so the frontend never has to guess.

Fields the team export does not provide (per-cutoff frame timestamps, masking
diagnostics, an inference time grid) are written as null / unavailable rather
than filled with zeros or invented values.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from analysis.attribution import pivotal_intervals
from analysis.gradcam_context import enrich_explanation

FOCUS_LABELS = "0 = focus player loses, 1 = focus player wins"
TERRAN_LABELS = "0 = Protoss wins, 1 = Terran wins"
INTERVAL_VISIBLE_DEFAULT = 3

ADAPTER_NOTE = (
    "Converted from viz.timeline/viz.gradcam output by analysis.adapt_viz. "
    "The model was trained on whole replays, so prefix predictions are illustrative and "
    "uncalibrated. Per-cutoff observation timestamps and the masking diagnostic are not "
    "part of the viz exports and are reported as unavailable. Membership of this replay "
    "in the checkpoint's training set is unknown; this report does not claim membership "
    "or independent-test status."
)


def subject_label(focus_target, focus_player_id, race):
    """Plain-language owner of the positive class."""
    return f"Player {focus_player_id} ({race})" if focus_target else "Terran"


def probability_statement(focus_target, focus_player_id, race):
    """One sentence that matches the actual label definition for this POV."""
    if focus_target:
        return (
            f"Probabilities are P(Player {focus_player_id} ({race}) wins), estimated from "
            "this POV's observations. They are not a match-level P(Terran wins) estimate; "
            "the two POVs use different observations and need not sum to 100%."
        )
    return "Probabilities are P(Terran wins), estimated from this POV's observations."


def interval_rule(threshold, merge_seconds):
    """Metadata for the candidate-interval display rule."""
    return {
        "algorithm": "candidate_adjacent_changes_consecutive_same_direction",
        "threshold": float(threshold),
        "max_merged_seconds": float(merge_seconds),
        "default_visible": INTERVAL_VISIBLE_DEFAULT,
        "note": (
            "A candidate starts at every adjacent probability change with "
            f"abs(delta) >= {float(threshold):g}. Consecutive same-direction changes may "
            f"merge only while their total span is at most {float(merge_seconds):g} s. "
            "This window is a local display rule, not a paper result and not a validated "
            "match-deciding event."
        ),
    }


def limitation_parts(focus_target, focus_player_id, race, threshold, merge_seconds,
                     has_explanations):
    """Keep the report caveats consistent with this POV's labels."""
    parts = [
        ADAPTER_NOTE,
        probability_statement(focus_target, focus_player_id, race),
        (
            "Candidate pivotal intervals use adjacent probability changes of at least "
            f"{float(threshold):g} (10.0 pp for 0.1). Each qualifying change is a candidate; "
            "consecutive same-direction changes may merge only while their total span is at "
            f"most {float(merge_seconds):g} s. The window is a local display rule, not a "
            "paper result and not a validated match-deciding event."
        ),
    ]
    if has_explanations:
        parts.append(
            "The Grad-CAM explains one pre-softmax class logit at block3; temporal bins "
            "overlap, so the map is evidence about sampled history, not an exact frame or "
            "a causal action.")
    return parts


def load_sample_metadata(sample_dir):
    """Read the POV row that owns this sample directory."""
    sample_dir = Path(sample_dir).expanduser().resolve()
    manifest_path = sample_dir.parent / "manifest.json"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next((s for s in manifest["samples"] if s["sample_id"] == sample_dir.name), None)
    if row is None:
        raise ValueError(f"{sample_dir.name} is not listed in {manifest_path}")
    return row, manifest_path


def checkpoint_epoch(checkpoint_path):
    """Best-effort epoch lookup; returns None when it cannot be read."""
    try:
        import torch
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        return None
    value = payload.get("epoch")
    return int(value) if isinstance(value, int) else None


def build(timeline_path, sample_dir, output, *, gradcam_dir=None, threshold=0.1,
          merge_seconds=120.0):
    timeline_path = Path(timeline_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {output}")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    row, manifest_path = load_sample_metadata(sample_dir)

    label_definition = timeline.get("label_definition")
    if label_definition == FOCUS_LABELS:
        focus_target = True
    elif label_definition == TERRAN_LABELS:
        focus_target = False
    else:
        raise ValueError(f"Unsupported label definition: {label_definition!r}")

    race = row["focus_player_race"]
    terran_win = int(row["label"])
    focus_outcome = terran_win if race == "Terran" else 1 - terran_win
    if focus_target:
        subject, probability_key = race, "focus_player_win_probability"
        label_value, result_text = focus_outcome, (
            f"Focus player ({race}, player {row['focus_player_id']}) "
            + ("won" if focus_outcome else "lost"))
        outcome_note = "1 = focus player wins, 0 = focus player loses"
    else:
        subject, probability_key = "Terran", "terran_win_probability"
        label_value, result_text = terran_win, ("Terran won" if terran_win else "Protoss won")
        outcome_note = "1 = Terran wins, 0 = Protoss wins"
    opponent = "Protoss" if subject == "Terran" else "Terran"

    points = []
    for point in timeline["points"]:
        points.append({
            "cutoff_seconds": float(point["seconds"]),
            probability_key: float(point["probability"]),
            "probability_draws": [float(value) for value in point.get("draws", [])],
        })
    if not points:
        raise ValueError("The timeline contains no prediction points")
    intervals = pivotal_intervals(
        [point["cutoff_seconds"] for point in points],
        [point[probability_key] for point in points],
        threshold=threshold, merge_seconds=merge_seconds)
    explanations = []
    gradcam_source = None
    if gradcam_dir is not None:
        gradcam_dir = Path(gradcam_dir).expanduser().resolve()
        gradcam_source = str(gradcam_dir)
        method = json.loads((gradcam_dir / "method.json").read_text(encoding="utf-8"))
        with np.load(gradcam_dir / "data.npz", allow_pickle=False) as archive:
            cutoffs = archive["cutoff_seconds"].copy()
            probabilities = archive["probabilities"].copy()
            native_bins = (int(archive["raw_heatmap"].shape[1])
                           if "raw_heatmap" in archive.files else None)
        if len(cutoffs) != 1:
            raise ValueError("Expected a static Grad-CAM export with exactly one cutoff")
        target = int(method["target_class"])
        if focus_target:
            target_meaning = "focus player wins" if target == 1 else "focus player loses"
        else:
            target_meaning = "Terran wins" if target == 1 else "Protoss wins"
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(gradcam_dir / "snapshot.png", output / "gradcam_1.png")
        explanation = {
            "cutoff_seconds": float(cutoffs[0]),
            # viz.gradcam stores only the cutoff and the native temporal bins; it does not
            # name the single frame that was displayed, so this stays null on purpose.
            "displayed_frame_seconds": None,
            "target_class": target,
            "target_meaning": target_meaning,
            "layer": method.get("layer"),
            "score": method.get("score"),
            "image": "gradcam_1.png",
            "arrays": None,
            "class_probabilities": [float(value) for value in probabilities[0]],
            "native_temporal_bins": native_bins,
            "masking_check": {"available": False,
                              "reason": "viz.gradcam does not export a masking diagnostic"},
        }
        try:
            explanation.update(enrich_explanation(
                sample_dir, output, float(cutoffs[0]),
                int(timeline.get("frames")), int(timeline.get("seed")),
                gradcam_dir / "data.npz", f"player{row['focus_player_id']}",
                "recorded_at_export"))
        except Exception as error:  # noqa: BLE001 - record unavailable context without faking it
            explanation["sampling"] = None
            explanation["context_error"] = f"{type(error).__name__}: {error}"
        explanations.append(explanation)
    else:
        output.mkdir(parents=True, exist_ok=False)

    limitations = limitation_parts(
        focus_target, row["focus_player_id"], race, threshold, merge_seconds,
        bool(explanations))

    report = {
        "sample_id": row["sample_id"],
        "replay_id": row["replay_id"],
        "focus_player_id": row["focus_player_id"],
        "focus_player_race": race,
        "label": label_value,
        "label_meaning": outcome_note,
        "result_text": result_text,
        "label_definition": label_definition,
        "probability_key": probability_key,
        "probability_subject": subject,
        "probability_opponent": opponent,
        "probability_subject_detail": (f"focus player {row['focus_player_id']} ({race})"
                                      if focus_target else "Terran"),
        "probability_statement": probability_statement(
            focus_target, row["focus_player_id"], race),
        "checkpoint": timeline.get("checkpoint"),
        "checkpoint_epoch": (checkpoint_epoch(timeline["checkpoint"])
                             if timeline.get("checkpoint") else None),
        "observation_setting": None,
        "threshold": threshold,
        "merge_seconds": merge_seconds,
        "interval_rule": interval_rule(threshold, merge_seconds),
        "intervals_source": "computed_by_adapter",
        "frames": timeline.get("frames"),
        "seed": timeline.get("seed"),
        "step_seconds": timeline.get("step"),
        "draws_per_point": timeline.get("draws"),
        "evaluation_status": "unknown_split_membership",
        "points": points,
        "pivotal_intervals": intervals,
        "explanations": explanations,
        "sources": {
            "timeline": str(timeline_path),
            "gradcam": gradcam_source,
            "sample": str(Path(sample_dir).expanduser().resolve()),
            "manifest": str(manifest_path),
        },
        "limitations": " ".join(limitations),
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def refresh_intervals(review_path, timeline_path=None, *, threshold=None, merge_seconds=120.0,
                      backup_suffix=".before-candidate-intervals.json"):
    """Recompute only candidate intervals from a saved timeline.

    The existing review gets a backup before the first rewrite. Probabilities and
    points are compared with the saved timeline and are never modified.
    """
    review_path = Path(review_path).expanduser().resolve()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    sources = review.get("sources") or {}
    timeline_path = Path(timeline_path or sources.get("timeline") or "").expanduser()
    if not timeline_path.is_file():
        raise FileNotFoundError(f"Saved timeline not found: {timeline_path}")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline_points = timeline.get("points") or []
    review_points = review.get("points") or []
    key = review.get("probability_key") or "terran_win_probability"
    if len(timeline_points) != len(review_points) or not review_points:
        raise ValueError("review points and saved timeline points differ in length")
    for saved, original in zip(review_points, timeline_points):
        if (abs(float(saved["cutoff_seconds"]) - float(original["seconds"])) > 1e-9
                or abs(float(saved[key]) - float(original["probability"])) > 1e-9):
            raise ValueError("review probabilities no longer match the saved timeline")
    threshold = float(review.get("threshold") if threshold is None else threshold)
    merge_seconds = float(merge_seconds)
    intervals = pivotal_intervals(
        [float(point["cutoff_seconds"]) for point in review_points],
        [float(point[key]) for point in review_points],
        threshold=threshold, merge_seconds=merge_seconds)
    backup_path = review_path.with_name(review_path.stem + backup_suffix)
    if not backup_path.exists():
        shutil.copyfile(review_path, backup_path)
    focus_target = key == "focus_player_win_probability"
    focus_player_id = review.get("focus_player_id")
    race = review.get("focus_player_race") or review.get("probability_subject") or "unknown"
    review.update({
        "threshold": threshold,
        "merge_seconds": merge_seconds,
        "interval_rule": interval_rule(threshold, merge_seconds),
        "intervals_source": "computed_from_saved_timeline",
        "probability_statement": probability_statement(
            focus_target, focus_player_id, race),
        "pivotal_intervals": intervals,
        "evaluation_status": review.get("evaluation_status") or "unknown_split_membership",
        "limitations": " ".join(limitation_parts(
            focus_target, focus_player_id, race, threshold, merge_seconds,
            bool(review.get("explanations")))),
    })
    tmp = review_path.with_name(review_path.name + ".tmp")
    tmp.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, review_path)
    return {"review": str(review_path), "backup": str(backup_path),
            "intervals": len(intervals), "threshold": threshold,
            "merge_seconds": merge_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--timeline", type=Path, required=True, help="viz.timeline probability.json")
    parser.add_argument("--sample", type=Path, required=True, help="Processed POV directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for review.json")
    parser.add_argument("--gradcam", type=Path, help="viz.gradcam output directory (optional)")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--merge-seconds", type=float, default=120.0,
                        help="Maximum span for merging consecutive same-direction changes")
    args = parser.parse_args()
    report = build(args.timeline, args.sample, args.output, gradcam_dir=args.gradcam,
                   threshold=args.threshold, merge_seconds=args.merge_seconds)
    print(json.dumps({"sample_id": report["sample_id"],
                      "probability_key": report["probability_key"],
                      "subject": report["probability_subject"],
                      "points": len(report["points"]),
                      "intervals": len(report["pivotal_intervals"]),
                      "explanations": len(report["explanations"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
