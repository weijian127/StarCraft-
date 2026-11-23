# `analysis.run_replay` — one replay, end to end

One command takes a raw `.SC2Replay` through validation, dependency checks, both-POV
extraction, both timelines, one automatically chosen Grad-CAM per POV, the adapter, and a
report index. It sequences existing code only: `data_processing.processor`,
`tools.prepare_replay_dependencies`, the team `viz.timeline` / `viz.gradcam`, and
`analysis.adapt_viz`. Nothing is trained, no video is produced, and no model, checkpoint,
label or version check is modified.

## Prerequisites

1. `StarCraft II` installed with `Versions/Base82457` present (the 5.0.4.82457 engine).
   If it is missing, obtain it once with the official mechanism:
   `python -m tools.prepare_replay_dependencies --replay <replay> --download`.
2. The team project (read-only) containing `viz/timeline.py`, `viz/gradcam.py` and
   `checkpoints/best.pt`. Pass `--team-dir` or set `SC2_TEAM_DIR`; alternatively pass an
   absolute `--checkpoint` and the team root is inferred from it.
3. The replay's Battle.net cache entries. They are checked before anything heavy runs. A
   missing archive stops the run with `missing_map_cache`; re-run with
   `--allow-download-map` to fetch **only the archives this replay references** from the
   official depot (`https://<region>-s2-depot.classic.blizzard.com/<sha256>.<type>`),
   each verified by content hash. Map sets and game packages are never downloaded.

## Complete command

```powershell
.\.venv\Scripts\python.exe -m analysis.run_replay `
  --replay data_sample\<replay>.SC2Replay `
  --team-dir <team-project> `
  --checkpoint checkpoints\best.pt `
  --frames 50 --step 60 --draws 1 --seed 2026 --device cpu
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--output <dir>` | New run directory. An existing path is refused (exit 2). Default: `runs/run-replay-<UTC timestamp>`. |
| `--sc2-path <dir>` | StarCraft II folder containing `Versions/`. Default: `SC2PATH` env or the local `config.py` value. |
| `--team-dir <dir>` | Team project (read-only). Default: `SC2_TEAM_DIR`, else inferred from `--checkpoint`. |
| `--checkpoint <path>` | Relative values resolve inside `--team-dir`. Default `checkpoints/best.pt`. |
| `--frames / --step / --draws / --seed` | Prediction sampling. Defaults `50 / 60 / 1 / 2026`. |
| `--device cpu\|cuda` | Default `cpu`. |
| `--allow-download-map` | Fetch this replay's missing cache archives (hash-verified) instead of failing. |
| `--bnet-cache <dir>` | Battle.net `Cache` root. Default: the engine's own cache. |
| `--fetch-timeout / --engine-timeout / --predict-timeout / --explain-timeout / --extract-timeout / --export-timeout` | Per-stage limits in seconds. |
| `--threshold / --merge-seconds` | Candidate-interval settings passed to the adapter. Default `0.1 / 120`; the 120 s cap is a local display rule. |
| `--python <exe>` | Interpreter for subprocesses. Default: the interpreter running this module. |
| `--lock-file <path>` | Single-run lock. Default `runs/.run_replay.lock`. |

Exit codes: `0` completed, `1` a stage failed (see `status.json`), `2` output directory
already exists, `3` another run holds the lock (`busy`).

## Output structure

```text
<run>/
  input/<replay>                 verbatim copy of the input
  input.json                     source path, stored path, sha256, bytes
  params.json                    every resolved parameter
  status.json                    machine-readable state (atomic replacement)
  logs/
    validating.log  checking_dependencies.log  extracting.log
    predicting.log  explaining.log  exporting.log
    engine_check.log  extracting-processor.log  predicting-playerN.log
    explaining-playerN.log  exporting-playerN.log  team_preflight.log
    prepare_replay_dependencies.log
  processed/both/manifest.json + one folder per POV
    (SpatialFeatures.npz, TimeInfo.npz, PlayerMetaInfo.json)
  timeline/player<N>/probability.json|png|svg
  gradcam/player<N>-<seconds>s/snapshot.png|svg, data.npz, method.json
  review/player<N>/review.json + gradcam_1.png
  report_index.json
```

`report_index.json` carries the replay identity (id, SHA-256, map, label), the measured
engine (`base_build`, `data_version`), the checkpoint hash, the run parameters, and one
entry per POV: player id, race, result, the focus-player label, relative `review` /
`timeline` / `gradcam` paths, the full probability curve, and the chosen explanation
(cutoff, target class, target meaning, selection rule, delta, image path). Each
explanation also records the deterministic sampled frame indices/seconds/game loops,
a 512×512 model-input minimap reference at the last sampled frame, and a raw Grad-CAM
peak summary; no model or timeline is rerun for these fields.

## `status.json`

Fields: `format_version`, `state` (`running` / `completed` / `failed`), `stage`,
`started_at`, `updated_at`, `finished_at`, `pov`, `error_code`, `error_message`,
`report_index`, and `stages[]` with per-stage `started_at` / `finished_at` / `seconds` /
`status`. No percentage is written: there is no real progress measurement. A failed run
ends as `failed` with a non-null `error_code`; it never stays `running` or claims
`completed`.

Stages: `validating` -> `checking_dependencies` -> `extracting` -> `predicting` ->
`explaining` -> `exporting` -> `completed`.

Common `error_code` values:

| Code | Meaning / next step |
| --- | --- |
| `invalid_replay` | The archive cannot be decoded (truncated/corrupt). Nothing heavy ran; the run folder with its copy and logs is kept. |
| `out_of_scope_replay` | Not 5.0.4.82457, not exactly two players, not Terr vs Prot, or not one Win/one Loss. The message lists what was found. |
| `missing_engine` | `Versions/Base82457` or its `SC2_x64.exe` is absent. Run the `--download` step once. |
| `missing_map_cache` | Cache archives are absent. Re-run with `--allow-download-map`. |
| `download_failed` / `cache_conflict` / `cache_write_failed` | Depot or cache problem; staged files and logs are kept, nothing is overwritten. |
| `engine_check_failed` | The dependency tool did not confirm the required engine or could not read the replay. See `logs/engine_check.log`. |
| `extraction_failed` / `extraction_incomplete` | The processor failed, or the manifest/NPZs/timestamps/player identity did not verify. |
| `timeline_failed` / `no_prediction_points` | `viz.timeline` failed or produced no usable points. |
| `gradcam_failed` / `explanation_mismatch` | `viz.gradcam` failed, or its probability did not match the timeline point (sampling/seed drift). |
| `adapter_failed` / `adapter_label_mismatch` | The adapter failed, or the focus-player label disagreed with the outcome. |
| `team_dir_missing` / `checkpoint_missing` | The team project or checkpoint could not be resolved. |
| `internal_error` / `stage_timeout` | Unexpected error or a stage exceeded its timeout; children are terminated. |

## Behaviour worth knowing

* **Label semantics.** `manifest.json` labels are match-level (`1 = Terran wins`). The team
  checkpoint is focus-player (`1 = focus player wins`). The adapter flips once, so player 2
  (Protoss, who lost) gets `label = 0`; the tool asserts this instead of trusting it.
* **Explanation choice.** With two or more points, the later point of the largest adjacent
  absolute probability change is explained; a single point explains itself; no points is an
  error. The target is class 1 when the point probability is `>= 0.5`, otherwise class 0,
  and its meaning is recorded (`focus player wins` / `focus player loses`), never assumed to
  be "Protoss". Candidate intervals are computed separately by the adapter: every
  adjacent change at or above the threshold starts a candidate, and only consecutive
  same-direction changes may merge while their total span is at most the configured cap
  (default 120 s).
* **Consistency.** The Grad-CAM uses the same checkpoint, POV, `--frames` and `--seed` as the
  timeline, and its exported probability is compared with the timeline point (`draws = 1`
  must match within `1e-6`; with `draws > 1` the single explanation must lie inside the
  recorded draw range, and both values are reported).
* **Isolation.** Team code runs only as subprocesses with the team directory as `cwd` and
  `PYTHONPATH` removed; a preflight proves `config`/`viz` resolve inside the team project.
  This project's modules are used for its own subprocesses, always via `cwd` = project root.
* **Failures are kept.** Failed run directories are never deleted; only the subprocesses a
  run started are terminated. Another run, the HTTP server, or other processes are never
  touched.

## Checking a finished run in the browser

Serve the project root once (this is the existing read-only server; the tool never starts
or stops it):

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Then open, for each entry in `report_index.json["povs"]`:

```text
http://127.0.0.1:8000/ui/index.html?review=<run>/review/player<N>
```

The point with the explanation is ringed and selected by default; selecting any other point
clears the previous heatmap and shows which cutoffs do have one.
