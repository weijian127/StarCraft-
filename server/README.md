# Backend for `analysis.run_replay`

A single-process, standard-library HTTP service (no new dependencies) that accepts one
`.SC2Replay` upload, runs the existing `analysis.run_replay` pipeline in the background,
and publishes the two POV reports the current viewer already understands.

- One analysis at a time; a second upload gets `409 busy` (no queue).
- Real stage reporting only; no fabricated percentage.
- The browser can never choose SC2 path, team dir, checkpoint, output directory or
  commands. Uploads are stored under a server-generated name; only the job's own public
  report is served.
- Local mode and public-demo mode are the same code. Public mode adds an isolated jobs
  root, 128-bit job links, upload rate limiting and a storage cap.

## Requirements

- Python from this project's `.venv` (numpy, torch, pysc2, s2protocol — unchanged).
- `StarCraft II` with `Versions/Base82457` (5.0.4.82457). If missing, run once:
  `python -m tools.prepare_replay_dependencies --replay <replay> --download`.
- The team project (read-only) with `viz/`, `checkpoints/best.pt`.
- Map cache entries for the replay (or start with `--allow-download-map`).

## Start (local mode)

```powershell
.\.venv\Scripts\python.exe -m server.app `
  --team-dir <team-project> `
  --jobs-root runs\backend-jobs
```

Useful flags: `--host` (default `127.0.0.1`), `--port` (default `8765`; if taken the next
free port is used, `0` lets the OS choose), `--sc2-path`, `--checkpoint`, `--max-upload-mib`
(default 20), `--allow-download-map`, `--python`, `--engine-timeout`, `--log-file`,
`--lock-file`, `--job-timeout`.

On start it prints `listening on http://127.0.0.1:<port>`; `/api/support` is the machine
readable description of the same settings. Jobs left `queued`/`running` by a previous
process are marked `failed` / `interrupted` at startup and reported as such.

## Start (public-demo mode)

Public mode is still loopback-only; a tunnel forwards to it. It must use a separate jobs
root and the **same** heavy-task lock as any local backend:

```powershell
.\.venv\Scripts\python.exe -m server.app `
  --public-demo --host 127.0.0.1 --port 8766 `
  --team-dir <team-project> `
  --jobs-root runs\public-jobs `
  --lock-file runs\backend-jobs\.active.lock `
  --allow-download-map `
  --uploads-per-hour 6 --rate-window-seconds 3600 `
  --job-timeout 3600 --max-jobs-mib 4096
```

- `--public-demo` refuses to run on `runs/backend-jobs`; old local reports are not exposed.
- Public job IDs end in 32 hex characters (128 random bits), not the six-hex local suffix.
- There is still no task-list endpoint. A report URL is a bearer link; anyone holding it
  can view that report.
- The public jobs root has a configurable byte cap (`--max-jobs-mib`). At the cap new
  uploads return `507 storage_full`; existing runs are never auto-deleted.
- Uploads are limited per client (`--uploads-per-hour`, sliding
  `--rate-window-seconds` window); the excess gets `429 rate_limited`.
- One wall-clock timeout (`--job-timeout`) covers the analysis child; its whole process
  tree is terminated and the job records `job_timeout`.
- `--lock-file` must point at the local backend's lock (the local default is
  `<jobs-root>/.active.lock`). Two backends with different `--jobs-root` values then
  cannot run heavy tasks concurrently.

## Stop

`Ctrl+C` (SIGINT/SIGTERM) stops accepting requests, terminates the analysis process **tree**
it started (analysis -> processor pool -> SC2 engine) and marks that job `interrupted`.
Nothing else is touched. Killing the process hard also ends the tree: the child is assigned
to a Windows job object with `KILL_ON_JOB_CLOSE`. The next start then marks the job
`interrupted`.

## API

### `GET /api/support`

```json
{
  "public_demo": {"enabled": true,
                  "report_access": "Anyone who has a report URL can view that report."},
  "engine": {"game_version": "5.0.4.82457", "base_build": 82457,
             "data_version": "D2707E265785612D12B381AF6ED9DBF4"},
  "replay_requirements": {"players": 2, "matchup": ["Terran", "Protoss"],
                          "results": "one player wins and the other loses"},
  "upload": {"field": "file", "max_bytes": 20971520, "max_mib": 20.0,
             "rate_limit": "6 uploads per 3600 s per client",
             "enforced_on": "bytes actually received, not only Content-Length"},
  "analysis": {"device": "cpu", "frames": 50, "step_seconds": 60.0,
               "draws": 1, "seed": 2026, "povs": 2},
  "jobs": {"max_concurrent": 1, "busy_status": 409, "progress_percent": null,
           "timeout_seconds": 3600},
  "limits": {"public_mode": true, "max_jobs_bytes": 4294967296}
}
```

### `POST /api/jobs` (multipart, one file part named `file`)

```
202 {"job_id": "job-<timestamp>-<random>", "state": "queued",
     "status_url": "/api/jobs/job-...", "report_url": "/api/jobs/.../report"}
```

Returns as soon as the upload is validated and the job is created; analysis continues in
the background. Validation happens before any heavy work: empty/damaged archives are
`400 invalid_replay`, out-of-scope replays `422 out_of_scope_replay`, oversized bodies
`413 upload_too_large`, a running analysis `409 busy`. Rejected uploads create no job
directory.

### `GET /api/jobs/<job_id>`

```json
{"job_id": "...", "state": "running", "stage": "extracting", "pov": null,
 "created_at": "...", "started_at": "...", "updated_at": "...", "finished_at": null,
 "error_code": null, "error_message": null,
 "replay": {"replay_id": "...", "map_title": "Submarine LE", "terran_win_label": 1},
 "report": null}
```

`state` is `queued` | `running` | `completed` | `failed`; `stage` mirrors the analysis
(`validating`, `checking_dependencies`, `extracting`, `predicting`, `explaining`,
`exporting`, `completed`, `failed`, `interrupted`). When completed, `report` lists
`index_url` plus, per POV, `review_url`, `image_url` and `ui_url`.

### `GET /api/jobs/<job_id>/report`

`200` with the browser-safe index (replay identity, engine, config, both POVs, points and
the chosen explanation with relative URLs). `409 not_ready` while the job is not completed,
`404` for an unknown id.

### `POST /api/jobs/<job_id>/heatmaps`

JSON body with exactly the controlled fields below; never a path or command:

```json
{"player": "player1", "cutoff_seconds": 660.0}
```

- `202` starts one queued heatmap and returns `status_url`. A completed cache hit returns
  `200` with `"cached": true`; the same in-flight request returns its current status.
  Another active heavy task (analysis or a different heatmap) returns `409 busy`.
- The cutoff must already be one of the report's saved prediction points
  (`422 unknown_cutoff`). The target class is derived from that point's saved probability.
- Generation reuses `run/processed/both/<sample_id>`, the checkpoint recorded by the report,
  and the report's POV/frames/seed/label definition. With `draws=1` the generated
  probability must equal the saved timeline probability; a mismatch is reported and the
  timeline/report probabilities are not modified.
- The job state stays `completed`; the heatmap has its own `queued|running|completed|failed`
  status. Repeated identical requests reuse the cache (job, POV, cutoff, target, checkpoint
  SHA-256, frames, seed, draws, device, label definition).
- New images get unique names such as
  `gradcam_player1_660s_d58445ec07.png`; the original `gradcam_1.png` is never overwritten.

### `GET /api/jobs/<job_id>/heatmaps/<heatmap_id>`

Returns the heatmap task state, target meaning, generated and timeline probabilities,
probability-match result, native temporal-bin count, error fields and, when complete,
`image_url`. No percentage is reported.

### `POST /api/jobs/<job_id>/game-view`

JSON body with exactly `{"player": "player1", "cutoff_seconds": 660.0}`. The cutoff must
already have a heatmap explanation with a saved context game loop. The backend re-renders
**one** RGB frame from the original replay using the original engine build and observed
player, under the same heavy-task slot as analysis/heatmaps. Results are cached in the
review's `game_view` field and the unique `rgb_<pov>_<loop>_<key>.png` file; a repeated
request returns the cached image without starting SC2. This is explicitly a
**Re-rendered game view**, not the model-input RGB, and it does not rerun the model or
analysis. The response is synchronous and contains `actual_game_loop`/`actual_seconds`
plus `image_url`; it does not expose a percentage. Failures leave the existing report
unchanged and return an error code/message.

### Static

- `GET /ui/<file>` — the existing viewer, read-only.
- `GET /jobs/<job_id>/report/<player>/review.json` and approved heatmap/minimap/RGB PNGs —
  the only report files served. `review.json` is sanitised (no absolute paths) so its
  relative image names resolve. The original `gradcam_1.png`, generated
  `gradcam_<player>_<cutoff>s_<key>.png`, `minimap_<player>_<cutoff>s<key>.png` and
  `rgb_<player>_<loop>_<key>.png` files are served only when the job record lists them
  for that player; arbitrary paths are rejected.

Open: `http://127.0.0.1:8765/ui/index.html?review=/jobs/<job_id>/report/player1`

## Testing it

```powershell
# 1. start the service, then describe it
curl.exe http://127.0.0.1:8765/api/support

# 2. upload (curl sends a file part named "file")
curl.exe -F "file=@data_sample\<replay>.SC2Replay" http://127.0.0.1:8765/api/jobs

# 3. poll until state is completed
#    (expect extracting -> predicting -> explaining -> exporting)
curl.exe http://127.0.0.1:8765/api/jobs/<job_id>

# 4. read the report
curl.exe -o report.json http://127.0.0.1:8765/api/jobs/<job_id>/report
curl.exe -o review1.json http://127.0.0.1:8765/jobs/<job_id>/report/player1/review.json
curl.exe -o cam1.png http://127.0.0.1:8765/jobs/<job_id>/report/player1/gradcam_1.png
```

Windows PowerShell 5.1 has no `-Form`, so use `curl.exe` or a small `http.client` script.
During a run, upload again to see `409 busy`; only one SC2 process starts.

## Errors and limits

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `missing_file` / `too_many_files` | multipart body has no single file part |
| 400 | `empty_file` / `empty_body` | zero-byte upload |
| 400 | `invalid_replay` | MPQ/metadata cannot be read |
| 400 | `unsupported_content_type` | not `multipart/form-data` |
| 409 | `busy` | an analysis or another heatmap is already running |
| 409 | `not_ready` | report requested before the job completed |
| 409 | `checkpoint_changed` / `processed_data_missing` | the original report inputs are unavailable |
| 409 | `context_time_missing` / `replay_missing` | a game view has no saved context loop or replay copy |
| 400 | `invalid_player` / `invalid_cutoff` / `invalid_json` | heatmap or game-view request fields are not the controlled pair |
| 404 | `unknown_pov` / `heatmap_not_found` | unknown POV or heatmap id |
| 422 | `unknown_cutoff` / `no_explanation` | cutoff is not a saved point or has no explanation |
| 500 | `game_view_failed` / `game_view_time_mismatch` | RGB rendering failed or reached a different game loop |
| 411 | `length_required` | no `Content-Length` (chunked uploads unsupported) |
| 413 | `upload_too_large` | body exceeds `--max-upload-mib` |
| 422 | `out_of_scope_replay` | not 5.0.4.82457 / not 1v1 / not Terr vs Prot / not Win+Loss |
| 429 | `rate_limited` | public-demo per-client upload limit reached; try later |
| 507 | `storage_full` | public-demo jobs root reached `--max-jobs-mib`; old runs kept |
| 404 | `job_not_found` / `not_found` | unknown job, player, file or route |

Not implemented on purpose: queueing, resuming an interrupted analysis, upload progress
percentages, arbitrary file reads, CORS, and a task-list endpoint. The public demo is an
opt-in loopback mode forwarded by a tunnel; it is not a general hosting platform.

## On disk

```text
<local jobs-root>/
  job-<UTC>-<6hex>/            historical local IDs remain readable
    ...
  .active.lock                 default local heavy-task lock

<public jobs-root>/
  job-<UTC>-<32hex>/           public demo: 128 random bits
    job.json                   state, timings, error, report links (atomic writes)
    upload/input.SC2Replay     server-named copy of the upload
    logs/run_replay.log        analysis stdout/stderr
    run/                       untouched analysis.run_replay output (status, npz, reviews)
    public/playerN/            sanitised review.json + approved PNGs (served)
  .incoming/                   temporary validated upload; not served
  server.log                   backend log (stack traces stay here)
```

Pass the same `--lock-file` to local and public backends so these separate roots share one
heavy-task slot. `run/` (NPZ archives, internal logs, the uploaded replay) is never served.
The upload field's filename is display-only: the stored file is always `input.SC2Replay`,
and request paths are resolved inside allow-listed directories only.
