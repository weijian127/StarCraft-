# Quick start: replay upload, report and public demo

This is the short path for the local replay service and the opt-in public demo. It accepts
one `.SC2Replay`, runs the existing `analysis.run_replay` pipeline, and serves both POV
reports in the browser. It does not train a model or produce video.

The repository is source-only. It deliberately excludes `.venv`, `runs`, processed
datasets, sample replays, the external team project, the model checkpoint, the StarCraft II
engine/maps/cache and tunnel binaries. **Cloning this repository alone is not enough to
analyze a replay**; prepare the external resources below first.

For implementation details see [`server/README.md`](server/README.md) and
[`ui/README.md`](ui/README.md). For dataset extraction, training and the offline
review exporter, see the sections in [`README.md`](README.md).

## External prerequisites

These values were verified in the development environment; replace the placeholders with
your own paths:

| Item | Required value |
| --- | --- |
| Python environment | `.venv\Scripts\python.exe`, Python 3.10.x |
| Python packages | existing `.venv`: torch 2.11.0+cu128, pysc2, s2protocol |
| SC2 installation | `<StarCraft II installation containing Versions/>` |
| Required engine | `Versions\Base82457\SC2_x64.exe`, **5.0.4.82457** |
| Engine build / data version | base build **82457**, data version **D2707E265785612D12B381AF6ED9DBF4** |
| Verified map | **Submarine LE** in the development runs |
| Team project | `<external team project>` (read-only; not in this repository) |
| Checkpoint | `<team-dir>\checkpoints\best.pt` (default; override with `--checkpoint`) |
| Jobs directory | `runs\backend-jobs` (created locally; not committed) |

The backend accepts only this engine version. The exact accepted scope is in
`GET /api/support`.

## First-time preparation

Skip any step that is already true. Do not reinstall or retrain for normal use.

1. **Python environment.** From the project root, create/sync the locked environment
   once:

   ```powershell
   Set-Location <path-to-cloned-repo>
   uv sync
   ```

   The development `.venv` contains the required packages and Python 3.10; it is not
   committed.

2. **StarCraft II engine.** Confirm the required build exists in your own installation:

   ```powershell
   Test-Path '<StarCraft II installation>\Versions\Base82457\SC2_x64.exe'
   ```

   If it is missing, obtain it once with a replay you provide:

   ```powershell
   .\.venv\Scripts\python.exe -m tools.prepare_replay_dependencies `
     --replay <path-to-replay>.SC2Replay --download
   ```

   The service never substitutes another installed build.

3. **Team code and checkpoint.** Supply the external team project separately; it must
   contain `viz\timeline.py`, `viz\gradcam.py` and `checkpoints\best.pt`. It is used
   read-only. Pass the project with `--team-dir`; a relative `--checkpoint` is resolved
   inside it. The model, weights and label definitions are not changed by this service.

4. **Map cache.** The replay's referenced Battle.net cache archives must exist. The
   development runs used `Submarine LE`. Start with `--allow-download-map` to fetch only
   this replay's missing archives from the official depot, each verified by content hash.
   Map sets and game packages are never downloaded.

## Daily start

Open a PowerShell terminal in the project root and run the complete command:

```powershell
Set-Location <path-to-cloned-repo>
.\.venv\Scripts\python.exe -m server.app `
  --team-dir "<external team project>" `
  --jobs-root "runs\backend-jobs" `
  --allow-download-map
```

Wait for a line like:

```text
listening on http://127.0.0.1:8765
```

The default host/port is `127.0.0.1:8765`. If 8765 is already occupied, the backend
tries the next free port; use the address it prints. Stop the previous backend before
starting a replacement.

Open the actual home page:

```text
http://127.0.0.1:8765/ui/index.html
```

`http://127.0.0.1:8765/` redirects there. The optional static file server on port 8000
is separate and can keep running; it only serves files and is not used for uploads.

## Use the page

1. Read the **Supported input** panel. It comes from `GET /api/support`.
2. Select one `.SC2Replay` file (maximum 20 MiB, 1v1 Terran vs Protoss, one winner
   and one loser, engine 5.0.4.82457).
3. Click **Start analysis**. Only one analysis runs at a time; a second upload gets
   `409 busy`.
4. Watch the real backend stages: validate, check engine, extract observations,
   predict, Grad-CAM, export. There is no fabricated percentage. A temporary status
   connection failure shows that the background job may still be running, retries
   with capped increasing delays, and can be retried manually without uploading again.
5. The report opens automatically when the job completes. Use the POV buttons to
   switch player perspectives; the curve, selected point and Grad-CAM heatmap reload.
   Opening a POV prepares up to three candidate-interval heatmaps in the background;
   the Grad-CAM pane uses the exact requested time and offers **Generate heatmap at
   [time]** for a cache miss. Moving the minimap slider never starts inference. Every
   request reuses the job's saved observations and original checkpoint and does not
   rerun the analysis. The report also shows a model-input minimap reference and can
   optionally render one cached RGB "Re-rendered game view" at the saved context game loop.
6. Keep the job id in the URL (`?job=...&pov=player2`) to return to the same report
   after a refresh.

After a job completes, keep its locally generated `?job=<job-id>&pov=player1` URL to
return to that report after a refresh. Jobs and reports are stored only under the local
`runs/` directory and are not shipped with the repository.

## Stop

Press `Ctrl+C` in the terminal running `server.app`. It stops accepting requests,
terminates only its own analysis process tree, and marks a running job interrupted.
The next start keeps partial files and reports the job as interrupted. Do not stop
the unrelated port-8000 static server.

## Public demo (Cloudflare Quick Tunnel)

This mode is for sharing a temporary course demo. The backend still listens only on
`127.0.0.1`; Cloudflare forwards to it. Do not forward `8000`/`8123` and do not open a
router inbound port.

### Start the public backend

Use a separate terminal from the local backend:

```powershell
Set-Location <path-to-cloned-repo>
.\.venv\Scripts\python.exe -m server.app `
  --public-demo --host 127.0.0.1 --port 8766 `
  --team-dir "<external team project>" `
  --jobs-root "runs\public-jobs" `
  --lock-file "runs\backend-jobs\.active.lock" `
  --allow-download-map `
  --uploads-per-hour 6 --rate-window-seconds 3600 `
  --job-timeout 3600 --max-jobs-mib 4096
```

Wait for `mode: public-demo` and `listening on http://127.0.0.1:8766` / the next free
port. `runs\public-jobs` is isolated; `runs\backend-jobs` and historical reports are not
served by this backend. Public job IDs have 32 hex random characters (128 bits).

The `--lock-file` must be the same file used by the local backend. The local default is
`runs\backend-jobs\.active.lock`; this is why the command above points there even though
the jobs root is different.

### Start the tunnel

If `cloudflared` is not already installed, this repository's existing ignored tools folder
can hold the official standalone binary:

```powershell
$cf = Join-Path (Get-Location) ".tools\cloudflared\cloudflared.exe"
if (-not (Test-Path $cf)) {
  New-Item -ItemType Directory -Force .tools\cloudflared | Out-Null
  Invoke-WebRequest `
    -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
    -OutFile $cf
}
& $cf tunnel --url "http://127.0.0.1:8766" --no-autoupdate
```

Wait for a line like:

```text
Your quick Tunnel has been created! Visit it at:
https://<random-words>.trycloudflare.com
```

That HTTPS string is the real public URL; there is no fixed address. Open it and test the
home page at `/ui/index.html`. The homepage reads `/api/support` and shows the actual
engine, matchup, upload limit, public rate limit, job timeout and demo notice.

### Stop public access

Press `Ctrl+C` in the cloudflared terminal first, then `Ctrl+C` in the public backend
terminal. If they were started as background processes with PID files, use:

```powershell
$tunnel = Get-Content runs\public-jobs\tunnel.pid
$backend = Get-Content runs\public-jobs\backend.pid
Stop-Process -Id $tunnel -Force
Stop-Process -Id $backend -Force
```

Stopping the tunnel makes the public URL stop working; it does not delete
`runs\public-jobs`. The local backend and `runs\backend-jobs` remain intact. Quick Tunnel
URLs can change after a restart, and service stops when this PC sleeps or shuts down.

### Restore local-only mode

Stop the public backend and tunnel, then start the local mode from the earlier **Daily
start** section, or keep using an already running `127.0.0.1:8765` backend. Do not expose
port 8765.

## Current scope and limits

- One analysis at a time, no queue, no cancel button, no upload percentage.
- Supported replay: engine 5.0.4.82457, exactly two human players, Terran vs Protoss,
  exactly one Win and one Loss. Unsupported versions, other matchups, non-1v1 and
  damaged archives are rejected.
- CPU prediction with 50 frames, step 60 s, 1 draw, seed 2026; one static Grad-CAM
  per POV. No training, video, tactical advice or persistent public hosting.
- The report is for pipeline inspection; probabilities are not calibrated and
  heatmaps are model evidence, not causal explanations.
- The local UI is English and has no CORS. Public demo mode is an opt-in backend mode
  forwarded by a temporary Cloudflare Quick Tunnel; it still has no CORS, no task list,
  no arbitrary file access and no persistent hosting guarantee. Uploads are stored under
  server-generated names; only the job's own public report is served.
- Public demo additionally uses 128-bit job IDs, a separate `runs\public-jobs` root,
  per-client upload rate limiting, a shared heavy-task lock, a job timeout and a
  configurable storage cap. At the cap it rejects new uploads and keeps old runs.
- Verification note: the current UI session verified upload, status, report, POV
  switching, refresh recovery and simulated connection interruption/recovery with
  real completed jobs. A real unsupported Zerg-vs-Protoss replay was rejected with
  `422 out_of_scope_replay`; the public upload route was also checked with an empty
  file and an oversized file.
