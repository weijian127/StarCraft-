# Replay review UI

A dependency-free HTML/CSS/JavaScript front end for the backend and for the
`review.json` files written by `analysis/review.py`. No build step, no npm packages, no
Python packages.

It uses same-origin relative paths (`/api/...`, `/jobs/...`, `/ui/...`) in both local and
public-demo mode, so a tunnel hostname never causes a request to `127.0.0.1` or
`localhost`. The homepage reads `/api/support`; public mode also shows a **PUBLIC COURSE
DEMO** notice and the bearer-link report-access warning.

It supports two flows in the same page:

| URL | What it does |
| --- | --- |
| `/ui/index.html` | Upload entry + live analysis status + report |
| `/ui/index.html?job=<job id>` | Reconnect to an existing job (status, then report) |
| `/ui/index.html?job=<job id>&pov=player2` | Open the report for one POV |
| `/ui/index.html?review=<directory>` | Legacy viewer for an exported review directory |

The upload entry reads `GET /api/support` and never hardcodes the accepted engine, limits
or analysis settings. A `.SC2Replay` is posted to `POST /api/jobs` as multipart/form-data
in the `file` field. Client-side checks cover the extension, an empty file and the
maximum size; server errors (`400`, `409 busy`, `413`, `422`, public `429 rate_limited`,
`507 storage_full`) are shown with the backend message. Only one analysis runs at a time,
so the status page offers no cancel or duplicate-submit control.

The status page polls `GET /api/jobs/<job_id>` and shows only the stage the backend
reports. There is no progress percentage. The job id is kept in the URL and in
`localStorage`, so a refresh resumes the same job; a completed job opens its report
automatically. Transient status-network failures keep the last known stage and are
retried after 3, 6, 12 and 24 seconds (capped at 30 s). Each retry queries only the
same job id; it never uploads again. After four automatic retries the page stops,
states that the background job may still be running, and offers **Retry connection**.
Leaving or switching the job aborts the pending request and clears the retry timer.
`completed`, `failed` and `interrupted` responses stop polling.

The report fetches `GET /api/jobs/<job_id>/report` for the POV list and then loads each
POV's sanitised `review.json`. The report is a review workspace rather than a long page:

- a toolbar with the match title (the report index's `map_title`, otherwise the replay id),
  the POV buttons, the selected player's actual result and a link back to the upload entry;
  the full match id lives in the collapsed **Match details** panel;
- a **Timeline** card with the probability curve, candidate bands, heatmap rings and the
  50% reference line;
- a **Suggested moments** card whose **Previous/Next** buttons follow the actual selected
  cutoff of each moment;
- a **Preview** card with Minimap / Game view / Grad-CAM tabs that share one image stage,
  so only one image is visible at a time;
- technical details last, collapsed by default.

The preview is driven by one selection. Changing the cutoff or the POV clears the previous
image, updates the selection strip and reloads each tab's content for the new moment.
Tabs only switch visibility; they never start a generation request. The Minimap tab is the
default and shows the replay-minimap browser described below.

### Timeline selection

Clicking a prediction dot or a candidate band, or using `Enter`/`Space` on a focused dot or
band, calls the same `selectPoint()` / `selectInterval()` used by the Previous/Next moment
buttons. The graph is **not** draggable: moving across it does not change the selection. The
selected dot keeps its drop line and probability label; the strip above the preview carries
the full selection text.

### Background explanations and custom Grad-CAM times

Opening a completed POV asks the backend to prepare at most **three** candidate
explanations in the background. Candidates are the existing Suggested moments ranked by
absolute net change; each uses its saved maximum-change cutoff, duplicates are removed and
already cached explanations are skipped. Fewer candidates mean fewer jobs; when there are
no intervals, the existing single-explanation fallback is preserved (an explanation that
already exists is reused, otherwise the later side of the largest adjacent change is used).
The report is rendered before preparation starts and the analysis job stays `completed`.

Generation uses the same single queue and heavy-task slot as explicit requests, one job at
a time, and releases the slot between explanations. Explicit button clicks are dequeued
before remaining background work; the operation already running is allowed to finish.
Repeated preparation calls are idempotent and reuse successful cache entries after a
restart. A failed background job is recorded in its own status file and never invalidates
the report; it can be retried by the button or by reopening the POV.

The Grad-CAM pane has its own requested time, separate from the selected prediction and
the displayed minimap frame:

- clicking a prediction uses that prediction's saved cutoff (even when its sampled frame
  is slightly earlier);
- moving the minimap slider sets the requested Grad-CAM time to the **browsing time**;
  merely moving it never starts inference or SC2;
- when the Grad-CAM tab is opened, an exact cached explanation is shown, otherwise the
  button reads **Generate heatmap at [time]** and generation starts only on click;
- a custom time outside the curve uses the actual probability from that inference. No
  nearby curve probability is copied and no curve point is added; its target class follows
  the existing policy (class 1 when the inference probability is at least 0.5);
- saved curve cutoffs keep the existing probability consistency checks (draws=1 equality,
  or the saved draw range);
- the heatmap is cleared when the requested time or POV changes. A late result may be
  cached, but it cannot replace the currently requested time.

Cache identity includes job, POV, exact cutoff, checkpoint SHA-256, frames, seed, draws,
device, label definition and target class; cutoffs are not rounded to whole minutes.

### Replay minimap time slider

Below the Minimap image there is a native `<input type="range">` that browses the POV's
saved observations, not prediction points:

- the slider spans every observation in the job's `TimeInfo.npz` (for example 3754 frames
  for a 22-minute replay, about 0.357 s apart), so it is much denser than the 23 prediction
  points and does not require an explanation;
- the image is rendered from that observation's saved `player_relative` and
  `visibility_map` with the existing ownership/visibility palette; `frame_index`,
  `frame_seconds` and `game_loop` come from TimeInfo. Nothing is interpolated and no frame
  is synthesised, so identical adjacent frames are allowed;
- the caption calls it a **Replay minimap** and says explicitly that it is not necessarily
  a model input sample. **Technical details** keeps the prediction's own sampled frame in
  *Prediction reference frame* and the browsed one in *Replay minimap frame*;
- *Browsing* follows the slider immediately; *Showing* only changes after that exact image
  has loaded, so the time label always describes the image on screen. While a new frame is
  loading the previous image and its label stay visible;
- the probability line uses the nearest saved prediction **at or before** the browse time
  and names its prediction time. Before the first prediction it says there is no prediction
  yet. It is never presented as a per-frame prediction;
- clicking a prediction or candidate jumps the slider to the observation that prediction
  used when the export names it, otherwise the server maps the cutoff to the last sampled
  observation;
- requests are throttled to one per 60 ms with a trailing request (leading + trailing), the
  previous request is aborted, and a request token plus the current slider index check
  before an image is displayed. A slow older response can therefore never replace the
  current frame, and POV changes clear the old image and frame list;
- the native slider supports mouse, touch and keyboard (`Left`/`Right`/`Home`/`End`); its
  `aria-valuetext` names the observation time and index. Dragging never calls the heatmap or
  game-view generators.

The model-input context minimap (the explanation's sampled frame and the optional
Grad-CAM peak outline) now lives in the **Grad-CAM** tab, so the peak outline is never
drawn over a different, browsed frame. Missing Grad-CAM/RGB still shows the existing empty
state with **Generate heatmap** / **Render game view**. Demo and legacy exports keep their
old "not recorded" behaviour because they have no job preview API.

The main view shows only the probability of the current object (`probability_key` /
`probability_subject`); the complement is not displayed because it is not the other POV's
prediction. The actual match result stays in the toolbar and in **Match details**.

The report:

- labels the curve as the estimated probability that the current POV's player wins,
  based on that POV's observations;
- uses POV buttons that reload every label, the curve, the selection and the preview;
- uses `target_meaning` for the heatmap class and keeps `displayed_frame_seconds`
  separate from the prediction cutoff;
- removes the previous image immediately for a point without an explanation;
- uses request tokens so a slow, older POV/image/heatmap request cannot overwrite a
  newer selection.

Candidate intervals are recomputed from saved timeline probabilities: every adjacent
change with `abs(delta) >= 0.1` is a candidate. Consecutive same-direction changes may
merge only while their total span is at most 120 s. The 120 s cap is a local display
rule, not a paper result. The default list shows the three largest by absolute net
change in time order; **Show all** expands the rest. Chart bands and list rows share the
same `intervalEntries()` data.

Each moment card shows its time range, `start → end` probability, the signed change in
percentage points and whether a heatmap exists. The maximum single step and the number of
adjacent changes are inside the card's own **Details** disclosure. The card area carries a
single sentence: “Large changes in model predictions; not confirmed causes of the result.”
The threshold, merge cap and “local display rule” note live in **Technical details**.

For a backend job, the Grad-CAM pane shows **Generate heatmap at [time]** when the exact
requested time has no cached explanation. The browser posts only `player` and
`cutoff_seconds`; the backend reuses the job's saved POV arrays and original checkpoint.
Saved prediction cutoffs keep their probability consistency checks; a custom browsing time
returns the probability produced by that inference. The UI shows real
queued/running/completed/failed states without a percentage. A completed result is written
into `review.json` with a unique image name, so a refresh keeps it; repeated requests use
the cache. Opening the POV also prepares up to three candidate explanations in the
background (see above). Static exports and demos show a clear “only available from a local
backend job” message and do not send an API request.

The **Replay minimap** slider (above) renders saved observations, not model samples. When an
explanation has saved sampling metadata, the **Grad-CAM** tab also shows the model-input
context minimap below the heatmap: self/enemy/neutral ownership colors, visibility colors
(hidden/seen/visible) and the last actually sampled frame. The legend labels are
`self`, `Neutral objects`, `enemy`, `hidden`, `seen`, `visible`. Neutral objects (including
resources) keep their original ownership color; secondary non-neutral ownership cells are
shown as self in the reference image. Raw unit categories and model inputs are unchanged.
The **Grad-CAM details and interpretation** panel names the context image time when the
export records `context_frame.frame_seconds`; **Technical details** has separate
*Prediction reference frame* and *Replay minimap frame* rows.

The **Grad-CAM** tab shows the heatmap with a short sentence that names the real
`target_meaning`; the full caption, the sampled history (`sampling.frames`, seconds and
source) and the separate image observation time stay inside the collapsed
**Grad-CAM details and interpretation** panel. The **Attribution summary** panel is a
separate collapsed panel in **Technical details**; it reads the raw `data.npz` maximum and
shows its temporal bin/cell or “no positive evidence”. An optional **Highest Grad-CAM
region** outline uses the coarse 4×4 grid. It is not masking or a decisive-position claim.

If that context loop is known, a backend job can also request one **Re-rendered game
view** RGB frame (640×480) from the original replay/engine. It is cached in `review.json`
and served as a unique `rgb_*.png`; it is explicitly not the model-input RGB. Clicking
the minimap, Grad-CAM PNG or RGB view opens a modal viewer with wheel zoom, drag panning,
Reset, Esc close and focus return.

The legacy `?review=` path is unchanged: it still reads a previously exported directory,
supports the same demo fixtures and keeps legacy Terran labels when the export has no
focus-player fields.

## Legacy viewer

It reads a previously exported review directory and shows:

- the match ID, player POV and actual result;
- a Terran win-probability curve over elapsed time, with the actual result marked;
- the detected pivotal intervals, selectable by click or keyboard;
- for a selected cutoff, the matching Grad-CAM heatmap, with the **prediction cutoff**
  and the **image observation time** labelled separately;
- which side the heatmap supports (`target_class`), and the masking diagnostic when
  the export contains one.

## Run it

Start the local backend from `server/README.md` (port 8765 by default), then open
`http://127.0.0.1:8765/ui/index.html` (`/` redirects there). The page must be served over
HTTP; opening `ui/index.html` directly from disk (`file://`) cannot work, because browsers
block `fetch()` on such pages. The page shows an explicit message in that case.

For the legacy viewer alone, a static server in the **project root** is enough:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

`http://127.0.0.1:8000/ui/index.html` then shows the upload page; `/api/support` is not
available there, so the upload controls report that the backend cannot be reached while
the legacy `?review=` links keep working.

## Public demo

The public-demo backend serves this same UI at its loopback origin. Open the HTTPS URL
printed by Cloudflare Quick Tunnel (for example
`https://<random-words>.trycloudflare.com/ui/index.html`). Every API, status, report and
image request is relative to that origin; do not replace the hostname with `127.0.0.1`.

The page shows the support conditions from that backend, a public-demo notice, the current
upload-rate/storage/timeout limits, and the statement that a report URL can be viewed by
anyone who has it. Public job links contain 32 hex random characters. The UI never offers a
job list and does not reveal internal paths.

## URL format

### Job and report URLs

```text
http://127.0.0.1:8765/ui/index.html
http://127.0.0.1:8765/ui/index.html?job=<job id>
http://127.0.0.1:8765/ui/index.html?job=<job id>&pov=<player1|player2>
```

- Without parameters the page is the upload entry.
- `job` resumes polling; a completed job is opened as a report.
- `pov` chooses the initial POV. The POV buttons keep this parameter in the URL, so a
  refresh stays on the same player.
- The report index comes from the backend; the page never guesses report paths.

### Legacy review URL

```text
http://127.0.0.1:8000/ui/index.html?review=<path-to-review-directory>
```

- `review` is resolved **from the server root**, not from `/ui/`, so
  `runs/my-run/review` and `/runs/my-run/review` both work.
- It may also point directly at a JSON file: `?review=runs/my-run/review/review.json`.
- Heatmap images are resolved **relative to the directory that contains `review.json`**,
  so the exported files must stay together:
  `review.json`, `gradcam_1.png`, `gradcam_2.png`, ... (`timeline.png` and the `.npz`
  arrays are not needed by this page).

### Demo URLs

All demo data is synthetic and is labelled **DEMO DATA** on the page:

| Scenario | URL |
| --- | --- |
| Two intervals, two heatmaps | `http://127.0.0.1:8000/ui/index.html?review=ui/demo/full` |
| No pivotal interval (last point shown) | `http://127.0.0.1:8000/ui/index.html?review=ui/demo/no-intervals` |
| Single prediction point, masking unavailable | `http://127.0.0.1:8000/ui/index.html?review=ui/demo/single-point` |
| Heatmap file missing on disk | `http://127.0.0.1:8000/ui/index.html?review=ui/demo/missing-image` |
| Empty export (`points: []`) | `http://127.0.0.1:8000/ui/index.html?review=ui/demo/empty` |

`ui/demo/missing-image/review.json` references `gradcam_1.png` on purpose but the file
is not present, to exercise the "image failed to load" path. That cutoff is the one the
page selects automatically, so the error state is visible on load; selecting the other
interval shows a working image.

> The demo numbers are invented for layout testing. They are not model predictions
> and do not describe a real match. Do not cite them anywhere.

## Fields used from review.json

| Field | Used for |
| --- | --- |
| `replay_id`, `sample_id`, `focus_player_id` | match header |
| `label` | actual result for the exported side: `1` = positive class won, `0` = lost |
| `result_text` | optional full result sentence (for example "Focus player (Protoss, player 2) lost") |
| `label_meaning` | optional; says whether the label is focus-player or Terran based |
| `probability_key` | which `points[].*` field holds the probability (default `terran_win_probability`) |
| `probability_subject` / `probability_opponent` | which race the probability refers to, and the other side |
| `probability_subject_detail` | optional, e.g. "focus player 2 (Protoss)" |
| `probability_statement` | export-generated sentence matching this POV's label definition |
| `checkpoint`, `checkpoint_epoch`, `checkpoint_sha256`, `engine` | technical details |
| `run_params` | original frames/step/draws/seed/device, shown in technical details |
| `time_grid.training` / `time_grid.inference` | technical details; training vs export cutoffs when recorded |
| `time_grid.override_applied` | shows the "cutoffs outside the training grid are unvalidated" warning |
| `time_grid.frames` | fallback frames-per-prediction value |
| `threshold`, `merge_seconds`, `interval_rule` | candidate-interval rule and parameters |
| `points[].cutoff_seconds` | x axis (displayed as m:ss) |
| `points[].<probability_key>` | y axis, 0..1 (Terran for legacy, current POV's player otherwise) |
| `points[].frame_seconds` | legacy optional sampled-frame list; the preview uses `explanations[].sampling` / `context_frame` instead |
| `pivotal_intervals[].start_seconds` / `end_seconds` | candidate band on the chart |
| `pivotal_intervals[].net_change` / `peak_change` | change in **percentage points** |
| `pivotal_intervals[].max_step_change` / `max_step_cutoff_seconds` | largest single adjacent change in the candidate |
| `pivotal_intervals[].changes` | number of consecutive qualified adjacent changes |
| `pivotal_intervals[].peak_cutoff_seconds` | fallback cutoff for selecting a heatmap |
| `explanations[].cutoff_seconds` | matched against the candidates and heatmaps |
| `explanations[].displayed_frame_seconds` | image observation time, when recorded; never substituted with the cutoff |
| `explanations[].target_class` | `1` = evidence for the positive class, `0` = negative |
| `explanations[].target_meaning` | plain-language class meaning, preferred over inferred race text |
| `explanations[].image` | heatmap file name, relative to `review.json` |
| `explanations[].generated_on_demand`, `generation` | on-demand cache identity and probability match |
| `explanations[].sampling` | actual frame indices, seconds, game loops, range, last sampled frame, source |
| `explanations[].context_image`, `context_frame`, `context_legend` | model-input minimap reference, its time and colors |
| `explanations[].attribution` | raw Grad-CAM max, temporal bin, spatial cell, ties, axis/normalization notes |
| `explanations[].game_view` | optional cached Re-rendered game view, actual loop/time and unique image |
| `explanations[].masking_check` | collapsible technical diagnostic only |
| `evaluation_status` | membership wording; unknown is shown as **unknown (not recorded)** |
| `limitations` | full original text inside collapsed Technical details; the page shows one short warning |
| `demo: true` | shows the demo banner (the path containing `/demo/` also triggers it) |

Notes:

- `threshold: 0.1` is an absolute probability change of **10.0 percentage points**.
  The UI never prints it as a percentage change.
- `merge_seconds` now means the maximum merged span, not a gap between events. New
  exports record `interval_rule`; legacy exports keep their stored intervals and are
  labelled as legacy.
- Newer exports describe **focus-player** probabilities (class 1 = the observed player
  wins). The chart title, readout, detail panel, result line and heatmap caption use
  `probability_subject` / `probability_subject_detail`, so a Protoss POV is never labelled
  "Terran" by mistake. Legacy exports without those fields keep the Terran wording.
- `cutoff_seconds` and `displayed_frame_seconds` are independent fields. They may be
  equal or different; both recorded values are displayed as they are. A field that the
  export does not contain is shown as **not recorded**, never as `0`.
- `points` may contain a single element. The chart then shows one dot and its readout.
- With no candidate intervals, the page selects the **last prediction point**, which is
  also the cutoff a real export explains in that case.

## Rendering notes

- The curve is an SVG drawn at the width of its own container (`viewBox` matches the
  CSS width), so labels stay at a 1:1 pixel scale; it is redrawn on window resize.
  X-axis ticks use readable `m:ss` labels at a coarse step.
- Only `points` and candidate `pivotal_intervals` are plotted. The `timeline.png` written
  by `analysis/review.py` is not used.
- Technical details are collapsible and show the full match hash, checkpoint hash/epoch,
  sampling values, time grids and interval-rule parameters. Action records are a short
  note inside that panel; the masking diagnostic is a collapsible technical subsection.
  Grad-CAM status, duplicate time/file fields and explanatory captions live in the
  default-closed **Grad-CAM details and interpretation** panel; **Attribution summary** is
  its own default-closed panel.
- The context minimap is a 512×512 nearest-neighbour rendering of the saved
  `player_relative` and `visibility_map` arrays, not a screenshot. The peak outline uses
  the same 4×4 row/column order as the raw array and is only a coarse reference.
- Minimap, Grad-CAM and RGB images open in a modal viewer. Wheel/buttons zoom, dragging
  pans, Reset restores the view, Esc closes and focus returns to the trigger.
- Print CSS hides the tabs, moment navigation, the time slider and on-demand actions, then
  prints the visible panes in order (the currently browsed replay minimap, Game view,
  Grad-CAM) without starting any generation. Figures and the chart stay whole, while text
  may flow across pages so a
  small technical panel no longer forces its own page. Legend swatches use
  `print-color-adjust: exact` plus a dark border, so Self, Enemy, Neutral and the
  visibility swatches remain distinguishable even when the browser's optional
  “background graphics” setting is off.
- Desktop uses a two-column grid (preview beside suggested moments, timeline below).
  At 900 px and narrower the cards stack in reading order: Timeline → Suggested moments →
  Preview → Technical details, with no horizontal scrolling.

## Empty and missing data

| Situation | Behaviour |
| --- | --- |
| No query parameter | upload entry with the support rules from `/api/support` |
| `review` parameter missing/invalid | legacy help panel with the expected URL format |
| `?job=` unknown or malformed | job error with a link back to the upload entry |
| Backend unreachable | upload controls disabled, support error shown; legacy `?review=` still works |
| `file://` page | explains that a local server is required |
| `review.json` missing / HTTP error / invalid JSON | error banner plus the requested URL |
| Report requested before the job completed | status polling resumes instead of showing a partial report |
| POV has no review file | report error naming the POV; POV buttons stay available |
| `points` empty or absent | "no prediction points" message, chart hidden |
| `points` has one element | dot, readout and detail panel, no polyline |
| No candidate interval | explanation plus the last point selected |
| No explanation for the selected cutoff | "No heatmap for this cutoff", previous image removed, listing which cutoffs have one |
| Explanation has no saved sampling/context | context minimap says not recorded; game-view button is hidden |
| Context image missing (`onerror`) | context panel reports the failed URL; heatmap remains visible |
| `sampling.source` recorded/reconstructed | shown as "recorded at export" or "reconstructed from saved observations" |
| No positive Grad-CAM values | Attribution summary says "No positive Grad-CAM evidence"; no peak outline |
| Cached game view missing (`onerror`) | error shown and Render game view offered again |
| Heatmap file absent (`onerror`) | "Heatmap image could not be loaded (no data available)" with the URL |
| `masking_check.available === false` | reason shown **in the diagnostic area only**; heatmap and probabilities stay visible |
| Action records | short "not available" note inside technical details |

## Not implemented

- No charting library, no upload library and no build pipeline; the curve is a hand-built SVG.
- No queue, cancel button, upload progress percentage or resumed analysis; the backend has
  no such endpoints, and only one analysis runs at a time.
- No video player, tactical advice, cross-match comparison or calibration display.
- The `.npz` heatmap arrays and `timeline.png` are ignored; only the PNG is displayed.
- The demo generator is not part of the repository; the committed demo files are
  static fixtures.
