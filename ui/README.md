# Replay review UI

A dependency-free HTML/CSS/JavaScript viewer for the `review.json` files written by
`analysis/review.py`. No build step, no npm packages, no Python packages.

It reads a previously exported review directory and shows:

- the match ID, player POV, actual result and observation setting;
- a Terran win-probability curve over elapsed time, with the actual result marked;
- the detected pivotal intervals, selectable by click or keyboard;
- for a selected cutoff, the matching Grad-CAM heatmap, with the **prediction cutoff**
  and the **image observation time** labelled separately;
- which side the heatmap supports (`target_class`), and the masking diagnostic when
  the export contains one.

## Run it

Start a static server in the **project root** (any Python 3 works; nothing is imported):

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

The page must be served over HTTP. Opening `ui/index.html` directly from disk
(`file://`) cannot work, because browsers block `fetch()` of `review.json` on
`file://` pages. The page shows an explicit message in that case.

## URL format

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
| `label` | actual result, `1` = Terran win, `0` = Protoss win |
| `observation_setting`, `checkpoint`, `checkpoint_epoch` | run information and provenance |
| `time_grid.training` / `time_grid.inference` | run information panel; training vs export cutoffs |
| `time_grid.override_applied` | shows the "cutoffs outside the training grid are unvalidated" warning |
| `time_grid.frames` | frames per prediction |
| `threshold`, `merge_seconds` | explanation of how intervals were detected |
| `points[].cutoff_seconds` | x axis (converted to minutes for display only) |
| `points[].terran_win_probability` | y axis, 0..1 |
| `points[].frame_seconds` | observed frame window of that prediction |
| `pivotal_intervals[].start_seconds` / `end_seconds` | shaded band on the chart |
| `pivotal_intervals[].net_change` / `peak_change` | change shown in **percentage points** |
| `pivotal_intervals[].peak_cutoff_seconds` | the cutoff that selects the heatmap |
| `explanations[].cutoff_seconds` | matched against `peak_cutoff_seconds` |
| `explanations[].displayed_frame_seconds` | image observation time |
| `explanations[].target_class` | "evidence supporting Terran" (1) or "Protoss" (0) |
| `explanations[].image` | heatmap file name, relative to `review.json` |
| `explanations[].masking_check` | diagnostic panel only |
| `limitations` | footer disclaimer |
| `demo: true` | shows the demo banner (the path containing `/demo/` also triggers it) |

Notes:

- `threshold: 0.1` is an absolute probability change of **10.0 percentage points**.
  The UI never prints it as a percentage change.
- `cutoff_seconds` and `displayed_frame_seconds` are independent fields. They may be
  equal or different; both recorded values are displayed as they are.
- `points` may contain a single element. The chart then shows one dot and its readout.
- With no pivotal intervals, the page selects the **last prediction point**, which is
  also the cutoff a real export explains in that case.

## Rendering notes

- The curve is an SVG drawn at the width of its own container (`viewBox` matches the
  CSS width), so labels stay at a 1:1 pixel scale; it is redrawn on window resize.
- Only `points` and `pivotal_intervals` are plotted. The `timeline.png` written by
  `analysis/review.py` is not used.
- The **Run information** panel is built from `checkpoint_epoch` and `time_grid`. An export
  that predates `time_grid` says so explicitly and repeats the `limitations` text instead of
  guessing a grid.

## Empty and missing data

| Situation | Behaviour |
| --- | --- |
| No `review` parameter | help panel with the expected URL format |
| `file://` page | explains that a local server is required |
| `review.json` missing / HTTP error / invalid JSON | error banner plus the requested URL |
| `points` empty or absent | "no prediction points" message, chart hidden |
| `points` has one element | dot, readout and detail panel, no polyline |
| No pivotal intervals | explanation plus the last point selected |
| No explanation for the selected cutoff | "No heatmap for this cutoff", listing which cutoffs have one |
| Heatmap file absent (`onerror`) | "Heatmap image could not be loaded (no data available)" with the URL |
| `masking_check.available === false` | reason shown **in the diagnostic area only**; heatmap and probabilities stay visible |
| Action records | always "No action records available": current exports contain no action history |

## Not implemented

- No charting library and no build pipeline; the curve is a hand-built SVG.
- The `.npz` heatmap arrays and `timeline.png` are ignored; only the PNG is displayed.
- No comparison across matches, no calibration display, no action annotations.
- The demo generator is not part of the repository; the committed demo files are
  static fixtures.
