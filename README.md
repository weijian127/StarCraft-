# SC2 predictor

Install the Python dependencies with `uv sync`.

Copy `.env.example` to `.env` and edit the paths for your machine. `.env` stays
out of Git; `.env.example` documents the settings for teammates.

| Setting | Location |
| --- | --- |
| `SC2PATH` | StarCraft II installation folder containing `Versions/` |
| `REPLAY_DIR` | Folder containing raw `.SC2Replay` files |
| `PROCESSED_DIR` | Parent directory for the three POV datasets |

Use forward slashes in `.env` paths, including on Windows. Relative paths start
at the project root. Absolute paths can point to another drive.

Run from the project root:

```powershell
uv run python -m data_processing.processor --pov single --limit 1
uv run python -m data_processing.processor --pov both --limit 1
uv run python -m data_processing.processor --pov fogless --limit 1
```

Command-line arguments override the configured paths:

```powershell
uv run python -m data_processing.processor --output data/another_run --pov both --limit 1
```

Choose a new output directory for each extraction. Existing shell environment
variables take precedence over `.env`. Other Python modules can import paths
from `config.py`.

## Parallel extraction

Use `--workers N` to process up to N matches simultaneously, with one SC2 engine
per worker. The default is one worker. Each worker reuses its engine between
matches; both perspectives stay together when using `--pov both`.

```powershell
uv run python -m data_processing.processor --pov both --workers 6 --output data/parallel_run
```

The main process filters duplicate matches, combines worker results in input order,
and writes one manifest. Workers save their arrays directly to disk. `--limit`
counts matches across the entire run. Fewer workers are started if fewer matches
are available. On interruption or a worker failure, pending tasks are cancelled
where possible and engines close after active work finishes. Use a new output
directory for another run.

## POV modes

| Mode | Observed player | Fog | Samples per accepted match |
| --- | --- | --- | ---: |
| `single` | Player 1 (the lowest participant ID) | Enabled | 1 |
| `both` | Each player separately | Enabled | 2 |
| `fogless` | Player 1 (the lowest participant ID) | Disabled | 1 |

The target is always **1 = Terran wins, 0 = Protoss wins**, regardless of the
observed player. Both views of a decisive match receive the same label.
Fogless mode requests disabled fog at replay start and on every observation.
It retains a focus player for self/enemy ownership encoding.
In the tested SC2 build, the visibility layer still describes that player's normal
sight, while the unit-type and ownership maps reveal units outside it. We retain
that visibility context instead of replacing it with a constant layer.

Each mode has a separate directory and manifest:

```text
data/processed/
    single/manifest.json
    both/manifest.json
    fogless/manifest.json
```

Each manifest has a `samples` list. Sample entries record `replay_id`, `sample_id`,
`focus_player_id`, `focus_player_race`, `pov_mode`, `disable_fog`, the Terran-win
label, and relative feature/timing paths. Sample IDs and directories include the
mode and player ID. Both views retain the same `replay_id`: future train/test
splits must group by that match ID. `--limit` counts matches, not POV samples.
For controlled comparisons, use the intersection of successfully extracted match
IDs across the modes, since playback failures can exclude different matches.

Extraction keeps valid recordings regardless of frame count. Training-specific
requirements, such as a 50-frame sampling window, belong in the dataset loader.

To inspect an export:

```powershell
uv run python -m data_processing.view_output --pov both --player 2 --frame 100
uv run python -m data_processing.gif_generator --pov fogless --player 1
```

GIF and MP4 exports are saved inside the selected sample's directory. MP4 uses
FFmpeg from PATH. Omitting `--pov` in the visualization scripts reads the older
flat exports directly under `PROCESSED_DIR`.
