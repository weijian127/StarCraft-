# SC2 predictor

Install the Python dependencies with `uv sync`.

For an NVIDIA GPU, use `uv sync --extra cuda` (CUDA 12.8 PyTorch wheels). This
project's local environment was verified on an RTX 5090 Laptop GPU. Keep
`--extra cuda` on subsequent `uv run` commands so uv retains the CUDA build.
No separate CUDA toolkit or SC2 installation is needed to train from arrays.

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

## Dataset loader

`data_processing.dataset` adapts `SC2ReplayDataset` and `replay_collate_fn` from
the [reference implementation](https://github.com/InsungBaek/StarCraft2/blob/master/utils_sc/data.py).
It reads our schema-v2 manifests and needs no running SC2 engine. Install the
locked Python 3.10 environment with `uv sync` (PyTorch and development tests are
included). Inspect a batch of the checked-in data with:

```powershell
uv run python -m data_processing.dataset --pov both --split train
uv run python -m data_processing.dataset --pov both --split test --num-workers 2
uv run python -m pytest
```

Use the dataset's configured collator so its short-recording filter and frame
sampling always use the same window:

```python
import torch
from torch.utils.data import DataLoader
from data_processing.dataset import SC2ReplayDataset

if __name__ == "__main__":  # Required for worker processes on Windows.
    torch.manual_seed(42)
    train_set = SC2ReplayDataset(pov="both", split="train", seed=42)
    test_set = SC2ReplayDataset(pov="both", split="test", seed=42)
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=train_set.collate_fn,
        generator=torch.Generator().manual_seed(42),
    )
    batch = next(iter(train_loader))
    inputs, targets = batch["inputs"], batch["targets"]
```

The default split reserves 20% of unique matches for testing, rounded to the
nearest whole match, with at least one match in each split. Both player POVs
stay together. Use the same manifest, `seed`, and `test_fraction` for train and
test; assignments occur before filtering short samples. This is a random,
unstratified split. Changing the manifest's set of matches can change assignments;
freeze the manifest for an experiment. Across POV modes, first use the same
intersection of match IDs. `split="all"` loads all eligible samples.

Like the reference collator, defaults select 50 distinct, chronologically sorted
frames with logarithmic weighting from the 10%-80% portion of each replay.
`start_prob` is the starting fraction and `length_prob` is the exclusive ending
fraction. Set them to `0.0` and `1.0` for the whole replay. `weighting="exp"` and
`"uniform"` are also supported. Samples with too few frames in the requested
window are excluded and counted in `dataset.skipped_samples`; an empty selection
raises an error. Exact-fit windows are accepted.

Each batch has `inputs` mapping feature names to **int64 [B, T, H, W]** tensors
and `targets` containing **int64 [B]** class indices, still 1=Terran win and
0=Protoss win. Feature order follows `include`, defaulting to visibility,
ownership, then unit type. One-frame batches retain the time dimension; squeeze
it explicitly for a 2D model. For a sigmoid/BCE head, convert targets to float.
Unit IDs are preserved, including IDs absent from older PySC2 vocabularies;
the model must provide a matching embedding vocabulary or an explicit ID mapping.
`dataset.samples` retains sample IDs, match IDs, and timing paths for inspection.

Two reference edge cases are corrected: logarithmic weights use `log(1+t+1)`
so frame zero can be selected, and exponential sampling operates in log space
to avoid overflow. Sampling uses PyTorch's RNG, including automatic worker
seeding. Reset `torch.manual_seed` for repeatable runs without workers, or use
a freshly seeded DataLoader generator with workers. Repeated evaluation without
resetting the RNG samples new frames; fix the seed/loader for comparable metrics.
These fraction-of-replay windows reproduce the baseline convention; a future
elapsed-time prediction loader must restrict observations to its cutoff.

## Baek model and training

`models.baek.BaekResNet3D` implements the **embedded-Gaussian non-local block
after residual block 3**, the best variant in Tables 3 and 4 of
[Baek et al., PLOS ONE (2022)](https://doi.org/10.1371/journal.pone.0264550).
Only that variant is included. The implementation follows
[InsungBaek/StarCraft2 at 664b17a](https://github.com/InsungBaek/StarCraft2/tree/664b17a),
specifically `ResNet3D_non_local`, `BasicBlock3D`, `CategoricalEmbedding`, and
`layers/non_local_embedded_gaussian.py`.

The three embeddings have widths 10, 10, and 100. Their 120 channels enter a
7x7x7 3D convolution, followed by residual stages with 64/128/256 channels,
the non-local block, global average pooling, dropout, and two output logits.
As in the reference, non-local attention uses **unscaled** embedded dot products,
softmax, spatially subsampled keys/values, and an initially zero output BN.
The output labels remain **0=Protoss wins, 1=Terran wins**, independent of POV.

Important implementation choices:

- Residual convolutions keep stride 1, matching what the reference actually
  executes even when its constructor receives `stride=2`. This is the author's
  custom 3D network, not a generic torchvision ResNet-18.
- A fixed 65,536-row unit embedding supports our raw uint16 IDs without clipping
  or consulting PySC2. This increases the parameter count to **11,532,508**.
- Final adaptive average pooling equals the reference's `(5,4,4)` average pool
  for 50x128x128 inputs. Shorter diagnostic clips are supported with at least
  nine frames and 64x64 spatial dimensions; they are not the paper setting.
- Embeddings are trainable, matching the released code. The paper says they are
  fixed; `--freeze-embeddings` selects that interpretation. Padding ID 0 stays zero.
- Adam at 0.001 and two-logit cross-entropy follow the reference code. The paper
  describes binary cross-entropy; two-class softmax CE expresses the same binary
  classification likelihood. Default batch size is 2 for laptop memory (paper: 4).
- The loader's default 10%-80% window is retained. For full-game sampling, pass
  `--start-prob 0 --length-prob 1`. Full-game results are retrospective predictions.

Run from the project root with the existing arrays:

```powershell
uv run --extra cuda python -m training.train --epochs 5 --device cuda
```

On this laptop uv is also available at `.tools/uv/bin/uv.exe`. After syncing,
the environment can be used directly, without running uv again:

```powershell
.\.venv\Scripts\python.exe -m training.train --epochs 5 --device cuda
```

CPU-only machines can use `uv run python -m training.train --device cpu`.
The full-resolution 3D model is expensive on CPU; `--max-timesteps 10
--max-train-batches 1 --max-val-batches 1 --epochs 1` provides a shorter smoke test.
`--no-amp` disables the default CUDA mixed precision. `--num-workers` defaults to
zero, avoiding extra full-array copies from Windows worker processes.

Training defaults to a **stratified split of whole matches**. For our sample set
this gives nine training matches (18 POVs) and two validation matches (four POVs),
with both outcomes in each split. This is opt-in via `stratify=True` when using
the dataset directly; its earlier random-split default stays unchanged.
The trainer uses fixed validation frame samples each epoch, fresh training
frame samples, and sample-weighted loss averages. `--no-stratify` selects the
original grouped random split. Saved validation metrics are per POV, not per match.

Each invocation creates a new ignored `runs/<timestamp>/` directory. An explicit
`--output runs/my-run` must also be new. It contains:

- `config.json`: settings, device, labels, and exact match/sample split IDs.
- `history.json`: epoch losses, accuracies, sample counts and confusion matrices
  (rows=true labels, columns=predictions), training/validation seconds, and CUDA
  peak allocated/reserved bytes. Timings include data loading and GPU work,
  synchronize CUDA at phase boundaries, and exclude setup/checkpoint writing.
  Memory peaks are per epoch for PyTorch's allocator; driver/context allocations
  and other applications are excluded. Reserved memory includes allocator caching.
- `last.pt`: latest model, optimizer and AMP scaler state.
- `best.pt`: the checkpoint with the lowest validation loss.

Checkpoints include the model configuration and label convention. Load one for
inference as follows (the loader returns a model in evaluation mode):

```python
import torch
from training.train import load_model

model = load_model("runs/my-run/best.pt", device="cuda")
# batch is an inputs/targets dictionary produced by dataset.collate_fn.
with torch.no_grad():
    logits = model({name: tensor.cuda() for name, tensor in batch["inputs"].items()})
    terran_win_probability = logits.softmax(dim=-1)[:, 1]
```

Training restart/resume and Grad-CAM are not implemented in this initial loop.
The 11-match set is for pipeline verification; its validation scores cannot
establish reproduction of the paper's reported predictive performance.

Run checks with `uv run --extra cuda python -m pytest -q` (omit the extra on CPU).
