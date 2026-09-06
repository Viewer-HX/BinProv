# Local training and Hugging Face release preparation

This is the *bounded, on-one-Mac* path to (a) train a BinProv opt4 model locally
on Apple Silicon, (b) evaluate it from the saved probabilities, and (c) package
a self-contained, upload-ready directory for a Hugging Face release — without
uploading, downloading, or touching the historical `checkpoints/`,
`reports/`/`results/explore` paths.

Everything is driven from two files:

| file | role |
|---|---|
| `configs/local_release.json` | the two profiles and their exact effective-batch recipes |
| `scripts/train_local_release.py` | the driver (dry-run default) |
| `scripts/experiment.py --grad-accum` | the training loop change that makes big effective batches fit MPS |
| `scripts/export_hf.py` | builds the upload-ready directory |
| `scripts/inference_template.py` | the minimal inference script copied into the export |

All generated output lands under `results/local_release/<profile>/` — never on
the archived paths the paper numbers were produced from.

## Hardware benchmark (measured)

Measured on an **Apple M4 Max, 48 GB unified memory**, `torch 2.10.0`, bf16
autocast on MPS (`engine.pick_device`). Footprints are the MPS peak
`torch.mps.current_allocated_memory` reached after **two optimizer steps** of the
real 12-layer / 768-hidden model (warmup steady state), for the two shapes these
profiles use:

| model input | micro-batch | peak footprint |
|---|---|---|
| 512 bytes (narrow) | 16 | **~18.7 GB** |
| 2048 bytes (wide) | 1 | **~12.4 GB** |

These are the reason the profiles use micro-batches 16 (512-byte) and 2
(2048-byte) with `workers 0`: at those sizes a full model plus an 18.7 GB peak
fits comfortably inside the 48 GB MPS budget, with room left for the OS, the
DataLoader and the ~1 GB per-epoch resume snapshots. `workers 0` keeps the fork
overhead and CPU contention out of the timing-critical loop.

## Duration estimates (rough)

**Do not read these as commitments.** The verified wide release pipeline took
12:42:09 on one A100 80 GB on GMU Hopper. An M-series Mac is a different and
much slower device, and the *full* local runs below have **not been executed**
in this repository state.

| profile | what it trains | rough wall-clock on one M4 Max |
|---|---|---|
| `opt4_narrow_seed13` | 512-byte MLM (10 epochs, eff. batch 256) + opt4 fine-tune seed 13 (4 epochs, eff. batch 128) | **2–4 days** (plugged in, unattended) |
| `opt4_wide_seed29` | same MLM512 + continued 2048-byte MLM (10 epochs, eff. batch 64) + opt4 fine-tune at 2048 B seed 29 (6 epochs, eff. batch 32) | **2 weeks+, likely more** |

Estimate basis: the archived per-step counts scaled by the measured MPS
throughput for the same shapes and by the accumulated effective-batch counts in
`configs/local_release.json`. Treat the wide profile as *optional* and the time
range as ±50%. **Keep the machine on AC power**: the runner wraps every training
command in `caffeinate -i -s` on macOS (see [Resuming and safety](#resuming-and-safety))
but caffeinate cannot save a machine running off battery.

## The two profiles

`configs/local_release.json` encodes every training stage as an **archived
recipe re-expressed** for MPS: micro-batch × `--grad-accum` reproduces the
archived *effective* batch exactly, workers are 0, and everything else (seed,
LR, warmup, epochs, task, split, val-seed, tta) is byte-for-byte the archived
`args.json`.

### `opt4_narrow_seed13` — recommended

Replays `reports/runs/best/r9_opt4_base_seed13` (task `opt4`, seed 13,
512-byte encoder):

| stage | archived | local recipe (effective = archived) |
|---|---|---|
| pretrain512 | `reports/runs/best/mlm512/pretrain_args.json` batch 256 | micro-batch 16 × accum 16 = **256**, workers 0, split `train`, 10 epochs, resume + save every epoch |
| finetune | `r9_opt4_base_seed13` batch 128 | micro-batch 16 × accum 8 = **128**, workers 0, seed 13 |

The archived run scored **81.11% sequence-level accuracy** (soft binary vote
92.82%) on the 116,321-window test set. A fresh run **will not land exactly on
81.11%**: the seed is pinned and the val partition is pinned (`--val-seed 1234`),
but torch/transformers version and the MPS vs Hopper CUDA backend still move
the number ~1–2 points. Treat 81% as the target, not the guarantee.

### `opt4_wide_seed29` — optional

Replays the 2048-byte round-7 path and `reports/runs/best/r9_opt4_wide_seed29`:

| stage | archived | local recipe (effective = archived) |
|---|---|---|
| pretrain512 | same MLM512 archive, batch 256 | micro-batch 16 × accum 16 = **256**, workers 0, 10 epochs, resume + save every epoch |
| pretrain2048 | round-7 continued MLM (best_results.json), batch 64 | micro-batch 2 × accum 32 = **64**, workers 0, init `mlm512`, 10 epochs, resume + save every epoch |
| finetune | `r9_opt4_wide_seed29` batch 32 | micro-batch 2 × accum 16 = **32**, workers 0, seed 29, seq 2048 |

The archived wide run scored **83.85% sequence** (soft binary vote 93.88%). Same
caveat: fresh results vary. This profile costs ~an order of magnitude more than
the narrow one; it exists to reproduce the 2048-byte encoder, not for a first
local run.

## `--grad-accum` in `experiment.py` (what changed)

`scripts/experiment.py` now accepts `--grad-accum N` (default 1). The semantics,
and why they are *mathematically* the archived ones:

- each micro-batch's **combined** loss (CE + any pair/margin terms) is divided by
  `N` before `backward()`, so `N` accumulated micro-batches produce exactly the
  mean gradient over one effective batch of `N × batch-size`;
- `optimizer.step()`, the LR `scheduler.step()` and gradient clipping happen once
  per effective batch (`N` micro-batches), never per micro-batch;
- `total_steps` counts **optimizer steps** (`steps_per_epoch × epochs`), so the
  warmup/cosine schedule still describes real updates;
- a trailing **partial group** at the end of an epoch (fewer than `N` micro-batches
  left) still performs one step, so its gradients cannot bleed into the next
  epoch — each epoch is an independent, correctly scaled set of updates;
- `--grad-accum 1` reproduces the archived loop exactly (step every micro-batch).

Validation is untouched. `--grad-accum` is saved into `args.json` like every other
flag. The helper `accum_groups`/`accum_steps_per_epoch` in `experiment.py` is the
single source of truth for the grouping; the regression tests in
`tests/test_grad_accum.py` prove the optimizer-step count *and* the loss scaling
on a tiny real model.

`experiment.py` also gained `--resume` (off by default): after every epoch it
snapshots model + optimizer + scheduler + RNG + the val history to
`training_state.pt` and persists the best-val weights as they are found, so a
resumed run still evaluates the val-selected epoch. The local runner always
passes `--resume`.

## Running it

```bash
# see the two profiles and their phase chains
python scripts/train_local_release.py --list

# recommended profile, dry run (default: prints the whole plan, mutates nothing)
python scripts/train_local_release.py --profile opt4_narrow_seed13

# actually train (pretrain512 -> finetune -> evaluate -> export), sequentially
python scripts/train_local_release.py --profile opt4_narrow_seed13 --execute

# continue an interrupted run (see below)
python scripts/train_local_release.py --profile opt4_narrow_seed13 --resume --execute

# a single phase, e.g. only the fine-tune
python scripts/train_local_release.py --profile opt4_narrow_seed13 \
    --phase finetune --execute

# evaluate from saved probabilities, or build the export only
python scripts/train_local_release.py --profile opt4_narrow_seed13 --phase evaluate --execute
python scripts/train_local_release.py --profile opt4_narrow_seed13 --phase export --execute
```

`--phase` selects `pretrain512 | pretrain2048 | finetune | evaluate | export |
all`; the wide profile owns `pretrain2048`, the narrow one does not. The runner
runs the training stages **sequentially through the existing scripts**
(`pretrain_mlm.py`, then `experiment.py`) as subprocesses — never `shell=True` —
teeing stdout to `results/local_release/<profile>/logs/<phase>.log` while
printing it. `results/local_release/status.json` is rewritten atomically after
each stage.

Before doing anything the runner verifies: the corpus exists and the **canonical
program split** matches `reports/tables/split_programs.txt` (47 test / 28
pinned-validation programs), the requested backend (`mps`) is actually
available, the machine is on **AC power** (warning/block), and there is enough
free disk (`disk_reserve_gb`). Dry runs report all of this without running or
writing.

### Evaluation and export

- **evaluate** does **not** re-run the model: it reads the exact
  `probs.npz`/`result.json` the finetune wrote and reports sequence accuracy
  (cross-checked against the recorded `result.json`) and binary-level soft-vote
  accuracy. Output: `results/local_release/<profile>/evaluate/report.json`.
- **export** runs `scripts/export_hf.py` on the completed finetune directory and
  builds `results/local_release/<profile>/export/` — weights, configs, label map,
  `metadata.json` (exact `experiment.py` args + eval results), `environment.json`,
  `LICENSE`, `README.md`/`MODEL_CARD.md`, bundled `binprov/` runtime,
  `requirements.txt`, and `evidence/` (args, result, evaluation + probabilities)
  and the standalone `inference.py`. It then **verifies** the export by reloading
  the copied weights and checking a deterministic prediction on small real-corpus
  windows matches the source checkpoint. Nothing is uploaded.

### The released model is not an `AutoModel`

The encoder is a RoBERTa-shaped byte encoder, but the pooling/classifier head is
BinProv's own, so the end-to-end checkpoint is **not** loadable with
`transformers.AutoModel`. The export never claims it is, never uses
`trust_remote_code`, and ships `inference.py` as the supported entry point:

```bash
# usage requires this repository installed (pip install .) for `binprov` + torch
python inference.py --model model --elf /path/to/binary.elf
python inference.py --model model --text-bytes raw_text.bin
```

It extracts `.text` with the repo's dependency-free ELF reader (or takes raw
bytes), cuts the model's windows exactly as the corpus cut does, predicts each,
and soft-votes per class over the windows.

## Resuming and safety

- A stage counts as **complete** only when the real files exist: for an MLM,
  HF weights (`config.json` + `pytorch_model.bin`) + `binprov_config.json` +
  `pretrain_args.json` + `baseline.json` + a `training_state.pt` whose epoch says
  the last requested epoch finished; for the finetune, `model.pt` +
  `binprov_config.json` + `head.json` + `result.json`. Partial files are never
  mistaken for a checkpoint.
- Before launching into an existing stage directory, the runner compares the
  previously saved `args.json`/`pretrain_args.json` against the exact args this
  profile would run. **Matching + interrupted** ⇒ safe resume (needs
  `--resume`). **Mismatched** ⇒ refused: move the directory aside rather than
  risk training that silently starts from the wrong recipe.
- The underlying scripts do the actual continuation from their per-epoch
  `training_state.pt` snapshots (MLM and, with `experiment.py --resume`, the
  fine-tune).
- **`caffeinate`**: on macOS the runner prefixes every command with
  `caffeinate -i -s` so the system does not idle-sleep during an unattended run
  (`-i` prevents idle sleep; `-s` asserts it while on AC). Equivalent manual
  invocation is documented in `scripts/train_local_release.py`. Keep the Mac on
  AC power for anything longer than an hour.

## See also

- Archived numbers behind the profiles: `docs/BEST_RESULTS.md`,
  `reports/runs/best/r9_opt4_base_seed13/result.json`,
  `reports/runs/best/r9_opt4_wide_seed29/result.json`.
- Gradient-accumulation correctness tests: `tests/test_grad_accum.py`.
- Runner/config/export tests: `tests/test_local_release.py`.
