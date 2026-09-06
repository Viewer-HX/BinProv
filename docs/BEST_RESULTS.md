# Best results

Headline BinProv results, their exact archived run
membership, and how to replay them. Machine-readable single source of truth:
[../configs/best_results.json](../configs/best_results.json). On 2026-09-05, the
five recipes with exact membership were independently recomputed from the supplied
probabilities and canonical corpus; see
[verified tables](../reports/tables/best/README.md). No model was retrained. Per-run accuracies are not
copied into the manifest or this page — they are read at runtime from each run's
archived `result.json`, so the driver displays the source measurements alongside each training command.

## Supported environment and corpus

| | |
|---|---|
| Training environment | one NVIDIA A100 80 GB, bf16 autocast |
| Verified full-pipeline time | 12:42:09 without restart |
| Recorded versions | python 3.12.11, torch 2.8.0+cu128, transformers 4.55.4, numpy 2.3.3 |
| Corpus | BinKit x86_64, program-grouped split (47 test / 28 val programs) |
| O2/O3 test set | 55,657 windows |
| opt4 test set | 116,321 windows |

## Unit vocabulary
| Term | Meaning |
|---|---|
| **stride 512 B** | evaluation unit is the paper's non-overlapping 512-byte window |
| **encoder input 2048 B** | wide encoder reads 2048 B per forward pass; stride stays 512 B (overlapping windows) |
| **nominal aperture 16.9 KB** | mean of a window and its 16 neighbours either side (radius 16); wide input context extends beyond those targets |
| **binary vote** | majority vote over all windows of a binary |

## Headline results
| Result | Score | Group |
|---|---|---|
| O2/O3, 7-wide ensemble @512 B (sequence level) | **75.10%** | `o2o3_7wide_512B` |
| Same 7 runs, 16.9 KB aperture (sequence level) | **81.10%** | `o2o3_7wide_16.9KB` |
| 2048-B encoder alone on O2/O3 — mean of **3** seeds | **71.98%** (SD 0.68 pp) | `o2o3_pretrained_wide_mean` |
| opt4, 6-run ensemble, 16.9 KB aperture (sequence level) | **87.62%** | `opt4_6run_16.9KB` |
| opt4, binary-level vote over the 3 wide runs | **95.21%** | `opt4_3wide_binary` |
| Archived 19-run O2/O3 binary vote | 93.09% | `o2o3_19run_binary` (no recipe) |

## Fresh release-model replication

On 2026-09-05/06, the repository's `opt4_wide_seed29` recipe was trained from
scratch through both MLM stages on one A100 80 GB. The selected epoch scored
**84.2255% sequence accuracy** (83.0951% balanced) and **93.8830% binary
soft-vote accuracy** on 116,321 windows from 376 binaries. This is 0.3755
percentage points above the archived seed-29 sequence score (83.85%); the
binary result reproduces the archived 93.88% to rounding.

The exact args, epoch history, confusion matrix, independent evaluation, and
pipeline timestamps are committed under
[`reports/runs/release/opt4_wide_seed29`](../reports/runs/release/opt4_wide_seed29/README.md).
The upload-ready 335 MiB model is generated under the gitignored
`results/gpu_release/opt4_wide_seed29/export/`. Hardware requirements are in
the repository README.

## O2/O3 — 75.10% and 81.10% (same 7 runs)
Both scores are the **same 7-run ensemble**; only the inference aperture changes.
All 7 are the 2048-byte encoder at stride 512. Exact membership:
`r4_ctx2048_dense` + `r6_dense2048_seed{7,13,29}` + `r7_wide2048_seed{7,13,29}`.

- `r4_ctx2048_dense` (seed 1234) and `r6_dense2048_seed{7,13,29}` init from the
  512-byte MLM (tiled positions); `r7_wide2048_seed{7,13,29}` init from the
  continued 2048-byte MLM.
- Archived args per run (no per-run accuracies claimed here):
  [`r4`](../reports/runs/best/r4_ctx2048_dense/args.json) ·
  [`r6_7`](../reports/runs/best/r6_dense2048_seed7/args.json) ·
  [`r6_13`](../reports/runs/best/r6_dense2048_seed13/args.json) ·
  [`r6_29`](../reports/runs/best/r6_dense2048_seed29/args.json) ·
  [`r7_7`](../reports/runs/best/r7_wide2048_seed7/args.json) ·
  [`r7_13`](../reports/runs/best/r7_wide2048_seed13/args.json) ·
  [`r7_29`](../reports/runs/best/r7_wide2048_seed29/args.json)
- Offline combination (equivalently the runner's `offline` phase):
  512 B → `scripts/combine.py <7 runs> --ensemble --context 0 --binary-vote`;
  16.9 KB → `... --context 16 --binary-vote`.

## O2/O3 — 2048-B encoder alone, 71.98% (mean of THREE seeds)
| run | archived args |
|---|---|
| `r7_wide2048_seed7` | [`args.json`](../reports/runs/best/r7_wide2048_seed7/args.json) |
| `r7_wide2048_seed13` | [`args.json`](../reports/runs/best/r7_wide2048_seed13/args.json) |
| `r7_wide2048_seed29` | [`args.json`](../reports/runs/best/r7_wide2048_seed29/args.json) |
| **mean** | **71.98%** (sd 0.68) |

**Three seeds, not four.** The earlier "4-run" reading was wrong: `r4` (seed 1234) and
`r6` (seeds 7/13/29) start from the 512-byte MLM, not from `mlm2048`, so they do not
measure the *pretrained wide* encoder and are excluded. These three runs are the
same `r7` runs inside the 7-wide ensemble above.

## opt4 — 87.62% and 95.21%
`opt4_6run_16.9KB` = 3 narrow (512-B) + 3 wide (2048-B) runs ensembled at the
16.9 KB aperture → **87.62%** sequence level:
`r9_opt4_base_seed{7,13,29}` + `r9_opt4_wide_seed{7,13,29}`.
- Archived args:
  [`base7`](../reports/runs/best/r9_opt4_base_seed7/args.json) ·
  [`base13`](../reports/runs/best/r9_opt4_base_seed13/args.json) ·
  [`base29`](../reports/runs/best/r9_opt4_base_seed29/args.json) ·
  [`wide7`](../reports/runs/best/r9_opt4_wide_seed7/args.json) ·
  [`wide13`](../reports/runs/best/r9_opt4_wide_seed13/args.json) ·
  [`wide29`](../reports/runs/best/r9_opt4_wide_seed29/args.json)
- `opt4_3wide_binary` → **95.21%** is a binary-level majority vote over the **3
  wide runs only** (`r9_opt4_wide_seed{7,13,29}`), not all six.

## Archived 19-run binary vote — 93.09% (no recipe)
Archived, **not reproducible by this runner**. The exact 19-member list is not
reconstructable from the archived `args.json` files (it was a post-hoc superset
of O2/O3 exploration runs), and recomputing it needs the archived `probs.npz`
set, **which is not in this repo**. Probability files alone are not sufficient to
recover membership. Do not invent the list or claim 93.09% as a runner output.

## Replay guide
Stable CLI, `scripts/run_best.py`. It reads the manifest and each run's archived
`args.json`; **dry-run by default**, `--execute` to run. It never downloads data.

```bash
python scripts/run_best.py --list
python scripts/run_best.py --group o2o3_7wide_512B                 # print plan
python scripts/run_best.py --group o2o3_7wide_16.9KB --execute     # best sequence score
python scripts/run_best.py --group opt4_6run_16.9KB --phase offline --archive-probs --execute
```

| Phase | What it does |
|---|---|
| `prepare` | checks the existing corpus + split are present (fetch/build commands in README must run first) |
| `pretrain` | MLMs → `results/best/ckpt/mlm512`, continued → `results/best/ckpt/mlm2048` |
| `train` | fine-tunes the group's runs from archived args |
| `offline` | resolves probabilities and computes the ensemble/mean score |
| `all` (default) | prepare → pretrain → train → offline |

- `--phase` defaults to `all`; dry-run is the default (`--dry-run` accepted).
- `offline` reads fresh fine-tune outputs under `results/best/finetune` by
  default; `--archive-probs` resolves probabilities from `results/explore` via
  the committed report metadata instead.
- All generated output is under `results/best`; nonempty training output
  directories are not overwritten. Start a new run from fresh directories, or
  execute a later phase using completed prerequisites.
- For archive recomputation, first extract `BinProv-probs-20260903.tar` from the
  repo root. It creates `results/explore/*/probs.npz`; the runner stages those
  files with matching metadata into `results/best/archive_inputs`. The original
  packed corpus is also required. The probability archive is distributed separately and has now been verified;
  its SHA-256 is `aeb1de809b6e1ad36145633b14073a483b5c9d078047758c2293b470125e4434`.
- `o2o3_19run_binary` is listed but archived/no-recipe: the runner rejects it because the exact ensemble membership is missing.

## Caveats
- **Compare distributions, not single runs.** Replicas of one configuration
  differ by ~1–2 points with seed alone; effects were established over multiple
  seeds. A different GPU/CUDA/library version is a plausible source of a similar
  shift.
- The two test sets (O2/O3 vs opt4) are not comparable by subtraction.
- The single wide seed-29 release recipe has been freshly trained and verified
  as described above. The multi-seed ensembles remain offline reproductions
  from archived probabilities rather than newly trained replicas.

## Cross-reference
- Rebuild context: [../README.md](../README.md)
- Paper reproduction: [RESULTS.md](RESULTS.md)
- Machine-readable manifest: [../configs/best_results.json](../configs/best_results.json)
- Curated best-run args/results: `../reports/runs/best/*/`
