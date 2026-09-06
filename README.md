# BinProv

BinProv identifies compilation provenance — which compiler and optimization
level — **without disassembling**: it reads the raw bytes of `.text`, embeds them
with a BERT-style byte-level encoder, and classifies. The 261-token vocabulary
(256 bytes + 5 special tokens) covers x86_64, x86_32, ARM and MIPS unchanged.

Paper: "BinProv: Binary Code Provenance Identification without Disassembly", RAID 2022 —
[10.1145/3545948.3545956](https://doi.org/10.1145/3545948.3545956). BibTeX: see [Citation](#citation).
> **Repository note.** The original implementation was lost; this is a rebuild on
> Hugging Face `transformers` (replacing the original fairseq setup). The choices
> needed to reproduce the paper and released model are documented explicitly.

## Pretrained model

The released x86-64 O0/O1/O2/O3 classifier is available at
**[XuViewer/binprov](https://huggingface.co/XuViewer/binprov)**. The model
repository includes weights, the minimal runtime, exact training metadata, and
an inference script for ELF files or extracted `.text` bytes.

```bash
hf download XuViewer/binprov --local-dir binprov-model
cd binprov-model
pip install -r requirements.txt
python inference.py --model model --elf /path/to/binary.elf
```

## Best results first

BinKit x86_64, **program-grouped split** (47 test programs); archived
measurements from one NVIDIA H200 (bf16). Full provenance and replay:
[docs/BEST_RESULTS.md](docs/BEST_RESULTS.md). Verified offline tables: [reports/tables/best/](reports/tables/best/).

**Units.** The encoder *input* is 2048 bytes, but accuracy is scored at the
paper's **non-overlapping 512-byte stride** (the wide encoder sees overlapping
windows). A nominal **16.9 KB aperture** = mean of the window and its 16
neighbours either side (radius 16) at inference, without retraining. This is nominal target-window coverage;
wide inputs also include context beyond each target window.

| Result | Score | Meaning | Group |
|---|---|---|---|
| O2/O3 (hard task), 7-wide ensemble @512 B | **75.10%** | sequence level, 55,657 windows | `o2o3_7wide_512B` |
| Same 7 runs, 16.9 KB aperture | **81.10%** | sequence level, mean over 33 windows | `o2o3_7wide_16.9KB` |
| 2048-B encoder alone on O2/O3 | **71.98%** (SD 0.68 pp) | mean of **three** seeds | `o2o3_pretrained_wide_mean` |
| 4-way O0/O1/O2/O3, 6-run ensemble | **87.62%** | sequence level, 116,321 windows, 16.9 KB aperture | `opt4_6run_16.9KB` |
| opt4, binary-level vote | **95.21%** | vote over the 3 *wide* runs only | `opt4_3wide_binary` |
| Archived 19-run O2/O3 binary vote | 93.09% | exact membership **not** reconstructable | `o2o3_19run_binary` |

**7-wide membership** (as in [configs/best_results.json](configs/best_results.json)):
`r4_ctx2048_dense` + `r6_dense2048_seed{7,13,29}` + `r7_wide2048_seed{7,13,29}`.
All are the 2048-byte encoder at stride 512. The r4/r6 runs init from the 512-byte MLM
(tiled positions); the r7 runs init from the continued 2048-byte MLM.

- **71.98% is three seeds** (`r7_wide2048_seed{7,13,29}`), *not* four: the
  r4 seed 1234 and r6 seeds 7/13/29 start from `mlm512`, not `mlm2048`.
- **93.09% is archived only**: its exact 19-run membership cannot be
  reconstructed from archived args, and recomputing it needs the archived
  `probs.npz` set, which is not in this repo. Probability files alone are not
  sufficient — do not treat it as reachable by the runner.
- Per-run accuracies are not stated here; they are read at runtime from each
  run's archived `result.json` (linked from `docs/BEST_RESULTS.md`).
## Install

```bash
conda create -n binprov python=3.12
conda activate binprov
pip install -r requirements.txt          # for BinKit fetch also: pip install gdown
```

`requirements.txt` bounds: torch ≥2.0, transformers ≥4.35, numpy ≥1.23; corpus
building needs no torch. **Recorded versions** (results produced with): python
3.12.11, torch 2.8.0+cu128, transformers 4.55.4, numpy 2.3.3. Accuracy moves 1–2
points with seed, so a different torch/transformers is a plausible source of a
similar shift — pin these when comparing against the tables.
## Getting the data (explicit — the runner never downloads)

The `prepare` phase only checks that the corpus and split already exist. Run
these first (from `configs/best_results.json`):

```bash
# fetch + selectively extract BinKit "normal" (Google Drive .7z).
# Shell script — run with bash, NOT python. Needs gdown (pip install gdown) + bsdtar.
bash scripts/fetch_binkit.sh normal

# pack .text into a flat corpus (steady state ~211 MB; binaries deletable after)
python scripts/build_corpus.py --root data/binkit/normal \
    --out data/corpus/binkit_x86_64 --arch x86_64 --compiler gcc clang \
    --opt O0 O1 O2 O3 --extra normal --workers 8 --yes
```

Split evidence: `reports/tables/split_programs.txt`.
## Replaying the best results

One driver, `scripts/run_best.py`, reads
[configs/best_results.json](configs/best_results.json) and each run's archived
`args.json`. **Dry-run by default** (prints, mutates nothing); `--execute` runs.

```bash
python scripts/run_best.py --list                        # groups + phases
python scripts/run_best.py --group NAME                  # print the plan (dry run)
python scripts/run_best.py --group NAME --execute        # run all phases
python scripts/run_best.py --group NAME --phase train    # choose one phase
```

- `--phase` defaults to `all`; dry-run is the default (`--dry-run` accepted).
- `offline` uses fresh fine-tune outputs under `results/best/finetune` by
  default; `--archive-probs` instead resolves probabilities from
  `results/explore` via the committed report metadata.
- All generated output lands under `results/best` (MLMs at `results/best/ckpt/mlm512` and `.../mlm2048`).

The separate `BinProv-probs-20260903.tar` archive is required for archived
probability recomputation. Extract it from the repository root with
`tar -xf /path/to/BinProv-probs-20260903.tar` (it creates `results/explore/*/probs.npz`).
The packed corpus is also required. New training generates its own probabilities.

After the data above is in place:

```bash
python scripts/run_best.py --group o2o3_7wide_16.9KB --execute         # best O2/O3 sequence result
python scripts/run_best.py --group opt4_6run_16.9KB --phase offline \
    --archive-probs --execute            # offline ensemble from archived probs
```

`o2o3_19run_binary` has no recipe and is explicitly rejected. Existing training
output directories are not overwritten; move them aside for a new run, or use
`--phase train` / `--phase offline` to start from completed prerequisite stages.
Run `--group all --execute` once to produce all supported result tables, reusing
shared model runs within that plan.
## Documentation

- [docs/HOPPER_TRAINING.md](docs/HOPPER_TRAINING.md) — Slurm recipe for training
  and exporting the best single `opt4_wide_seed29` model on an A100/H100 80 GB GPU;
  the verified A100 replica reached **84.23% sequence / 93.88% binary accuracy**.
- [docs/HUGGINGFACE_RELEASE.md](docs/HUGGINGFACE_RELEASE.md) — reviewed model
  card, repository naming, upload source, and publication checks.
- [docs/LOCAL_TRAINING.md](docs/LOCAL_TRAINING.md) — **bounded local training on
  one Mac** (MPS): the `opt4_narrow_seed13` / `opt4_wide_seed29` profiles, memory
  measurements, `--grad-accum`, resume/export workflow.
- [docs/BEST_RESULTS.md](docs/BEST_RESULTS.md) — **start here for results**:
  headline numbers, run membership, replay guide.
- Supporting documentation: [docs/RESULTS.md](docs/RESULTS.md) (paper
  reproduction), [docs/REPRODUCTION.md](docs/REPRODUCTION.md) (paper-table →
  command map), and [docs/DATA.md](docs/DATA.md) (disk budget, corpus format).
- Curated artifacts for the best recipes and released model: [reports/](reports/)
  (weights are hosted on Hugging Face).
## Citation

```
@inproceedings{xu2022binprov,
author = {He, Xu and Wang, Shu and Xing, Yunlong and Feng, Pengbin and Wang, Haining and Li, Qi and Chen, Songqing and Sun, Kun},
title = {BinProv: Binary Code Provenance Identification without Disassembly},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 25th International Symposium on Research in Attacks, Intrusions and Defenses},
pages = {350–363},
numpages = {14},
location = {Limassol, Cyprus},
series = {RAID '22}
}
```
