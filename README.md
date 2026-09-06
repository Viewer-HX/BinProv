# BinProv

BinProv identifies compilation provenance — which compiler and optimization
level — **without disassembling**: it reads the raw bytes of `.text`, embeds them
with a BERT-style byte-level encoder, and classifies. The 261-token vocabulary
(256 bytes + 5 special tokens) covers x86_64, x86_32, ARM and MIPS unchanged.

Paper: "BinProv: Binary Code Provenance Identification without Disassembly", RAID 2022 —
[10.1145/3545948.3545956](https://doi.org/10.1145/3545948.3545956). BibTeX: see [Citation](#citation).
> **Repository note.** This is a rebuild version, implemented on
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

## Measured results

BinKit x86_64, **program-grouped split** (47 test programs). The supported
full-training profile uses one A100 80 GB GPU.
Full provenance and replay:
[docs/RESULTS.md](docs/RESULTS.md). Verified offline tables: [reports/tables/best/](reports/tables/best/).

**Units.** The encoder *input* is 2048 bytes, but accuracy is scored at the
paper's **non-overlapping 512-byte stride** (the wide encoder sees overlapping
windows). A nominal **16.9 KB aperture** = mean of the window and its 16
neighbours either side (radius 16) at inference, without retraining. This is nominal target-window coverage;
wide inputs also include context beyond each target window.

| Result | Score | Meaning | Group |
|---|---|---|---|
| Released `opt4_wide_seed29` model | **84.23%** | sequence level, 116,321 windows | single model |
| Released `opt4_wide_seed29` model | **93.88%** | binary soft vote, 376 binaries | single model |
| O2/O3 (hard task), 7-wide ensemble @512 B | **75.10%** | sequence level, 55,657 windows | `o2o3_7wide_512B` |
| Same 7 runs, 16.9 KB aperture | **81.10%** | sequence level, mean over 33 windows | `o2o3_7wide_16.9KB` |
| 2048-B encoder alone on O2/O3 | **71.98%** (SD 0.68 pp) | mean of **three** seeds | `o2o3_pretrained_wide_mean` |
| 4-way O0/O1/O2/O3, 6-run ensemble | **87.62%** | sequence level, 116,321 windows, 16.9 KB aperture | `opt4_6run_16.9KB` |
| opt4, binary-level vote | **95.21%** | vote over the 3 *wide* runs only | `opt4_3wide_binary` |

Exact ensemble membership and per-run evidence are recorded in
[docs/RESULTS.md](docs/RESULTS.md).
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

## Hardware requirements for training

The released `opt4_wide_seed29` recipe was verified with the following
allocation. Inference is much lighter and does not require this hardware.

| resource | verified full-training configuration |
|---|---|
| GPU | 1× NVIDIA A100 80 GB, bf16 |
| CPU | 8 cores |
| system memory | 64 GB |
| free disk | at least 80 GB for corpus, checkpoints, logs, and export |
| software | Python 3.12.11, PyTorch 2.8.0, Transformers 4.55.4, NumPy 2.3.3 |
| elapsed time | 12:42:09 for MLM512, MLM2048, fine-tuning, evaluation, and export |

The exact CUDA micro-batches and gradient accumulation are in
[configs/gpu_release.json](configs/gpu_release.json). An equivalent CUDA GPU
may work; configurations with less than 80 GB of VRAM require smaller
micro-batches and matching increases in gradient accumulation, and have not
been verified here.

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
## Reproducing the measurements

One driver, `scripts/run_best.py`, reads
[configs/best_results.json](configs/best_results.json) and each run's recorded
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
python scripts/run_best.py --group o2o3_7wide_16.9KB --execute         # measured O2/O3 ensemble
python scripts/run_best.py --group opt4_6run_16.9KB --phase offline \
    --archive-probs --execute            # offline ensemble from archived probs
```

Existing training output directories are not overwritten; move them aside for a new run, or use
`--phase train` / `--phase offline` to start from completed prerequisite stages.
Run `--group all --execute` once to produce all supported result tables, reusing
shared model runs within that plan.
## Documentation

- [docs/LOCAL_TRAINING.md](docs/LOCAL_TRAINING.md) — **bounded local training on
  one Mac** (MPS): the `opt4_narrow_seed13` / `opt4_wide_seed29` profiles, memory
  measurements, `--grad-accum`, resume/export workflow.
- [docs/RESULTS.md](docs/RESULTS.md) — measured metrics, configuration
  membership, evidence, and reproduction.
- Supporting documentation: [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
  (method and command map) and [docs/DATA.md](docs/DATA.md) (disk budget and
  corpus format).
- Curated artifacts for the measured configurations and released model: [reports/](reports/)
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
