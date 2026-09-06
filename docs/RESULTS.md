# Results on BinKit

This page summarizes the initial paper-reproduction measurements. The repository
keeps detailed evidence only for the selected best recipes; see
[BEST_RESULTS.md](BEST_RESULTS.md) and [../reports/](../reports/).

What this rebuild actually produced, measured on BinKit Normal. Read
[REPRODUCTION.md](REPRODUCTION.md) alongside it — several numbers depend on a
choice the paper leaves unspecified, and those choices are listed there.

**Summary: most of the paper reproduces; the O2/O3 task does not.** Table 3's
headline "Overall" lands within 0.2 points, compiler and O0/O1 identification meet
or exceed the paper, and every qualitative finding reproduces. Fine-grained
O2/O3 comes out 16.6 points low, and eight hypotheses for why were tested and
refuted. Two methodological gaps in the paper were found and quantified along the
way.

## Setup

| | |
|---|---|
| Dataset | BinKit Normal, `gcc-8.2.0` + `clang-7.0`, O0–O3 |
| Main corpus | x86_64, 1,880 binaries, 211 MB `.text`, 315,710 sequences of 512 B |
| Pooled corpus | 4 architectures, 7,520 binaries, 903 MB `.text`, 1,366,685 sequences |
| Split | 8:2 grouped by **program** (188 train / 47 test programs, zero overlap) |
| Encoder | 12 layers, 768 hidden, 86.2M parameters |
| MLM | 10 epochs, batch 256, lr 1e-4 → val loss 0.941, masked-byte acc 75.15% |
| Fine-tuning | 3 epochs per task, batch 128, lr 3e-5 encoder / 1e-4 head |

Compiler versions are pinned to one per family on purpose: BinKit ships 5 GCC and
4 Clang versions, and using all of them would put 9 compilers into a 2-class task.
7,520 binaries across 4 architectures is the closest match to the paper's Table 2
count of 6,280.

## Table 3 — basic tasks, sequence level

| Basic task | This rebuild | Paper | majority baseline |
|---|---|---|---|
| Compiler (GCC/Clang) | **99.81%** | 95.47% | 50.25% |
| Opt level (High/Low) | 95.11% | 98.90% | 52.15% |
| **Overall (both correct)** | **94.98%** | **94.77%** | — |

## Table 4 — fine-grained optimization level, sequence level

| Opt level | This rebuild | Paper | majority baseline |
|---|---|---|---|
| O0/O1/O2/O3 | 79.98% | 91.07% | 32.42% |
| O0/O1 | **99.80%** | 98.49% | 62.17% |
| O2/O3 | 67.08% | 83.64% | 52.95% |

The majority baseline is well above `1/num_classes` because the corpus is balanced
per *binary*, not per *sequence*: O0 binaries are longer, so they contribute more
sequences. This independently reproduces the paper's Figure 2(b) — O1 shortest,
O2 < O3.

## Table 5 — per class, split by compiler (4-way task)

| Metric | GCC O0 | GCC O1 | GCC O2 | GCC O3 | Clang O0 | Clang O1 | Clang O2 | Clang O3 |
|---|---|---|---|---|---|---|---|---|
| Precision | 99.79% | 98.47% | 65.35% | 74.00% | 99.82% | 64.09% | 58.74% | 63.13% |
| Recall | 99.75% | 98.62% | 71.36% | 68.17% | 99.75% | 80.31% | 51.12% | 58.91% |
| F1 | 99.77% | 98.54% | 68.22% | 70.97% | 99.79% | 71.29% | 54.66% | 60.95% |

Confusion matrix (rows = truth):

| true \ pred | O0 | O1 | O2 | O3 |
|---|---|---|---|---|
| O0 | **37618** | 14 | 55 | 25 |
| O1 | 32 | **20674** | 1153 | 1093 |
| O2 | 19 | 2747 | **15976** | 7443 |
| O3 | 23 | 2246 | 8442 | **18761** |

O2↔O3 accounts for 15,885 of the errors — the overwhelming majority. GCC O0/O1
match the paper almost exactly (99/98 vs the paper's 99/98). The gap is
concentrated on the Clang side.

## Table 6 — joint inference by majority voting

| Provenance | sequence | function | binary | paper (seq / func / bin) |
|---|---|---|---|---|
| Compiler | 99.81% | 90.51% → **99.27%** | **100.00%** | 95.47 / 97.60 / 100 |
| Opt (High/Low) | 95.11% | 83.97% → 92.26% | 98.40% | 98.90 / 98.20 / 100 |
| Opt (O2/O3) | 67.08% | 59.00% → 64.18% | 85.64% | 83.64 / 94.70 / 99.8 |
| Opt (O0/O1/O2/O3) | 79.98% | 68.57% | 89.89% | 91.07 / — / — |

The arrow is the function-level number without → with `--function-context` (see
finding 3 below). Binary-level voting behaves exactly as the paper describes:
+18.6 points on O2/O3, +9.9 on the 4-way task, and 100% on compiler.

## What reproduces

- **Table 3's "Overall": 94.98% vs 94.77%** — within 0.2 points.
- **Compiler identification** exceeds the paper at sequence level (99.81% vs
  95.47%) and matches it at binary level (100%).
- **O0/O1** exceeds the paper (99.80% vs 98.49%).
- **O2↔O3 dominates the confusion matrix**, the paper's central claim about which
  boundary is hard.
- **Clang O1 is over-predicted.** §5.1(b) reports "BinProv is prone to misclassify
  the sequences on O2/O3 as O1 under Clang"; here Clang O1 precision is 64.09%
  against GCC's 98.47% — the same asymmetry, more severe.
- **GCC outperforms Clang**, as §5.1(b) reports.
- **Binary-level voting gives large gains** on every task.
- **ARM/MIPS beat x86** (Table 7's ISA rows, see below) — fixed-width
  instructions keep sequences aligned.
- **MLM pre-training transfers positively.** Controlled at equal budget on the
  compiler task: warm start 99.77% vs random init 92.78%. At step 200 the
  warm-started model was already at 92% where random init was at 58%.

## What does not: O2/O3

67.08% against the paper's 83.64%. Eight hypotheses were tested; all were
refuted. Each row changed one thing and re-measured.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Under-trained | epoch trajectory + lr | ✗ train 83.3% / test 67.1% — overfitting, not underfitting. |
| 2 | Paper used a looser split | rebuild split by binary | ✗ 60.15%, *worse* (and the arm has label-prior shift — see below) |
| 3 | Encoder too weak | pre-train on 5.6× data (4 arch, 903 MB) | ✗ 67.67%, +0.59 |
| 4 | Test set imbalance | reweight existing predictions to balanced | ✗ 67.05%, −0.03 |
| 5 | Paper's number is a voted one | compare at all three granularities | ✗ gap at every one: −16.6 / −35.7 / −14.2 |
| 6 | Function-level input distribution | centred full-length windows | **◐ +5.2 points at function level**, nothing at sequence level |
| 7 | Training set imbalance | `--balance` | ✗ 66.45%, −0.63 |
| 8 | Fine-tuning data too narrow | fine-tune on 4.1× data (4 arch) | ✗ 67.51% on x86_64, +0.43 |

Five orthogonal dimensions — pre-training data ×5.6, fine-tuning data ×4.1, class
balancing, split protocol, aggregation granularity — all land in a 66–68% band.
The gap is not on any axis that was varied.

Two cautions about that table, both of which cost real time to notice:

- **Hypothesis 2's experiment is flawed.** Splitting by binary at random does not
  preserve the O2:O3 *sequence* ratio: test came out 60.3% O3 against a 51.8%
  training prior. Its 60.15% is roughly the majority baseline (60.29%), so that
  arm measures label-prior shift as much as split granularity. A correct version
  would stratify by (program, optimization level).
- **Hypothesis 8 nearly produced a false positive.** On the mixed 4-architecture
  test set it scored 73.66%, which looks like a 6.6-point win over 67.08%. Split
  by architecture, x86_64 alone is 67.51%. The entire apparent gain came from
  ARM/MIPS being easier. Numbers computed over different test-set compositions are
  not comparable, however similar the metric name.

## Two findings about the paper

### 1. Function-level voting cannot work as described

Table 6 reports function-level accuracy *above* sequence-level (O2/O3: 94.70% vs
83.64%). Majority voting can only help when a function contains several sequences.
Measured on BinKit's x86_64 test set (137,928 functions):

| percentile | function size |
|---|---|
| 50% | **127 bytes** |
| 75% | 359 bytes |
| 90% | 947 bytes |
| ≥ 512 bytes | only **18.5%** of functions |

So 81.5% of functions cannot hold even one full 512-byte sequence, and the median
sequences-per-function stays at 1 for *any* stride:

| stride | median seqs/function | mean |
|---|---|---|
| 512 | 1 | 1.5 |
| 128 | 1 | 2.8 |
| 64 | 1 | 4.6 |
| 32 | 1 | 8.0 |

#### The ceiling this puts on function-level accuracy

The size distribution is enough to bound the reported number, using nothing but the
paper's own accuracies. Function-level accuracy is a weighted average over
*functions*:

```
acc_function = f1 * acc(functions yielding 1 sequence)
             + f2 * acc_voted(functions yielding >= 2 sequences)
```

A one-sequence group has one vote, so voting cannot change it — its accuracy is
exactly its per-sequence accuracy. The ceiling therefore comes from assuming voting
is *perfect* on every multi-sequence function. On the O2/O3 test split, cut the
paper's way (512-byte non-overlapping windows within each function), `f1 = 81.08%`
and `f2 = 18.92%` of 60,361 functions:

| assumed accuracy on the 81.08% of functions with one sequence | function-level ceiling | paper reports |
|---|---|---|
| 57.28% — measured here (see below) | 65.36% | 94.70% |
| **83.64% — the paper's own sequence-level accuracy** | **86.74%** | **94.70%** |
| 90%, generously | 91.89% | 94.70% |
| 100%, degenerate | 100.00% | 94.70% |

The second row is the load-bearing one. Grant the paper its own sequence-level
accuracy on short functions *and* perfect voting on every function long enough to
vote, and function-level still cannot exceed **86.74%** against the 94.70% reported.
Reaching 94.70% would require the one-sequence functions to be classified at
**93.5%**. Both inputs to that bound — the paper's 83.64% and the BinKit function
size distribution at a 512-byte cut — are properties of the paper's own setup, not
of this rebuild.

And the requirement runs the wrong way. Measured with the O2/O3 model on
within-function windows:

| functions | per-sequence accuracy |
|---|---|
| yielding 1 sequence (short) | **57.28%** |
| yielding ≥ 2 sequences (long) | **68.40%** |

Short functions are **11.1 points harder**, not easier: at a 127-byte median, three
quarters of a padded 512-byte window is padding, so it carries far less context than
an ordinary sequence. This is the same mechanism as the context-width result: accuracy on this task is governed by how many real bytes the prediction can draw on.

Voting these groups gives **59.79%** at the function level against **69.23%** at the
sequence level: it moves *down* by 9.4 points, which is the direction the mechanism
predicts and the opposite of the paper's +11.1.

So the conclusion is stronger than "the numbers disagree". Under the reading that
§3.4's wording most directly supports — majority voting over the 512-byte sequences
lying inside a function — the reported function-level accuracy is not attainable at
any per-sequence accuracy the paper also reports. The paper does not say how
sequences are assigned to functions, so some other assignment must be intended;
finding 3 below measures the most plausible alternative (windows *centred* on the
function, borrowing neighbouring bytes) and it moves in the right direction but
recovers only 5.2 of the 11.1 points, while no longer being function-local.

`evaluate.py` prints the sequences-per-group distribution and warns when the median
is 1, so this cannot pass unnoticed.

### 2. Function-level evaluation needs context from outside the function

The corollary: a short function padded to 512 bytes carries far less context than
an ordinary sequence, making the function level *harder* rather than easier.
Centring a full-length window on the function instead — `--function-context`,
which raises the share of full 512-byte windows from 64.6% to 93.5% — improves
every task with no retraining:

| task | function bytes only | centred context window | change |
|---|---|---|---|
| Compiler | 90.51% | **99.27%** | +8.76 |
| Opt (High/Low) | 83.97% | 92.26% | +8.29 |
| Opt (O2/O3) | 59.00% | 64.18% | +5.18 |

Compiler identification then exceeds the paper's 97.60%. This is the one
hypothesis of the eight that was confirmed, and it is off by default because it
has a cost worth stating: the window includes neighbouring functions' bytes. That
is not label leakage — provenance is a property of the whole binary, so the
borrowed bytes share the label — but it does mean a "function-level" prediction
uses information from outside the function.

## Table 7 — ISA rows (O2/O3)

One model fine-tuned on all four architectures, then evaluated per architecture.

| Architecture | sequence | Paper | gap | binary (voted) |
|---|---|---|---|---|
| x86_64 | 67.51% | 83.64% | −16.13 | 86.70% |
| x86_32 | 71.12% | 82.25% | −11.13 | 86.70% |
| arm_64 | 71.88% | 87.46% | −15.58 | 91.49% |
| mips_64 | **81.94%** | 87.91% | **−5.97** | 92.02% |

The paper's main claim about architectures holds: MIPS_64 and ARM_64 are the
easiest, and it attributes that to fixed-width instructions keeping sequences
aligned at the boundaries. Difficulty ordering here is
`mips_64 > arm_64 > x86_32 > x86_64` against the paper's
`mips_64 > arm_64 > x86_64 > x86_32` — the top two agree, and only the two x86
variants swap. The paper explains its x86_32 < x86_64 result by x86_32's extra
auxiliary functions (`__x86.get_pc_thunk.bx`), a subtle enough effect that a flip
between two adjacent values is weak evidence either way.

MIPS_64 also comes closest to the paper in absolute terms (−5.97). It would be
tidy to read the gap as tracking instruction-encoding regularity, but the data
does not support that: ARM_64 is fixed-width too and is 15.6 points down, further
than variable-width x86_32. There is no clean pattern in the residual, which is
consistent with the O2/O3 gap having a cause that was not isolated.

## Pre-training

| | x86_64 | 4 architectures pooled |
|---|---|---|
| Unique data | 161 MB | 903 MB |
| Sequences | 315,710 | 1,366,685 |
| Schedule | 10 epochs × 1,208 steps | 4 epochs × 5,231 steps |
| Wall clock | 1h11m | 2h01m |
| Unigram baseline | 4.351 nats / 13.98% | 4.563 nats / 17.49% |
| Final val loss | 0.9413 | **0.9276** |
| Final masked-byte acc | 75.15% | **75.78%** |
| Below its own baseline | 3.410 nats | **3.635 nats** |

Read an MLM curve against the unigram baseline, never against zero: `0x00` alone
is ~14% of x86_64 `.text`, so a model that has learned only the byte histogram
still reports ~15% masked-byte accuracy. `pretrain_mlm.py` prints the baseline on
startup for this reason.

The recorded curve has a sharp phase transition rather than smooth convergence — flat at
the unigram baseline for ~1,800 steps, then a collapse from 4.07 to 2.94 nats
around step 2,000. Two earlier attempts ended before that transition and looked
like slow convergence; they were not converging at all.

## Reproducing these numbers

```bash
pip install gdown
scripts/fetch_binkit.sh normal
CUDA_VISIBLE_DEVICES=<free gpu> ARCH=x86_64 \
  CORPUS=data/corpus/binkit_x86_64 CKPT=checkpoints/binkit_x86_64 \
  RESULTS=results/binkit_x86_64 FT_EPOCHS=3 BATCH=128 \
  scripts/run_binkit.sh
```

For a measured compute budget, see the hardware requirements in the repository
README: the complete wide training, evaluation, and export pipeline finished
in 12:42:09 on one A100 80 GB.
