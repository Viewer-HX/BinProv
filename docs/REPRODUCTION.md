# Reproducing the paper

The original implementation was lost, so this is a rebuild from the paper text.
Where the paper specifies something, this code follows it. Where it does not,
the choice is listed under [Interpretations](#interpretations) below, with the
reasoning — so you can see which numbers depend on a judgement call of mine
rather than on the paper.

Paper: He et al., *BinProv: Binary Code Provenance Identification without
Disassembly*, RAID 2022. <https://doi.org/10.1145/3545948.3545956>

---

## The pipeline

```
                       scripts/fetch_binkit.sh          (or build_local_dataset.py)
                                  |
                          tree of ELF binaries
                                  |
                       scripts/build_corpus.py          §3.1  .text -> byte sequences
                                  |
                       data/corpus/<name>/               packed, binaries deletable
                                  |
                    +-------------+-------------+
                    |                           |
        scripts/pretrain_mlm.py        (warm start)
        §3.2  MLM, 20% masked                   |
                    |                           |
              checkpoints/mlm ------------------+
                                                |
                                    scripts/finetune.py     §3.3  one task per model
                                                |
                                       checkpoints/<task>
                                                |
                                    scripts/evaluate.py     §3.4  majority voting
                                                |
                                        results/tables.md
```

## Commands, end to end

This machine's GPUs are shared and usually memory-full, so pin a device
explicitly. Check what is free first:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sort -t, -k2 -rn
export CUDA_VISIBLE_DEVICES=<a GPU with real free memory, not a guess>
export PY=python3
```

### 1. Corpus

```bash
$PY scripts/build_corpus.py --root data/binkit/normal --out data/corpus/x86_64 \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal
```

### 2. Pre-train the embedding model (§3.2)

```bash
$PY scripts/pretrain_mlm.py --corpus data/corpus/x86_64 \
    --out checkpoints/mlm_x86_64 --epochs 10 --batch-size 64
```

12 layers, 768 hidden, 12 heads → 86.2M trainable parameters, matching the
paper's "around 80 million". Masking is 20% of bytes, of which 50% become
`<mask>` and 50% a random byte, re-drawn every epoch (§4.1).

### 3. Fine-tune one model per task (§3.3)

```bash
for task in compiler opt_hl opt4 opt_o0o1 opt_o2o3; do
  $PY scripts/finetune.py --corpus data/corpus/x86_64 --task $task \
      --init-from checkpoints/mlm_x86_64 --out checkpoints/$task --epochs 5
done
```

### 4. Evaluate, with voting (§3.4)

```bash
$PY scripts/evaluate.py --corpus data/corpus/x86_64 --out results/x86_64 \
    --ckpt compiler=checkpoints/compiler \
    --ckpt opt_hl=checkpoints/opt_hl \
    --ckpt opt4=checkpoints/opt4 \
    --ckpt opt_o0o1=checkpoints/opt_o0o1 \
    --ckpt opt_o2o3=checkpoints/opt_o2o3
```

Writes `results/x86_64/tables.md` (the paper's tables) and `results.json`.

## Table → command map

| Paper | What it reports | How to get it |
|---|---|---|
| Table 3 | compiler and High/Low opt, sequence level, plus joint "Overall" | `evaluate.py` with `--ckpt compiler=… --ckpt opt_hl=…` |
| Table 4 | O0/O1/O2/O3, O0/O1, O2/O3, sequence level | `--ckpt opt4=… --ckpt opt_o0o1=… --ckpt opt_o2o3=…` |
| Table 5 | per-level precision/recall/F1, split by compiler | `--ckpt opt4=…` (emitted automatically) |
| Table 6 | function- and binary-level voting | same run; `--levels binary function` is the default |
| Table 7, ISA rows | per-architecture accuracy | build one corpus per `--arch`, repeat steps 2–4 |
| Table 7, OBF rows | obfuscated binaries | `fetch_binkit.sh obfus`, then `--obfuscation sub bcf fla` |
| §5.1(d) | architecture identification | one corpus with several `--arch`, then `--task arch` |

Baselines (Origin, O-glassesX) and the case studies of §5.2–5.4 are **not**
implemented — see [Not implemented](#not-implemented).

## Interpretations

Places where the paper does not pin a detail down. Each is a real fork in the
road, and each is exposed as a flag so you can go the other way.

### 1. Function-level voting needs overlapping sequences

The paper cuts a binary into fixed 512-byte sequences (§3.1) and then votes "over
the sequences belonging to the same function" (§3.4), reporting a large gain from
doing so — O2/O3 rises from 83.64% to 94.70% (Tables 4, 6).

But most functions are considerably shorter than 512 bytes. Under a
non-overlapping cut, a typical function contains at most one sequence, and voting
over one vote changes nothing. So the paper must be doing something else, and it
does not say what.

**This has now been measured, and the mechanism as described cannot produce the
reported result.** On BinKit's x86_64 test set the median function is 127 bytes
and only 18.5% reach 512 bytes, so the median sequences-per-function stays at 1
for *any* stride (512 → 1, 128 → 1, 64 → 1, 32 → 1). For 81.5% of functions there
is nothing to vote over, whatever the stride. Numbers in
[RESULTS.md](RESULTS.md#1-function-level-voting-cannot-work-as-described).

This code cuts *within* each function using an overlapping stride
(`--function-stride`, default `seq_len // 4`), so the minority of large functions
do get several sequences and no sequence straddles a boundary. The
`### voting group sizes` table in the output reports the mean and median
sequences per group, and `evaluate.py` prints an explicit caveat when the median
is 1 — so a function-level number that is not a voting result cannot be read as
one by accident.

The corollary, which *was* confirmed: a short function padded to 512 bytes carries
much less context than an ordinary sequence, so the function level measures a
**harder** input distribution rather than a voted one — which is why it comes out
below the sequence level here while the paper has it above.
`--function-context` centres a full-length window on the function instead,
borrowing neighbouring bytes, and gains 5–9 points across tasks with no
retraining (compiler then exceeds the paper). It is off by default because those
borrowed bytes lie outside the function: not label leakage, since provenance is a
binary-level property, but a "function-level" prediction that uses information
from outside the function.

The remaining alternative reading — assign each whole-binary sequence to the
function containing its midpoint — is not implemented; it gives even fewer
sequences per function.

Binary-level voting has no such ambiguity: `level="binary"` with the default
stride is exactly §3.1's cut, and it behaves as the paper describes (+18.6 points
on O2/O3).

### 2. The classifier's border-weakening first layer

§3.3: "The first layer aims to reshape the input vectors and weaken the border
weights of the embeddings", motivated by x86's variable-length instructions being
cut mid-instruction at a sequence boundary. No mechanism is given.

Implemented as `BorderTaperedPool`: a learned attention over token positions,
whose logits are *initialised* with a linear ramp so that the outermost 32
positions start down-weighted. Training can undo that if it turns out not to
help. Ablate with `--pool mean` (plain masked mean) or `--pool cls`.

### 3. Train/test split granularity

§4.1 says the ratio is 8:2 over "different binaries", and that train and test do
not overlap. It does not say whether the variants of one program are kept
together.

Default here is `--split-group-by program`: every compiled variant of a program
(both compilers, all four optimization levels) lands on the same side. Otherwise
`ls` compiled at O2 could be in training while `ls` at O3 is in test, and the
model could recognise the program rather than the optimization level — which
would inflate exactly the O2/O3 number the paper highlights as hard. Use
`--split-group-by binary` for the looser reading.

**Measured: the looser split is *worse*, not better** — O2/O3 drops from 67.08%
to 60.15%. Program identity carries no information about optimization level, so
memorising "program X was O2" actively misleads the model when X@O3 appears in
test. Note that arm is also confounded: random binary selection does not preserve
the O2:O3 sequence ratio (test came out 60.3% O3 against a 51.8% training prior),
so it measures label-prior shift as well as split granularity. A clean version
would stratify by (program, optimization level). Either way, the strict split is
the defensible default and is what the reported numbers use.

### 4. Pre-training set and leakage

§4.1: "we first construct the pre-training set by selecting at least one binary
(2 × 4 variants) from each software project". Implemented as one program per
package, all its variants (`--pretrain-per-package`).

The paper does not say whether the pre-training set excludes test binaries. Here
it does — `pretrain_bids(restrict_to=train)`. MLM uses no labels, but a model
that has already fit the byte distribution of a test binary is not measuring
generalization. Pass `--split-name train` to pre-train on all training binaries
instead.

**Watch the size of this set.** On BinKit's 51 packages, one program per package
× 8 variants is a few hundred binaries and tens of megabytes of `.text` — enough
for MLM. On a small corpus it is not: the local dataset here has 17 packages,
giving 128 binaries and ~2,300 sequences (≈1.2 MB of code), and MLM plateaus at
roughly unigram-model quality (masked-byte accuracy ~15%, loss ~4.4 nats) no
matter how many epochs you run. `pretrain_mlm.py` prints the sequence count on
startup — if it is not in the tens of thousands, pre-train on `--split-name
train` instead, and do not expect the warm start to be worth much.

### 5. Segment embedding `E_s`

§3.2 defines a segment sequence marking "which binary program each byte belongs
to, when a byte sequence contains multiple fragments from different programs".
With one binary per sequence that embedding is constant and learns nothing. The
paper never says it spliced fragments, so the default is single-segment;
`--pair-prob` in `pretrain_mlm.py` turns splicing on if you want `E_s` to carry
information.

### 6. Optimizer and schedule

Not specified beyond "gradient descent back-propagation". Used here: AdamW,
linear warmup then cosine decay, bf16 autocast, and lr 1e-4 for MLM with 3e-5 on
the encoder / 1e-4 on the head for fine-tuning (§3.3 notes the head needs larger
adjustments than the pre-trained encoder).

**The MLM learning rate has to be scaled to the batch size, and this bites.** The
first attempt used RoBERTa's published 6e-4, which goes with a batch of 8192. At
batch 64 — what one shared GPU comfortably runs — the loss fell to 4.15 nats
during warmup and then *rose* back toward 4.20 once the rate reached its 5e-4
peak, drifting back to the unigram solution. Scale it down with the batch
(sqrt scaling from RoBERTa's setting gives ~1e-4 at batch 256), or raise the
batch. This is the same sensitivity that collapses fine-tuning at 1e-4, and in
both cases the symptom is a loss curve that stops improving rather than an error.

### 6b. Read the MLM loss against the unigram baseline, not against zero

`pretrain_mlm.py` prints the corpus's unigram cross-entropy on startup, because
without it the MLM curve cannot be interpreted. Machine code is dominated by a
few byte values — `0x00` alone is ~14-15% of BinKit's x86_64 `.text`, and the
unigram entropy is ~4.33 nats — so a model that has learned nothing but the byte
frequencies still reports a plausible-looking ~15% masked-byte accuracy.

Measured on BinKit x86_64: unigram is 4.35 nats / 13.98% top-1. Treat a curve
flattening anywhere near 4.35 as "learned the histogram", not "learned context",
and expect it to make a poor warm start.

### 7. Framework

The paper built the network with fairseq. This uses HuggingFace `transformers`,
because fairseq pins old PyTorch versions and no longer installs cleanly. The
architecture is the same — a 12-layer, 768-hidden bidirectional encoder over a
261-token vocabulary (256 bytes + `<pad> <s> </s> <unk> <mask>`).

### 8. Non-code bytes in `.text`

The paper acknowledges this as a limitation (§6): alignment and padding bytes
between functions are fed to the model as if they were code. No filtering is done
here either, matching the paper.

## Training failure mode you will probably hit

The model collapses easily — near-constant logits, every sequence assigned the
same class, loss pinned at exactly `ln(num_classes)`. In an accuracy table this
is indistinguishable from "the task is hard", so `finetune.py` detects it and
prints the likely causes. Measured on the small local corpus (2,048 sequences,
300 steps, GCC-vs-Clang, `--pool border`):

| encoder | init | lr | train acc | logit spread |
|---|---|---|---|---|
| 4L / 256 | random | 1e-4 | 90.6% | 2.85 |
| 12L / 768 | random | 1e-4 | 53.7% | 0.14 ← collapsed |
| 12L / 768 | random | 3e-5 | 69.8% | 2.69 |
| 12L / 768 | random | 1e-5 | 80.6% | 2.66 |
| 12L / 768 | weak MLM | 3e-5 | 53.0% | 0.16 ← collapsed |
| 12L / 768 | weak MLM | 1e-5 | 53.5% | 0.23 ← collapsed |

Two things to take from this:

1. **An under-trained MLM checkpoint is worse than no pre-training.** The last two
   rows collapse where the *same architecture from random init at the same
   learning rate* trains fine. An MLM that has only learned the marginal byte
   distribution appears to drive token representations toward a degenerate
   solution the classifier cannot recover from. So verify the warm start is
   earning its place: run `finetune.py --from-scratch` as a control. If scratch
   wins, your MLM stage is too short, not your classifier.
2. **Raising the learning rate does not compensate.** It is the natural reflex
   when a model will not move, and here it makes things strictly worse — 1e-4
   collapses the 12-layer encoder outright. On small data the ordering was
   monotone in favour of *lower* rates.

The paper's 12-layer/768-hidden encoder is sized for BinKit and does not train on
a few thousand sequences at any learning rate tried. `scripts/run_smoke.sh`
therefore uses `--layers 4 --hidden 256`, which reaches ~96% on the local corpus'
compiler task. Use the paper's size only with a BinKit-scale corpus.

## Not implemented

Deliberately out of scope for this rebuild, in rough order of how much work each
would be:

- **Baselines** — Origin, O-glassesX (both need external tools plus objdump
  features), and the paper's own BinRNN / O-glassesX* variants (Table 8).
- **Analysis experiments** — the NCD/length/position studies of §5.2 and
  Figures 2, 5, 6. `Corpus.sequences()` already returns the byte offsets needed
  for the position analysis, and `SequenceIndex` carries `bid`, so the plumbing
  exists.
- **Case studies** — compiler helper-function detection (§5.3, Table 10) and
  binary-similarity bucketing (§5.4, Table 11). Function *names* are kept in the
  corpus index specifically so helper-function labels can be derived later.

## What has actually been verified here

Full numbers, the eight refuted hypotheses for the O2/O3 gap, and two findings
about the paper's own method are in **[RESULTS.md](RESULTS.md)**. In short:

- `tests/test_pipeline.py` — 20 tests, all passing. They cover what would corrupt
  results rather than crash: label parsing across both real layouts, byte/`<pad>`
  separation (0x00 is everywhere in real `.text`), sequence cutting never crossing
  a function boundary, program-name extraction (getting it wrong silently defeats
  the leak-free split), split leakage, the pre-training set staying inside train,
  MLM masking ratios, and voting arithmetic.
- The ELF reader matches `readelf` on `.text` offset, size and virtual address.
- Label parsing was checked against **all 67,680 paths** of BinKit Normal:
  architecture, compiler, compiler version and optimization level are correct for
  every one.
- Model parameter count 86.2M, consistent with the paper's ~80M.
- Corpus packing measured 5.2x smaller than the source binaries on BinKit
  (1.1 GB -> 211 MB), 12x on the locally built dataset.
- **Reproduced on BinKit x86_64**: Table 3's "Overall" 94.98% vs the paper's
  94.77%; compiler 99.81% (paper 95.47%) and 100% at binary level; O0/O1 99.80%
  (paper 98.49%); and every qualitative finding — O2<->O3 dominating the confusion
  matrix, Clang O1 over-prediction, large binary-level voting gains, GCC over
  Clang, fixed-width ISAs over x86.
- **Not reproduced**: O2/O3, 67.08% against the paper's 83.64%, and the 4-way task
  that depends on it (79.98% vs 91.07%). Eight hypotheses tested and refuted
  across five orthogonal dimensions; see RESULTS.md.
- MLM pre-training transfers positively, measured under a controlled comparison at
  equal budget: warm start 99.77% vs random init 92.78% on the compiler task.
