# Datasets and disk budget

BinProv reads exactly two things out of a binary: the bytes of the `.text`
section, and — only for function-level voting — the function boundaries in the
symbol table. Everything else in an ELF file (`.data`, `.rodata`, relocations,
DWARF, the symbol *names*) is dead weight for this model. The pipeline is built
around that fact, so that a multi-GB dataset collapses to a corpus you can keep
around indefinitely.

## Measured, on this machine

All numbers below were measured, not estimated.

**BinKit x86_64**, the paper's main experiment (1,880 binaries, gcc-8.2.0 +
clang-7.0, O0–O3):

| stage | on disk | time |
|---|---|---|
| `normal.7z` archive | 3.5 GB | 1m36s to download |
| extracted subset, 4 architectures (7,520 binaries) | 4.5 GB | 3m52s |
| — of which x86_64 (1,880 binaries) | 1.1 GB | |
| **packed x86_64 corpus** (`text.u8` + index) | **211 MB** | 1.8s |

A **5.2× reduction** for BinKit, and **12×** on the locally built dataset
(31.3 MB of binaries → 2.5 MB corpus). BinKit does worse because its `.text` is a
larger share of each file (~40%, versus ~8% for the local build).

The resulting corpus holds 315,710 training sequences of 512 bytes, and all 1,880
binaries carry function symbols, so function-level voting is available.

## BinKit sizing

| | binaries | avg size | total |
|---|---|---|---|
| BinKit Normal archive, all 288 toolchains | 67,680 | ~600 KB | **~40 GB** |
| 4 architectures, one compiler version each | 7,520 | ~600 KB | **4.5 GB** |
| x86_64 only | 1,880 | ~600 KB | **1.1 GB** |

Note the ~600 KB average, not the ~201 KB the BinKit paper quotes: the
distributed archive is `gnu_debug`, i.e. built with debug info. Expansion from
the 3.5 GB archive is therefore about 11×, not 4×.

Two selections matter for size, and the defaults in `scripts/fetch_binkit.sh`
apply both:

- **Architectures.** The paper uses 4 of BinKit's 8.
- **Compiler versions.** BinKit Normal ships **5 GCC and 4 Clang versions**.
  Taking all of them would put 9 compilers into a 2-class "GCC vs Clang" task and
  multiply the corpus ~4.5×. The defaults pin `gcc-8.2.0` and `clang-7.0` — the
  versions named in the paper's own Figure 2 legend — giving 7,520 binaries
  across 4 architectures, the closest match to Table 2's 6,280. Override with
  `KEEP_COMPILERS="gcc clang"`, or at corpus-build time with
  `--compiler-version`.

Peak disk while working through the pipeline is **~8 GB** (3.5 GB archive +
4.5 GB extracted). Steady state after packing and deleting both is ~211 MB per
architecture, so well under 1 GB even with all four.

### What the download actually is (probed, not assumed)

- The Normal dataset is **`normal_dataset.7z`, 3.5 GB** — a 7z archive, not a
  tarball. Plan tooling accordingly.
- Google Drive delivered **~40 MB/s** through the proxy on this host, so the
  download is **~1.5 minutes**. It is not the bottleneck; decompression is.
- No Drive quota error was returned, but that can change — a popular research
  dataset can start replying "too many users have viewed or downloaded this
  file". If the downloaded file turns out to be a small HTML document, that is
  what happened, and the script says so.

Two consequences of 7z worth knowing before you plan around it:

- **It is not streamable.** 7z keeps its index at the end of the file, so you
  cannot list or extract anything until the whole 3.5 GB has arrived. A tarball
  could have been filtered mid-stream; this cannot.
- **Solid compression.** Extracting one ninth of the entries may still require
  decompressing whole solid blocks, so the selective extraction saves *disk* but
  not necessarily *time*.

`bsdtar` (libarchive) is the extraction tool of choice: it is already present
here, it is C rather than Python, and it reads `.7z` and `.tar.*` with the same
wildcard syntax. `pip install py7zr` is the fallback if libarchive lacks 7z
support — it also does selective extraction, via
`SevenZipFile.extract(targets=[...])`, just more slowly.

Peak and steady-state footprint for the Normal dataset (measured):

```
1. download archive                     += 3.5 GB                (transient)
2. selective extract (7,520 binaries)   += 4.5 GB                (transient)
3. delete archive                       -= 3.5 GB
4. build corpus (.text only)            += 211 MB per arch        (keep this)
5. rm -rf the extracted binaries        -= 4.5 GB
                                        ------------------------
   peak 8 GB, steady state under 1 GB
```

Step 5 is safe: nothing downstream reads the binaries again. If you want it done
automatically, `build_corpus.py --purge-source --yes` deletes each file as soon
as it has been packed (it insists on `--yes` because it is destructive, and you
should only use it for data you can re-download).

## Knobs that bound corpus size

In rough order of preference:

- `--per-toolchain-limit N` — at most N binaries per compilation configuration.
  The best knob, because it shrinks the corpus while keeping it balanced across
  the classes you are trying to separate.
- `--packages a b c` — restrict to particular packages.
- `--arch / --compiler / --opt / --extra` — the primary filters; always set
  these, since the default is "everything found".
- `--no-func-names` — keeps function boundaries but drops the names. Cheap
  saving; names are only needed if you extend this to the paper's compiler
  helper-function study.
- `--no-functions` — skips the symbol table entirely. Faster and smaller, but
  disables function-level voting (paper Table 6).
- `--max-text-bytes N` — truncates each `.text`. **Use with care**: §5.2(c) of
  the paper shows the head of a binary is the easiest part to classify, so
  truncation flatters the model and invalidates any position analysis.

Sequences are *not* stored — `Corpus.sequences()` cuts them on the fly from the
memory-mapped `text.u8`. Changing sequence length or stride therefore needs no
rebuild and costs no extra disk.

## Corpus layout

```
data/corpus/<name>/
  text.u8             every .text concatenated, uint8, no padding
  binaries.jsonl.gz   one record per binary: offsets, labels, function table
  meta.json           counts, label tallies, the exact build parameters
  splits/default.json binary-id lists for train / test / pretrain
```

`meta.json` records the filters the corpus was built with, so a corpus is
self-describing after the fact.

## The three data paths

### 1. BinKit (what the paper used)

```bash
pip install gdown
scripts/fetch_binkit.sh normal --list     # inspect the archive layout first
scripts/fetch_binkit.sh normal
python scripts/build_corpus.py --root data/binkit/normal --out data/corpus/binkit_x86_64 \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3
```

The Google Drive file IDs come from the BinKit README.

**The actual layout** (confirmed against all 67,680 entries) puts the toolchain
in the *filename*, not in directories:

```
gnu_debug/a2ps/a2ps-4.14_clang-7.0_x86_64_O2_a2ps.elf
          |    {package}-{version}_{compiler}-{ver}_{arch}_{opt}_{program}
          `-- package directory
```

Two consequences that are easy to get wrong, both handled but worth knowing if
you adapt this code:

- Extraction patterns must be **compiler-before-architecture**. A pattern like
  `*x86_64*gcc*` matches *nothing*, because the compiler field comes first.
- The program name has to be parsed out of the filename
  (`binprov.discover.program_name`). Using the whole filename would make every
  compiled variant look like a distinct program and silently defeat the
  program-grouped split — letting a binary train at O2 and be tested at O3.

Still run `--list` and `--dry-run` before a long build: BinKit's naming has
varied between releases, label parsing is scan-based rather than template-based
so that a change degrades into something visible, and `build_corpus.py`
cross-checks the path-derived compiler against each ELF's `.comment` section and
warns if too many disagree.

One quirk of that cross-check: a Clang-built BinKit binary's `.comment` holds
*both* a GCC fingerprint (from the cross-toolchain's startup objects) and a Clang
one. `elf.guess_compiler` checks Clang first for exactly this reason.

### 2. Local build (for validating the pipeline)

```bash
# this machine has gcc but no clang; get one first
bash scripts/setup_toolchains.sh          # or see the note below
export PATH="$PWD/data/cache/llvm14/bin:$PATH"

python scripts/build_local_dataset.py --out data/local \
    --source synthetic gnu --synthetic-count 240 --synthetic-packages 12
python scripts/build_corpus.py --root data/local --out data/corpus/local \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3
```

Sources: `synthetic` (generated C, offline, deterministic), `gnu` (small
autotools packages from ftp.gnu.org), `algorithms` (TheAlgorithms/C, closest to
the paper's algorithm dataset). Build trees are deleted after the executables are
copied out.

**Do not report numbers from this dataset as reproductions.** Two compiler
versions on one architecture is a much narrower problem than BinKit's 51 packages
across 4 architectures, and conda/LLVM-supplied Clang links against a different
sysroot than the system GCC, which gives the model startup-code shortcuts that a
consistently built dataset would not.

#### Getting Clang on this machine

`bash scripts/setup_toolchains.sh` handles it, and is worth a note on *why* it
does what it does.

`conda` cannot reach its Artifactory mirror here: the corporate CA certificate is
rejected by OpenSSL 3 (`Basic Constraints of CA cert not marked critical`).
Disabling certificate verification would get around that and is not worth doing,
so the script fetches an upstream LLVM release with `curl` instead, which works
with the `CURL_CA_BUNDLE` already exported in the shell profile.

It extracts only the driver and its resource headers — a working compiler in
247 MB rather than the ~2.5 GB full tree — and deletes the 600 MB tarball
afterwards. It pins the Ubuntu 18.04 build deliberately: that targets glibc 2.27
and this host has 2.28, whereas newer LLVM releases ship only 22.04 builds
needing glibc 2.35, which will not run here.

### 3. Your own binaries

Any tree whose paths carry the provenance works, since discovery is
scan-based. The layout the other two paths use is:

```
<root>/<package>/<arch>-<compiler>-<version>-<opt>/<program>
```

Underscores work as well as dashes, and component order does not matter.
