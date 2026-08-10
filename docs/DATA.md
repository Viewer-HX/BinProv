# Data preparation

BinProv consumes raw bytes from the ELF `.text` section and, when available,
function boundaries from the symbol table. The corpus builder packs only the
information required by downstream training and evaluation.

## BinKit configuration

The main reconstructed experiment uses BinKit Normal with:

| Field | Selection |
|---|---|
| Architectures | x86_64, x86_32, arm_64, mips_64 |
| Compiler families | GCC and Clang |
| Compiler versions | GCC 8.2.0 and Clang 7.0 |
| Optimization levels | O0, O1, O2, O3 |
| Variant | normal |

The default settings in `scripts/fetch_binkit.sh` select this subset. Restrict
the extraction further with environment variables:

```bash
KEEP_ARCHES="x86_64" scripts/fetch_binkit.sh normal
KEEP_ARCHES="arm_64 mips_64" scripts/fetch_binkit.sh normal
```

## Download and inspect

Install the downloader and ensure `bsdtar`/libarchive is available:

```bash
pip install gdown
scripts/fetch_binkit.sh normal --list
scripts/fetch_binkit.sh normal
```

`--list` shows the archive layout before extraction. The script validates the
archive type, selects matching toolchains, and places extracted files under:

```text
data/binkit/normal/
```

If the dataset has already been downloaded, place the archive under
`data/binkit/` using the dataset name and a supported archive extension.

## Expected storage

Measured sizes for the normal dataset are:

| Stage | Approximate size |
|---|---:|
| Downloaded archive | 3.5 GB |
| Extracted four-architecture subset | 4.5 GB |
| Extracted x86_64 binaries | 1.1 GB |
| Packed x86_64 corpus | 211 MB |

Allow approximately 8 GB of temporary free space when downloading and
extracting the default four-architecture subset. Once the packed corpus has been
verified, downstream training no longer needs the original ELF files.

## Filename layout and labels

The distributed BinKit files encode provenance in their names. A typical path
is:

```text
gnu_debug/a2ps/a2ps-4.14_clang-7.0_x86_64_O2_a2ps.elf
```

The corpus builder extracts:

- package and program identity;
- compiler family and version;
- target architecture;
- optimization level;
- experiment variant.

Label parsing is scan-based so that common directory and filename layouts are
both supported. Always run a dry run before building a new corpus:

```bash
python scripts/build_corpus.py \
    --root data/binkit/normal \
    --dry-run \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal
```

Review the reported counts and compiler cross-check warnings before continuing.

## Build a packed corpus

```bash
python scripts/build_corpus.py \
    --root data/binkit/normal \
    --out data/corpus/binkit_x86_64 \
    --arch x86_64 \
    --compiler gcc clang \
    --opt O0 O1 O2 O3 \
    --extra normal
```

The resulting layout is:

```text
data/corpus/binkit_x86_64/
  text.u8
  binaries.jsonl.gz
  meta.json
  splits/default.json
```

- `text.u8` concatenates every selected `.text` section without padding.
- `binaries.jsonl.gz` records offsets, labels, package/program identities, and
  function boundaries.
- `meta.json` records counts and the exact build filters.
- `splits/default.json` stores train, test, and pre-training binary IDs.

Sequences are generated from the memory-mapped `text.u8` file at runtime rather
than stored separately. Sequence length and stride can therefore be changed
without rebuilding the corpus.

## Split policy

The default split is grouped by source program. Every compiler and optimization
variant of the same program is assigned to the same side of the train/test
boundary. This prevents compiled variants of one program from appearing in both
sets.

The saved split also includes a pre-training subset with at least one program
per package, entirely inside the training split. The stored full-scale
validation run uses all training binaries via `--split-name train`. The same
split file is reused by all task-specific runs.

## Corpus-size controls

Useful `build_corpus.py` options include:

- `--per-toolchain-limit N`: cap binaries per compilation configuration.
- `--packages ...`: keep selected packages.
- `--arch`, `--compiler`, `--compiler-version`, `--opt`, `--extra`: apply
  provenance filters.
- `--no-func-names`: keep function boundaries without symbol names.
- `--no-functions`: omit function metadata when only sequence/binary evaluation
  is needed.
- `--max-text-bytes N`: limit bytes per binary for development runs.

For publishable measurements, record all filters from `meta.json` and avoid
changing the corpus selection between task-specific runs.

## Local validation dataset

A small local corpus can be generated without BinKit:

```bash
python scripts/build_local_dataset.py \
    --out data/local \
    --source synthetic gnu

python scripts/build_corpus.py \
    --root data/local \
    --out data/corpus/local \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3
```

This path is intended for installation checks, tests, and development. Use the
BinKit configuration for measurements compared with the paper.

## Custom binaries

Custom datasets can use a directory layout such as:

```text
<root>/<package>/<arch>-<compiler>-<version>-<opt>/<program>
```

For example:

```text
dataset/coreutils/x86_64-gcc-8.2.0-O2/ls
```

Underscores and dashes are both supported. Use `--dry-run` to confirm that every
required label is discovered before building the corpus.
