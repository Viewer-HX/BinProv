# Data preparation

The reported experiments use the x86_64 subset of BinKit Normal with
GCC 8.2.0, Clang 7.0, and optimization levels O0–O3. Data under `data/` is
ignored by Git.

## 1. Download and extract BinKit

Install the download and archive tools first:

```bash
pip install gdown
# Install bsdtar/libarchive with your operating system's package manager.
```

From the repository root, download and extract only x86_64 binaries:

```bash
KEEP_ARCHES=x86_64 bash scripts/fetch_binkit.sh normal
```

The script downloads the BinKit archive from Google Drive and writes the
selected ELF files under:

```text
data/binkit/normal/
```

It selects GCC 8.2.0 and Clang 7.0 by default. To inspect an existing archive
without extracting it, run:

```bash
KEEP_ARCHES=x86_64 bash scripts/fetch_binkit.sh normal --list
```

## 2. Build the packed corpus

Check the selected binaries first:

```bash
python scripts/build_corpus.py \
  --root data/binkit/normal \
  --dry-run \
  --arch x86_64 \
  --compiler gcc clang \
  --opt O0 O1 O2 O3 \
  --extra normal
```

Then build the corpus used by the committed configurations:

```bash
python scripts/build_corpus.py \
  --root data/binkit/normal \
  --out data/corpus/binkit_x86_64 \
  --arch x86_64 \
  --compiler gcc clang \
  --opt O0 O1 O2 O3 \
  --extra normal \
  --workers 8
```

Once this command succeeds, training reads only the packed corpus. The
extracted ELF files may be deleted:

```bash
rm -rf data/binkit/normal
```

## Directory structure

During preparation, the relevant paths are:

```text
data/
├── binkit/
│   └── normal/                    downloaded and extracted ELF files
└── corpus/
    └── binkit_x86_64/
        ├── text.u8                concatenated .text bytes
        ├── binaries.jsonl.gz      offsets, labels, and function metadata
        ├── meta.json              corpus counts and build parameters
        └── splits/
            └── default.json       train, test, and pretrain binary IDs
```

The released training configurations expect exactly:

```text
data/corpus/binkit_x86_64/
```

Sequence windows are created from `text.u8` while training, so changing the
window length or stride does not require rebuilding the corpus.

## Expected output

For the selection above, `meta.json` should report 1,880 binaries and
220,741,868 bytes of `.text` data, balanced across GCC/Clang and O0/O1/O2/O3.
`splits/default.json` should contain 1,504 train, 376 test, and 336 pretraining
binary IDs.

It is valid for `data/binkit/` to be empty after the corpus has been built and
the extracted source binaries have been removed. Training and evaluation need
only `data/corpus/binkit_x86_64/`.
