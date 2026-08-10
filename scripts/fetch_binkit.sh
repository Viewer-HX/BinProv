#!/usr/bin/env bash
# Download BinKit and extract only the toolchains BinProv needs.
#
# Space is the reason this script exists rather than a one-line download-and-
# unpack. BinKit Normal is 67,680 debug-built binaries averaging ~600 KB: a 3.5 GB
# archive that expands to roughly 40 GB across 288 toolchains. The defaults below
# keep 4 architectures x 1 version per compiler family x 4 optimization levels --
# about 11% of it. Extracting selectively, packing to a corpus, then deleting the
# binaries keeps the steady state at ~211 MB per architecture.
#
# Stages, with peak disk at each point (measured):
#   1. download archive                       ~= 3.5 GB for Normal
#   2. selective extract (7,520 binaries)     += ~4.5 GB
#   3. delete archive                         -= 3.5 GB
#   4. build corpus (.text only)              += ~211 MB per architecture
#   5. optional: purge extracted binaries     -= ~4.5 GB
#
# Measured on this host: Google Drive delivers ~40 MB/s through the proxy, so the
# download is ~1.5 minutes and is not the bottleneck. Decompression is.
#
# Usage:
#   scripts/fetch_binkit.sh normal            # download + extract the paper subset
#   scripts/fetch_binkit.sh normal --list     # just show the archive layout
#   scripts/fetch_binkit.sh obfus             # the obfuscation set (Table 7)
#   KEEP_ARCHES="x86_64" scripts/fetch_binkit.sh normal   # narrow it further
#
# Requires `gdown` (pip install gdown) because the datasets are hosted on Google
# Drive, and `bsdtar` (libarchive) which reads .7z as well as .tar.*. If you
# already have the archive, drop it at $DATA_ROOT/<name>.<ext> to skip the
# download.

set -euo pipefail

# --- Google Drive ids, taken from the BinKit README (paper-era datasets) ------
declare -A GDRIVE_IDS=(
  [normal]=1K9ef-OoRBr0X5u8g2mlnYqh9o1i6zFij
  [sizeopt]=1QgwbEfd8vdzg5glNZFL7dg4l4hrkoWO3
  [noinline]=1wt7GY-DDp8J_2zeBBVUrcfWIyerg_xLO
  [pie]=1IfEbnS9RtHhVhW8oiqnE7G75uPej1FPx
  [lto]=1Tsd-WNO_JDlEX0GylBOxsFjOPUmUyeGh
  [obfus]=1H5k3pfJH9zN4anfxKi1WvNqTKmjVjUUU
)

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data/binkit}"
# Toolchains to keep: the paper's four architectures, all optimization levels.
#
# Compilers are pinned to ONE version per family on purpose. BinKit Normal ships
# 5 GCC and 4 Clang versions (67,680 binaries in total); taking all of them would
# put 9 different compilers into a 2-class "GCC vs Clang" task and multiply the
# corpus ~4.5x. gcc-8.2.0 and clang-7.0 are the versions named in the paper's
# own Figure 2 legend, and they give 7,520 binaries across four architectures
# (1,880 for x86_64 alone) — the closest match to Table 2's 6,280.
#
# Set KEEP_COMPILERS="gcc clang" to take every version instead.
: "${KEEP_ARCHES:=x86_64 x86_32 arm_64 mips_64}"
: "${KEEP_COMPILERS:=gcc-8.2.0 clang-7.0}"

NAME="${1:-}"
shift || true
LIST_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --list) LIST_ONLY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "$NAME" || -z "${GDRIVE_IDS[$NAME]:-}" ]]; then
  echo "usage: $0 {${!GDRIVE_IDS[*]}} [--list]" >&2
  exit 2
fi

# bsdtar handles .tar.gz, .tar.xz and .7z with one wildcard syntax. Everything
# below depends on that, so check up front rather than failing after a 3.5 GB
# download.
if ! command -v bsdtar >/dev/null; then
  echo "bsdtar (libarchive) not found -- needed to read BinKit's .7z archives." >&2
  echo "Install libarchive, or extract manually with:" >&2
  echo "  pip install py7zr && python -m py7zr x <archive>.7z" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "=== disk before ==="
df -h "$DATA_ROOT" | tail -1

# --- 1. download -------------------------------------------------------------
ARCHIVE=""
for cand in "$NAME".7z "$NAME".tar.gz "$NAME".tar.xz "$NAME".tar "$NAME".tar.zst; do
  [[ -f "$cand" ]] && ARCHIVE="$cand" && break
done

if [[ -z "$ARCHIVE" ]]; then
  if ! command -v gdown >/dev/null; then
    echo "gdown not found. Install it (pip install gdown) or place the archive at" >&2
    echo "  $DATA_ROOT/$NAME.7z" >&2
    echo "Drive link: https://drive.google.com/file/d/${GDRIVE_IDS[$NAME]}/view" >&2
    exit 1
  fi
  echo "=== downloading $NAME from Google Drive (~3.5 GB for normal) ==="
  # Large Drive files need the confirm-token dance; gdown handles it.
  gdown "${GDRIVE_IDS[$NAME]}" -O "$NAME.download"
  # The Drive filename is not exposed reliably, so identify the container.
  case "$(file -b --mime-type "$NAME.download")" in
    application/x-7z-compressed) ARCHIVE="$NAME.7z" ;;
    application/gzip)            ARCHIVE="$NAME.tar.gz" ;;
    application/x-xz)            ARCHIVE="$NAME.tar.xz" ;;
    application/zstd)            ARCHIVE="$NAME.tar.zst" ;;
    application/x-tar)           ARCHIVE="$NAME.tar" ;;
    *)
      echo "unrecognised archive type:" >&2
      file -b "$NAME.download" >&2
      echo "(a small HTML file here usually means a Drive quota error)" >&2
      exit 1
      ;;
  esac
  mv "$NAME.download" "$ARCHIVE"
fi

echo "archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

# --- 2. inspect layout -------------------------------------------------------
# BinKit's directory template has changed between releases, so look before
# extracting rather than assuming a path shape. Note 7z keeps its index at the
# end of the file, so this needs the whole archive present -- it cannot stream.
echo "=== archive layout (first 25 entries) ==="
bsdtar -tf "$ARCHIVE" 2>/dev/null | head -25 || true

# Auto-detect a single wrapping directory instead of hard-coding
# --strip-components=1, which silently mangles paths when the archive has none.
TOPS=$(bsdtar -tf "$ARCHIVE" 2>/dev/null | head -2000 | cut -d/ -f1 | sort -u | wc -l)
STRIP=0
if [[ "$TOPS" -eq 1 ]]; then
  STRIP=1
  echo "(single top-level directory detected -> --strip-components=1)"
fi

if [[ "$LIST_ONLY" == 1 ]]; then
  echo
  echo "--list: stopping before extraction."
  echo "Check the paths above, then set KEEP_ARCHES / KEEP_COMPILERS if the"
  echo "naming differs from the defaults ($KEEP_ARCHES | $KEEP_COMPILERS)."
  exit 0
fi

# --- 3. selective extract ----------------------------------------------------
DEST="$DATA_ROOT/$NAME"
mkdir -p "$DEST"

PATTERNS=()
for a in $KEEP_ARCHES; do
  for c in $KEEP_COMPILERS; do
    # Order matters and it is compiler-before-architecture. BinKit encodes the
    # toolchain in the filename as
    #   {pkg}-{ver}_{compiler}-{cver}_{arch}_{opt}_{program}.elf
    # so a pattern like "*x86_64*gcc*" matches nothing at all. Verified against
    # the real listing of all 67,680 entries.
    # A KEEP_COMPILERS entry may be a bare family ("gcc") or a pinned version
    # ("gcc-8.2.0"); the trailing "*" after $c covers both.
    PATTERNS+=("*_${c}*_${a}_O*")
  done
done

echo "=== extracting subset into $DEST ==="
echo "patterns: ${PATTERNS[*]}"
echo "(7z uses solid compression, so this may decompress more than it writes;"
echo " the saving is in disk, not necessarily in time)"
# bsdtar exits non-zero when a pattern matches nothing, which is expected --
# not every arch/compiler pair exists in every dataset.
bsdtar -xf "$ARCHIVE" -C "$DEST" --strip-components="$STRIP" "${PATTERNS[@]}" || true

N_FILES=$(find "$DEST" -type f | wc -l)
if [[ "$N_FILES" -eq 0 ]]; then
  echo "extracted nothing -- the patterns did not match the archive layout." >&2
  echo "Re-run with --list, then set KEEP_ARCHES/KEEP_COMPILERS to match." >&2
  exit 1
fi
echo "extracted $N_FILES files, $(du -sh "$DEST" | cut -f1)"

# --- 4. drop the archive -----------------------------------------------------
ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)
if [[ "${PURGE_ARCHIVE:-ask}" == "yes" ]]; then
  rm -f "$ARCHIVE"
  echo "deleted $ARCHIVE (PURGE_ARCHIVE=yes)"
elif [[ ! -t 0 || "${PURGE_ARCHIVE:-ask}" == "no" ]]; then
  # Non-interactive (a nohup'd or CI run): never guess on a destructive step.
  # Keeping it also avoids re-downloading when extracting another architecture.
  echo "kept $ARCHIVE ($ARCHIVE_SIZE)"
  echo "  delete it with: rm -f $DATA_ROOT/$ARCHIVE"
  echo "  or re-run with PURGE_ARCHIVE=yes"
else
  read -r -p "Delete the archive $ARCHIVE to reclaim $ARCHIVE_SIZE? [y/N] " ans
  if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
    rm -f "$ARCHIVE"
    echo "deleted $ARCHIVE"
  else
    echo "kept $ARCHIVE"
  fi
fi

echo "=== disk after ==="
df -h "$DATA_ROOT" | tail -1

cat <<EOF

Next: pack .text into a corpus (this is what shrinks the footprint), e.g.

  python scripts/build_corpus.py --root $DEST --dry-run \\
      --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal

  python scripts/build_corpus.py --root $DEST --out data/corpus/${NAME}_x86_64 \\
      --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal

Once the corpus exists the extracted binaries are no longer needed:

  rm -rf $DEST     # or pass --purge-source --yes to build_corpus.py
EOF
