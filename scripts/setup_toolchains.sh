#!/usr/bin/env bash
# Install Clang, so that the compiler-identification task has two classes.
#
# This host ships GCC but no Clang, and the paper's Table 3/6 task is GCC vs
# Clang, so a locally built dataset needs both.
#
# It fetches an upstream LLVM release with curl rather than using conda: conda
# cannot reach its Artifactory mirror here, because the corporate CA certificate
# is rejected by OpenSSL 3 ("Basic Constraints of CA cert not marked critical").
# curl, which uses $CURL_CA_BUNDLE from the shell profile, works fine. Disabling
# certificate verification would also "work" and is not worth doing.
#
# Only the driver and its resource headers are extracted -- 247 MB instead of the
# ~2.5 GB full tree. Everything lands under /data; home is a small shared volume.
#
# Usage:
#   bash scripts/setup_toolchains.sh
#   LLVM_VERSION=13.0.0 bash scripts/setup_toolchains.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LLVM_VERSION="${LLVM_VERSION:-14.0.0}"
# The 18.04 build targets glibc 2.27; this host has 2.28, so it runs. Newer
# LLVM releases only ship 22.04 builds (glibc 2.35), which will NOT run here.
UBUNTU="${UBUNTU:-18.04}"
CACHE="${CACHE:-data/cache}"
DEST="$CACHE/llvm${LLVM_VERSION%%.*}"

if [[ -x "$DEST/bin/clang" ]]; then
  echo "clang already present at $DEST/bin/clang"
  "$DEST/bin/clang" --version | head -1
  echo
  echo "export PATH=\"$PWD/$DEST/bin:\$PATH\""
  exit 0
fi

STEM="clang+llvm-$LLVM_VERSION-x86_64-linux-gnu-ubuntu-$UBUNTU"
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/$STEM.tar.xz"
TARBALL="$CACHE/$STEM.tar.xz"

mkdir -p "$CACHE" "$DEST"

if [[ ! -f "$TARBALL" ]]; then
  echo "=== downloading $STEM (~600 MB) ==="
  if ! curl -fL --progress-bar -o "$TARBALL.part" "$URL"; then
    echo "download failed. Check that this release exists:" >&2
    echo "  https://github.com/llvm/llvm-project/releases/tag/llvmorg-$LLVM_VERSION" >&2
    rm -f "$TARBALL.part"
    exit 1
  fi
  mv "$TARBALL.part" "$TARBALL"
fi

echo "=== extracting driver + resource headers only ==="
tar -xf "$TARBALL" -C "$DEST" --strip-components=1 \
  "$STEM/bin/clang" \
  "$STEM/bin/clang-${LLVM_VERSION%%.*}" \
  "$STEM/lib/clang"

rm -f "$TARBALL"
echo "reclaimed the tarball; $DEST is $(du -sh "$DEST" | cut -f1)"

"$DEST/bin/clang" --version | head -1

cat <<EOF

Put it on PATH for the dataset build:

  export PATH="$PWD/$DEST/bin:\$PATH"

Then:

  python scripts/build_local_dataset.py --out data/local --source synthetic gnu

A caveat on realism: this Clang targets a different sysroot than the system GCC,
so a GCC-vs-Clang model trained on locally built binaries can key on startup and
libc differences as well as on codegen. Fine for validating the pipeline; use
BinKit, whose toolchains are built consistently, for numbers you intend to report.
EOF
