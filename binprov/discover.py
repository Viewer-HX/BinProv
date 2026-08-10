"""Find labelled binaries in a directory tree.

Two layouts are handled, because the two data paths differ:

**BinKit** puts the whole toolchain in the *filename*, two levels deep::

    gnu_debug/a2ps/a2ps-4.14_clang-7.0_x86_64_O2_a2ps.elf
              |    |         |         |       |  `-- program
              |    |         |         |       `-- optimization level
              |    |         |         `-- architecture
              |    |         `-- compiler and version
              |    `-- package and version
              `-- package directory

**scripts/build_local_dataset.py** puts it in a directory instead::

    <root>/<package>/<arch>-<compiler>-<version>-<opt>/<program>

Labels come from :func:`binprov.provenance.parse_labels`, which scans every path
component for recognisable tokens and so copes with both. Verified against all
67,680 paths of BinKit Normal: architecture, compiler, compiler version and
optimization level are recovered correctly for every one.

The *program name* needs more care than the labels do. Under the BinKit layout a
naive ``path.name`` makes each compiled variant look like a different program,
which would silently defeat the program-grouped train/test split — the same
binary at O2 could train while its O3 twin is tested. :func:`program_name`
therefore parses the filename grammar to recover the real program.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import elf
from .provenance import Labels, parse_labels, toolchain_key

#: never worth parsing — build artefacts, sources, docs
SKIP_SUFFIXES = frozenset(
    """.c .h .cc .cpp .hpp .py .sh .m4 .am .in .ac .md .txt .log .json .yaml .yml
    .gz .xz .bz2 .zip .tar .patch .diff .po .pot .1 .3 .html .css .js .png .jpg
    .o .lo .la .a .cmake .pc .mk .d .gmo .info .texi""".split()
)


@dataclass
class Candidate:
    """A file on disk that looks like a labelled binary."""

    path: Path
    package: str
    program: str
    labels: Labels

    @property
    def toolchain(self) -> str:
        return toolchain_key(self.labels.as_dict())


def _looks_like_source_tree(name: str) -> bool:
    return name in {".git", ".svn", "autom4te.cache", "__pycache__", "node_modules"}


# BinKit's filename grammar:
#   {package}-{version}_{compiler}-{cver}_{arch}_{width}_{opt}_{program}
# The program is the trailing group, so names containing underscores survive
# (gdbm_dump, gdbm_load are the two real cases in BinKit Normal).
_BINKIT_STEM = re.compile(
    r"^(?P<pkgver>.+?)"
    r"_(?P<compiler>(?:gcc|clang|g\+\+|clang\+\+)-[0-9][0-9.]*)"
    r"_(?P<arch>(?:x86|x64|i386|arm|aarch64|mips|mipseb|mipsel|ppc|riscv)_(?:32|64))"
    r"_(?P<opt>O[0-3s]|Ofast)"
    r"_(?P<program>.+)$"
)


def program_name(path: Path) -> str:
    """The program a binary was built from, stripped of provenance tokens.

    Returns the trailing program field for a BinKit-style filename, and the plain
    filename otherwise (the directory-based layout, where the filename already
    *is* the program). This is the grouping key for leak-free splits, so getting
    it wrong does not fail loudly — it just quietly inflates accuracy.
    """
    stem = path.name
    for suffix in (".elf", ".exe", ".so"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    m = _BINKIT_STEM.match(stem)
    return m.group("program") if m else path.name


def walk(root: str | os.PathLike, *, follow_symlinks: bool = False):
    """Yield paths of ELF files under ``root``, cheaply.

    Filters by extension first and reads only 4 bytes to confirm the ELF magic,
    so walking a build tree full of objects and sources stays fast.
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if not _looks_like_source_tree(d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in SKIP_SUFFIXES:
                continue
            if p.is_symlink() and not follow_symlinks:
                continue
            try:
                if p.stat().st_size < 128:
                    continue
            except OSError:
                continue
            if elf.is_elf(p):
                yield p


def discover(
    root: str | os.PathLike,
    *,
    arch=None,
    compiler=None,
    compiler_version=None,
    opt=None,
    extra=None,
    obfuscation=None,
    require_complete: bool = True,
    per_toolchain_limit: int | None = None,
    packages=None,
):
    """Yield :class:`Candidate` objects for binaries matching the filters.

    Filters accept a scalar or a collection; ``None`` means "no constraint".
    They are applied *before* the file is read, which is the whole point — the
    paper uses 4 architectures x 2 compilers x 4 opt levels, i.e. 32 of BinKit
    Normal's 288 toolchains, so filtering here avoids touching ~89% of the
    dataset.

    Args:
        per_toolchain_limit: keep at most this many binaries per compilation
            configuration. The cheapest way to bound corpus size while staying
            balanced across configurations.
        packages: restrict to these package names.
    """
    root = Path(root)
    want = {
        "arch": arch,
        "compiler": compiler,
        "compiler_version": compiler_version,
        "opt": opt,
        "extra": extra,
        "obfuscation": obfuscation,
    }
    pkg_filter = None if packages is None else set(packages)
    seen_per_toolchain: Counter[str] = Counter()

    for path in walk(root):
        rel = path.relative_to(root)
        parts = rel.parts
        package = parts[0] if len(parts) > 1 else "_root"
        if pkg_filter is not None and package not in pkg_filter:
            continue

        labels = parse_labels(str(rel))
        if require_complete and not labels.is_complete():
            continue

        d = labels.as_dict()
        if not _matches(d, want):
            continue

        if per_toolchain_limit is not None:
            key = toolchain_key(d)
            if seen_per_toolchain[key] >= per_toolchain_limit:
                continue
            seen_per_toolchain[key] += 1

        yield Candidate(
            path=path, package=package, program=program_name(path), labels=labels
        )


def _matches(got: dict, want: dict) -> bool:
    for key, expected in want.items():
        if expected is None:
            continue
        value = got.get(key)
        if isinstance(expected, (list, tuple, set, frozenset)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def summarize(candidates) -> str:
    """A short report of what discovery found, for ``--dry-run``."""
    cands = list(candidates)
    if not cands:
        return "no binaries matched"
    by_tc = Counter(c.toolchain for c in cands)
    by_pkg = Counter(c.package for c in cands)
    lines = [
        f"{len(cands)} binaries, {len(by_pkg)} packages, {len(by_tc)} toolchains",
        "",
        "toolchains (count):",
    ]
    for tc, n in sorted(by_tc.items()):
        lines.append(f"  {n:7d}  {tc}")
    lines += ["", "sample paths:"]
    for c in cands[:8]:
        lines.append(f"  {c.package} | {c.program} | {c.toolchain}")
    return "\n".join(lines)
