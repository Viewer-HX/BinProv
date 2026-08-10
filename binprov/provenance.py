"""Provenance labels and the classification tasks defined on them.

The paper decomposes provenance identification into independent sub-tasks
(§3.3) rather than predicting the 8-way compiler×optimization product, because
"specialized classifiers have better performance". Each entry in :data:`TASKS`
is one such sub-task: which binaries it keeps, and how a kept binary maps to a
class index.

Tasks (with the paper table they feed):

    compiler   GCC vs Clang                            Tables 3, 6, 7
    opt_hl     low (O0/O1) vs high (O2/O3)             Tables 3, 6
    opt4       O0 vs O1 vs O2 vs O3                    Tables 4, 5, 7
    opt_o0o1   O0 vs O1  (only O0/O1 binaries)         Table 4
    opt_o2o3   O2 vs O3  (only O2/O3 binaries)         Tables 4, 6, 7
    arch       instruction set architecture            §5.1(d)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

COMPILERS = ("gcc", "clang")
OPT_LEVELS = ("O0", "O1", "O2", "O3")
#: architectures present in BinKit; the paper evaluates the first four
ARCHES = (
    "x86_64",
    "x86_32",
    "arm_64",
    "arm_32",
    "mips_64",
    "mips_32",
    "mipseb_64",
    "mipseb_32",
)

# ---------------------------------------------------------------------------
# label extraction from a path
# ---------------------------------------------------------------------------

# BinKit's on-disk layout has changed between releases (and mirrors repack it),
# so rather than hard-coding one directory template we scan every path
# component for recognisable tokens. `scripts/build_corpus.py --dry-run` prints
# what this produced so a new layout can be spotted before a long build.
_ARCH_RE = re.compile(
    r"(?<![a-z0-9])(x86|x64|i386|arm|aarch64|mips|mipseb|mipsel|ppc|riscv)"
    r"(?:[-_]?(32|64))?(?![a-z0-9])",
    re.I,
)
_COMPILER_RE = re.compile(
    r"(?<![a-z])(gcc|g\+\+|clang|clang\+\+)(?:[-_]?([0-9]+(?:\.[0-9]+){0,2}))?(?![a-z])", re.I
)
_OPT_RE = re.compile(r"(?<![a-z0-9])-?O(0|1|2|3|s|fast)(?![a-z0-9])")
_EXTRA_RE = re.compile(
    r"(?<![a-z])(normal|nopie|pie|noinline|lto|sizeopt|obfus(?:2loop)?|bcf|fla|sub|all)(?![a-z])",
    re.I,
)
# Obfuscator-LLVM transforms, reported separately in Table 7
_OBF_TOKENS = {"bcf", "fla", "sub", "all"}


@dataclass
class Labels:
    """Everything we know about how one binary was produced."""

    arch: str | None = None
    compiler: str | None = None
    compiler_version: str | None = None
    opt: str | None = None
    extra: str = "normal"  # normal | pie | lto | noinline | sizeopt | ...
    obfuscation: str | None = None  # bcf | fla | sub | all

    def is_complete(self) -> bool:
        """True when the binary can be used for every task in :data:`TASKS`."""
        return bool(self.arch and self.compiler and self.opt)

    def as_dict(self) -> dict:
        return {
            "arch": self.arch,
            "compiler": self.compiler,
            "compiler_version": self.compiler_version,
            "opt": self.opt,
            "extra": self.extra,
            "obfuscation": self.obfuscation,
        }


def _norm_arch(base: str, width: str | None) -> str:
    base = base.lower()
    if base in ("x64",):
        return "x86_64"
    if base in ("i386",):
        return "x86_32"
    if base == "aarch64":
        return "arm_64"
    if base == "mipsel":
        base = "mips"
    if width:
        return f"{base}_{width}"
    # widths that are implied by the architecture name alone
    return {"arm": "arm_32", "x86": "x86_32", "mips": "mips_32"}.get(base, base)


def parse_labels(path: str) -> Labels:
    """Derive provenance labels by scanning the components of ``path``.

    Later components win, so a per-binary directory overrides a package-level
    one. Compiler family/version are only taken together, to avoid pairing
    "gcc" from one component with a version number from another.
    """
    lab = Labels()
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    for part in parts:
        if m := _ARCH_RE.search(part):
            lab.arch = _norm_arch(m.group(1), m.group(2))
        if m := _COMPILER_RE.search(part):
            fam = m.group(1).lower().rstrip("+")
            lab.compiler = "gcc" if fam in ("gcc", "g") else "clang"
            if m.group(2):
                lab.compiler_version = m.group(2)
        if m := _OPT_RE.search(part):
            lab.opt = "O" + m.group(1)
        # finditer, not search: a component can carry both the family and the
        # specific transform, e.g. "x86_64-clang-obfus-sub-O2"
        for m in _EXTRA_RE.finditer(part):
            tok = m.group(1).lower()
            if tok in _OBF_TOKENS:
                lab.obfuscation = tok
                lab.extra = "obfus"
            elif tok.startswith("obfus"):
                lab.extra = "obfus"
            else:
                lab.extra = tok
    return lab


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """A classification task: a filter plus a label map."""

    name: str
    classes: tuple[str, ...]
    keep: Callable[[dict], bool]
    to_class: Callable[[dict], str]
    note: str = ""  # which paper table this task feeds

    @property
    def num_labels(self) -> int:
        return len(self.classes)

    def label_of(self, labels: dict) -> int | None:
        """Class index for a binary, or None if it is out of scope."""
        if not self.keep(labels):
            return None
        cls = self.to_class(labels)
        try:
            return self.classes.index(cls)
        except ValueError:
            return None


def _opt_in(*levels: str) -> Callable[[dict], bool]:
    allowed = set(levels)
    return lambda d: d.get("opt") in allowed


TASKS: dict[str, Task] = {
    "compiler": Task(
        name="compiler",
        classes=COMPILERS,
        keep=lambda d: d.get("compiler") in COMPILERS,
        to_class=lambda d: d["compiler"],
        note="Table 3/6/7: GCC vs Clang",
    ),
    "opt_hl": Task(
        name="opt_hl",
        classes=("low", "high"),
        keep=_opt_in(*OPT_LEVELS),
        to_class=lambda d: "low" if d["opt"] in ("O0", "O1") else "high",
        note="Table 3/6: O0,O1 vs O2,O3",
    ),
    "opt4": Task(
        name="opt4",
        classes=OPT_LEVELS,
        keep=_opt_in(*OPT_LEVELS),
        to_class=lambda d: d["opt"],
        note="Table 4/5/7: O0/O1/O2/O3",
    ),
    "opt_o0o1": Task(
        name="opt_o0o1",
        classes=("O0", "O1"),
        keep=_opt_in("O0", "O1"),
        to_class=lambda d: d["opt"],
        note="Table 4: O0 vs O1",
    ),
    "opt_o2o3": Task(
        name="opt_o2o3",
        classes=("O2", "O3"),
        keep=_opt_in("O2", "O3"),
        to_class=lambda d: d["opt"],
        note="Table 4/6/7: O2 vs O3 (the hard one)",
    ),
    "arch": Task(
        name="arch",
        classes=ARCHES,
        keep=lambda d: d.get("arch") in ARCHES,
        to_class=lambda d: d["arch"],
        note="§5.1(d): instruction set architecture",
    ),
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise SystemExit(
            f"unknown task {name!r}; available: {', '.join(sorted(TASKS))}"
        ) from None


def toolchain_key(labels: dict) -> str:
    """Stable identifier for a compilation configuration.

    Used to balance the corpus across configurations and to report per-cell
    numbers (e.g. Table 5, which splits optimization results by compiler).
    """
    parts: Sequence[str] = (
        labels.get("arch") or "?",
        labels.get("compiler") or "?",
        labels.get("compiler_version") or "-",
        labels.get("opt") or "?",
        labels.get("obfuscation") or labels.get("extra") or "normal",
    )
    return "/".join(parts)
