#!/usr/bin/env python3
"""Compile a small provenance dataset locally, in BinKit's directory shape.

Why this exists: BinKit is a multi-GB Google Drive download, and you cannot test
a pipeline you cannot run. This builds the same *kind* of data — one program
compiled by several compilers at O0/O1/O2/O3 — from sources on this machine, so
the whole pipeline can be exercised end to end in minutes.

Read this before quoting any number produced from it: a locally built dataset is
for **validating the pipeline, not reproducing the paper**. The paper's numbers
come from BinKit's 51 GNU packages across 4 architectures; a handful of native
x86_64 packages built with one GCC and one Clang version is a far easier and far
narrower problem. Use `scripts/fetch_binkit.sh` for the real thing.

Sources
-------
synthetic  Generated C programs. No network, no dependencies, deterministic.
gnu        Small GNU packages fetched from ftp.gnu.org and built with autotools.
           Realistic code; needs network and a few minutes per configuration.
algorithms Clones TheAlgorithms/C and compiles each file standalone. Closest to
           the paper's "algorithm dataset" of §4.2.

Output layout, which `binprov.discover` reads without extra configuration::

    <out>/<package>/x86_64-<compiler>-<version>-<opt>/<program>
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov import elf  # noqa: E402

# A build runs for minutes with its output redirected to a log; unbuffered
# progress is the difference between "working" and "apparently hung".
print = functools.partial(print, flush=True)

GNU_PACKAGES = {
    # name: (version, tarball url)  -- small, plain-autotools, quick to build
    "hello": ("2.12.1", "https://ftp.gnu.org/gnu/hello/hello-2.12.1.tar.gz"),
    "gzip": ("1.13", "https://ftp.gnu.org/gnu/gzip/gzip-1.13.tar.gz"),
    "sed": ("4.9", "https://ftp.gnu.org/gnu/sed/sed-4.9.tar.gz"),
    "diffutils": ("3.10", "https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz"),
    "grep": ("3.11", "https://ftp.gnu.org/gnu/grep/grep-3.11.tar.gz"),
    "which": ("2.21", "https://ftp.gnu.org/gnu/which/which-2.21.tar.gz"),
    # opt-in: many binaries per configuration, but a slower build
    "coreutils": ("9.4", "https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz"),
    "findutils": ("4.9.0", "https://ftp.gnu.org/gnu/findutils/findutils-4.9.0.tar.xz"),
}
DEFAULT_GNU = ["hello", "gzip", "sed", "which", "diffutils"]

ALGORITHMS_REPO = "https://github.com/TheAlgorithms/C.git"


@dataclass
class Toolchain:
    family: str  # gcc | clang
    version: str
    cc: str  # executable path

    @property
    def tag(self) -> str:
        return f"{self.family}-{self.version}"


def detect_toolchain(cc: str) -> Toolchain | None:
    """Identify a compiler executable, or return None if it does not run."""
    exe = shutil.which(cc)
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    low = first.lower()
    family = "clang" if "clang" in low else "gcc"
    import re

    m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", first) or re.search(r"([0-9]+\.[0-9]+)", first)
    return Toolchain(family=family, version=m.group(1) if m else "unknown", cc=exe)


# ---------------------------------------------------------------------------
# synthetic source
# ---------------------------------------------------------------------------

_TEMPLATES = [
    # Each template is parameterised by an integer so that generated programs
    # differ in constants, loop bounds and array sizes. The mix deliberately
    # spans code shapes that optimization levels treat very differently:
    # loops (vectorised at O3), recursion (inlined at O2+), switch tables,
    # float math, and struct copies.
    """
    static int loop_{k}(const int *a, int n) {{
        int acc = {k};
        for (int i = 0; i < n; i++) {{
            acc += a[i] * {m} - (a[i] >> 2);
            if (acc > 100000) acc -= 100000;
        }}
        return acc;
    }}""",
    """
    static long recur_{k}(long n) {{
        if (n <= 1) return {k};
        return recur_{k}(n - 1) + recur_{k}(n - 2) % {m};
    }}""",
    """
    static double fmath_{k}(double x, int n) {{
        double s = 0.0;
        for (int i = 1; i <= n; i++) s += x / (double)i + (double){k} / (double)(i * {m} + 1);
        return s;
    }}""",
    """
    static int switch_{k}(int op, int a, int b) {{
        switch (op % 8) {{
            case 0: return a + b + {k};
            case 1: return a - b * {m};
            case 2: return a ^ (b << 3);
            case 3: return (a | b) & {k};
            case 4: return a ? b / (a % {m} + 1) : {k};
            case 5: return a * a - b * b;
            case 6: return (a > b) ? a % ({m} + 1) : b % ({k} + 1);
            default: return a + {m};
        }}
    }}""",
    """
    struct rec_{k} {{ int id; double w; char tag[{m}]; }};
    static double sumrec_{k}(const struct rec_{k} *r, int n) {{
        double t = 0;
        for (int i = 0; i < n; i++) t += r[i].w * (double)(r[i].id % {m} + 1);
        return t;
    }}""",
    """
    static void sort_{k}(int *a, int n) {{
        for (int i = 1; i < n; i++) {{
            int key = a[i], j = i - 1;
            while (j >= 0 && a[j] > key) {{ a[j + 1] = a[j]; j--; }}
            a[j + 1] = key;
        }}
    }}""",
    """
    static unsigned hash_{k}(const char *s) {{
        unsigned h = {k}u;
        while (*s) h = h * 31u + (unsigned char)*s++;
        return h ^ (h >> {m});
    }}""",
]


def generate_synthetic(out_dir: Path, count: int, seed: int = 0) -> list[Path]:
    """Write ``count`` self-contained C programs. Deterministic given ``seed``."""
    import random

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(count):
        n_fns = rng.randint(4, len(_TEMPLATES))
        chosen = rng.sample(range(len(_TEMPLATES)), n_fns)
        bodies, calls = [], []
        for j, t_idx in enumerate(chosen):
            k = rng.randint(2, 97)
            m = rng.randint(3, 61)
            bodies.append(textwrap.dedent(_TEMPLATES[t_idx]).format(k=k, m=m))
            name = _TEMPLATES[t_idx].strip().split("{k}")[0].split()[-1].rstrip("_") + f"_{k}"
            calls.append((t_idx, name, k, m))

        src = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <math.h>"]
        src += bodies
        src.append("int main(int argc, char **argv) {")
        src.append("    int n = argc > 1 ? atoi(argv[1]) : 64; if (n < 4) n = 4; if (n > 4096) n = 4096;")
        src.append("    int *buf = malloc(sizeof(int) * n); double acc = 0;")
        src.append("    for (int i = 0; i < n; i++) buf[i] = (i * 7919) % 1021;")
        for t_idx, name, k, m in calls:
            if t_idx == 0:
                src.append(f"    acc += loop_{k}(buf, n);")
            elif t_idx == 1:
                src.append(f"    acc += (double)recur_{k}(n % 24 + 4);")
            elif t_idx == 2:
                src.append(f"    acc += fmath_{k}((double)n, n);")
            elif t_idx == 3:
                src.append(f"    for (int i = 0; i < n; i++) acc += switch_{k}(i, buf[i], i);")
            elif t_idx == 4:
                src.append(
                    f"    {{ struct rec_{k} *r = calloc(n, sizeof(*r));"
                    f" for (int i = 0; i < n; i++) {{ r[i].id = buf[i]; r[i].w = i * 0.5; }}"
                    f" acc += sumrec_{k}(r, n); free(r); }}"
                )
            elif t_idx == 5:
                src.append(f"    sort_{k}(buf, n);")
            else:
                src.append(f'    acc += hash_{k}(argv[0]);')
        src.append("    printf(\"%.3f %d\\n\", acc, buf[n / 2]);")
        src.append("    free(buf); return 0;")
        src.append("}")

        path = out_dir / f"prog{i:04d}.c"
        path.write_text("\n".join(src) + "\n")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _run(cmd, cwd=None, env=None, timeout=1800) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as exc:
        return 1, str(exc)


def _collect_binaries(tree: Path, dest: Path, *, min_text: int = 1024) -> int:
    """Copy ELF executables out of a build tree into ``dest``.

    Filters out libtool wrappers, shared objects that are just stubs, and
    anything whose ``.text`` is too small to yield a single byte sequence.
    """
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    seen: set[str] = set()
    for path in tree.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix in (".o", ".a", ".la", ".lo", ".c", ".h", ".sh"):
            continue
        if not elf.is_elf(path):
            continue
        try:
            sec = elf.parse(path.read_bytes(), want_functions=False)
        except (elf.ElfError, OSError):
            continue
        if len(sec.data) < min_text or path.name in seen:
            continue
        shutil.copy2(path, dest / path.name)
        seen.add(path.name)
        n += 1
    return n


def build_synthetic(sources: list[Path], tc: Toolchain, opt: str, out: Path, jobs: int) -> int:
    """Compile each generated program standalone."""
    out.mkdir(parents=True, exist_ok=True)

    def one(src: Path) -> bool:
        exe = out / src.stem
        rc, _ = _run([tc.cc, f"-{opt}", "-w", str(src), "-o", str(exe), "-lm"], timeout=300)
        if rc != 0:
            exe.unlink(missing_ok=True)
            return False
        return True

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return sum(pool.map(one, sources))


def build_algorithms(repo: Path, tc: Toolchain, opt: str, out: Path, jobs: int, limit: int) -> int:
    """Compile TheAlgorithms/C files standalone; most, not all, will build."""
    files = [p for p in sorted(repo.rglob("*.c")) if "test" not in p.name.lower()][:limit]
    out.mkdir(parents=True, exist_ok=True)

    def one(src: Path) -> bool:
        exe = out / src.stem
        if exe.exists():
            return False
        rc, _ = _run([tc.cc, f"-{opt}", "-w", str(src), "-o", str(exe), "-lm"], timeout=120)
        if rc != 0:
            exe.unlink(missing_ok=True)
            return False
        try:
            sec = elf.parse(exe.read_bytes(), want_functions=False)
        except elf.ElfError:
            exe.unlink(missing_ok=True)
            return False
        if len(sec.data) < 1024:
            exe.unlink(missing_ok=True)
            return False
        return True

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return sum(pool.map(one, files))


def build_gnu(tarball: Path, name: str, tc: Toolchain, opt: str, out: Path, jobs: int, workdir: Path) -> int:
    """Configure+make one GNU package at one optimization level.

    The build tree is thrown away afterwards; only the ELF executables survive.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        top = tf.getnames()[0].split("/")[0]
        # filter="data" is the safe extraction mode (Python 3.12+); fall back
        # for older interpreters
        try:
            tf.extractall(workdir, filter="data")
        except TypeError:
            tf.extractall(workdir)
    src = workdir / top

    env = dict(os.environ)
    env["CC"] = tc.cc
    # -w suppresses warnings that some packages promote to errors under Clang
    env["CFLAGS"] = f"-{opt} -w"
    env["LDFLAGS"] = ""
    rc, log = _run(
        ["./configure", "--disable-nls", "--disable-dependency-tracking", f"CC={tc.cc}",
         f"CFLAGS=-{opt} -w"],
        cwd=src,
        env=env,
    )
    if rc != 0:
        print(f"    configure failed for {name} {tc.tag} {opt}: {log.splitlines()[-1:]}")
        shutil.rmtree(src, ignore_errors=True)
        return 0
    rc, log = _run(["make", f"-j{jobs}"], cwd=src, env=env)
    if rc != 0:
        # a partial build still leaves usable binaries behind
        print(f"    make returned {rc} for {name} {tc.tag} {opt} (keeping what built)")
    n = _collect_binaries(src, out)
    shutil.rmtree(src, ignore_errors=True)
    return n


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return dest


# ---------------------------------------------------------------------------



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default="data/local", help="output root")
    ap.add_argument(
        "--source",
        nargs="+",
        default=["synthetic"],
        choices=["synthetic", "gnu", "algorithms"],
        help="which program sources to compile",
    )
    ap.add_argument("--opt", nargs="+", default=["O0", "O1", "O2", "O3"])
    ap.add_argument("--cc", nargs="+", default=["gcc", "clang"], help="compilers to try")
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--synthetic-count", type=int, default=200)
    ap.add_argument("--synthetic-seed", type=int, default=0)
    ap.add_argument(
        "--synthetic-packages",
        type=int,
        default=10,
        help="spread the generated programs over this many pseudo-packages, so "
        "that package-grouped splits and the pre-training selection behave as "
        "they would on a multi-project dataset",
    )
    ap.add_argument("--gnu-packages", nargs="+", default=DEFAULT_GNU, choices=list(GNU_PACKAGES))
    ap.add_argument("--algorithms-limit", type=int, default=400)
    ap.add_argument("--cache", default="data/cache", help="where tarballs and clones live")
    ap.add_argument("--keep-cache", action="store_true", help="do not delete downloaded tarballs")
    args = ap.parse_args()

    out_root = Path(args.out)
    cache = Path(args.cache)
    out_root.mkdir(parents=True, exist_ok=True)

    toolchains = []
    for cc in args.cc:
        tc = detect_toolchain(cc)
        if tc is None:
            print(f"skipping {cc!r}: not found on PATH")
            continue
        toolchains.append(tc)
        print(f"found {tc.family} {tc.version} at {tc.cc}")
    if not toolchains:
        print("no usable compiler found", file=sys.stderr)
        return 1
    families = {t.family for t in toolchains}
    if len(families) < 2:
        print(
            f"\nWARNING: only the {sorted(families)[0]} family is available, so the "
            "compiler-identification task (Tables 3/6) cannot be trained.\n"
            "Install the other one, e.g.:  bash scripts/setup_toolchains.sh\n"
        )

    totals: dict[str, int] = {}
    arch = "x86_64"  # native builds only; BinKit supplies the cross-compiled ones

    # ---- synthetic -------------------------------------------------------
    if "synthetic" in args.source:
        print(f"\n== synthetic: generating {args.synthetic_count} programs ==")
        gen_dir = cache / "synthetic_src"
        sources = generate_synthetic(gen_dir, args.synthetic_count, args.synthetic_seed)
        # Spread the programs over several pseudo-packages. Without this the
        # whole source is one "package", and build_corpus.py's pre-training
        # selection ("one program per package", paper §4.1) degenerates to a
        # single program.
        n_pkg = max(1, args.synthetic_packages)
        buckets: list[list[Path]] = [[] for _ in range(n_pkg)]
        for i, src in enumerate(sources):
            buckets[i % n_pkg].append(src)
        for pkg_i, bucket in enumerate(buckets):
            if not bucket:
                continue
            pkg = "synthetic" if n_pkg == 1 else f"synthetic{pkg_i:02d}"
            for tc in toolchains:
                for opt in args.opt:
                    dest = out_root / pkg / f"{arch}-{tc.tag}-{opt}"
                    n = build_synthetic(bucket, tc, opt, dest, args.jobs)
                    totals[f"{pkg}/{tc.tag}/{opt}"] = n
        built = sum(v for k, v in totals.items() if k.startswith("synthetic"))
        print(f"  {built} binaries across {n_pkg} pseudo-package(s), "
              f"{len(toolchains)} compilers x {len(args.opt)} opt levels")

    # ---- algorithms ------------------------------------------------------
    if "algorithms" in args.source:
        print("\n== algorithms: TheAlgorithms/C ==")
        repo = cache / "TheAlgorithms-C"
        if not repo.exists():
            rc, log = _run(["git", "clone", "--depth", "1", ALGORITHMS_REPO, str(repo)])
            if rc != 0:
                print(f"  clone failed, skipping: {log.splitlines()[-1:]}")
                repo = None
        if repo and repo.exists():
            for tc in toolchains:
                for opt in args.opt:
                    dest = out_root / "algorithms" / f"{arch}-{tc.tag}-{opt}"
                    n = build_algorithms(repo, tc, opt, dest, args.jobs, args.algorithms_limit)
                    totals[f"algorithms/{tc.tag}/{opt}"] = n
                    print(f"  {tc.tag} {opt}: {n} built")

    # ---- gnu -------------------------------------------------------------
    if "gnu" in args.source:
        print("\n== gnu packages ==")
        with tempfile.TemporaryDirectory(prefix="binprov-build-", dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
            for name in args.gnu_packages:
                version, url = GNU_PACKAGES[name]
                try:
                    tarball = fetch(url, cache / "tarballs" / url.rsplit("/", 1)[-1])
                except Exception as exc:  # network/URL problems should not abort the rest
                    print(f"  {name}: download failed ({exc}); skipping")
                    continue
                pkg = f"{name}-{version}"
                for tc in toolchains:
                    for opt in args.opt:
                        dest = out_root / pkg / f"{arch}-{tc.tag}-{opt}"
                        if dest.exists() and any(dest.iterdir()):
                            print(f"  {pkg} {tc.tag} {opt}: already built, skipping")
                            continue
                        n = build_gnu(
                            tarball, pkg, tc, opt, dest, args.jobs,
                            Path(tmp) / f"{pkg}-{tc.tag}-{opt}",
                        )
                        totals[f"{pkg}/{tc.tag}/{opt}"] = n
                        print(f"  {pkg} {tc.tag} {opt}: {n} binaries")
                if not args.keep_cache:
                    tarball.unlink(missing_ok=True)

    # ---- report ----------------------------------------------------------
    n_total = sum(totals.values())
    size = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"\n{n_total} binaries under {out_root} ({size / 1e6:.1f} MB)")
    if n_total == 0:
        print("nothing was built; check the compiler and configure logs above")
        return 1
    print("\nnext:")
    print(f"  python scripts/build_corpus.py --root {out_root} --out data/corpus/local \\")
    print("      --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
