#!/usr/bin/env python3
"""Pack a tree of binaries into a BinProv corpus.

This is the disk-space chokepoint of the whole project, so it is worth being
deliberate here. A BinKit binary averages ~200 KB; its ``.text`` section is a
fraction of that, and ``.text`` plus function boundaries is all BinProv reads.
Packing therefore shrinks the working set several-fold, after which the source
binaries can be deleted (``--purge-source``).

Examples
--------
Inspect what would be selected, without writing anything (do this first — it is
how you check that labels were parsed correctly from a new directory layout)::

    python scripts/build_corpus.py --root data/binkit/normal \\
        --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --dry-run

Build the paper's main x86_64 corpus (Tables 3-6)::

    python scripts/build_corpus.py --root .../binkit/normal --out data/corpus/x86_64 \\
        --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal

Cap the size while staying balanced across the 8 configurations::

    ... --per-toolchain-limit 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov import discover, elf  # noqa: E402
from binprov.corpus import Corpus, CorpusWriter  # noqa: E402
from binprov.provenance import toolchain_key  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _parse_one(args):
    """Worker: read one binary and return its .text plus metadata.

    Runs in a subprocess. Returns ``None`` on any failure so the build can skip
    the file and carry on — a few unparseable files in a 6000-binary dataset
    should not abort a long job.
    """
    path, want_functions, max_text_bytes = args
    try:
        with open(path, "rb") as fh:
            buf = fh.read()
        sec = elf.parse(buf, want_functions=want_functions)
    except (elf.ElfError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}

    text = sec.data
    functions = [[f.offset, f.size] for f in sec.functions]
    names = [f.name for f in sec.functions]
    if max_text_bytes and len(text) > max_text_bytes:
        text = text[:max_text_bytes]
        functions_trimmed, names_trimmed = [], []
        for (off, size), name in zip(functions, names):
            if off + size <= max_text_bytes:
                functions_trimmed.append([off, size])
                names_trimmed.append(name)
        functions, names = functions_trimmed, names_trimmed

    return {
        "error": None,
        "path": str(path),
        "text": text,
        "vaddr": sec.vaddr,
        "arch_elf": sec.arch,
        "stripped": sec.stripped,
        "functions": functions,
        "func_names": names,
        "comment_compiler": elf.guess_compiler(sec.comment)[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pack .text sections of a binary tree into a BinProv corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", required=True, help="directory tree of labelled binaries")
    ap.add_argument("--out", help="output corpus directory (omit with --dry-run)")

    sel = ap.add_argument_group("selection (all filters accept several values)")
    sel.add_argument("--arch", nargs="*", default=None, help="e.g. x86_64 arm_64")
    sel.add_argument("--compiler", nargs="*", default=None, help="gcc clang")
    sel.add_argument(
        "--compiler-version",
        nargs="*",
        default=None,
        help="pin exact versions, e.g. 8.2.0 7.0. BinKit Normal ships 5 GCC "
        "and 4 Clang versions; leaving this open mixes all 9 into the two "
        "compiler classes and multiplies the corpus by ~4.5x",
    )
    sel.add_argument("--opt", nargs="*", default=None, help="O0 O1 O2 O3")
    sel.add_argument("--extra", nargs="*", default=None, help="normal pie lto noinline ...")
    sel.add_argument("--obfuscation", nargs="*", default=None, help="bcf fla sub all")
    sel.add_argument("--packages", nargs="*", default=None, help="restrict to these packages")
    sel.add_argument(
        "--per-toolchain-limit",
        type=int,
        default=None,
        help="max binaries per compilation configuration; the safest size knob, "
        "since it keeps the corpus balanced",
    )
    sel.add_argument(
        "--max-text-bytes",
        type=int,
        default=None,
        help="truncate each .text to this many bytes. Saves disk but biases "
        "toward the head of the binary, which paper §5.2(c) shows is the "
        "easiest part to classify -- do not use for the location analysis",
    )
    sel.add_argument(
        "--no-functions",
        action="store_true",
        help="skip symbol-table parsing (faster; disables function-level voting)",
    )
    sel.add_argument(
        "--no-func-names",
        action="store_true",
        help="keep function boundaries but drop names (smaller index)",
    )

    sp = ap.add_argument_group("splits")
    sp.add_argument("--train-ratio", type=float, default=0.8, help="paper §4.1 uses 8:2")
    sp.add_argument(
        "--split-group-by",
        choices=["program", "package", "binary"],
        default="program",
        help="'program' keeps every compiled variant of a program on one side, "
        "so a test program is never seen at another optimization level",
    )
    sp.add_argument("--split-seed", type=int, default=1234)
    sp.add_argument(
        "--pretrain-per-package",
        type=int,
        default=1,
        help="programs per package in the MLM pre-training set (paper: "
        "'at least one binary (2x4 variants) from each software project')",
    )

    run = ap.add_argument_group("run")
    run.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    run.add_argument("--dry-run", action="store_true", help="report selection and exit")
    run.add_argument(
        "--purge-source",
        action="store_true",
        help="DELETE each source binary once packed. Requires --yes. Only do "
        "this for a dataset you can re-download",
    )
    run.add_argument("--yes", action="store_true", help="confirm --purge-source")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        ap.error(f"--root {root} is not a directory")
    if args.purge_source and not args.yes:
        ap.error("--purge-source deletes files; re-run with --yes to confirm")

    print(f"scanning {root} ...", flush=True)
    t0 = time.time()
    candidates = list(
        discover.discover(
            root,
            arch=args.arch,
            compiler=args.compiler,
            compiler_version=args.compiler_version,
            opt=args.opt,
            extra=args.extra,
            obfuscation=args.obfuscation,
            packages=args.packages,
            per_toolchain_limit=args.per_toolchain_limit,
        )
    )
    print(f"scan took {time.time() - t0:.1f}s")
    print(discover.summarize(candidates))

    if candidates:
        src_bytes = sum(c.path.stat().st_size for c in candidates)
        print(f"\nsource binaries on disk: {human(src_bytes)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not args.out:
        ap.error("--out is required unless --dry-run")
    if not candidates:
        print("nothing to do")
        return 1

    out_dir = Path(args.out)
    build_params = {
        "root": str(root),
        "filters": {
            k: getattr(args, k)
            for k in ("arch", "compiler", "compiler_version", "opt", "extra",
                      "obfuscation", "packages")
        },
        "per_toolchain_limit": args.per_toolchain_limit,
        "max_text_bytes": args.max_text_bytes,
        "functions": not args.no_functions,
    }

    errors: Counter[str] = Counter()
    label_mismatch = 0
    arch_mismatch = 0
    n_purged = 0
    work = [(c.path, not args.no_functions, args.max_text_bytes) for c in candidates]

    print(f"\npacking into {out_dir} with {args.workers} workers ...", flush=True)
    t0 = time.time()
    with CorpusWriter(out_dir, build_params=build_params) as writer:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for cand, res in zip(candidates, pool.map(_parse_one, work, chunksize=8)):
                if res["error"]:
                    errors[res["error"].split(":")[0]] += 1
                    continue
                labels = cand.labels.as_dict()

                # cross-check the path-derived compiler against the .comment
                # fingerprint; a systematic mismatch means the layout parser is
                # reading the wrong component
                if res["comment_compiler"] and labels.get("compiler"):
                    if res["comment_compiler"] != labels["compiler"]:
                        label_mismatch += 1
                if labels.get("arch") and res["arch_elf"] != labels["arch"]:
                    arch_mismatch += 1
                    labels["arch"] = res["arch_elf"]  # trust the ELF header

                writer.add(
                    res["text"],
                    path=res["path"],
                    package=cand.package,
                    program=cand.program,
                    labels=labels,
                    text_vaddr=res["vaddr"],
                    arch_elf=res["arch_elf"],
                    stripped=res["stripped"],
                    functions=res["functions"],
                    func_names=[] if args.no_func_names else res["func_names"],
                )
                if args.purge_source:
                    try:
                        cand.path.unlink()
                        n_purged += 1
                    except OSError as exc:
                        print(f"  warning: could not delete {cand.path}: {exc}")

        n_bin, n_bytes = writer.num_binaries, writer.num_bytes

    dt = time.time() - t0
    print(f"packed {n_bin} binaries, {human(n_bytes)} of .text in {dt:.1f}s")
    if candidates:
        print(f"  compression vs source binaries: {n_bytes / max(1, src_bytes):.2f}x of original")
    if errors:
        print(f"  skipped {sum(errors.values())}: {dict(errors)}")
    if arch_mismatch:
        print(f"  note: {arch_mismatch} arch labels corrected from the ELF header")
    if label_mismatch:
        pctm = 100 * label_mismatch / max(1, n_bin)
        print(
            f"  WARNING: {label_mismatch} ({pctm:.1f}%) binaries whose .comment "
            "disagrees with the path-derived compiler."
        )
        if pctm > 20:
            print(
                "  That is high enough to suspect the layout parser. Re-run with "
                "--dry-run and check the printed toolchains."
            )
    if n_purged:
        print(f"  deleted {n_purged} source binaries (--purge-source)")

    # -- splits ------------------------------------------------------------
    corpus = Corpus(out_dir)
    splits = corpus.make_splits(
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        group_by=args.split_group_by,
    )
    splits["pretrain"] = corpus.pretrain_bids(
        per_package=args.pretrain_per_package,
        restrict_to=splits["train"],
        seed=args.split_seed,
    )
    path = corpus.save_splits("default", splits)
    print(
        f"\nsplits -> {path}\n"
        f"  train {len(splits['train'])} / test {len(splits['test'])} binaries "
        f"(grouped by {args.split_group_by}, ratio {args.train_ratio})\n"
        f"  pretrain {len(splits['pretrain'])} binaries (subset of train)"
    )

    # a quick look at what training will actually see
    idx = corpus.sequences(seq_len=512, level="binary", bids=splits["train"])
    print(f"  -> {len(idx):,} training sequences of 512 bytes")
    n_with_funcs = sum(1 for r in corpus.records if r.functions)
    print(f"  -> {n_with_funcs}/{len(corpus)} binaries have function boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
