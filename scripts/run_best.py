#!/usr/bin/env python3
"""Reproduce the measured results from their exact recorded parameters.

Reads configs/best_results.json and replays each recorded run by translating its
saved args.json back into a command line (flag whitelist, correct bool / nargs
handling). Omitted flags stay omitted: early wide runs (r4, r6) have no
``--val-seed`` and none is invented for them. Everything is written under
``results/best`` (gitignored); nothing under ``reports/`` is ever touched.

Phases (dependency order)::

    prepare    corpus + split preflight checks (no downloads)
    pretrain   512-byte MLM, then 2048-byte MLM by continued pre-training
    train      fine-tune every run in the selected group(s)
    offline    recompute the reported number from fresh probs.npz / result.json

Dry-run is the default: the whole plan is printed, nothing is created or run.
Use ``--execute`` to actually run. ``--list`` shows the groups and memberships.

Examples::

    python scripts/run_best.py --list
    python scripts/run_best.py --group o2o3_7wide_512B                 # dry-run
    python scripts/run_best.py --group o2o3_7wide_512B --phase train   # dry-run
    python scripts/run_best.py --group opt4_3wide_binary --phase offline --execute
    python scripts/run_best.py --group all --execute                   # full suite
"""

from __future__ import annotations

import argparse
import json
import statistics
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "best_results.json"
PY = sys.executable

# Every experiment.py dest that still has a live CLI flag. Anything in an
# archived args.json that is not here is skipped with a warning, never passed.
EXPERIMENT_FLAGS = {
    "task", "splits", "init_from", "from_scratch", "tag", "arch", "extra",
    "compiler", "version", "test_version", "train_compiler",
    "max_seqs_per_binary", "stride", "val_frac", "val_seed", "split_mode",
    "seq_bytes", "eval_window_bytes", "jitter", "pos_extend", "pool",
    "pool_heads", "mil_tau", "taper", "dropout", "classifier_dropout",
    "epochs", "batch_size", "lr", "head_lr", "llrd", "weight_decay",
    "warmup_ratio", "clip_grad", "label_smoothing", "use_pairs", "pair_loss",
    "pair_ce", "pair_margin", "pairs_per_batch", "pair_min_size", "pair_jitter",
    "workers", "seed", "log_every", "val_max_seqs", "tta", "tta_views",
    "tta_span", "save_probs", "no_save_weights",
}

PRETRAIN_FLAGS = {
    "splits", "split_name", "seq_bytes", "init_from", "pos_extend", "layers",
    "hidden", "heads", "intermediate", "epochs", "max_steps", "batch_size",
    "grad_accum", "lr", "weight_decay", "warmup_ratio", "clip_grad",
    "mask_prob", "mask_replace", "random_replace", "max_seqs_per_binary",
    "pair_prob", "val_fraction", "workers", "seed", "fp32", "log_every",
    "save_every_epoch", "resume",
}

def verify_corpus(cfg: dict) -> None:
    """Validate the shipped test partition and pinned validation reference."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from binprov.corpus import Corpus
    c = Corpus(ROOT / cfg["corpus_dir"])
    splits = c.load_splits("default")
    lines = [l.strip() for l in (ROOT / cfg["split_evidence"]).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    cut = lines.index("[VALIDATION]")
    expected = lines[lines.index("[TEST]") + 1:cut]
    train = {c.records[b].group for b in splits["train"]}
    test = {c.records[b].group for b in splits["test"]}
    if sorted(test) != expected or train & test:
        raise ValueError("SPLIT MISMATCH: test programs differ or train/test programs overlap")
    groups = sorted(train)
    np.random.default_rng(1234).shuffle(groups)
    val = sorted(groups[:max(1, round(len(groups) * 0.15))])
    if val != lines[cut + 1:]:
        raise ValueError("SPLIT MISMATCH: pinned validation programs differ")
    print(f"Verified {len(test)} test and {len(val)} pinned validation programs")


def checkpoint_ready(path: Path) -> bool:
    return (path / "binprov_config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def offline_dir(cfg: dict, tag: str) -> Path:
    return (ROOT / "results/explore" / tag if cfg.get("archive_probs")
            else fresh_dir(cfg, tag))


def offline_result(cfg: dict, tag: str) -> Path:
    local = offline_dir(cfg, tag) / "result.json"
    if local.is_file() or not cfg.get("archive_probs"):
        return local
    return ROOT / cfg["runs"][tag]["archived_result"]


def err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def fresh_dir(cfg: dict, tag: str) -> Path:
    return ROOT / cfg["results_dir"] / "finetune" / tag


def rel_of(path: Path | str) -> str:
    """Print/command-friendly path: repo-relative when under ROOT."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def archived_ckpt_name(cfg: dict, path: str) -> str | None:
    """'checkpoints/binkit_x86_64/mlm2048' -> 'mlm2048' (or None)."""
    for name, cp in cfg["checkpoint_paths"].items():
        if cp["archived"] == path:
            return name
    return None


def fresh_ckpt_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["checkpoint_paths"][name]["fresh"]


# ---------------------------------------------------------------------------
# archived args -> command line
# ---------------------------------------------------------------------------


def replay_argv(script: str, args: dict, *, corpus: str, out: str,
                init_from: str | None = None) -> list[str]:
    """Translate one archived vars(args) dict back into a CLI argv.

    corpus/out/init_from are substituted by the caller (fresh paths), never taken
    from the archive. Booleans are store_true flags (present only when True);
    lists are nargs='*' values; None values and non-whitelisted keys are skipped.
    """
    flags = EXPERIMENT_FLAGS if script == "experiment.py" else PRETRAIN_FLAGS
    argv = [PY, f"scripts/{script}"]
    skipped: list[str] = []
    for key, value in args.items():
        if key in ("corpus", "out", "init_from"):
            continue
        if key not in flags:
            skipped.append(key)
            continue
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.append(flag)
            argv.append(str(value))
    if skipped:
        print(f"  (skipped non-replayable archived keys: {', '.join(skipped)})",
              file=sys.stderr)
    argv += ["--corpus", corpus]
    if init_from:
        argv += ["--init-from", init_from]
    argv += ["--out", out]
    return argv


def resolve_init(cfg: dict, archived_init: str | None) -> str | None:
    """Point --init-from at the fresh checkpoint a reproduce run produces."""
    if not archived_init:
        return None
    name = archived_ckpt_name(cfg, archived_init)
    if name is None:
        return archived_init
    return cfg["checkpoint_paths"][name]["fresh"]


# ---------------------------------------------------------------------------
# plan steps
# ---------------------------------------------------------------------------


def prereq_for_run(cfg: dict, run: dict) -> list[str]:
    """Which pre-training checkpoints a run needs (mlm2048 pulls in mlm512)."""
    args = read_json(run["archived_args"])
    need: list[str] = []
    init = resolve_init(cfg, args.get("init_from"))
    if init is None:
        return need
    name = archived_ckpt_name(cfg, args.get("init_from") or "")
    if name == "mlm2048":
        need.append("mlm512")
        need.append("mlm2048")
    elif name == "mlm512":
        need.append("mlm512")
    return need


def add_pretrain_steps(cfg: dict, steps: list[dict], names: list[str]) -> None:
    for name in names:
        prereq = cfg["prerequisites"][name]
        out = cfg["checkpoint_paths"][name]["fresh"]
        if "command" in prereq:
            tokens = {"{corpus}": cfg["corpus_dir"],
                      "{mlm512}": cfg["checkpoint_paths"]["mlm512"]["fresh"],
                      "{mlm2048}": cfg["checkpoint_paths"]["mlm2048"]["fresh"]}
            cmd = [PY, *[tokens.get(t, t) for t in prereq["command"][1:]]]
        else:
            archived = read_json(prereq["archived_args"])
            cmd = replay_argv(
                prereq["script"], archived,
                corpus=cfg["corpus_dir"], out=out,
            )
        inputs = [cfg["prerequisites"]["corpus"]["meta_file"]]
        if name == "mlm2048":
            inputs.append(cfg["checkpoint_paths"]["mlm512"]["fresh"])
        steps.append({
            "phase": "pretrain",
            "name": name,
            "title": prereq["description"],
            "cmd": cmd,
            "outputs": [out],
            "inputs": inputs,
        })


def add_train_step(cfg: dict, steps: list[dict], tag: str) -> None:
    run = cfg["runs"][tag]
    args = read_json(run["archived_args"])
    out = rel_of(fresh_dir(cfg, tag))
    cmd = replay_argv(
        run["script"], args,
        corpus=cfg["corpus_dir"], out=out,
        init_from=resolve_init(cfg, args.get("init_from")),
    )
    acc = ""
    res_rel = run.get("archived_result")
    if res_rel and (ROOT / res_rel).exists():
        res = read_json(res_rel)
        try:
            acc = f"recorded test acc {res['test']['accuracy'] * 100:.2f}%"
            if res.get("test_tta") is not None:
                acc += f" (tta {res['test_tta']['accuracy'] * 100:.2f}%)"
        except (KeyError, TypeError):
            acc = ""
    steps.append({
        "phase": "train",
        "name": tag,
        "title": tag,
        "cmd": cmd,
        "archived_result": res_rel,
        "archived_acc": acc,
        "outputs": [out],
        "inputs": [run["archived_args"], cfg["prerequisites"]["corpus"]["meta_file"]],
        "checkpoint": resolve_init(cfg, args.get("init_from")),
    })


def add_offline_step(cfg: dict, steps: list[dict], gkey: str, group: dict) -> None:
    if group["kind"] == "archived_no_recipe":
        return
    runs = group["runs"]
    meta = cfg["prerequisites"]["corpus"]["meta_file"]
    table_out = rel_of(ROOT / cfg["results_dir"] / "tables" / f"{gkey}.md")
    if group["kind"] == "mean":
        steps.append({
            "phase": "offline",
            "name": gkey,
            "title": group["description"],
            "cmd": None,
            "mean": True,
            "expected": group.get("expected_accuracy"),
            "expected_sd": group.get("expected_sd"),
            "runs": runs,
            "outputs": [table_out],
            "inputs": [rel_of(offline_result(cfg, t)) for t in runs] + [meta],
        })
        return
    radii = [str(r) for r in group.get("context", [0])]
    cmd = [PY, "scripts/combine.py", "--corpus", cfg["corpus_dir"]]
    dirs = [(ROOT / cfg["results_dir"] / "archive_inputs" / t)
            if cfg.get("archive_probs") else fresh_dir(cfg, t) for t in runs]
    cmd += [rel_of(d) for d in dirs]
    cmd += ["--ensemble", "--context", *radii]
    if group.get("binary_vote"):
        cmd.append("--binary-vote")
    cmd += ["--out", table_out]
    steps.append({
        "phase": "offline",
        "name": gkey,
        "title": group["description"],
        "cmd": cmd,
        "expected": group.get("expected_accuracy"),
        "outputs": [table_out],
        "inputs": [rel_of(offline_dir(cfg, t) / "probs.npz") for t in runs]
                  + [rel_of(offline_result(cfg, t)) for t in runs] + [meta],
        "probs_required": True,
        "runs": runs,
    })


def group_tags(cfg: dict, gkey: str) -> list[str]:
    return list(cfg["groups"][gkey]["runs"])


def plan(cfg: dict, selected: list[str], phase: str) -> list[dict]:
    """Assemble the ordered steps for the selected groups and requested phase."""
    steps: list[dict] = []
    show = ({"prepare", "pretrain", "train", "offline"} if phase == "all"
            else {phase})

    tags: list[str] = []
    for gkey in selected:
        for t in group_tags(cfg, gkey):
            if t not in tags:
                tags.append(t)

    if "prepare" in show:
        steps.append(prepare_step(cfg))

    if "pretrain" in show:
        prereq_names: list[str] = []
        for t in tags:
            for p in prereq_for_run(cfg, cfg["runs"][t]):
                if p not in prereq_names:
                    prereq_names.append(p)
        if prereq_names:
            add_pretrain_steps(cfg, steps, prereq_names)

    if "train" in show:
        for t in tags:
            add_train_step(cfg, steps, t)

    if "offline" in show:
        for gkey in selected:
            add_offline_step(cfg, steps, gkey, cfg["groups"][gkey])
    return steps


def prepare_step(cfg: dict) -> dict:
    meta = ROOT / cfg["prerequisites"]["corpus"]["meta_file"]
    return {
        "phase": "prepare",
        "name": "corpus",
        "title": "corpus + split preflight",
        "cmd": None,
        "meta": meta,
        "outputs": [],
        "inputs": [],
    }


# ---------------------------------------------------------------------------
# rendering / running
# ---------------------------------------------------------------------------


def run_step(step: dict, cfg: dict, execute: bool) -> int | None:
    """Run one plan step. Returns subprocess rc, or None for no-op steps."""
    phase = step["phase"]
    print(f"\n### [{phase}] {step['name']} — {step['title']}")
    if step.get("archived_acc"):
        print(f"  {step['archived_acc']}")
    if phase == "prepare":
        if not execute:
            print("  (preflight: corpus + split check; run --execute to verify)")
            return None
        verify_corpus(cfg)
        return 0
    if not step.get("cmd"):
        return None
    print("  cmd: " + shlex.join(step["cmd"]))
    if not execute:
        return None
    destination = ROOT / step["outputs"][0]
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"output already exists: {destination}; move it aside before retraining")
    print("  -> running ...", flush=True)
    return subprocess.call(step["cmd"], cwd=str(ROOT))


def check_step_inputs(step: dict, cfg: dict) -> list[str]:
    """Human-readable list of missing inputs that would block execution."""
    missing: list[str] = []
    for p in step.get("inputs", []):
        if not (ROOT / p).exists():
            missing.append(p)
    if step["phase"] == "prepare":
        if not step["meta"].exists():
            missing.append(rel_of(step["meta"]))
    ckpt = step.get("checkpoint")
    if step["phase"] == "pretrain" and step["name"] == "mlm2048":
        ckpt = cfg["checkpoint_paths"]["mlm512"]["fresh"]
    if ckpt and not checkpoint_ready(ROOT / ckpt):
        missing.append(f"{ckpt} (complete MLM weights and binprov_config.json required)")
    return missing


def require_inputs(step: dict, cfg: dict, execute: bool) -> int | None:
    """Dry runs annotate missing inputs; execution refuses to start on them."""
    missing = check_step_inputs(step, cfg)
    if not missing:
        return None
    hints: list[str] = []
    if step["phase"] == "prepare":
        hints.append(
            "build the corpus first: scripts/fetch_binkit.sh normal, then "
            + " ".join(cfg["prerequisites"]["corpus"]["build_command"])
        )
    elif "meta.json" in " ".join(missing):
        hints.append("build the corpus first (see README / docs/DATA.md)")
    if step.get("probs_required"):
        hints.append("run --phase train first: probs.npz is produced by experiment.py")
    if step["phase"] == "offline":
        hints.insert(0, "offline mode needs the corpus and per-run probs.npz/result.json")
    msg = f"missing input for [{step['phase']}] {step['name']}: {', '.join(missing)}"
    if hints:
        msg += " — " + "; ".join(hints)
    if execute:
        print(f"error: {msg}", file=sys.stderr)
        return 2
    print(f"  [missing input: {', '.join(missing)} — will block --execute]")
    return None


def run_offline(step: dict, cfg: dict, execute: bool) -> int | None:
    print(f"\n### [offline] {step['name']} — {step['title']}")
    if not execute:
        if step.get("cmd"):
            print("  cmd: " + shlex.join(step["cmd"]))
        return None
    if step.get("mean"):
        rows = []
        accs = []
        for t in step["runs"]:
            res = read_json(str(offline_result(cfg, t)))
            acc = res["test"]["accuracy"]
            accs.append(acc)
            rows.append((t, acc))
        mean = statistics.fmean(accs)
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        out = step["outputs"][0]
        out = ROOT / out
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        md = ["### " + step["title"], "",
              "| run | test accuracy |",
              "|---|---|"]
        for t, acc in rows:
            md.append(f"| {t} | {acc * 100:.4f}% |")
        md.append("")
        md.append(f"**mean** {mean * 100:.2f}%  (sd {sd * 100:.2f}%)  "
                  f"expected {step['expected'] * 100:.2f}%")
        Path(out).write_text("\n".join(md) + "\n")
        print(f"  mean over {len(accs)} runs: {mean * 100:.2f}%  "
              f"sd {sd * 100:.2f}%  expected {step['expected'] * 100:.2f}%")
        print(f"  wrote {out}")
        return 0
    if cfg.get("archive_probs"):
        for tag in step["runs"]:
            dest = ROOT / cfg["results_dir"] / "archive_inputs" / tag
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(offline_dir(cfg, tag) / "probs.npz", dest / "probs.npz")
            shutil.copy2(offline_result(cfg, tag), dest / "result.json")
    (ROOT / step["outputs"][0]).parent.mkdir(parents=True, exist_ok=True)
    cmd = step["cmd"]
    print("  cmd: " + shlex.join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_groups(cfg: dict) -> None:
    print("Measured-result groups (configs/best_results.json)")
    print(f"{'group':32s} {'runs':>4s}  metric / expected")
    for gkey, g in cfg["groups"].items():
        runs = g.get("runs")
        n = len(runs) if isinstance(runs, list) else 0
        kind = g["kind"]
        if kind == "archived_no_recipe":
            line = f"{gkey:32s} {'—':>4s}  {g['unit']}: {g['expected_accuracy']:.4f} (archived, no exact recipe)"
        else:
            line = (f"{gkey:32s} {n:4d}  {g['unit']}: "
                    f"{g['expected_accuracy']:.4f}")
        print(line)
    print("\nSpecial groups: all  (run every reproducible group + prerequisites)")
    print("Phases: prepare, pretrain, train, offline, all (default). "
          "Dry-run by default; add --execute to run.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay the measured results from configs/best_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--group", default=None,
                    help="group name from --list, or 'all'")
    ap.add_argument("--phase", default="all",
                    choices=["all", "prepare", "pretrain", "train", "offline"],
                    help="which phase(s) of the plan to show/run")
    ap.add_argument("--corpus", default=None,
                    help="override corpus dir (default: config corpus_dir)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print only (the default)")
    ap.add_argument("--archive-probs", action="store_true",
                    help="offline only: use a separately stored probability bundle")
    mode.add_argument("--execute", action="store_true",
                    help="actually run (default is a dry run: print, mutate nothing)")
    ap.add_argument("--list", action="store_true", help="list groups and exit")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.corpus:
        cfg["corpus_dir"] = args.corpus
        cfg["prerequisites"]["corpus"]["meta_file"] = str(Path(args.corpus) / "meta.json")
    cfg["archive_probs"] = args.archive_probs
    if args.archive_probs and args.phase != "offline":
        return err("--archive-probs requires --phase offline")

    if args.list:
        list_groups(cfg)
        return 0

    if not args.group:
        return err("pass --group NAME (see --list) or --group all")

    if args.group == "all":
        selected = [k for k, g in cfg["groups"].items()
                    if g["kind"] != "archived_no_recipe"]
        group_title = "all reproducible groups"
    elif args.group in cfg["groups"]:
        selected = [args.group]
        group_title = args.group
        g = cfg["groups"][args.group]
        if g.get("kind") == "archived_no_recipe":
            return err(
                f"group '{args.group}' has no exact recipe: {g['note']} "
                "It is recorded evidence only (expected accuracy "
                f"{g['expected_accuracy']:.4f}); do not attempt to replay it."
            )
    else:
        return err(f"unknown group '{args.group}'. Available: "
                   + ", ".join(cfg["groups"]) + ", all")

    execute = args.execute
    print(f"== BinProv measured-results plan: group '{group_title}' "
          f"(phase {args.phase}, {'EXECUTE' if execute else 'dry-run'}) ==")
    print(f"corpus: {cfg['corpus_dir']}   outputs under: {cfg['results_dir']}/")

    steps = plan(cfg, selected, args.phase)
    if not steps:
        return err("nothing to do for that group/phase combination")

    if execute and args.phase != "all":
        preflight = prepare_step(cfg)
        if require_inputs(preflight, cfg, True):
            return 2
        verify_corpus(cfg)
    for i, step in enumerate(steps, 1):
        rc = require_inputs(step, cfg, execute)
        if rc:
            return rc
        if step["phase"] == "offline":
            rc = run_offline(step, cfg, execute)
        else:
            rc = run_step(step, cfg, execute)
        if rc:
            return err(f"step [{step['phase']}] {step['name']} failed (rc {rc})")

    print("\n" + ("done." if execute else
                  "dry-run complete: nothing was created or executed. "
                  "Re-run with --execute to run the plan."))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(err(str(exc)))
