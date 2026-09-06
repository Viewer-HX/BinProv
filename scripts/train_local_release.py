#!/usr/bin/env python3
"""Bounded local-training and release-preparation driver for one Mac.

Reads ``configs/local_release.json`` and runs each profile's stages sequentially
through the existing training scripts (``pretrain_mlm.py``, ``experiment.py``)
plus an in-process evaluation over the saved probabilities and
``scripts/export_hf.py``. Every stage is an *archived recipe re-expressed* for a
small MPS micro-batch with gradient accumulation, so the archived effective
batch size is preserved exactly (see configs/local_release.json). Output is
written only under ``results/local_release/<profile>/`` — never to the
historical ``checkpoints/``, ``reports/`` or ``results/explore`` paths.

Dry-run is the default: the whole plan is printed, nothing is created or run.
Use ``--execute`` to run. On macOS every training command is wrapped in
``caffeinate`` so an unattended run does not fall asleep; keep the machine
plugged in for the durations documented in docs/LOCAL_TRAINING.md.

Resume policy (safe by construction): a stage output is only ever launched into
again when its previously saved ``args.json``/``pretrain_args.json`` exactly
matches the arguments this run would use (``--resume`` must be passed to touch
an existing, interrupted stage). A completed checkpoint counts only when the
real files are present — weights + config + the training result/state — never a
partial set. ``results/local_release/status.json`` records stage outcomes and is
rewritten atomically.

Examples::

    python scripts/train_local_release.py --list
    python scripts/train_local_release.py --profile opt4_narrow_seed13        # dry-run
    python scripts/train_local_release.py --profile opt4_narrow_seed13 --execute
    python scripts/train_local_release.py --profile opt4_narrow_seed13 \
        --phase finetune --resume --execute        # continue an interrupted run
    python scripts/train_local_release.py --profile opt4_narrow_seed13 \
        --phase export --execute
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "configs" / "local_release.json"
PY = sys.executable

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import run_best  # noqa: E402  (flag whitelists / corpus verification)

# experiment.py gained --grad-accum and --resume; extend the shared whitelist so
# replays of local finetune args carry them. run_best's own replays are untouched
# (their archived args never contained these keys).
EXPERIMENT_FLAGS = set(run_best.EXPERIMENT_FLAGS) | {"grad_accum", "resume"}
PRETRAIN_FLAGS = set(run_best.PRETRAIN_FLAGS)
# files an experiment.py run may legitimately leave before args.json is written
_TRANSIENT = {"train_log.jsonl"}

PHASES = ("pretrain512", "pretrain2048", "finetune", "evaluate", "export")


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    return json.loads(cfg_path.read_text())


def read_json(rel: str | Path) -> dict:
    return json.loads((ROOT / rel).read_text())


def rel_of(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# config access
# ---------------------------------------------------------------------------


def profile_dir(cfg: dict, profile: str) -> Path:
    return ROOT / cfg["meta"]["results_dir"] / profile


def status_file_for(cfg: dict) -> Path:
    return ROOT / cfg["meta"]["results_dir"] / "status.json"


def step_out_dir(cfg: dict, profile: str, step: dict) -> Path:
    return profile_dir(cfg, profile) / step["out"]


def profile_steps(cfg: dict, profile: str) -> list[dict]:
    return cfg["profiles"][profile]["steps"]


def finetune_step(cfg: dict, profile: str) -> dict:
    for s in profile_steps(cfg, profile):
        if s["phase"] == "finetune":
            return s
    raise KeyError(f"profile {profile!r} has no finetune step")


# ---------------------------------------------------------------------------
# the exact arguments a stage must run with
# ---------------------------------------------------------------------------

_DEFAULTS_CACHE: dict[str, dict] = {}


def script_defaults(script: str) -> dict:
    """vars(args) of the script's parser when no optional flags are given (cached).

    Archived args.json files predate a few current parser knobs (e.g.
    experiment.py's later defaulted flags), so a freshly launched run would save
    keys the archive never recorded. Filling those defaults into the desired
    dict is what makes the saved-args comparison exact on resume.
    """
    if script in _DEFAULTS_CACHE:
        return _DEFAULTS_CACHE[script]
    import importlib.util

    name = f"lr_defaults_{script.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # required flags get throwaway values; only the *defaults* of the optional
    # flags matter and every required one is substituted by the runner anyway.
    probe = ["_", "--corpus", "_c", "--out", "_o"]
    if script == "experiment.py":
        probe += ["--task", "_t"]
    old = list(sys.argv)
    sys.argv[:] = probe
    try:
        ns = vars(mod.parse_args())
    finally:
        sys.argv[:] = old
    _DEFAULTS_CACHE[script] = ns
    return ns


def step_desired(cfg: dict, profile: str, step: dict) -> dict:
    """The canonical vars(args)-shaped dict this stage must run with.

    Built from the archived args (or an explicit full recipe), completed with the
    script's current parser defaults, the profile's overrides, and the local
    output paths substituted by the runner. Comparing this to the stage's saved
    args.json/pretrain_args.json is how an incompatible pre-existing output is
    detected, and how a safe resume is recognised.
    """
    if step.get("args") is not None:
        d = dict(step["args"])
    else:
        d = dict(read_json(step["archived_args"]))
    for key, value in script_defaults(step["script"]).items():
        d.setdefault(key, value)
    d.update(step.get("overrides") or {})
    d["corpus"] = cfg["meta"]["corpus_dir"]
    d["out"] = rel_of(step_out_dir(cfg, profile, step))
    init_step = step.get("init_from_step")
    if init_step:
        for s in profile_steps(cfg, profile):
            if s["name"] == init_step:
                d["init_from"] = rel_of(step_out_dir(cfg, profile, s))
                break
        else:
            raise KeyError(f"profile {profile!r}: unknown init_from_step {init_step!r}")
    else:
        d["init_from"] = None
    return d


def replay_argv(script: str, d: dict) -> list[str]:
    """Translate a vars(args)-shaped dict back into a CLI argv (like run_best)."""
    flags = EXPERIMENT_FLAGS if script == "experiment.py" else PRETRAIN_FLAGS
    argv = [PY, rel_of(SCRIPTS / script)]
    for key, value in d.items():
        if key in ("corpus", "out", "init_from"):
            continue
        if key not in flags:
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
    if d.get("init_from"):
        argv += ["--init-from", str(d["init_from"])]
    argv += ["--corpus", d["corpus"]]
    argv += ["--out", d["out"]]
    return argv


def saved_args_name(script: str) -> str:
    return "pretrain_args.json" if script == "pretrain_mlm.py" else "args.json"


# ---------------------------------------------------------------------------
# stage completeness / resume safety (files are the authority)
# ---------------------------------------------------------------------------


def _dir_entries(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {e.name for e in path.iterdir()}


def _has_hf_weights(out: Path) -> bool:
    return (out / "config.json").is_file() and any(
        (out / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def training_state_epoch(out: Path) -> int | None:
    """Epoch of the latest training_state.pt, or None if it cannot be read."""
    try:
        import torch

        payload = torch.load(out / "training_state.pt", map_location="cpu",
                             weights_only=False)
        return int(payload["epoch"])
    except Exception:  # noqa: BLE001 - a corrupt/unreadable snapshot must never
        return None  # crash a status query; the stage is simply not complete


def pretrain_complete(cfg: dict, profile: str, step: dict, desired: dict) -> bool:
    out = step_out_dir(cfg, profile, step)
    need = {"binprov_config.json", "pretrain_args.json", "baseline.json",
            "training_state.pt"}
    if not need.issubset(_dir_entries(out)) or not _has_hf_weights(out):
        return False
    epochs = int(desired["epochs"])
    ep = training_state_epoch(out)
    return ep is not None and ep >= epochs - 1


def finetune_complete(out: Path) -> bool:
    return all(
        (out / f).is_file()
        for f in ("model.pt", "binprov_config.json", "head.json", "result.json")
    )


def stage_status(cfg: dict, profile: str, step: dict) -> tuple[str, str]:
    """Classify a training-stage output directory.

    Returns ``(state, detail)`` with state in
    ``empty | complete | resumable | incompatible | unexpected``. The saved-args
    comparison is what separates a *safe resume* from an *incompatible
    overwrite*; nothing here ever deletes or overwrites anything.
    """
    out = step_out_dir(cfg, profile, step)
    entries = _dir_entries(out)
    if not entries:
        return "empty", "no output yet"

    script = step["script"]
    args_file = saved_args_name(script)
    desired = step_desired(cfg, profile, step)

    if step["phase"] in ("pretrain512", "pretrain2048"):
        if pretrain_complete(cfg, profile, step, desired):
            return "complete", "MLM weights + config + training state present"
        if (out / "training_state.pt").is_file() and (out / args_file).is_file():
            saved = read_json(out / args_file)
            if saved == desired:
                ep = training_state_epoch(out)
                if ep is None:
                    return "resumable", "interrupted (training_state.pt present)"
                return "resumable", f"interrupted at epoch {ep + 1}/{desired['epochs']}"
            return "incompatible", "saved args differ from this profile's recipe"
        if (out / args_file).is_file():
            saved = read_json(out / args_file)
            if saved != desired:
                return "incompatible", "saved args differ from this profile's recipe"
    else:  # finetune
        if finetune_complete(out):
            return "complete", "result.json + weights + config present"
        if (out / args_file).is_file():
            saved = read_json(out / args_file)
            if saved == desired:
                return "resumable", "interrupted before result.json was written"
            return "incompatible", "saved args differ from this profile's recipe"

    leftover = entries - _TRANSIENT
    if not leftover:
        return "empty", "only transient log files (safe to restart fresh)"
    return "unexpected", f"unrecognised contents: {sorted(entries)}"


# ---------------------------------------------------------------------------
# environment preflight
# ---------------------------------------------------------------------------


def power_state() -> str:
    """'ac' | 'battery' | 'unknown' — best-effort on macOS."""
    if sys.platform != "darwin":
        return "unknown"
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:  # noqa: BLE001 - a failed probe is a warning, not a crash
        return "unknown"
    if "AC Power" in out or "AC attached" in out:
        return "ac"
    if "Battery Power" in out:
        return "battery"
    return "unknown"


def backend_available(want: str) -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if want == "mps":
        return bool(torch.backends.mps.is_available())
    if want == "cuda":
        return bool(torch.cuda.is_available())
    return True


def preflight(cfg: dict, profile: str, execute: bool) -> list[str]:
    """Return a list of blocking problems (empty when OK). Warnings print."""
    problems: list[str] = []
    meta = cfg["meta"]
    prof = cfg["profiles"][profile]

    # corpus + canonical split (train/test partition and pinned validation)
    if not (ROOT / meta["corpus_dir"] / "meta.json").is_file():
        problems.append(
            f"corpus {meta['corpus_dir']} missing; build it first "
            "(docs/DATA.md / configs/best_results.json prepare phase)"
        )
    else:
        try:
            run_best.verify_corpus(
                {"corpus_dir": meta["corpus_dir"], "split_evidence": meta["split_evidence"]}
            )
        except ValueError as exc:
            problems.append(f"split verification failed: {exc}")

    # backend
    want = prof.get("backend") or meta.get("backend", "mps")
    if backend_available(want):
        print(f"  backend OK: {want}")
    else:
        problems.append(
            f"profile needs {want!r} but it is not available here; refusing to "
            "train on a fallback device"
        )

    # Power checks only apply to the macOS local path. Remote GPU nodes do not expose
    # pmset and should not emit laptop-specific warnings.
    if sys.platform == "darwin":
        ps = power_state()
        if ps == "battery":
            problems.append(
                "running on battery: multi-day training must be on AC power "
                "(runner wraps commands in caffeinate; that cannot help a dying battery)"
            )
        elif ps == "ac":
            print("  power OK: AC")
        else:
            print("  WARNING: could not confirm AC power; plug the machine in for long runs")

    # disk
    results_base = ROOT / meta.get("results_dir", "results/local_release")
    free_gb = shutil.disk_usage(results_base if results_base.exists() else ROOT).free / 2**30
    reserve = meta.get("disk_reserve_gb", 80)
    if free_gb < reserve:
        problems.append(
            f"only {free_gb:.0f} GiB free; the runner wants >= {reserve} GiB reserved "
            "for checkpoints, snapshots, probs and the export dir"
        )
    else:
        print(f"  disk OK: {free_gb:.0f} GiB free")
    return problems


# ---------------------------------------------------------------------------
# status bookkeeping
# ---------------------------------------------------------------------------


def status_read(status_file: Path) -> dict:
    if status_file.is_file():
        return json.loads(status_file.read_text())
    return {"version": 1, "stages": {}}


def status_mark(profile: str, phase: str, state: str, rc: int, log_rel: str | None,
                status_file: Path) -> None:
    st = status_read(status_file)
    st["stages"].setdefault(profile, {})[phase] = {
        "state": state,
        "rc": rc,
        "finished": now(),
        "log": log_rel,
    }
    st["updated"] = now()
    atomic_write_json(status_file, st)


# ---------------------------------------------------------------------------
# running one training stage
# ---------------------------------------------------------------------------


def run_command(cmd: list[str], log_path: Path) -> int:
    """Run cmd (no shell) with stdout/stderr streamed to the console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  cmd: " + shlex.join(cmd))
    with open(log_path, "ab") as fh:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line.encode("utf-8", "replace"))
            fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    return proc.wait()


def macos_wrap(cmd: list[str]) -> list[str]:
    """Prevent idle sleep for an unattended run (macOS only)."""
    if sys.platform != "darwin":
        return cmd
    return ["caffeinate", "-i", "-s", *cmd]


def run_training_stage(cfg: dict, profile: str, step: dict, log_path: Path,
                       resume: bool) -> int:
    """Launch one pretrain/finetune stage; never overwrites incompatible output."""
    state, detail = stage_status(cfg, profile, step)
    desired = step_desired(cfg, profile, step)
    if state == "complete":
        print(f"  already complete ({detail}) — nothing to run")
        return 0
    if state == "resumable" and not resume:
        raise ValueError(
            f"stage output exists and is interrupted ({detail}). Re-run with "
            "--resume to continue it (saved args match this profile), or move it "
            "aside to start fresh."
        )
    if state in ("incompatible", "unexpected"):
        raise ValueError(
            f"refusing to touch {rel_of(step_out_dir(cfg, profile, step))}: {detail}. "
            "Move it aside before running a different recipe."
        )
    # state == 'empty': fresh launch (or only transient logs). Passing --resume to
    # the scripts is harmless on a fresh start and required on a real resume.
    cmd = macos_wrap(replay_argv(step["script"], desired))
    rc = run_command(cmd, log_path)
    # reclassify after the run so status reflects the real files
    state, _detail = stage_status(cfg, profile, step)
    log_rel = rel_of(log_path)
    if rc == 0 and state == "complete":
        status_mark(profile, step["phase"], "done", rc, log_rel, status_file_for(cfg))
    else:
        status_mark(profile, step["phase"], "failed", rc, log_rel, status_file_for(cfg))
    if rc != 0:
        raise ValueError(
            f"stage [{step['phase']}] {step['name']} failed (rc {rc}); "
            f"log: {log_rel}. Re-run with --resume to continue from its snapshot."
        )
    return 0


def export_complete(export_dir: Path) -> bool:
    return all(
        (export_dir / f).is_file()
        for f in ("metadata.json", "environment.json", "labels.json",
                  "README.md", "MODEL_CARD.md", "inference.py", "LICENSE",
                  "requirements.txt", "inference_config.json", ".gitignore")
    ) and (export_dir / "model" / "model.pt").is_file() \
      and (export_dir / "binprov" / "model.py").is_file()


def run_export(cfg: dict, profile: str, log_path: Path) -> int:
    """Build the self-contained HF-ready export dir from the finetune output."""
    prof = cfg["profiles"][profile]
    ckpt = step_out_dir(cfg, profile, finetune_step(cfg, profile))
    if not finetune_complete(ckpt):
        raise ValueError(
            f"finetune checkpoint {rel_of(ckpt)} is not complete; run "
            "--phase finetune first"
        )
    export_dir = profile_dir(cfg, profile) / "export"
    if export_complete(export_dir):
        print(f"  already exported ({rel_of(export_dir)}) — nothing to do")
        return 0
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError(
            f"export dir {rel_of(export_dir)} exists but is incomplete; move it "
            "aside before rebuilding"
        )
    cmd = [
        PY, rel_of(SCRIPTS / "export_hf.py"),
        "--checkpoint", rel_of(ckpt),
        "--out", rel_of(export_dir),
        "--model-name", prof.get("model_name") or profile,
        "--corpus", cfg["meta"]["corpus_dir"],
    ]
    eval_json = profile_dir(cfg, profile) / "evaluate" / "report.json"
    if eval_json.is_file():
        cmd += ["--eval-json", rel_of(eval_json)]
    cmd = macos_wrap(cmd)
    rc = run_command(cmd, log_path)
    if rc == 0:
        status_mark(profile, "export", "done", rc, rel_of(log_path), status_file_for(cfg))
    else:
        status_mark(profile, "export", "failed", rc, rel_of(log_path), status_file_for(cfg))
        raise ValueError(f"export failed (rc {rc}); log: {rel_of(log_path)}")
    return 0


# ---------------------------------------------------------------------------
# evaluation (saved probabilities / result.json)
# ---------------------------------------------------------------------------


def evaluate_profile(cfg: dict, profile: str) -> dict:
    """Sequence- and binary-level accuracy from the saved test probabilities.

    Uses the exact probs.npz/result.json the finetune wrote (no re-running the
    model), and cross-checks the recomputed sequence accuracy against the
    recorded one.
    """
    import numpy as np

    from binprov.metrics import accuracy, per_class_prf

    step = finetune_step(cfg, profile)
    out = step_out_dir(cfg, profile, step)
    if not (out / "probs.npz").is_file() or not (out / "result.json").is_file():
        raise ValueError(
            f"{rel_of(out)} has no probs.npz/result.json; run the finetune stage first"
        )
    npz = np.load(out / "probs.npz")
    res = json.loads((out / "result.json").read_text())
    prob = npz["prob"].astype(np.float64)
    mmap = npz["marginal_map"]
    if mmap.size:  # factorized run: sum the fine classes onto the coarse ones
        coarse = np.zeros((prob.shape[0], int(mmap.max()) + 1), dtype=prob.dtype)
        for fine, c in enumerate(mmap.tolist()):
            coarse[:, c] += prob[:, fine]
        prob = coarse
        lut = np.asarray(mmap, dtype=np.int64)
        true = lut[npz["true"].astype(np.int64)]
    else:
        true = npz["true"].astype(np.int64)
    bid = npz["bid"].astype(np.int64)
    pred = prob.argmax(1)

    n = prob.shape[1]
    seq_acc = float(accuracy(pred, true))
    prf = per_class_prf(pred, true, n)
    sup = prf["support"] > 0
    bal = float(prf["recall"][sup].mean()) if sup.any() else 0.0

    uniq, inv = np.unique(bid, return_inverse=True)
    tally = np.zeros((len(uniq), n), dtype=prob.dtype)
    np.add.at(tally, inv, prob)
    truth = np.full(len(uniq), -1, dtype=np.int64)
    truth[inv] = true
    bpred = tally.argmax(1)
    bin_acc = float(accuracy(bpred, truth))

    recorded = None
    try:
        recorded = float(res["test"]["accuracy"])
    except (KeyError, TypeError):
        pass
    report = {
        "profile": profile,
        "task": res.get("task"),
        "scored_classes": res.get("scored_classes"),
        "sequence_accuracy": round(seq_acc, 6),
        "balanced_accuracy": round(bal, 6),
        "binary_accuracy": round(bin_acc, 6),
        "num_sequences": int(len(true)),
        "num_binaries": int(len(uniq)),
        "recorded_test_accuracy": recorded,
        "recomputed_matches_recorded": (
            recorded is not None and abs(recorded - seq_acc) < 1e-9
        ),
        "checkpoint": rel_of(out),
        "result_json": res.get("test"),
        "evaluated_at": now(),
    }
    eval_dir = profile_dir(cfg, profile) / "evaluate"
    atomic_write_json(eval_dir / "report.json", report)
    return report


def print_eval(report: dict) -> None:
    seq = report["sequence_accuracy"]
    bal = report["balanced_accuracy"]
    binacc = report["binary_accuracy"]
    print(f"\n  {report['profile']} ({report['task']}):")
    print(f"    sequence accuracy  {seq * 100:.4f}%  (n={report['num_sequences']:,})")
    print(f"    balanced accuracy  {bal * 100:.4f}%")
    print(f"    binary accuracy    {binacc * 100:.4f}%  (n={report['num_binaries']} binaries)")
    if report["recorded_test_accuracy"] is not None:
        ok = "match" if report["recomputed_matches_recorded"] else "MISMATCH"
        print(f"    recorded test acc  {report['recorded_test_accuracy'] * 100:.4f}% ({ok})")


# ---------------------------------------------------------------------------
# plan / rendering
# ---------------------------------------------------------------------------


def build_units(cfg: dict, profile: str, phase: str) -> list[dict]:
    """Ordered execution units for one profile filtered to the requested phase."""
    units: list[dict] = []
    for step in profile_steps(cfg, profile):
        units.append({"kind": "train", "phase": step["phase"], "step": step})
    units.append({
        "kind": "evaluate", "phase": "evaluate", "step": finetune_step(cfg, profile),
    })
    units.append({"kind": "export", "phase": "export", "step": finetune_step(cfg, profile)})

    if phase == "all":
        return units
    wanted = {phase}
    sel = [u for u in units if u["phase"] in wanted]
    if not sel:
        raise ValueError(
            f"phase {phase!r} is not part of profile {profile!r}; phases: "
            + ", ".join(u["phase"] for u in units)
        )
    return sel


def missing_inputs(cfg: dict, profile: str, unit: dict) -> list[str]:
    """Human-readable blockers: prerequisite checkpoints not yet complete."""
    out: list[str] = []
    step = unit["step"]
    if unit["kind"] == "train":
        for s in profile_steps(cfg, profile):
            if s["name"] == step.get("init_from_step"):
                state, _detail = stage_status(cfg, profile, s)
                if state != "complete":
                    out.append(
                        f"{rel_of(step_out_dir(cfg, profile, s))} "
                        f"(prerequisite {s['phase']} not complete)"
                    )
        state, _ = stage_status(cfg, profile, step)
        if state not in ("empty", "resumable"):
            out.append("stage output is not runnable (see status above)")
    if unit["kind"] == "evaluate":
        fin = step_out_dir(cfg, profile, finetune_step(cfg, profile))
        if not (fin / "result.json").is_file():
            out.append(f"{rel_of(fin)} has no result.json; run --phase finetune first")
    if unit["kind"] == "export":
        if not finetune_complete(step_out_dir(cfg, profile, finetune_step(cfg, profile))):
            out.append("finetune checkpoint not complete; run --phase finetune first")
    return out


def show_plan(cfg: dict, profile: str, units: list[dict], execute: bool) -> None:
    prof = cfg["profiles"][profile]
    print(f"== BinProv local release: profile '{profile}' "
          f"({prof['description']}) ==")
    print(f"corpus: {cfg['meta']['corpus_dir']}   outputs under: "
          f"{cfg['meta']['results_dir']}/{profile}/")
    for i, unit in enumerate(units, 1):
        step = unit["step"]
        print(f"\n### [{unit['phase']}] {step.get('name', unit['phase'])} — "
              f"{step.get('title', step.get('phase'))}")
        if unit["kind"] == "train":
            state, detail = stage_status(cfg, profile, step)
            print(f"  status: {state} ({detail})")
            desired = step_desired(cfg, profile, step)
            print(f"  micro-batch {desired['batch_size']} x accum "
                  f"{desired['grad_accum']} = effective {desired['batch_size'] * desired['grad_accum']}"
                  f"  (archived effective {step['effective_batch']})")
            for m in missing_inputs(cfg, profile, unit):
                print(f"  [missing input: {m}]")
            cmd = macos_wrap(replay_argv(step["script"], desired))
            print("  cmd: " + shlex.join(cmd))
            if execute and state not in ("empty", "resumable") and state != "complete":
                print("  [will block --execute]")
        elif unit["kind"] == "evaluate":
            fin = rel_of(step_out_dir(cfg, profile, step))
            print(f"  reads {fin}/probs.npz + result.json -> "
                  f"{cfg['meta']['results_dir']}/{profile}/evaluate/report.json")
            for m in missing_inputs(cfg, profile, unit):
                print(f"  [missing input: {m}]")
        else:
            fin = rel_of(step_out_dir(cfg, profile, step))
            print(f"  builds self-contained export dir from {fin} -> "
                  f"{cfg['meta']['results_dir']}/{profile}/export/")
            for m in missing_inputs(cfg, profile, unit):
                print(f"  [missing input: {m}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_profiles(cfg: dict) -> None:
    print("Local-training profiles")
    print(f"{'profile':24s} {'task':6s} {'recommended':12s} phases")
    for name, p in cfg["profiles"].items():
        phases = ", ".join(s["phase"] for s in p["steps"])
        print(f"{name:24s} {p['task']:6s} {'yes' if p.get('recommended') else '':12s} "
              f"{phases} + evaluate + export")
    print("\nPhases per profile (run in dependency order):")
    for name, p in cfg["profiles"].items():
        print(f"  {name}: " + " -> ".join(s["phase"] for s in p["steps"])
              + " -> evaluate -> export")
    print("\nDry-run by default; add --execute to run. --resume continues an "
          "interrupted stage whose saved args match the profile.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bounded local training + HF release prep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default=None,
                    help="path to JSON config (default: configs/local_release.json)")
    ap.add_argument("--profile", default=None,
                    help="profile name from --list (default: the recommended one)")
    ap.add_argument("--phase", choices=PHASES + ("all",), default="all",
                    help="which stage(s) to show/run")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print only (the default)")
    mode.add_argument("--execute", action="store_true",
                      help="actually run (default is a dry run: print, mutate nothing)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted stage output whose saved args match "
                         "the profile (required to touch an existing partial output)")
    ap.add_argument("--list", action="store_true", help="list profiles and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.list:
        list_profiles(cfg)
        return 0

    if args.profile is None:
        args.profile = next(n for n, p in cfg["profiles"].items() if p.get("recommended"))
    if args.profile not in cfg["profiles"]:
        print(f"error: unknown profile '{args.profile}'. Available: "
              + ", ".join(cfg["profiles"]), file=sys.stderr)
        return 2

    execute = args.execute
    print(f"profile: {args.profile}   phase: {args.phase}   "
          f"{'EXECUTE' if execute else 'dry-run'}")
    problems = preflight(cfg, args.profile, execute)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        if execute:
            return 2
        print("\n  (problems above would block --execute)")

    try:
        units = build_units(cfg, args.profile, args.phase)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not execute:
        show_plan(cfg, args.profile, units, execute=False)
        print("\ndry-run complete: nothing was created or executed. "
              "Re-run with --execute to run the plan.")
        return 0

    show_plan(cfg, args.profile, units, execute=True)
    if problems:
        return 2

    logs_base = profile_dir(cfg, args.profile) / cfg["meta"]["logs_dir"]
    try:
        for unit in units:
            missing = missing_inputs(cfg, args.profile, unit)
            if missing:
                raise ValueError(
                    f"missing input for [{unit['phase']}]: " + "; ".join(missing)
                )
            print(f"\n### [{unit['phase']}] {unit['step'].get('name', unit['phase'])}")
            if unit["kind"] == "train":
                log_path = logs_base / f"{unit['phase']}.log"
                run_training_stage(cfg, args.profile, unit["step"], log_path,
                                   resume=args.resume)
            elif unit["kind"] == "evaluate":
                report = evaluate_profile(cfg, args.profile)
                print_eval(report)
            else:
                log_path = logs_base / "export.log"
                run_export(cfg, args.profile, log_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
