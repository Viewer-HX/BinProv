"""Regression tests for scripts/train_local_release.py + scripts/export_hf.py.

Covers the three failure modes that matter for a local-release workflow: the
profile->effective-batch mapping silently drifting from the archived recipes,
the runner mutating anything on a dry run, and the export directory not
round-tripping (load + deterministic prediction). Run directly
(``python tests/test_local_release.py``) or under pytest.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("train_local_release")
export_hf = _load("export_hf")
cfg_all = runner.load_config()
# script parsers, for the "emitted command parses back to the recipe" check
parsers = {name: _load(name) for name in ("experiment", "pretrain_mlm")}


# ---------------------------------------------------------------------------
# profile -> effective-batch mapping (config is the contract with the archives)
# ---------------------------------------------------------------------------


def test_every_profile_step_preserves_the_archived_effective_batch():
    for pname, prof in cfg_all["profiles"].items():
        for step in prof["steps"]:
            # the adaptation must be exactly: micro-batch x accum -> archived batch
            assert step["micro_batch"] * step["grad_accum"] == step["effective_batch"], (
                pname, step["phase"])
            assert step["effective_batch"] == step["archived_effective"], (
                pname, step["phase"])
            assert step["workers"] == 0, pname
            assert step["grad_accum"] >= 1
            if step.get("archived_args"):
                arc = runner.read_json(step["archived_args"])
                # archived recipes had grad_accum 1, so archived batch == archived
                # effective batch
                assert arc["batch_size"] == step["archived_effective"], (
                    pname, step["phase"])
            # the runner's desired args must carry the same micro-batch mapping
            d = runner.step_desired(cfg_all, pname, step)
            assert d["batch_size"] == step["micro_batch"]
            assert d["grad_accum"] == step["grad_accum"]
            assert d["workers"] == step["workers"]


def test_runner_commands_parse_to_the_mapped_batches():
    """Every training command the runner emits must parse to the same recipe."""
    for pname, prof in cfg_all["profiles"].items():
        for step in prof["steps"]:
            d = runner.step_desired(cfg_all, pname, step)
            argv = runner.replay_argv(step["script"], d)
            script = Path(argv[1]).stem  # argv[0] is the interpreter
            cmd = ["local-run"] + argv[2:]
            with patch_or_none(sys, "argv", cmd):
                parsed = vars(parsers[script].parse_args())
            assert parsed["batch_size"] == step["micro_batch"], (pname, step["phase"])
            assert parsed["grad_accum"] == step["grad_accum"], (pname, step["phase"])
            assert parsed["workers"] == 0, (pname, step["phase"])
            assert parsed["seed"] == int(d.get("seed") or 0), (pname, step["phase"])


# ---------------------------------------------------------------------------
# dry run never mutates anything
# ---------------------------------------------------------------------------


def test_dry_run_is_mutation_free():
    release_dir = ROOT / "results" / "local_release"
    existed_before = release_dir.exists()
    with patch_or_none(runner.subprocess, "Popen", AssertionError("ran a command")), \
         patch_or_none(runner.subprocess, "run", _fake_pmset_ac):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
             contextlib.redirect_stderr(io.StringIO()):
            rc = runner.main(["--profile", "opt4_narrow_seed13", "--dry-run"])
    assert rc == 0
    if existed_before:
        assert release_dir.is_dir()  # never delete anything a user made
    else:
        assert not release_dir.exists(), "dry run created output under results/local_release"
    assert "nothing was created or executed" in out.getvalue()


def _fake_pmset_ac(*_a, **_k):
    import subprocess

    return subprocess.CompletedProcess([], 0, stdout="Now drawing from 'AC Power'\n")


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_or_none(container, attr, value):
    """patch.object when available, else a no-op context manager."""
    try:
        from unittest.mock import patch

        return patch.object(container, attr, value)
    except Exception:  # noqa: BLE001
        return _noop()


# ---------------------------------------------------------------------------
# resume safety: saved-args comparison, partial vs complete vs incompatible
# ---------------------------------------------------------------------------


def test_hf_checkpoint_accepts_safetensors_and_legacy_bin():
    """Transformers defaults to safetensors, while older runs used .bin."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        (out / "config.json").write_text("{}")
        assert not runner._has_hf_weights(out)
        (out / "model.safetensors").write_bytes(b"weights")
        assert runner._has_hf_weights(out)
        (out / "model.safetensors").unlink()
        (out / "pytorch_model.bin").write_bytes(b"weights")
        assert runner._has_hf_weights(out)


def _mini_cfg(tmp: Path) -> dict:
    """A single-profile config pointing its outputs into tmp (absolute path)."""
    return {
        "meta": {
            "results_dir": str(tmp / "lr"),
            "corpus_dir": cfg_all["meta"]["corpus_dir"],
            "split_evidence": cfg_all["meta"]["split_evidence"],
        },
        "profiles": {
            "p": {
                "task": "opt4",
                "steps": [
                    {
                        "phase": "finetune",
                        "name": "opt4",
                        "script": "experiment.py",
                        "archived_args":
                            "reports/runs/best/r9_opt4_base_seed13/args.json",
                        "overrides": {
                            "grad_accum": 8, "batch_size": 16, "workers": 0,
                            "resume": True, "tag": "p",
                        },
                        "out": "opt4",
                        "init_from_step": None,
                        "effective_batch": 128,
                    }
                ],
            }
        },
    }


def _write_desired_args(cfg, out: Path, step) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(
        json.dumps(runner.step_desired(cfg, "p", step), default=str)
    )


def test_complete_requires_weights_config_and_result():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = _mini_cfg(tmp)
        step = cfg["profiles"]["p"]["steps"][0]
        out = tmp / "lr" / "p" / "opt4"
        _write_desired_args(cfg, out, step)
        # partial files are NOT complete
        (out / "model.pt").write_bytes(b"x")
        (out / "binprov_config.json").write_text("{}")
        assert runner.stage_status(cfg, "p", step)[0] == "resumable"
        (out / "head.json").write_text("{}")
        assert runner.stage_status(cfg, "p", step)[0] == "resumable"
        (out / "result.json").write_text('{"test":{"accuracy":0.5}}')
        assert runner.stage_status(cfg, "p", step)[0] == "complete"


def test_incompatible_saved_args_are_refused_not_resumed():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = _mini_cfg(tmp)
        step = cfg["profiles"]["p"]["steps"][0]
        out = tmp / "lr" / "p" / "opt4"
        out.mkdir(parents=True)
        (out / "args.json").write_text(
            json.dumps({"batch_size": 999, "epochs": 2})  # a different recipe
        )
        state, _detail = runner.stage_status(cfg, "p", step)
        assert state == "incompatible"


def test_empty_and_transient_only_directories_are_fresh():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = _mini_cfg(tmp)
        step = cfg["profiles"]["p"]["steps"][0]
        out = tmp / "lr" / "p" / "opt4"
        assert runner.stage_status(cfg, "p", step)[0] == "empty"
        out.mkdir(parents=True)
        (out / "train_log.jsonl").write_text("x\n")
        assert runner.stage_status(cfg, "p", step)[0] == "empty"


# ---------------------------------------------------------------------------
# evaluate from saved probabilities
# ---------------------------------------------------------------------------


def test_evaluate_profile_reports_sequence_and_binary_accuracy():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg = _mini_cfg(tmp)
        step = cfg["profiles"]["p"]["steps"][0]
        out = tmp / "lr" / "p" / "opt4"
        out.mkdir(parents=True)
        # 4 binaries x 3 windows; binaries 0 and 3 are clean, 1 and 2 tie->O0
        bid = np.repeat(np.arange(4), 3)
        true = np.repeat([0, 1, 2, 3], 3)
        prob = np.zeros((12, 4))
        prob[np.arange(12), true] = 1.0
        prob[3:6, :] = 0.25  # binary 1: uniform -> votes O0, wrong (true O1)
        prob[6:9, :] = 0.25  # binary 2: uniform -> votes O0, wrong (true O2)
        np.savez(out / "probs.npz", prob=prob.astype(np.float32),
                 true=true.astype(np.int16), bid=bid,
                 marginal_map=np.zeros(0, dtype=np.int16))
        (out / "result.json").write_text(
            json.dumps({"task": "opt4", "scored_classes": ["O0", "O1", "O2", "O3"],
                        "test": {"accuracy": 0.5}})
        )
        report = runner.evaluate_profile(cfg, "p")
        # sequence: binaries 0 & 3 correct (6 of 12 windows) = 0.5
        assert report["sequence_accuracy"] == 0.5
        assert report["recomputed_matches_recorded"] is True
        # binary soft vote: clean binaries 0 & 3 vote right (2/4)
        assert report["binary_accuracy"] == 0.5
        assert report["num_binaries"] == 4


# ---------------------------------------------------------------------------
# export fixture: build, load, deterministic prediction
# ---------------------------------------------------------------------------


def _tiny_checkpoint(base: Path) -> Path:
    import torch  # noqa: F401

    from binprov.model import BinProvConfig, BinProvForProvenance

    ck = base / "ckpt"
    ck.mkdir(parents=True)
    cfg = BinProvConfig(seq_bytes=64, hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64)
    model = BinProvForProvenance(cfg, 4, pool="border", classifier_dropout=0.0)
    model.save(ck, extra={"task": "opt4", "tag": "fixture"})
    (ck / "args.json").write_text(json.dumps({"epochs": 1, "tag": "fixture"}))
    (ck / "result.json").write_text(
        json.dumps({"task": "opt4", "scored_classes": ["O0", "O1", "O2", "O3"],
                    "test": {"accuracy": 0.5}})
    )
    return ck


def test_export_dir_roundtrips_and_predicts_deterministically():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skipping export test (torch not installed)")
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        out = tmp / "export"
        eval_json = tmp / "report.json"
        eval_json.write_text(json.dumps({
            "sequence_accuracy": 0.5,
            "binary_accuracy": 0.75,
            "balanced_accuracy": 0.4,
            "num_sequences": 4,
            "num_binaries": 2,
        }))
        summary = export_hf.build_export(
            ck, out, model_name="binprov-fixture", eval_json=str(eval_json),
            corpus_dir=None
        )
        assert (out / "inference.py").is_file()
        assert (out / "MODEL_CARD.md").is_file()
        assert (out / "README.md").is_file()
        assert (out / "LICENSE").is_file()
        assert (out / "metadata.json").is_file()
        assert (out / "labels.json").is_file()
        assert (out / "requirements.txt").is_file()
        assert (out / "binprov" / "model.py").is_file()
        assert (out / "evidence" / "evaluation.json").is_file()
        assert (out / "model" / "model.pt").is_file()
        inference_cfg = json.loads((out / "model" / "inference_config.json").read_text())
        assert inference_cfg == {"seq_bytes": 64, "stride": 64, "min_bytes": 16}
        card = (out / "README.md").read_text()
        assert card.startswith("---\nlicense: mit\n")
        assert "language:\n- machine-code" not in card
        assert "pip install -r requirements.txt" in card
        # label map has the four opt classes
        labels = json.loads((out / "labels.json").read_text())
        assert labels["label_map"]["0"] == "O0"
        # verification actually reloaded and compared the export model
        assert summary["verification"]["verified"] is True
        assert summary["binary_accuracy"] == 0.75
        assert summary["verification"]["windows"] >= 1
        assert len(summary["verification"]["mean_prob"]) == 4


def test_release_inference_uses_overlapping_stride_and_bounded_batches():
    inference = _load("inference_template")
    windows = inference.cut_windows(bytes(2048), seq_bytes=2048, stride=512)
    assert [len(chunk) for chunk, _index in windows] == [2048, 1536]


def test_export_dir_rejects_rebuild_over_existing_output():
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ck = _tiny_checkpoint(tmp)
        out = tmp / "export"
        export_hf.build_export(ck, out, model_name="binprov-fixture", corpus_dir=None)
        # second build must refuse (never clobber an existing export)
        try:
            export_hf.build_export(ck, out, model_name="binprov-fixture",
                                   corpus_dir=None)
        except SystemExit:
            return
        raise AssertionError("export must refuse to overwrite an existing dir")


# ---------------------------------------------------------------------------
# --config flag and alternate config loading
# ---------------------------------------------------------------------------

GPU_CFG = ROOT / "configs" / "gpu_release.json"


def test_config_flag_loads_default_when_omitted():
    """Omitting --config loads configs/local_release.json."""
    cfg = runner.load_config()
    assert "profiles" in cfg
    assert "opt4_narrow_seed13" in cfg["profiles"]


def test_config_flag_loads_gpu_release():
    """--config configs/gpu_release.json loads the CUDA profile."""
    cfg = runner.load_config(str(GPU_CFG))
    assert "opt4_wide_seed29" in cfg["profiles"]
    assert cfg["meta"]["backend"] == "cuda"
    assert cfg["meta"]["results_dir"] == "results/gpu_release"


def test_config_flag_rejects_missing_file():
    """A nonexistent config path raises FileNotFoundError."""
    try:
        runner.load_config("/nonexistent/path.json")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_gpu_config_batch_invariants():
    """Every GPU step's effective batch equals micro-batch times accumulation."""
    cfg = runner.load_config(str(GPU_CFG))
    for pname, prof in cfg["profiles"].items():
        for step in prof["steps"]:
            assert step["micro_batch"] * step["grad_accum"] == step["effective_batch"], (
                pname, step["phase"])
            assert step["effective_batch"] == step["archived_effective"], (
                pname, step["phase"])
            assert step["workers"] == 6, pname


def test_gpu_config_dry_run():
    """Dry-run with --config gpu_release.json produces no output."""
    if not GPU_CFG.is_file():
        print("  skipping: configs/gpu_release.json not present")
        return
    with patch_or_none(runner.subprocess, "Popen", AssertionError("ran a command")), \
         patch_or_none(runner.subprocess, "run", _fake_pmset_ac):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
             contextlib.redirect_stderr(io.StringIO()):
            rc = runner.main(["--config", str(GPU_CFG), "--dry-run"])
    assert rc == 0
    assert "nothing was created or executed" in out.getvalue()
    assert "gpu_release" in out.getvalue()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
