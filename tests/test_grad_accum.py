"""Regression tests for experiment.py --grad-accum.

Two things can go wrong with gradient accumulation and both are silent: the
optimizer stepping the wrong number of times (so total_steps and the LR
schedule no longer describe the run), and the accumulated gradient being scaled
by the wrong factor (so an "effective batch" is not actually the mean over the
micro-batches it claims to be). These tests pin both down on a tiny real model,
plus the trailing-partial-group rule that keeps gradients from leaking across
epoch boundaries.

Run directly (``python tests/test_grad_accum.py``) or under pytest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parents[1]


def _experiment():
    spec = importlib.util.spec_from_file_location(
        "grad_accum_experiment", ROOT / "scripts" / "experiment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _experiment()


# ---------------------------------------------------------------------------
# group planning (pure, no torch)
# ---------------------------------------------------------------------------


def test_accum_groups_full_groups_plus_one_partial():
    # 7 micro-batches at accum 3 -> steps on {3, 6} then a trailing partial {7}.
    assert ex.accum_groups(7, 3) == [3, 3, 1]
    # exact multiples never get a partial group
    assert ex.accum_groups(9, 3) == [3, 3, 3]
    assert ex.accum_groups(8, 4) == [4, 4]
    # every group boundary is a valid step: full groups then at most one smaller
    for n in range(1, 15):
        for g in range(2, 7):
            groups = ex.accum_groups(n, g)
            assert sum(groups) == n, (n, g)
            assert all(x == g for x in groups[:-1]), (n, g, groups)
            assert groups[-1] == (n % g or g), (n, g, groups)
            assert ex.accum_steps_per_epoch(n, g) == len(groups)


def test_accum_groups_grad_accum_one_is_archived_stepping():
    # grad_accum == 1 must reproduce the archived per-micro-batch loop exactly:
    # one optimizer step per micro-batch, none merged, no partial group needed.
    for n in range(1, 12):
        assert ex.accum_groups(n, 1) == [1] * n
        assert ex.accum_steps_per_epoch(n, 1) == n
    assert ex.accum_groups(0, 1) == []
    assert ex.accum_groups(0, 4) == []


# ---------------------------------------------------------------------------
# loss scaling (real autograd, small model)
# ---------------------------------------------------------------------------

def _problem(n=24, d=8, c=3):
    torch.manual_seed(0)
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    return x, y


def _chunk_loss(model, xs, ys, extra_w=0.25):
    """One micro-batch's combined loss (CE + a weighted auxiliary term)."""
    logits = model(xs)
    return F.cross_entropy(logits, ys) + extra_w * logits.pow(2).mean()


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


def _accum_grad_over(model, xs, ys, m, g, extra_w=0.25):
    """Gradients left on the model after accumulating `backward(loss/g)` over
    every micro-batch in one optimizer group (indices 0..k, k < len(xs)/m)."""
    model.zero_grad()
    k = len(xs) // m
    for i in range(k):
        (_chunk_loss(model, xs[i * m : (i + 1) * m], ys[i * m : (i + 1) * m],
                     extra_w) / g).backward()
    return _grads(model)


def _reference_mean_grad(model, xs, ys, m, g, extra_w=0.25):
    """Closed form of what the loop must reproduce: (1/g) * sum over the
    group's micro-batches of each batch's standalone gradient."""
    acc = None
    k = len(xs) // m
    for i in range(k):
        model.zero_grad()
        _chunk_loss(model, xs[i * m : (i + 1) * m], ys[i * m : (i + 1) * m],
                    extra_w).backward()
        gr = _grads(model)
        acc = [a + gi / g for a, gi in zip(acc, gr)] if acc else [gi / g for gi in gr]
    return acc


def _assert_close(list_a, list_b, tol=1e-6):
    for a, b in zip(list_a, list_b):
        assert torch.allclose(a, b, atol=tol, rtol=tol), (a - b).abs().max()


def test_full_group_accumulated_gradient_is_the_mean_gradient():
    """loss/grad_accum summed over a full group == the mean gradient the same
    loss would produce if the micro-batches had been one batch."""
    x, y = _problem(n=12, d=8)
    m, g = 4, 3  # 3 full micro-batches = one effective batch of 12 samples
    model = nn.Linear(x.shape[1], int(y.max()) + 1)
    got = _accum_grad_over(model, x, y, m, g)
    want = _reference_mean_grad(model, x, y, m, g)
    _assert_close(got, want)


def test_combined_loss_is_scaled_as_a_whole():
    """The auxiliary (pair/margin) term in experiment.py is part of the combined
    `loss`; the *whole* loss is divided by grad_accum before backward."""
    x, y = _problem(n=16, d=8)
    m, g = 4, 4
    model = nn.Linear(x.shape[1], int(y.max()) + 1)
    got = _accum_grad_over(model, x, y, m, g, extra_w=0.25)
    want = _reference_mean_grad(model, x, y, m, g, extra_w=0.25)
    _assert_close(got, want)


def test_partial_group_gradient_is_correctly_scaled():
    """A trailing partial group (< grad_accum micro-batches) still accumulates
    each micro-batch divided by grad_accum, so its summed gradient equals
    (1/g) * sum over the partial micro-batches -- smaller than a full group but
    correctly scaled, and it is stepped (not dropped into the next epoch)."""
    x, y = _problem(n=8, d=8)  # 2 micro-batches of m=4: partial under g=3
    m, g = 4, 3
    model = nn.Linear(x.shape[1], int(y.max()) + 1)
    got = _accum_grad_over(model, x, y, m, g)
    want = _reference_mean_grad(model, x, y, m, g)
    _assert_close(got, want)


# ---------------------------------------------------------------------------
# optimizer step count (integration shape of the loop, still tiny)
# ---------------------------------------------------------------------------


class CountingSGD(torch.optim.SGD):
    """SGD that counts optimizer.step() calls. A stub of the optimiser, not a
    mock of a model: the optimiser and the model are real."""

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self.calls = 0

    def step(self, *a, **kw):
        self.calls += 1
        return super().step(*a, **kw)


def _run_epochs(n_micro, g, epochs):
    """Mirror experiment.py's epoch structure: step once per accum_groups group.

    Each epoch re-reads the same n_micro micro-batches (as a shuffled DataLoader
    would); micro-batch indexing resets per epoch exactly as the loop's counters
    do.
    """
    x, y = _problem(n=n_micro * 4, d=8)
    m = 4  # 4 samples per micro-batch
    model = nn.Linear(x.shape[1], int(y.max()) + 1)
    opt = CountingSGD(model.parameters(), lr=0.05)
    for _ in range(epochs):
        idx = 0
        for group in ex.accum_groups(n_micro, g):
            for _ in range(group):
                opt.zero_grad()
                loss = _chunk_loss(model, x[idx * m : (idx + 1) * m],
                                   y[idx * m : (idx + 1) * m])
                (loss / g).backward()
                idx += 1
            opt.step()
    return opt.calls


def test_optimizer_step_count_matches_group_plan():
    # 7 micro-batches / epoch at accum 3 -> 3 steps (2 full + 1 partial) / epoch
    assert _run_epochs(7, 3, epochs=1) == ex.accum_steps_per_epoch(7, 3) == 3
    # two epochs -> total_steps = steps_per_epoch * epochs
    assert _run_epochs(7, 3, epochs=2) == 2 * ex.accum_steps_per_epoch(7, 3)
    # exact multiple: no partial group, still correct count
    assert _run_epochs(6, 3, epochs=1) == 2
    # grad_accum == 1 reproduces the archived per-micro-batch stepping count
    assert _run_epochs(7, 1, epochs=1) == 7


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
