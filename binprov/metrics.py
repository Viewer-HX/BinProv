"""Metrics and table formatting.

The paper reports accuracy as its headline number, because the dataset is
balanced across compilers and optimization levels (§4.2), and adds
precision/recall/F1 per optimization level to expose *where* the O2/O3
confusion sits (Table 5). Both are here, with no sklearn dependency so that
evaluation runs in any environment that can run the model.
"""

from __future__ import annotations

import numpy as np


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) == 0:
        return 0.0
    return float((pred == true).mean())


def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are ground truth, columns are predictions."""
    flat = true.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_prf(pred: np.ndarray, true: np.ndarray, num_classes: int) -> dict:
    """Precision / recall / F1 per class, plus macro averages (Table 5)."""
    cm = confusion_matrix(pred, true, num_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual.astype(np.int64),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "confusion": cm,
    }


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table with columns padded to a readable width."""
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def confusion_as_markdown(cm: np.ndarray, classes) -> str:
    header = ["true \\ pred", *classes]
    rows = [[classes[i], *[str(int(v)) for v in cm[i]]] for i in range(len(classes))]
    return markdown_table(header, rows)
