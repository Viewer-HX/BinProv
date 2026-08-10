"""BinProv's networks: a byte-level RoBERTa encoder and a provenance classifier.

Two pieces, matching the paper's two training phases:

* :func:`build_mlm_model` — the embedding model ``f_e`` with a softmax head,
  pre-trained on masked language modeling (§3.2, Eq. 1). 12 transformer layers,
  hidden size 768, which lands at ~85M trainable parameters — the paper's
  "around 80 million".
* :class:`BinProvForProvenance` — ``f_c(f_e(...))``, the encoder plus the
  two-layer fully connected classifier of §3.3, fine-tuned end-to-end.

The paper describes the classifier's first layer as one that "reshape[s] the
input vectors and weaken[s] the border weights of the embeddings", to cope with
x86's variable-length instructions being cut mid-instruction at a sequence
boundary. It does not give the mechanism. :class:`BorderTaperedPool` implements
it as a learned attention over token positions whose logits are *initialised*
with a taper at both ends: borders start down-weighted, and training is free to
change that. Ablate it with ``--pool mean`` to see what it buys.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModel

from .vocab import BOS_ID, EOS_ID, MASK_ID, PAD_ID, VOCAB_SIZE


@dataclass
class BinProvConfig:
    """Architecture knobs. Stored next to every checkpoint."""

    seq_bytes: int = 512  # bytes per sequence (paper §3.1)
    hidden_size: int = 768
    num_hidden_layers: int = 12  # "consists of 12 transformers" (§3.2)
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_segments: int = 2  # segment embedding E_s (§3.2)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    @property
    def seq_tokens(self) -> int:
        """Sequence length in tokens: the bytes plus <s> and </s>."""
        return self.seq_bytes + 2

    def to_roberta(self) -> RobertaConfig:
        return RobertaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            # HF's RoBERTa derives position ids as padding_idx + 1 + i, so the
            # table needs two slots of headroom beyond the token count.
            max_position_embeddings=self.seq_tokens + 2,
            type_vocab_size=self.max_segments,
            pad_token_id=PAD_ID,
            bos_token_id=BOS_ID,
            eos_token_id=EOS_ID,
            mask_token_id=MASK_ID,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            position_embedding_type="absolute",
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> BinProvConfig:
        return cls(**json.loads(Path(path).read_text()))


def build_mlm_model(cfg: BinProvConfig) -> RobertaForMaskedLM:
    """The pre-training model: encoder + softmax head over the 261-byte vocab."""
    return RobertaForMaskedLM(cfg.to_roberta())


class BorderTaperedPool(nn.Module):
    """Pool token embeddings into one vector, down-weighting sequence borders.

    This is the first of the classifier's two layers (§3.3). Weights are a
    softmax over learned per-position logits, masked to the valid tokens, so
    padding and the <s>/</s> markers contribute nothing. Logits start as a
    linear ramp over the outermost ``taper`` byte positions, encoding the prior
    that a byte near the cut is likely part of a broken instruction.
    """

    def __init__(self, n_positions: int, hidden_size: int, taper: int = 32, taper_floor: float = -2.0):
        super().__init__()
        logits = torch.zeros(n_positions)
        taper = max(0, min(taper, n_positions // 2))
        if taper:
            ramp = torch.linspace(taper_floor, 0.0, taper + 1)[:-1]
            logits[:taper] = ramp
            logits[n_positions - taper :] = ramp.flip(0)
        self.pos_logits = nn.Parameter(logits)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, H); attention_mask: (B, T) with 1 on real tokens
        n = hidden.size(1)
        logits = self.pos_logits[:n].unsqueeze(0).expand(hidden.size(0), -1)
        logits = logits.masked_fill(attention_mask == 0, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        return self.norm(F.gelu(self.proj(pooled)))


class MeanPool(nn.Module):
    """Plain masked mean pooling — the ablation baseline for the head."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.norm(F.gelu(self.proj(pooled)))


class BinProvForProvenance(nn.Module):
    """Encoder + classifier, i.e. Eq. (2) of the paper.

    ``y_hat = f_c(f_e(concat(E_b, E_p, E_s)))``
    """

    def __init__(
        self,
        cfg: BinProvConfig,
        num_labels: int,
        *,
        pool: str = "border",
        classifier_dropout: float = 0.1,
        taper: int = 32,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        self.pool_kind = pool
        self.encoder = RobertaModel(cfg.to_roberta(), add_pooling_layer=False)
        if pool == "border":
            self.pool = BorderTaperedPool(cfg.seq_tokens, cfg.hidden_size, taper=taper)
        elif pool == "mean":
            self.pool = MeanPool(cfg.hidden_size)
        elif pool == "cls":
            self.pool = None
        else:
            raise ValueError(f"unknown pool {pool!r} (border|mean|cls)")
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(cfg.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state  # E_final, shape (B, T, H)
        pooled = hidden[:, 0] if self.pool is None else self.pool(hidden, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)
        return {"loss": loss, "logits": logits}

    # -- checkpointing -----------------------------------------------------

    def load_encoder_from_mlm(self, mlm_dir: str | Path, *, strict: bool = False) -> list[str]:
        """Warm-start the encoder from an MLM checkpoint (the transfer step).

        Returns the list of encoder parameters that were *not* found, which
        should be empty for a matching config — worth asserting in a run log,
        since a silent mismatch here would quietly turn fine-tuning into
        training from scratch.
        """
        mlm = RobertaForMaskedLM.from_pretrained(str(mlm_dir))
        missing, unexpected = self.encoder.load_state_dict(
            mlm.roberta.state_dict(), strict=strict
        )
        del mlm
        if strict and (missing or unexpected):
            raise RuntimeError(f"encoder mismatch: missing={missing} unexpected={unexpected}")
        return list(missing)

    def save(self, out_dir: str | Path, *, extra: dict | None = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "model.pt")
        self.cfg.save(out / "binprov_config.json")
        (out / "head.json").write_text(
            json.dumps(
                {
                    "num_labels": self.num_labels,
                    "pool": self.pool_kind,
                    **(extra or {}),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, ckpt_dir: str | Path, *, map_location="cpu") -> tuple[BinProvForProvenance, dict]:
        d = Path(ckpt_dir)
        cfg = BinProvConfig.load(d / "binprov_config.json")
        head = json.loads((d / "head.json").read_text())
        model = cls(cfg, head["num_labels"], pool=head.get("pool", "border"))
        state = torch.load(d / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model, head


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    n = count_parameters(model)
    return f"{type(model).__name__}: {n:,} trainable parameters ({n / 1e6:.1f}M)"


def param_groups(model: nn.Module, weight_decay: float = 0.01):
    """AdamW groups that exempt biases and LayerNorm from weight decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    # Empty groups are dropped: a frozen encoder would otherwise hand AdamW a
    # group with no parameters.
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.02):
    """Linear warmup then cosine decay — standard for MLM pre-training."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
