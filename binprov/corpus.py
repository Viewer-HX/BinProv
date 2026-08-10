"""The packed corpus: the only thing BinProv needs to keep on disk.

Motivation is disk space. A BinKit binary averages ~200 KB, but BinProv reads
nothing except the ``.text`` bytes and (for function-level voting) the function
boundaries. So corpus building concatenates every ``.text`` section into one
flat ``uint8`` file and records offsets alongside — after which **the binaries
themselves can be deleted**. In practice that is a 3-5x reduction versus keeping
the ELF files, and it makes training I/O a single memory-mapped sequential read.

On-disk layout of a corpus directory::

    text.u8            all .text sections concatenated, no padding
    binaries.jsonl.gz  one JSON record per binary (offset, length, labels, funcs)
    meta.json          corpus-level counts, label tallies, build parameters
    splits/<name>.json binary-id lists for train/test/pretrain

Sequences (the paper's fixed-length 512-byte units, §3.1) are *not* stored.
They are cut on the fly by :meth:`Corpus.sequences`, so changing the sequence
length or stride costs nothing and needs no rebuild.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TEXT_FILE = "text.u8"
INDEX_FILE = "binaries.jsonl.gz"
META_FILE = "meta.json"
SPLIT_DIR = "splits"


@dataclass
class BinaryRecord:
    """One binary's slice of the packed corpus."""

    bid: int
    path: str  # original path, for traceability only
    package: str  # e.g. "coreutils-8.24"
    program: str  # e.g. "ls" — the grouping key for leak-free splits
    labels: dict  # see provenance.Labels.as_dict()
    text_off: int  # offset into text.u8
    text_len: int
    text_vaddr: int  # virtual address of .text[0], for symbol mapping
    arch_elf: str  # architecture as read from the ELF header
    stripped: bool
    functions: list = field(default_factory=list)  # [[off, size], ...] within .text
    func_names: list = field(default_factory=list)  # parallel to functions, may be empty

    @property
    def group(self) -> str:
        """Split-grouping key: all compiled variants of one program share it."""
        return f"{self.package}/{self.program}"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


class CorpusWriter:
    """Streams ``.text`` sections into a packed corpus.

    Use as a context manager. ``text.u8`` is appended to as we go, so peak
    memory stays at one binary regardless of corpus size.
    """

    def __init__(self, out_dir: str | os.PathLike, *, build_params: dict | None = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._text = open(self.dir / TEXT_FILE, "wb")
        self._index = gzip.open(self.dir / INDEX_FILE, "wt", compresslevel=6)
        self._offset = 0
        self._n = 0
        self._build_params = dict(build_params or {})
        self._label_counts: dict[str, dict[str, int]] = {}

    def add(
        self,
        text: bytes,
        *,
        path: str,
        package: str,
        program: str,
        labels: dict,
        text_vaddr: int,
        arch_elf: str,
        stripped: bool,
        functions: list | None = None,
        func_names: list | None = None,
    ) -> BinaryRecord:
        rec = BinaryRecord(
            bid=self._n,
            path=path,
            package=package,
            program=program,
            labels=labels,
            text_off=self._offset,
            text_len=len(text),
            text_vaddr=text_vaddr,
            arch_elf=arch_elf,
            stripped=stripped,
            functions=functions or [],
            func_names=func_names or [],
        )
        self._text.write(text)
        self._index.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
        self._offset += len(text)
        self._n += 1
        for key in ("arch", "compiler", "opt", "extra", "obfuscation"):
            val = labels.get(key)
            if val is not None:
                self._label_counts.setdefault(key, {})
                self._label_counts[key][val] = self._label_counts[key].get(val, 0) + 1
        return rec

    @property
    def num_binaries(self) -> int:
        return self._n

    @property
    def num_bytes(self) -> int:
        return self._offset

    def close(self) -> None:
        self._text.close()
        self._index.close()
        meta = {
            "num_binaries": self._n,
            "num_text_bytes": self._offset,
            "label_counts": self._label_counts,
            "build_params": self._build_params,
        }
        (self.dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


@dataclass
class SequenceIndex:
    """A set of fixed-length byte sequences cut out of a corpus.

    Arrays are parallel and all the same length:

        bid    which binary each sequence came from
        start  absolute offset into ``text.u8``
        length valid bytes (< seq_len only for the tail of a unit)
        unit   index into ``record.functions``, or -1 for a binary-level cut

    ``unit`` is what makes function-level majority voting (§3.4) possible: it
    identifies the entity a sequence votes for.
    """

    bid: np.ndarray
    start: np.ndarray
    length: np.ndarray
    unit: np.ndarray
    seq_len: int
    level: str  # "binary" or "function"

    def __len__(self) -> int:
        return int(self.bid.shape[0])

    def group_ids(self) -> np.ndarray:
        """Integer id of the entity each sequence votes for.

        Binary level: one group per binary. Function level: one group per
        (binary, function) pair.
        """
        if self.level == "binary":
            keys = self.bid.astype(np.int64)
        else:
            keys = self.bid.astype(np.int64) * (2**24) + (self.unit.astype(np.int64) + 1)
        # dense re-indexing keeps downstream bincount cheap
        _, inverse = np.unique(keys, return_inverse=True)
        return inverse.astype(np.int64)

    def subset(self, mask: np.ndarray) -> SequenceIndex:
        return SequenceIndex(
            bid=self.bid[mask],
            start=self.start[mask],
            length=self.length[mask],
            unit=self.unit[mask],
            seq_len=self.seq_len,
            level=self.level,
        )


class Corpus:
    """Read-only view over a packed corpus directory."""

    def __init__(self, path: str | os.PathLike):
        self.dir = Path(path)
        if not (self.dir / TEXT_FILE).exists():
            raise FileNotFoundError(f"{self.dir} is not a corpus (no {TEXT_FILE})")
        self.meta = json.loads((self.dir / META_FILE).read_text())
        self.records: list[BinaryRecord] = []
        with gzip.open(self.dir / INDEX_FILE, "rt") as fh:
            for line in fh:
                self.records.append(BinaryRecord(**json.loads(line)))
        # np.memmap keeps the whole thing out of RAM; the OS page cache does the
        # right thing for the sequential-ish access pattern of training.
        self.text = np.memmap(self.dir / TEXT_FILE, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        mb = self.text.shape[0] / 1e6
        return f"<Corpus {self.dir.name}: {len(self.records)} binaries, {mb:.1f} MB .text>"

    # -- selection ---------------------------------------------------------

    def filter_bids(self, **constraints) -> list[int]:
        """Binary ids whose labels match every constraint.

        A constraint value may be a scalar or a collection; ``None`` means "no
        constraint". Example::

            corpus.filter_bids(arch="x86_64", extra="normal", opt=("O2", "O3"))
        """
        out = []
        for rec in self.records:
            ok = True
            for key, want in constraints.items():
                if want is None:
                    continue
                got = rec.labels.get(key)
                if isinstance(want, (list, tuple, set, frozenset)):
                    if got not in want:
                        ok = False
                        break
                elif got != want:
                    ok = False
                    break
            if ok:
                out.append(rec.bid)
        return out

    # -- sequence cutting --------------------------------------------------

    def sequences(
        self,
        *,
        seq_len: int = 512,
        stride: int | None = None,
        level: str = "binary",
        bids: list[int] | None = None,
        min_bytes: int = 16,
        max_seqs_per_binary: int | None = None,
        function_context: bool = False,
        rng: random.Random | None = None,
    ) -> SequenceIndex:
        """Cut the corpus into fixed-length byte sequences.

        Args:
            seq_len: bytes per sequence (paper uses 512, a multiple of 8 so that
                fixed-width ARM/MIPS instructions stay aligned).
            stride: step between sequence starts. Defaults to ``seq_len``
                (non-overlapping, as in §3.1). A smaller stride is used for
                function-level evaluation, where most functions are shorter than
                one sequence and non-overlapping cuts would give nothing to vote
                over — see docs/REPRODUCTION.md.
            level: ``"binary"`` cuts the whole ``.text``; ``"function"`` cuts
                each function separately so no sequence straddles a boundary.
            min_bytes: drop trailing fragments shorter than this.
            max_seqs_per_binary: uniformly subsample when a binary is huge.
                Keeps a corpus-wide cap on training cost without biasing toward
                long binaries.
            function_context: at ``level="function"``, give a function shorter
                than ``seq_len`` a full-length window centred on it rather than
                its own bytes padded. See the comment at the use site: most
                functions are far shorter than one sequence, so without this the
                function level measures a harder input distribution than the
                sequence level rather than a voted one.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        step = stride or seq_len
        if step <= 0:
            raise ValueError("stride must be positive")
        if level not in ("binary", "function"):
            raise ValueError(f"level must be 'binary' or 'function', got {level!r}")

        wanted = self.records if bids is None else [self.records[b] for b in bids]
        rng = rng or random.Random(0)
        b_out: list[int] = []
        s_out: list[int] = []
        l_out: list[int] = []
        u_out: list[int] = []

        for rec in wanted:
            if level == "binary":
                spans = [(-1, 0, rec.text_len)]
            else:
                spans = [(i, off, size) for i, (off, size) in enumerate(rec.functions)]

            local: list[tuple[int, int, int]] = []  # (unit, start_abs, length)
            for unit, span_off, span_len in spans:
                if function_context and unit >= 0 and span_len < seq_len:
                    # A short function padded to seq_len sees far less context
                    # than an ordinary sequence, which makes it *harder*, not
                    # easier -- and in BinKit 81.5% of functions are shorter than
                    # 512 bytes (median 127), so this is the common case. Centre a
                    # full-length window on the function instead, borrowing
                    # neighbouring bytes and clamping to the section. The label is
                    # still the function's, and provenance is a property of the
                    # whole binary, so the borrowed bytes carry the same label.
                    centre = span_off + span_len // 2
                    start = max(0, min(centre - seq_len // 2, rec.text_len - seq_len))
                    n = min(seq_len, rec.text_len)
                    local.append((unit, rec.text_off + max(0, start), n))
                    continue
                pos = 0
                while pos < span_len:
                    n = min(seq_len, span_len - pos)
                    # keep a short unit whole (a 40-byte helper function is a
                    # legitimate sample), but drop a mere tail fragment
                    if n < min_bytes and pos > 0:
                        break
                    local.append((unit, rec.text_off + span_off + pos, n))
                    if n < seq_len:
                        break
                    pos += step
            if max_seqs_per_binary is not None and len(local) > max_seqs_per_binary:
                local = rng.sample(local, max_seqs_per_binary)
                local.sort(key=lambda t: t[1])
            for unit, start, n in local:
                b_out.append(rec.bid)
                s_out.append(start)
                l_out.append(n)
                u_out.append(unit)

        return SequenceIndex(
            bid=np.asarray(b_out, dtype=np.int32),
            start=np.asarray(s_out, dtype=np.int64),
            length=np.asarray(l_out, dtype=np.int32),
            unit=np.asarray(u_out, dtype=np.int32),
            seq_len=seq_len,
            level=level,
        )

    # -- splits ------------------------------------------------------------

    def make_splits(
        self,
        *,
        train_ratio: float = 0.8,
        seed: int = 1234,
        group_by: str = "program",
        bids: list[int] | None = None,
    ) -> dict[str, list[int]]:
        """Split binary ids into train/test.

        The paper splits 8:2 and states that train and test must not overlap so
        that generalization is measured honestly (§4.1). ``group_by="program"``
        goes one step further: every compiled variant of a program (all
        compilers, all optimization levels) lands on the same side, so the model
        cannot recognise a test program it has already seen at another
        optimization level. Use ``group_by="binary"`` for the looser
        binary-level split.
        """
        pool = self.records if bids is None else [self.records[b] for b in bids]
        rng = random.Random(seed)
        if group_by == "binary":
            keys = [str(r.bid) for r in pool]
        elif group_by == "program":
            keys = [r.group for r in pool]
        elif group_by == "package":
            keys = [r.package for r in pool]
        else:
            raise ValueError(f"unknown group_by {group_by!r}")

        uniq = sorted(set(keys))
        rng.shuffle(uniq)
        n_train = int(round(len(uniq) * train_ratio))
        train_keys = set(uniq[:n_train])
        train, test = [], []
        for rec, key in zip(pool, keys):
            (train if key in train_keys else test).append(rec.bid)
        return {"train": train, "test": test}

    def pretrain_bids(
        self,
        *,
        per_package: int = 1,
        restrict_to: list[int] | None = None,
        seed: int = 1234,
    ) -> list[int]:
        """Select the MLM pre-training set (paper §4.1).

        "We first construct the pre-training set by selecting at least one
        binary (2 x 4 variants) from each software project." So: pick
        ``per_package`` programs per package and take *all* their compiled
        variants, giving broad coverage of projects at modest size.

        ``restrict_to`` should normally be the training split — pre-training on
        binaries that later appear in the test set would leak, even though the
        MLM task has no labels.
        """
        allowed = None if restrict_to is None else set(restrict_to)
        by_pkg: dict[str, dict[str, list[int]]] = {}
        for rec in self.records:
            if allowed is not None and rec.bid not in allowed:
                continue
            by_pkg.setdefault(rec.package, {}).setdefault(rec.program, []).append(rec.bid)

        rng = random.Random(seed)
        out: list[int] = []
        for pkg in sorted(by_pkg):
            programs = sorted(by_pkg[pkg])
            rng.shuffle(programs)
            for prog in programs[:per_package]:
                out.extend(by_pkg[pkg][prog])
        return sorted(out)

    def save_splits(self, name: str, splits: dict[str, list[int]]) -> Path:
        d = self.dir / SPLIT_DIR
        d.mkdir(exist_ok=True)
        path = d / f"{name}.json"
        path.write_text(json.dumps({k: sorted(v) for k, v in splits.items()}, indent=1))
        return path

    def load_splits(self, name: str) -> dict[str, list[int]]:
        path = self.dir / SPLIT_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no split {name!r} in {self.dir}; create one with scripts/build_corpus.py"
            )
        return json.loads(path.read_text())
