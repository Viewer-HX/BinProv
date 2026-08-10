"""Tests for the parts that are easy to get subtly wrong.

Runnable either with pytest or directly::

    python tests/test_pipeline.py

Deliberately covers the pieces where a silent error would corrupt results rather
than crash: label parsing, sequence cutting, voting-group identity, split
leakage, and the MLM masking ratios.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov import vocab  # noqa: E402
from binprov.corpus import Corpus, CorpusWriter  # noqa: E402
from binprov.provenance import get_task, parse_labels  # noqa: E402
from binprov.vote import group_truth, majority_vote, vote_report  # noqa: E402


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


def test_vocab_roundtrip():
    assert vocab.VOCAB_SIZE == 261, "256 byte values + 5 special tags (paper §3.2)"
    data = bytes(range(256))
    assert vocab.decode(vocab.encode(data)) == data
    # special ids must not collide with byte ids
    assert max(vocab.NON_BYTE_IDS) < vocab.BYTE_OFFSET
    assert vocab.byte_to_id(0x00) == vocab.BYTE_OFFSET
    assert vocab.byte_to_id(0xFF) == vocab.VOCAB_SIZE - 1


# ---------------------------------------------------------------------------
# label parsing
# ---------------------------------------------------------------------------


def test_parse_labels_layout_variants():
    cases = [
        ("coreutils-8.24/x86_64-gcc-8.2.0-O2/ls", "x86_64", "gcc", "8.2.0", "O2"),
        ("pkg/x86_64_clang_7.0_O0/prog", "x86_64", "clang", "7.0", "O0"),
        ("a/arm_64-gcc-9.4.0-O3/b", "arm_64", "gcc", "9.4.0", "O3"),
        ("a/mipseb_64-clang-6.0.1-O1/b", "mipseb_64", "clang", "6.0.1", "O1"),
        ("a/x86_32-gcc-O0/b", "x86_32", "gcc", None, "O0"),
    ]
    for path, arch, comp, ver, opt in cases:
        lab = parse_labels(path)
        assert lab.arch == arch, (path, lab.arch)
        assert lab.compiler == comp, (path, lab.compiler)
        assert lab.compiler_version == ver, (path, lab.compiler_version)
        assert lab.opt == opt, (path, lab.opt)
        assert lab.is_complete()


def test_parse_labels_binkit_filename_layout():
    """BinKit puts the whole toolchain in the filename, not in directories.

    Checked against the real archive listing: these shapes cover all 67,680
    entries of BinKit Normal.
    """
    cases = [
        ("a2ps/a2ps-4.14_clang-7.0_x86_64_O2_a2ps.elf", "x86_64", "clang", "7.0", "O2"),
        ("gdbm/gdbm-1.15_gcc-8.2.0_arm_32_O0_gdbm_dump.elf", "arm_32", "gcc", "8.2.0", "O0"),
        ("binutils/binutils-2.30_gcc-4.9.4_mipseb_64_O3_ar.elf", "mipseb_64", "gcc", "4.9.4", "O3"),
        ("bool/bool-0.2_clang-4.0_mips_32_O1_bool.elf", "mips_32", "clang", "4.0", "O1"),
    ]
    for path, arch, comp, ver, opt in cases:
        lab = parse_labels(path)
        assert lab.arch == arch, (path, lab.arch)
        assert lab.compiler == comp, (path, lab.compiler)
        assert lab.compiler_version == ver, (path, lab.compiler_version)
        assert lab.opt == opt, (path, lab.opt)


def test_program_name_strips_provenance():
    """The grouping key must be the program, not the whole filename.

    If every compiled variant looks like a distinct program, the
    program-grouped split stops preventing leakage — a binary at O2 could train
    while its O3 twin is tested — and it fails silently, by inflating accuracy.
    """
    from binprov.discover import program_name

    # BinKit layout: strip the toolchain, keep underscores inside the program
    assert program_name(Path("a2ps-4.14_clang-7.0_x86_64_O2_a2ps.elf")) == "a2ps"
    assert program_name(Path("a2ps-4.14_clang-7.0_x86_64_O2_fixnt.elf")) == "fixnt"
    assert program_name(Path("gdbm-1.15_gcc-8.2.0_arm_32_O0_gdbm_dump.elf")) == "gdbm_dump"
    assert program_name(Path("gdbm-1.15_gcc-8.2.0_arm_32_O0_gdbm_load.elf")) == "gdbm_load"

    # all four variants of one program must collapse to the same key
    variants = [
        f"coreutils-8.24_gcc-8.2.0_x86_64_{o}_ls.elf" for o in ("O0", "O1", "O2", "O3")
    ] + [f"coreutils-8.24_clang-7.0_x86_64_{o}_ls.elf" for o in ("O0", "O1", "O2", "O3")]
    assert {program_name(Path(v)) for v in variants} == {"ls"}

    # directory layout: the filename already IS the program
    assert program_name(Path("ls")) == "ls"
    assert program_name(Path("gdbm_dump")) == "gdbm_dump"


def test_parse_labels_obfuscation():
    lab = parse_labels("pkg/x86_64-clang-obfus-sub-O2/prog")
    assert lab.obfuscation == "sub"
    assert lab.extra == "obfus"


def test_parse_labels_rejects_incomplete():
    assert not parse_labels("just/a/path/prog").is_complete()


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def test_task_scoping():
    o2o3 = get_task("opt_o2o3")
    assert o2o3.label_of({"opt": "O0", "compiler": "gcc"}) is None, "O0 is out of scope"
    assert o2o3.label_of({"opt": "O3", "compiler": "gcc"}) == 1

    hl = get_task("opt_hl")
    assert hl.label_of({"opt": "O0"}) == 0 and hl.label_of({"opt": "O1"}) == 0
    assert hl.label_of({"opt": "O2"}) == 1 and hl.label_of({"opt": "O3"}) == 1


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def _toy_corpus(tmp: Path) -> Corpus:
    """Two packages x two programs x two opt levels, with known .text bytes."""
    with CorpusWriter(tmp) as w:
        for pkg in ("pkgA", "pkgB"):
            for prog in ("p1", "p2"):
                for comp in ("gcc", "clang"):
                    for opt in ("O0", "O1", "O2", "O3"):
                        n = 1300  # 2 full 512-byte sequences + a 276-byte tail
                        text = bytes((i * 7 + len(opt)) % 256 for i in range(n))
                        w.add(
                            text,
                            path=f"{pkg}/x86_64-{comp}-{opt}/{prog}",
                            package=pkg,
                            program=prog,
                            labels={
                                "arch": "x86_64", "compiler": comp,
                                "compiler_version": "1.0", "opt": opt,
                                "extra": "normal", "obfuscation": None,
                            },
                            text_vaddr=0x1000,
                            arch_elf="x86_64",
                            stripped=False,
                            # three functions, the last shorter than a sequence
                            functions=[[0, 600], [600, 600], [1200, 100]],
                            func_names=["f0", "f1", "f2"],
                        )
    return Corpus(tmp)


def test_corpus_roundtrip_and_offsets():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        assert len(c) == 2 * 2 * 2 * 4 == 32
        # every record's slice must be exactly the bytes that were written
        for rec in c.records:
            got = np.asarray(c.text[rec.text_off : rec.text_off + rec.text_len])
            want = np.array(
                [(i * 7 + len(rec.labels["opt"])) % 256 for i in range(rec.text_len)],
                dtype=np.uint8,
            )
            assert np.array_equal(got, want), f"bytes for bid {rec.bid} do not round-trip"
        assert c.meta["num_text_bytes"] == sum(r.text_len for r in c.records)
    finally:
        shutil.rmtree(tmp)


def test_sequence_cutting_binary_level():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        idx = c.sequences(seq_len=512, level="binary", bids=[0])
        # 1300 bytes -> 512 + 512 + 276
        assert list(idx.length) == [512, 512, 276]
        assert list(idx.start) == [0, 512, 1024]
        assert set(idx.unit) == {-1}, "binary-level cuts carry no function id"
        # no sequence may run past the binary's own bytes
        rec = c.records[0]
        assert (idx.start + idx.length <= rec.text_off + rec.text_len).all()
    finally:
        shutil.rmtree(tmp)


def test_sequence_cutting_respects_function_bounds():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        idx = c.sequences(seq_len=512, stride=512, level="function", bids=[0])
        rec = c.records[0]
        for start, length, unit in zip(idx.start, idx.length, idx.unit):
            f_off, f_size = rec.functions[unit]
            lo = rec.text_off + f_off
            assert lo <= start < lo + f_size
            assert start + length <= lo + f_size, "a sequence crossed a function boundary"
        # the 100-byte function must still produce one (short) sequence
        assert 100 in list(idx.length)
    finally:
        shutil.rmtree(tmp)


def test_overlapping_stride_gives_more_votes():
    """The function-level voting of §3.4 needs >1 sequence per function.

    Note the sequence length has to be well below the function length for an
    overlapping stride to buy anything: a 600-byte function admits only 89
    distinct full 512-byte windows, so at seq_len=512 a smaller stride adds
    almost nothing. That is exactly why evaluate.py defaults the function-level
    stride to seq_len/4 and why it reports sequences-per-group -- a
    function-level "voting" number over groups of size 1 is not a voting result.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        wide = c.sequences(seq_len=128, stride=128, level="function", bids=[0])
        tight = c.sequences(seq_len=128, stride=32, level="function", bids=[0])
        assert len(tight) > len(wide), (len(tight), len(wide))
        assert np.bincount(tight.group_ids()).max() > 1, "need multiple votes per function"
        # and no window may escape its function even with a small stride
        rec = c.records[0]
        for start, length, unit in zip(tight.start, tight.length, tight.unit):
            f_off, f_size = rec.functions[unit]
            assert rec.text_off + f_off <= start
            assert start + length <= rec.text_off + f_off + f_size
    finally:
        shutil.rmtree(tmp)


def test_splits_have_no_program_leakage():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        sp = c.make_splits(train_ratio=0.5, group_by="program")
        assert set(sp["train"]) & set(sp["test"]) == set()
        assert len(sp["train"]) + len(sp["test"]) == len(c)
        train_groups = {c.records[b].group for b in sp["train"]}
        test_groups = {c.records[b].group for b in sp["test"]}
        assert train_groups & test_groups == set(), (
            "a program appeared on both sides; its other optimization levels leak"
        )
    finally:
        shutil.rmtree(tmp)


def test_pretrain_set_is_inside_train():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _toy_corpus(tmp)
        sp = c.make_splits(train_ratio=0.5, group_by="program")
        pre = c.pretrain_bids(per_package=1, restrict_to=sp["train"])
        assert pre, "pre-training set must not be empty"
        assert set(pre) <= set(sp["train"]), "MLM must not see test binaries"
        # "at least one binary (2 x 4 variants) from each software project"
        pkgs = {c.records[b].package for b in pre}
        train_pkgs = {c.records[b].package for b in sp["train"]}
        assert pkgs == train_pkgs
        for pkg in pkgs:
            variants = [b for b in pre if c.records[b].package == pkg]
            assert len(variants) == 8, f"{pkg}: expected 2 compilers x 4 opt, got {len(variants)}"
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# voting
# ---------------------------------------------------------------------------


def test_majority_vote_beats_noisy_sequences():
    # 3 groups of 5 sequences; 2 of 5 predictions wrong in each
    groups = np.repeat(np.arange(3), 5)
    truth = np.repeat([0, 1, 0], 5)
    pred = truth.copy()
    for g in range(3):
        pred[g * 5 : g * 5 + 2] = 1 - truth[g * 5]
    assert (pred == truth).mean() == 0.6
    rep = vote_report(groups, pred, truth, 2)
    assert rep["vote_accuracy"] == 1.0, "3/5 correct votes must carry each group"
    assert rep["num_groups"] == 3
    assert rep["mean_seqs_per_group"] == 5


def test_majority_vote_cannot_fix_a_lost_majority():
    groups = np.repeat(np.arange(2), 5)
    truth = np.zeros(10, dtype=np.int64)
    pred = truth.copy()
    pred[:3] = 1  # group 0 loses its majority, group 1 is clean
    rep = vote_report(groups, pred, truth, 2)
    assert rep["vote_accuracy"] == 0.5


def test_soft_vote_uses_confidence():
    # hard voting loses 1-2; soft voting wins because the minority is confident
    groups = np.zeros(3, dtype=np.int64)
    truth = np.zeros(3, dtype=np.int64)
    pred = np.array([1, 1, 0])
    probs = np.array([[0.49, 0.51], [0.49, 0.51], [0.99, 0.01]])
    assert majority_vote(groups, pred, 2)[1][0] == 1
    assert majority_vote(groups, pred, 2, probs=probs, mode="soft")[1][0] == 0


def test_group_truth_rejects_inconsistent_labels():
    groups = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 1, 1])  # group 0 disagrees with itself
    try:
        group_truth(groups, labels)
    except ValueError:
        return
    raise AssertionError("group_truth must reject a group with mixed labels")


def test_empty_vote_is_not_a_crash():
    empty = np.zeros(0, dtype=np.int64)
    groups, voted, counts = majority_vote(empty, empty, 2)
    assert len(groups) == len(voted) == len(counts) == 0


# ---------------------------------------------------------------------------
# MLM masking (needs torch)
# ---------------------------------------------------------------------------


def test_mlm_masking_ratios():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skipping MLM masking test (torch not installed)")
        return
    from binprov.data import IGNORE_INDEX, MLMCollator

    seq_bytes, n = 512, 400
    rng = np.random.default_rng(0)
    batch = [(rng.integers(0, 256, seq_bytes, dtype=np.uint8), -1, i) for i in range(n)]
    out = MLMCollator(seq_bytes + 2, mask_prob=0.20, seed=0)(batch)

    ids = out["input_ids"].numpy()
    labels = out["labels"].numpy()
    selected = labels != IGNORE_INDEX

    # 20% of bytes selected (paper §4.1)
    rate = selected.sum() / (n * seq_bytes)
    assert 0.19 < rate < 0.21, rate
    # never mask <s>, </s> or padding
    assert not selected[:, 0].any() and not selected[:, -1].any()
    # of the selected, ~50% <mask> and ~50% a random byte, and none left as-is
    masked = (ids[selected] == vocab.MASK_ID).mean()
    assert 0.47 < masked < 0.53, masked
    assert (ids[selected] >= vocab.BYTE_OFFSET).sum() + (ids[selected] == vocab.MASK_ID).sum() \
        == selected.sum(), "a selected token became something other than <mask> or a byte"
    # unselected positions must be untouched
    assert (ids[~selected & (labels == IGNORE_INDEX)] != vocab.MASK_ID).all()


def test_zero_bytes_are_not_padding():
    """0x00 is everywhere in real .text (alignment, padding between functions).

    It must not collide with <pad>, or the model would treat genuine zero bytes
    as absent. This is why byte values are offset by the 5 special ids.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skipping zero-byte test (torch not installed)")
        return
    from binprov.data import ClassificationCollator

    seq_bytes = 64
    zeros = np.zeros(seq_bytes, dtype=np.uint8)
    short = np.zeros(8, dtype=np.uint8)
    out = ClassificationCollator(seq_bytes + 2)([(zeros, 0, 0), (short, 1, 1)])
    ids = out["input_ids"].numpy()
    attn = out["attention_mask"].numpy()

    # row 0: <s> + 64 zero bytes + </s>, all attended, no PAD id among the bytes
    assert attn[0].sum() == seq_bytes + 2
    assert (ids[0, 1 : 1 + seq_bytes] == vocab.byte_to_id(0)).all()
    assert (ids[0, 1 : 1 + seq_bytes] != vocab.PAD_ID).all()
    # row 1: 8 real zero bytes attended, the rest padded and not attended
    assert attn[1].sum() == 8 + 2
    assert (ids[1, 1:9] == vocab.byte_to_id(0)).all()
    assert (ids[1, 10:] == vocab.PAD_ID).all()
    assert attn[1, 10:].sum() == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 - a test report wants every failure
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
