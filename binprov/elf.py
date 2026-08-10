"""A small, dependency-free ELF reader.

BinProv's whole input is the ``.text`` section (paper §3.1), plus — only for the
function-level majority voting of §3.4 — the function boundaries recorded in the
symbol table. That is all this module extracts. It deliberately avoids
``pyelftools``/``lief`` so that corpus building has no native dependencies and
so that the same code path handles every architecture in the paper
(x86_32, x86_64, ARM_64, MIPS_64), including big-endian MIPS.

Everything is parsed from an in-memory ``bytes``/``memoryview`` buffer.
Malformed or truncated files raise :class:`ElfError` rather than crashing, so a
corpus build can skip them and keep going.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

ELF_MAGIC = b"\x7fELF"

# section header types we care about
SHT_SYMTAB = 2
SHT_DYNSYM = 11
SHT_NOBITS = 8

STT_FUNC = 2

# e_machine -> canonical architecture name. The width/endianness suffix is
# resolved from EI_CLASS / EI_DATA, because e.g. EM_MIPS covers 4 variants.
_MACHINE = {
    3: "x86",
    62: "x86",  # EM_X86_64 (width comes from EI_CLASS)
    40: "arm",
    183: "arm",  # EM_AARCH64
    8: "mips",
    10: "mips",  # EM_MIPS_RS3_LE
    20: "ppc",
    21: "ppc",
    22: "s390",
    243: "riscv",
    258: "loongarch",
}


class ElfError(Exception):
    """Raised for anything that is not a parseable ELF file."""


@dataclass(frozen=True)
class Function:
    """A function boundary taken from the symbol table."""

    name: str
    offset: int  # byte offset within the .text section
    size: int  # size in bytes


@dataclass
class TextSection:
    """The extracted ``.text`` section and its metadata."""

    data: bytes
    vaddr: int  # virtual address of .text[0]
    file_offset: int
    arch: str  # e.g. "x86_64", "mips_64", "mipseb_64"
    bits: int  # 32 or 64
    little_endian: bool
    functions: list[Function] = field(default_factory=list)
    comment: str = ""  # contents of .comment, if present
    stripped: bool = True  # no SHT_SYMTAB present

    def __len__(self) -> int:
        return len(self.data)


def _u(buf: memoryview, fmt: str, off: int):
    """Unpack ``fmt`` at ``off``, raising ElfError on truncation."""
    size = struct.calcsize(fmt)
    if off < 0 or off + size > len(buf):
        raise ElfError(f"truncated file: want {size} bytes at {off}, have {len(buf)}")
    return struct.unpack_from(fmt, buf, off)


def _cstr(buf: memoryview, off: int) -> str:
    """Read a NUL-terminated string; tolerates a missing terminator."""
    if not 0 <= off < len(buf):
        return ""
    end = bytes(buf[off:]).find(b"\x00")
    raw = bytes(buf[off:] if end < 0 else buf[off : off + end])
    return raw.decode("utf-8", "replace")


def _arch_name(machine: int, bits: int, little_endian: bool) -> str:
    base = _MACHINE.get(machine, f"machine{machine}")
    # BinKit names its big-endian MIPS targets "mipseb"; mirror that so labels
    # from paths and from ELF headers agree.
    if base == "mips" and not little_endian:
        base = "mipseb"
    return f"{base}_{bits}"


def parse(buf: bytes, *, want_functions: bool = True) -> TextSection:
    """Extract ``.text`` (and optionally function boundaries) from an ELF image.

    Args:
        buf: the whole file contents.
        want_functions: parse the symbol table for function boundaries. Skip it
            when only doing binary-level work — it saves a scan over the
            symbol table, which in a debug build can be larger than ``.text``.

    Raises:
        ElfError: not an ELF, unsupported class, or no ``.text`` section.
    """
    mv = memoryview(buf)
    if len(mv) < 64 or bytes(mv[:4]) != ELF_MAGIC:
        raise ElfError("not an ELF file")

    ei_class, ei_data = mv[4], mv[5]
    if ei_class == 1:
        bits = 32
    elif ei_class == 2:
        bits = 64
    else:
        raise ElfError(f"bad EI_CLASS {ei_class}")
    if ei_data == 1:
        little_endian = True
    elif ei_data == 2:
        little_endian = False
    else:
        raise ElfError(f"bad EI_DATA {ei_data}")

    e = "<" if little_endian else ">"
    (machine,) = _u(mv, e + "H", 18)

    # --- ELF header: locate the section header table -----------------------
    if bits == 64:
        e_shoff, = _u(mv, e + "Q", 40)
        e_shentsize, e_shnum, e_shstrndx = _u(mv, e + "HHH", 58)
    else:
        e_shoff, = _u(mv, e + "I", 32)
        e_shentsize, e_shnum, e_shstrndx = _u(mv, e + "HHH", 46)

    if e_shoff == 0 or e_shnum == 0:
        raise ElfError("no section header table (stripped of sections?)")
    if e_shentsize < (64 if bits == 64 else 40):
        raise ElfError(f"implausible e_shentsize {e_shentsize}")

    # --- section headers ---------------------------------------------------
    # (name_off, type, addr, offset, size, link, entsize)
    sections: list[tuple[int, int, int, int, int, int, int]] = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        if bits == 64:
            sh_name, sh_type = _u(mv, e + "II", base)
            sh_addr, sh_offset, sh_size = _u(mv, e + "QQQ", base + 16)
            sh_link, = _u(mv, e + "I", base + 40)
            sh_entsize, = _u(mv, e + "Q", base + 56)
        else:
            sh_name, sh_type = _u(mv, e + "II", base)
            sh_addr, sh_offset, sh_size = _u(mv, e + "III", base + 12)
            sh_link, = _u(mv, e + "I", base + 24)
            sh_entsize, = _u(mv, e + "I", base + 36)
        sections.append((sh_name, sh_type, sh_addr, sh_offset, sh_size, sh_link, sh_entsize))

    # section-name string table
    shstr_off = 0
    if 0 <= e_shstrndx < len(sections):
        shstr_off = sections[e_shstrndx][3]

    def sec_name(idx: int) -> str:
        return _cstr(mv, shstr_off + sections[idx][0]) if shstr_off else ""

    names = [sec_name(i) for i in range(len(sections))]

    # --- .text -------------------------------------------------------------
    try:
        text_idx = names.index(".text")
    except ValueError:
        raise ElfError("no .text section") from None

    _, t_type, t_addr, t_off, t_size, _, _ = sections[text_idx]
    if t_type == SHT_NOBITS or t_size == 0:
        raise ElfError(".text is empty")
    if t_off + t_size > len(mv):
        # Truncated download / corrupt file: take what is actually there.
        t_size = max(0, len(mv) - t_off)
        if t_size == 0:
            raise ElfError(".text lies past end of file")
    text = bytes(mv[t_off : t_off + t_size])

    # --- .comment (compiler fingerprint, used to sanity-check labels) ------
    comment = ""
    if ".comment" in names:
        _, _, _, c_off, c_size, _, _ = sections[names.index(".comment")]
        if c_size and c_off + c_size <= len(mv):
            comment = bytes(mv[c_off : c_off + c_size]).replace(b"\x00", b" ").decode(
                "utf-8", "replace"
            ).strip()

    stripped = not any(s[1] == SHT_SYMTAB for s in sections)

    result = TextSection(
        data=text,
        vaddr=t_addr,
        file_offset=t_off,
        arch=_arch_name(machine, bits, little_endian),
        bits=bits,
        little_endian=little_endian,
        comment=comment,
        stripped=stripped,
    )
    if want_functions:
        result.functions = _read_functions(mv, e, bits, sections, text_idx, t_addr, t_size, machine)
    return result


def _read_functions(
    mv: memoryview,
    e: str,
    bits: int,
    sections,
    text_idx: int,
    text_vaddr: int,
    text_size: int,
    machine: int,
) -> list[Function]:
    """Collect STT_FUNC symbols that live inside ``.text``.

    Prefers SHT_SYMTAB; falls back to SHT_DYNSYM for stripped binaries (which
    yields far fewer functions, but is better than nothing).
    """
    symtabs = [s for s in sections if s[1] == SHT_SYMTAB]
    if not symtabs:
        symtabs = [s for s in sections if s[1] == SHT_DYNSYM]
    if not symtabs:
        return []

    sym_size = 24 if bits == 64 else 16
    out: dict[int, Function] = {}  # offset -> Function, de-duplicates aliases

    for _, _, _, sh_offset, sh_size, sh_link, sh_entsize in symtabs:
        entsize = sh_entsize or sym_size
        if entsize < sym_size or sh_size == 0:
            continue
        # linked string table
        strtab_off = sections[sh_link][3] if 0 <= sh_link < len(sections) else 0
        n = sh_size // entsize
        for i in range(n):
            base = sh_offset + i * entsize
            try:
                if bits == 64:
                    st_name, = _u(mv, e + "I", base)
                    st_info = mv[base + 4]
                    st_shndx, = _u(mv, e + "H", base + 6)
                    st_value, st_sz = _u(mv, e + "QQ", base + 8)
                else:
                    st_name, st_value, st_sz = _u(mv, e + "III", base)
                    st_info = mv[base + 12]
                    st_shndx, = _u(mv, e + "H", base + 14)
            except ElfError:
                break  # truncated symbol table; keep what we have

            if (st_info & 0xF) != STT_FUNC or st_sz == 0 or st_shndx != text_idx:
                continue
            # ARM/Thumb interworking sets bit 0 of st_value; MIPS16/microMIPS
            # does the same. Clear it so the offset lands on a real byte.
            if machine in (40, 8, 10):
                st_value &= ~1
            off = st_value - text_vaddr
            if off < 0 or off >= text_size:
                continue
            size = min(st_sz, text_size - off)
            if size <= 0 or off in out:
                continue
            out[off] = Function(name=_cstr(mv, strtab_off + st_name), offset=off, size=size)

    return sorted(out.values(), key=lambda f: f.offset)


def guess_compiler(comment: str) -> tuple[str | None, str | None]:
    """Best-effort ``(family, version)`` from a ``.comment`` string.

    Used to cross-check labels derived from file paths — a cheap guard against a
    mislabeled corpus. Returns ``(None, None)`` when nothing is recognisable.
    Note that a binary linked from both GCC- and Clang-compiled objects can
    carry both fingerprints; the first match wins.
    """
    import re

    if not comment:
        return None, None
    low = comment.lower()
    m = re.search(r"clang version ([0-9]+(?:\.[0-9]+)*)", low)
    if m:
        return "clang", m.group(1)
    m = re.search(r"gcc[:\s(]+.*?([0-9]+\.[0-9]+\.[0-9]+)", low)
    if m:
        return "gcc", m.group(1)
    if "clang" in low:
        return "clang", None
    if "gcc" in low or "gnu" in low:
        return "gcc", None
    return None, None


def is_elf(path) -> bool:
    """Cheap check used to filter build trees before a full parse."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == ELF_MAGIC
    except OSError:
        return False
