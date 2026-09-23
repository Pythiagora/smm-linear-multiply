"""Schönhage (1980), §6: integer multiplication in linear time on a Storage Modification Machine.

This module follows A. Schönhage, "Storage Modification Machines", SIAM J. Comput. 9(3):490–508,
1980, §6 ("Integer-multiplication in linear time"), as literally as a working program allows:

* the machine model and its instruction language are those of §2 — alphabet Δ = {P, Q, S, W}, the
  instructions ``new W; set W to V; if U = V then σ; if U ≠ V then σ; input λ0, λ1; output β;
  goto λ; halt;`` — with unit cost per executed instruction (a conditional counts once whatever
  the outcome).  Programs can be printed in, and parsed from, that syntax (:func:`parse`,
  :meth:`Program.listing`);
* arithmetic is B-ary (B = 2^b, b = n/3) on the "B-scale" of §6.1, a cyclic list D_0 … D_{B-1}
  with successor (S), doubling (P) and halving (Q) pointers (6.1)–(6.2).  A digit is a pointer to
  D_p.  The basic digital operations are (6.3) p+q = r+sB, (6.4) p−q = r·(−1)^s and (6.5) pq = r+sB,
  executed through an intermediate translation to binary in O(b), O(b) and O(b²) steps;
* the numerical work is done in the *interpreting mode* of §6.2: every complex multiplication or
  butterfly is a small straight-line program encoded in the Δ-structure; one sweep advances every
  program by one basic operation, the operations are collected in three lines (additions,
  subtractions, multiplications), each line is sorted with respect to equal operands by the
  radix-sort program printed in §6.2 (transcribed verbatim in :meth:`Builder.sort_line`), each
  distinct operation is computed once, and the results are distributed (Lemma 6.2: O(m + b²B²)
  per sweep for m programs);
* the multiplication itself is §6.3: n minimal, multiple of 3, with n·2^n ≥ 2N (and n ≥ 6); the
  factors split into 2^{n-1} pieces of n bits (6.8); fixed-point complex numbers of 19 B-ary digits
  (18 fractional, one leading) with modulus ≤ 1; roots of unity w_2 = i, w_{ν+1} = (1+w_ν)/|1+w_ν|
  in "any crude manner" in binary, their powers by (6.11) in n−2 stages of simultaneous complex
  multiplications; two FFTs of n stages, pointwise products, the back transformation; and the
  final translation to binary and summation (6.12).

Deviations, all of them choices the paper leaves open ("for instance"):
the forward transform is decimation in frequency and the inverse decimation in time, so no
bit-reversal is needed; signed fixed-point numbers use B's complement; a real product keeps the
partial products of columns ≥ 17 of 37 (one guard column), the host model shows a worst-case error
below 0.02·2^{-4n} against the 0.5·2^{-4n} the rounding step tolerates; complex products use the
three-multiplication formula with the twiddle sums w_re ± w_im precomputed once per root; at most
2^9 programs are interpreted per session (the programs of a stage are independent, this only bounds
the live structure); one instruction ``set P to []`` precedes the ``new P`` of the sorting program,
since the paper's text leaves the previous contents of A P unspecified and ``new`` copies them into
the auxiliary node (without it every sorted line of every sweep would stay reachable).

The runtime keeps only the part of the Δ-structure accessible from the centre (the reduction the
semantics of ``set`` prescribe), by an occasional mark-and-compact pass; ``RunResult.nodes`` counts
every ``new`` executed.  Everything the machine computes is computed by pointer manipulation; the
host only encodes the input tape, decodes the output tape, and — in the tests — recomputes the
expected values.

Measurements (``--bench``, poito, 23/09/2026; ``results/bench_faithful.jsonl``).  The work of a run
is fixed by n, so the honest abscissa is the largest N of each class, N = n·2^{n-1}:

    n   B      N          steps    steps/N   steps/(n·2^n)   peak live nodes
    6   4    192    251,790,964  1,311,411         655,706        8.0 M
    9   8   2304  2,787,786,704  1,209,977         604,988       13.2 M
   12  16  24576   (see results/bench_faithful.jsonl)

Steps per butterfly slot (n·2^n of them over the two transforms and the back transformation) are
flat, and steps/N falls with n because the O(b²B²) part of a sweep (Lemma 6.2) shrinks against
m = 2^{n-1}: the linear bound with a large constant, as the paper has it.  Of the steps, ≈ 59 % go
to the two forward transforms, 31 % to the back transformation, 7 % to the pointwise products and
4 % to the powers of the roots; everything else (input, parameters, B-scale, roots in binary,
rounding, output) is below 0.5 %.  ``--test`` runs 31 checks (paper's §2 counterexample, syntax
round trip, all digit pairs for B = 4, 8, 16, every numerical program against the host model,
roots and powers digit for digit, reference interpreter = compiled machine to the step, products
against Python integers up to N = 2304).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from typing import Final

__all__ = [
    "ALPHABET",
    "Asm",
    "Builder",
    "FastSMM",
    "FixedModel",
    "Instr",
    "Op",
    "Program",
    "RunResult",
    "build_program",
    "decode_output",
    "encode_tape",
    "parse",
    "run_reference",
    "smm_multiply",
]

ALPHABET: Final = "PQSW"
EMPTY_WORD: Final = "[]"  # how the empty path (the centre) is written in listings

# ======================================================================================
# Part 1 — the Storage Modification Machine of §2, alphabet Δ = {P, Q, S, W}
# ======================================================================================


class Op(IntEnum):
    NEW = 0
    SET = 1
    GOTO = 2
    HALT = 3
    INPUT = 4
    OUTPUT = 5


@dataclass(frozen=True, slots=True)
class Instr:
    """One instruction of §2, possibly wrapped in ``if U = V then`` / ``if U ≠ V then``.

    ``a`` and ``b`` are paths (words over Δ, "" = the centre) or labels/bits depending on ``op``;
    ``cond`` is None, or (U, V, equal) for the conditional forms (2.7)/(2.8).
    """

    op: Op
    a: str = ""
    b: str = ""
    cond: tuple[str, str, bool] | None = None
    label: str = ""  # label carried by this instruction, "" if none

    def text(self) -> str:
        body = {
            Op.NEW: lambda: f"new {_w(self.a)};",
            Op.SET: lambda: f"set {_w(self.a)} to {_w(self.b)};",
            Op.GOTO: lambda: f"goto {self.a};",
            Op.HALT: lambda: "halt;",
            Op.INPUT: lambda: f"input {self.a}, {self.b};",
            Op.OUTPUT: lambda: f"output {self.a};",
        }[self.op]()
        if self.cond is not None:
            u, v, eq = self.cond
            body = f"if {_w(u)} {'=' if eq else '≠'} {_w(v)} then {body}"
        return f"{self.label}: {body}" if self.label else body


def _w(path: str) -> str:
    return path if path else EMPTY_WORD


@dataclass
class Program:
    code: list[Instr]
    labels: dict[str, int]
    note: dict[str, str] = field(default_factory=dict)  # free-form documentation (register map …)

    def listing(self) -> str:
        return "\n".join(ins.text() for ins in self.code)

    def __len__(self) -> int:
        return len(self.code)


_TOKEN = re.compile(
    r"\s*(?:(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*:(?!=)|(?P<instr>[^;]+;))",
)
_PATH = re.compile(r"^(?:\[\]|[PQSW]*)$")


def parse(text: str) -> Program:
    """Parse a program written in the syntax of §2 (see :meth:`Instr.text`)."""
    code: list[Instr] = []
    labels: dict[str, int] = {}
    pending = ""
    pos = 0
    text = re.sub(r"#[^\n]*", "", text)
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise SyntaxError(f"cannot parse near: {text[pos : pos + 40]!r}")
        pos = m.end()
        if m.group("label"):
            pending = m.group("label")
            continue
        ins = _parse_instr(m.group("instr").strip()[:-1].strip(), pending)
        if pending:
            if pending in labels:
                raise SyntaxError(f"duplicate label {pending}")
            labels[pending] = len(code)
            pending = ""
        code.append(ins)
    if pending:
        raise SyntaxError(f"label {pending} at end of program")
    for ins in code:
        for lab in _targets(ins):
            if lab not in labels:
                raise SyntaxError(f"undefined label {lab}")
    return Program(code, labels)


def _targets(ins: Instr) -> list[str]:
    if ins.op is Op.GOTO:
        return [ins.a]
    if ins.op is Op.INPUT:
        return [ins.a, ins.b]
    return []


def _path(tok: str) -> str:
    tok = tok.strip()
    if not _PATH.match(tok):
        raise SyntaxError(f"bad path {tok!r}")
    return "" if tok == EMPTY_WORD else tok


def _parse_instr(s: str, label: str) -> Instr:
    cond: tuple[str, str, bool] | None = None
    if s.startswith("if "):
        m = re.match(r"if\s+(\S+)\s*(=|≠|!=|/=)\s*(\S+)\s+then\s+(.*)$", s)
        if not m:
            raise SyntaxError(f"bad conditional {s!r}")
        cond = (_path(m.group(1)), _path(m.group(3)), m.group(2) == "=")
        s = m.group(4).strip()
    words = s.split()
    kw = words[0]
    if kw == "new" and len(words) == 2:
        return Instr(Op.NEW, _path(words[1]), cond=cond, label=label)
    if kw == "set" and len(words) == 4 and words[2] == "to":
        return Instr(Op.SET, _path(words[1]), _path(words[3]), cond=cond, label=label)
    if kw == "goto" and len(words) == 2:
        return Instr(Op.GOTO, words[1], cond=cond, label=label)
    if kw == "halt" and len(words) == 1:
        return Instr(Op.HALT, cond=cond, label=label)
    if kw == "output" and len(words) == 2 and words[1] in ("0", "1"):
        return Instr(Op.OUTPUT, words[1], cond=cond, label=label)
    if kw == "input":
        m = re.match(r"input\s+(\S+)\s*,\s*(\S+)$", s)
        if m:
            return Instr(Op.INPUT, m.group(1), m.group(2), cond=cond, label=label)
    raise SyntaxError(f"bad instruction {s!r}")


class StepLimitError(RuntimeError):
    pass


@dataclass
class RunResult:
    output: list[int]
    steps: int
    nodes: int
    halted: bool
    ptr: tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]  # final structure, per letter
    centre: int
    phase_steps: dict[str, int] = field(default_factory=dict)
    peak_nodes: int = 0  # largest node array during the run (FastSMM)
    collections: int = 0  # garbage collections performed (FastSMM)

    def follow(self, node: int, path: str) -> int:
        for ch in path:
            node = self.ptr[ALPHABET.index(ch)][node]
        return node

    def node(self, path: str) -> int:
        return self.follow(self.centre, path)


def run_reference(prog: Program, inp: Sequence[int], max_steps: int | None = None) -> RunResult:
    """Direct interpreter of the semantics (2.1)–(2.8).  Slow, kept as the oracle for FastSMM."""
    code = prog.code
    labels = prog.labels
    ptr: tuple[list[int], list[int], list[int], list[int]] = ([0], [0], [0], [0])
    idx = {ch: i for i, ch in enumerate(ALPHABET)}
    centre = 0
    out: list[int] = []
    pc = 0
    steps = 0
    ipos = 0
    n_in = len(inp)
    halted = False
    phase_steps: dict[str, int] = {}
    cur_phase = ""

    def star(path: str) -> int:
        node = centre
        for ch in path:
            node = ptr[idx[ch]][node]
        return node

    last_mark = 0
    while pc < len(code):
        ins = code[pc]
        steps += 1
        if max_steps is not None and steps > max_steps:
            raise StepLimitError(f"more than {max_steps} steps")
        if ins.label.startswith("phase_"):
            phase_steps[cur_phase] = phase_steps.get(cur_phase, 0) + (steps - 1 - last_mark)
            last_mark = steps - 1
            cur_phase = ins.label[6:]
        pc += 1
        if ins.cond is not None:
            u, v, eq = ins.cond
            if (star(u) == star(v)) != eq:
                continue
        op = ins.op
        if op is Op.NEW:
            w = ins.a
            old = star(w)
            y = len(ptr[0])
            for lst in ptr:
                lst.append(old)
            if w == "":
                centre = y
            else:
                ptr[idx[w[-1]]][star(w[:-1])] = y
        elif op is Op.SET:
            target = star(ins.b)
            if ins.a == "":
                centre = target
            else:
                ptr[idx[ins.a[-1]]][star(ins.a[:-1])] = target
        elif op is Op.GOTO:
            pc = labels[ins.a]
        elif op is Op.HALT:
            halted = True
            break
        elif op is Op.INPUT:
            if ipos < n_in:
                bit = inp[ipos]
                ipos += 1
                pc = labels[ins.b if bit else ins.a]
        elif op is Op.OUTPUT:
            out.append(int(ins.a))
    else:
        halted = True
    phase_steps[cur_phase] = phase_steps.get(cur_phase, 0) + (steps - last_mark)
    phase_steps.pop("", None)
    return RunResult(out, steps, len(ptr[0]), halted, ptr, centre, phase_steps)


class FastSMM:
    """The same machine, with each basic block compiled once into a Python function.

    Step counts, node counts and the final structure are identical to :func:`run_reference`.
    Phase accounting: an instruction carrying a label ``phase_<name>`` starts phase ``<name>``.
    """

    def __init__(self, prog: Program) -> None:
        self.prog = prog
        self.blocks: list[Callable[..., int]] = []
        self.block_of_pc: dict[int, int] = {}
        self.phase_of_block: dict[int, str] = {}
        self._compile()

    # -- compilation -------------------------------------------------------------------
    def _compile(self) -> None:
        code = self.prog.code
        labels = self.prog.labels
        starts = {0, len(code)}
        for pc, ins in enumerate(code):
            for lab in _targets(ins):
                starts.add(labels[lab])
            if ins.op in (Op.GOTO, Op.INPUT, Op.HALT) or (ins.cond is not None and ins.op is Op.GOTO):
                starts.add(pc + 1)
            if ins.label:
                starts.add(pc)
        order = sorted(starts)
        self.block_of_pc = {pc: i for i, pc in enumerate(order)}
        n_blocks = len(order)
        for i, start in enumerate(order):
            end = order[i + 1] if i + 1 < n_blocks else len(code)
            if start < len(code) and code[start].label.startswith("phase_"):
                self.phase_of_block[i] = code[start].label[6:]
            self.blocks.append(self._compile_block(code[start:end], start, end))

    def _compile_block(self, block: list[Instr], start: int, end: int) -> Callable[..., int]:
        """Return f(st) -> next block index (or -1 to halt).  st is the machine state object."""
        lines: list[str] = ["def blk(st):", "    P, Q, S, W = st.P, st.Q, st.S, st.W", "    c = st.centre", "    n = 0"]
        idx = {ch: i for i, ch in enumerate(ALPHABET)}
        names = "PQSW"

        def expr(path: str) -> str:
            e = "c"
            for ch in path:
                e = f"{names[idx[ch]]}[{e}]"
            return e

        def emit(ind: str, ins: Instr) -> None:
            op = ins.op
            if op is Op.NEW:
                w = ins.a
                lines.append(f"{ind}old = {expr(w)}")
                lines.append(f"{ind}y = len(P); P.append(old); Q.append(old); S.append(old); W.append(old)")
                if w == "":
                    lines.append(f"{ind}c = y")
                else:
                    lines.append(f"{ind}{names[idx[w[-1]]]}[{expr(w[:-1])}] = y")
            elif op is Op.SET:
                if ins.a == "":
                    lines.append(f"{ind}c = {expr(ins.b)}")
                else:
                    lines.append(f"{ind}{names[idx[ins.a[-1]]]}[{expr(ins.a[:-1])}] = {expr(ins.b)}")
            elif op is Op.OUTPUT:
                lines.append(f"{ind}st.out.append({ins.a})")
            elif op is Op.HALT:
                lines.append(f"{ind}st.centre = c; st.steps += n; return -1")
            elif op is Op.GOTO:
                lines.append(f"{ind}st.centre = c; st.steps += n; return {self.block_of_pc[self.prog.labels[ins.a]]}")
            elif op is Op.INPUT:
                lines.append(f"{ind}if st.ipos < st.n_in:")
                lines.append(f"{ind}    bit = st.inp[st.ipos]; st.ipos += 1; st.centre = c; st.steps += n")
                lines.append(
                    f"{ind}    return {self.block_of_pc[self.prog.labels[ins.b]]} if bit else "
                    f"{self.block_of_pc[self.prog.labels[ins.a]]}"
                )
                # input exhausted: control falls through to the next instruction

        for ins in block:
            lines.append("    n += 1")
            if ins.cond is None:
                emit("    ", ins)
            else:
                u, v, eq = ins.cond
                lines.append(f"    if {expr(u)} {'==' if eq else '!='} {expr(v)}:")
                emit("        ", ins)
        lines.append("    st.centre = c; st.steps += n")
        lines.append(f"    return {self.block_of_pc[end] if end < len(self.prog.code) else -1}")
        src = "\n".join(lines)
        ns: dict[str, object] = {}
        exec(src, ns)  # noqa: S102 — trusted, generated from the program itself
        return ns["blk"]  # type: ignore[return-value]

    # -- execution ---------------------------------------------------------------------
    def run(self, inp: Sequence[int], max_steps: int | None = None, gc_limit: int = 8_000_000) -> RunResult:
        """Run on the input bits.  ``gc_limit``: collect garbage when the node arrays exceed this size."""
        st = _State(inp, gc_limit)
        blocks = self.blocks
        phase_of_block = self.phase_of_block
        phase_steps: dict[str, int] = {}
        cur = ""
        last = 0
        b = 0
        limit = max_steps if max_steps is not None else float("inf")
        base = 0  # nodes existing at the last collection (allocation accounting)
        P = st.P
        while b >= 0:
            if b in phase_of_block:
                phase_steps[cur] = phase_steps.get(cur, 0) + st.steps - last
                last = st.steps
                cur = phase_of_block[b]
            b = blocks[b](st)
            if len(st.P) > st.gc_limit:
                st.allocs += len(st.P) - base
                st.peak = max(st.peak, len(st.P))
                st.collect()
                base = len(st.P)
                P = st.P
            if st.steps > limit:
                raise StepLimitError(f"more than {max_steps} steps")
        st.allocs += len(P) - base
        st.peak = max(st.peak, len(st.P))
        phase_steps[cur] = phase_steps.get(cur, 0) + st.steps - last
        phase_steps.pop("", None)
        res = RunResult(st.out, st.steps, st.allocs, True, (st.P, st.Q, st.S, st.W), st.centre, phase_steps)
        res.peak_nodes = st.peak
        res.collections = st.gcs
        return res


class _State:
    """Machine state.  Pointers live in four lists (one per letter), indexed by node number.

    The Δ-structure of §2 is always reduced to the part accessible from the centre, so nodes that
    became unreachable may be dropped: :meth:`collect` marks from the centre and compacts, keeping
    memory proportional to the live structure.  ``allocs`` counts every ``new`` ever executed.
    """

    __slots__ = ("P", "Q", "S", "W", "allocs", "centre", "gc_limit", "gcs", "inp", "ipos", "n_in", "out", "peak", "steps")

    def __init__(self, inp: Sequence[int], gc_limit: int) -> None:
        self.P: list[int] = [0]
        self.Q: list[int] = [0]
        self.S: list[int] = [0]
        self.W: list[int] = [0]
        self.centre = 0
        self.inp = list(inp)
        self.ipos = 0
        self.n_in = len(self.inp)
        self.out: list[int] = []
        self.steps = 0
        self.allocs = 0
        self.peak = 1
        self.gcs = 0
        self.gc_limit = gc_limit

    def collect(self) -> None:
        """Mark (breadth first from the centre) and compact; node numbers change, the structure does not."""
        from array import array

        P, Q, S, W = self.P, self.Q, self.S, self.W
        n = len(P)
        new_id = array("i", bytes(4 * n))  # 0 = not yet seen; ids are stored +1
        order = [self.centre]
        new_id[self.centre] = 1
        k = 0
        while k < len(order):
            node = order[k]
            k += 1
            for arr in (P, Q, S, W):
                t = arr[node]
                if not new_id[t]:
                    new_id[t] = len(order) + 1
                    order.append(t)
        live = len(order)
        self.P = [new_id[P[node]] - 1 for node in order]
        self.Q = [new_id[Q[node]] - 1 for node in order]
        self.S = [new_id[S[node]] - 1 for node in order]
        self.W = [new_id[W[node]] - 1 for node in order]
        self.centre = 0
        self.gcs += 1
        self.gc_limit = max(self.gc_limit, 4 * live)


# ======================================================================================
# Part 2 — assembler: labels, structured control, the register tree
# ======================================================================================


class _Cond:
    """``if U = V then …`` / ``if U ≠ V then …`` (2.7)/(2.8): one instruction, one step."""

    def __init__(self, asm: Asm, u: str, v: str, eq: bool) -> None:
        self.asm, self.cond = asm, (u, v, eq)

    def new(self, w: str) -> None:
        self.asm._emit(Instr(Op.NEW, w, cond=self.cond))

    def set(self, w: str, v: str) -> None:
        self.asm._emit(Instr(Op.SET, w, v, cond=self.cond))

    def goto(self, label: str) -> None:
        self.asm._emit(Instr(Op.GOTO, label, cond=self.cond))

    def halt(self) -> None:
        self.asm._emit(Instr(Op.HALT, cond=self.cond))

    def output(self, bit: int) -> None:
        self.asm._emit(Instr(Op.OUTPUT, str(bit), cond=self.cond))


class Asm:
    """Emits §2 instructions; labels are attached to the next instruction emitted."""

    def __init__(self) -> None:
        self.code: list[Instr] = []
        self.labels: dict[str, int] = {}
        self._pending: list[str] = []
        self._alias: dict[str, str] = {}
        self._counter = 0

    # -- labels ------------------------------------------------------------------------
    def fresh(self, base: str = "L") -> str:
        self._counter += 1
        return f"{base}{self._counter}"

    def label(self, name: str | None = None) -> str:
        name = name or self.fresh()
        self._pending.append(name)
        return name

    def phase(self, name: str) -> None:
        self.label(f"phase_{name}")

    def _emit(self, ins: Instr) -> None:
        if self._pending:
            phases = [lab for lab in self._pending if lab.startswith("phase_")]
            first = phases[0] if phases else self._pending[0]
            for lab in self._pending:
                self.labels[lab] = len(self.code)
                if lab != first:
                    self._alias[lab] = first
            ins = Instr(ins.op, ins.a, ins.b, ins.cond, first)
            self._pending = []
        self.code.append(ins)

    # -- instructions --------------------------------------------------------------------
    def new(self, w: str) -> None:
        self._emit(Instr(Op.NEW, w))

    def set(self, w: str, v: str) -> None:
        self._emit(Instr(Op.SET, w, v))

    def goto(self, label: str) -> None:
        self._emit(Instr(Op.GOTO, label))

    def halt(self) -> None:
        self._emit(Instr(Op.HALT))

    def input(self, l0: str, l1: str) -> None:
        self._emit(Instr(Op.INPUT, l0, l1))

    def output(self, bit: int) -> None:
        self._emit(Instr(Op.OUTPUT, str(bit)))

    def if_eq(self, u: str, v: str) -> _Cond:
        return _Cond(self, u, v, True)

    def if_ne(self, u: str, v: str) -> _Cond:
        return _Cond(self, u, v, False)

    # -- structured control (all expand to gotos) ----------------------------------------
    def loop(self) -> Iterator[str]:
        """``with asm.loop() as brk:`` — infinite loop; leave it with ``asm.if_eq(u, v).goto(brk)``."""
        return self._loop()

    def _loop(self) -> Iterator[str]:
        from contextlib import contextmanager  # local import keeps the module header short

        @contextmanager
        def cm() -> Iterator[str]:
            top = self.label()
            brk = self.fresh()
            yield brk
            self.goto(top)
            self.label(brk)

        return cm()

    def while_ne(self, u: str, v: str) -> Iterator[None]:
        from contextlib import contextmanager

        @contextmanager
        def cm() -> Iterator[None]:
            top = self.label()
            end = self.fresh()
            self.if_eq(u, v).goto(end)
            yield
            self.goto(top)
            self.label(end)

        return cm()

    def while_eq(self, u: str, v: str) -> Iterator[None]:
        from contextlib import contextmanager

        @contextmanager
        def cm() -> Iterator[None]:
            top = self.label()
            end = self.fresh()
            self.if_ne(u, v).goto(end)
            yield
            self.goto(top)
            self.label(end)

        return cm()

    def if_block(self, u: str, v: str, eq: bool) -> Iterator[None]:
        """Multi-instruction conditional block, entered iff (p*(U) = p*(V)) == eq."""
        from contextlib import contextmanager

        @contextmanager
        def cm() -> Iterator[None]:
            end = self.fresh()
            (self.if_ne(u, v) if eq else self.if_eq(u, v)).goto(end)
            yield
            self.label(end)

        return cm()

    # -- finish -----------------------------------------------------------------------------
    def program(self, note: dict[str, str] | None = None) -> Program:
        if self._pending:  # labels at the very end: attach to an explicit halt
            self.halt()
        code = []
        for ins in self.code:
            if ins.op is Op.GOTO:
                ins = Instr(ins.op, self._alias.get(ins.a, ins.a), ins.b, ins.cond, ins.label)
            elif ins.op is Op.INPUT:
                ins = Instr(ins.op, self._alias.get(ins.a, ins.a), self._alias.get(ins.b, ins.b), ins.cond, ins.label)
            code.append(ins)
        labels = {lab: i for lab, i in self.labels.items() if lab not in self._alias}
        return Program(code, labels, note or {})


class Registers:
    """Named registers = the 256 pointers of the depth-3 leaves of the tree hanging from A Q.

    A register named ``X`` is a path ``Q`` + 4 letters; ``reg("X") + "S"`` is the S-pointer of the node
    it holds, and so on — every access is one instruction whatever the depth.
    """

    def __init__(self) -> None:
        self._paths: dict[str, str] = {}
        self._free = ["Q" + a + b + c + d for a in ALPHABET for b in ALPHABET for c in ALPHABET for d in ALPHABET]

    def __call__(self, name: str) -> str:
        if name not in self._paths:
            if not self._free:
                raise RuntimeError("register file exhausted (256)")
            self._paths[name] = self._free.pop(0)
        return self._paths[name]

    def build(self, asm: Asm) -> None:
        asm.new("Q")
        for a in ALPHABET:
            asm.new("Q" + a)
            for b in ALPHABET:
                asm.new("Q" + a + b)
                for c in ALPHABET:
                    asm.new("Q" + a + b + c)

    def map(self) -> dict[str, str]:
        return dict(sorted(self._paths.items(), key=lambda kv: kv[1]))


# ======================================================================================
# Part 3 — the multiplication program (§6.3), built phase by phase
# ======================================================================================

DIGITS: Final = 19  # 18 fractional B-ary digits and one leading digit (§6.3)
FRAC_DIGITS: Final = 18
GUARD_COLUMN: Final = 17  # lowest column of the 37-column product that is computed
BATCH_LOG2: Final = 9  # at most 2^9 programs per interpreting session (memory bound, see end_program)


class Builder:
    """Emits the whole SMM program.  Every method ``ph_*`` is one phase; ``emit_*`` are macros.

    Conventions of the Δ-structure (A = centre):
      A Q  — the register tree (:class:`Registers`);  A W — D_0 of the B-scale (as in Fig. 6.1);
      A S, A P — scratch, used exactly as in the sort program of §6.2.
      NIL  — a fixed node that ends every list;  ZERO/ONE — two fixed nodes standing for the bits.
      list node: S → next (NIL at the end), P → previous, W → payload.
      digit cell: W → D_p (the digit), S → next cell (towards the more significant digit), P → previous.
      fixed-point number: 19 cells, least significant first, B's complement, value = Σ c_k B^(k-18).
      complex number Z: Z P → real chain, Z Q → imaginary chain, Z S → next Z, Z W → extra sums
        (twiddles only: (Z W) P → re+im chain, (Z W) Q → im−re chain).
    """

    def __init__(self, stop_after: str | None = None) -> None:
        self.a = Asm()
        self.r = Registers()
        self.stop_after = stop_after

    # ------------------------------------------------------------------ small idioms
    def fresh(self, reg: str) -> None:
        """reg := a new node all of whose pointers point to NIL."""
        self.a.set(reg, self.r("NIL"))
        self.a.new(reg)

    def append(self, tail: str, link_back: bool = True) -> None:
        """Append a NIL-initialised node after the node in ``tail``; ``tail`` then names the new node."""
        a = self.a
        a.set(tail + "S", self.r("NIL"))
        a.new(tail + "S")
        if link_back:
            a.set(tail + "SP", tail)
        a.set(tail, tail + "S")

    def new_list(self, head: str, tail: str) -> None:
        self.fresh(head)
        self.a.set(tail, head)

    def walk(self, cur: str, start: str) -> Iterator[None]:
        """``with self.walk("CUR", head + "S"):`` — for each node of the list starting at ``start``."""
        from contextlib import contextmanager

        a = self.a
        reg = self.r(cur)

        @contextmanager
        def cm() -> Iterator[None]:
            a.set(reg, start)
            with a.while_ne(reg, self.r("NIL")):
                yield
                a.set(reg, reg + "S")

        return cm()

    # ------------------------------------------------------------------ phases
    def build(self) -> Program:
        phases: list[tuple[str, Callable[[], None]]] = [
            ("setup", self.ph_setup),
            ("input", self.ph_input),
            ("params", self.ph_params),
            ("scale", self.ph_scale),
            ("bitchains", self.ph_bitchains),
            ("interp_setup", self.ph_interp_setup),
            ("roots", self.ph_roots),
            ("powers", self.ph_powers),
            ("arrays", self.ph_arrays),
            ("forward", self.ph_forward),
            ("pointwise", self.ph_pointwise),
            ("inverse", self.ph_inverse),
            ("rounding", self.ph_rounding),
            ("output", self.ph_output),
        ]
        tests = {"test_digit_ops": ("bitchains", self.ph_test_digit_ops), "test_numeric": ("interp_setup", self.ph_test_numeric)}
        if self.stop_after in tests:  # a test phase runs right after its prerequisite
            after, fn = tests[self.stop_after]
            names = [name for name, _ in phases]
            phases = phases[: names.index(after) + 1] + [(self.stop_after, fn)]
        for name, fn in phases:
            self.a.phase(name)
            fn()
            if name == self.stop_after:
                break
        self.a.halt()
        return self.a.program(note=self.r.map())

    def ph_setup(self) -> None:
        a, r = self.a, self.r
        r.build(a)
        a.new(r("NIL"))
        for ch in ALPHABET:
            a.set(r("NIL") + ch, r("NIL"))
        self.fresh(r("ZERO"))
        self.fresh(r("ONE"))

    def ph_input(self) -> None:
        """Read the 2N input bits into a doubly linked list; find where y starts (two cursors)."""
        a, r = self.a, self.r
        self.new_list(r("IN_H"), r("IN_T"))
        read = a.label()
        z, o, done = a.fresh("z"), a.fresh("o"), a.fresh("done")
        a.input(z, o)
        a.goto(done)
        a.label(z)
        self.append(r("IN_T"))
        a.set(r("IN_T") + "W", r("ZERO"))
        a.goto(read)
        a.label(o)
        self.append(r("IN_T"))
        a.set(r("IN_T") + "W", r("ONE"))
        a.goto(read)
        a.label(done)
        a.set(r("SLOW"), r("IN_H") + "S")
        a.set(r("FAST"), r("IN_H") + "S")
        with a.while_ne(r("FAST"), r("NIL")):
            a.set(r("FAST"), r("FAST") + "SS")
            a.set(r("SLOW"), r("SLOW") + "S")
        a.set(r("Y_H"), r("SLOW"))  # first bit of y (most significant)
        a.set(r("X_L"), r("SLOW") + "P")  # last bit of x (least significant)

    def ph_params(self) -> None:
        """n = least multiple of 3, n ≥ 6, with n·2^n ≥ 2N (§6.3); lists of n, 2^n and b = n/3 nodes."""
        a, r = self.a, self.r
        self.new_list(r("N_H"), r("N_T"))
        for _ in range(6):
            self.append(r("N_T"))
        self.new_list(r("C_H"), r("C_T"))
        self.append(r("C_T"))  # 2^0
        with self.walk("NC", r("N_H") + "S"):
            self.emit_double_count()
        check = a.label()
        ok = a.fresh("ok")
        a.set(r("CUR"), r("IN_H") + "S")
        with self.walk("NC", r("N_H") + "S"), self.walk("CC", r("C_H") + "S"):
            a.if_eq(r("CUR"), r("NIL")).goto(ok)
            a.set(r("CUR"), r("CUR") + "S")
        a.if_eq(r("CUR"), r("NIL")).goto(ok)
        a.set(r("OLD_NT"), r("N_T"))
        for _ in range(3):
            self.append(r("N_T"))
        with self.walk("NC", r("OLD_NT") + "S"):
            self.emit_double_count()
        a.goto(check)
        a.label(ok)
        self.new_list(r("B_H"), r("B_T"))
        a.set(r("NC"), r("N_H") + "S")
        with a.loop() as brk:
            self.append(r("B_T"))
            a.set(r("NC"), r("NC") + "SSS")
            a.if_eq(r("NC"), r("NIL")).goto(brk)

    def emit_double_count(self) -> None:
        """Double the length of the list C_H … C_T."""
        a, r = self.a, self.r
        a.set(r("OLD_CT"), r("C_T"))
        a.set(r("CUR2"), r("C_H") + "S")
        with a.loop() as brk:
            self.append(r("C_T"))
            a.if_eq(r("CUR2"), r("OLD_CT")).goto(brk)
            a.set(r("CUR2"), r("CUR2") + "S")

    def ph_scale(self) -> None:
        """The B-scale (6.1)–(6.2), by b doublings of the one-node scale, in O(B) steps in total."""
        a, r = self.a, self.r
        a.set("W", r("NIL"))
        a.new("W")
        a.set("WS", "W")
        a.set("WP", "W")
        a.set("WQ", "W")
        with self.walk("BC", r("B_H") + "S"):
            self.new_list(r("NH2"), r("NT2"))
            a.set(r("CI"), r("NH2"))  # cursor E_i over the new list; NH2 = "not started"
            a.set(r("OLD"), "W")
            with a.loop() as brk:
                self.append(r("NT2"), link_back=False)
                a.set(r("EA"), r("NT2"))
                self.append(r("NT2"), link_back=False)
                a.if_eq(r("CI"), r("NH2")).set(r("CI"), r("EA"))
                a.set(r("EA") + "Q", r("CI"))  # D_{2i} Q = D_i
                a.set(r("NT2") + "Q", r("CI"))  # D_{2i+1} Q = D_i
                a.set(r("CI") + "P", r("EA"))  # D_i P = D_{2i}
                a.set(r("CI"), r("CI") + "S")
                a.set(r("OLD"), r("OLD") + "S")
                a.if_eq(r("OLD"), "W").goto(brk)
            a.set(r("NT2") + "S", r("NH2") + "S")  # D_{B-1} S = D_0
            a.set("W", r("NH2") + "S")
        # constants on the scale
        a.set(r("D1"), "WS")
        a.set(r("BM1"), "W")
        with a.while_ne(r("BM1") + "S", "W"):
            a.set(r("BM1"), r("BM1") + "S")
        a.set(r("HALFB"), "WS")
        with self.walk("BC", r("B_H") + "SS"):
            a.set(r("HALFB"), r("HALFB") + "P")

    def ph_bitchains(self) -> None:
        """Fixed bit lists of b (two operands) and 2b (product) bits for the digital operations."""
        a, r = self.a, self.r
        for name in ("BP", "BQ", "AC"):
            self.new_list(r(name + "_H"), r(name + "_T"))
            with self.walk("BC", r("B_H") + "S"):
                self.append(r(name + "_T"))
                a.set(r(name + "_T") + "W", r("ZERO"))
        a.set(r("AC_M"), r("AC_T"))
        with self.walk("BC", r("B_H") + "S"):
            self.append(r("AC_T"))
            a.set(r("AC_T") + "W", r("ZERO"))

    def ph_test_digit_ops(self) -> None:
        """Test-only phase: apply the three basic operations to every pair (p, q) of digits."""
        a, r = self.a, self.r
        for name, emit in (("TA", self.emit_op_add), ("TS", self.emit_op_sub), ("TM", self.emit_op_mul)):
            self.new_list(r(name + "_H"), r(name + "_T"))
            a.set(r("OPP"), "W")
            with a.loop() as brk_p:
                a.set(r("OPQ"), "W")
                with a.loop() as brk_q:
                    emit()
                    self.append(r(name + "_T"))
                    a.set(r(name + "_T") + "P", r("RES_R"))
                    a.set(r(name + "_T") + "Q", r("RES_S"))
                    a.set(r("OPQ"), r("OPQ") + "S")
                    a.if_eq(r("OPQ"), "W").goto(brk_q)
                a.set(r("OPP"), r("OPP") + "S")
                a.if_eq(r("OPP"), "W").goto(brk_p)

    # ------------------------------------------------------------------ digits ↔ bits (§6.1)
    def emit_digit_to_bits(self, digit: str, head: str) -> None:
        """Write the b bits of the digit (a pointer to D_p) into the list after ``head``, LSB first."""
        a, r = self.a, self.r
        a.set(r("DX"), digit)
        with self.walk("BIT", head + "S"):
            a.if_eq(r("DX") + "QP", r("DX")).set(r("BIT") + "W", r("ZERO"))  # even iff D_p Q P = D_p
            a.if_ne(r("DX") + "QP", r("DX")).set(r("BIT") + "W", r("ONE"))
            a.set(r("DX"), r("DX") + "Q")

    def emit_bits_to_digit(self, tail: str, head: str, out: str) -> None:
        """out := D_p for the bits between ``head`` (excluded) and ``tail`` (included), read MSB first."""
        a, r = self.a, self.r
        a.set(out, "W")
        a.set(r("BIT"), tail)
        with a.while_ne(r("BIT"), head):
            a.set(out, out + "P")
            a.if_eq(r("BIT") + "W", r("ONE")).set(out, out + "S")
            a.set(r("BIT"), r("BIT") + "P")

    def emit_full_add(self, abit: str, bbit: str, carry: str) -> None:
        """abit := abit ⊕ bbit ⊕ carry; carry := majority.  Bits are pointers to ZERO/ONE."""
        a, r = self.a, self.r
        a.set(r("NX"), r("ONE"))
        a.if_eq(carry, r("ONE")).set(r("NX"), r("ZERO"))  # NX = ¬carry
        a.set(r("TB"), carry)
        a.if_ne(abit, bbit).set(r("TB"), r("NX"))  # sum bit
        a.if_eq(abit, bbit).set(carry, abit)  # carry' = a if a = b else carry
        a.set(abit, r("TB"))

    def emit_full_sub(self, abit: str, bbit: str, borrow: str) -> None:
        """abit := abit − bbit − borrow (mod 2); borrow := new borrow."""
        a, r = self.a, self.r
        a.set(r("NX"), r("ONE"))
        a.if_eq(borrow, r("ONE")).set(r("NX"), r("ZERO"))
        a.set(r("TB"), borrow)
        a.if_ne(abit, bbit).set(r("TB"), r("NX"))
        a.if_ne(abit, bbit).set(borrow, bbit)  # borrow' = b if a ≠ b else borrow
        a.set(abit, r("TB"))

    def emit_negate_bits(self, head: str) -> None:
        """Two's complement of the bit list after ``head``, in place."""
        a, r = self.a, self.r
        a.set(r("CY"), r("ONE"))
        with self.walk("BIT", head + "S"):
            a.set(r("NA"), r("ONE"))
            a.if_eq(r("BIT") + "W", r("ONE")).set(r("NA"), r("ZERO"))  # ¬a
            a.set(r("NX"), r("ONE"))
            a.if_eq(r("CY"), r("ONE")).set(r("NX"), r("ZERO"))  # ¬c
            a.set(r("TB"), r("CY"))
            a.if_eq(r("NA"), r("ONE")).set(r("TB"), r("NX"))  # ¬a ⊕ c
            a.if_eq(r("NA"), r("ZERO")).set(r("CY"), r("ZERO"))  # c' = ¬a ∧ c
            a.set(r("BIT") + "W", r("TB"))

    # ------------------------------------------------------------------ the basic digital operations
    def emit_op_add(self) -> None:
        """(6.3): OPP + OPQ = RES_R + RES_S·B, through binary, O(b) steps."""
        a, r = self.a, self.r
        self.emit_digit_to_bits(r("OPP"), r("BP_H"))
        self.emit_digit_to_bits(r("OPQ"), r("BQ_H"))
        a.set(r("CY"), r("ZERO"))
        a.set(r("BB"), r("BQ_H") + "S")
        with self.walk("BA", r("BP_H") + "S"):
            self.emit_full_add(r("BA") + "W", r("BB") + "W", r("CY"))
            a.set(r("BB"), r("BB") + "S")
        self.emit_bits_to_digit(r("BP_T"), r("BP_H"), r("RES_R"))
        a.set(r("RES_S"), "W")
        a.if_eq(r("CY"), r("ONE")).set(r("RES_S"), "WS")

    def emit_op_sub(self) -> None:
        """(6.4): OPP − OPQ = RES_R·(−1)^RES_S, with RES_S = 0 for OPP = OPQ."""
        a, r = self.a, self.r
        self.emit_digit_to_bits(r("OPP"), r("BP_H"))
        self.emit_digit_to_bits(r("OPQ"), r("BQ_H"))
        a.set(r("CY"), r("ZERO"))
        a.set(r("BB"), r("BQ_H") + "S")
        with self.walk("BA", r("BP_H") + "S"):
            self.emit_full_sub(r("BA") + "W", r("BB") + "W", r("CY"))
            a.set(r("BB"), r("BB") + "S")
        a.set(r("RES_S"), "W")
        with a.if_block(r("CY"), r("ONE"), True):  # p < q: negate, s = 1
            self.emit_negate_bits(r("BP_H"))
            a.set(r("RES_S"), "WS")
        self.emit_bits_to_digit(r("BP_T"), r("BP_H"), r("RES_R"))

    def emit_op_mul(self) -> None:
        """(6.5): OPP · OPQ = RES_R + RES_S·B, shift-and-add in binary, O(b²) steps."""
        a, r = self.a, self.r
        self.emit_digit_to_bits(r("OPP"), r("BP_H"))
        self.emit_digit_to_bits(r("OPQ"), r("BQ_H"))
        with self.walk("BIT", r("AC_H") + "S"):
            a.set(r("BIT") + "W", r("ZERO"))
        a.set(r("AK"), r("AC_H") + "S")  # accumulator position of bit k of q
        with self.walk("BB", r("BQ_H") + "S"):
            with a.if_block(r("BB") + "W", r("ONE"), True):
                a.set(r("CY"), r("ZERO"))
                a.set(r("BA"), r("BP_H") + "S")
                with self.walk("CC", r("AK")):
                    a.set(r("B2"), r("ZERO"))
                    a.if_ne(r("BA"), r("NIL")).set(r("B2"), r("BA") + "W")
                    a.if_ne(r("BA"), r("NIL")).set(r("BA"), r("BA") + "S")
                    self.emit_full_add(r("CC") + "W", r("B2"), r("CY"))
            a.set(r("AK"), r("AK") + "S")
        self.emit_bits_to_digit(r("AC_M"), r("AC_H"), r("RES_R"))
        self.emit_bits_to_digit(r("AC_T"), r("AC_M"), r("RES_S"))

    # ================================================================== §6.2 interpreting mode
    def ph_interp_setup(self) -> None:
        """Opcode markers, constant cells, the junk cell, the three collector lines, the instance list."""
        a, r = self.a, self.r
        for name in ("ADDM", "SUBM", "MULM", "JUNK"):
            self.fresh(r(name))
        for name, digit in (("ZC", "W"), ("OC", "WS"), ("BM1C", r("BM1")), ("HC", r("HALFB"))):
            self.fresh(r(name))
            a.set(r(name) + "W", digit)
        for name in ("LA", "LS", "LM", "IL"):
            self.new_list(r(name + "_H"), r(name + "_T"))
        # at most BATCH instances per interpreting session (bounds the live structure)
        self.new_list(r("BAT_H"), r("BAT_T"))
        self.append(r("BAT_T"))
        for _ in range(BATCH_LOG2):
            a.set(r("OLD_CT"), r("BAT_T"))
            a.set(r("CUR2"), r("BAT_H") + "S")
            with a.loop() as brk:
                self.append(r("BAT_T"))
                a.if_eq(r("CUR2"), r("OLD_CT")).goto(brk)
                a.set(r("CUR2"), r("CUR2") + "S")
        a.set(r("BATC"), r("BAT_H") + "S")

    def emit_new_cell(self, reg: str) -> None:
        """reg := a fresh digit cell holding D_0."""
        self.fresh(reg)
        self.a.set(reg + "W", "W")

    def emit_new_number(self, head: str) -> None:
        """head := a fresh 19-cell number, all digits 0 (dummy head; head P → last cell)."""
        a, r = self.a, self.r
        self.fresh(head)
        a.set(r("NT"), head)
        for _ in range(DIGITS):
            self.append(r("NT"))
            a.set(r("NT") + "W", "W")
        a.set(head + "P", r("NT"))

    def emit_instr(self, opm: str, p: str, q: str, rdst: str | None, sdst: str | None) -> None:
        """Append one basic operation to the program under construction (PG_T).

        p, q: paths of the operand cells.  rdst/sdst: path of an existing destination cell, or None
        for a fresh cell — the fresh cells are left in registers RC and SC.
        """
        a, r = self.a, self.r
        self.fresh(r("K"))
        a.set(r("K") + "W", opm)
        if rdst is None:
            self.emit_new_cell(r("RC"))
            a.set(r("K") + "P", r("RC"))
        else:
            a.set(r("K") + "P", rdst)
        if sdst is None:
            self.emit_new_cell(r("SC"))
            a.set(r("K") + "Q", r("SC"))
        else:
            a.set(r("K") + "Q", sdst)
        self.append(r("PG_T"), link_back=False)
        a.set(r("PG_T") + "P", p)
        a.set(r("PG_T") + "Q", q)
        a.set(r("PG_T") + "W", r("K"))

    def begin_program(self) -> None:
        self.new_list(self.r("PG_H"), self.r("PG_T"))

    def end_program(self) -> None:
        """Wrap the instruction list into an instance (header H: W → first instruction) on the IL list.

        Every BATCH instances the pending session is run at once (the programs of one stage are
        independent, so splitting a stage into sessions changes nothing but the live structure).
        """
        a, r = self.a, self.r
        self.append(r("IL_T"), link_back=False)
        a.set(r("IL_T") + "W", r("PG_H") + "S")
        a.set(r("BATC"), r("BATC") + "S")
        with a.if_block(r("BATC"), r("NIL"), True):
            self.emit_run_programs()
            a.set(r("BATC"), r("BAT_H") + "S")

    def emit_run_programs(self) -> None:
        """Interpreting mode: sweeps until every instance has halted (Lemma 6.2)."""
        a, r = self.a, self.r
        with a.loop() as done:
            a.set(r("ACTIVE"), r("ZERO"))
            with self.walk("H", r("IL_H") + "S"), a.if_block(r("H") + "W", r("NIL"), False):
                a.set(r("ACTIVE"), r("ONE"))
                a.set(r("I"), r("H") + "W")
                a.set(r("K"), r("I") + "W")
                self.fresh(r("C"))
                a.set(r("C") + "P", r("I") + "PW")  # C_i P = D_p  (6.7)
                a.set(r("C") + "Q", r("I") + "QW")  # C_i Q = D_q
                a.set(r("C") + "W", r("K"))  # C_i W = R_i, the destination
                for line, opm in (("LA", "ADDM"), ("LS", "SUBM"), ("LM", "MULM")):
                    a.if_eq(r("K") + "W", r(opm)).set(r(line + "_T") + "S", r("C"))
                    a.if_eq(r("K") + "W", r(opm)).set(r(line + "_T"), r("C"))
                a.set(r("H") + "W", r("I") + "S")  # advance the instruction pointer
            a.if_eq(r("ACTIVE"), r("ZERO")).goto(done)
            for line, emit in (("LA", self.emit_op_add), ("LS", self.emit_op_sub), ("LM", self.emit_op_mul)):
                with a.if_block(r(line + "_T"), r(line + "_H"), False):
                    a.set(r(line + "_T") + "S", "")  # the end of the line points back to A
                    a.set("S", r(line + "_H") + "S")  # the line hangs from A S
                    self.sort_line("Q")
                    self.sort_line("P")
                    self.emit_execute_sorted(emit)
                    a.set(r(line + "_T"), r(line + "_H"))
        a.set(r("IL_H") + "S", r("NIL"))  # all instances halted: drop the list
        a.set(r("IL_T"), r("IL_H"))

    def sort_line(self, key: str) -> None:
        """The sorting program of §6.2, verbatim (key = "Q" sorts by q, key = "P" by p).

        The line to be sorted hangs from A S and ends with an S-pointer back to A; A W = D_0.
        Afterwards nodes with equal keys are adjacent and the sorted line hangs from A S again.
        """
        a = self.a
        empty, insert, test = a.fresh("empty"), a.fresh("insert"), a.fresh("test")
        a.set("P", "")  # (added) the auxiliary node gets its pointers from A P: point it at A first,
        a.new("P")  # otherwise every sorted line of every sweep would stay reachable through it
        a.set("PS", "")
        a.set("WW", "W")
        a.label(empty)
        a.set("WW", "WWS")
        a.set("WWW", "P")
        a.if_ne("WW", "P").goto(empty)
        a.goto(test)
        a.label(insert)
        a.set("PW", "S")
        a.set("S", "SS")
        a.set("PWS", "PW" + key + "WS")
        a.set("PW" + key + "WS", "PW")
        a.set("PW" + key + "W", "PW")
        a.label(test)
        a.if_ne("S", "").goto(insert)
        a.set("S", "PS")

    def emit_execute_sorted(self, emit_op: Callable[[], None]) -> None:
        """Walk the sorted line at A S: compute each distinct (p, q) once, deliver r and s to all."""
        a, r = self.a, self.r
        a.set(r("LASTP"), r("NIL"))
        a.set(r("LASTQ"), r("NIL"))
        a.set(r("CUR"), "S")
        with a.while_ne(r("CUR"), ""):
            a.set(r("NEEDC"), r("ZERO"))
            a.if_ne(r("CUR") + "P", r("LASTP")).set(r("NEEDC"), r("ONE"))
            a.if_ne(r("CUR") + "Q", r("LASTQ")).set(r("NEEDC"), r("ONE"))
            with a.if_block(r("NEEDC"), r("ONE"), True):
                a.set(r("OPP"), r("CUR") + "P")
                a.set(r("OPQ"), r("CUR") + "Q")
                emit_op()
                a.set(r("LASTP"), r("OPP"))
                a.set(r("LASTQ"), r("OPQ"))
            a.set(r("CUR") + "WPW", r("RES_R"))  # deliver r into the destination cell
            a.set(r("CUR") + "WQW", r("RES_S"))  # deliver s
            a.set(r("CUR"), r("CUR") + "S")

    def emit_bits_to_number(self, cur: str, dst: str) -> None:
        """dst := the number whose 19 digits are the next 19·b input bits at ``cur`` (MSB first)."""
        a, r = self.a, self.r
        a.set(r("CD"), dst + "P")  # c_18
        with a.while_ne(r("CD"), dst):
            a.set(r("DG"), "W")
            with self.walk("BC", r("B_H") + "S"):
                a.set(r("DG"), r("DG") + "P")
                a.if_eq(cur + "W", r("ONE")).set(r("DG"), r("DG") + "S")
                a.set(cur, cur + "S")
            a.set(r("CD") + "W", r("DG"))
            a.set(r("CD"), r("CD") + "P")

    def ph_test_numeric(self) -> None:
        """Test-only phase: numbers XA, XB from the input bits; programs for +, −, ·, /2, complex products."""
        a, r = self.a, self.r
        a.set(r("CUR"), r("IN_H") + "S")
        for name in ("XA", "XB"):
            self.emit_new_number(r(name))
            self.emit_bits_to_number(r("CUR"), r(name))
        self.begin_program()
        self.emit_new_number(r("SUM"))
        self.emit_prog_add(r("XA"), r("XB"), r("SUM"), r("ZC"))
        self.emit_new_number(r("DIF"))
        self.emit_prog_sub(r("XA"), r("XB"), r("DIF"))
        self.emit_prog_mul(r("XA"), r("XB"))
        a.set(r("PROD"), r("MRES"))
        self.emit_new_number(r("HALF"))
        self.emit_prog_halve(r("XA"), r("HALF"))
        self.end_program()
        # complex: T = XA + i·XB (with sums), Z = XB + i·XA
        self.begin_program()
        self.fresh(r("T"))
        a.set(r("T") + "P", r("XA"))
        a.set(r("T") + "Q", r("XB"))
        self.emit_prog_sums(r("T"))
        self.end_program()
        self.emit_run_programs()
        self.begin_program()
        self.fresh(r("Z"))
        a.set(r("Z") + "P", r("XB"))
        a.set(r("Z") + "Q", r("XA"))
        self.emit_new_complex(r("CM"))
        self.emit_prog_cmul(r("Z"), r("T"), False, r("CM") + "P", r("CM") + "Q")
        self.emit_new_complex(r("CMC"))
        self.emit_prog_cmul(r("Z"), r("T"), True, r("CMC") + "P", r("CMC") + "Q")
        self.end_program()
        self.emit_run_programs()

    # ================================================================== numerical programs (B-ary)
    def emit_prog_add(self, x: str, y: str, dst: str, carry_in: str) -> None:
        """Program: dst := x + y (+ carry_in cell), 19 digits, B's complement, 3 operations per digit."""
        a, r = self.a, self.r
        a.set(r("CC"), carry_in)
        a.set(r("CB"), y + "S")
        a.set(r("CD"), dst + "S")
        with self.walk("CA", x + "S"):
            self.emit_instr(r("ADDM"), r("CA"), r("CB"), None, None)
            a.set(r("T1"), r("RC"))
            a.set(r("C1"), r("SC"))
            self.emit_instr(r("ADDM"), r("T1"), r("CC"), r("CD"), None)
            with a.if_block(r("CA") + "S", r("NIL"), False):  # no carry out of the top digit
                self.emit_instr(r("ADDM"), r("C1"), r("SC"), None, r("JUNK"))
                a.set(r("CC"), r("RC"))
            a.set(r("CB"), r("CB") + "S")
            a.set(r("CD"), r("CD") + "S")

    def emit_prog_complement(self, x: str, dst: str) -> None:
        """Program: dst_k := (B−1) − x_k for every digit (6.4 with p = B−1, s = 0 always)."""
        a, r = self.a, self.r
        a.set(r("CD"), dst + "S")
        with self.walk("CA", x + "S"):
            self.emit_instr(r("SUBM"), r("BM1C"), r("CA"), r("CD"), r("JUNK"))
            a.set(r("CD"), r("CD") + "S")

    def emit_prog_sub(self, x: str, y: str, dst: str) -> None:
        """Program: dst := x − y  as  x + (B−1−y) + 1."""
        r = self.r
        self.emit_new_number(r("CPL"))
        self.emit_prog_complement(y, r("CPL"))
        self.emit_prog_add(x, r("CPL"), dst, r("OC"))

    def emit_prog_mul(self, x: str, y: str) -> None:
        """Program: MRES := x · y truncated to 18 fractional digits (columns ≥ GUARD_COLUMN kept).

        Row-wise accumulation into the 20 columns 17…36; B's-complement sign corrections
        −neg_x·y·B^19 − neg_y·x·B^19; the result chain is built from the accumulator cells 18…36.
        """
        a, r = self.a, self.r
        # accumulator: 20 zero cells (columns 17 … 36)
        self.fresh(r("AH"))
        a.set(r("NT"), r("AH"))
        for _ in range(DIGITS + 1):
            self.append(r("NT"))
            a.set(r("NT") + "W", "W")
        # sign digits neg_x, neg_y = carry of x_18 + x_18, y_18 + y_18
        self.emit_instr(r("ADDM"), x + "P", x + "P", r("JUNK"), None)
        a.set(r("NX"), r("SC"))
        self.emit_instr(r("ADDM"), y + "P", y + "P", r("JUNK"), None)
        a.set(r("NY"), r("SC"))
        # rows
        a.set(r("YS"), y + "S")
        for _ in range(GUARD_COLUMN):
            a.set(r("YS"), r("YS") + "S")  # y_17
        a.set(r("AS"), r("AH") + "S")  # column 17
        with self.walk("CX", x + "S"):
            a.set(r("SPREV"), r("ZC"))
            a.set(r("CARRY"), r("ZC"))
            a.set(r("CAK"), r("AS"))
            with self.walk("CYJ", r("YS")):
                self.emit_instr(r("MULM"), r("CX"), r("CYJ"), None, None)
                a.set(r("PR"), r("RC"))
                a.set(r("PS"), r("SC"))
                self.emit_accumulate(r("PR"))
                a.set(r("SPREV"), r("PS"))
                a.set(r("CAK"), r("CAK") + "S")
            with a.if_block(r("CAK"), r("NIL"), False):
                self.emit_accumulate(None)
                a.set(r("CAK"), r("CAK") + "S")
            with a.while_ne(r("CAK"), r("NIL")):  # propagate the carry to the top
                self.emit_instr(r("ADDM"), r("CAK"), r("CARRY"), r("CAK"), None)
                a.set(r("CARRY"), r("SC"))
                a.set(r("CAK"), r("CAK") + "S")
            a.if_eq(r("YS"), y + "S").set(r("AS"), r("AS") + "S")
            a.if_ne(r("YS"), y + "S").set(r("YS"), r("YS") + "P")
        # sign corrections: columns 19 … 36 receive (B−1) − neg·v_j, plus 1 at column 19
        for neg, v in ((r("NX"), y), (r("NY"), x)):
            a.set(r("CAK"), r("AH") + "SSS")
            a.set(r("CV"), v + "S")
            a.set(r("CARRY"), r("OC"))
            with a.while_ne(r("CAK"), r("NIL")):
                self.emit_instr(r("MULM"), neg, r("CV"), None, r("JUNK"))
                a.set(r("T1"), r("RC"))
                self.emit_instr(r("SUBM"), r("BM1C"), r("T1"), None, r("JUNK"))
                a.set(r("T1"), r("RC"))
                self.emit_instr(r("ADDM"), r("CAK"), r("T1"), None, None)
                a.set(r("T2"), r("RC"))
                a.set(r("C1"), r("SC"))
                self.emit_instr(r("ADDM"), r("T2"), r("CARRY"), r("CAK"), None)
                self.emit_instr(r("ADDM"), r("C1"), r("SC"), None, r("JUNK"))
                a.set(r("CARRY"), r("RC"))
                a.set(r("CAK"), r("CAK") + "S")
                a.set(r("CV"), r("CV") + "S")
        # result = accumulator cells of columns 18 … 36, relinked under a fresh head
        self.fresh(r("MRES"))
        a.set(r("MRES") + "S", r("AH") + "SS")
        a.set(r("MRES") + "SP", r("MRES"))
        a.set(r("MRES") + "P", r("NT"))

    def emit_accumulate(self, pr: str | None) -> None:
        """CAK := CAK (+ pr) + SPREV + CARRY; CARRY := carry out (≤ 3 < B)."""
        a, r = self.a, self.r
        if pr is not None:
            self.emit_instr(r("ADDM"), r("CAK"), pr, None, None)
            a.set(r("T1"), r("RC"))
            a.set(r("C1"), r("SC"))
            self.emit_instr(r("ADDM"), r("T1"), r("SPREV"), None, None)
        else:
            self.emit_instr(r("ADDM"), r("CAK"), r("SPREV"), None, None)
            a.set(r("C1"), r("ZC"))
        a.set(r("T2"), r("RC"))
        a.set(r("C2"), r("SC"))
        self.emit_instr(r("ADDM"), r("T2"), r("CARRY"), r("CAK"), None)
        a.set(r("C3"), r("SC"))
        self.emit_instr(r("ADDM"), r("C1"), r("C2"), None, r("JUNK"))
        a.set(r("T1"), r("RC"))
        self.emit_instr(r("ADDM"), r("T1"), r("C3"), None, r("JUNK"))
        a.set(r("CARRY"), r("RC"))

    def emit_prog_halve(self, u: str, dst: str) -> None:
        """Program: dst := u / 2 (arithmetic shift), via u_k · (B/2) = (u_k mod 2)·B/2 + ⌊u_k/2⌋·B."""
        a, r = self.a, self.r
        a.set(r("CD"), dst + "S")
        self.emit_instr(r("MULM"), u + "S", r("HC"), None, None)
        a.set(r("HS"), r("SC"))  # ⌊u_0/2⌋
        with self.walk("CA", u + "SS"):
            self.emit_instr(r("MULM"), r("CA"), r("HC"), None, None)
            a.set(r("HR"), r("RC"))  # (u_k mod 2)·B/2
            self.emit_instr(r("ADDM"), r("HS"), r("HR"), r("CD"), r("JUNK"))
            a.set(r("HS"), r("SC"))
            a.set(r("CD"), r("CD") + "S")
        # top digit: ⌊u_18/2⌋ + sign·B/2
        self.emit_instr(r("ADDM"), u + "P", u + "P", r("JUNK"), None)
        a.set(r("T1"), r("SC"))
        self.emit_instr(r("MULM"), r("T1"), r("HC"), None, r("JUNK"))
        a.set(r("T1"), r("RC"))
        self.emit_instr(r("ADDM"), r("HS"), r("T1"), r("CD"), r("JUNK"))

    def emit_prog_cmul(self, z: str, t: str, conj: bool, dre: str, dim: str) -> None:
        """Program: (dre, dim) := z · t  (or z · conj(t)), t a twiddle with precomputed sums.

        Three real products: k1 = t_re·(z_re + z_im), k2 = z_re·(t_im − t_re), k3 = z_im·(t_re + t_im);
        z·t = (k1 − k3) + i(k1 + k2).  For conj(t) swap the two sums: re = k1 + z_im·(t_im − t_re),
        im = k1 − z_re·(t_re + t_im).
        """
        a, r = self.a, self.r
        self.emit_new_number(r("SUMZ"))
        self.emit_prog_add(z + "P", z + "Q", r("SUMZ"), r("ZC"))
        self.emit_prog_mul(t + "P", r("SUMZ"))
        a.set(r("K1"), r("MRES"))
        sp, sm = t + "WP", t + "WQ"
        self.emit_prog_mul(z + "P", sp if conj else sm)
        a.set(r("K2"), r("MRES"))
        self.emit_prog_mul(z + "Q", sm if conj else sp)
        a.set(r("K3"), r("MRES"))
        if conj:
            self.emit_prog_add(r("K1"), r("K3"), dre, r("ZC"))
            self.emit_prog_sub(r("K1"), r("K2"), dim)
        else:
            self.emit_prog_sub(r("K1"), r("K3"), dre)
            self.emit_prog_add(r("K1"), r("K2"), dim, r("ZC"))

    def emit_prog_sums(self, z: str) -> None:
        """Program: attach to complex z the sums node (z W): P → re+im, Q → im−re (for twiddles)."""
        a, r = self.a, self.r
        self.fresh(r("SN"))
        a.set(z + "W", r("SN"))
        self.emit_new_number(r("SPN"))
        a.set(r("SN") + "P", r("SPN"))
        self.emit_prog_add(z + "P", z + "Q", r("SPN"), r("ZC"))
        self.emit_new_number(r("SMN"))
        a.set(r("SN") + "Q", r("SMN"))
        self.emit_prog_sub(z + "Q", z + "P", r("SMN"))

    def emit_new_complex(self, reg: str) -> None:
        """reg := fresh complex node with two fresh zero numbers."""
        a, r = self.a, self.r
        self.fresh(reg)
        self.emit_new_number(r("NRE"))
        a.set(reg + "P", r("NRE"))
        self.emit_new_number(r("NIM"))
        a.set(reg + "Q", r("NIM"))

    # ================================================================== roots of unity (§6.3), binary
    def emit_new_bits(self, head: str, count_head: str) -> None:
        """head := a fresh list of ZERO bits, one per node of the list at ``count_head``; head P → tail."""
        a, r = self.a, self.r
        self.fresh(head)
        a.set(r("NT"), head)
        with self.walk("KC", count_head + "S"):
            self.append(r("NT"))
            a.set(r("NT") + "W", r("ZERO"))
        a.set(head + "P", r("NT"))

    def emit_bits_clear(self, head: str) -> None:
        with self.walk("BIT", head + "S"):
            self.a.set(self.r("BIT") + "W", self.r("ZERO"))

    def emit_bits_copy(self, src: str, dst: str) -> None:
        a, r = self.a, self.r
        a.set(r("BIT2"), dst + "S")
        with self.walk("BIT", src + "S"):
            a.set(r("BIT2") + "W", r("BIT") + "W")
            a.set(r("BIT2"), r("BIT2") + "S")

    def emit_bits_shl(self, head: str, newbit: str) -> None:
        """Shift the list one position towards the MSB (the top bit is lost); bit 0 := newbit."""
        a, r = self.a, self.r
        a.set(r("BIT"), head + "P")
        with a.while_ne(r("BIT") + "P", head):
            a.set(r("BIT") + "W", r("BIT") + "PW")
            a.set(r("BIT"), r("BIT") + "P")
        a.set(r("BIT") + "W", newbit)

    def emit_bits_sub_if_ge(self, rem: str, sub: str, tmp: str) -> None:
        """If rem ≥ sub: rem := rem − sub and FLAG := ONE, else FLAG := ZERO (equal lengths)."""
        a, r = self.a, self.r
        a.set(r("CY"), r("ZERO"))
        a.set(r("BIT2"), sub + "S")
        a.set(r("BIT3"), tmp + "S")
        with self.walk("BIT", rem + "S"):
            a.set(r("BIT3") + "W", r("BIT") + "W")
            self.emit_full_sub(r("BIT3") + "W", r("BIT2") + "W", r("CY"))
            a.set(r("BIT2"), r("BIT2") + "S")
            a.set(r("BIT3"), r("BIT3") + "S")
        a.set(r("FLAG"), r("ONE"))
        a.if_eq(r("CY"), r("ONE")).set(r("FLAG"), r("ZERO"))
        with a.if_block(r("FLAG"), r("ONE"), True):
            self.emit_bits_copy(tmp, rem)

    def emit_bits_place_shifted(self, src: str, dst: str) -> None:
        """dst := src · 2^L  (copy the bits of src into dst starting at position L; dst cleared first)."""
        a, r = self.a, self.r
        self.emit_bits_clear(dst)
        a.set(r("BIT2"), dst + "S")
        with self.walk("LC", r("L_H") + "S"):
            a.set(r("BIT2"), r("BIT2") + "S")
        with self.walk("BIT", src + "S"), a.if_block(r("BIT2"), r("NIL"), False):
            a.set(r("BIT2") + "W", r("BIT") + "W")
            a.set(r("BIT2"), r("BIT2") + "S")

    def emit_bits_sqrt(self, x: str, res: str) -> None:
        """res := ⌊√x⌋ on K-bit lists (restoring, two bits of x per step)."""
        a, r = self.a, self.r
        self.emit_bits_clear(r("REM"))
        self.emit_bits_clear(res)
        a.set(r("CX"), x + "P")
        with a.while_ne(r("CX"), x):
            a.set(r("B1"), r("CX") + "W")
            a.set(r("B0"), r("CX") + "PW")
            a.set(r("CX"), r("CX") + "PP")
            self.emit_bits_shl(r("REM"), r("B1"))
            self.emit_bits_shl(r("REM"), r("B0"))
            self.emit_bits_copy(res, r("TRIAL"))
            self.emit_bits_shl(r("TRIAL"), r("ZERO"))
            self.emit_bits_shl(r("TRIAL"), r("ONE"))  # 4·res + 1
            self.emit_bits_sub_if_ge(r("REM"), r("TRIAL"), r("TMPB"))
            self.emit_bits_shl(res, r("FLAG"))

    def emit_bits_div(self, x: str, d: str, quo: str) -> None:
        """quo := ⌊x / d⌋ on K-bit lists (restoring long division)."""
        a, r = self.a, self.r
        self.emit_bits_clear(r("REM"))
        self.emit_bits_clear(quo)
        a.set(r("CX"), x + "P")
        with a.while_ne(r("CX"), x):
            self.emit_bits_shl(r("REM"), r("CX") + "W")
            self.emit_bits_sub_if_ge(r("REM"), d, r("TMPB"))
            self.emit_bits_shl(quo, r("FLAG"))
            a.set(r("CX"), r("CX") + "P")

    def emit_bits_to_bary(self, src: str, dst: str) -> None:
        """dst (a fresh 19-digit number) := the binary fixed-point list src truncated to 18 digits.

        Bits 8 … 8+19b−1 of src (L−18b = 8 guard bits dropped) are grouped b at a time, MSB first.
        """
        a, r = self.a, self.r
        self.emit_new_number(dst)
        a.set(r("BIT"), src + "S")
        for _ in range(7):
            a.set(r("BIT"), r("BIT") + "S")
        for _ in range(DIGITS):
            with self.walk("BC", r("B_H") + "S"):
                a.set(r("BIT"), r("BIT") + "S")
        a.set(r("CD"), dst + "P")  # BIT is now the top bit of digit 18; walk down
        with a.while_ne(r("CD"), dst):
            a.set(r("DG"), "W")
            with self.walk("BC", r("B_H") + "S"):
                a.set(r("DG"), r("DG") + "P")
                a.if_eq(r("BIT") + "W", r("ONE")).set(r("DG"), r("DG") + "S")
                a.set(r("BIT"), r("BIT") + "P")
            a.set(r("CD") + "W", r("DG"))
            a.set(r("CD"), r("CD") + "P")

    def emit_const_number(self, head: str, top_digit: str) -> None:
        """head := fresh number with all digits 0 except the leading one (1 → "1", B−1 → "−1")."""
        self.emit_new_number(head)
        self.a.set(head + "PW", top_digit)

    def emit_const_complex(self, reg: str, re_top: str | None, im_top: str | None, sp_top: str, sm_top: str) -> None:
        a, r = self.a, self.r
        self.fresh(reg)
        for letter, top in (("P", re_top), ("Q", im_top)):
            self.emit_new_number(r("CN"))
            if top is not None:
                a.set(r("CN") + "PW", top)
            a.set(reg + letter, r("CN"))
        self.fresh(r("SN"))
        a.set(reg + "W", r("SN"))
        self.emit_const_number(r("CN"), sp_top)
        a.set(r("SN") + "P", r("CN"))
        self.emit_const_number(r("CN"), sm_top)
        a.set(r("SN") + "Q", r("CN"))

    def ph_roots(self) -> None:
        """w_2 = i and w_{ν+1} = (1 + w_ν)/|1 + w_ν| for ν = 2 … n−1, in binary fixed point with
        L = 6n + 8 fractional bits (K = 2L + 8 bit lists), then translated into B-ary form.
        ROOTS list: node ν−2 (ν = 2 … n) W → complex node w_ν (with sums)."""
        a, r = self.a, self.r
        # L chain (6n + 8 nodes) and K chain (2L + 8 nodes)
        self.new_list(r("L_H"), r("L_T"))
        for _ in range(6):
            with self.walk("NC", r("N_H") + "S"):
                self.append(r("L_T"))
        for _ in range(8):
            self.append(r("L_T"))
        self.new_list(r("K_H"), r("K_T"))
        for _ in range(2):
            with self.walk("LC", r("L_H") + "S"):
                self.append(r("K_T"))
        for _ in range(8):
            self.append(r("K_T"))
        for name in ("REM", "TRIAL", "TMPB", "CB", "SB", "M2", "XB", "MB", "DB"):
            self.emit_new_bits(r(name), r("K_H"))
        # position L of a K-list, kept as an offset walker: bit L of CB/SB is the units bit
        self.new_list(r("ROOTS_H"), r("ROOTS_T"))
        self.emit_const_complex(r("WI"), None, "WS", "WS", "WS")  # i: re 0, im 1; sums 1, 1
        self.append(r("ROOTS_T"))
        a.set(r("ROOTS_T") + "W", r("WI"))
        # binary c = 0, s = 1 (bit L of SB)
        a.set(r("BIT"), r("SB") + "S")
        with self.walk("LC", r("L_H") + "S"):
            a.set(r("BIT"), r("BIT") + "S")
        a.set(r("BIT") + "W", r("ONE"))
        a.set(r("UNIT_OFF"), r("BIT"))  # (not reused; documents where bit L sits)
        # ν = 2 … n−1: one iteration per node of the N chain beyond the first two
        with self.walk("NC", r("N_H") + "SSS"):
            # M2 := 1 + c, then ×2
            self.emit_bits_copy(r("CB"), r("M2"))
            a.set(r("BIT"), r("M2") + "S")
            with self.walk("LC", r("L_H") + "S"):
                a.set(r("BIT"), r("BIT") + "S")
            a.set(r("BIT") + "W", r("ONE"))  # c < 1, so bit L of c is 0: 1 + c
            self.emit_bits_shl(r("M2"), r("ZERO"))  # 2 + 2c
            self.emit_bits_place_shifted(r("M2"), r("XB"))
            self.emit_bits_sqrt(r("XB"), r("MB"))  # m = |1 + w|, L fractional bits
            # c' = (1 + c)/m
            self.emit_bits_copy(r("CB"), r("M2"))
            a.set(r("BIT"), r("M2") + "S")
            with self.walk("LC", r("L_H") + "S"):
                a.set(r("BIT"), r("BIT") + "S")
            a.set(r("BIT") + "W", r("ONE"))
            self.emit_bits_place_shifted(r("M2"), r("XB"))
            self.emit_bits_div(r("XB"), r("MB"), r("DB"))
            self.emit_bits_copy(r("DB"), r("CB"))
            # s' = s/m
            self.emit_bits_place_shifted(r("SB"), r("XB"))
            self.emit_bits_div(r("XB"), r("MB"), r("DB"))
            self.emit_bits_copy(r("DB"), r("SB"))
            # B-ary complex node
            self.fresh(r("WN"))
            self.emit_bits_to_bary(r("CB"), r("CN"))
            a.set(r("WN") + "P", r("CN"))
            self.emit_bits_to_bary(r("SB"), r("CN"))
            a.set(r("WN") + "Q", r("CN"))
            self.append(r("ROOTS_T"))
            a.set(r("ROOTS_T") + "W", r("WN"))
        # sums re+im, im−re of every root ν ≥ 3, by one interpreting session
        with self.walk("RC2", r("ROOTS_H") + "SS"):
            self.begin_program()
            self.emit_prog_sums(r("RC2") + "W")
            self.end_program()
        self.emit_run_programs()

    # ================================================================== powers of the roots, (6.11)
    def ph_powers(self) -> None:
        """LV list: node ν (ν = 1 … n) W → list of the complex nodes w_ν^κ, κ < 2^(ν−1), with sums."""
        a, r = self.a, self.r
        self.new_list(r("LV_H"), r("LV_T"))
        self.emit_const_complex(r("ONEZ"), "WS", None, "WS", r("BM1"))  # 1: sums 1, −1
        # level 1 = {1}
        self.new_list(r("LH"), r("LT"))
        self.append(r("LT"), link_back=False)
        a.set(r("LT") + "P", r("ONEZ") + "P")
        a.set(r("LT") + "Q", r("ONEZ") + "Q")
        a.set(r("LT") + "W", r("ONEZ") + "W")
        self.append(r("LV_T"))
        a.set(r("LV_T") + "W", r("LH"))
        # level 2 = {1, i}
        self.new_list(r("LH"), r("LT"))
        for src in (r("ONEZ"), r("ROOTS_H") + "SW"):
            self.append(r("LT"), link_back=False)
            a.set(r("LT") + "P", src + "P")
            a.set(r("LT") + "Q", src + "Q")
            a.set(r("LT") + "W", src + "W")
        self.append(r("LV_T"))
        a.set(r("LV_T") + "W", r("LH"))
        # levels 3 … n
        a.set(r("RTC"), r("ROOTS_H") + "SS")  # root w_3
        with self.walk("NC", r("N_H") + "SSS"):
            a.set(r("PREV"), r("LV_T") + "W")
            self.new_list(r("LH"), r("LT"))
            with self.walk("ZC2", r("PREV") + "S"):
                self.append(r("LT"), link_back=False)  # even power: shares the chains of w_ν^κ
                a.set(r("LT") + "P", r("ZC2") + "P")
                a.set(r("LT") + "Q", r("ZC2") + "Q")
                a.set(r("LT") + "W", r("ZC2") + "W")
                self.append(r("LT"), link_back=False)  # odd power: w_ν^κ · w_{ν+1}
                self.emit_new_number(r("CN"))
                a.set(r("LT") + "P", r("CN"))
                self.emit_new_number(r("CN"))
                a.set(r("LT") + "Q", r("CN"))
                self.begin_program()
                self.emit_prog_cmul(r("ZC2"), r("RTC") + "W", False, r("LT") + "P", r("LT") + "Q")
                self.emit_prog_sums(r("LT"))
                self.end_program()
            self.emit_run_programs()
            self.append(r("LV_T"))
            a.set(r("LV_T") + "W", r("LH"))
            a.set(r("RTC"), r("RTC") + "S")

    # ================================================================== the arrays and the input pieces (6.8)
    def ph_arrays(self) -> None:
        a, r = self.a, self.r
        for arr in ("AX", "AY"):
            self.new_list(r(arr + "_H"), r(arr + "_T"))
            with self.walk("CC", r("C_H") + "S"):
                self.append(r(arr + "_T"), link_back=False)
                self.emit_new_number(r("CN"))
                a.set(r(arr + "_T") + "P", r("CN"))
                self.emit_new_number(r("CN"))
                a.set(r(arr + "_T") + "Q", r("CN"))
        # n-bit buffer with the group boundaries after b and 2b bits
        self.emit_new_bits(r("TMPN"), r("N_H"))
        a.set(r("G1"), r("TMPN") + "S")
        with self.walk("BC", r("B_H") + "SS"):
            a.set(r("G1"), r("G1") + "S")
        a.set(r("G2"), r("G1"))
        with self.walk("BC", r("B_H") + "S"):
            a.set(r("G2"), r("G2") + "S")
        for arr, start, stop in (("AX", r("X_L"), r("IN_H")), ("AY", r("IN_T"), r("X_L"))):
            a.set(r("CUR"), start)
            a.set(r("EL"), r(arr + "_H") + "S")
            with a.while_ne(r("CUR"), stop):
                with self.walk("BIT", r("TMPN") + "S"):
                    a.set(r("BIT") + "W", r("ZERO"))
                    a.if_ne(r("CUR"), stop).set(r("BIT") + "W", r("CUR") + "W")
                    a.if_ne(r("CUR"), stop).set(r("CUR"), r("CUR") + "P")
                a.set(r("CD"), r("EL") + "PS")
                for _ in range(12):
                    a.set(r("CD"), r("CD") + "S")
                for tail, head in ((r("G1"), r("TMPN")), (r("G2"), r("G1")), (r("TMPN") + "P", r("G2"))):
                    self.emit_bits_to_digit(tail, head, r("DG"))
                    a.set(r("CD") + "W", r("DG"))
                    a.set(r("CD"), r("CD") + "S")
                a.set(r("EL"), r("EL") + "S")

    # ================================================================== the transforms
    def emit_pairs(self, arr: str, body: Callable[[], None]) -> None:
        """Enumerate the butterflies of one stage: pairs (F, G = F + span) with twiddle TN, span = |TW|."""
        a, r = self.a, self.r
        a.set(r("F"), arr + "S")
        a.set(r("G"), r("F"))
        with self.walk("TN", r("TW") + "S"):
            a.set(r("G"), r("G") + "S")
        with a.while_ne(r("F"), r("NIL")):
            with self.walk("TN", r("TW") + "S"):
                body()
                a.set(r("F"), r("F") + "S")
                a.set(r("G"), r("G") + "S")
            a.set(r("F"), r("G"))
            with self.walk("TN", r("TW") + "S"):
                a.set(r("G"), r("G") + "S")

    def emit_dif_butterfly(self) -> None:
        """Program: (F, G) := (F + G, (F − G)·TN)."""
        a, r = self.a, self.r
        self.begin_program()
        self.fresh(r("VZ"))
        self.emit_new_number(r("CN"))
        a.set(r("VZ") + "P", r("CN"))
        self.emit_prog_sub(r("F") + "P", r("G") + "P", r("CN"))
        self.emit_new_number(r("CN"))
        a.set(r("VZ") + "Q", r("CN"))
        self.emit_prog_sub(r("F") + "Q", r("G") + "Q", r("CN"))
        self.emit_prog_add(r("F") + "P", r("G") + "P", r("F") + "P", r("ZC"))
        self.emit_prog_add(r("F") + "Q", r("G") + "Q", r("F") + "Q", r("ZC"))
        self.emit_prog_cmul(r("VZ"), r("TN"), False, r("G") + "P", r("G") + "Q")
        self.end_program()

    def emit_dit_butterfly(self) -> None:
        """Program: (F, G) := ((F + G·conj TN)/2, (F − G·conj TN)/2)."""
        r = self.r
        self.begin_program()
        self.emit_new_complex(r("GW"))
        self.emit_prog_cmul(r("G"), r("TN"), True, r("GW") + "P", r("GW") + "Q")
        for letter in ("P", "Q"):
            self.emit_new_number(r("UN"))
            self.emit_prog_add(r("F") + letter, r("GW") + letter, r("UN"), r("ZC"))
            self.emit_new_number(r("VN"))
            self.emit_prog_sub(r("F") + letter, r("GW") + letter, r("VN"))
            self.emit_prog_halve(r("UN"), r("F") + letter)
            self.emit_prog_halve(r("VN"), r("G") + letter)
        self.end_program()

    def ph_forward(self) -> None:
        """Decimation in frequency, n stages, both arrays in the same interpreting sessions."""
        a, r = self.a, self.r
        a.set(r("LVC"), r("LV_T"))  # level n → stage span 2^(n−1)
        with a.while_ne(r("LVC"), r("LV_H")):
            a.set(r("TW"), r("LVC") + "W")
            for arr in ("AX", "AY"):
                self.emit_pairs(r(arr + "_H"), self.emit_dif_butterfly)
            self.emit_run_programs()
            a.set(r("LVC"), r("LVC") + "P")

    def ph_pointwise(self) -> None:
        """AX_k := AX_k · AY_k, the sums of AY_k computed inside each program."""
        a, r = self.a, self.r
        a.set(r("EY"), r("AY_H") + "S")
        with self.walk("EL", r("AX_H") + "S"):
            self.begin_program()
            self.emit_prog_sums(r("EY"))
            self.emit_prog_cmul(r("EL"), r("EY"), False, r("EL") + "P", r("EL") + "Q")
            self.end_program()
            a.set(r("EY"), r("EY") + "S")
        self.emit_run_programs()

    def ph_inverse(self) -> None:
        """Decimation in time with conjugate twiddles and halving; the input is in bit-reversed order."""
        a, r = self.a, self.r
        with self.walk("LVC", r("LV_H") + "S"):
            a.set(r("TW"), r("LVC") + "W")
            self.emit_pairs(r("AX_H"), self.emit_dit_butterfly)
            self.emit_run_programs()

    # ================================================================== rounding, (6.12), output
    def ph_rounding(self) -> None:
        """Add ½·2^-4n (= B/2 at cell 5) to every z_j and propagate the carry up to the leading digit."""
        a, r = self.a, self.r
        with self.walk("EL", r("AX_H") + "S"):
            self.begin_program()
            a.set(r("CD"), r("EL") + "PS")
            for _ in range(5):
                a.set(r("CD"), r("CD") + "S")
            self.emit_instr(r("ADDM"), r("CD"), r("HC"), r("CD"), None)
            a.set(r("CARRY"), r("SC"))
            a.set(r("CD"), r("CD") + "S")
            with a.while_ne(r("CD"), r("NIL")):
                self.emit_instr(r("ADDM"), r("CD"), r("CARRY"), r("CD"), None)
                a.set(r("CARRY"), r("SC"))
                a.set(r("CD"), r("CD") + "S")
            self.end_program()
        self.emit_run_programs()

    def ph_output(self) -> None:
        """Z_j = digits 6 … 14 of z_j → 3n bits; xy = Σ Z_j 2^{nj} (6.12), bit-serial; output MSB first."""
        a, r = self.a, self.r
        # bit lists of the elements (EL W → list head)
        with self.walk("EL", r("AX_H") + "S"):
            self.new_list(r("BL_H"), r("BL_T"))
            a.set(r("EL") + "W", r("BL_H"))
            a.set(r("CD"), r("EL") + "PS")
            for _ in range(6):
                a.set(r("CD"), r("CD") + "S")
            for _ in range(9):
                a.set(r("DX"), r("CD") + "W")
                with self.walk("BC", r("B_H") + "S"):
                    self.append(r("BL_T"))
                    a.set(r("BL_T") + "W", r("ZERO"))
                    a.if_ne(r("DX") + "QP", r("DX")).set(r("BL_T") + "W", r("ONE"))
                    a.set(r("DX"), r("DX") + "Q")
                a.set(r("CD"), r("CD") + "S")
        # the counting scale K_0 … K_5: W = parity, P = K_{⌊k/2⌋}
        self.new_list(r("KN"), r("KT"))
        for k in range(6):
            self.append(r("KT"), link_back=False)
            a.set(r("KT") + "W", r("ONE") if k % 2 else r("ZERO"))
        a.set(r("KT"), r("KN") + "S")  # K_0
        for k in range(6):
            a.set(r("KN") + "S" * (k + 1) + "P", r("KN") + "S" * (k // 2 + 1))
        # bit-serial summation over 2N positions (the input list is the counter)
        for name in ("CA", "CB", "CCU"):
            a.set(r(name), r("NIL"))
        a.set(r("CARRY"), r("KT"))
        a.set(r("EL"), r("AX_H") + "S")
        a.set(r("NCNT"), r("NIL"))
        a.set(r("OUT_TOP"), r("NIL"))
        with self.walk("POS", r("IN_H") + "S"):
            with a.if_block(r("NCNT"), r("NIL"), True):  # a new element starts every n bits
                a.set(r("NCNT"), r("N_H") + "S")
                a.set(r("CCU"), r("CB"))
                a.set(r("CB"), r("CA"))
                a.set(r("CA"), r("EL") + "WS")  # NIL when the elements are exhausted
                a.set(r("EL"), r("EL") + "S")
            a.set(r("CNT"), r("CARRY"))
            for name in ("CA", "CB", "CCU"):
                a.if_eq(r(name) + "W", r("ONE")).set(r("CNT"), r("CNT") + "S")
            a.set(r("OUTN"), r("OUT_TOP"))
            a.new(r("OUTN"))
            a.set(r("OUTN") + "W", r("CNT") + "W")  # OUTN S = old top (new-semantics)
            a.set(r("OUT_TOP"), r("OUTN"))
            a.set(r("CARRY"), r("CNT") + "P")
            for name in ("CA", "CB", "CCU"):
                a.set(r(name), r(name) + "S")
            a.set(r("NCNT"), r("NCNT") + "S")
        with self.walk("OUTN", r("OUT_TOP")):
            a.if_eq(r("OUTN") + "W", r("ONE")).output(1)
            a.if_ne(r("OUTN") + "W", r("ONE")).output(0)


def build_program(stop_after: str | None = None) -> Program:
    return Builder(stop_after).build()


# ======================================================================================
# Part 4 — reading the structure back (tests only)
# ======================================================================================


class Inspector:
    """Decodes the final Δ-structure of a run: registers, lists, the B-scale, digits, numbers."""

    def __init__(self, res: RunResult, prog: Program) -> None:
        self.res = res
        self.regs = prog.note
        self.nil = self.reg("NIL")
        self.zero = self.reg("ZERO")
        self.one = self.reg("ONE")
        d0 = res.node("W")
        self.scale: list[int] = [d0]
        node = res.follow(d0, "S")
        while node != d0 and len(self.scale) <= 1 << 20:
            self.scale.append(node)
            node = res.follow(node, "S")
        self.index = {nd: i for i, nd in enumerate(self.scale)}

    def reg(self, name: str) -> int:
        return self.res.node(self.regs[name])

    def lst(self, head: int) -> list[int]:
        out = []
        node = self.res.follow(head, "S")
        while node != self.nil:
            out.append(node)
            node = self.res.follow(node, "S")
        return out

    def bit(self, node: int) -> int:
        w = self.res.follow(node, "W")
        if w == self.zero:
            return 0
        if w == self.one:
            return 1
        raise ValueError("not a bit")

    def digit(self, cell: int) -> int:
        return self.index[self.res.follow(cell, "W")]

    def B(self) -> int:
        return len(self.scale)

    def number(self, head: int) -> int:
        """Signed value of a 19-digit B's-complement number (as an integer multiple of B^-18)."""
        cells = self.lst(head)
        assert len(cells) == DIGITS, len(cells)
        B = self.B()
        v = 0
        for cell in reversed(cells):
            v = v * B + self.digit(cell)
        M = B**DIGITS
        return v - M if v >= M // 2 else v

    def complex(self, z: int) -> tuple[int, int]:
        return self.number(self.res.follow(z, "P")), self.number(self.res.follow(z, "Q"))


# ======================================================================================
# Part 5 — host model of the same fixed-point arithmetic (verification only)
# ======================================================================================


class FixedModel:
    """Exact emulation of the machine's B-ary fixed-point arithmetic (§6.3), on host integers.

    A number is an int X with |X| < B^19/2 standing for X·B^-18.  Every method mirrors one of the
    numerical programs the SMM builds, operation for operation, so intermediate values can be
    compared digit by digit.
    """

    def __init__(self, n: int) -> None:
        if n % 3 or n < 6:
            raise ValueError("n must be a multiple of 3, n ≥ 6")
        self.n = n
        self.b = n // 3
        self.B = 1 << self.b
        self.M = self.B**DIGITS

    @staticmethod
    def choose_n(N: int) -> int:
        n = 6
        while n * (1 << n) < 2 * N:
            n += 3
        return n

    def wrap(self, X: int) -> int:
        X %= self.M
        return X - self.M if X >= self.M // 2 else X

    def add(self, X: int, Y: int) -> int:
        return self.wrap(X + Y)

    def sub(self, X: int, Y: int) -> int:
        return self.wrap(X - Y)

    @staticmethod
    def half(X: int) -> int:
        return X >> 1

    def mul(self, X: int, Y: int) -> int:
        B, M = self.B, self.M
        Xu, Yu = X % M, Y % M
        xd = [(Xu // B**i) % B for i in range(DIGITS)]
        yd = [(Yu // B**j) % B for j in range(DIGITS)]
        P = 0
        for i in range(DIGITS):
            for j in range(max(0, GUARD_COLUMN - i), DIGITS):
                P += xd[i] * yd[j] * B ** (i + j)
        P -= B**DIGITS * (int(X < 0) * Yu + int(Y < 0) * Xu)
        return self.wrap((P % (M * M)) // B**FRAC_DIGITS)

    def cmul(self, z: tuple[int, int], t: tuple[int, int], conj: bool = False) -> tuple[int, int]:
        zr, zi = z
        tr, ti = t
        sp, sm = self.add(tr, ti), self.sub(ti, tr)
        k1 = self.mul(tr, self.add(zr, zi))
        if conj:
            return self.add(k1, self.mul(zi, sm)), self.sub(k1, self.mul(zr, sp))
        return self.sub(k1, self.mul(zi, sp)), self.add(k1, self.mul(zr, sm))

    def from_frac(self, q: Fraction) -> int:
        return self.wrap((q * self.B**FRAC_DIGITS).__floor__())

    def to_frac(self, X: int) -> Fraction:
        return Fraction(X, self.B**FRAC_DIGITS)

    # -- roots of unity, "in any crude manner" (§6.3) -----------------------------------------
    def root_bits(self) -> int:
        return 6 * self.n + 8  # fractional bits of the binary fixed-point root computation

    def roots(self) -> list[tuple[int, int]]:
        """w_ν for ν = 2 … n (index ν), each as a B-ary pair, by w_{ν+1} = (1 + w_ν)/|1 + w_ν|."""
        L = self.root_bits()
        one = 1 << L
        c, s = 0, one
        out: list[tuple[int, int]] = [(0, 0), (0, 0), (0, self.from_frac(Fraction(1)))]
        from math import isqrt

        for _nu in range(2, self.n):
            m2 = 2 * one + 2 * c  # |1 + w|² = 2 + 2 cos, with L fractional bits
            m = isqrt(m2 << L)  # L fractional bits
            c = (((one + c) << L) // m) if m else 0
            s = ((s << L) // m) if m else 0
            out.append((self.bits_to_bary(c, L), self.bits_to_bary(s, L)))
        return out

    def bits_to_bary(self, v: int, L: int) -> int:
        return self.wrap(v >> (L - self.b * FRAC_DIGITS))

    def powers(self) -> list[list[tuple[int, int]]]:
        """pw[ν] = [w_ν^κ for κ < 2^(ν−1)], ν = 1 … n, from (6.11)."""
        one = self.from_frac(Fraction(1))
        roots = self.roots()
        pw: list[list[tuple[int, int]]] = [[], [(one, 0)], [(one, 0), roots[2]]]
        for nu in range(2, self.n):
            prev = pw[nu]
            w = roots[nu + 1]
            cur: list[tuple[int, int]] = []
            for k in range(len(prev)):
                cur.append(prev[k])
                cur.append(self.cmul(prev[k], w))
            pw.append(cur)
        return pw

    # -- transforms ---------------------------------------------------------------------------
    def dif(self, a: list[tuple[int, int]], pw: list[list[tuple[int, int]]]) -> None:
        for t in range(self.n - 1, -1, -1):
            span = 1 << t
            tw = pw[t + 1]
            for blk in range(0, 1 << self.n, 2 * span):
                for j in range(span):
                    f, g = a[blk + j], a[blk + j + span]
                    a[blk + j] = (self.add(f[0], g[0]), self.add(f[1], g[1]))
                    a[blk + j + span] = self.cmul((self.sub(f[0], g[0]), self.sub(f[1], g[1])), tw[j])

    def dit_inverse(self, a: list[tuple[int, int]], pw: list[list[tuple[int, int]]]) -> None:
        for t in range(self.n):
            span = 1 << t
            tw = pw[t + 1]
            for blk in range(0, 1 << self.n, 2 * span):
                for j in range(span):
                    f, g = a[blk + j], a[blk + j + span]
                    gw = self.cmul(g, tw[j], conj=True)
                    a[blk + j] = (self.half(self.add(f[0], gw[0])), self.half(self.add(f[1], gw[1])))
                    a[blk + j + span] = (self.half(self.sub(f[0], gw[0])), self.half(self.sub(f[1], gw[1])))

    def pieces(self, x: int) -> list[tuple[int, int]]:
        """(6.8): x = 2^{2n} Σ x_j 2^{nj}, x_j = piece·2^{-2n} = piece·B^{-6}: digits at cells 12, 13, 14."""
        n, B = self.n, self.B
        half = 1 << (n - 1)
        vals = [(x >> (n * j)) & ((1 << n) - 1) for j in range(half)]
        return [(self.wrap(v * B ** (FRAC_DIGITS - 6)), 0) for v in vals] + [(0, 0)] * half

    def multiply(self, x: int, y: int) -> tuple[int, Fraction]:
        """The whole algorithm; returns (product, worst |error| of any z_j in units of 2^-4n)."""
        n, B = self.n, self.B
        a, c = self.pieces(x), self.pieces(y)
        pw = self.powers()
        self.dif(a, pw)
        self.dif(c, pw)
        z = [self.cmul(a[k], c[k]) for k in range(1 << n)]
        self.dit_inverse(z, pw)
        half = 1 << (n - 1)
        xs = [(x >> (n * j)) & ((1 << n) - 1) for j in range(half)]
        ys = [(y >> (n * j)) & ((1 << n) - 1) for j in range(half)]
        worst = Fraction(0)
        prod = 0
        for j in range(1 << n):
            exact = sum(xs[i] * ys[j - i] for i in range(max(0, j - half + 1), min(j, half - 1) + 1))
            worst = max(worst, abs(self.to_frac(z[j][0]) * B**12 - exact), abs(self.to_frac(z[j][1]) * B**12))
            rounded = (z[j][0] + (B // 2) * B**5) % self.M  # add ½·B^-12, keep digits 6 … 14
            prod += ((rounded // B**6) % B**9) << (n * j)
        return prod, worst


# ======================================================================================
# Part 6 — host interface, tests, benchmark, command line
# ======================================================================================


def encode_tape(x: int, y: int, N: int | None = None) -> list[int]:
    """The 2N-bit input of Theorem 6.1: x then y, N bits each, most significant bit first."""
    if x < 0 or y < 0:
        raise ValueError("operands must be non-negative")
    N = N or max(1, max(x, y).bit_length())
    if max(x, y) >= 1 << N:
        raise ValueError("operand does not fit in N bits")
    return [int(c) for c in format(x, f"0{N}b")] + [int(c) for c in format(y, f"0{N}b")]


def decode_output(bits: Sequence[int]) -> int:
    return int("".join(map(str, bits)), 2) if bits else 0


_MACHINE: dict[str, FastSMM] = {}


def machine() -> FastSMM:
    if "m" not in _MACHINE:
        _MACHINE["m"] = FastSMM(build_program())
    return _MACHINE["m"]


def smm_multiply(x: int, y: int, N: int | None = None) -> tuple[int, RunResult]:
    """Multiply on the SMM; returns (product, run statistics)."""
    res = machine().run(encode_tape(x, y, N))
    return decode_output(res.output), res


def param_n(N: int) -> int:
    return FixedModel.choose_n(N)


# -- tests --------------------------------------------------------------------------------------


@dataclass
class Report:
    passed: int = 0
    failed: int = 0

    def check(self, ok: bool, what: str) -> None:
        self.passed += ok
        self.failed += not ok
        print(("ok   " if ok else "FAIL ") + what, flush=True)


def _identical(a: RunResult, b: RunResult) -> bool:
    return (a.output, a.steps, a.nodes) == (b.output, b.steps, b.nodes)


def test_runtime(rep: Report) -> None:
    """§2 semantics: the paper's counterexample, a round trip through the syntax, reference = fast."""
    prog = parse("start: new PP; set QQ to P; halt;")
    res = run_reference(prog, [])
    rep.check(res.node("PP") == res.node("QQ") == res.centre != res.node("P"), "§2 counterexample: p*(PP) = p*(QQ) = a ≠ p*(P)")
    rev = parse(
        "set S to []; read: input z, o; goto done; z: new S; set SW to []; goto read; o: new S; set SW to S; goto read;"
        " done: if S = [] then halt; if SW = S then output 1; if SW ≠ S then output 0; set S to SS; goto done;"
    )
    rep.check(parse(rev.listing()).listing() == rev.listing(), "listing ↔ parse round trip (small program)")
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    a, b = run_reference(rev, bits), FastSMM(rev).run(bits)
    rep.check(a.output == bits[::-1] == b.output and _identical(a, b), "reverse-copy program: reference = compiled")
    full = build_program()
    rep.check(
        parse(full.listing()).listing() == full.listing(), f"listing ↔ parse round trip (multiplier, {len(full)} instructions)"
    )


def test_digit_ops(rep: Report) -> None:
    """(6.3)–(6.5) for every pair of digits, B = 4, 8, 16; reference interpreter = compiled."""
    prog = build_program(stop_after="test_digit_ops")
    for N in (5, 193, 2305):
        tape = [0] * (2 * N)
        res = FastSMM(prog).run(tape)
        ins = Inspector(res, prog)
        B = ins.B()
        bad = 0
        for name, f in (
            ("TA", lambda p, q, B: divmod(p + q, B)[::-1]),
            ("TS", lambda p, q, B: (abs(p - q), int(p < q))),
            ("TM", lambda p, q, B: divmod(p * q, B)[::-1]),
        ):
            nodes = ins.lst(ins.reg(name + "_H"))
            for k, node in enumerate(nodes):
                p, q = divmod(k, B)
                got = (ins.index[res.follow(node, "P")], ins.index[res.follow(node, "Q")])
                bad += got != f(p, q, B)
        rep.check(bad == 0 and len(nodes) == B * B, f"basic digital operations, all {3 * B * B} pairs, B = {B}")
        if N == 5:
            rep.check(_identical(run_reference(prog, tape), res), "digit-op program: reference = compiled (steps, nodes, output)")


def test_scale(rep: Report) -> None:
    """Parameters n, 2^n, b and the pointers (6.1)–(6.2) of the B-scale."""
    prog = build_program(stop_after="bitchains")
    for N, n_exp in ((1, 6), (192, 6), (193, 9), (2304, 9), (2305, 12)):
        res = FastSMM(prog).run([0] * (2 * N))
        ins = Inspector(res, prog)
        B = ins.B()
        n = len(ins.lst(ins.reg("N_H")))
        ok = n == n_exp == param_n(N) and len(ins.lst(ins.reg("C_H"))) == 1 << n and 1 << (n // 3) == B
        ok &= all(res.follow(ins.scale[i], "P") == ins.scale[2 * i] for i in range(B // 2))
        ok &= all(res.follow(ins.scale[i], "Q") == ins.scale[i // 2] for i in range(B))
        rep.check(ok, f"N = {N}: n = {n}, B = {B}, scale pointers")


def test_numeric_programs(rep: Report, rng: random.Random) -> None:
    """The interpreted programs (+, −, ·, /2, complex products) against the host model, digit for digit."""
    prog = build_program(stop_after="test_numeric")
    for N, n in ((100, 6), (200, 9), (2400, 12)):
        fx = FixedModel(n)
        XA = rng.randrange(-fx.M // 2, fx.M // 2)
        XB = rng.randrange(-fx.M // 2, fx.M // 2)
        bits = format(XA % fx.M, f"0{DIGITS * fx.b}b") + format(XB % fx.M, f"0{DIGITS * fx.b}b")
        tape = [int(c) for c in bits.ljust(N, "0")] + [0] * N
        res = FastSMM(prog).run(tape)
        ins = Inspector(res, prog)
        got = {k: ins.number(ins.reg(k)) for k in ("XA", "XB", "SUM", "DIF", "PROD", "HALF")}
        exp = {"XA": XA, "XB": XB, "SUM": fx.add(XA, XB), "DIF": fx.sub(XA, XB), "PROD": fx.mul(XA, XB), "HALF": fx.half(XA)}
        ok = got == exp
        ok &= ins.complex(ins.reg("CM")) == fx.cmul((XB, XA), (XA, XB))
        ok &= ins.complex(ins.reg("CMC")) == fx.cmul((XB, XA), (XA, XB), conj=True)
        rep.check(ok, f"numerical programs +, −, ·, /2, complex products, B = {fx.B}")


def test_roots(rep: Report, quick: bool) -> None:
    prog = build_program(stop_after="powers")
    for N, n in ((1, 6), (193, 9)) + (() if quick else ((2305, 12),)):
        fx = FixedModel(n)
        res = FastSMM(prog).run([0] * (2 * N))
        ins = Inspector(res, prog)
        roots = [ins.complex(res.follow(nd, "W")) for nd in ins.lst(ins.reg("ROOTS_H"))]
        ok = roots == fx.roots()[2:]
        levels = ins.lst(ins.reg("LV_H"))
        pw = fx.powers()
        ok &= len(levels) == n
        for nu, node in enumerate(levels, start=1):
            got = [ins.complex(z) for z in ins.lst(res.follow(node, "W"))]
            ok &= got == pw[nu]
        rep.check(ok, f"roots w_2 … w_{n} and their powers (6.11), n = {n}")


def test_model(rep: Report, rng: random.Random, trials: int) -> None:
    """The host model alone: product exact and worst rounding error far below ½·2^-4n."""
    for n in (6, 9):
        fx = FixedModel(n)
        Nmax = n << (n - 1)
        worst = Fraction(0)
        bad = 0
        cases = [((1 << Nmax) - 1, (1 << Nmax) - 1), ((1 << Nmax) - 1, 1), (0, 0)]
        cases += [(rng.getrandbits(Nmax), rng.getrandbits(Nmax)) for _ in range(trials)]
        for x, y in cases:
            p, e = fx.multiply(x, y)
            bad += p != x * y
            worst = max(worst, e)
        rep.check(
            bad == 0 and worst < Fraction(1, 2),
            f"host model n = {n}: {len(cases)} products exact, worst error {float(worst):.4f}·2^-4n",
        )


def test_multiply(rep: Report, rng: random.Random, cases: Sequence[tuple[int, int, int]]) -> None:
    """The machine against Python integers."""
    for N, x, y in cases:
        t0 = time.time()
        prod, res = smm_multiply(x, y, N)
        rep.check(
            prod == x * y,
            f"N = {N} (n = {param_n(N)}): {res.steps:,} steps, {res.nodes:,} nodes allocated, {time.time() - t0:.0f} s",
        )


def run_tests(quick: bool, seed: int) -> bool:
    rng = random.Random(seed)
    rep = Report()
    test_runtime(rep)
    test_scale(rep)
    test_digit_ops(rep)
    test_numeric_programs(rep, rng)
    test_roots(rep, quick)
    test_model(rep, rng, 3 if quick else 20)
    cases: list[tuple[int, int, int]] = [(8, 200, 123), (1, 1, 1), (64, (1 << 64) - 1, (1 << 64) - 1)]
    if not quick:
        cases += [(192, (1 << 192) - 1, (1 << 192) - 1)]
        cases += [(N, rng.getrandbits(N), rng.getrandbits(N)) for N in (100, 150, 192, 192)]
        cases += [(N, rng.getrandbits(N), rng.getrandbits(N)) for N in (300, 2304)]
    test_multiply(rep, rng, cases)
    print(f"\n{rep.passed} passed, {rep.failed} failed")
    return rep.failed == 0


# -- benchmark --------------------------------------------------------------------------------


def bench(N_values: Sequence[int], out: str | None, seed: int) -> None:
    rng = random.Random(seed)
    rows = []
    for N in N_values:
        x, y = rng.getrandbits(N) | (1 << (N - 1)), rng.getrandbits(N) | (1 << (N - 1))
        t0 = time.time()
        prod, res = smm_multiply(x, y, N)
        row = {
            "N": N,
            "n": param_n(N),
            "B": 1 << (param_n(N) // 3),
            "ok": prod == x * y,
            "steps": res.steps,
            "steps_per_bit": res.steps / N,
            "steps_per_butterfly": res.steps / (param_n(N) * (1 << param_n(N))),
            "nodes_allocated": res.nodes,
            "peak_nodes": res.peak_nodes,
            "collections": res.collections,
            "seconds": round(time.time() - t0, 1),
            "phases": res.phase_steps,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    print("\n   N      n    B        steps   steps/N  steps/(n·2^n)  ok")
    for row in rows:
        print(
            f"{row['N']:6d} {row['n']:6d} {row['B']:4d} {row['steps']:12,d} {row['steps_per_bit']:9.0f} {row['steps_per_butterfly']:14.0f}  {row['ok']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", action="store_true", help="run the verification suite")
    ap.add_argument("--quick", action="store_true", help="with --test: the small cases only")
    ap.add_argument("--bench", nargs="*", type=int, metavar="N", help="multiply random N-bit numbers and log the step counts")
    ap.add_argument("--out", help="with --bench: append one JSON line per run to this file")
    ap.add_argument("--multiply", nargs=2, type=int, metavar=("X", "Y"), help="multiply two integers on the machine")
    ap.add_argument("--listing", action="store_true", help="print the program in the syntax of §2")
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args(argv)
    sys.set_int_max_str_digits(0)
    if args.listing:
        prog = build_program()
        print(f"# {len(prog)} instructions; registers (A Q …): " + ", ".join(f"{k}={v}" for k, v in prog.note.items()))
        print(prog.listing())
        return 0
    if args.multiply:
        x, y = args.multiply
        prod, res = smm_multiply(x, y)
        print(f"{x} × {y} = {prod}  ({'correct' if prod == x * y else 'WRONG'}; {res.steps:,} steps, {res.nodes:,} nodes)")
        return 0 if prod == x * y else 1
    if args.bench is not None:
        bench(args.bench or [8, 64, 192, 256, 1024, 2304], args.out, args.seed)
        return 0
    if args.test:
        return 0 if run_tests(args.quick, args.seed) else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
