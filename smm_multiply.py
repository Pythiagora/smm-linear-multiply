#!/usr/bin/env python3
"""Linear-time integer multiplication on a Schönhage Storage Modification Machine.

A. Schönhage, "Storage Modification Machines", SIAM J. Comput. 9(3):490-508, 1980,
proves that an SMM multiplies two n-bit integers in O(n) steps. This module builds
the machine (alphabet Δ = {0, 1}) and a *fixed* SMM program that does it. The
whole computation runs as pointer manipulation inside the machine; the host only
interprets instructions.

Machine model (Dexter, Doyle & Gurevich, J.UCS 3(4), 1997, §2)
    A Δ-structure is a set of nodes, a centre node, and for every node one pointer
    per letter of Δ. A word W ∈ Δ* names the node p*(W) reached from the centre.
      NEW W        W = Uδ: create y, redirect the δ-pointer of p*(U) to y; every
                   pointer of y points to the node formerly named by W.
      SET W, V     W = Uδ: redirect the δ-pointer of p*(U) to p*(V).
      TEST_EQ U, V, λ   jump to λ iff p*(U) = p*(V).
      GOTO λ
    Memory is mutated by NEW and SET only. Input/output are the model's
    bit-tape instructions INPUT λ0, λ1 (falls through at end of input),
    OUTPUT β, and HALT. Every instruction costs one step (Schönhage's measure);
    words have bounded length, fixed by the program. Total pointer hops are
    also reported.

Why "block trie + diagonal summation" alone is not linear
    Summing the block products a_i·b_j along the diagonals i + j = m costs
    Θ((n/k)²) block products for blocks of k = Θ(log n) bits. A single lookup
    T[x][y] on a bounded-out-degree pointer machine cannot be O(1) in the worst
    case either. The answer would have to lie within c steps of x, of y or of
    the centre. For a table injective in each argument, that reaches at most
    3·2^c·2^k of the 2^{2k} pairs. Schönhage's linear bound (Theorem 6.1)
    needs two ingredients. (1) A fast transform: O(N log N) word operations on
    N = Θ(n / log n) words, i.e. O(n). (2) "Mass production" (§6.2, Lemma 6.2):
    the digit operations of all simultaneous programs are collected per sweep,
    radix-sorted by operand in linear time, and each distinct digit operation
    is executed once; a sweep over m programs costs O(m + b²B²). The diagonal
    sums c_m = Σ_{i+j=m} a_i b_j are still computed, exactly, but through the
    transform.

Relation to Schönhage (1980), §6 (what is the same, what differs)
    Same: a digit universe of B = 2^b pointer nodes with successor and
    doubling pointers (his "B-scale", Fig. 6.1); words of O(1) digits of
    Θ(log n) bits; an FFT-type transform with O(1) sweeps per stage; digit
    operations executed in bulk after a linear-time bucket sort on the
    operands; binary ↔ digit translation in O(b) per digit; final
    translation to binary and summation of the shifted coefficients (6.12).
    Differs:
      - Ring: he uses the complex FFT in fixed point (19 B-ary digits,
        b = n/3, B = O(N^(1/3)), round-off analysis after Schönhage-Strassen).
        This file uses an exact NTT modulo q = h·β² + 1 on 3-digit words, so
        it has no round-off analysis but a modulus search instead.
      - Digit operations: he sorts by both operands and computes each of the
        ≤ 3B² distinct operations once, in O(b²) through binary. This file
        precomputes the full β² table (β² ∈ {4N, 8N}) and, per round, streams
        the rows of the first operands found in the batch.
      - Alphabet: he uses Δ = {P, Q, S, W}; the brief asked for {0, 1}, so
        records are fixed-depth binary trees (one instruction still reaches
        any field).
      - The programs are compiled into rounds, not interpreted with
        per-program instruction pointers (his "interpreting mode").

The program (every step below is SMM code)
    1.  Read both operands (bit lists) and count the bits in a binary counter.
    2.  Choose parameters with unary arithmetic on a pointer "number line":
        digit size s = ⌈L/2⌉ + 1 (β = 2^s, so that β² ∈ {4N, 8N}) and chunk
        size d = max(1, ⌊(L-1)/4⌋) = Θ(log n). The transform length N = 2^L
        is the least one with ⌈|A|/d⌉ + ⌈|B|/d⌉ - 1 ≤ N.
    3.  Build the digit universe (β nodes) with the MSB-first lookup trie and
        the LSB-first bit lists.
    4.  Build the digit-pair table: β² = O(N) entries holding x±y and x·y,
        indexed by a binary trie for sequential (direct) lookups.
    5.  Search the modulus q = h·β² + 1 (h a digit) and an element a with
        a^((q-1)/2) ≡ -1 (mod q). Then ψ = a^h has order β² ≥ N, and
        ω = ψ^(β²/N) is a principal N-th root of unity. Primality of q is
        never needed.
    6.  Split the operands into d-bit chunks: Horner evaluation with the
        digits' doubling pointers.
    7.  Forward NTT (decimation in frequency) of both operands in lock-step,
        pointwise products, inverse NTT (decimation in time), and a final
        scaling by N^-1. All modular arithmetic is on 3-digit words with a
        Montgomery reduction specialised to q ≡ 1 (mod β²). Each vector
        operation is compiled into rounds of batched table lookups.
    8.  Carry propagation: the coefficients c_m (the diagonal sums) are added
        bit-serially into a sliding accumulator, and d bits are emitted per
        coefficient.

Cost. Input, chunking, the element pool and the output are O(n). The
N/2 twiddles are sequential Montgomery products with trie lookups: O(N·s) =
O(n). Digits, tables and the modulus search are o(n): O(β²) = O(N) plus
polylog n. The transform performs O(N log N) = O(n) word operations at O(1)
amortised steps each. `--bench` logs steps per input bit. Its control run,
`--lookup direct` (trie lookups, Θ(log n) steps each), shows the Θ(n log n)
growth that batching removes.

Division of labour. Every bit of every intermediate value lives in the pointer
structure and is produced by SMM instructions. Host code only prepares the
program (compile time, independent of the data), encodes the operands on the
input tape, interprets instructions (copying and comparing node identifiers),
and counts steps. Host integer arithmetic appears in the test oracle (`x * y`,
modular checks, the parameter replica) and in the step counters, never on
the operands inside the machine.

Usage
    python3 smm_multiply.py --test             # verification suite
    python3 smm_multiply.py --bench            # step counts vs n
    python3 smm_multiply.py 12345 67890        # multiply two integers on the SMM
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache, partial
from typing import Literal

# ════════════════════════════════════════════════════════════════════════════
# Part 1 — the Storage Modification Machine
# ════════════════════════════════════════════════════════════════════════════


class Op(IntEnum):
    NEW = 0
    SET = 1
    TEST_EQ = 2
    GOTO = 3
    INPUT = 4
    OUTPUT = 5
    HALT = 6


POINTER_OPS = (Op.NEW, Op.SET, Op.TEST_EQ, Op.GOTO)  # the four pointer-machine operations
MUTATING_OPS = (Op.NEW, Op.SET)  # the only instructions that modify memory


@dataclass(frozen=True, slots=True)
class Instr:
    """One SMM instruction. Words are strings over {'0', '1'}; '' is the centre."""

    op: Op
    w: str = ""  # NEW/SET destination word; TEST_EQ left word
    v: str = ""  # SET source word; TEST_EQ right word
    t0: int = -1  # TEST_EQ/GOTO target; INPUT target on bit 0
    t1: int = -1  # INPUT target on bit 1
    bit: int = 0  # OUTPUT bit

    def __str__(self) -> str:
        w = self.w or "ε"
        v = self.v or "ε"
        match self.op:
            case Op.NEW:
                return f"NEW {w}"
            case Op.SET:
                return f"SET {w} ← {v}"
            case Op.TEST_EQ:
                return f"TEST_EQ {w} = {v} → {self.t0}"
            case Op.GOTO:
                return f"GOTO {self.t0}"
            case Op.INPUT:
                return f"INPUT 0→{self.t0} 1→{self.t1}"
            case Op.OUTPUT:
                return f"OUTPUT {self.bit}"
            case Op.HALT:
                return "HALT"
        raise AssertionError(self.op)


@dataclass(frozen=True)
class Program:
    code: tuple[Instr, ...]
    labels: dict[str, int]
    phases: tuple[tuple[str, int, int], ...]  # (name, first pc, last pc + 1)
    registers: dict[str, str] = field(default_factory=dict)  # name → word (for host inspection)

    def listing(self) -> str:
        at = {pc: name for name, pc in self.labels.items()}
        return "\n".join(f"{pc:6d}  {str(ins):60s}{'  ; ' + at[pc] if pc in at else ''}" for pc, ins in enumerate(self.code))

    def phase_of(self) -> list[str]:
        names = ["?"] * (len(self.code) + 1)
        for name, lo, hi in self.phases:
            for pc in range(lo, hi):
                names[pc] = name
        return names


@dataclass
class RunResult:
    output: list[int]
    steps: int
    by_op: dict[str, int]
    by_phase: dict[str, int]
    hops: int  # total pointer dereferences (sum of word lengths evaluated)
    nodes: int
    halted_at: int  # pc at halt (len(code) if control ran off the end)
    p0: list[int] = field(repr=False, default_factory=list)
    p1: list[int] = field(repr=False, default_factory=list)
    center: int = 0

    def node(self, w: str) -> int:
        """p*(W) in the final structure (host-side inspection only)."""
        x = self.center
        for ch in w:
            x = self.p1[x] if ch == "1" else self.p0[x]
        return x


class StepLimitError(RuntimeError):
    pass


def run_reference(prog: Program, inp: Sequence[int], max_steps: int | None = None) -> RunResult:
    """Literal interpreter of the SMM semantics; the specification for `run_fast`."""
    p0: list[int] = [0]
    p1: list[int] = [0]
    center = 0
    code = prog.code
    counts = [0] * len(code)
    out: list[int] = []
    pos = 0
    pc = 0
    steps = 0

    def walk(word: str) -> int:
        x = center
        for ch in word:
            x = p1[x] if ch == "1" else p0[x]
        return x

    while 0 <= pc < len(code):
        ins = code[pc]
        counts[pc] += 1
        steps += 1
        if max_steps is not None and steps > max_steps:
            raise StepLimitError(f"more than {max_steps} steps")
        op = ins.op
        if op is Op.NEW:
            y = len(p0)
            if ins.w == "":
                p0.append(center)
                p1.append(center)
                center = y
            else:
                x = walk(ins.w[:-1])
                arr = p1 if ins.w[-1] == "1" else p0
                old = arr[x]
                p0.append(old)
                p1.append(old)
                arr[x] = y
            pc += 1
        elif op is Op.SET:
            target = walk(ins.v)
            if ins.w == "":
                center = target
            else:
                x = walk(ins.w[:-1])
                (p1 if ins.w[-1] == "1" else p0)[x] = target
            pc += 1
        elif op is Op.TEST_EQ:
            pc = ins.t0 if walk(ins.w) == walk(ins.v) else pc + 1
        elif op is Op.GOTO:
            pc = ins.t0
        elif op is Op.INPUT:
            if pos >= len(inp):
                pc += 1
            else:
                pc = ins.t1 if inp[pos] else ins.t0
                pos += 1
        elif op is Op.OUTPUT:
            out.append(ins.bit)
            pc += 1
        else:  # HALT
            break
    return _tally(prog, counts, out, p0, p1, center, pc)


def _tally(
    prog: Program, pc_counts: Sequence[int], out: list[int], p0: list[int], p1: list[int], center: int, pc: int
) -> RunResult:
    by_op = {op.name: 0 for op in Op}
    by_phase: dict[str, int] = {}
    phase = prog.phase_of()
    steps = hops = 0
    for i, c in enumerate(pc_counts):
        if c:
            ins = prog.code[i]
            by_op[ins.op.name] += c
            by_phase[phase[i]] = by_phase.get(phase[i], 0) + c
            steps += c
            hops += c * (len(ins.w) + len(ins.v))
    return RunResult(out, steps, by_op, by_phase, hops, len(p0), pc, p0, p1, center)


class FastSMM:
    """Same semantics as `run_reference`, compiled to one Python closure per basic block.

    Node identifiers are indices into two pointer arrays; the generated code only
    indexes, assigns and compares them. Step counts are exact: every instruction
    of an executed block is counted once.
    """

    def __init__(self, prog: Program) -> None:
        self.prog = prog
        code = prog.code
        n = len(code)
        leaders = {0, n}
        for i, ins in enumerate(code):
            if ins.op in (Op.TEST_EQ, Op.GOTO, Op.INPUT, Op.HALT):
                leaders.add(i + 1)
            if ins.op in (Op.TEST_EQ, Op.GOTO):
                leaders.add(ins.t0)
            if ins.op is Op.INPUT:
                leaders.update((ins.t0, ins.t1))
        self.starts = sorted(x for x in leaders if 0 <= x <= n)
        self.block_of = {s: b for b, s in enumerate(self.starts)}
        moves_centre = any(ins.op in MUTATING_OPS and ins.w == "" for ins in code)
        centre = "CEN[0]" if moves_centre else "0"

        def word(w: str) -> str:
            expr = centre
            for ch in w:
                expr = f"P{ch}[{expr}]"
            return expr

        lines = ["def _factory(P0, P1, CEN, OUT, INP, POS, NIN):", "    FNS = []"]
        for b, start in enumerate(self.starts):
            lines.append(f"    def _b{b}(P0=P0, P1=P1, CEN=CEN, OUT=OUT, INP=INP, POS=POS, NIN=NIN):")
            body: list[str] = []
            if start == n:
                body.append("return -1")
            else:
                end = self.starts[b + 1]
                for i in range(start, end):
                    ins = code[i]
                    last = i == end - 1
                    op = ins.op
                    if op is Op.NEW:
                        if ins.w == "":
                            body += ["_y = len(P0)", f"P0.append({centre})", f"P1.append({centre})", "CEN[0] = _y"]
                        else:
                            d = ins.w[-1]
                            body += [
                                f"_x = {word(ins.w[:-1])}",
                                f"_o = P{d}[_x]",
                                "_y = len(P0)",
                                "P0.append(_o)",
                                "P1.append(_o)",
                                f"P{d}[_x] = _y",
                            ]
                    elif op is Op.SET:
                        if ins.w == "":
                            body.append(f"CEN[0] = {word(ins.v)}")
                        else:
                            body.append(f"P{ins.w[-1]}[{word(ins.w[:-1])}] = {word(ins.v)}")
                    elif op is Op.TEST_EQ:
                        body.append(f"if {word(ins.w)} == {word(ins.v)}: return {self.block_of[ins.t0]}")
                    elif op is Op.GOTO:
                        body.append(f"return {self.block_of[ins.t0]}")
                    elif op is Op.INPUT:
                        body += [
                            "_i = POS[0]",
                            f"if _i >= NIN: return {self.block_of[i + 1]}",
                            "POS[0] = _i + 1",
                            f"return {self.block_of[ins.t1]} if INP[_i] else {self.block_of[ins.t0]}",
                        ]
                    elif op is Op.OUTPUT:
                        body.append(f"OUT.append({ins.bit})")
                    else:
                        body.append("return -1")
                    if last and op not in (Op.GOTO, Op.INPUT, Op.HALT):
                        body.append(f"return {self.block_of[end]}")
            lines += ["        " + s for s in body]
            lines.append(f"    FNS.append(_b{b})")
        lines.append("    return FNS")
        namespace: dict[str, object] = {}
        exec(compile("\n".join(lines), f"<smm:{n} instrs>", "exec"), namespace)
        self._factory = namespace["_factory"]

    def run(self, inp: Sequence[int]) -> RunResult:
        p0: list[int] = [0]
        p1: list[int] = [0]
        cen = [0]
        out: list[int] = []
        pos = [0]
        fns = self._factory(p0, p1, cen, out, list(inp), pos, len(inp))  # type: ignore[operator]
        cnt = [0] * len(fns)
        b = 0
        while b >= 0:
            cnt[b] += 1
            b = fns[b]()
        n = len(self.prog.code)
        pc_counts = [0] * n
        halted_at = n
        for blk, c in enumerate(cnt):
            if c:
                start = self.starts[blk]
                if start == n:
                    continue
                end = self.starts[blk + 1]
                for i in range(start, end):
                    pc_counts[i] += c
                if self.prog.code[end - 1].op is Op.HALT:
                    halted_at = end - 1
        return _tally(self.prog, pc_counts, out, p0, p1, cen[0], halted_at)


# ════════════════════════════════════════════════════════════════════════════
# Part 2 — assembler, register file, record layouts
# ════════════════════════════════════════════════════════════════════════════


class Asm:
    """Symbolic assembler: labels, fresh names and phase markers."""

    def __init__(self) -> None:
        self._code: list[tuple[Op, str, str, str | None, str | None, int]] = []
        self._labels: dict[str, int] = {}
        self._uid = 0
        self._phases: list[tuple[str, int, int]] = []
        self._open: tuple[str, int] | None = None

    def fresh(self, stem: str = "L") -> str:
        self._uid += 1
        return f"{stem}.{self._uid}"

    def label(self, name: str) -> None:
        if name in self._labels:
            raise ValueError(f"duplicate label {name}")
        self._labels[name] = len(self._code)

    def phase(self, name: str) -> None:
        if self._open is not None:
            self._phases.append((self._open[0], self._open[1], len(self._code)))
        self._open = (name, len(self._code))

    def _emit(self, op: Op, w: str = "", v: str = "", t0: str | None = None, t1: str | None = None, bit: int = 0) -> None:
        if w.strip("01") or v.strip("01"):
            raise ValueError(f"bad word {w!r} / {v!r}")
        self._code.append((op, w, v, t0, t1, bit))

    def new(self, w: str) -> None:
        self._emit(Op.NEW, w)

    def set(self, w: str, v: str) -> None:
        self._emit(Op.SET, w, v)

    def jeq(self, u: str, v: str, target: str) -> None:
        self._emit(Op.TEST_EQ, u, v, target)

    def goto(self, target: str) -> None:
        self._emit(Op.GOTO, t0=target)

    def input(self, on0: str, on1: str) -> None:
        self._emit(Op.INPUT, t0=on0, t1=on1)

    def output(self, bit: int) -> None:
        self._emit(Op.OUTPUT, bit=bit)

    def halt(self) -> None:
        self._emit(Op.HALT)

    def assemble(self, registers: dict[str, str] | None = None) -> Program:
        self.phase("(end)")
        lab = self._labels
        code = tuple(
            Instr(op, w, v, lab[t0] if t0 is not None else -1, lab[t1] if t1 is not None else -1, bit)
            for op, w, v, t0, t1, bit in self._code
        )
        phases = tuple(p for p in self._phases if p[2] > p[1])
        return Program(code, dict(lab), phases, dict(registers or {}))


class Kind:
    """A record: a complete binary tree of nodes whose leaf pointers are the fields."""

    def __init__(self, name: str, fields: Sequence[str]) -> None:
        self.name = name
        self.depth = max(1, (len(fields) - 1).bit_length())
        self.fields = {f: format(i, f"0{self.depth}b") for i, f in enumerate(fields)}

    def __getattr__(self, f: str) -> str:
        try:
            return self.__dict__["fields"][f]
        except KeyError:
            raise AttributeError(f) from None

    def internal_prefixes(self) -> list[str]:
        return [format(i, f"0{lv}b") for lv in range(1, self.depth) for i in range(1 << lv)]


N_JOBS = 10
N_SLOTS = 50
EL = Kind("EL", ["u", "v", "w", "nx", *(f"j{i}" for i in range(N_JOBS)), *(f"s{i}" for i in range(N_SLOTS))])
JB = Kind("JB", ["x", "y", "r", "link"])
CO = Kind("CO", ["d0", "d1", "d2", "nx"])
DG = Kind("DG", ["nx", "succ", "pred", "mx", "nz", "neg", "dbl0", "dbl1", "msb", "lsb", "row", "rowhead", "bucket", "tmp"])
EN = Kind("EN", ["alo", "acy", "slo", "sbw", "mlo", "mhi", "nx"])
KINDS = (EL, JB, CO, DG, EN)
FIELD_PATH: dict[str, str] = {f"{k.name}.{f}": p for k in KINDS for f, p in k.fields.items()}
FIELD_PATH["BIT.not"] = "0"  # bit nodes B0/B1 point at each other through their 0-pointer

# Raw (field-less) nodes: bit/list cells (.0 item, .1 next), trie nodes (.0/.1 children),
# unary number line (.0 predecessor, .1 successor).


class RegFile:
    """Registers are leaf pointers of a fixed tree hanging from the centre.

    Four hot registers sit at depth 3 (under '0'), the next 32 at depth 7 (under
    '10'), 128 more at depth 9 (under '11'). The tree is built once at start-up and
    never modified afterwards.
    """

    HOT = ("E", "J", "Z", "T")

    def __init__(self) -> None:
        self.paths = {name: "0" + format(i, "02b") for i, name in enumerate(self.HOT)}
        self._next = 0

    def __call__(self, name: str) -> str:
        path = self.paths.get(name)
        if path is None:
            i = self._next
            if i < 32:
                path = "10" + format(i, "05b")
            elif i < 160:
                path = "11" + format(i - 32, "07b")
            else:
                raise RuntimeError("register file exhausted")
            self.paths[name] = path
            self._next += 1
        return path

    @staticmethod
    def tree_nodes() -> list[str]:
        def under(prefix: str, depth: int) -> list[str]:
            return [prefix + (format(i, f"0{lv}b") if lv else "") for lv in range(depth) for i in range(1 << lv)]

        return ["0", "00", "01", "1", *under("10", 5), *under("11", 7)]


# ════════════════════════════════════════════════════════════════════════════
# Part 3 — per-element digit kernels (IR), arithmetic library, host oracle
# ════════════════════════════════════════════════════════════════════════════
# A kernel is a straight-line program run once per element of a vector. Values are
# digit nodes, bit nodes (B0/B1) or table entries. The only binary operation is LK
# (look up the table entry for a digit pair); everything else is unary pointer
# chasing with short branches. The compiler schedules LKs into rounds by dependency
# depth; each round is one batched lookup over the whole vector.


@dataclass(frozen=True, slots=True)
class Val:
    base: tuple[str, object]  # ('in', relpath) | ('const', register) | ('out', (op index, output index))
    fields: tuple[str, ...] = ()

    def f(self, name: str) -> Val:
        if name == "BIT.not" and self.base[0] == "const" and not self.fields and self.base[1] in ("B0", "B1"):
            return B1V if self.base[1] == "B0" else B0V
        return Val(self.base, self.fields + (name,))


B0V = Val(("const", "B0"))
B1V = Val(("const", "B1"))
ZEROV = Val(("const", "ZERO"))
HV = Val(("const", "H"))


@dataclass(frozen=True, slots=True)
class KOp:
    kind: Literal["lk", "addc", "subb", "or", "and", "sel"]
    args: tuple[Val, ...]
    nout: int


class Kernel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: list[KOp] = []
        self.stores: list[tuple[tuple[str, ...], Val]] = []

    def _op(self, kind: Literal["lk", "addc", "subb", "or", "and", "sel"], args: tuple[Val, ...], nout: int) -> tuple[Val, ...]:
        i = len(self.ops)
        self.ops.append(KOp(kind, args, nout))
        return tuple(Val(("out", (i, k))) for k in range(nout))

    def lk(self, x: Val, y: Val) -> Val:
        return self._op("lk", (x, y), 1)[0]

    def addc(self, v: Val, c: Val) -> tuple[Val, Val]:
        """(v + c) mod β and the carry; c is a bit."""
        if c == B0V:
            return v, B0V
        if c == B1V:
            return v.f("DG.succ"), v.f("DG.mx")
        d, k = self._op("addc", (v, c), 2)
        return d, k

    def subb(self, v: Val, b: Val) -> tuple[Val, Val]:
        """(v - b) mod β and the borrow; b is a bit."""
        if b == B0V:
            return v, B0V
        if b == B1V:
            return v.f("DG.pred"), v.f("DG.nz").f("BIT.not")
        d, k = self._op("subb", (v, b), 2)
        return d, k

    def or_(self, x: Val, y: Val) -> Val:
        if B1V in (x, y):
            return B1V
        if x == B0V:
            return y
        if y == B0V or x == y:
            return x
        return self._op("or", (x, y), 1)[0]

    def and_(self, x: Val, y: Val) -> Val:
        if B0V in (x, y):
            return B0V
        if x == B1V:
            return y
        if y == B1V or x == y:
            return x
        return self._op("and", (x, y), 1)[0]

    def sel(self, c: Val, x: Val, y: Val) -> Val:
        """x if c = 1 else y."""
        if x == y or c == B1V:
            return x
        if c == B0V:
            return y
        return self._op("sel", (c, x, y), 1)[0]

    def store(self, dst: tuple[str, ...], v: Val) -> None:
        self.stores.append((dst, v))

    def store_num(self, rel: str, digits: Sequence[Val]) -> None:
        for i, v in enumerate(digits):
            self.store((rel, f"CO.d{i}"), v)

    def optimized(self) -> Kernel:
        """Dead-code elimination; renumbers the surviving operations."""
        live: set[int] = set()
        stack = [v for _, v in self.stores]
        while stack:
            v = stack.pop()
            if v.base[0] == "out":
                i = v.base[1][0]  # type: ignore[index]
                if i not in live:
                    live.add(i)
                    stack.extend(self.ops[i].args)
        order = sorted(live)
        remap = {old: new for new, old in enumerate(order)}

        def rv(v: Val) -> Val:
            if v.base[0] == "out":
                i, k = v.base[1]  # type: ignore[misc]
                return Val(("out", (remap[i], k)), v.fields)
            return v

        out = Kernel(self.name)
        out.ops = [KOp(self.ops[i].kind, tuple(rv(a) for a in self.ops[i].args), self.ops[i].nout) for i in order]
        out.stores = [(d, rv(v)) for d, v in self.stores]
        return out

    def levels(self) -> list[int]:
        """Round index of every op: an LK is one round after its latest input."""
        lv: list[int] = []
        for op in self.ops:
            m = max((lv[a.base[1][0]] for a in op.args if a.base[0] == "out"), default=0)  # type: ignore[index]
            lv.append(m + 1 if op.kind == "lk" else m)
        return lv


def num_in(rel: str) -> list[Val]:
    return [Val(("in", (rel, f"CO.d{i}"))) for i in range(3)]


def num_const(reg: str) -> list[Val]:
    return [Val(("const", reg), (f"CO.d{i}",)) for i in range(3)]


# Residues mod q = h·β² + 1 (1 ≤ h < β/2) are 3-digit words (x0, x1, x2), 0 ≤ x < q.


def cond_sub_q(k: Kernel, x: Sequence[Val]) -> list[Val]:
    """x - q if x ≥ q else x, for 0 ≤ x < 2q.  q = (1, 0, h)."""
    d0, b = k.subb(x[0], B1V)
    d1, b1 = k.subb(x[1], b)
    e = k.lk(HV, x[2])  # slo = (h - x2) mod β, sbw = [h < x2]
    below = k.and_(e.f("EN.sbw").f("BIT.not"), e.f("EN.slo").f("DG.nz"))  # [x2 < h]
    d2, b2 = k.subb(e.f("EN.slo").f("DG.neg"), b1)  # (x2 - h - b1) mod β
    under = k.or_(below, b2)
    return [k.sel(under, xi, di) for xi, di in zip(x, (d0, d1, d2), strict=True)]


def cond_add_q(k: Kernel, x: Sequence[Val], flag: Val) -> list[Val]:
    """(x + q) mod β³ if flag else x."""
    e0, c = k.addc(x[0], B1V)
    e1, c1 = k.addc(x[1], c)
    e = k.lk(HV, x[2])
    e2, _ = k.addc(e.f("EN.alo"), c1)
    return [k.sel(flag, ei, xi) for ei, xi in zip((e0, e1, e2), x, strict=True)]


def addsub_mod(k: Kernel, x: Sequence[Val], y: Sequence[Val]) -> tuple[list[Val], list[Val]]:
    """((x + y) mod q, (x - y) mod q), sharing one lookup per digit pair."""
    e = [k.lk(x[i], y[i]) for i in range(3)]
    s0, c = e[0].f("EN.alo"), e[0].f("EN.acy")
    s1, c1 = k.addc(e[1].f("EN.alo"), c)
    c = k.or_(e[1].f("EN.acy"), c1)
    s2, _ = k.addc(e[2].f("EN.alo"), c)
    d0, b = e[0].f("EN.slo"), e[0].f("EN.sbw")
    d1, b1 = k.subb(e[1].f("EN.slo"), b)
    b = k.or_(e[1].f("EN.sbw"), b1)
    d2, b2 = k.subb(e[2].f("EN.slo"), b)
    negative = k.or_(e[2].f("EN.sbw"), b2)
    return cond_sub_q(k, (s0, s1, s2)), cond_add_q(k, (d0, d1, d2), negative)


def sum_columns(k: Kernel, cols: Sequence[Sequence[Val]], carries_in: Sequence[Val] = ()) -> list[Val]:
    """Digits of Σ_c (Σ cols[c]) β^c + Σ carries_in; the carry out of the top column is dropped."""
    carry: list[list[Val]] = [list(carries_in)] + [[] for _ in cols]
    out: list[Val] = []
    for c, col in enumerate(cols):
        terms = list(col)
        while len(terms) > 1:  # pairwise lookup adds; every pair yields a carry bit
            nxt = []
            for i in range(0, len(terms) - 1, 2):
                e = k.lk(terms[i], terms[i + 1])
                nxt.append(e.f("EN.alo"))
                carry[c + 1].append(e.f("EN.acy"))
            if len(terms) % 2:
                nxt.append(terms[-1])
            terms = nxt
        digit = terms[0] if terms else ZEROV
        bits = carry[c]
        if len(bits) >= 8:
            raise AssertionError("too many carry bits for β ≥ 8")
        wrap: Val | None = None
        for cb in bits:  # ≤ 7 unit increments of a digit < β wrap at most once (β ≥ 8)
            digit, co = k.addc(digit, cb)
            wrap = co if wrap is None else k.or_(wrap, co)
        if wrap is not None:
            carry[c + 1].append(wrap)
        out.append(digit)
    return out


def montmul(k: Kernel, a: Sequence[Val], b: Sequence[Val]) -> list[Val]:
    """a·b·β⁻³ mod q (Montgomery, R = β³), for 0 ≤ a, b < q.

    With q ≡ 1 (mod β²), q' = -q⁻¹ mod β³ = h·β² - 1, so m = T·q' mod β³
    = u·β² - T mod β³ with u = h·t0 mod β, and (T + m·q)/β³ = T_hi + ε + (u + m·h)/β.
    """
    cols: list[list[Val]] = [[] for _ in range(6)]
    for i in range(3):
        for j in range(3):
            e = k.lk(a[i], b[j])
            cols[i + j].append(e.f("EN.mlo"))
            cols[i + j + 1].append(e.f("EN.mhi"))
    t = sum_columns(k, cols)  # T = a·b, six digits
    u = k.lk(HV, t[0]).f("EN.mlo")
    m0, b1 = t[0].f("DG.neg"), t[0].f("DG.nz")  # m = u·β² - T_lo mod β³, ε = final borrow
    n1, k1 = t[1].f("DG.neg"), t[1].f("DG.nz")
    m1, k2 = k.subb(n1, b1)
    b2 = k.or_(k1, k2)
    e2 = k.lk(u, t[2])
    m2, k4 = k.subb(e2.f("EN.slo"), b2)
    eps = k.or_(e2.f("EN.sbw"), k4)
    qh = [k.lk(HV, m) for m in (m0, m1, m2)]  # m·h
    v = sum_columns(  # (u + m·h)/β; u + lo(m0·h) is 0 or β
        k,
        [[qh[0].f("EN.mhi"), qh[1].f("EN.mlo")], [qh[1].f("EN.mhi"), qh[2].f("EN.mlo")], [qh[2].f("EN.mhi")]],
        carries_in=(u.f("DG.nz"),),
    )
    r = sum_columns(k, [[t[3], v[0]], [t[4], v[1]], [t[5], v[2]]], carries_in=(eps,))  # < 1.5q
    return cond_sub_q(k, r)


@cache
def kernels() -> dict[str, Kernel]:
    ks: dict[str, Kernel] = {}
    u, v, w = num_in("EL.u"), num_in("EL.v"), num_in("EL.w")

    k = Kernel("dif")  # (u, v) ← (u + v, (u - v)·ω^j)
    s, d = addsub_mod(k, u, v)
    k.store_num("EL.u", s)
    k.store_num("EL.v", montmul(k, d, w))
    ks["dif"] = k

    k = Kernel("dit")  # (u, v) ← (u + v·ω^j, u - v·ω^j)
    s, d = addsub_mod(k, u, montmul(k, v, w))
    k.store_num("EL.u", s)
    k.store_num("EL.v", d)
    ks["dit"] = k

    k = Kernel("pointwise")  # u ← u·v·R⁻¹
    k.store_num("EL.u", montmul(k, u, v))
    ks["pointwise"] = k

    k = Kernel("scale")  # u ← u·KR·R⁻¹, KR = N⁻¹·R² mod q
    k.store_num("EL.u", montmul(k, u, num_const("KR")))
    ks["scale"] = k

    k = Kernel("add")  # w ← u + v mod q
    k.store_num("EL.w", addsub_mod(k, u, v)[0])
    ks["add"] = k

    k = Kernel("sub")  # w ← u - v mod q
    k.store_num("EL.w", addsub_mod(k, u, v)[1])
    ks["sub"] = k

    k = Kernel("mul")  # w ← u·v·R⁻¹ mod q
    k.store_num("EL.w", montmul(k, u, v))
    ks["mul"] = k
    return {name: kk.optimized() for name, kk in ks.items()}


def eval_kernel_host(
    k: Kernel, beta: int, consts: dict[str, object], inputs: dict[tuple[str, ...], object]
) -> dict[tuple[str, ...], int]:
    """Test oracle: evaluates a kernel with host integers (never used by the SMM)."""

    def field_fn(name: str, x: object) -> object:
        if name.startswith("CO.d"):
            return x[int(name[-1])]  # type: ignore[index]
        if name == "BIT.not":
            return 1 - x  # type: ignore[operator]
        if name.startswith("EN."):
            _, a, b = x  # type: ignore[misc]
            return {
                "EN.alo": (a + b) % beta,
                "EN.acy": int(a + b >= beta),
                "EN.slo": (a - b) % beta,
                "EN.sbw": int(a < b),
                "EN.mlo": (a * b) % beta,
                "EN.mhi": (a * b) // beta,
            }[name]
        z: int = x  # type: ignore[assignment]
        return {
            "DG.succ": (z + 1) % beta,
            "DG.pred": (z - 1) % beta,
            "DG.mx": int(z == beta - 1),
            "DG.nz": int(z != 0),
            "DG.neg": (-z) % beta,
        }[name]

    vals: dict[tuple[int, int], object] = {}

    def get(v: Val) -> object:
        kind, ref = v.base
        x = inputs[ref] if kind == "in" else consts[ref] if kind == "const" else vals[ref]  # type: ignore[index]
        for f in v.fields:
            x = field_fn(f, x)
        return x

    for i, op in enumerate(k.ops):
        a = [get(x) for x in op.args]
        if op.kind == "lk":
            vals[(i, 0)] = ("E", a[0], a[1])
        elif op.kind == "addc":
            s = a[0] + a[1]  # type: ignore[operator]
            vals[(i, 0)], vals[(i, 1)] = s % beta, int(s >= beta)
        elif op.kind == "subb":
            s = a[0] - a[1]  # type: ignore[operator]
            vals[(i, 0)], vals[(i, 1)] = s % beta, int(s < 0)
        elif op.kind == "or":
            vals[(i, 0)] = a[0] | a[1]  # type: ignore[operator]
        elif op.kind == "and":
            vals[(i, 0)] = a[0] & a[1]  # type: ignore[operator]
        else:
            vals[(i, 0)] = a[1] if a[0] else a[2]
    return {dst: get(v) for dst, v in k.stores}  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════
# Part 4 — the multiplication program
# ════════════════════════════════════════════════════════════════════════════

Lookup = Literal["batched", "direct"]


class Builder:
    WARM = ("NIL", "B0", "B1", "JH", "bl.jn", "bl.x", "DIG0", "DEND", "dl.t", "dl.b", "SE", "H", "U0", "VEND", "POOL")

    def __init__(self, lookup: Lookup = "batched") -> None:
        self.a = Asm()
        self.R = RegFile()
        self.lookup = lookup
        self.ks = kernels()
        self._depth = 0
        for name in self.WARM:  # the depth-7 tier: registers of the inner loops
            self.R(name)

    # ── small macros ────────────────────────────────────────────────────────
    def alloc(self, w: str, kind: Kind) -> None:
        self.a.new(w)
        for pre in kind.internal_prefixes():
            self.a.new(w + pre)

    def append_cell(self, head: str, tail: str, item: str) -> None:
        """Append a raw cell (.0 = item, .1 = NIL) to the list (head, tail)."""
        a, R = self.a, self.R
        empty, done = a.fresh("app"), a.fresh("app")
        a.jeq(head, R("NIL"), empty)
        a.new(tail + "1")  # the new cell's pointers start at the old tail.1 = NIL
        a.set(tail, tail + "1")
        a.goto(done)
        a.label(empty)
        a.new(head)
        a.set(tail, head)
        a.label(done)
        a.set(tail + "0", item)

    def counter_inc(self, head: str, tail: str) -> None:
        """Increment a binary counter stored as an LSB-first bit list (amortised O(1))."""
        a, R = self.a, self.R
        c = R("cnt.p")
        loop, one, app, done = (a.fresh("inc") for _ in range(4))
        a.set(c, head)
        a.label(loop)
        a.jeq(c, R("NIL"), app)
        a.jeq(c + "0", R("B0"), one)
        a.set(c + "0", R("B0"))
        a.set(c, c + "1")
        a.goto(loop)
        a.label(one)
        a.set(c + "0", R("B1"))
        a.goto(done)
        a.label(app)
        self.append_cell(head, tail, R("B1"))
        a.label(done)

    def list_length(self, dst: str, head: str) -> None:
        """dst ← length of a raw list, as a point of the unary number line."""
        a, R = self.a, self.R
        p = R("len.p")
        loop, done = a.fresh("len"), a.fresh("len")
        a.set(dst, R("U0"))
        a.set(p, head)
        a.label(loop)
        a.jeq(p, R("NIL"), done)
        a.set(dst, dst + "1")
        a.set(p, p + "1")
        a.goto(loop)
        a.label(done)

    def repeat(self, count: str, body: Callable[[], None]) -> None:
        """Run body `count` times (count is a unary number; it is copied first)."""
        a, R = self.a, self.R
        t = R(f"rep{self._depth}")
        self._depth += 1
        loop, done = a.fresh("rep"), a.fresh("rep")
        a.set(t, count)
        a.label(loop)
        a.jeq(t, R("U0"), done)
        body()
        a.set(t, t + "0")
        a.goto(loop)
        a.label(done)
        self._depth -= 1

    # unary numbers: nodes of the number line; .0 = predecessor (saturating), .1 = successor
    def u_add(self, dst: str, x: str, y: str, sub: bool = False) -> None:
        a, R = self.a, self.R
        t = R("u.t")
        loop, done = a.fresh("uadd"), a.fresh("uadd")
        a.set(t, y)
        a.set(dst, x)
        a.label(loop)
        a.jeq(t, R("U0"), done)
        a.set(dst, dst + ("0" if sub else "1"))
        a.set(t, t + "0")
        a.goto(loop)
        a.label(done)

    def u_le(self, x: str, y: str, if_true: str, if_false: str) -> None:
        a, R = self.a, self.R
        t1, t2 = R("u.l1"), R("u.l2")
        loop = a.fresh("ule")
        a.set(t1, x)
        a.set(t2, y)
        a.label(loop)
        a.jeq(t1, R("U0"), if_true)
        a.jeq(t2, R("U0"), if_false)
        a.set(t1, t1 + "0")
        a.set(t2, t2 + "0")
        a.goto(loop)

    def u_half(self, dst: str, x: str, ceil: bool) -> None:
        a, R = self.a, self.R
        t = R("u.h")
        loop, done = a.fresh("uhalf"), a.fresh("uhalf")
        a.set(t, x)
        a.set(dst, R("U0"))
        a.label(loop)
        a.jeq(t, R("U0"), done)
        a.set(t, t + "0")
        if ceil:
            a.set(dst, dst + "1")
        a.jeq(t, R("U0"), done)
        a.set(t, t + "0")
        if not ceil:
            a.set(dst, dst + "1")
        a.goto(loop)
        a.label(done)

    def param_s(self, dst: str, L: str) -> None:
        """s(L) = ⌈L/2⌉ + 1."""
        self.u_half(dst, L, ceil=True)
        self.a.set(dst, dst + "1")

    def param_d(self, dst: str, L: str) -> None:
        """d(L) = max(1, ⌊(L - 1)/4⌋)."""
        a, R = self.a, self.R
        tmp = R("pd.t")
        ok = a.fresh("pd")
        a.set(tmp, L + "0")
        self.u_half(dst, tmp, ceil=False)
        a.set(tmp, dst)
        self.u_half(dst, tmp, ceil=False)
        a.jeq(dst, R("U0"), ok + "z")
        a.goto(ok)
        a.label(ok + "z")
        a.set(dst, dst + "1")
        a.label(ok)

    def u_log2(self, dst: str, x: str) -> None:
        """⌊log2 x⌋ for x ≥ 1."""
        a, R = self.a, self.R
        cur, nxt = R("lg.c"), R("lg.n")
        loop, done = a.fresh("lg"), a.fresh("lg")
        a.set(dst, R("U0"))
        a.set(cur, x)
        a.label(loop)
        self.u_half(nxt, cur, ceil=False)
        a.jeq(nxt, R("U0"), done)
        a.set(dst, dst + "1")
        a.set(cur, nxt)
        a.goto(loop)
        a.label(done)

    def copy_num(self, dst: str, src: str) -> None:
        for i in range(3):
            self.a.set(dst + CO.fields[f"d{i}"], src + CO.fields[f"d{i}"])

    # ── kernel compilation ─────────────────────────────────────────────────
    def _vpath(self, v: Val, base: str, slot: dict[tuple[int, int], int]) -> str:
        kind, ref = v.base
        if kind == "in":
            p = base + "".join(FIELD_PATH[f] for f in ref)  # type: ignore[union-attr]
        elif kind == "const":
            p = self.R(ref)  # type: ignore[arg-type]
        else:
            p = base + EL.fields[f"s{slot[ref]}"]  # type: ignore[index]
        return p + "".join(FIELD_PATH[f] for f in v.fields)

    def _emit_unary(self, op: KOp, outs: list[str], vp: Callable[[Val], str]) -> None:
        a, R = self.a, self.R
        B0, B1 = R("B0"), R("B1")
        yes, done = a.fresh("k"), a.fresh("k")
        if op.kind == "addc":
            v, c = (vp(x) for x in op.args)
            a.jeq(c, B1, yes)
            a.set(outs[0], v)
            a.set(outs[1], B0)
            a.goto(done)
            a.label(yes)
            a.set(outs[0], v + DG.succ)
            a.set(outs[1], v + DG.mx)
        elif op.kind == "subb":
            v, b = (vp(x) for x in op.args)
            a.jeq(b, B1, yes)
            a.set(outs[0], v)
            a.set(outs[1], B0)
            a.goto(done)
            a.label(yes)
            a.set(outs[0], v + DG.pred)
            a.set(outs[1], v + DG.nz + FIELD_PATH["BIT.not"])
        elif op.kind in ("or", "and"):
            x, y = (vp(z) for z in op.args)
            dominant = B1 if op.kind == "or" else B0
            a.jeq(x, dominant, yes)
            a.set(outs[0], y)
            a.goto(done)
            a.label(yes)
            a.set(outs[0], dominant)
        elif op.kind == "sel":
            c, x, y = (vp(z) for z in op.args)
            a.jeq(c, B1, yes)
            a.set(outs[0], y)
            a.goto(done)
            a.label(yes)
            a.set(outs[0], x)
        else:
            raise AssertionError(op.kind)
        a.label(done)

    @staticmethod
    def _allocate(events: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]) -> dict[tuple[int, int], int]:
        """Linear-scan slot allocation; outputs never share a slot with inputs of the same event."""
        last: dict[tuple[int, int], int] = {}
        for i, (_, uses) in enumerate(events):
            for u in uses:
                last[u] = i
        free = list(range(N_SLOTS - 1, -1, -1))
        slot: dict[tuple[int, int], int] = {}
        for i, (defs, uses) in enumerate(events):
            for d in defs:
                if not free:
                    raise RuntimeError("kernel needs more element slots")
                slot[d] = free.pop()
            for d in defs:
                if last.get(d, -1) <= i:
                    free.append(slot[d])
            for u in dict.fromkeys(uses):
                if last[u] == i:
                    free.append(slot[u])
        return slot

    @staticmethod
    def _uses(vals: Sequence[Val]) -> list[tuple[int, int]]:
        return [v.base[1] for v in vals if v.base[0] == "out"]  # type: ignore[misc]

    def emit_kernel_batched(self, k: Kernel, vhead: str, vend: str) -> None:
        a, R = self.a, self.R
        E, JH = R("E"), R("JH")
        lv = k.levels()
        rounds = max((lv[i] for i, op in enumerate(k.ops) if op.kind == "lk"), default=0)
        lks = {r: [i for i, op in enumerate(k.ops) if op.kind == "lk" and lv[i] == r] for r in range(1, rounds + 1)}
        job = {i: j for r in lks for j, i in enumerate(lks[r])}
        if max((len(x) for x in lks.values()), default=0) > N_JOBS:
            raise RuntimeError("kernel needs more job records per element")
        plan: list[list[tuple[str, object]]] = []
        for r in range(rounds + 1):
            ev: list[tuple[str, object]] = [("copy", i) for i in lks.get(r, [])]
            ev += [("unary", i) for i, op in enumerate(k.ops) if op.kind != "lk" and lv[i] == r]
            ev += [("setup", i) for i in lks.get(r + 1, [])]
            if r == rounds:
                ev += [("store", s) for s in k.stores]
            plan.append(ev)
        flat: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        for ev in plan:
            for kind, x in ev:
                if kind == "copy":
                    flat.append(([(x, 0)], []))  # type: ignore[list-item]
                elif kind == "unary":
                    op = k.ops[x]  # type: ignore[index]
                    flat.append(([(x, j) for j in range(op.nout)], self._uses(op.args)))  # type: ignore[list-item]
                elif kind == "setup":
                    flat.append(([], self._uses(k.ops[x].args)))  # type: ignore[index]
                else:
                    flat.append(([], self._uses([x[1]])))  # type: ignore[index]
        slot = self._allocate(flat)

        def vp(v: Val) -> str:
            return self._vpath(v, E, slot)

        for r, ev in enumerate(plan):
            top, end = a.fresh("round"), a.fresh("round")
            a.set(E, vhead)
            if r < rounds:
                a.set(JH, R("NIL"))
            a.label(top)
            a.jeq(E, vend, end)
            for kind, x in ev:
                if kind == "copy":
                    a.set(E + EL.fields[f"s{slot[(x, 0)]}"], E + EL.fields[f"j{job[x]}"] + JB.r)  # type: ignore[index]
                elif kind == "unary":
                    op = k.ops[x]  # type: ignore[index]
                    outs = [E + EL.fields[f"s{slot[(x, j)]}"] for j in range(op.nout)]  # type: ignore[index]
                    self._emit_unary(op, outs, vp)
                elif kind == "setup":
                    jp = E + EL.fields[f"j{job[x]}"]  # type: ignore[index]
                    xa, ya = k.ops[x].args  # type: ignore[index]
                    a.set(jp + JB.x, vp(xa))
                    a.set(jp + JB.y, vp(ya))
                    a.set(jp + JB.link, JH)
                    a.set(JH, jp)
                else:
                    dst, v = x  # type: ignore[misc]
                    a.set(E + "".join(FIELD_PATH[f] for f in dst), vp(v))
            a.set(E, E + EL.nx)
            a.goto(top)
            a.label(end)
            if r < rounds:
                self.emit_batched_lookup()

    def emit_batched_lookup(self) -> None:
        """For every job j on list JH: j.r ← T[j.x][j.y].

        Jobs are bucketed by x (O(1) each); then for every digit x whose bucket is
        non-empty, row x is streamed into the digits' `tmp` pointers (β steps) and
        each job reads j.y.tmp. Cost O(#jobs + β · #distinct x) = O(#jobs + β²).
        """
        a, R = self.a, self.R
        NIL, J, JN, X, Z, T = R("NIL"), R("J"), R("bl.jn"), R("bl.x"), R("Z"), R("T")
        p1, p2, rows, nextx, inst, ans, ansl, clr, done = (a.fresh("bl") for _ in range(9))
        a.set(J, R("JH"))
        a.label(p1)
        a.jeq(J, NIL, p2)
        a.set(JN, J + JB.link)
        a.set(J + JB.link, J + JB.x + DG.bucket)
        a.set(J + JB.x + DG.bucket, J)
        a.set(J, JN)
        a.goto(p1)
        a.label(p2)
        a.set(X, R("DIG0"))
        a.label(rows)
        a.jeq(X, R("DEND"), done)
        a.jeq(X + DG.bucket, NIL, nextx)
        a.set(T, X + DG.rowhead)
        a.set(Z, R("DIG0"))
        a.label(inst)
        a.jeq(Z, R("DEND"), ans)
        for _ in range(4):  # β is a multiple of 4
            a.set(Z + DG.tmp, T)
            a.set(T, T + EN.nx)
            a.set(Z, Z + DG.nx)
        a.goto(inst)
        a.label(ans)
        a.set(J, X + DG.bucket)
        a.label(ansl)
        a.jeq(J, NIL, clr)
        a.set(J + JB.r, J + JB.y + DG.tmp)
        a.set(J, J + JB.link)
        a.goto(ansl)
        a.label(clr)
        a.set(X + DG.bucket, NIL)
        a.label(nextx)
        a.set(X, X + DG.nx)
        a.goto(rows)
        a.label(done)

    def emit_kernel_direct(self, k: Kernel, base: str) -> None:
        """Straight-line kernel on the element at `base`; each LK walks the lookup trie."""
        a = self.a
        events: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        for i, op in enumerate(k.ops):
            events.append(([(i, j) for j in range(op.nout)], self._uses(op.args)))
        for _, v in k.stores:
            events.append(([], self._uses([v])))
        slot = self._allocate(events)

        def vp(v: Val) -> str:
            return self._vpath(v, base, slot)

        for i, op in enumerate(k.ops):
            outs = [base + EL.fields[f"s{slot[(i, j)]}"] for j in range(op.nout)]
            if op.kind == "lk":
                self.emit_direct_lookup(vp(op.args[0]), vp(op.args[1]), outs[0])
            else:
                self._emit_unary(op, outs, vp)
        for dst, v in k.stores:
            a.set(base + "".join(FIELD_PATH[f] for f in dst), vp(v))

    def emit_direct_lookup(self, x: str, y: str, dst: str) -> None:
        """dst ← T[x][y]: from row root x.row follow the s bits of y (MSB first)."""
        a, R = self.a, self.R
        DT, DB = R("dl.t"), R("dl.b")
        loop, one, done = a.fresh("dl"), a.fresh("dl"), a.fresh("dl")
        a.set(DT, x + DG.row)
        a.set(DB, y + DG.msb)
        a.label(loop)
        a.jeq(DB, R("NIL"), done)
        a.jeq(DB + "0", R("B1"), one)
        a.set(DT, DT + "0")
        a.set(DB, DB + "1")
        a.goto(loop)
        a.label(one)
        a.set(DT, DT + "1")
        a.set(DB, DB + "1")
        a.goto(loop)
        a.label(done)
        a.set(dst, DT)

    def run_vector_kernel(self, name: str, vhead: str, vend: str) -> None:
        k = self.ks[name]
        if self.lookup == "batched":
            self.emit_kernel_batched(k, vhead, vend)
            return
        a, R = self.a, self.R
        E = R("E")
        loop, done = a.fresh("vd"), a.fresh("vd")
        a.set(E, vhead)
        a.label(loop)
        a.jeq(E, vend, done)
        self.emit_kernel_direct(k, E)
        a.set(E, E + EL.nx)
        a.goto(loop)
        a.label(done)

    def scalar(self, name: str, u: str, v: str, w: str) -> None:
        """w ← kernel(u, v) on the set-up element SE (sequential, direct lookups)."""
        a, R = self.a, self.R
        SE = R("SE")
        a.set(SE + EL.u, u)
        a.set(SE + EL.v, v)
        a.set(SE + EL.w, w)
        self.emit_kernel_direct(self.ks[name], SE)

    # ── the program, phase by phase ────────────────────────────────────────
    def build(self) -> Program:
        self.phase_init()
        self.phase_input()
        self.phase_parameters()
        self.phase_digits()
        self.phase_tables()
        self.phase_lists()
        self.phase_modulus()
        self.phase_twiddles()
        self.phase_transform()
        self.phase_output()
        return self.a.assemble(self.R.paths)

    def phase_init(self) -> None:
        a, R = self.a, self.R
        a.phase("init")
        for w in RegFile.tree_nodes():
            a.new(w)
        NIL, B0, B1 = R("NIL"), R("B0"), R("B1")
        a.new(NIL)
        a.set(NIL + "0", NIL)
        a.set(NIL + "1", NIL)
        for b in (B0, B1):
            a.new(b)
            a.set(b + "1", b)
        a.set(B0 + "0", B1)
        a.set(B1 + "0", B0)

    def phase_input(self) -> None:
        """Operands arrive LSB-first, each bit as '1b', each operand closed by '0'."""
        a, R = self.a, self.R
        a.phase("input")
        NIL = R("NIL")
        for reg in ("A.h", "A.t", "B.h", "B.t", "C.h", "C.t"):
            a.set(R(reg), NIL)
        for opnd in ("A", "B"):
            head, tail = R(f"{opnd}.h"), R(f"{opnd}.t")
            top, bit, zero, one, end = (a.fresh("in" + opnd) for _ in range(5))
            a.label(top)
            a.input(end, bit)
            a.goto(end)  # end of tape
            a.label(bit)
            a.input(zero, one)
            a.goto(end)
            for lab, b in ((zero, "B0"), (one, "B1")):
                a.label(lab)
                self.append_cell(head, tail, R(b))
                self.counter_inc(R("C.h"), R("C.t"))
                a.goto(top)
            a.label(end)

    def phase_parameters(self) -> None:
        a, R = self.a, self.R
        a.phase("parameters")
        NIL, U0 = R("NIL"), R("U0")
        # number line: length 4·λ + 40, λ = bit length of n = |A| + |B|
        a.new(U0)
        a.set(U0 + "0", U0)
        a.set(U0 + "1", NIL)
        ue = R("u.end")
        a.set(ue, U0)

        def extend() -> None:
            a.new(ue + "1")
            a.set(ue + "10", ue)
            a.set(ue, ue + "1")

        for _ in range(4):
            p = R("np.p")
            loop, done = a.fresh("line"), a.fresh("line")
            a.set(p, R("C.h"))
            a.label(loop)
            a.jeq(p, NIL, done)
            extend()
            a.set(p, p + "1")
            a.goto(loop)
            a.label(done)
        for _ in range(40):
            extend()
        LAM, L, D, FL, X, Y = R("lam"), R("L"), R("D"), R("fl"), R("px"), R("py")
        self.list_length(LAM, R("C.h"))
        # L_up: least L ≥ 3 with λ + 1 ≤ L + ⌊log2 d(L)⌋ (a sufficient condition)
        a.set(L, U0 + "111")
        up, found = a.fresh("Lup"), a.fresh("Lup")
        a.label(up)
        self.param_d(D, L)
        self.u_log2(FL, D)
        self.u_add(X, L, FL)
        a.set(Y, LAM + "1")
        nope = a.fresh("Lup")
        self.u_le(Y, X, found, nope)
        a.label(nope)
        a.set(L, L + "1")
        a.goto(up)
        a.label(found)
        # refine downwards with the exact test  ⌈|A|/d⌉ + ⌈|B|/d⌉ - 1 ≤ 2^L
        refine, stop, take = a.fresh("ref"), a.fresh("ref"), a.fresh("ref")
        L2, ELL = R("L2"), R("ell")
        a.label(refine)
        a.jeq(L, U0 + "111", stop)
        a.set(L2, L + "0")
        self.param_d(D, L2)
        self.chunk_count_bits(ELL, D)
        self.u_le(ELL, L2, take, stop)
        a.label(take)
        a.set(L, L2)
        a.goto(refine)
        a.label(stop)
        S, CP, E2, S3, K6, LM1, T = R("S"), R("CP"), R("E2"), R("S3"), R("K6"), R("LM1"), R("pt")
        self.param_s(S, L)
        self.param_d(D, L)
        self.u_add(T, S, S)
        self.u_add(CP, T, L, sub=True)  # c' = 2s - L ∈ {2, 3}
        self.u_add(T, D, D)
        self.u_add(E2, T, CP, sub=True)  # e = max(0, 2d - c'): h ≥ 2^e
        self.u_add(T, S, S)
        self.u_add(S3, T, S)  # 3s
        self.u_add(T, S3, S3)
        self.u_add(K6, T, L, sub=True)  # 6s - L
        a.set(LM1, L + "0")

    def chunk_count_bits(self, dst: str, d: str) -> None:
        """dst ← bit length of (⌈|A|/d⌉ + ⌈|B|/d⌉ - 2), clamped at 0."""
        a, R = self.a, self.R
        NIL, U0 = R("NIL"), R("U0")
        H, T, SKIP, P, CNT = R("cc.h"), R("cc.t"), R("cc.skip"), R("cc.p"), R("cc.c")
        a.set(H, NIL)
        a.set(T, NIL)
        a.set(SKIP, U0 + "11")
        for opnd in ("A", "B"):
            chunk, inner, count, done = (a.fresh("cc") for _ in range(4))
            a.set(P, R(f"{opnd}.h"))
            a.label(chunk)
            a.jeq(P, NIL, done)
            a.jeq(SKIP, U0, count)
            a.set(SKIP, SKIP + "0")
            skipped = a.fresh("cc")
            a.goto(skipped)
            a.label(count)
            self.counter_inc(H, T)
            a.label(skipped)
            a.set(CNT, d)
            a.label(inner)
            a.jeq(CNT, U0, chunk)
            a.jeq(P, NIL, chunk)
            a.set(P, P + "1")
            a.set(CNT, CNT + "0")
            a.goto(inner)
            a.label(done)
        self.list_length(dst, H)

    def phase_digits(self) -> None:
        """β = 2^s digit records, in order, with MSB-first and LSB-first bit lists."""
        a, R = self.a, self.R
        a.phase("digits")
        NIL, B0, B1 = R("NIL"), R("B0"), R("B1")
        LH = R("lv.h")
        lists = {}
        for order in ("msb", "lsb"):
            a.set(LH, NIL)
            a.new(LH)  # one cell holding the empty bit list (NIL)
            passes = [(B0,), (B1,)] if order == "msb" else [(B0, B1)]
            self.repeat(R("S"), partial(self._bit_level, passes))
            lists[order] = R(f"lv.{order}")
            a.set(lists[order], LH)
        DH, DT, PM, PL = R("DIG0"), R("dg.t"), R("dg.pm"), R("dg.pl")
        a.set(DH, NIL)
        a.set(PM, lists["msb"])
        a.set(PL, lists["lsb"])
        loop, first, made, done = (a.fresh("dg") for _ in range(4))
        a.label(loop)
        a.jeq(PM, NIL, done)
        a.jeq(DH, NIL, first)
        self.alloc(DT + DG.nx, DG)
        a.set(DT, DT + DG.nx)
        a.goto(made)
        a.label(first)
        self.alloc(DH, DG)
        a.set(DT, DH)
        a.label(made)
        a.set(DT + DG.msb, PM + "0")
        a.set(DT + DG.lsb, PL + "0")
        a.set(DT + DG.bucket, NIL)
        a.set(PM, PM + "1")
        a.set(PL, PL + "1")
        a.goto(loop)
        a.label(done)
        DEND, DLAST = R("DEND"), R("DLAST")
        a.new(DEND)
        a.set(DT + DG.nx, DEND)
        a.set(DLAST, DT)
        a.set(R("ZERO"), DH)
        a.set(R("ONE"), DH + DG.nx)
        p, q = R("dg.p"), R("dg.q")
        # succ (wrapping), mx, nz
        loop, done = a.fresh("dg"), a.fresh("dg")
        a.set(p, DH)
        a.label(loop)
        a.jeq(p, DEND, done)
        a.set(p + DG.succ, p + DG.nx)
        a.set(p + DG.mx, B0)
        a.set(p + DG.nz, B1)
        a.set(p, p + DG.nx)
        a.goto(loop)
        a.label(done)
        a.set(DLAST + DG.succ, DH)
        a.set(DLAST + DG.mx, B1)
        a.set(DH + DG.nz, B0)
        # pred
        loop, done = a.fresh("dg"), a.fresh("dg")
        a.set(DH + DG.pred, DLAST)
        a.set(q, DH)
        a.set(p, DH + DG.nx)
        a.label(loop)
        a.jeq(p, DEND, done)
        a.set(p + DG.pred, q)
        a.set(q, p)
        a.set(p, p + DG.nx)
        a.goto(loop)
        a.label(done)
        # neg: 0 ↦ 0, v ↦ β - v
        loop, done = a.fresh("dg"), a.fresh("dg")
        a.set(DH + DG.neg, DH)
        a.set(p, DH + DG.nx)
        a.set(q, DLAST)
        a.label(loop)
        a.jeq(p, DEND, done)
        a.set(p + DG.neg, q)
        a.set(p, p + DG.nx)
        a.set(q, q + DG.pred)
        a.goto(loop)
        a.label(done)
        # dbl0/dbl1: v ↦ 2v, 2v + 1 (mod β)
        loop, done = a.fresh("dg"), a.fresh("dg")
        a.set(p, DH)
        a.set(q, DH)
        a.label(loop)
        a.jeq(p, DEND, done)
        a.set(p + DG.dbl0, q)
        a.set(p + DG.dbl1, q + DG.succ)
        a.set(q, q + DG.succ + DG.succ)
        a.set(p, p + DG.nx)
        a.goto(loop)
        a.label(done)

    def _bit_level(self, passes: Sequence[tuple[str, ...]]) -> None:
        """From the list of k-bit values build the (k+1)-bit list.

        MSB order: one pass per leading bit (all 0x, then all 1x).
        LSB order: one pass, x0 and x1 after each x.
        """
        a, R = self.a, self.R
        NIL, LH, NH, NT, P = R("NIL"), R("lv.h"), R("lv.nh"), R("lv.nt"), R("lv.p")
        a.set(NH, NIL)
        a.set(NT, NIL)
        for bits in passes:
            loop, done = a.fresh("lv"), a.fresh("lv")
            a.set(P, LH)
            a.label(loop)
            a.jeq(P, NIL, done)
            for b in bits:
                self.append_cell(NH, NT, NIL)
                a.new(NT + "0")  # bit node (b, rest)
                a.set(NT + "00", b)
                a.set(NT + "01", P + "0")
            a.set(P, P + "1")
            a.goto(loop)
            a.label(done)
        a.set(LH, NH)

    def phase_tables(self) -> None:
        """The β² digit-pair entries (x±y, x·y), rows streamed in y order, plus the lookup trie."""
        a, R = self.a, self.R
        a.phase("tables")
        NIL, B0, B1, DH, DEND = R("NIL"), R("B0"), R("B1"), R("DIG0"), R("DEND")
        X, Y, ENH, ENT = R("tb.x"), R("tb.y"), R("EN.h"), R("tb.t")
        RA, CA, RS, BS, LO, HI, T, Z = R("tb.ra"), R("tb.ca"), R("tb.rs"), R("tb.bs"), R("tb.lo"), R("tb.hi"), R("T"), R("Z")
        a.set(ENH, NIL)
        rows, rows_done = a.fresh("tb"), a.fresh("tb")
        a.set(X, DH)
        a.label(rows)
        a.jeq(X, DEND, rows_done)
        # pass 1: entries of row x with sum/difference fields
        a.set(RA, X)
        a.set(CA, B0)
        a.set(RS, X)
        a.set(BS, B0)
        a.set(Y, DH)
        y1, p1done, first, made, notfirst, nc, nb = (a.fresh("tb") for _ in range(7))
        a.label(y1)
        a.jeq(Y, DEND, p1done)
        a.jeq(ENH, NIL, first)
        self.alloc(ENT + EN.nx, EN)
        a.set(ENT, ENT + EN.nx)
        a.goto(made)
        a.label(first)
        self.alloc(ENH, EN)
        a.set(ENT, ENH)
        a.label(made)
        a.set(ENT + EN.nx, NIL)
        a.jeq(Y, DH, notfirst + "h")
        a.goto(notfirst)
        a.label(notfirst + "h")
        a.set(X + DG.rowhead, ENT)
        a.label(notfirst)
        a.set(ENT + EN.alo, RA)
        a.set(ENT + EN.acy, CA)
        a.set(ENT + EN.slo, RS)
        a.set(ENT + EN.sbw, BS)
        a.jeq(RA + DG.mx, B0, nc)
        a.set(CA, B1)
        a.label(nc)
        a.set(RA, RA + DG.succ)
        a.jeq(RS + DG.nz, B1, nb)
        a.set(BS, B1)
        a.label(nb)
        a.set(RS, RS + DG.pred)
        a.set(Y, Y + DG.nx)
        a.goto(y1)
        a.label(p1done)
        # pass 2: z.tmp ← entry(x, z)
        inst, idone = a.fresh("tb"), a.fresh("tb")
        a.set(T, X + DG.rowhead)
        a.set(Z, DH)
        a.label(inst)
        a.jeq(Z, DEND, idone)
        a.set(Z + DG.tmp, T)
        a.set(T, T + EN.nx)
        a.set(Z, Z + DG.nx)
        a.goto(inst)
        a.label(idone)
        # pass 3: x·y = x·(y-1) + x, using entry(x, lo).alo/acy
        p3, p3done, noinc = a.fresh("tb"), a.fresh("tb"), a.fresh("tb")
        a.set(LO, DH)
        a.set(HI, DH)
        a.set(T, X + DG.rowhead)
        a.set(Y, DH)
        a.label(p3)
        a.jeq(Y, DEND, p3done)
        a.set(T + EN.mlo, LO)
        a.set(T + EN.mhi, HI)
        a.jeq(LO + DG.tmp + EN.acy, B0, noinc)
        a.set(HI, HI + DG.succ)
        a.label(noinc)
        a.set(LO, LO + DG.tmp + EN.alo)
        a.set(T, T + EN.nx)
        a.set(Y, Y + DG.nx)
        a.goto(p3)
        a.label(p3done)
        a.set(X, X + DG.nx)
        a.goto(rows)
        a.label(rows_done)
        # lookup trie: s bottom-up pairing rounds over the entries; level s = row roots
        LH, NH, NT, P = R("tr.h"), R("tr.nh"), R("tr.nt"), R("tr.p")
        a.set(NH, NIL)
        a.set(NT, NIL)
        a.set(P, ENH)
        loop, done = a.fresh("tr"), a.fresh("tr")
        a.label(loop)
        a.jeq(P, NIL, done)
        self.append_cell(NH, NT, NIL)
        a.new(NT + "0")
        a.set(NT + "00", P)
        a.set(NT + "01", P + EN.nx)
        a.set(P, P + EN.nx + EN.nx)
        a.goto(loop)
        a.label(done)
        a.set(LH, NH)

        def pair_level() -> None:
            loop, done = a.fresh("tr"), a.fresh("tr")
            a.set(NH, NIL)
            a.set(NT, NIL)
            a.set(P, LH)
            a.label(loop)
            a.jeq(P, NIL, done)
            self.append_cell(NH, NT, NIL)
            a.new(NT + "0")
            a.set(NT + "00", P + "0")
            a.set(NT + "01", P + "10")
            a.set(P, P + "11")
            a.goto(loop)
            a.label(done)
            a.set(LH, NH)

        self.repeat(R("S") + "0", pair_level)
        loop, done = a.fresh("tr"), a.fresh("tr")
        a.set(P, LH)
        a.set(X, DH)
        a.label(loop)
        a.jeq(X, DEND, done)
        a.set(X + DG.row, P + "0")
        a.set(P, P + "1")
        a.set(X, X + DG.nx)
        a.goto(loop)
        a.label(done)

    def new_zero_num(self, w: str) -> None:
        self.alloc(w, CO)
        for i in range(3):
            self.a.set(w + CO.fields[f"d{i}"], self.R("ZERO"))
        self.a.set(w + CO.nx, self.R("NIL"))

    def phase_lists(self) -> None:
        """Coefficient lists of length N = 2^L, the operands' chunks, the element pool."""
        a, R = self.a, self.R
        a.phase("chunks")
        NIL, U0 = R("NIL"), R("U0")
        for opnd in ("A", "B"):
            H, T = R(f"{opnd}C.h"), R("cl.t")
            self.new_zero_num(H)
            a.set(T, H)
            self.repeat(R("L"), partial(self._double_list, H, T))
            # chunks: d bits → digit by Horner on the dbl pointers
            BP, C, STK, CNT, V, Q = R("ch.bp"), R("ch.c"), R("ch.stk"), R("ch.cnt"), R("ch.v"), R("ch.q")
            outer, col, horner, hl, one, store, done = (a.fresh("ch") for _ in range(7))
            a.set(BP, R(f"{opnd}.h"))
            a.set(C, H)
            a.label(outer)
            a.jeq(BP, NIL, done)
            a.set(STK, NIL)
            a.set(CNT, R("D"))
            a.label(col)
            a.jeq(CNT, U0, horner)
            a.jeq(BP, NIL, horner)
            a.new(STK)  # push: the new cell's .1 is the old top
            a.set(STK + "0", BP + "0")
            a.set(BP, BP + "1")
            a.set(CNT, CNT + "0")
            a.goto(col)
            a.label(horner)
            a.set(V, R("ZERO"))
            a.set(Q, STK)
            a.label(hl)
            a.jeq(Q, NIL, store)
            a.jeq(Q + "0", R("B1"), one)
            a.set(V, V + DG.dbl0)
            a.set(Q, Q + "1")
            a.goto(hl)
            a.label(one)
            a.set(V, V + DG.dbl1)
            a.set(Q, Q + "1")
            a.goto(hl)
            a.label(store)
            a.set(C + CO.d0, V)
            a.set(C, C + CO.nx)
            a.goto(outer)
            a.label(done)
        a.phase("pool")
        POOL, PT, P = R("POOL"), R("po.t"), R("po.p")
        loop, first, made, done = (a.fresh("po") for _ in range(4))
        a.set(POOL, NIL)
        a.set(P, R("AC.h"))
        a.label(loop)
        a.jeq(P, NIL, done)
        a.jeq(POOL, NIL, first)
        self.alloc(PT + EL.nx, EL)
        a.set(PT, PT + EL.nx)
        a.goto(made)
        a.label(first)
        self.alloc(POOL, EL)
        a.set(PT, POOL)
        a.label(made)
        if self.lookup == "batched":
            for j in range(N_JOBS):
                self.alloc(PT + EL.fields[f"j{j}"], JB)
        a.set(P, P + CO.nx)
        a.goto(loop)
        a.label(done)
        a.set(PT + EL.nx, NIL)
        self.alloc(R("SE"), EL)

    def _double_list(self, head: str, tail: str) -> None:
        """Append one zero record per existing record."""
        a, R = self.a, self.R
        P, END = R("cl.p"), R("cl.end")
        loop, done = a.fresh("cl"), a.fresh("cl")
        a.set(P, head)
        a.set(END, tail)
        a.label(loop)
        self.new_zero_num(tail + CO.nx)
        a.set(tail, tail + CO.nx)
        a.jeq(P, END, done)
        a.set(P, P + CO.nx)
        a.goto(loop)
        a.label(done)

    def phase_modulus(self) -> None:
        """Find h (a digit, 2^e ≤ h < β/2) and a ∈ {3, 5, 7} with a^((q-1)/2) ≡ -1 (mod q)."""
        a, R = self.a, self.R
        a.phase("modulus")
        NIL, B1 = R("NIL"), R("B1")
        for rec in ("RM", "M1", "AR", "XR", "OM", "Z3", "KR"):
            self.new_zero_num(R(rec))
        H = R("H")
        a.set(H, R("ONE"))
        self.repeat(R("E2"), lambda: a.set(H, H + DG.dbl0))
        hloop, found, nexth = a.fresh("mod"), a.fresh("mod"), a.fresh("mod")
        a.label(hloop)
        a.jeq(H + DG.msb + "0", B1, "FAIL")
        # R mod q = 2^(3s) mod q by doubling 1
        a.set(R("RM") + CO.d0, R("ONE"))
        a.set(R("RM") + CO.d1, R("ZERO"))
        a.set(R("RM") + CO.d2, R("ZERO"))
        self.repeat(R("S3"), lambda: self.scalar("add", R("RM"), R("RM"), R("RM")))
        self.scalar("sub", R("Z3"), R("RM"), R("M1"))  # Montgomery form of -1
        self.copy_num(R("AR"), R("RM"))
        tries = R("mod.tries")
        a.set(tries, R("U0") + "111")
        attempt = a.fresh("mod")
        a.label(attempt)
        a.jeq(tries, R("U0"), nexth)
        a.set(tries, tries + "0")
        self.scalar("add", R("AR"), R("RM"), R("AR"))  # a·R for a = 3, 5, 7
        self.scalar("add", R("AR"), R("RM"), R("AR"))
        XR = R("XR")
        self.copy_num(XR, R("RM"))
        BIT, bl, bdone, nomul = R("mod.bit"), a.fresh("mod"), a.fresh("mod"), a.fresh("mod")
        a.set(BIT, H + DG.msb)  # ψ = a^h by square-and-multiply over the bits of h
        a.label(bl)
        a.jeq(BIT, NIL, bdone)
        self.scalar("mul", XR, XR, XR)
        a.jeq(BIT + "0", R("B0"), nomul)
        self.scalar("mul", XR, R("AR"), XR)
        a.label(nomul)
        a.set(BIT, BIT + "1")
        a.goto(bl)
        a.label(bdone)
        self.repeat(R("CP"), lambda: self.scalar("mul", XR, XR, XR))  # ω = ψ^(2^c')
        self.copy_num(R("OM"), XR)
        self.repeat(R("LM1"), lambda: self.scalar("mul", XR, XR, XR))  # ω^(N/2)
        for i in range(3):
            a.jeq(XR + CO.fields[f"d{i}"], R("M1") + CO.fields[f"d{i}"], f"{attempt}.eq{i}")
            a.goto(attempt)
            a.label(f"{attempt}.eq{i}")
        a.goto(found)
        a.label(nexth)
        a.set(H, H + DG.succ)
        a.goto(hloop)
        a.label(found)
        # KR = 2^(6s - L) mod q = N⁻¹·R² mod q  (scale factor after pointwise Montgomery products)
        KR = R("KR")
        a.set(KR + CO.d0, R("ONE"))
        a.set(KR + CO.d1, R("ZERO"))
        a.set(KR + CO.d2, R("ZERO"))
        self.repeat(R("K6"), lambda: self.scalar("add", KR, KR, KR))

    def phase_twiddles(self) -> None:
        """W = [ω^j·R mod q : j < N/2]; sub-lists W_h = every (N/2h)-th entry; stage lists."""
        a, R = self.a, self.R
        a.phase("twiddles")
        NIL = R("NIL")
        WH, WT, P = R("W.h"), R("tw.t"), R("tw.p")
        self.new_zero_num(WH)
        self.copy_num(WH, R("RM"))
        a.set(WT, WH)
        loop, done = a.fresh("tw"), a.fresh("tw")
        a.set(P, R("AC.h") + CO.nx + CO.nx)
        a.label(loop)
        a.jeq(P, NIL, done)
        self.new_zero_num(WT + CO.nx)
        self.scalar("mul", WT, R("OM"), WT + CO.nx)
        a.set(WT, WT + CO.nx)
        a.set(P, P + CO.nx + CO.nx)
        a.goto(loop)
        a.label(done)
        SH, ST, NH, NT = R("sl.h"), R("sl.t"), R("sl.nh"), R("sl.nt")
        a.set(SH, NIL)
        a.set(ST, NIL)
        loop, done = a.fresh("sl"), a.fresh("sl")
        a.set(P, WH)
        a.label(loop)
        a.jeq(P, NIL, done)
        self.append_cell(SH, ST, P)
        a.set(P, P + CO.nx)
        a.goto(loop)
        a.label(done)
        DIF, DIFT, DIT = R("st.dif"), R("st.dift"), R("st.dit")
        a.set(DIF, NIL)
        a.set(DIFT, NIL)
        a.set(DIT, NIL)
        stage, last = a.fresh("sl"), a.fresh("sl")
        a.label(stage)
        self.append_cell(DIF, DIFT, SH)  # decreasing h
        a.new(DIT)  # push: increasing h
        a.set(DIT + "0", SH)
        a.jeq(SH + "1", NIL, last)
        a.set(NH, NIL)
        a.set(NT, NIL)
        loop, done = a.fresh("sl"), a.fresh("sl")
        a.set(P, SH)
        a.label(loop)
        a.jeq(P, NIL, done)
        self.append_cell(NH, NT, P + "0")
        a.set(P, P + "11")  # NIL.1 = NIL, so this is safe at the end
        a.goto(loop)
        a.label(done)
        a.set(SH, NH)
        a.goto(stage)
        a.label(last)

    def build_butterflies(self, stage: str, operands: Sequence[str]) -> None:
        """Pool elements (u, v, w) = (x_j, x_{j+h}, W_h[j]) for every block of 2h, h = |W_h|."""
        a, R = self.a, self.R
        NIL, E = R("NIL"), R("E")
        P, Q, U, V, T = R("bf.p"), R("bf.q"), R("bf.u"), R("bf.v"), R("bf.t")
        a.set(E, R("POOL"))
        for opnd in operands:
            blk, adv, pair, pl, blkend, done = (a.fresh("bf") for _ in range(6))
            a.set(P, R(f"{opnd}C.h"))
            a.label(blk)
            a.jeq(P, NIL, done)
            a.set(Q, P)
            a.set(T, stage)
            a.label(adv)
            a.jeq(T, NIL, pair)
            a.set(Q, Q + CO.nx)
            a.set(T, T + "1")
            a.goto(adv)
            a.label(pair)
            a.set(U, P)
            a.set(V, Q)
            a.set(T, stage)
            a.label(pl)
            a.jeq(T, NIL, blkend)
            a.set(E + EL.u, U)
            a.set(E + EL.v, V)
            a.set(E + EL.w, T + "0")
            a.set(E, E + EL.nx)
            a.set(U, U + CO.nx)
            a.set(V, V + CO.nx)
            a.set(T, T + "1")
            a.goto(pl)
            a.label(blkend)
            a.set(P, V)
            a.goto(blk)
            a.label(done)
        a.set(R("VEND"), E)

    def phase_transform(self) -> None:
        a, R = self.a, self.R
        NIL, E = R("NIL"), R("E")
        a.phase("forward NTT")
        ST = R("st.cur")
        loop, done = a.fresh("fw"), a.fresh("fw")
        a.set(ST, R("st.dif"))
        a.label(loop)
        a.jeq(ST, NIL, done)
        self.build_butterflies(ST + "0", ("A", "B"))
        self.run_vector_kernel("dif", R("POOL"), R("VEND"))
        a.set(ST, ST + "1")
        a.goto(loop)
        a.label(done)
        a.phase("pointwise")
        U, V = R("pw.u"), R("pw.v")
        loop, done = a.fresh("pw"), a.fresh("pw")
        a.set(E, R("POOL"))
        a.set(U, R("AC.h"))
        a.set(V, R("BC.h"))
        a.label(loop)
        a.jeq(U, NIL, done)
        a.set(E + EL.u, U)
        a.set(E + EL.v, V)
        a.set(E, E + EL.nx)
        a.set(U, U + CO.nx)
        a.set(V, V + CO.nx)
        a.goto(loop)
        a.label(done)
        a.set(R("VEND"), E)
        self.run_vector_kernel("pointwise", R("POOL"), R("VEND"))
        a.phase("inverse NTT")
        loop, done = a.fresh("iv"), a.fresh("iv")
        a.set(ST, R("st.dit"))
        a.label(loop)
        a.jeq(ST, NIL, done)
        self.build_butterflies(ST + "0", ("A",))
        self.run_vector_kernel("dit", R("POOL"), R("VEND"))
        a.set(ST, ST + "1")
        a.goto(loop)
        a.label(done)
        # c_m = N⁻¹ · Y_{-m mod N}: reverse positions 1 .. N-1
        PREV, P, NXT = R("rv.prev"), R("rv.p"), R("rv.n")
        loop, done = a.fresh("rv"), a.fresh("rv")
        a.set(PREV, NIL)
        a.set(P, R("AC.h") + CO.nx)
        a.label(loop)
        a.jeq(P, NIL, done)
        a.set(NXT, P + CO.nx)
        a.set(P + CO.nx, PREV)
        a.set(PREV, P)
        a.set(P, NXT)
        a.goto(loop)
        a.label(done)
        a.set(R("AC.h") + CO.nx, PREV)
        loop, done = a.fresh("sc"), a.fresh("sc")
        a.set(E, R("POOL"))
        a.set(U, R("AC.h"))
        a.label(loop)
        a.jeq(U, NIL, done)
        a.set(E + EL.u, U)
        a.set(E, E + EL.nx)
        a.set(U, U + CO.nx)
        a.goto(loop)
        a.label(done)
        a.set(R("VEND"), E)
        self.run_vector_kernel("scale", R("POOL"), R("VEND"))

    def phase_output(self) -> None:
        """Σ c_m·2^(d·m): add each c_m bit-serially into an accumulator, emit d bits per m."""
        a, R = self.a, self.R
        a.phase("carry+output")
        NIL, B0, B1, U0 = R("NIL"), R("B0"), R("B1"), R("U0")
        AH, AT, C, CELL, CARRY, BP, CNT = R("acc.h"), R("acc.t"), R("out.c"), R("out.cell"), R("out.cy"), R("out.bp"), R("out.n")
        a.set(AH, NIL)
        a.set(AT, NIL)
        a.set(C, R("AC.h"))
        cl, flush = a.fresh("out"), a.fresh("out")
        a.label(cl)
        a.jeq(C, NIL, flush)
        a.set(CELL, AH)
        a.set(CARRY, B0)
        for i in range(3):
            bl, bdone, have, a1, a0b1, a1b1, fdone = (a.fresh("fa") for _ in range(7))
            a.set(BP, C + CO.fields[f"d{i}"] + DG.lsb)
            a.label(bl)
            a.jeq(BP, NIL, bdone)
            a.jeq(CELL, NIL, have + "n")
            a.goto(have)
            a.label(have + "n")
            self.append_cell(AH, AT, B0)
            a.set(CELL, AT)
            a.label(have)
            # full adder: (cell, carry) ← cell + bit + carry
            a.jeq(CELL + "0", B1, a1)
            a.jeq(BP + "0", B1, a0b1)
            a.set(CELL + "0", CARRY)
            a.set(CARRY, B0)
            a.goto(fdone)
            a.label(a0b1)
            a.set(CELL + "0", CARRY + "0")
            a.goto(fdone)
            a.label(a1)
            a.jeq(BP + "0", B1, a1b1)
            a.set(CELL + "0", CARRY + "0")
            a.goto(fdone)
            a.label(a1b1)
            a.set(CELL + "0", CARRY)
            a.set(CARRY, B1)
            a.label(fdone)
            a.set(CELL, CELL + "1")
            a.set(BP, BP + "1")
            a.goto(bl)
            a.label(bdone)
        prop, have, emit, flip, pdone = (a.fresh("cy") for _ in range(5))
        a.label(prop)
        a.jeq(CARRY, B0, emit)
        a.jeq(CELL, NIL, have + "n")
        a.goto(have)
        a.label(have + "n")
        self.append_cell(AH, AT, B0)
        a.set(CELL, AT)
        a.label(have)
        a.jeq(CELL + "0", B1, flip)
        a.set(CELL + "0", B1)
        a.set(CARRY, B0)
        a.goto(pdone)
        a.label(flip)
        a.set(CELL + "0", B0)
        a.label(pdone)
        a.set(CELL, CELL + "1")
        a.goto(prop)
        a.label(emit)
        el, zero, o1, popped, nextc = (a.fresh("em") for _ in range(5))
        a.set(CNT, R("D"))
        a.label(el)
        a.jeq(CNT, U0, nextc)
        a.set(CNT, CNT + "0")
        a.jeq(AH, NIL, zero)
        a.jeq(AH + "0", B1, o1)
        a.output(0)
        a.goto(popped)
        a.label(o1)
        a.output(1)
        a.label(popped)
        a.set(AH, AH + "1")
        a.jeq(AH, NIL, zero + "t")
        a.goto(el)
        a.label(zero + "t")
        a.set(AT, NIL)
        a.goto(el)
        a.label(zero)
        a.output(0)
        a.goto(el)
        a.label(nextc)
        a.set(C, C + CO.nx)
        a.goto(cl)
        a.label(flush)
        fl, f1, fdone = a.fresh("fl"), a.fresh("fl"), a.fresh("fl")
        a.label(fl)
        a.jeq(AH, NIL, fdone)
        a.jeq(AH + "0", B1, f1)
        a.output(0)
        a.set(AH, AH + "1")
        a.goto(fl)
        a.label(f1)
        a.output(1)
        a.set(AH, AH + "1")
        a.goto(fl)
        a.label(fdone)
        a.halt()
        a.phase("fail")
        a.label("FAIL")  # modulus search exhausted (never observed; see tests)
        a.halt()


@cache
def build_program(lookup: Lookup = "batched") -> Program:
    return Builder(lookup).build()


@cache
def fast_machine(lookup: Lookup = "batched") -> FastSMM:
    return FastSMM(build_program(lookup))


# ════════════════════════════════════════════════════════════════════════════
# Part 5 — host interface (tape encoding) and host-side reference model
# ════════════════════════════════════════════════════════════════════════════


def encode_tape(x: int, y: int) -> list[int]:
    """Each operand LSB-first, every bit written as '1' + bit, operand closed by '0'."""
    tape: list[int] = []
    for z in (x, y):
        if z < 0:
            raise ValueError("operands must be non-negative")
        for ch in reversed(format(z, "b")) if z else ():
            tape += [1, 1 if ch == "1" else 0]
        tape.append(0)
    return tape


def decode_output(bits: Sequence[int]) -> int:
    s = "".join("1" if b else "0" for b in reversed(bits))
    return int(s, 2) if s else 0


def smm_multiply(
    x: int, y: int, lookup: Lookup = "batched", engine: Literal["fast", "reference"] = "fast"
) -> tuple[int, RunResult]:
    tape = encode_tape(x, y)
    prog = build_program(lookup)
    res = fast_machine(lookup).run(tape) if engine == "fast" else run_reference(prog, tape)
    if res.halted_at == prog.labels["FAIL"]:
        raise RuntimeError("modulus search exhausted")
    return decode_output(res.output), res


@dataclass(frozen=True)
class Params:
    L: int
    s: int
    d: int
    h: int
    a: int

    @property
    def N(self) -> int:
        return 1 << self.L


def reference_params(nx: int, ny: int) -> Params:
    """Host replica of the SMM's parameter rule (test oracle)."""

    def s_of(L: int) -> int:
        return (L + 1) // 2 + 1

    def d_of(L: int) -> int:
        return max(1, (L - 1) // 4)

    lam = (nx + ny).bit_length()
    L = 3
    while lam + 1 > L + d_of(L).bit_length() - 1:
        L += 1

    def ell(d: int) -> int:
        return max(0, -(-nx // d) + -(-ny // d) - 2).bit_length()

    while L > 3 and ell(d_of(L - 1)) <= L - 1:
        L -= 1
    s, d = s_of(L), d_of(L)
    e = max(0, 2 * d - (2 * s - L))
    for h in range(1 << e, 1 << (s - 1)):
        q = h * (1 << (2 * s)) + 1
        for a in (3, 5, 7):
            if pow(a, (q - 1) // 2, q) == q - 1:
                return Params(L, s, d, h, a)
    raise RuntimeError(f"no modulus for L={L}")


def machine_params(prog: Program, res: RunResult) -> dict[str, int]:
    """Parameters as the SMM left them in its final pointer structure (host inspection only)."""
    reg = prog.registers
    u0 = res.node(reg["U0"])

    def unary(name: str) -> int:
        x, k = res.node(reg[name]), 0
        while x != u0:
            x, k = res.p0[x], k + 1
        return k

    def digit(name: str) -> int:
        x, target, k = res.node(reg["DIG0"]), res.node(reg[name]), 0
        nx = DG.nx
        while x != target:
            for ch in nx:
                x = res.p1[x] if ch == "1" else res.p0[x]
            k += 1
        return k

    return {"L": unary("L"), "s": unary("S"), "d": unary("D"), "h": digit("H")}


# ════════════════════════════════════════════════════════════════════════════
# Part 6 — verification suite and scaling benchmark
# ════════════════════════════════════════════════════════════════════════════


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}", flush=True)
        if not ok:
            self.failures.append(name)


def audit_program(prog: Program) -> tuple[bool, str]:
    """The instruction set is exactly the SMM's; only NEW and SET write memory; words ∈ {0,1}*."""
    ops = {ins.op for ins in prog.code}
    words_ok = all(not (ins.w.strip("01") or ins.v.strip("01")) for ins in prog.code)
    writers = {ins.op for ins in prog.code if ins.w and ins.op not in (Op.TEST_EQ,)}
    targets_ok = all(0 <= ins.t0 <= len(prog.code) for ins in prog.code if ins.op in (Op.TEST_EQ, Op.GOTO, Op.INPUT)) and all(
        0 <= ins.t1 <= len(prog.code) for ins in prog.code if ins.op is Op.INPUT
    )
    centre_fixed = not any(ins.op in MUTATING_OPS and ins.w == "" for ins in prog.code)
    ok = ops <= set(Op) and words_ok and writers <= set(MUTATING_OPS) and targets_ok
    counts = {op.name: sum(ins.op is op for ins in prog.code) for op in Op}
    return (
        ok,
        f"{len(prog.code)} instructions {counts}, max word length {max(len(i.w) + len(i.v) for i in prog.code)}, centre fixed: {centre_fixed}",
    )


def _digits3(beta: int, x: int) -> tuple[int, int, int]:
    return x % beta, (x // beta) % beta, x // (beta * beta)


def check_kernels(rng: random.Random, trials: int) -> tuple[bool, str]:
    """Host oracle for the digit kernels: every kernel equals modular arithmetic mod q = h·β² + 1."""
    ks = kernels()
    n = 0
    for s in (3, 4, 5, 6, 7, 9, 12):
        beta = 1 << s
        dig = partial(_digits3, beta)
        for t in range(trials):
            h = rng.randrange(1, beta // 2) if t else beta // 2 - 1
            q = h * beta * beta + 1
            rinv = pow(beta**3, -1, q)
            u, v, w, kk = (rng.randrange(q) for _ in range(4))
            if t == 1:
                u, v, w = q - 1, q - 1, q - 1
            consts: dict[str, object] = {"B0": 0, "B1": 1, "H": h, "ZERO": 0, "KR": dig(kk)}
            inp: dict[tuple[str, ...], object] = {}
            for rel, val in (("EL.u", u), ("EL.v", v), ("EL.w", w)):
                for i, dd in enumerate(dig(val)):
                    inp[(rel, f"CO.d{i}")] = dd

            tw = v * w * rinv % q
            expect = {
                ("dif", "EL.u"): (u + v) % q,
                ("dif", "EL.v"): (u - v) * w * rinv % q,
                ("dit", "EL.u"): (u + tw) % q,
                ("dit", "EL.v"): (u - tw) % q,
                ("pointwise", "EL.u"): u * v * rinv % q,
                ("scale", "EL.u"): u * kk * rinv % q,
                ("add", "EL.w"): (u + v) % q,
                ("sub", "EL.w"): (u - v) % q,
                ("mul", "EL.w"): u * v * rinv % q,
            }
            for (name, rel), e in expect.items():
                n += 1
                out = eval_kernel_host(ks[name], beta, consts, inp)
                if sum(out[(rel, f"CO.d{i}")] * beta**i for i in range(3)) != e:
                    return False, f"{name} wrong for s={s} h={h} u={u} v={v} w={w}"
    return True, f"{n} kernel evaluations"


def random_operand(rng: random.Random, bits: int) -> int:
    return rng.getrandbits(bits) | (1 << (bits - 1)) if bits else 0


def run_tests(quick: bool = False, seed: int = 20260923) -> bool:
    rng = random.Random(seed)
    rep = Report()
    t0 = time.perf_counter()
    print("SMM program")
    for lookup in ("batched", "direct"):
        ok, detail = audit_program(build_program(lookup))
        rep.check(f"instruction audit ({lookup})", ok, detail)

    print("Digit kernels (host oracle)")
    ok, detail = check_kernels(rng, 20 if quick else 150)
    rep.check("kernels = modular arithmetic", ok, detail)

    print("Parameter rule")
    bad = [L for L in range(3, 41) if not _modulus_exists(L)]
    rep.check("a modulus q = h·β²+1 with a^((q-1)/2) ≡ -1 exists for every L in 3..40", not bad, f"missing: {bad}" if bad else "")

    print("Reference interpreter vs compiled interpreter")
    for x, y in ((0, 0), (5, 3), (random_operand(rng, 41), random_operand(rng, 23))):
        zr, rr = smm_multiply(x, y, engine="reference")
        zf, rf = smm_multiply(x, y, engine="fast")
        same = (rr.output, rr.steps, rr.hops, rr.nodes, rr.by_phase) == (rf.output, rf.steps, rf.hops, rf.nodes, rf.by_phase)
        rep.check(f"identical runs for {x.bit_length()}×{y.bit_length()} bits", same and zr == zf == x * y, f"{rf.steps:,} steps")

    print("Products checked against Python integers")
    cases: list[tuple[int, int]] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 3),
        (255, 255),
        (1 << 63, 1 << 63),
        ((1 << 100) - 1, (1 << 100) - 1),
    ]
    cases += [((1 << a) - 1, 1 << b) for a, b in ((7, 13), (64, 1), (1, 64))]
    sizes = (
        (1, 2, 3, 4, 7, 8, 15, 31, 32, 63, 100, 257, 700)
        if quick
        else (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 200, 257, 500, 777, 1024, 1500, 2048)
    )
    for nx in sizes:
        for _ in range(2 if quick else 4):
            ny = rng.choice((nx, rng.randint(1, nx), rng.randint(nx, 2 * nx + 1)))
            cases.append((random_operand(rng, nx), random_operand(rng, ny)))
    for lookup in ("batched", "direct"):
        bad = []
        todo = cases if lookup == "batched" else cases[:: 3 if quick else 2]
        for x, y in todo:
            z, _ = smm_multiply(x, y, lookup)  # type: ignore[arg-type]
            if z != x * y:
                bad.append((x, y))
        rep.check(f"{len(todo)} random/edge products ({lookup} lookups)", not bad, f"first failure {bad[0]}" if bad else "")

    print("Parameters computed by the SMM (read back from its pointer structure)")
    prog = build_program("batched")
    mism = []
    for nx, ny in ((1, 1), (10, 3), (64, 64), (300, 100), (1024, 1024), (2048, 1)):
        x, y = random_operand(rng, nx), random_operand(rng, ny)
        _, res = smm_multiply(x, y)
        got = machine_params(prog, res)
        ref = reference_params(nx, ny)
        want = {"L": ref.L, "s": ref.s, "d": ref.d, "h": ref.h}
        if got != want:
            mism.append((nx, ny, got, want))
    rep.check("L, s, d, h match the rule", not mism, f"{mism[0]}" if mism else "")

    dt = time.perf_counter() - t0
    print(f"{'ALL PASSED' if not rep.failures else 'FAILED: ' + ', '.join(rep.failures)}  ({dt:.0f} s)")
    return not rep.failures


def _modulus_exists(L: int) -> bool:
    s, d = (L + 1) // 2 + 1, max(1, (L - 1) // 4)
    e = max(0, 2 * d - (2 * s - L))
    for h in range(1 << e, 1 << (s - 1)):
        q = h * (1 << (2 * s)) + 1
        if any(pow(a, (q - 1) // 2, q) == q - 1 for a in (3, 5, 7)):
            return True
    return False


def bench_sizes_full_transforms(L_values: Sequence[int]) -> list[int]:
    """Operand sizes (bits each) that fill the transform exactly: n = d(L)·2^(L-1)."""
    return [max(1, (L - 1) // 4) << (L - 1) for L in L_values]


def bench_one(n: int, lookup: Lookup, seed: int = 1) -> dict[str, object]:
    rng = random.Random(seed * 1_000_003 + n)
    x, y = random_operand(rng, n), random_operand(rng, n)
    t = time.perf_counter()
    z, res = smm_multiply(x, y, lookup)
    dt = time.perf_counter() - t
    prm = machine_params(build_program(lookup), res)
    N = 1 << prm["L"]
    ntt = sum(res.by_phase.get(ph, 0) for ph in ("forward NTT", "pointwise", "inverse NTT"))
    return {
        "lookup": lookup,
        "bits_per_operand": n,
        "input_bits": 2 * n,
        "correct": z == x * y,
        **prm,
        "N": N,
        "steps": res.steps,
        "hops": res.hops,
        "nodes": res.nodes,
        "ntt_steps": ntt,
        "by_op": res.by_op,
        "by_phase": res.by_phase,
        "seconds": round(dt, 2),
    }


BENCH_HEADER = (
    f"{'lookup':8s} {'n':>9s} {'L':>3s} {'s':>3s} {'d':>3s} {'steps':>15s} {'steps/n':>8s} "
    f"{'NEW/n':>6s} {'SET/n':>6s} {'TEST/n':>6s} {'GOTO/n':>6s} {'NTT/(N·L)':>9s} {'nodes/n':>7s} {'ok':>2s} {'sec':>7s}"
)


def print_bench(rows: Sequence[dict[str, object]]) -> None:
    """n = input bits (both operands); NTT/(N·L) = transform steps per butterfly slot."""
    for r in rows:
        n: int = r["input_bits"]  # type: ignore[assignment]
        op: dict[str, int] = r["by_op"]  # type: ignore[assignment]
        print(
            f"{r['lookup']:8s} {n:>9,} {r['L']:>3} {r['s']:>3} {r['d']:>3} {r['steps']:>15,} "
            f"{r['steps'] / n:>8,.0f} {op['NEW'] / n:>6,.0f} {op['SET'] / n:>6,.0f} {op['TEST_EQ'] / n:>6,.0f} "  # type: ignore[operator]
            f"{op['GOTO'] / n:>6,.0f} {r['ntt_steps'] / (r['N'] * r['L']):>9,.0f} "  # type: ignore[operator]
            f"{r['nodes'] / n:>7,.1f} {'✓' if r['correct'] else '✗':>2s} {r['seconds']:>7}"  # type: ignore[operator]
        )


def summarize_bench(rows: Sequence[dict[str, object]], min_L: int = 10) -> None:
    """Least-squares slope of log(steps) against log(n), and the spread of steps/n."""
    import math

    for lookup in sorted({str(r["lookup"]) for r in rows}):
        sel = [r for r in rows if r["lookup"] == lookup and r["L"] >= min_L]  # type: ignore[operator]
        if len(sel) < 2:
            continue
        xs = [math.log(r["input_bits"]) for r in sel]  # type: ignore[arg-type]
        ys = [math.log(r["steps"]) for r in sel]  # type: ignore[arg-type]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
        per_n = [r["steps"] / r["input_bits"] for r in sel]  # type: ignore[operator]
        per_op = [r["ntt_steps"] / (r["N"] * r["L"]) for r in sel]  # type: ignore[operator]
        lo, hi = min(r["input_bits"] for r in sel), max(r["input_bits"] for r in sel)  # type: ignore[type-var]
        print(
            f"{lookup}: n = {lo:,}..{hi:,} (L ≥ {min_L}): slope d log(steps)/d log(n) = {slope:.3f}; "
            f"steps/n in [{min(per_n):,.0f}, {max(per_n):,.0f}]; NTT steps per butterfly slot in [{min(per_op):,.0f}, {max(per_op):,.0f}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("operands", nargs="*", type=int, help="two non-negative integers to multiply")
    ap.add_argument(
        "--lookup",
        choices=("batched", "direct"),
        default="batched",
        help="batched = O(n) (default); direct = trie walks, Θ(n log n)",
    )
    ap.add_argument("--reference", action="store_true", help="run on the literal (slow) interpreter")
    ap.add_argument("--test", action="store_true", help="run the verification suite")
    ap.add_argument("--quick", action="store_true", help="smaller verification suite")
    ap.add_argument("--bench", action="store_true", help="step counts vs n")
    ap.add_argument(
        "--L", type=int, nargs="*", default=list(range(8, 15)), help="bench: transform exponents (operand size d(L)·2^(L-1) bits)"
    )
    ap.add_argument("--sizes", type=int, nargs="*", help="bench: explicit operand sizes in bits (overrides --L)")
    ap.add_argument("--both", action="store_true", help="bench: also run the direct-lookup control")
    ap.add_argument("--json", help="bench: append result rows (JSON lines) to this file")
    ap.add_argument("--summarize", metavar="FILE", help="print the table and fit for a JSON-lines bench file")
    ap.add_argument("--listing", action="store_true", help="print the SMM program")
    args = ap.parse_args(argv)
    if args.listing:
        print(build_program(args.lookup).listing())
        return 0
    if args.test:
        return 0 if run_tests(args.quick) else 1
    if args.bench:
        sizes = args.sizes or bench_sizes_full_transforms(args.L)
        rows = []
        print(BENCH_HEADER)
        for lookup in ("batched", "direct") if args.both else (args.lookup,):
            for n in sizes:
                row = bench_one(n, lookup)
                rows.append(row)
                print_bench([row])
                if args.json:
                    import json

                    with open(args.json, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
        print()
        summarize_bench(rows)
        return 0 if all(r["correct"] for r in rows) else 1
    if args.summarize:
        import json

        with open(args.summarize) as fh:
            rows = sorted((json.loads(line) for line in fh), key=lambda r: (r["lookup"], r["input_bits"]))
        print(BENCH_HEADER)
        print_bench(rows)
        print()
        summarize_bench(rows)
        return 0
    if len(args.operands) == 2:
        x, y = args.operands
        t0 = time.perf_counter()
        z, res = smm_multiply(x, y, args.lookup, "reference" if args.reference else "fast")
        dt = time.perf_counter() - t0
        print(f"{x} × {y} = {z}   [{'OK' if z == x * y else 'MISMATCH'}]")
        print(f"SMM steps {res.steps:,}  pointer hops {res.hops:,}  nodes {res.nodes:,}  ({dt:.2f} s)")
        print("steps by instruction:", {k: v for k, v in res.by_op.items() if v})
        return 0 if z == x * y else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
