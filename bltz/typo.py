"""Typo-cluster augmentation (docs/32 AE 补救工具箱 #2, 2026-09-20 立项拍板).

corrupt(s, rng) -> a typo'd variant of s. The TRAINING TARGET IS THE TYPO'D
STRING ITSELF (not denoising to the clean form): each real string becomes a
dense point cloud of near-miss variants, so a slightly-off predicted embedding
decodes to a typo instead of OOD garbage.

Char units: ASCII printable bytes are single-byte units; UTF-8 multibyte
characters are indivisible units (only whole-char delete/duplicate — no
invalid UTF-8 inside). Ops mix (2026-09-20 拍板): substitute 40 / delete 20 /
duplicate 15 / transpose 15 / case-flip 10 (%). Edits per selected string:
1 (85%) or 2 (15%). Length gate: len<3 never augmented; len 3-4 only
substitute/case-flip. Frequency throttle lives in train_ae (top-freq strings
get p x freq_scale).
"""
from __future__ import annotations

import random

ASCII_PRINT = set(range(32, 127))
LETTER_LO = b"abcdefghijklmnopqrstuvwxyz"
LETTER_UP = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = b"0123456789"


def char_units(s: bytes) -> list[bytes]:
    """Split into indivisible units: ASCII byte or whole UTF-8 char."""
    out: list[bytes] = []
    i = 0
    n = len(s)
    while i < n:
        b = s[i]
        if b < 0x80:
            out.append(s[i : i + 1]); i += 1
            continue
        L = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
        if i + L <= n and all(0x80 <= s[i + k] < 0xC0 for k in range(1, L)):
            out.append(s[i : i + L]); i += L
        else:
            out.append(s[i : i + 1]); i += 1  # malformed: byte-sized unit
    return out


def _is_ascii_letter(u: bytes) -> bool:
    return len(u) == 1 and (u[0] in LETTER_LO or u[0] in LETTER_UP)


def corrupt(
    s: bytes,
    rng: random.Random,
    ops: tuple[float, ...] = (0.40, 0.20, 0.15, 0.15, 0.10),
    two_edit_p: float = 0.15,
    l_max: int = 32,
) -> bytes | None:
    """One typo variant of s, or None if augmentation is inapplicable/no-op."""
    if len(s) < 3:
        return None
    allow = ("sub", "case") if len(s) <= 4 else ("sub", "del", "dup", "trans", "case")
    w = dict(zip(("sub", "del", "dup", "trans", "case"), ops))
    cur = s
    for _ in range(1 + (rng.random() < two_edit_p)):
        cur2 = None
        for _try in range(8):
            op = rng.choices(list(w), weights=[w[k] for k in w])[0]
            if op not in allow:
                continue
            cur2 = _apply(cur, op, rng, l_max)
            if cur2 is not None:
                break
        if cur2 is None:
            break
        cur = cur2
    if cur == s or not cur or len(cur) > l_max:
        return None
    return cur


def _apply(s: bytes, op: str, rng: random.Random, l_max: int) -> bytes | None:
    u = char_units(s)
    if op == "sub":
        cands = [i for i, x in enumerate(u) if len(x) == 1 and (_is_ascii_letter(x) or x in DIGITS)]
        if not cands:
            return None
        i = rng.choice(cands)
        b = u[i][0]
        if b in LETTER_LO:
            nb = bytes([rng.choice(LETTER_LO)])
        elif b in LETTER_UP:
            nb = bytes([rng.choice(LETTER_UP)])
        else:
            nb = bytes([rng.choice(DIGITS)])
        if nb == u[i]:
            return None
        return b"".join(u[:i]) + nb + b"".join(u[i + 1 :])
    if op == "del":
        if len(u) < 2:
            return None
        i = rng.randrange(len(u))
        return b"".join(u[:i]) + b"".join(u[i + 1 :])
    if op == "dup":
        i = rng.randrange(len(u))
        if len(s) + len(u[i]) > l_max:
            return None
        return b"".join(u[: i + 1]) + u[i] + b"".join(u[i + 1 :])
    if op == "trans":
        cands = [i for i in range(len(u) - 1)
                 if len(u[i]) == 1 and len(u[i + 1]) == 1 and u[i][0] in ASCII_PRINT
                 and u[i + 1][0] in ASCII_PRINT]
        if not cands:
            return None
        i = rng.choice(cands)
        u[i], u[i + 1] = u[i + 1], u[i]
        return b"".join(u)
    if op == "case":
        cands = [i for i, x in enumerate(u) if _is_ascii_letter(x)]
        if not cands:
            return None
        i = rng.choice(cands)
        b = u[i][0]
        nb = bytes([b - 32 if b in LETTER_LO else b + 32])
        return b"".join(u[:i]) + nb + b"".join(u[i + 1 :])
    return None


def p_eff(s: bytes, p: float, freq_set: frozenset[bytes] | None, freq_scale: float) -> float:
    """Length gate x frequency throttle (2026-09-20 拍板自适应版)."""
    n = len(s)
    if n < 3:
        return 0.0
    g = 0.5 if n <= 4 else 1.0
    f = freq_scale if (freq_set is not None and s in freq_set) else 1.0
    return p * g * f
