"""Typo augmentation sanity: char units, op invariants, gates.
Run: python tests/test_typo.py
"""
from __future__ import annotations

import random

from bltz.typo import char_units, corrupt, p_eff

# char units: ASCII singles + whole UTF-8 chars
u = char_units("A你b".encode("utf-8"))
assert u == [b"A", "你".encode(), b"b"], u
assert b"".join(char_units("aBc123 !".encode())) == b"aBc123 !"

# len<3 never augmented
assert corrupt(b"ab", random.Random(0)) is None
assert corrupt(b"x", random.Random(0)) is None

# invariants over many trials: non-empty, != original (when applied), <= l_max
s = b"intelligence"
diffs = 0
for i in range(500):
    c = corrupt(s, random.Random(i))
    if c is None:
        continue
    assert 0 < len(c) <= 32 and c != s
    diffs += 1
assert diffs > 400, diffs  # applicable nearly always for a 12-byte word

# len 3-4: only length-preserving ops (substitute/case-flip)
for i in range(300):
    c = corrupt(b"the", random.Random(i))
    if c is not None:
        assert len(c) == 3, c

# op diversity: on a longer string, observe all four length-change signatures
saw = set()
for i in range(800):
    c = corrupt(b"complicated", random.Random(1000 + i))
    if c is None:
        continue
    if len(c) < 11:
        saw.add("shorter")
    elif len(c) > 11:
        saw.add("longer")
    elif sum(a != b for a, b in zip(c, b"complicated")) == 1:
        saw.add("one-diff")
    elif sorted(c) == sorted(b"complicated"):
        saw.add("anagram")  # transpose
assert saw >= {"shorter", "longer", "one-diff", "anagram"}, saw

# p_eff gates
fs = frozenset({b"common"})
assert p_eff(b"ab", 0.4, fs, 0.25) == 0.0
assert p_eff(b"the", 0.4, None, 0.25) == 0.2       # len 3 -> x0.5
assert p_eff(b"common", 0.4, fs, 0.25) == 0.1     # freq -> x0.25
assert p_eff(b"rareword", 0.4, fs, 0.25) == 0.4
print("test_typo: OK")
