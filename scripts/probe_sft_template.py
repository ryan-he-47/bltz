"""SFT template / marker probe (2026-10-26, docs/48 §2 前置验证).

Answers the open risks in docs/48 §2 with measurements on REAL smol-smoltalk
rows:
  1. how the proposed chat template segments under `segment_word` (marker units,
     cross-turn symbol merges such as `|>\\n<|`);
  2. whether the frozen word-AE reconstructs marker units as well as ordinary
     content units (verify flag rate; the AE is the judge);
  3. which turn separator (newline / space / none) minimises weird merges;
  4. text conservation (units re-join == rendered bytes) and target ratio.

CPU-only (AE forward on CPU is fine for a few hundred conversations); runs on
the tiny partition via slurm/bltz_sft_template.sbatch.

  python scripts/probe_sft_template.py --parquet-dir DIR --tok DIR --ae CKPT \
      --n 100 --out json
  python scripts/probe_sft_template.py --synthetic --tok DIR --ae CKPT --n 5
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.segment_v2 import segment_word
from bltz.verify import VerifyResplitter

MARK = {
    "system": b"<|system|>",
    "user": b"<|user|>",
    "assistant": b"<|assistant|>",
    "end": b"<|end|>",
}
SEPS = {"newline": b"\n", "space": b" ", "none": b""}


def _turns(row) -> list[tuple[str, str]]:
    out = []
    for m in row["messages"]:
        out.append((str(m.get("role", "")), str(m.get("content", ""))))
    return out


def render(turns: list[tuple[str, str]], sep: bytes) -> tuple[bytes, list[tuple[int, int, bool]]]:
    """Render with the docs/48 template; return bytes + (start,end,is_target) spans."""
    parts: list[bytes] = []
    spans: list[tuple[int, int, bool]] = []
    pos = 0
    for role, content in turns:
        role_l = role.lower()
        if role_l in MARK:
            m = MARK[role_l]
            parts.append(m)
            pos += len(m)
        b = content.encode("utf-8", errors="replace")
        parts.append(b)
        spans.append((pos, pos + len(b), role_l == "assistant"))
        pos += len(b)
        if role_l == "assistant":
            parts.append(MARK["end"])
            spans.append((pos, pos + len(MARK["end"]), True))
            pos += len(MARK["end"])
        parts.append(sep)
        pos += len(sep)
    return b"".join(parts), spans


def target_flags(units: list[bytes], spans) -> list[int]:
    flags, pos = [], 0
    for u in units:
        tgt = 0
        for s0, s1, is_t in spans:
            if s0 <= pos < s1:
                tgt = int(is_t)
                break
        flags.append(tgt)
        pos += len(u)
    return flags


def markerish(u: bytes) -> bool:
    return b"<|" in u or b"|>" in u or u in (b"user", b"assistant", b"system", b"end")


def cross_turn(u: bytes) -> bool:
    """A unit that contains BOTH an end-of-turn marker tail and a new-marker head."""
    return b"|>" in u and b"<|" in u


def synthetic_rows(n: int = 5):
    return [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize the following text: The quick brown fox."},
            {"role": "assistant", "content": "A fast fox jumps."},
            {"role": "user", "content": "Give me a haiku about it."},
            {"role": "assistant", "content": "Fox runs through the field\nsummer grass bends in its wake\nquick shadow, gone now"},
        ]},
        {"messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."},
        ]},
    ] * max(1, n // 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.synthetic:
        rows = synthetic_rows(args.n)
        print(f"[tmpl] SYNTHETIC rows={len(rows)}")
    else:
        files = sorted(Path(args.parquet_dir).glob("*.parquet"))
        rows = []
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(batch_size=256):
                rows.extend(b.to_pylist())
                if len(rows) >= args.n:
                    break
            if len(rows) >= args.n:
                break
        rows = rows[: args.n]
        print(f"[tmpl] real rows={len(rows)} from {len(files)} parquet")

    ae_state = __import__("torch").load(args.ae, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    ae.eval()
    ver = VerifyResplitter(ae, args.tok, ent_thresh=0.2, floor_bytes=3,
                           device="cpu", log_every=0)
    print(f"[tmpl] AE loaded (l_max={ae.l_max}), verify on CPU")

    results = {}
    for sep_name, sep in SEPS.items():
        units_all: list[bytes] = []
        n_target = n_units = 0
        n_convs = 0
        n_merge = 0
        roundtrip_ok = 0
        marker_units: collections.Counter = collections.Counter()
        for row in rows:
            turns = _turns(row)
            text, spans = render(turns, sep)
            units = segment_word(text, args.tok, l_max=32)
            if b"".join(units) != text:
                continue
            roundtrip_ok += 1
            n_convs += 1
            fl = target_flags(units, spans)
            n_units += len(units)
            n_target += sum(fl)
            for u in units:
                if markerish(u):
                    marker_units[u] += 1
                if cross_turn(u):
                    n_merge += 1
            units_all.extend(units)

        # AE verify: marker-ish vs content (one pass over all units, cached)
        em, ent = ver._verify(units_all)
        n_flag = int((~em).sum())
        flag_marker = flag_content = cnt_marker = cnt_content = 0
        flagged_marker_forms: collections.Counter = collections.Counter()
        for u, ok, e in zip(units_all, em, ent):
            bad = (not ok) or e > ver.ent_thresh
            if markerish(u):
                cnt_marker += 1
                flag_marker += int(bad)
                if bad:
                    flagged_marker_forms[u] += 1
            else:
                cnt_content += 1
                flag_content += int(bad)
        results[sep_name] = {
            "convs": n_convs,
            "roundtrip_ok": roundtrip_ok,
            "units": n_units,
            "target_units": n_target,
            "target_share": n_target / max(n_units, 1),
            "cross_turn_merges": n_merge,
            "cross_turn_merge_share": n_merge / max(n_units, 1),
            "ae_flag_all": n_flag / max(len(units_all), 1),
            "marker_units_total": cnt_marker,
            "marker_flag_rate": flag_marker / max(cnt_marker, 1),
            "content_units_total": cnt_content,
            "content_flag_rate": flag_content / max(cnt_content, 1),
            "marker_unit_top": {u.decode("utf-8", "replace"): c
                                for u, c in marker_units.most_common(15)},
            "flagged_marker_top": {u.decode("utf-8", "replace"): c
                                   for u, c in flagged_marker_forms.most_common(15)},
        }
        r = results[sep_name]
        print(f"\n[tmpl] sep={sep_name!r}: convs {r['convs']} units {r['units']} "
              f"target {r['target_share']:.1%} | cross-turn merges {r['cross_turn_merges']} "
              f"({r['cross_turn_merge_share']:.2%}) | AE flag all {r['ae_flag_all']:.2%} "
              f"(marker {r['marker_flag_rate']:.2%} of {r['marker_units_total']}, "
              f"content {r['content_flag_rate']:.2%})")
        print(f"  marker unit top: {list(r['marker_unit_top'].items())[:8]}")

    best = min(results, key=lambda k: (results[k]["cross_turn_merge_share"],
                                       results[k]["ae_flag_all"]))
    print(f"\n[tmpl] recommended separator: {best!r}")
    out = {"n_rows": len(rows), "variants": results, "recommended": best}
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"[tmpl] wrote {args.out}")


if __name__ == "__main__":
    main()
