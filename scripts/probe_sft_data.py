"""SFT data recon for smol-smoltalk (2026-10-26, docs/48 施工方案前置测量).

Measures, on a sample of the HF parquet shards:
  * schema + conversation structure (turns, roles);
  * rendered chat-template lengths in chars / Qwen BPE tokens / word-arm units
    (`segment_word`, the training-aligned segmenter) and assistant-target units;
  * truncation rate at S=4096 units;
  * marker tokenization (`<|user|>` / `<|assistant|>` / `<|end|>`) as units;
  * full-dataset extrapolation: total units -> steps/epoch at the CPT batch
    (196,608 units/step = 24 x 2 GPU x 4096).

CPU-only; runs on the tiny partition via slurm/bltz_sft_probe.sbatch.

  python scripts/probe_sft_data.py --data-dir DIR --tok DIR [--n 5000] [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from bltz.segment_v2 import segment_word

MARK_USER = b"<|user|>"
MARK_ASSIST = b"<|assistant|>"
MARK_END = b"<|end|>"
S = 4096
UNITS_PER_STEP = 196_608


def _turns(sample) -> list[tuple[str, str]]:
    """Normalize a smol-smoltalk row to [(role, content), ...]."""
    if "messages" in sample:
        msgs = sample["messages"]
        out = []
        for m in msgs:
            role = str(m.get("role", ""))
            content = m.get("content", "")
            if isinstance(content, list):  # rare multimodal rows
                content = " ".join(str(c.get("text", "")) for c in content
                                   if isinstance(c, dict))
            out.append((role, str(content)))
        return out
    conv = sample.get("conversations") or sample.get("conversation")
    out = []
    for m in conv:
        role = str(m.get("from", m.get("role", "")))
        content = m.get("value", m.get("content", ""))
        out.append((role, str(content)))
    return out


def render(turns: list[tuple[str, str]]) -> tuple[bytes, list[tuple[int, int, bool]]]:
    """Render to bytes; return text + list of (start, end, is_target) spans
    (assistant content + end marker are training targets)."""
    parts: list[bytes] = []
    spans: list[tuple[int, int, bool]] = []
    pos = 0
    for role, content in turns:
        role_l = role.lower()
        if role_l in ("user", "human"):
            parts.append(MARK_USER)
            pos += len(MARK_USER)
            b = content.encode("utf-8", errors="replace")
            parts.append(b)
            spans.append((pos, pos + len(b), False))
            pos += len(b)
        elif role_l in ("assistant", "gpt", "bot"):
            parts.append(MARK_ASSIST)
            pos += len(MARK_ASSIST)
            b = content.encode("utf-8", errors="replace")
            parts.append(b)
            spans.append((pos, pos + len(b), True))
            pos += len(b)
            parts.append(MARK_END)
            spans.append((pos, pos + len(MARK_END), True))
            pos += len(MARK_END)
        else:
            b = content.encode("utf-8", errors="replace")
            parts.append(b)
            pos += len(b)
    return b"".join(parts), spans


def unit_target_flags(units: list[bytes], spans: list[tuple[int, int, bool]]) -> np.ndarray:
    """Per-unit target flag by first-byte offset inside target spans."""
    flags = np.zeros(len(units), dtype=np.uint8)
    pos = 0
    for i, u in enumerate(units):
        start = pos
        pos += len(u)
        tgt = False
        for s0, s1, is_t in spans:
            if s0 <= start < s1:
                tgt = is_t
                break
        flags[i] = int(tgt)
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    print(f"[probe] {len(files)} parquet files")
    total_rows = 0
    for f in files:
        meta = pq.ParquetFile(f).metadata
        total_rows += meta.num_rows
        print(f"  {f.name}: rows={meta.num_rows} size={f.stat().st_size/1e6:.0f}MB")
    first = pq.ParquetFile(files[0])
    print(f"[probe] schema:\n{first.schema_arrow}")

    # ---- marker tokenization -------------------------------------------------
    print("\n[probe] marker units under segment_word:")
    for m in (MARK_USER, MARK_ASSIST, MARK_END):
        print(f"  {m!r} -> {segment_word(m, args.tok, l_max=32)}")

    # ---- sample pass ---------------------------------------------------------
    rows = []
    for f in files:
        pqf = pq.ParquetFile(f)
        for batch in pqf.iter_batches(batch_size=1024):
            rows.extend(batch.to_pylist())
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break
    rows = rows[: args.n]
    print(f"\n[probe] sampled {len(rows)} rows; first row keys: {sorted(rows[0])}")

    turns_hist: dict[int, int] = {}
    chars, tokens, units_n, tgt_units = [], [], [], []
    trunc = 0
    sample_show = 0
    for r in rows:
        t = _turns(r)
        if not t:
            continue
        turns_hist[len(t)] = turns_hist.get(len(t), 0) + 1
        text, spans = render(t)
        u = segment_word(text, args.tok, l_max=32)
        chars.append(len(text))
        units_n.append(len(u))
        tgt_units.append(int(unit_target_flags(u, spans).sum()))
        if len(u) > S:
            trunc += 1
        if sample_show < 2:
            print(f"\n[probe] sample {sample_show}: turns={len(t)} "
                  f"units={len(u)} target_units={tgt_units[-1]} "
                  f"chars={len(text)}")
            print(f"  head: {text[:200]!r}")
            print(f"  units[:12]: {u[:12]}")
            sample_show += 1

    n = len(units_n)
    units_a = np.array(units_n)
    tgt_a = np.array(tgt_units)
    ch_a = np.array(chars)
    print(f"\n[probe] n={n} rows")
    print(f"  chars: mean {ch_a.mean():.0f} p50 {np.percentile(ch_a,50):.0f} "
          f"p90 {np.percentile(ch_a,90):.0f} p99 {np.percentile(ch_a,99):.0f} max {ch_a.max()}")
    print(f"  units: mean {units_a.mean():.1f} p50 {np.percentile(units_a,50):.0f} "
          f"p90 {np.percentile(units_a,90):.0f} p99 {np.percentile(units_a,99):.0f} max {units_a.max()}")
    print(f"  target units: mean {tgt_a.mean():.1f} "
          f"(target share {tgt_a.sum()/units_a.sum():.1%})")
    print(f"  truncation >{S} units: {trunc}/{n} = {trunc/n:.2%}")
    print(f"  turns hist: {dict(sorted(turns_hist.items()))}")

    mean_units = float(units_a.mean())
    mean_tgt = float(tgt_a.mean())
    est_units = mean_units * total_rows
    out = {
        "files": [f.name for f in files],
        "total_rows": total_rows,
        "sampled": n,
        "marker_units": {m.decode(): [x.decode("utf-8", "replace") for x in
                                      segment_word(m, args.tok, l_max=32)]
                         for m in (MARK_USER, MARK_ASSIST, MARK_END)},
        "chars_mean": float(ch_a.mean()),
        "chars_p99": float(np.percentile(ch_a, 99)),
        "units_mean": mean_units,
        "units_p50": float(np.percentile(units_a, 50)),
        "units_p99": float(np.percentile(units_a, 99)),
        "units_max": int(units_a.max()),
        "target_units_mean": mean_tgt,
        "target_share": float(tgt_a.sum() / units_a.sum()),
        "trunc_rate_4096": float(trunc / n),
        "turns_hist": {str(k): v for k, v in sorted(turns_hist.items())},
        "est_total_units": est_units,
        "est_steps_per_epoch": est_units / UNITS_PER_STEP,
        "est_hours_per_epoch_11_3s": est_units / UNITS_PER_STEP * 11.3 / 3600,
    }
    print(f"\n[probe] extrapolation: {total_rows} convs x {mean_units:.1f} units "
          f"= {est_units/1e6:.1f}M units; steps/epoch {out['est_steps_per_epoch']:.0f}; "
          f"~{out['est_hours_per_epoch_11_3s']:.1f}h/epoch at 11.3s/step")
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
