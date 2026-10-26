"""Acceptance battery driver (docs/38 §1-§5) for a v2 word-arm ckpt.

Reproduces the five-section battery used for the p1/p2 0.5B milestone
acceptance (checkpoints/word_05b_p{1,2}_accept.log) so the CPT S-tier
acceptance log is directly comparable:

  1. diag_v2 backbone suite  : held-out NLL / top5 / 3-readout EM / pi stats / gen
  2. snap protocol-500k      : PPL/bpb vs temperature (forgetting monitor)
  3. pushforward byteCE      : table-free byte marginal bpb
  4. diag_v2_semantic        : NMI(h)/NMI(lambda) semantic structure
  5. gen_accept              : extended generation 8 prompts x 6 readouts

Eval set + protocol table are the OLD word cache (data/fineweb_word_local,
188.6M units) — the exact p2 acceptance scale; the CPT model sees it only as
a held-out forgetting probe (zero replay training).

  python scripts/run_accept.py <v2_ckpt> <ae_ckpt> --tag cpt_s [--sections 1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
AE_DEFAULT = "checkpoints/ae_48_256_word_full/best.pt"
PKL = "data/fineweb_word_local/eval_sample.pkl"
CACHE = "data/fineweb_word_local"


def section_cmd(n: int, ckpt: str, ae: str, tag: str) -> list[str]:
    if n == 1:
        return [PY, "-u", "scripts/diag_v2.py", ckpt, ae, "--units-pkl", PKL]
    if n == 2:
        return [PY, "-u", "scripts/snap_decode.py", ckpt, ae,
                "--units-pkl", PKL, "--cache", CACHE, "--topn", "500000",
                "--ppl", "--no-qwen"]
    if n == 3:
        return [PY, "-u", "scripts/pushforward_bpb.py", tag, ckpt, ae, "word"]
    if n == 4:
        return [PY, "-u", "scripts/diag_v2_semantic.py", ckpt, ae, "--units-pkl", PKL]
    if n == 5:
        return [PY, "-u", "scripts/gen_accept.py", ckpt, ae,
                "--steps", "48", "--topn", "500000",
                "--table-cache", "checkpoints/protocol_table_500k.pt"]
    raise ValueError(f"unknown section {n}")


TITLES = {
    1: "backbone suite (diag_v2: NLL/top5/readout EM/pi/gen)",
    2: "snap protocol-500k PPL/bpb (old-dist forgetting monitor)",
    3: "pushforward byteCE (table-free byte marginal bpb)",
    4: "semantic structure (NMI h vs lambda)",
    5: "extended generation (8 prompts x 6 readouts)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("ae", nargs="?", default=AE_DEFAULT)
    ap.add_argument("--tag", default="cpt_s")
    ap.add_argument("--sections", default="1,2,3,4,5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sections = [int(s) for s in args.sections.split(",") if s.strip()]
    out = Path(args.out or f"checkpoints/word_{args.tag}_accept.log")
    out.parent.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    with open(out, "w", encoding="utf-8", errors="replace") as f:
        def w(line: str) -> None:
            f.write(line + "\n")
            f.flush()

        w(f"# acceptance battery: ckpt={args.ckpt} ae={args.ae} tag={args.tag}")
        w(f"# sections={sections} protocol pkl={PKL} table cache={CACHE} (top-500k, no-qwen)")
        for n in sections:
            w(f"\n=== {n}. {TITLES[n]} {time.strftime('%H:%M:%S')} ===")
            t0 = time.time()
            cmd = section_cmd(n, args.ckpt, args.ae, args.tag)
            w(f"$ {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                w(line.rstrip("\n"))
            rc = proc.wait()
            w(f"[section {n}] exit={rc} elapsed={time.time() - t0:.0f}s")
            if rc != 0:
                w(f"[section {n}] FAILED — stopping battery")
                print(f"[run_accept] section {n} failed rc={rc}", flush=True)
                return
        w(f"\nWORD_{args.tag.upper()}_ACCEPT_DONE (total {time.time() - t_all:.0f}s)")
    print(f"[run_accept] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
