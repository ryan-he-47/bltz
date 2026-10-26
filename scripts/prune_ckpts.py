"""Scratch ckpt pruning manifest (2026-10-26, quota incident: bltz scratch at
276GB while the per-user scratch soft quota is 300GB and the SFT stage needs
~30GB more).

Policy (AGENTS/cluster discipline): deletions MUST run as a SLURM job (login
node is read-only); this script is DRY-RUN by default and only acts on an
explicit list — it never globs a directory to wipe.

  python scripts/prune_ckpts.py            # dry-run: print sizes + totals
  python scripts/prune_ckpts.py --apply    # actually delete (job only)

Keep-list rationale:
  * p2  last.pt / best.pt / ckpt_pre_decay.pt  -> old-dist eval baseline +
    S-tier branch point (docs/41 assets);
  * CPT stable last/best + s0070000 (kept as the edu5b matched-compute
    comparator, docs/50 §8), anneal last/best/pre_decay/s0095000 ->
    acceptance assets; CPT terminal weights = SFT init / M-tier candidate;
  * AE best.pt, all data caches (活资产), SFT data.
Delete-list rationale: rolling full-state ckpts of FINISHED runs (optimizer
state is useless once a run is done; weights of the same step exist as
last.pt/best.pt), plus milestones of finished runs whose numbers are already
in docs/39, plus the 2026-09 pre-CPT ablation runs (results documented).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRATCH = Path("/gpfs1/scratch/yihe47/bltz")
CK = SCRATCH / "ckpts"

# (path, reason)
DELETE_MANIFEST: list[tuple[str, str]] = [
    # --- 2026-09/10 ablation runs (results in docs/39 #18-#32; no further use)
    ("ckpts/v2_bpe_60k", "BPE arm closed 09-30; results documented"),
    ("ckpts/v2_word_1b_learnin2", "learnable-input arm closed 09-30"),
    ("ckpts/v2_word_long", "120M word-long arm superseded by p1/p2"),
    ("ckpts/v2_lin_2b", "linear-baseline arm closed"),
    ("ckpts/v2_word_1b", "word-1b predecessor of p1, superseded"),
    # --- p1 (superseded by p2/CPT; keep last.pt only)
    ("ckpts/word_05b_p1/ckpt_full.pt", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p1/ckpt_full.pt.1", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p1/ckpt_full.pt.2", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p1/best.pt", "p1 superseded; last.pt kept"),
    ("ckpts/word_05b_p1/ckpt_pre_decay.pt", "p1 branch point superseded by p2 pre_decay"),
    ("ckpts/word_05b_p1/ckpt_s0010000.pt", "finished-run milestone (docs/39 #39)"),
    ("ckpts/word_05b_p1/ckpt_s0020000.pt", "finished-run milestone (docs/39 #39)"),
    # --- p2 (keep last/best/pre_decay)
    ("ckpts/word_05b_p2/ckpt_full.pt", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p2/ckpt_full.pt.1", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p2/ckpt_full.pt.2", "finished run: rolling optimizer state"),
    ("ckpts/word_05b_p2/ckpt_s0025000.pt", "finished-run milestone (docs/39 #40)"),
    ("ckpts/word_05b_p2/ckpt_s0030000.pt", "finished-run milestone (docs/39 #40)"),
    ("ckpts/word_05b_p2/ckpt_s0035000.pt", "finished-run milestone (docs/39 #40)"),
    ("ckpts/word_05b_p2/ckpt_s0040000.pt", "finished-run milestone (docs/39 #40)"),
    ("ckpts/word_05b_p2/ckpt_s0045000.pt", "finished-run milestone (docs/39 #40)"),
    ("ckpts/word_05b_p2/ckpt_s0050000.pt", "finished-run milestone (docs/39 #40)"),
    # --- CPT S stable phase (keep last/best)
    ("ckpts/word_cpt_s_stable/ckpt_full.pt", "finished phase: rolling optimizer state"),
    ("ckpts/word_cpt_s_stable/ckpt_full.pt.1", "finished phase: rolling optimizer state"),
    ("ckpts/word_cpt_s_stable/ckpt_full.pt.2", "finished phase: rolling optimizer state"),
    ("ckpts/word_cpt_s_stable/ckpt_s0050000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0055000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0060000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0065000.pt", "finished-phase milestone"),
    # s0070000 KEPT: matched-compute comparator for the edu5b experiment
    # (docs/50 §8 item 3, +24,571 steps vs edu5b's +25,432); 1.93GB, do not delete.
    ("ckpts/word_cpt_s_stable/ckpt_s0075000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0080000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0085000.pt", "finished-phase milestone"),
    ("ckpts/word_cpt_s_stable/ckpt_s0090000.pt", "finished-phase milestone"),
    # --- CPT S anneal phase (keep last/best/pre_decay/s0095000 for the chain)
    ("ckpts/word_cpt_s_anneal/ckpt_full.pt", "finished run: rolling optimizer state"),
    ("ckpts/word_cpt_s_anneal/ckpt_full.pt.1", "finished run: rolling optimizer state"),
    ("ckpts/word_cpt_s_anneal/ckpt_full.pt.2", "finished run: rolling optimizer state"),
]


def size_of(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    apply = "--apply" in sys.argv
    total = 0
    print(f"[prune] mode={'APPLY' if apply else 'DRY-RUN'}")
    for rel, reason in DELETE_MANIFEST:
        p = SCRATCH / rel
        sz = size_of(p)
        if sz == 0:
            print(f"  MISSING/EMPTY {rel} (skip)")
            continue
        total += sz
        print(f"  {sz/1e9:7.2f}GB {rel}  # {reason}")
        if apply:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    print(f"[prune] reclaimable {total/1e9:.1f}GB "
          f"({total/1024**3:.1f}GiB)")
    if apply:
        print("[prune] deletion done; re-run dry-run to confirm")
    else:
        print("[prune] DRY-RUN only; submit a job with --apply after approval")


if __name__ == "__main__":
    main()
