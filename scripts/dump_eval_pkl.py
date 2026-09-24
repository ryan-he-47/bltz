"""Dump the eval-source pkl for a v2 cache (login-node friendly, CPU only).

Same sampler as diag_ae_v2.load_sources (freq_units / val_units / probe_pool /
distinct_seq_units) so every downstream diag (diag_v2 / snap_decode / semantic)
runs identically on any cache.

  python scripts/dump_eval_pkl.py <cache_dir> <out.pkl>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_ae_v2 import load_sources

src = load_sources(sys.argv[1])
with open(sys.argv[2], "wb") as f:
    pickle.dump(src, f)
print(f"dumped {sys.argv[2]}: {len(src['distinct_seq_units'])} distinct seqs, "
      f"{len(src['freq_units'])} freq units, total {src['total_units'] / 1e6:.1f}M units, "
      f"{src['n_seq']} sequences", flush=True)
