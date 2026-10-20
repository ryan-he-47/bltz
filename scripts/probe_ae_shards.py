"""AE reconstruction probe on cache shards (2026-10-20, docs/43 答疑:
DCLM/Edu 新语料上的 AE 编码质量直测)。

对每个给定的缓存池(shard 目录)抽样 N 个 unit,测 AE 重建:
teacher-forced CE / byte-acc / 贪婪解码 EM(复用 probe_ae_ood.eval_set)。
对照基线 = 旧 FineWeb 缓存(fineweb10b_v2_word)。

  python -u scripts/probe_ae_shards.py --ae <ae_ckpt> \
      --sets dclm:<dir1> edu:<dir2> oldweb:<dir3> --n 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from probe_ae_ood import eval_set

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sample_units(cache_dir: Path, n: int, seed: int = 20261020) -> list[bytes]:
    """从缓存池全部 shard 的并集均匀抽 n 个 unit(读 npy 数组, mmap)。"""
    pools: list[tuple[np.ndarray, np.ndarray]] = []
    total = 0
    for d in sorted(cache_dir.glob("shard-*")):
        if not (d / "meta.json").exists():
            continue
        lens = np.load(d / "unit_len.npy", mmap_mode="r")
        pools.append((d, lens))
        total += len(lens)
    rng = np.random.default_rng(seed)
    take = rng.choice(total, size=min(n, total), replace=False)
    units: list[bytes] = []
    base = 0
    for d, lens in pools:
        sel = take[(take >= base) & (take < base + len(lens))] - base
        if len(sel):
            raw = np.load(d / "bytes.npy", mmap_mode="r")
            offs = np.concatenate([[0], np.cumsum(lens[:])])
            units.extend(bytes(raw[offs[i]:offs[i + 1]]) for i in sel)
        base += len(lens)
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="label:cache_dir ...")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    state = torch.load(args.ae, map_location="cpu", weights_only=False)
    model = ByteStringAE(Cfg(state["cfg"])).to(DEV).eval()
    model.load_state_dict(state["model"])
    print(f"[probe] AE step {state.get('step')} l_max={model.l_max} dev={DEV}",
          flush=True)

    for spec in args.sets:
        label, cdir = spec.split(":", 1)
        units = sample_units(Path(cdir), args.n)
        r = eval_set(model, units, DEV)
        print(f"\n=== {label} ({cdir}) ===", flush=True)
        print(f"  n={r['n']}  mean_unit_bytes={r['mean_bytes']:.2f}  "
              f"CE {r['ce']:.4f}  bacc {r['bacc']:.4f}  greedyEM {r['em']:.4f}",
              flush=True)
        for orig, pred in r["fails"][:5]:
            print(f"  MISS orig={orig!r}", flush=True)
            print(f"       pred={pred!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
