"""Assemble final CPT cache dirs via symlinks + unit-count report (docs/43 §2).

Runs INSIDE the cache-build SLURM job (10-18 用户纪律: 长任务不上登录节点).

Pools (built by build_cache_v2.py --set build.src_dir=... build.shard_offset=...):
  cpt_cache_dclm        shard-00000..  DCLM (text-only parquets)
  cpt_cache_edu         shard-00000..  FineWeb-Edu (id-deduped)
  cpt_cache_anneal_pool offsets per ANNEAL_POOL_MAP below.

Final dirs (symlinks only, renumbered shard-00000..):
  cpt_cache_main   = ALL dclm + ALL edu shards
                     (60:40 falls out of the 33GB/22GB text build)
  cpt_cache_anneal = all dolmino (web ~85%) + stack_edu (code ~10%) + finemath
                     (math ~5%) shards, chosen by meta.json n_units.

Prints per-source shard/unit counts and achieved ratios, then decodes 3 units
of sequence 0 from each final cache as a sanity peek.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (source, lo, hi): shard idx range [lo, hi) inside cpt_cache_anneal_pool.
# MUST match the build calls in slurm/bltz_cpt_cachebuild.sbatch.
ANNEAL_POOL_MAP = [
    ("dolmino_dclm", 0, 50),
    ("dolmino_pes2o", 50, 100),
    ("dolmino_wiki", 100, 150),
    ("dolmino_stackexchange", 150, 200),
    ("stack_edu_python", 200, 210),
    ("stack_edu_javascript", 210, 220),
    ("stack_edu_markdown", 220, 230),
    ("finemath", 230, 400),
]
ANNEAL_WEB = ("dolmino_dclm", "dolmino_pes2o", "dolmino_wiki", "dolmino_stackexchange")
ANNEAL_CODE = ("stack_edu_python", "stack_edu_javascript", "stack_edu_markdown")
ANNEAL_MATH = ("finemath",)
TARGET_WEB, TARGET_CODE, TARGET_MATH = 0.85, 0.10, 0.05


def log(msg: str) -> None:
    print(msg, flush=True)


def scan(pool: Path) -> list[tuple[int, Path, int]]:
    """-> [(shard_idx, dir, n_units)] sorted by idx."""
    out = []
    for d in sorted(pool.glob("shard-*")):
        meta = json.loads((d / "meta.json").read_text())
        out.append((int(d.name.split("-")[1]), d, int(meta["n_units"])))
    return sorted(out)


def link_all(final: Path, shards: list[tuple[int, Path, int]]) -> None:
    final.mkdir(parents=True, exist_ok=True)
    for old in final.glob("shard-*"):
        if old.is_symlink():
            old.unlink()
        else:
            raise RuntimeError(f"{old} is a real directory, refusing to touch")
    for i, (_, src, _) in enumerate(shards):
        os.symlink(os.path.realpath(src), final / f"shard-{i:05d}")


def report(name: str, groups: dict[str, list[tuple[int, Path, int]]]) -> int:
    total = 0
    for src, shards in groups.items():
        u = sum(u for _, _, u in shards)
        total += u
        log(f"  {src:24s} shards={len(shards):3d}  units={u/1e9:.3f}B")
    log(f"  {'TOTAL':24s} shards={sum(len(s) for s in groups.values()):3d}  "
        f"units={total/1e9:.3f}B")
    return total


def peek(cache: Path) -> None:
    from bltz.shards import ShardReader
    try:
        r = ShardReader(str(cache))
        units = r.sequence_units(0, 512)[:3]
        for u in units:
            log(f"  peek {cache.name}: {bytes(u)!r}")
    except Exception as e:  # noqa: BLE001 - sanity only
        log(f"  [peek] {cache.name} failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    data = Path(args.data)

    # ---- main mix: everything ---------------------------------------------
    log("== cpt_cache_main ==")
    main_groups = {
        "dclm": scan(data / "cpt_cache_dclm"),
        "fweb_edu": scan(data / "cpt_cache_edu"),
    }
    main_total = report("main", main_groups)
    link_all(data / "cpt_cache_main",
             main_groups["dclm"] + main_groups["fweb_edu"])

    # ---- anneal mix ---------------------------------------------------------
    log("\n== cpt_cache_anneal ==")
    pool = scan(data / "cpt_cache_anneal_pool")
    by_src: dict[str, list[tuple[int, Path, int]]] = {}
    for src, lo, hi in ANNEAL_POOL_MAP:
        by_src[src] = [(i, d, u) for i, d, u in pool if lo <= i < hi]
    web = [s for src in ANNEAL_WEB for s in by_src[src]]
    u_web = sum(u for _, _, u in web)
    total_t = u_web / TARGET_WEB
    code_t, math_t = TARGET_CODE * total_t, TARGET_MATH * total_t

    def pick(srcs: tuple[str, ...], target: float) -> list[tuple[int, Path, int]]:
        chosen, acc = [], 0
        for src in srcs:
            for s in by_src[src]:
                chosen.append(s)
                acc += s[2]
                if acc >= target:
                    return chosen
        return chosen

    code = pick(ANNEAL_CODE, code_t)
    math = pick(ANNEAL_MATH, math_t)
    anneal_groups = {"web": web, "code": code, "math": math}
    anneal_total = report("anneal", anneal_groups)
    log(f"  ratios: web {sum(u for _,_,u in web)/anneal_total:.3f} / "
        f"code {sum(u for _,_,u in code)/anneal_total:.3f} / "
        f"math {sum(u for _,_,u in math)/anneal_total:.3f} "
        f"(target {TARGET_WEB}/{TARGET_CODE}/{TARGET_MATH})")
    link_all(data / "cpt_cache_anneal", web + code + math)

    # ---- sanity peek ---------------------------------------------------------
    log("\n== peek ==")
    peek(data / "cpt_cache_main")
    peek(data / "cpt_cache_anneal")
    log("\n[done] caches assembled")


if __name__ == "__main__":
    sys.exit(main())
