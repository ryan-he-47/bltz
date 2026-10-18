"""Download CPT data slices from HF to cluster scratch (docs/43 §1, 2026-10-18).

Phases (resumable via .done markers, per-file heartbeats with rate/ETA):
  fw10bt_ids : dl FineWeb sample/10BT 000..013 (the exact files that built the
               OLD cache; build_cache_v2 N_FILES=14), extract `id` column into
               an int64-hash parquet, then DELETE the raw parquets
               (re-downloadable, not a live asset).
  dclm       : dl dclm-baseline-1.0 parquets (sorted) until TARGET_GB_DCLM.
  fweb_edu   : dl fineweb-edu parquets, DROP rows whose id-hash is in the
               fw10bt set (~9% expected), write filtered parquets until
               TARGET_GB_EDU (post-filter bytes).
  anneal     : finemath-4plus slice + stack-edu {Python,JavaScript,Markdown}
               slices (normalized to a `text` column) for the decay tail.
  sft        : smol-smoltalk full (kept as-is; SFT processing comes later).

Run (cluster, SLURM job — never on the login node):
  python -u scripts/download_cpt_data.py --phase all --out $SCRATCH/data
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

# parquet GB targets (FineWeb calibration: 30.1GB parquet ~= 9.18B units)
TARGET_GB_DCLM = 22.0          # ~6B units (60% of 10B) + headroom
TARGET_GB_EDU = 15.0           # ~4B units post-dedup (40%) + headroom
TARGET_GB_FINEMATH = 2.0
TARGET_GB_STACKEDU_PER_LANG = 1.0
STACK_EDU_LANGS = ["Python", "JavaScript", "Markdown"]
FW10BT_FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]  # 000..013

GB = 1024 ** 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def id_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


class Rate:
    def __init__(self, target_bytes: float):
        self.t0 = time.time()
        self.done = 0
        self.target = target_bytes

    def beat(self, path: str, size: int) -> None:
        self.done += size
        el = max(time.time() - self.t0, 1e-9)
        rate = self.done / el / 1e6
        eta = (self.target - self.done) / max(self.done / el, 1e-9) / 60
        log(f"  {Path(path).name} +{size/GB:.2f}GB | cum {self.done/GB:.1f}/"
            f"{self.target/GB:.0f}GB | {rate:.0f} MB/s | ETA {eta:.0f} min")


def list_parquets(api, repo: str, prefix: str) -> list[str]:
    files = api.list_repo_files(repo, repo_type="dataset")
    out = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not out:
        raise RuntimeError(f"{repo}: no parquet under {prefix!r}")
    return out


def dl(api, repo: str, fn: str, raw_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset",
                        local_dir=str(raw_dir))
    return Path(p)


def phase_fw10bt_ids(api, out: Path) -> Path:
    import pyarrow.parquet as pq
    import pyarrow as pa
    ids_path = out / "cpt_ids" / "fw10bt_idhash.parquet"
    if ids_path.exists():
        log(f"[fw10bt_ids] already done: {ids_path}")
        return ids_path
    raw = out / "raw" / "fineweb"
    rate = Rate(30.1 * GB)
    hashes: list[int] = []
    for fn in FW10BT_FILES:
        p = dl(api, "HuggingFaceFW/fineweb", fn, raw)
        pf = pq.ParquetFile(p)
        for rb in pf.iter_batches(batch_size=100_000, columns=["id"]):
            hashes.extend(id_hash(s) for s in rb.column("id").to_pylist())
        size = p.stat().st_size
        p.unlink()  # delete raw parquet right after id extraction
        rate.beat(fn, size)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id_hash": pa.array(hashes, type=pa.int64())}), ids_path)
    log(f"[fw10bt_ids] {len(hashes)} ids -> {ids_path} "
        f"({ids_path.stat().st_size/1e6:.0f} MB)")
    return ids_path


def load_id_set(ids_path: Path) -> set[int]:
    import pyarrow.parquet as pq
    return set(pq.read_table(ids_path).column("id_hash").to_pylist())


def phase_dclm(api, out: Path) -> None:
    raw = out / "cpt_dclm"
    raw.mkdir(parents=True, exist_ok=True)
    done = sum(p.stat().st_size for p in raw.rglob("*.parquet"))
    target = TARGET_GB_DCLM * GB
    rate = Rate(target)
    rate.done = done
    for fn in list_parquets(api, "mlfoundations/dclm-baseline-1.0", "global-shard_"):
        if done >= target:
            break
        if (raw / fn).exists():
            continue
        p = dl(api, "mlfoundations/dclm-baseline-1.0", fn, raw)
        size = p.stat().st_size
        done += size
        rate.beat(fn, size)
    log(f"[dclm] total {done/GB:.1f} GB in {raw}")


def phase_fweb_edu(api, out: Path, ids_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst = out / "cpt_fweb_edu"
    dst.mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / "fineweb-edu"
    seen = load_id_set(ids_path)
    log(f"[fweb_edu] loaded {len(seen)} seen id-hashes")
    done = sum(p.stat().st_size for p in dst.rglob("*.parquet"))
    target = TARGET_GB_EDU * GB
    rate = Rate(target)
    rate.done = done
    n_in = n_kept = 0
    for fn in list_parquets(api, "HuggingFaceFW/fineweb-edu", "data/"):
        if done >= target:
            break
        rel = Path(fn).name
        out_path = dst / f"{len(list(dst.glob('*.parquet'))):04d}_{rel}"
        if out_path.exists():
            continue
        p = dl(api, "HuggingFaceFW/fineweb-edu", fn, raw)
        pf = pq.ParquetFile(p)
        writer = None
        for rb in pf.iter_batches(batch_size=50_000):
            tbl = pa.Table.from_batches([rb])
            masks = [id_hash(s) not in seen for s in tbl.column("id").to_pylist()]
            n_in += len(masks)
            filt = tbl.filter(pa.array(masks))
            n_kept += filt.num_rows
            if filt.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, filt.schema)
                writer.write_table(filt)
        if writer is not None:
            writer.close()
        size = p.stat().st_size
        p.unlink()  # raw copy deleted; filtered copy stays
        if out_path.exists():
            done += out_path.stat().st_size
        rate.beat(fn, size)
        log(f"  [fweb_edu] keep ratio so far {n_kept/max(n_in,1):.3f}")
    log(f"[fweb_edu] total {done/GB:.1f} GB (kept {n_kept}/{n_in} rows) in {dst}")


def _normalize_write(src: Path, dst: Path) -> int:
    """Rewrite parquet keeping only a `text` column (rename `content` if needed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src)
    cols = pf.schema_arrow.names
    col = "text" if "text" in cols else ("content" if "content" in cols else None)
    if col is None:
        raise RuntimeError(f"{src}: no text/content column in {cols}")
    writer = None
    n = 0
    for rb in pf.iter_batches(batch_size=50_000, columns=[col]):
        tbl = pa.Table.from_batches([rb]).rename_columns(["text"])
        if writer is None:
            writer = pq.ParquetWriter(dst, tbl.schema)
        writer.write_table(tbl)
        n += tbl.num_rows
    if writer is not None:
        writer.close()
    return n


def phase_anneal(api, out: Path) -> None:
    base = out / "cpt_anneal"
    for repo, prefix, target_gb, sub in [
        ("HuggingFaceTB/finemath", "finemath-4plus/", TARGET_GB_FINEMATH, "finemath"),
    ] + [
        ("HuggingFaceTB/stack-edu", f"{lang}/", TARGET_GB_STACKEDU_PER_LANG,
         f"stack_edu_{lang.lower()}")
        for lang in STACK_EDU_LANGS
    ]:
        dst_dir = base / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        done = sum(p.stat().st_size for p in dst_dir.glob("*.parquet"))
        target = target_gb * GB
        rate = Rate(target)
        rate.done = done
        for fn in list_parquets(api, repo, prefix):
            if done >= target:
                break
            out_path = dst_dir / Path(fn).name
            if out_path.exists():
                continue
            p = dl(api, repo, fn, out / "raw" / repo.split("/")[-1])
            _normalize_write(p, out_path)
            size = p.stat().st_size
            p.unlink()
            done += out_path.stat().st_size
            rate.beat(fn, size)
        log(f"[anneal:{sub}] total {done/GB:.1f} GB")


def phase_sft(api, out: Path) -> None:
    dst = out / "sft" / "smol-smoltalk"
    dst.mkdir(parents=True, exist_ok=True)
    files = list_parquets(api, "HuggingFaceTB/smol-smoltalk", "data/")
    total = 0
    rate = Rate(2.0 * GB)
    for fn in files:
        if (dst / fn).exists():
            continue
        p = dl(api, "HuggingFaceTB/smol-smoltalk", fn, dst)
        size = p.stat().st_size
        total += size
        rate.beat(fn, size)
    log(f"[sft] {total/GB:.1f} GB in {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "fw10bt_ids", "dclm", "fweb_edu", "anneal", "sft"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()
    # fail fast if compute node has no HF route
    api.repo_info("HuggingFaceFW/fineweb", repo_type="dataset")
    log("[net] HF reachable from this node")

    ids_path = out / "cpt_ids" / "fw10bt_idhash.parquet"
    phases = ["fw10bt_ids", "dclm", "fweb_edu", "anneal", "sft"]
    for ph in phases if args.phase == "all" else [args.phase]:
        marker = out / f".done_{ph}"
        if marker.exists():
            log(f"[skip] {ph} (done marker)")
            continue
        t0 = time.time()
        log(f"===== phase {ph} =====")
        if ph == "fw10bt_ids":
            ids_path = phase_fw10bt_ids(api, out)
        elif ph == "dclm":
            phase_dclm(api, out)
        elif ph == "fweb_edu":
            phase_fweb_edu(api, out, ids_path)
        elif ph == "anneal":
            phase_anneal(api, out)
        elif ph == "sft":
            phase_sft(api, out)
        marker.touch()
        log(f"===== {ph} done in {(time.time()-t0)/60:.0f} min =====")
    log("[all] download pipeline finished")


if __name__ == "__main__":
    sys.exit(main())
