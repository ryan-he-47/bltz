"""Download CPT data slices from HF to cluster scratch (docs/43 §1, 2026-10-18).

Phases (resumable via .done markers + per-source-file .progress.json,
heartbeats with rate/ETA):
  fw10bt_ids : dl FineWeb sample/10BT 000..013 (the exact files that built the
               OLD cache; build_cache_v2 N_FILES=14), extract `id` column into
               an int64-hash parquet, then DELETE the raw parquets
               (re-downloadable, not a live asset). [DONE 2026-10-18]
  dclm       : dl dclm-baseline-1.0 (ships as .jsonl.zst, NOT parquet) from
               global-shard_01_of_10, stream-decompress, keep only `text`,
               write one parquet per source file, until TARGET_TEXT_GB_DCLM.
  fweb_edu   : dl fineweb-edu parquets, DROP rows whose id-hash is in the
               fw10bt set (~9% expected), write filtered parquets until
               TARGET_TEXT_GB_EDU (post-filter text bytes).
  anneal     : finemath-4plus slice + stack-edu {Python,JavaScript,Markdown}
               slices (normalized to a `text` column) for the decay tail.
  sft        : smol-smoltalk full (kept as-is; SFT processing comes later).

Size targets are in *text bytes* (FineWeb calibration: 46GB text = 9.178B
units, ~5.0 B/unit): DCLM 33GB (~6.6B units, 60%), Edu 22GB kept (~4.4B
units, 40%) — headroom over the 10B-units goal included.

Run (cluster, SLURM job — never on the login node):
  python -u scripts/download_cpt_data.py --phase all --out $SCRATCH/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TARGET_TEXT_GB_DCLM = 33.0     # ~6.6B units (60% of 10B) + headroom
TARGET_TEXT_GB_EDU = 22.0      # ~4.4B units post-dedup (40%) + headroom
TARGET_GB_FINEMATH = 2.0
TARGET_GB_STACKEDU_PER_LANG = 1.0
STACK_EDU_LANGS = ["python", "javascript"]  # code_search_net configs
FW10BT_FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]  # 000..013
DCLM_REPO = "mlfoundations/dclm-baseline-1.0"
EDU_REPO = "HuggingFaceFW/fineweb-edu"
# 2026-10-18 两轮换源: HuggingFaceTB/stack-edu 只有 blob_id 元数据(正文在
# gated StarCoder2 blobs); bigcode/the-stack-smol 本身也 gated(BigCode 条款,
# 403 未授权) -> code-search-net/code_search_net(公开, parquet, per-language,
# 取 whole_func_string 列)。markdown 无对应语言,从配比中去掉(其散文性质与
# Dolmino web 重叠)。
STACK_REPO = "code-search-net/code_search_net"

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

    def beat(self, label: str, amount: int) -> None:
        self.done += amount
        el = max(time.time() - self.t0, 1e-9)
        rate = self.done / el / 1e6
        eta = (self.target - self.done) / max(self.done / el, 1e-9) / 60
        log(f"  {label} | cum {self.done/GB:.1f}/{self.target/GB:.0f}GB | "
            f"{rate:.0f} MB/s | ETA {eta:.0f} min")


def list_files(api, repo: str, prefix: str, suffix: str) -> list[str]:
    files = api.list_repo_files(repo, repo_type="dataset")
    out = sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))
    if not out:
        raise RuntimeError(f"{repo}: no {suffix} under {prefix!r}")
    return out


def dl(api, repo: str, fn: str, raw_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset",
                        local_dir=str(raw_dir))
    return Path(p)


def load_progress(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_progress(path: Path, prog: dict) -> None:
    path.write_text(json.dumps(prog))


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
    import io
    import zstandard as zstd
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst = out / "cpt_dclm"
    dst.mkdir(parents=True, exist_ok=True)
    prog_path = dst / ".progress.json"
    prog = load_progress(prog_path)
    target = TARGET_TEXT_GB_DCLM * GB
    done = sum(v["text_bytes"] for v in prog.values())
    rate = Rate(target)
    rate.done = done
    for fn in list_files(api, DCLM_REPO, "global-shard_01_of_10/", ".jsonl.zst"):
        if done >= target:
            break
        if fn in prog:
            continue
        p = dl(api, DCLM_REPO, fn, out / "raw" / "dclm")
        zsize = p.stat().st_size
        texts: list[str] = []
        dctx = zstd.ZstdDecompressor()
        with open(p, "rb") as fh, dctx.stream_reader(fh) as rdr:
            for line in io.TextIOWrapper(rdr, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec.get("text") or rec.get("content")
                if t:
                    texts.append(t)
        tb = sum(len(t.encode("utf-8")) for t in texts)
        out_path = dst / (Path(fn).name.replace(".jsonl.zst", "") + ".parquet")
        pq.write_table(pa.table({"text": pa.array(texts)}), out_path)
        p.unlink()
        prog[fn] = {"text_bytes": tb, "rows": len(texts)}
        save_progress(prog_path, prog)
        done += tb
        rate.beat(f"{Path(fn).name} (zst {zsize/1e6:.0f}MB, {len(texts)} docs)", tb)
    log(f"[dclm] total text {done/GB:.1f} GB in {dst}")


def phase_fweb_edu(api, out: Path, ids_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    dst = out / "cpt_fweb_edu"
    dst.mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / "fineweb-edu"
    prog_path = dst / ".progress.json"
    prog = load_progress(prog_path)
    seen = load_id_set(ids_path)
    log(f"[fweb_edu] loaded {len(seen)} seen id-hashes")
    target = TARGET_TEXT_GB_EDU * GB
    done = sum(v["text_bytes"] for v in prog.values())
    rate = Rate(target)
    rate.done = done
    n_in = n_kept = 0
    for fn in list_files(api, EDU_REPO, "data/", ".parquet"):
        if done >= target:
            break
        if fn in prog:
            continue
        p = dl(api, EDU_REPO, fn, raw)
        pf = pq.ParquetFile(p)
        out_path = dst / f"{len(prog):04d}_{Path(fn).name}"
        writer = None
        tb = 0
        rows_in = rows_kept = 0
        for rb in pf.iter_batches(batch_size=50_000):
            tbl = pa.Table.from_batches([rb])
            masks = [id_hash(s) not in seen for s in tbl.column("id").to_pylist()]
            rows_in += len(masks)
            filt = tbl.filter(pa.array(masks))
            rows_kept += filt.num_rows
            if filt.num_rows:
                tb += sum(len(t.encode("utf-8"))
                          for t in filt.column("text").to_pylist())
                if writer is None:
                    writer = pq.ParquetWriter(out_path, filt.schema)
                writer.write_table(filt)
        if writer is not None:
            writer.close()
        p.unlink()  # raw copy deleted; filtered copy stays
        n_in += rows_in
        n_kept += rows_kept
        prog[fn] = {"text_bytes": tb, "rows_in": rows_in, "rows_kept": rows_kept}
        save_progress(prog_path, prog)
        done += tb
        rate.beat(f"{Path(fn).name} (keep {rows_kept}/{rows_in})", tb)
    log(f"[fweb_edu] total text {done/GB:.1f} GB kept "
        f"({n_kept}/{n_in} rows = {n_kept/max(n_in,1):.3f}) in {dst}")


def _normalize_write(src: Path, dst: Path) -> int:
    """Rewrite parquet keeping only a `text` column (rename `content` if needed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src)
    cols = pf.schema_arrow.names
    col = next((c for c in ("text", "content", "whole_func_string",
                            "func_code_string") if c in cols), None)
    if col is None:
        raise RuntimeError(f"{src}: no usable text column in {cols}")
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


def _json_to_text_parquet(src: Path, dst: Path) -> int:
    """JSONL (one json object per line) -> parquet with a single `text` column."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    texts: list[str] = []
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("content") or rec.get("text")
            if t:
                texts.append(t)
    pq.write_table(pa.table({"text": pa.array(texts)}), dst)
    return len(texts)


def phase_anneal(api, out: Path) -> None:
    base = out / "cpt_anneal"
    # finemath-4plus (parquet shards)
    dst_dir = base / "finemath"
    dst_dir.mkdir(parents=True, exist_ok=True)
    done = sum(p.stat().st_size for p in dst_dir.glob("*.parquet"))
    target = TARGET_GB_FINEMATH * GB
    rate = Rate(target)
    rate.done = done
    for fn in list_files(api, "HuggingFaceTB/finemath", "finemath-4plus/", ".parquet"):
        if done >= target:
            break
        out_path = dst_dir / Path(fn).name
        if out_path.exists():
            continue
        p = dl(api, "HuggingFaceTB/finemath", fn, out / "raw" / "finemath")
        _normalize_write(p, out_path)
        size = p.stat().st_size
        p.unlink()
        done += out_path.stat().st_size
        rate.beat(fn, size)
    log(f"[anneal:finemath] total {done/GB:.1f} GB")
    # code_search_net (公开, per-language parquet, whole_func_string 列)
    for lang in STACK_EDU_LANGS:
        dst_dir = base / f"stack_edu_{lang}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / f"{lang}.parquet"
        if out_path.exists():
            continue
        p = dl(api, STACK_REPO, f"{lang}/train-00000-of-00001.parquet",
               out / "raw" / "code_search_net")
        n = _normalize_write(p, out_path)
        size = p.stat().st_size
        p.unlink()
        log(f"[anneal:stack_edu_{lang}] {n} docs, "
            f"{out_path.stat().st_size/1e6:.0f} MB (src {size/1e6:.0f} MB)")


def phase_dolmino(api, out: Path) -> None:
    """Dolmino high-quality web fill for the decay tail (10-18 拍板 §3.4):
    dclm/pes2o/wiki/stackexchange slices; math subdir skipped (FineMath covers
    math at the approved 5%), flan skipped (instruction format, SFT handles it).
    """
    base = out / "cpt_anneal"
    for prefix, sub in [("data/dclm/", "dclm"), ("data/pes2o/", "pes2o"),
                        ("data/wiki/", "wiki"), ("data/stackexchange/", "stackexchange")]:
        dst_dir = base / f"dolmino_{sub}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        done = sum(p.stat().st_size for p in dst_dir.glob("*.parquet"))
        target = 1.3 * GB
        rate = Rate(target)
        rate.done = done
        for fn in list_files(api, "allenai/dolmino-mix-1124", prefix, ".parquet"):
            if done >= target:
                break
            out_path = dst_dir / Path(fn).name
            if out_path.exists():
                continue
            p = dl(api, "allenai/dolmino-mix-1124", fn, out / "raw" / "dolmino")
            _normalize_write(p, out_path)
            size = p.stat().st_size
            p.unlink()
            done += out_path.stat().st_size
            rate.beat(fn, size)
        log(f"[dolmino:{sub}] total {done/GB:.1f} GB")


def phase_sft(api, out: Path) -> None:
    dst = out / "sft" / "smol-smoltalk"
    dst.mkdir(parents=True, exist_ok=True)
    files = list_files(api, "HuggingFaceTB/smol-smoltalk", "data/", ".parquet")
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
                    choices=["all", "fw10bt_ids", "dclm", "fweb_edu", "anneal",
                             "dolmino", "sft"])
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
    phases = ["fw10bt_ids", "dclm", "fweb_edu", "anneal", "dolmino", "sft"]
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
        elif ph == "dolmino":
            phase_dolmino(api, out)
        elif ph == "sft":
            phase_sft(api, out)
        marker.touch()
        log(f"===== {ph} done in {(time.time()-t0)/60:.0f} min =====")
    log("[all] download pipeline finished")


if __name__ == "__main__":
    sys.exit(main())
