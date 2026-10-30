"""edu5b 病因实验报告生成器(docs/50 §5;2026-10-30)。

读取训后评估产物(本机 pull 下来的),输出:
  1. 三 ckpt 的 MC 8 项 unit-space 主指标 acc_unit 表 + Δ vs p2pre / Δ vs CPT 对照;
  2. 预注册判据(docs/50 §5.2)的 H1/H2 判定;
  3. 内部协议关键量(held-out NLL / top5 / mode EM / snap T=10 bpb / NMI);
  4. 遗忘红线(旧分布 snap bpb 相对 p2 2.5390 退化 ≤ +0.02)。

用法(在本机仓库根目录,先把集群产物拉到 checkpoints/ 下):
  python scripts/report_edu5b.py \
      --mc checkpoints/mc_eval_p2pre.json,checkpoints/mc_eval_cpts7.json,checkpoints/mc_eval_edu5b.json \
      --accept checkpoints/word_p2pre_accept.log,checkpoints/word_cpts7_accept.log,checkpoints/word_edu5b_accept.log \
      --diag "checkpoints/edu5b_eval_edunll_*.log,checkpoints/edu5b_eval_oldnll_*.log"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

try:  # Windows 控制台默认 GBK,报告含 Δ/± 等字符会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TAGS = ["p2pre", "cpts7", "edu5b"]
P2_SNAP_BPB = 2.5390          # p2 acceptance log, protocol T=10, docs/39 #41
FORGET_REDLINE = 0.02         # docs/50 §5.2
# lm-eval 0.4.13 的 winogrande 是"续写型"任务(doc_to_target 返回续写文本,
# doc_to_choice 返回整句),mc_eval 的 2 选 1 gold 下标解析拿不到标签 → 三个
# 模型恒 0%,不参与均值(Δ 不受影响,因三者同为 0)。修复待办:gold 从
# doc["answer"] (1/2) 取,或改用该任务的 generate 口径。
EXCLUDE = {"winogrande"}

RE_NLL = re.compile(r"held-out NLL\s+(-?\d+(?:\.\d+)?)")
RE_TOP5 = re.compile(r"true string among decoded top-5 candidates:\s+([\d.]+)")
RE_EM = re.compile(r"^\s*argmax_pi\s+([\d.]+)", re.M)
RE_SNAP_T10 = re.compile(r"\*\s+T=10.*?bpb\s+([\d.]+)")
RE_NMI = re.compile(r"NMI\(h-clusters, types\)\s*=\s*([\d.]+)")


def parse_diag(path: str) -> dict:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    m = RE_NLL.search(txt)
    if m:
        out["nll"] = float(m.group(1))
    m = RE_TOP5.search(txt)
    if m:
        out["top5"] = float(m.group(1))
    m = RE_EM.search(txt)
    if m:
        out["mode_em"] = float(m.group(1))
    m = RE_NMI.search(txt)
    if m:
        out["nmi_h"] = float(m.group(1))
    m = RE_SNAP_T10.search(txt)
    if m:
        out["snap_bpb_t10"] = float(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True, help="逗号分隔的 mc_eval_<tag>.json(顺序 p2pre,cpts7,edu5b)")
    ap.add_argument("--accept", default="", help="逗号分隔的 accept log")
    ap.add_argument("--diag", default="", help="逗号/通配符分隔的 diag 日志(自动按 tag 归属)")
    args = ap.parse_args()

    mc = {}
    for p in args.mc.split(","):
        p = p.strip()
        if not p or not Path(p).exists():
            print(f"[warn] missing mc json: {p}")
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        mc[d["tag"]] = d
    tags = [t for t in TAGS if t in mc]
    if not tags:
        sys.exit("[err] no mc results found")

    tasks = list(mc[tags[0]]["results"].keys())
    print("\n## 1. MC 8 项 unit-space 主指标 acc_unit(docs/49)\n")
    hdr = "| task | n | " + " | ".join(tags) + " | d_edu5b-p2pre | d_cpts7-p2pre | d_edu5b-cpts7 |"
    print(hdr)
    print("|---|---|" + "---|" * len(tags) + "---|---|---|")
    deltas_edu, deltas_cpt, deltas_vs_cpt = [], [], []
    for t in tasks:
        row = [f"| {t} | {mc[tags[0]]['results'][t]['n']} "]
        vals = {}
        for tag in tags:
            v = mc[tag]["results"][t]["metrics"]["acc_unit"]
            vals[tag] = v
            row.append(f"| {v*100:.2f}% ")
        d_e = vals.get("edu5b", float("nan")) - vals.get("p2pre", float("nan"))
        d_c = vals.get("cpts7", float("nan")) - vals.get("p2pre", float("nan"))
        d_ec = vals.get("edu5b", float("nan")) - vals.get("cpts7", float("nan"))
        row += [f"| {d_e*100:+.2f}pp ", f"| {d_c*100:+.2f}pp ", f"| {d_ec*100:+.2f}pp "]
        row.append("| *invalid, excluded* " if t in EXCLUDE else "| ")
        if t not in EXCLUDE:
            deltas_edu.append(d_e)
            deltas_cpt.append(d_c)
            deltas_vs_cpt.append(d_ec)
        print("".join(row))

    n_valid = len(deltas_edu)
    print(f"\n### 汇总(有效 {n_valid} 项均值;{sorted(EXCLUDE)} 因 harness 任务格式不计)\n")
    for key in ["acc_unit", "acc_raw", "acc_char", "acc_unit_pmi"]:
        line = f"- {key}: " + " | ".join(
            f"{tag} {mc[tag]['summary'].get(key, float('nan'))*100:.2f}%" for tag in tags)
        print(line)
    import statistics
    d_edu = statistics.mean(deltas_edu) * 100
    d_cpt = statistics.mean(deltas_cpt) * 100
    d_ec = statistics.mean(deltas_vs_cpt) * 100
    print(f"\n**Δacc_unit 均值**: edu5b−p2pre = **{d_edu:+.2f}pp**; "
          f"cpts7−p2pre = {d_cpt:+.2f}pp; edu5b−cpts7 = {d_ec:+.2f}pp")

    print("\n### 预注册判据(docs/50 §5.2)\n")
    h2 = d_edu <= 1.0 and d_cpt <= 1.0
    h1 = d_edu >= 2.0 and (d_edu - d_cpt) >= 1.0
    if h2:
        print("- **H2 支持**(0.5B 尺度能力上限):Δ_edu ≤ +1.0pp 且 Δ_CPT ≤ +1.0pp")
    elif h1:
        print("- **H1 支持**(数据切换/混合动力学吃掉增益):Δ_edu ≥ +2.0pp 且 Δ_edu−Δ_CPT ≥ +1.0pp")
    else:
        print("- **不确定**:未触发两条预注册分支,报告逐任务表与置信区间,不下强结论")
    print(f"  (Δ_edu={d_edu:+.2f}pp, Δ_CPT={d_cpt:+.2f}pp, Δ_edu−Δ_CPT={d_edu-d_cpt:+.2f}pp)")

    if args.accept:
        print("\n## 2. 内部协议(docs/38 四段;run_accept)\n")
        print("| 指标 | " + " | ".join(TAGS) + " |")
        print("|---|" + "---|" * len(TAGS))
        acc = {}
        for tag, p in zip(TAGS, [x.strip() for x in args.accept.split(",")]):
            if p and Path(p).exists():
                acc[tag] = parse_diag(p)
            else:
                acc[tag] = {}
        keys = [("nll", "held-out NLL(旧分布)"), ("top5", "top5 覆盖"),
                ("mode_em", "argmax_pi EM"), ("snap_bpb_t10", "snap T=10 bpb"),
                ("nmi_h", "NMI(h)")]
        for k, label in keys:
            row = [f"| {label} "]
            for tag in TAGS:
                v = acc.get(tag, {}).get(k)
                row.append(f"| {v:.4f} " if isinstance(v, float) else "| — ")
            print("".join(row) + "|")
        bpb = acc.get("edu5b", {}).get("snap_bpb_t10")
        if bpb is not None:
            dd = bpb - P2_SNAP_BPB
            flag = "OK(红线内)" if dd <= FORGET_REDLINE else "**RED LINE BREACH**"
            print(f"\n- 遗忘红线: snap-500k T=10 bpb = **{bpb:.4f}** vs p2 {P2_SNAP_BPB:.4f} "
                  f"(Δ={dd:+.4f} ≤ +{FORGET_REDLINE}) → {flag}")

    if args.diag:
        print("\n## 3. held-out NLL(edu 分布 / 旧分布)\n")
        files = []
        for pat in args.diag.split(","):
            files += glob.glob(pat.strip())
        print("| log | NLL |")
        print("|---|---|")
        for f in sorted(files):
            d = parse_diag(f)
            if "nll" in d:
                print(f"| {Path(f).name} | {d['nll']:.3f} |")


if __name__ == "__main__":
    main()
