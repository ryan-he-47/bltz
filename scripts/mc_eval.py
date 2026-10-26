"""mc_eval.py — unit-space MC 评测驱动(docs/49,2026-10-27).

用 lm-eval 0.4.13 的任务定义/数据集/0-shot prompt 构造请求(prompt 与文献一致),
指标层由本脚本实现(docs/49 §4-§5):

  S_raw      = Σ_{候选项} log p(λ(u_i) | 前文)          # 适配器返回(raw nats)
  acc_unit   = argmax S_raw / n_units                     # ★ 主指标(项数归一)
  acc_raw    = argmax S_raw                              # 标准 acc
  acc_char   = argmax S_raw / n_chars                    # 对齐 lm-eval acc_norm(字符)
  acc_unit_pmi = argmax (S_raw − S_uncond)/n_units       # 可选去偏(标准 acc_mutual_info)

本机用法:
  python scripts/mc_eval.py --ckpt checkpoints/word_cpt_s_last.pt --tag cpt_s_pilot \
      --limit 100 --out-dir checkpoints
集群用法见 slurm/bltz_mc_eval.sbatch。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from lmeval_bltz import BltzEvalLM

TASKS_DEFAULT = ("arc_easy,arc_challenge,hellaswag,piqa,winogrande,"
                 "openbookqa,boolq,commonsense_qa")
DEV = "cuda"


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _argmax(xs: list[float]) -> int:
    """np.argmax 语义(并列取最小下标)。"""
    best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


def _gold_index(task, doc, choices: list[str]) -> int:
    gold = task.doc_to_target(doc)
    if isinstance(gold, list):
        return gold[0] if gold else -1
    if isinstance(gold, str):
        if gold.isdigit():
            return int(gold)
        if gold in choices:
            return choices.index(gold)
        return -1
    return int(gold)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    a, b = np.asarray(xs, float), np.asarray(ys, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stderr(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))


# --------------------------------------------------------------------------- #
# 无条件分数(PMI):批量前向,候选链短,按题内候选 batch
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batch_unconditional(lm: BltzEvalLM, chains: list[list[bytes]]) -> list[float]:
    """候选 unit 链在空 context(仅 BOS)下的 Σ log p(λ(u_i)|u_<i)。"""
    B = len(chains)
    lens = [len(c) for c in chains]
    S = max(lens) if lens else 0
    if S == 0:
        return [float("-inf")] * B
    L = lm.l_max
    ids = torch.zeros(B, S, L, dtype=torch.long, device=lm.dev)
    pm = torch.ones(B, S, L, dtype=torch.bool, device=lm.dev)
    for b, units in enumerate(chains):
        for i, u in enumerate(units):
            n = min(len(u), L)
            ids[b, i, :n] = torch.from_numpy(
                np.frombuffer(u[:n], dtype=np.uint8).astype(np.int64))
            pm[b, i, :n] = False
        # 填充位必须是"合法 unit"(不能全 mask:AE 编码器对全 pad 行会出 NaN,
        # 2026-10-27 实测 -> 污染整批;填该链最后一个真实 unit,其输出不参与求和)
        if len(units) < S:
            last = units[-1]
            n = min(len(last), L)
            for i in range(len(units), S):
                ids[b, i, :n] = torch.from_numpy(
                    np.frombuffer(last[:n], dtype=np.uint8).astype(np.int64))
                pm[b, i, :n] = False
    lam = lm.model.encode_units(ids, pm)                 # (B,S,D)
    x = torch.cat([lm.model.bos.expand(B, 1, -1), lam[:, :-1]], dim=1)
    h = lm.model.backbone(lm.model.adapter(x))           # (B,S,d_model)
    nll = lm.model.head.nll(h, lam)                      # (B,S)
    return [-float(nll[b, :lens[b]].sum()) for b in range(B)]


# --------------------------------------------------------------------------- #
# 单任务评测
# --------------------------------------------------------------------------- #
def eval_task(lm: BltzEvalLM, name: str, limit: int, pmi: bool,
              chunk: int, verbose: bool = True) -> dict:
    from lm_eval.tasks import get_task_dict

    t0 = time.time()
    task = get_task_dict([name])[name]
    task.build_all_requests(limit=(limit or None), rank=0, world_size=1)
    insts = task.instances
    n_inst = len(insts)

    # ---- 条件分数(Raw S_raw;适配器返回 raw 语义)------------------------
    lls: list[float] = []
    metas: list[dict] = []
    for s in range(0, n_inst, chunk):
        part = insts[s:s + chunk]
        res = lm.loglikelihood(part)
        lls.extend(r[0] for r in res)
        metas.extend(lm.last_meta)
        if verbose and (s + chunk) % (chunk * 20) == 0:
            done = min(s + chunk, n_inst)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  [{name}] {done}/{n_inst} req ({rate:.2f} req/s, "
                  f"ETA {((n_inst-done)/rate/60):.1f} min)", flush=True)

    # ---- 按 doc 分组(instances 是逐 doc 展平的)----------------------------
    groups: list[list[int]] = []
    seen: dict[int, int] = {}
    for k, inst in enumerate(insts):
        did = id(inst.doc)
        if did not in seen:
            seen[did] = len(groups)
            groups.append([])
        groups[seen[did]].append(k)

    items = []
    for g in groups:
        inst0 = insts[g[0]]
        doc = inst0.doc
        choices = list(task.doc_to_choice(doc))
        gold = _gold_index(task, doc, choices)
        raw = [lls[k] for k in g]
        n_units = [metas[k]["n_units"] for k in g]
        n_chars = [max(len(c), 1) for c in choices]
        n_bytes = [max(len(c.encode("utf-8")), 1) for c in choices]
        unit = [raw[i] / max(n_units[i], 1) for i in range(len(raw))]
        char = [raw[i] / n_chars[i] for i in range(len(raw))]
        items.append({
            "ctx": insts[g[0]].args[0],
            "choices": choices,
            "gold": gold,
            "raw": raw,
            "n_units": n_units,
            "n_chars": n_chars,
            "n_bytes": n_bytes,
            "unit": unit,
            "char": char,
        })

    # ---- PMI(可选,题内批处理)-------------------------------------------
    if pmi:
        for it in items:
            chains = [lm._split_units(it["ctx"], " " + c)[1] for c in it["choices"]]
            uncond = batch_unconditional(lm, chains)
            it["uncond"] = uncond
            it["pmi_unit"] = [(it["raw"][i] - uncond[i]) / max(it["n_units"][i], 1)
                              for i in range(len(it["raw"]))]

    # ---- 指标 -------------------------------------------------------------
    def acc_of(pred_fn):
        n_ok = 0
        for it in items:
            if it["gold"] < 0:
                continue
            preds = pred_fn(it)
            n_ok += int(preds == it["gold"])
        return n_ok / max(len(items), 1)

    metrics = {
        "acc_unit": acc_of(lambda it: _argmax(it["unit"])),
        "acc_raw": acc_of(lambda it: _argmax(it["raw"])),
        "acc_char": acc_of(lambda it: _argmax(it["char"])),
    }
    if pmi:
        metrics["acc_unit_pmi"] = acc_of(lambda it: _argmax(it["pmi_unit"]))

    n = len(items)
    stderrs = {k: _stderr(v, n) for k, v in metrics.items()}
    # 诊断: 长度相关 / 平局 / argmax==最短
    # 注意: 跨题汇总的 length-Score 相关会被"题目难度/候选整体长度"混淆,
    # 真正要查的是**题内**长度偏置 -> 用题内 centering 后的 pooled 相关。
    ties = 0
    eq_short = eq_long = 0
    xs_raw, ys_len = [], []
    xs_unit, ys_nu = [], []
    xs_unit_w, ys_nu_w = [], []
    xs_pmi_w, ys_pmi_w = [], []
    for it in items:
        for i in range(len(it["raw"])):
            xs_raw.append(it["n_chars"][i])
            ys_len.append(it["raw"][i])
            xs_unit.append(it["n_units"][i])
            ys_nu.append(it["unit"][i])
        nu = np.asarray(it["n_units"], float)
        su = np.asarray(it["unit"], float)
        xs_unit_w.extend((nu - nu.mean()).tolist())
        ys_nu_w.extend((su - su.mean()).tolist())
        if "pmi_unit" in it:
            pu = np.asarray(it["pmi_unit"], float)
            xs_pmi_w.extend((nu - nu.mean()).tolist())
            ys_pmi_w.extend((pu - pu.mean()).tolist())
        vals = it["unit"]
        m = max(vals)
        if sum(1 for v in vals if abs(v - m) < 1e-9) > 1:
            ties += 1
        pred_u = _argmax(vals)
        short_idx = int(np.argmin(it["n_chars"]))
        long_idx = int(np.argmax(it["n_chars"]))
        eq_short += int(pred_u == short_idx)
        eq_long += int(pred_u == long_idx)

    out = {
        "task": name,
        "n": n,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "stderr": {k: round(v, 6) for k, v in stderrs.items()},
        "diag": {
            "n_units_mean": round(float(np.mean([np.mean(it["n_units"]) for it in items])), 3),
            "pearson_raw_vs_n_chars": round(_pearson(xs_raw, ys_len), 4),
            "pearson_unit_vs_n_units": round(_pearson(xs_unit, ys_nu), 4),
            "pearson_unit_vs_n_units_within": round(_pearson(xs_unit_w, ys_nu_w), 4),
            "pearson_pmi_vs_n_units_within": (
                round(_pearson(xs_pmi_w, ys_pmi_w), 4) if xs_pmi_w else None),
            "argmax_eq_shortest": round(eq_short / max(n, 1), 4),
            "argmax_eq_longest": round(eq_long / max(n, 1), 4),
            "tie_rate": round(ties / max(n, 1), 4),
            "elapsed_s": round(time.time() - t0, 1),
        },
        "items": items,
    }
    print(f"[mc_eval] {name}: n={n} " + " ".join(
        f"{k}={v*100:.2f}%±{stderrs[k]*100:.2f}" for k, v in metrics.items())
        + f" | r_raw_len={out['diag']['pearson_raw_vs_n_chars']} "
          f"short={out['diag']['argmax_eq_shortest']*100:.1f}%", flush=True)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ae", default="checkpoints/ae_48_256_word_full/best.pt")
    ap.add_argument("--tok", default="data/qwen35_tokenizer")
    ap.add_argument("--tag", default="mc")
    ap.add_argument("--tasks", default=TASKS_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--device", default=DEV)
    ap.add_argument("--noverify", action="store_true")
    ap.add_argument("--no-pmi", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                    device=args.device, verify=not args.noverify)
    lm.collect_meta = True

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {}
    t0 = time.time()
    for t in tasks:
        results[t] = eval_task(lm, t, args.limit, pmi=not args.no_pmi,
                               chunk=args.chunk)

    # 汇总(主指标 acc_unit;其余口径并列展示)
    keys = ["acc_unit", "acc_raw", "acc_char"] + (
        [] if args.no_pmi else ["acc_unit_pmi"])
    summary = {k: round(float(np.mean([results[t]["metrics"][k] for t in tasks])), 6)
               for k in keys}
    payload = {
        "ckpt": args.ckpt,
        "tag": args.tag,
        "limit": args.limit,
        "tasks": tasks,
        "summary": summary,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    js = out_dir / f"mc_eval_{args.tag}.json"
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md = out_dir / f"mc_eval_{args.tag}.md"
    lines = [f"# MC eval (unit-space, docs/49)  tag={args.tag}  ckpt={args.ckpt}",
             "", f"limit={args.limit or 'full'}  elapsed={payload['elapsed_s']}s",
             "", "| task | n | acc_unit ★ | acc_raw | acc_char | acc_unit_pmi | "
             "r(raw,len) | argmax=shortest |", "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        r = results[t]
        mm = r["metrics"]
        lines.append(
            f"| {t} | {r['n']} | {mm['acc_unit']*100:.2f}% | {mm['acc_raw']*100:.2f}% | "
            f"{mm['acc_char']*100:.2f}% | "
            f"{mm.get('acc_unit_pmi', float('nan'))*100:.2f}% | "
            f"{r['diag']['pearson_raw_vs_n_chars']} | "
            f"{r['diag']['argmax_eq_shortest']*100:.1f}% |")
    lines += ["", "| summary | " + " | ".join(
        f"{k}={summary[k]*100:.2f}%" for k in keys) + " |", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n[mc_eval] SUMMARY " + " ".join(f"{k}={summary[k]*100:.2f}%"
                                            for k in keys), flush=True)
    print(f"[mc_eval] wrote {js} and {md}", flush=True)


if __name__ == "__main__":
    main()
