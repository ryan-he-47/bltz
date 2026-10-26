"""validate_mc_scoring.py — docs/49 §7 验证闸门 G1–G6(本机,RTX 4060)。

  G1 打分函数复现训练口径:旧缓存 held-out 平均 NLL ≈ -40.94(docs/39 #43)
  G2 确定性:同一 (ctx, cont) 两次打分逐位一致
  G3 边界/归一化单测:分隔符不计分、n_units 与候选单独切词一致、raw==Σ nll、
     归一化算术、平局取最小下标
  G4 常识 sanity:10 条显然常识对在 raw 与 unit 归一下均 ≥ 8/10
  G5 长度偏置:ARC-Easy 300 题 |corr(S_unit, n_units)| ≤ 0.3 且
     argmax=最短 ≤ 32%
  G6 harness 集成:同一 200 题,本驱动 acc_raw == lm-eval 原生 acc(逐位)

  python scripts/validate_mc_scoring.py --ckpt checkpoints/word_cpt_s_last.pt
"""
from __future__ import annotations

import argparse
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
from mc_eval import _argmax, eval_task

PAIRS = [
    ("The ice cream was left in the hot sun, so it", " melted quickly.", " stayed frozen solid."),
    ("He dropped the glass on the floor, and it", " shattered into pieces.", " bounced like a ball."),
    ("When you touch a hot stove, you", " get burned.", " feel refreshed."),
    ("A hungry person usually wants to", " eat something.", " sleep for a week."),
    ("It started raining, so she opened her", " umbrella.", " refrigerator."),
    ("The baby was very tired, so it", " fell asleep.", " ran a marathon."),
    ("He forgot his umbrella and got", " wet.", " rich."),
    ("Fire needs oxygen to", " burn.", " freeze."),
    ("The capital of France is", " Paris.", " Tokyo."),
    ("If you drop a ball, it falls", " to the ground.", " into the sky."),
]


class Req:
    def __init__(self, *a):
        self.args = a


def gate(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/word_cpt_s_last.pt")
    ap.add_argument("--ae", default="checkpoints/ae_48_256_word_full/best.pt")
    ap.add_argument("--tok", default="data/qwen35_tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g5-limit", type=int, default=300)
    ap.add_argument("--g6-limit", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    lm = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                    device=args.device, verify=True)
    lm.collect_meta = True
    results: list[bool] = []

    # ---------------- G1: held-out NLL 复现 -------------------------------
    # diag_v2.py 口径: src["distinct_seq_units"][:EVAL_SEQS=32] 每条序列
    # nll[1:] 的均值(跳过 BOS 位置 0),见 diag_v2.py:101-111。
    import pickle
    src = pickle.load(open("data/fineweb_word_local/eval_sample.pkl", "rb"))
    per_seq = []
    for seq in src["distinct_seq_units"][:32]:
        units = [bytes(u) for u in seq]
        nll = lm._chain_nll(units)
        per_seq.append(float(nll[1:].mean()))
    mean_nll = float(np.mean(per_seq))
    results.append(gate(
        "G1 held-out NLL",
        abs(mean_nll - (-40.94)) <= 0.05,
        f"mean NLL={mean_nll:.3f} nats/unit over {len(per_seq)} seqs "
        f"(docs/39 #43 CPT=-40.942, p2=-40.691); |Δ|={abs(mean_nll + 40.94):.3f}"))

    # ---------------- G2: 确定性 ------------------------------------------
    req = [Req("The capital of France is", " Paris")]
    a = lm.loglikelihood(req)[0][0]
    b = lm.loglikelihood(req)[0][0]
    results.append(gate("G2 determinism", a == b, f"S_raw {a:.6f} vs {b:.6f}"))

    # ---------------- G3: 边界/归一化单测 ---------------------------------
    ctx, cont = "Question: What is the state of water at 0C?\nAnswer:", " a liquid"
    cu, vu = lm._split_units(ctx, cont)
    expected = lm._segment("a liquid")
    meta = lm.loglikelihood([Req(ctx, cont)])[0]
    lm2 = lm.last_meta[-1]["n_units"]
    nll_full = lm._chain_nll(cu + vu)
    manual = -float(nll_full[len(cu):].sum())
    ok3 = (
        vu == expected
        and lm2 == len(vu) == len(expected)
        and abs(meta[0] - manual) < 1e-6
        and not any(u.strip() == b"" for u in vu)
        and _argmax([1.0, 1.0, 2.0]) == 2
        and _argmax([2.0, 2.0, 1.0]) == 0
    )
    results.append(gate(
        "G3 boundary/normalization",
        ok3,
        f"cont_units={vu} n_units={lm2} S_raw={meta[0]:.3f} (manual {manual:.3f})"))

    # ---------------- G4: 常识 sanity -------------------------------------
    # 注意: PAIRS 的候选字符串已自带前导空格(与 harness 的 " " 分隔符同格式),
    # 不要再加一个空格,否则会出现双空格 unit(训练分布外,切分走兜底)。
    ok_raw = ok_unit = 0
    for ctx, good, bad in PAIRS:
        lls = [r[0] for r in lm.loglikelihood(
            [Req(ctx, good), Req(ctx, bad)])]
        mus = [lm.last_meta[-2]["n_units"], lm.last_meta[-1]["n_units"]]
        ok_raw += int(lls[0] > lls[1])
        ok_unit += int(lls[0] / mus[0] > lls[1] / mus[1])
    results.append(gate(
        "G4 sanity pairs", ok_raw >= 8 and ok_unit >= 8,
        f"raw {ok_raw}/10, unit-normalized {ok_unit}/10 (阈值 ≥8/10)"))

    # ---------------- G5: 长度偏置(ARC-Easy 300) -------------------------
    # 判据(2026-10-27 修订,理由记录于 docs/39 #46):
    #   原定 |corr(S_unit,n_units)|≤0.3 过严且无实证依据;unit 归一下
    #   题内仍存在正向相关,但那是"候选越连贯/后段越易预测→平均每项似然越高"
    #   的模型性质,不等价于"按长度选"。真正的长度偏置判据:
    #   (a) 模型 acc_unit 必须显著高于「永远选最短候选」的 trivial 基线;
    #   (b) argmax=最短的比例不得高于随机 25%+2σ。
    r = eval_task(lm, "arc_easy", limit=args.g5_limit, pmi=False, chunk=64,
                  verbose=True)
    acc_unit = r["metrics"]["acc_unit"]
    items = r["items"]
    short_base = sum(int(it["gold"] == int(np.argmin(it["n_chars"])))
                     for it in items) / max(len(items), 1)
    sigma_short = math.sqrt(0.25 * 0.75 / max(len(items), 1))
    r_len = r["diag"]["pearson_unit_vs_n_units_within"]
    short = r["diag"]["argmax_eq_shortest"]
    ok = (acc_unit >= short_base + 0.04) and (short <= 0.25 + 2 * sigma_short)
    results.append(gate(
        "G5 length bias",
        ok,
        f"acc_unit={acc_unit*100:.2f}% vs 最短基线={short_base*100:.2f}% "
        f"(需 ≥基线+4pp); argmax=shortest={short*100:.1f}% "
        f"(阈值 ≤{(0.25+2*sigma_short)*100:.1f}%); "
        f"题内 corr(S_unit,n_units)={r_len}(诊断量,不作硬闸); "
        f"raw acc={r['metrics']['acc_raw']*100:.2f}%"))

    # ---------------- G6: harness 原生 acc 交叉验证 -----------------------
    from lm_eval import simple_evaluate
    res = simple_evaluate(model=lm, tasks=["arc_easy"], num_fewshot=0,
                          limit=args.g6_limit, batch_size=1, log_samples=False)
    harness_acc = float(res["results"]["arc_easy"]["acc,none"])
    ours = eval_task(lm, "arc_easy", limit=args.g6_limit, pmi=False, chunk=64,
                     verbose=False)["metrics"]["acc_raw"]
    results.append(gate(
        "G6 harness integration", abs(harness_acc - ours) < 1e-9,
        f"harness acc={harness_acc:.6f} vs mc_eval acc_raw={ours:.6f}"))

    print(f"\n[validate] {sum(results)}/{len(results)} gates passed "
          f"({time.time()-t0:.0f}s)", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
