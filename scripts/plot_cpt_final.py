"""Final CPT S-tier loss curve (2026-10-26, acceptance): full run vs p1/p2.

Absolute step axis: p1 [0,25431] -> p2 [20004,50863] (same FineWeb branch) ->
CPT [45429,96292] (NEW data DCLM 60% + Edu 40%, zero replay). Phase A =
plateau on cpt_cache_main (45429->91205); phase B = anneal tail on
cpt_cache_anneal (91205->96292). p2's old-data tail after the branch point is
kept as gray dashed reference.

Panel A: overlay of all runs (rolling-500 loss).
Panel B: CPT only, phase markers + LR schedule on a twin axis.

  python -u scripts/plot_cpt_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "word_cpt_s_final_vs_p12.png"
ROLL = 10  # 50-step cadence -> 500-step rolling window
BRANCH = 45429   # p2 ckpt_pre_decay (CPT branch point)
PHASE_B = 91205  # A->B data switch (anneal cache)
END = 96292      # planned absolute endpoint


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, loss, lr = [], [], []
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in r:
            steps.append(r["step"])
            loss.append(r["loss"])
            lr.append(float(r.get("lr", "nan")))
    return np.array(steps), np.array(loss), np.array(lr)


def roll(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.ones(ROLL) / ROLL, mode="valid")


def merge(s1, l1, r1, s2, l2, r2):
    """Concatenate two log segments, dedupe by step (first occurrence wins)."""
    seen: dict[int, int] = {}
    steps, loss, lr = [], [], []
    for s, l, r in zip(list(s1) + list(s2), list(l1) + list(l2), list(r1) + list(r2)):
        if int(s) in seen:
            continue
        seen[int(s)] = len(steps)
        steps.append(s)
        loss.append(l)
        lr.append(r)
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(loss)[order], np.array(lr)[order]


p1s, p1l, _ = load(ROOT / "checkpoints" / "word_05b_p1_train.log")
p2s, p2l, _ = load(ROOT / "checkpoints" / "word_05b_p2_train.log")
as_, al, _ = load(ROOT / "checkpoints" / "word_cpt_s_stable_train.log")
bs, bl, blr = load(ROOT / "checkpoints" / "word_cpt_s_anneal_train.log")
cs, cl, clr = merge(as_, al, np.zeros_like(al), bs, bl, blr)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9.2),
                              gridspec_kw={"height_ratios": [1.45, 1.0]})

# ---------------- panel A: overlay ----------------
def plot_run(a, steps, loss, color, label, alpha=0.18):
    a.plot(steps, loss, color=color, alpha=alpha, lw=0.6)
    a.plot(steps[ROLL - 1:], roll(loss), color=color, lw=1.8, label=label)


plot_run(ax, p1s, p1l, "tab:cyan", "p1 (FineWeb 5B units, 0-25.4k)")
plot_run(ax, p2s, p2l, "tab:blue", "p2 (FineWeb to 10B units, 20k-50.9k)")
m = p2s >= BRANCH
ax.plot(p2s[m][ROLL - 1:], roll(p2l[m]), color="gray", ls="--", lw=1.2,
        label="p2 tail on OLD data (superseded by CPT branch)")
ax.plot(p2s[m], p2l[m], color="gray", alpha=0.12, lw=0.6)
plot_run(ax, cs, cl, "tab:red", "CPT S (NEW data DCLM60+Edu40, 45429-96292)")

ax.axvline(20004, color="gray", ls=":", lw=1)
ax.axvline(BRANCH, color="tab:orange", ls=":", lw=1.4)
ax.axvline(PHASE_B, color="purple", ls=":", lw=1.2)
ax.text(BRANCH + 150, -46.5, "CPT data switch", color="tab:orange", fontsize=7.5,
        rotation=90, va="bottom")
ax.text(PHASE_B + 150, -46.5, "A->B anneal cache", color="purple", fontsize=7.5,
        rotation=90, va="bottom")
ax.set_xlim(0, END + 800)
yl = np.concatenate([p1l[p1s >= 2000], p2l[p2s >= 2000], cl[cs >= 2000]])
ax.set_ylim(yl.min() - 0.8, yl.max() + 1.2)
ax.set_xlabel("absolute step (global counter continues across resumes)")
ax.set_ylabel("MTP mixture loss (rolling 500 steps)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
n_new = int(cs[-1] - BRANCH)
ax.set_title(f"bltz word-arm 0.5B — p1/p2 vs CPT S-tier final (step {cs[-1]}, "
             f"+{n_new} new-data steps = {n_new * 196608 / 1e9:.1f}B units; "
             f"anneal 91205-96292)", fontsize=10)

# ---------------- panel B: CPT only + LR ----------------
ma = cs <= 91204
ax2.plot(cs[ma][ROLL - 1:], roll(cl[ma]), color="tab:red", lw=1.8,
         label="CPT phase A (plateau, cpt_cache_main)")
mb = cs >= 91205
ax2.plot(cs[mb][ROLL - 1:], roll(cl[mb]), color="tab:purple", lw=1.8,
         label="CPT phase B (anneal tail, cpt_cache_anneal)")
ax2.axvline(PHASE_B, color="purple", ls=":", lw=1.2)
ax2.set_xlabel("absolute step")
ax2.set_ylabel("MTP mixture loss (rolling 500)", color="tab:red")
ax2.grid(alpha=0.3)
lr_ok = np.isfinite(clr)
ax3 = ax2.twinx()
ax3.plot(cs[lr_ok], clr[lr_ok], color="gray", lw=1.0, ls="--", label="LR")
ax3.set_ylabel("learning rate (WSD)", color="gray")
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
ax2.legend(h2 + h3, l2 + l3, loc="upper right", fontsize=8)
ax2.set_title("CPT S-tier alone: plateau -> anneal (loss absolute level shifts with "
              "corpus entropy; not comparable across phases)", fontsize=10)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
print(f"p1 last {p1l[-1]:.3f} | p2 end {p2l[-1]:.3f} | "
      f"A end {al[-1]:.3f} (step {as_[-1]}) | B end {bl[-1]:.3f} (step {bs[-1]})")
print(f"A rolling10 tail {np.mean(al[-10:]):.3f} | B rolling10 tail {np.mean(bl[-10:]):.3f} | "
      f"CPT rolling min {roll(cl).min():.3f}")
