"""AE OOD probe — anneal-mix gate (docs/43 §3.3, 2026-10-18).

Question: the AE (ae_48_256_word_full) was trained on FineWeb web text; can it
reconstruct MATH (LaTeX) and CODE text? If reconstruction collapses, mixing
math/code into the decay tail would feed the backbone OOD latents.

Sets:
  cache_val : real FineWeb units from eval_sample.pkl (in-distribution control)
  en_prose  : English prose (BLT paper excerpt) re-segmented by segment_word
  math      : synthetic LaTeX-heavy math text, re-segmented
  code      : synthetic code mix (Python/JS/C/shell), re-segmented

Per set: n_units, mean unit bytes, teacher-forced CE / byte-acc (batch_loss),
greedy-decode EM (exact string match), and a handful of failed strings.

  python -u scripts/probe_ae_ood.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.segment_v2 import segment_word
from train_ae import batch_loss, tensorize

DEV = "cuda" if torch.cuda.is_available() else "cpu"

MATH_TEXT = r"""
\section{Convergence of stochastic gradient descent}
Consider the objective $f(w) = \frac{1}{n} \sum_{i=1}^{n} f_i(w)$ where each
$f_i : \mathbb{R}^d \to \mathbb{R}$ is $L$-smooth and $\mu$-strongly convex.
The update rule $w_{t+1} = w_t - \eta_t \nabla f_{i_t}(w_t)$ with step size
$\eta_t = \frac{1}{\mu(t + t_0)}$ satisfies, for all $t \geq 1$,
\begin{equation}
  \mathbb{E}\left[ \| w_t - w^* \|^2 \right] \leq
  \frac{2 G^2}{\mu^2 (t + t_0)} + \frac{t_0}{t + t_0} \| w_0 - w^* \|^2 .
\end{equation}
\begin{proof}
Expanding the squared norm and taking conditional expectations gives
$\|w_{t+1} - w^*\|^2 = \|w_t - w^*\|^2 - 2\eta_t \langle \nabla f_{i_t}(w_t),
w_t - w^* \rangle + \eta_t^2 \|\nabla f_{i_t}(w_t)\|^2$.
Strong convexity yields $\langle \nabla f(w_t), w_t - w^* \rangle \geq
\mu \|w_t - w^*\|^2$, and the result follows by induction on $t$.
\end{proof}
The eigenvalues of the Hessian $\nabla^2 f(w) \in \mathbb{R}^{d \times d}$
satisfy $\mu I \preceq \nabla^2 f(w) \preceq L I$, i.e.
$\lambda_{\min}(\nabla^2 f) \geq \mu$ and $\lambda_{\max}(\nabla^2 f) \leq L$;
the condition number is $\kappa = L / \mu$.
For the quadratic case $f(w) = \frac{1}{2} w^\top A w - b^\top w$ with
$A = A^\top \succ 0$, the optimum solves $A w^* = b$, and Richardson's
iteration $w_{t+1} = w_t - \eta (A w_t - b)$ contracts as
$\|w_t - w^*\| \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|w_0 - w^*\|$
when $\eta = 2 / (\lambda_1 + \lambda_d)$.
Integrals and series behave as expected: $\int_0^\infty e^{-x^2} \, dx =
\frac{\sqrt{\pi}}{2}$, $\sum_{k=1}^{\infty} k^{-2} = \frac{\pi^2}{6}$, and
$\oint_{|z|=1} \frac{dz}{z} = 2 \pi i$ by the residue theorem.
A binomial coefficient identity: $\binom{n}{k} = \binom{n-1}{k-1} +
\binom{n-1}{k}$, with generating function $\sum_{k=0}^{n} \binom{n}{k} x^k =
(1 + x)^n$; differentiation under the integral sign gives
$\frac{d}{dt} \int_a^b g(x, t) \, dx = \int_a^b \frac{\partial g}{\partial t}
(x, t) \, dx$ under mild regularity assumptions.
Matrix calculus: if $F(X) = \log \det X$ for $X \succ 0$, then
$\nabla F(X) = X^{-1}$ and $\nabla^2 F(X)[H] = - X^{-1} H X^{-1}$.
The Kronecker product satisfies $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$,
and $\operatorname{vec}(AXB) = (B^\top \otimes A) \operatorname{vec}(X)$.
""".strip()

CODE_TEXT = r'''
import asyncio
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ShardReader:
    root: str
    shards: list = field(default_factory=list)

    def sequence_units(self, idx: int, n_patches: int = 512) -> list[bytes]:
        """Return the unit list for sequence `idx` (zero-copy when resident)."""
        seq = self._index[idx]
        return [u for u in seq.units[:n_patches] if u is not None]

def quicksort(xs):
    if len(xs) <= 1:
        return xs
    pivot = xs[len(xs) // 2]
    left = [x for x in xs if x < pivot]
    mid = [x for x in xs if x == pivot]
    right = [x for x in xs if x > pivot]
    return quicksort(left) + mid + quicksort(right)

async def fetch_all(urls, timeout=30):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(u, timeout=timeout)) for u in urls]
    return {u: t.result() for u, t in zip(urls, tasks)}

// JavaScript section --------------------------------------------------------
const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (!cache.has(key)) cache.set(key, fn(...args));
    return cache.get(key);
  };
};

export async function* streamLines(resp) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder("utf-8");
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n");
    buf = parts.pop() ?? "";
    for (const line of parts) yield line;
  }
}

/* C section ---------------------------------------------------------------- */
#include <stdint.h>
#include <string.h>

typedef struct node {
    int64_t key;
    struct node *left, *right;
} node_t;

node_t *bst_insert(node_t *root, int64_t key) {
    if (!root) {
        node_t *n = malloc(sizeof(node_t));
        n->key = key; n->left = n->right = NULL;
        return n;
    }
    if (key < root->key) root->left = bst_insert(root->left, key);
    else if (key > root->key) root->right = bst_insert(root->right, key);
    return root;
}

# shell section
for shard in data/shard-*.parquet; do
    echo "[build] $shard $(date +%T)"
    zstd -dc "$shard" | wc -l >> counts.log
done
'''.strip()


def as_bytes(u) -> bytes:
    return u if isinstance(u, (bytes, bytearray)) else str(u).encode("utf-8")


def eval_set(model, strings: list[bytes], dev: str) -> dict:
    """Teacher-forced CE/bacc + greedy EM + failed examples."""
    ce_t = bacc_t = 0.0
    nb = 0
    fails: list[tuple[bytes, bytes]] = []
    n_em = 0
    with torch.no_grad():
        for i in range(0, len(strings), 512):
            chunk = strings[i : i + 512]
            byte_ids, lens, pad_mask = tensorize(chunk, model.l_max, dev)
            ce, bacc, _em = batch_loss(model, byte_ids, lens, pad_mask)
            ce_t += float(ce)
            bacc_t += bacc
            nb += 1
            lam = model.encoder(byte_ids, pad_mask)
            for orig, pred in zip(chunk, model.decode(lam)):
                pb = bytes(pred)
                if pb == bytes(orig):
                    n_em += 1
                elif len(fails) < 12:
                    fails.append((bytes(orig), pb))
    n = len(strings)
    return {
        "n": n,
        "mean_bytes": sum(len(s) for s in strings) / max(n, 1),
        "ce": ce_t / max(nb, 1),
        "bacc": bacc_t / max(nb, 1),
        "em": n_em / max(n, 1),
        "fails": fails,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ae_48_256_word_full/best.pt")
    ap.add_argument("--pkl", default="data/fineweb_word_local/eval_sample.pkl")
    ap.add_argument("--prose", default="reference_projects/2412.09871v1.txt")
    ap.add_argument("--tok", default="data/qwen35_tokenizer")
    ap.add_argument("--max-units", type=int, default=800)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ByteStringAE(Cfg(state["cfg"])).to(DEV).eval()
    model.load_state_dict(state["model"])
    l_max = model.l_max
    print(f"[probe] AE ckpt {args.ckpt} step {state.get('step')} l_max={l_max} dev={DEV}",
          flush=True)

    with open(args.pkl, "rb") as f:
        src = pickle.load(f)
    sets: dict[str, list[bytes]] = {
        "cache_val": [as_bytes(u) for u in src["val_units"][: args.max_units]],
    }
    prose = Path(args.prose).read_text(encoding="utf-8", errors="replace")[4000:14000]
    for name, text in [("en_prose", prose), ("math", MATH_TEXT), ("code", CODE_TEXT)]:
        units = segment_word(text.encode("utf-8"), args.tok, l_max=l_max)
        sets[name] = [as_bytes(u) for u in units[: args.max_units]]
    for name, ss in sets.items():
        r = eval_set(model, ss, DEV)
        print(f"\n=== {name} ===", flush=True)
        print(f"  n={r['n']}  mean_unit_bytes={r['mean_bytes']:.2f}  "
              f"CE {r['ce']:.4f}  bacc {r['bacc']:.4f}  greedyEM {r['em']:.4f}",
              flush=True)
        for orig, pred in r["fails"][:6]:
            print(f"  MISS orig={orig!r}", flush=True)
            print(f"       pred={pred!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
