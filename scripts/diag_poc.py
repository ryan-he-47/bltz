"""POC qualitative diagnostic suite (docs/08 §4). NO pass/fail gates — every
section prints descriptive measurements; judgment is left to the reader
(D-1 verdict framework abolished 2026-09-13).

Sections: gen gcos edelta enc bound demb bemb (default: all)
Run:
  python scripts/diag_poc.py <ckpt.pt> [section ...] [--cache data/cache_dryrun]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# generated text may contain arbitrary bytes; the Windows console is GBK —
# reconfigure stdout/stderr to utf-8 with replacement instead of crashing.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.data import collate_sequences, tensorize_units
from bltz.infer import generate
from bltz.models import BltzLM
from bltz.objectives import mtp_targets
from bltz.segment import Segmenter
from bltz.shards import ShardReader

DEV = "cuda"
VIZ = Path("viz")
CKPT = sys.argv[1]
CACHE = "data/cache_dryrun"
SECTIONS = ["gen", "gcos", "gcos_ext", "edelta", "enc", "bound", "demb", "bemb"]
args = sys.argv[2:]
if "--cache" in args:
    i = args.index("--cache")
    CACHE = args[i + 1]
    del args[i : i + 2]
traj_extra: list[str] = []
if "gcos_traj" in args:
    i = args.index("gcos_traj")
    traj_extra = args[i + 1 :]
    args = args[: i + 1]
todo = args or SECTIONS


def load_model(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = Cfg(state["cfg"])
    model = BltzLM(mcfg)
    model.load_state_dict(state["model"])
    return model.to(DEV).eval(), mcfg, state


model, mcfg, state = load_model(CKPT)
seg = Segmenter(mcfg.segment.l_max, 0.0, 0.0)
print(f"model: step {state['step']}, params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


def char_class(b: int) -> str:
    if b in (32, 9, 10, 13):
        return "space"
    if 48 <= b <= 57:
        return "digit"
    if (65 <= b <= 90) or (97 <= b <= 122):
        return "letter"
    if b in (46, 44, 33, 63, 59, 58, 39, 34, 45, 40, 41):
        return "punct"
    return "other"


# ----------------------------------------------------------------------------
# gcos_ext: corrected gcos measurement per docs/11 (D-3 spec = ∇_h, not
# full-param). Red lines honored: old sec_gcos untouched; no verdict gates;
# >=10 batches with inter-batch range; nothing outside this script modified.

_GCOS_PAIRS = ((1, 2), (1, 3), (2, 3))


def _masked_task_loss(h: torch.Tensor, q: dict, w: torch.Tensor) -> torch.Tensor:
    """Weighted CE of MTP queries under an explicit weight map w (B, S-1, k_max).
    h: (B, S-1, d) backbone output (graph-carrying). Mirrors mtp_loss's query
    construction exactly (mtp_targets is the single source of truth)."""
    B, Sq, k_max = w.shape
    j_all = q["j_idx"].view(1, Sq, 1).expand(B, Sq, k_max).reshape(-1)
    d_all = q["delta"].view(1, 1, k_max).expand(B, Sq, k_max).reshape(-1)
    t_all = q["targets"].reshape(-1)
    b_all = torch.arange(B, device=DEV).view(B, 1, 1).expand(B, Sq, k_max).reshape(-1)
    sel = (w.reshape(-1) > 0).nonzero(as_tuple=True)[0]
    logits = model.head(h[b_all[sel], j_all[sel]], d_all[sel])
    ce = F.cross_entropy(logits.float(), t_all[sel], reduction="none")
    return (ce * w.reshape(-1)[sel]).sum() / w.sum().clamp_min(1e-8)


# module groups in model.parameters() registration order (encoder, backbone,
# head partition adapts to head_type: concat = delta_emb/mlp/out;
# film = delta_emb/body(in_proj+films+rest)/out — contiguous either way).
# Used to slice ONE full-param grad call per task into per-module vectors.
def _param_groups() -> list:
    groups = [
        ("encoder", list(model.encoder.parameters())),
        ("backbone", list(model.backbone.parameters())),
        ("head.delta_emb", list(model.head.delta_emb.parameters())),
    ]
    if hasattr(model.head, "mlp"):
        groups.append(("head.mlp", list(model.head.mlp.parameters())))
    else:
        groups.append(("head.body", list(model.head.in_proj.parameters())
                       + list(model.head.films.parameters()) + list(model.head.rest.parameters())))
    groups.append(("head.out", list(model.head.out.parameters())))
    return groups


def _group_names() -> list:
    return [n for n, _ in _param_groups()]


def _group_offsets(params: list) -> dict[str, tuple[int, int]]:
    offs: dict[str, tuple[int, int]] = {}
    acc = 0
    for name, ps in _param_groups():
        n = sum(p.numel() for p in ps)
        offs[name] = (acc, acc + n)
        acc += n
    assert acc == sum(p.numel() for p in params), "group partition mismatch"
    return offs


def _gcos_point(reader, mcfg, n_batches: int, seed: int, randomize: bool) -> dict[str, Any]:
    """One measurement point: per-k ∇_h cosines, per-module cosines, combined
    alignment, and (batch 0) the micro-step sub-loss change. randomize=True
    replaces byte content with IID random bytes (same unit lengths) = T4."""
    S = mcfg.data.n_patches
    l_max = mcfg.segment.l_max
    n, lam, k_max = mcfg.loss.n_patches_ahead, float(mcfg.loss.lam), mcfg.model.k_max
    n_seq = reader.n_sequences(S)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    out: dict[str, list] = {
        f"h_cos{a}{b}": [] for a, b in _GCOS_PAIRS
    }
    for gname in _group_names():
        for a, b in _GCOS_PAIRS:
            out[f"mod:{gname}:{a}{b}"] = []
    for kv in (1, 2, 3):
        out[f"comb_cos{kv}"] = []
    out["dl"] = []  # (dL1, dL2, dL3) micro-step records

    for bi in range(n_batches):
        idx = [rng.randrange(n_seq) for _ in range(4)]
        if not randomize:
            batch = reader.make_batch(idx, S, augment=False, p_split=0.0, p_merge=0.0, rng=rng)
        else:
            seqs = []
            for gi in idx:
                units = reader.sequence_units(gi, S)
                lens = torch.tensor([len(u) for u in units], dtype=torch.long)
                F_len = int(lens.sum())
                rand_flat = torch.from_numpy(
                    nrng.integers(0, 256, F_len, dtype=np.uint8).copy()
                ).long()
                seqs.append(tensorize_units(rand_flat, lens, l_max))
            batch = collate_sequences(seqs)
        batch = {k: v.to(DEV) for k, v in batch.items()}

        q = mtp_targets(batch, n, lam, k_max)
        h = model(batch["byte_ids"], batch["pad_mask"])[:, : S - 1]
        w_full = q["w"]
        w_k = {kv: w_full * (q["k"] == kv).float() for kv in (1, 2, 3)}
        loss_k = {kv: _masked_task_loss(h, q, w_k[kv]) for kv in (1, 2, 3)}
        loss_full = _masked_task_loss(h, q, w_full)

        # T1: hidden-state gradients
        gh = {}
        for kv in (1, 2, 3):
            gh[kv] = torch.autograd.grad(loss_k[kv], h, retain_graph=True)[0].reshape(-1).float()
        for a, b in _GCOS_PAIRS:
            out[f"h_cos{a}{b}"].append(F.cosine_similarity(gh[a], gh[b], dim=0).item())

        # T2: ONE full-param grad call per task; per-module cosines by slicing
        params = list(model.parameters())
        offs = _group_offsets(params)
        g_k_full: dict[int, torch.Tensor] = {}
        for kv in (1, 2, 3):
            g = torch.autograd.grad(loss_k[kv], params, retain_graph=True)
            g_k_full[kv] = torch.cat([x.reshape(-1).float() for x in g])
        for gname in _group_names():
            o0, o1 = offs[gname]
            for a, b in _GCOS_PAIRS:
                out[f"mod:{gname}:{a}{b}"].append(
                    F.cosine_similarity(g_k_full[a][o0:o1], g_k_full[b][o0:o1], dim=0).item()
                )

        # T3: combined-gradient alignment (true full loss, full params)
        g_comb = torch.autograd.grad(loss_full, params, retain_graph=True)
        g_comb = torch.cat([x.reshape(-1).float() for x in g_comb])
        for kv in (1, 2, 3):
            out[f"comb_cos{kv}"].append(F.cosine_similarity(g_comb, g_k_full[kv], dim=0).item())

        # T3b: micro-step on batch 0 — one combined step, per-k sub-loss change
        if bi == 0:
            eta = 4e-4
            with torch.no_grad():
                saved = [p.detach().clone() for p in params]
                ofs = 0
                for p in params:
                    n_ = p.numel()
                    p -= eta * g_comb[ofs : ofs + n_].view_as(p).to(p.dtype)
                    ofs += n_
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(mcfg.train.bf16)):
                    h2 = model(batch["byte_ids"], batch["pad_mask"])[:, : S - 1]
                    dl = tuple(
                        float(_masked_task_loss(h2, q, w_k[kv]).item() - loss_k[kv].item())
                        for kv in (1, 2, 3)
                    )
                for p, sp in zip(params, saved):
                    p.copy_(sp)
            out["dl"].append(dl)

        del batch, q, h, gh, g_k_full, g_comb
        torch.cuda.empty_cache()
    return out


def _gcos_report(tag: str, res: dict[str, Any]) -> None:
    def ms(v: list[float]) -> str:
        return f"{np.mean(v):+.3f} [{min(v):+.3f},{max(v):+.3f}]"

    print(f"\n--- {tag} ---")
    print("T1 ∇_h cosine (D-3 spec):")
    for a, b in _GCOS_PAIRS:
        print(f"  k{a}-k{b}: {ms(res[f'h_cos{a}{b}'])}")
    print("T2 per-module cosines:")
    for gname in _group_names():
        row = "  ".join(f"{a}{b}:{ms(res[f'mod:{gname}:{a}{b}'])}" for a, b in _GCOS_PAIRS)
        print(f"  {gname:15s} {row}")
    print("T3 combined-update alignment cos(g_combined, g_k):")
    for kv in (1, 2, 3):
        print(f"  k={kv}: {ms(res[f'comb_cos{kv}'])}")
    if res["dl"]:
        dl = np.array(res["dl"])
        print("T3b micro-step (eta=4e-4) mean dL_k (negative = improved):")
        print(f"  k=1 {dl[:,0].mean():+.4f}  k=2 {dl[:,1].mean():+.4f}  k=3 {dl[:,2].mean():+.4f}")


def sec_gcos_ext() -> None:
    """docs/11 T1-T4 on the loaded ckpt: ∇_h + per-module + combined alignment
    + random-byte control. >=10 batches, mean + inter-batch range."""
    v = torch.randn(4096)
    assert abs(F.cosine_similarity(v, v, dim=0).item() - 1.0) < 1e-5
    print("self-check cos(v,v)=1: OK")

    reader = ShardReader(CACHE)
    res = _gcos_point(reader, mcfg, n_batches=10, seed=0, randomize=False)
    _gcos_report("T1-T3 real data (10 batches)", res)
    res_r = _gcos_point(reader, mcfg, n_batches=10, seed=100, randomize=True)
    _gcos_report("T4 random-byte control (10 batches)", res_r)
    print("\nreading (descriptive, no gates): h_cos near/above 0 => trunk not torn; "
          "negative concentrated in head.* => readout geometry; random-control "
          "h_cos~0 => data-driven near/far structure, still-strong-neg => head artifact")


def sec_gcos_traj(extra_ckpts: list[str]) -> None:
    """docs/11 T5: ∇_h cosine trajectory across ckpts. NOTE: the 20k run predates
    the milestone feature — only {12000(frozen), 18920(best), 20000(last)} survive."""
    reader = ShardReader(CACHE)
    pts = [CKPT] + list(extra_ckpts)
    rows = []
    global model, state
    for p in pts:
        model, mcfg_t, state_t = load_model(p)
        globals()["mcfg"] = mcfg_t
        res = _gcos_point(reader, mcfg_t, n_batches=6, seed=0, randomize=False)
        rows.append((state_t["step"], res))
        print(f"point step={state_t['step']}: "
              + "  ".join(f"h{a}{b}={np.mean(res[f'h_cos{a}{b}']):+.3f}" for a, b in _GCOS_PAIRS))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, b in _GCOS_PAIRS:
        xs = [s for s, _ in rows]
        ys = [np.mean(r[f"h_cos{a}{b}"]) for _, r in rows]
        yerr = [np.std(r[f"h_cos{a}{b}"]) for _, r in rows]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=f"h_cos k{a}-k{b}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("∇_h cosine")
    ax.set_title("gcos trajectory (surviving points only; full traj needs next run's milestones)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / "gcos_traj.png", dpi=150)
    print("saved viz/gcos_traj.png")


# ----------------------------------------------------------------------------
def sec_gen() -> None:
    """Generation samples: D11 space-boundary stop (fallback sweep) vs legacy
    pure-entropy stop; emergent segmentation stats under the D11 rule."""
    prompts = [
        "The history of artificial intelligence began",
        "In a surprising turn of events,",
        "The quick brown fox",
    ]
    print("=== D11 rule: space-like boundary stop, entropy fallback sweep ===")
    for fb in (float("inf"), 3.5, 2.5):
        print(f"\n--- stop_mode=space, theta_fallback={fb} ---")
        for p in prompts:
            r = generate(model, p, seg, mcfg, max_commits=32, stop_mode="space",
                         theta_fallback=fb, temperature=0.0)
            text = r["text"].decode("utf-8", errors="replace")
            print(f"[{p[:30]!r}] spans={r['spans'][:12]}{'...' if len(r['spans'])>12 else ''}")
            print(f"  -> {text!r}")

    print("\n--- legacy: stop_mode=entropy, theta_stop=1.5 (comparison) ---")
    for p in prompts:
        r = generate(model, p, seg, mcfg, max_commits=32, stop_mode="entropy",
                     theta_stop=1.5, temperature=0.0)
        text = r["text"].decode("utf-8", errors="replace")
        print(f"[{p[:30]!r}] spans={r['spans'][:12]}{'...' if len(r['spans'])>12 else ''}")
        print(f"  -> {text!r}")

    # emergent segmentation: are commit boundaries word-aligned under D11?
    print("\n=== emergent segmentation stats (space mode, fallback=inf) ===")
    for p in prompts[:2]:
        r = generate(model, p, seg, mcfg, max_commits=32, stop_mode="space",
                     theta_fallback=float("inf"), temperature=0.0)
        cont = bytes(r["text"])[len(p):]
        spans = r["spans"]
        ends = np.cumsum(spans)
        ends = ends[ends < len(cont)]
        if len(ends) == 0:
            continue
        aligned = sum(1 for e in ends if e >= 1 and cont[e - 1] == 32)  # D11: span ENDS WITH the space
        base = sum(1 for i in range(len(cont)) if cont[i] == 32) / max(len(cont), 1)
        print(f"  [{p[:24]!r}] commits={len(spans)} aligned@space={aligned}/{len(ends)} "
              f"({aligned/len(ends):.2f}) base-rate={base:.2f} "
              f"span mean={np.mean(spans):.1f} median={np.median(spans):.0f}")


# ----------------------------------------------------------------------------
def sec_gcos() -> None:
    """Gradient cosine similarity between the k=1/2/3 future-patch subtasks."""
    reader = ShardReader(CACHE)
    S = mcfg.data.n_patches
    n_seq = reader.n_sequences(S)
    rng = random.Random(0)
    n, lam, k_max = mcfg.loss.n_patches_ahead, float(mcfg.loss.lam), mcfg.model.k_max

    def task_grad(batch, kv: int) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        B, Sq = batch["byte_ids"].shape[0], batch["byte_ids"].shape[1]
        h = model(batch["byte_ids"], batch["pad_mask"])[:, : Sq - 1]
        q = mtp_targets(batch, n, lam, k_max)
        w = q["w"] * (q["k"] == kv).float()
        b_all = torch.arange(B, device=DEV).view(B, 1, 1).expand(B, Sq - 1, k_max).reshape(-1)
        j_all = q["j_idx"].view(1, Sq - 1, 1).expand(B, Sq - 1, k_max).reshape(-1)
        d_all = q["delta"].view(1, 1, k_max).expand(B, Sq - 1, k_max).reshape(-1)
        sel = (w.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        h_q = h[b_all[sel], j_all[sel]]
        logits = model.head(h_q, d_all[sel])
        ce = F.cross_entropy(logits.float(), q["targets"].reshape(-1)[sel], reduction="none")
        loss = (ce * w.reshape(-1)[sel]).sum() / w.sum().clamp_min(1e-8)
        g = torch.autograd.grad(loss, list(model.parameters()))
        return torch.cat([x.reshape(-1).float() for x in g])

    cos_acc: dict[tuple[int, int], list[float]] = {(1, 2): [], (1, 3): [], (2, 3): []}
    for trial in range(3):
        idx = [rng.randrange(n_seq) for _ in range(4)]
        batch = {k: v.to(DEV) for k, v in reader.make_batch(idx, S, augment=False, p_split=0.0, p_merge=0.0, rng=rng).items()}
        g = {kv: task_grad(batch, kv) for kv in (1, 2, 3)}
        for a, b in cos_acc:
            c = F.cosine_similarity(g[a], g[b], dim=0).item()
            cos_acc[(a, b)].append(c)
        del batch, g
        torch.cuda.empty_cache()
    print("\ngradient cosine between subtasks (3 batches):")
    for (a, b), cs in cos_acc.items():
        print(f"  k={a} vs k={b}: {np.mean(cs):+.3f} (per-batch {['%+.3f' % c for c in cs]})")
    print("  reading: <0 = conflict, ~0 = orthogonal, >0 = aligned")


# ----------------------------------------------------------------------------
def sec_edelta() -> None:
    """Per-query CE stratified by word length / char class / position-in-word."""
    reader = ShardReader(CACHE)
    S = mcfg.data.n_patches
    n_seq = reader.n_sequences(S)
    rng = random.Random(1)
    n, lam, k_max = mcfg.loss.n_patches_ahead, float(mcfg.loss.lam), mcfg.model.k_max

    all_ce: list[np.ndarray] = []
    all_delta: list[np.ndarray] = []
    all_meta: list[np.ndarray] = []  # (class_id, word_len, pos_in_word, next_is_space)
    classes = ["space", "digit", "letter", "punct", "other"]

    with torch.no_grad():
        for gi in range(n_seq - 8, n_seq):
            units = reader.sequence_units(gi, S)
            raw = b"".join(units)
            meta = np.zeros((len(raw), 4), dtype=np.int64)
            wlen = np.zeros(len(raw), dtype=np.int64)
            # word metadata: split on space
            start = 0
            for i in range(len(raw) + 1):
                if i == len(raw) or raw[i] == 32:
                    L = i - start
                    for j in range(start, i):
                        wlen[j] = L
                        meta[j, 2] = 0 if j == start else (2 if j == i - 1 else 1)
                    start = i + 1
            for i in range(len(raw)):
                meta[i, 0] = classes.index(char_class(raw[i]))
                meta[i, 1] = min(wlen[i], 12)
                meta[i, 3] = 1 if i + 1 < len(raw) and raw[i + 1] == 32 else 0

            batch = {k: v.to(DEV) for k, v in reader.make_batch([gi], S, augment=False, p_split=0.0, p_merge=0.0, rng=rng).items()}
            q = mtp_targets(batch, n, lam, k_max)
            h = model(batch["byte_ids"], batch["pad_mask"])[:, : S - 1]
            h0 = h[0]  # (S-1, d); B=1 here so index with j only
            B, Sq = 1, S
            j_all = q["j_idx"].view(1, Sq - 1, 1).expand(B, Sq - 1, k_max).reshape(-1)
            d_all = q["delta"].view(1, 1, k_max).expand(B, Sq - 1, k_max).reshape(-1)
            t_all = q["targets"].reshape(-1)
            v_all = q["valid"].reshape(-1)
            ends = batch["ends"][0, : S - 1]
            tpos = (ends[:, None] + q["delta"].view(1, 1, k_max)[0]).reshape(-1)
            sel = v_all.nonzero(as_tuple=True)[0]
            for s in range(0, sel.numel(), 100000):
                ss = sel[s : s + 100000]
                logits = model.head(h0[j_all[ss]], d_all[ss])
                ce = F.cross_entropy(logits.float(), t_all[ss], reduction="none")
                pos = tpos[ss].cpu().numpy()
                ok = pos < len(raw)
                all_ce.append(ce.cpu().numpy()[ok])
                all_delta.append(d_all[ss].cpu().numpy()[ok] + 1)
                all_meta.append(meta[pos[ok]])

    ce = np.concatenate(all_ce)
    delta = np.concatenate(all_delta)
    meta = np.concatenate(all_meta)
    print(f"\nqueries: {len(ce)}, mean CE {ce.mean():.3f}")

    def mask_wlen(lo: int, hi: int) -> np.ndarray:
        return (meta[:, 1] >= lo) & (meta[:, 1] <= hi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for lo, hi, lab in [(1, 2, "1-2"), (3, 5, "3-5"), (6, 8, "6-8"), (9, 12, "9+")]:
        m = mask_wlen(lo, hi)
        curve = [ce[m & (delta == d)].mean() if (m & (delta == d)).any() else np.nan for d in range(1, 17)]
        ax.plot(range(1, 17), curve, marker=".", label=f"wordlen {lab} (n={m.sum()//1000}k)")
    ax.set_xlabel("delta")
    ax.set_ylabel("mean CE (nats)")
    ax.set_title("CE vs delta, by target word length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = np.arange(len(classes))
    ax.bar(xs, [ce[meta[:, 0] == i].mean() for i in xs])
    ax.set_xticks(xs, classes)
    ax.set_ylabel("mean CE")
    ax.set_title("CE by target byte class")
    ax.grid(alpha=0.3)

    ax = axes[2]
    labs = ["word-first", "word-mid", "word-last", "space-before-word"]
    masks = [
        (meta[:, 0] == classes.index("letter")) & (meta[:, 2] == 0),
        (meta[:, 0] == classes.index("letter")) & (meta[:, 2] == 1),
        (meta[:, 0] == classes.index("letter")) & (meta[:, 2] == 2),
        (meta[:, 0] == classes.index("space")),
    ]
    ax.bar(np.arange(4), [ce[m].mean() for m in masks])
    ax.set_xticks(np.arange(4), labs, rotation=15)
    ax.set_ylabel("mean CE")
    ax.set_title("CE by position-in-word")
    ax.grid(alpha=0.3)

    for a in axes:
        a.tick_params(labelsize=8)
    fig.suptitle(f"edelta: per-query CE stratification (step {state['step']})")
    fig.tight_layout()
    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / "poc_edelta.png", dpi=150)
    print("saved viz/poc_edelta.png")

    print("\nmean CE by class:", {c: round(float(ce[meta[:, 0] == i].mean()), 3) for i, c in enumerate(classes)})
    print("mean CE by pos-in-word:", {lab: round(float(ce[m].mean()), 3) for lab, m in zip(labs, masks)})
    print("mean CE next-byte-is-word-end(Δ=1):",
          round(float(ce[(delta == 1) & (meta[:, 3] == 1)].mean()), 3),
          "vs not:", round(float(ce[(delta == 1) & (meta[:, 3] == 0)].mean()), 3))


# ----------------------------------------------------------------------------
def sec_enc() -> None:
    """Encoder geometry: anisotropy, order usage, char/semantic 2x2 pairs."""
    words = (
        "the of and to in is was he for it with as his on be at by had not are but from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people mr how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three".split()
    )
    words = [w for w in dict.fromkeys(words) if 2 <= len(w) <= 12]

    def latent(bs: bytes) -> torch.Tensor:
        L = len(bs)
        byte_ids = torch.tensor([list(bs)], dtype=torch.long, device=DEV).reshape(1, 1, L)
        pad_mask = torch.zeros(1, 1, L, dtype=torch.bool, device=DEV)
        with torch.no_grad():
            return model.encoder(byte_ids, pad_mask)[0, 0].float()

    lat = torch.stack([latent(w.encode()) for w in words])
    lat_n = F.normalize(lat, dim=-1)
    sim = lat_n @ lat_n.T
    n = len(words)
    off = sim[~torch.eye(n, dtype=bool, device=DEV)]
    print(f"\nencoder anisotropy over {n} words: mean pairwise cos {off.mean():+.3f} "
          f"(isotropic gaussian ~0; cone => +)")
    X = (lat - lat.mean(0)).cpu().numpy()
    sv = np.linalg.svd(X, compute_uv=False)
    pr = sv.sum() ** 2 / (sv**2).sum()
    print(f"participation ratio: {pr:.1f} / {lat.shape[1]} dims (effective dimensionality)")
    # kNN clustering coefficient (k=5)
    k = 5
    knn = sim.fill_diagonal_(-2).topk(k, dim=1).indices.cpu().numpy()
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, knn[i]] = True
    adj |= adj.T
    cc = []
    for i in range(n):
        nb = np.nonzero(adj[i])[0]
        if len(nb) < 2:
            continue
        links = adj[np.ix_(nb, nb)].sum() / 2
        cc.append(links / (len(nb) * (len(nb) - 1) / 2))
    print(f"kNN(k=5) clustering coefficient: {np.mean(cc):.3f} (1=cliquey, 0=tree-like)")

    rng = random.Random(0)
    cos_order = []
    for w in words:
        if len(w) < 4:
            continue
        b = list(w.encode())
        sh = b[:]
        rng.shuffle(sh)
        cos_order.append(F.cosine_similarity(latent(bytes(b)), latent(bytes(sh)), dim=0).item())
    print(f"\norder usage: cos(word, shuffled) mean {np.mean(cos_order):.3f} +- {np.std(cos_order):.3f} "
          f"(~1 = bag-of-bytes/position-ignored; <<1 = position used)")

    groups = {
        "char-sim + sem-related": [("cat", "cats"), ("compute", "computer"), ("create", "creation"), ("music", "musical")],
        "char-sim + sem-unrelated": [("cat", "car"), ("dog", "dot"), ("tree", "three"), ("light", "night")],
        "char-far + sem-related": [("cat", "feline"), ("dog", "canine"), ("car", "vehicle"), ("happy", "joyful")],
        "char-far + sem-unrelated": [("cat", "banana"), ("dog", "microscope"), ("tree", "algebra"), ("happy", "granite")],
    }
    print("\n2x2 pair cosines:")
    for gname, pairs in groups.items():
        cs = [F.cosine_similarity(latent(a.encode()), latent(b.encode()), dim=0).item() for a, b in pairs]
        print(f"  {gname}: mean {np.mean(cs):+.3f}  {['%+.2f' % c for c in cs]}")


# ----------------------------------------------------------------------------
def sec_bound() -> None:
    """Prediction distributions: word-boundary vs mid-word (autoregressive cut)."""
    A = b"The quick brown"
    B = b"The quick brown fo"
    C = b"The quick brown fox"

    def dist(ctx: bytes, Q: int = 8):
        patches = [list(u) for u in seg.segment_bytes(ctx)]
        S = len(patches)
        l_max = mcfg.segment.l_max
        from bltz.data import PAD_ID
        byte_ids = torch.full((1, S, l_max), PAD_ID, dtype=torch.long)
        pad_mask = torch.ones(1, S, l_max, dtype=torch.bool)
        for j, pb in enumerate(patches):
            L = min(len(pb), l_max)
            byte_ids[0, j, :L] = torch.tensor(pb[:L], dtype=torch.long)
            pad_mask[0, j, :L] = False
        byte_ids, pad_mask = byte_ids.to(DEV), pad_mask.to(DEV)
        with torch.no_grad():
            h = model(byte_ids, pad_mask)[:, -1]
            d = torch.arange(Q, device=DEV)
            logits = model.head(h.expand(Q, -1), d).float()
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        H = -(p * logp).sum(-1)
        return p, H

    pa, Ha = dist(A)
    pb, Hb = dist(B)
    pc, Hc = dist(C)

    def show(name, p, H):
        print(f"  {name}: segments={[bytes(u) for u in seg.segment_bytes(name)]}")
        for d in range(4):
            top = p[d].topk(5)
            s = ", ".join(f"{repr(chr(b))}:{v:.2f}" for v, b in zip(top.values.tolist(), top.indices.tolist()))
            print(f"    d={d+1} H={H[d]:.2f} top5: {s}")

    print("\n[A] word boundary 'The quick brown':")
    show(A, pa, Ha)
    print("[B] mid-word 'The quick brown fo':")
    show(B, pb, Hb)
    print("[C] completed word 'The quick brown fox':")
    show(C, pc, Hc)
    kl = (pa[0] * (pa[0].log() - pb[0].log())).sum().item()
    print(f"\n  H(d=1): A(boundary) {Ha[0]:.2f}  B(mid-word) {Hb[0]:.2f}  C(after-word) {Hc[0]:.2f}")
    print(f"  KL(A_d1 || B_d1) = {kl:.3f} nats")


# ----------------------------------------------------------------------------
def sec_demb() -> None:
    """Delta-embedding geometry (the head's learned distance field)."""
    W = model.head.delta_emb.weight.detach().float()  # (k_max, d_delta)
    Wn = F.normalize(W, dim=-1)
    sim = (Wn @ Wn.T).cpu().numpy()
    k_max = W.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("cos(delta_emb i, j)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    dists, means = [], []
    for d in range(1, k_max):
        vals = [sim[i, i + d] for i in range(k_max - d)]
        dists.append(d)
        means.append(np.mean(vals))
    axes[1].plot(dists, means, marker=".")
    axes[1].set_xlabel("|i - j|")
    axes[1].set_ylabel("mean cosine")
    axes[1].set_title("delta-emb similarity vs distance")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"delta embedding geometry (step {state['step']})")
    fig.tight_layout()
    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / "poc_demb.png", dpi=150)
    print("saved viz/poc_demb.png")
    print(f"  adjacent cos(d,d+1) mean {np.mean([sim[i,i+1] for i in range(k_max-1)]):+.3f}; "
          f"cos(1, k_max) {sim[0, k_max-1]:+.3f}")


# ----------------------------------------------------------------------------
def sec_bemb() -> None:
    """Byte-embedding geometry (the only learned 'vocabulary')."""
    E = model.encoder.byte_emb.weight[:256].detach().float().cpu().numpy()
    labels = np.array([char_class(b) for b in range(256)])
    X = E - E.mean(0)
    u, sv, vh = np.linalg.svd(X, compute_uv=True)
    xy = X @ vh.T[:, :2]

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, color in [("letter", "tab:blue"), ("digit", "tab:orange"), ("space", "tab:green"), ("punct", "tab:red"), ("other", "gray")]:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.6, label=c, c=color)
    for b in [65, 97, 69, 101, 32, 46, 48, 57, 10]:
        ax.annotate(repr(chr(b)), (xy[b, 0], xy[b, 1]), fontsize=9)
    ax.legend()
    ax.set_title(f"byte embedding PCA (step {state['step']})")
    fig.tight_layout()
    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / "poc_bemb.png", dpi=150)
    print("saved viz/poc_bemb.png")

    En = torch.from_numpy(E)
    En = F.normalize(En, dim=-1)
    sim = En @ En.T
    for b in [97, 101, 116, 65, 48, 32, 46, 10]:
        top = sim[b].topk(9).indices.tolist()[1:9]
        print(f"  NN of {chr(b)!r}: {[chr(t) if 32 <= t < 127 else hex(t) for t in top]}")


# ----------------------------------------------------------------------------
secs = {
    "gen": sec_gen,
    "gcos": sec_gcos,
    "gcos_ext": sec_gcos_ext,
    "edelta": sec_edelta,
    "enc": sec_enc,
    "bound": sec_bound,
    "demb": sec_demb,
    "bemb": sec_bemb,
}
for name in todo:
    print(f"\n{'='*70}\n[{name}]\n{'='*70}")
    if name == "gcos_traj":
        sec_gcos_traj(traj_extra)
    else:
        secs[name]()
print("\ndiag_poc.py: DONE")
