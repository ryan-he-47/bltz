# bltz — Architecture & Design Notes

> English-language entry point for external readers. The full technical archive
> (design history, experiment chronicle, evaluation protocols) lives under
> `docs/` (in Chinese): current phase summary `docs/37`, evaluation bible
> `docs/38`, experiment index `docs/39`, near-term plan `docs/40`, code map
> `docs/07`. Status: **active** (v2 era, word-segmentation main line).

## Motivation

Large-vocabulary tokenizers are a fixed tax: every new language, domain, or
fine-tuning setup pays it again. bltz (*byte-aware learnable tokenizer*) asks
whether a language model can keep the proven token-LM paradigm while replacing
its fixed I/O with **learnable byte-level I/O** — no vocabulary table, no
learned tokenizer. The first generation (v1: Set-Transformer patch encoder +
conditional byte head, purely parallel marginal decoding) failed in principle
(conditioned-independent parallel marginals cannot express multimodal joints —
the "cat/dog/dot" ghost-word pathology). The current generation (v2) rebuilds
the I/O around an **autoencoder-shaped latent space** and **density modeling**.

## Architecture (v2, word-arm, current main line)

```
raw text ── segment_word ──> units (whole words / digit singles / symbol runs,
                              one trailing space merged, BPE fallback >32B,
                              fully deterministic)
         ── online verify+resplit (EM + entropy gate vs the frozen AE;
             failures BPE-resplit and re-verified, char-safe <=3B floor;
             the cache on disk is never touched)
units ── frozen ByteStringAE encoder ──> λ (48-dim) ── detach ──> MDN targets
      └─ LN + wide MLP adapter (48->2048->768) ──> backbone input
backbone: causal transformer (RMSNorm/RoPE/SwiGLU, 768x12 @120M / 1024x28 @0.5B)
head: MDN — GMM over λ (K=64, diagonal σ, linear head, σ_floor=0.05)
training: per-position GMM NLL (fp32 logsumexp)
```

Design pillars (docs/31):
1. **Reconstruction-shaped frozen target space** — prediction targets come from
   a frozen reconstruction-trained AE (word-arm: whole-word deterministic
   segmentation, trailing-space merged, digits alone); gradients never cross
   between AE and backbone (hard cut).
2. **Density fitting, not per-word classification** — the backbone fits a GMM
   over λ; components are density basis functions, not words (cooperative
   peaks: the mode may sit on no component center).
3. **Density-functional readout** — generation reads the density, never the
   bare component means: `mode` (global MAP by mean-shift), `modes-peak`
   (table-free temperature sampling over local maxima, peak-altitude weighted),
   `snap` (nearest-legal-unit reprojection against a corpus-frequency table,
   protocol table = top-500k units).

## Evaluation (docs/38)

Headline metric: **calibrated snap-PPL / bpb** (temperature-calibrated,
protocol table 500k, byte-normalized — the only cross-segmentation currency).
Sidecar: **pushforward byte CE** (table-free byte-level marginal CE through the
decoder). Generation quality is judged by eyeball (grammar skeleton /
attractor morphology / degeneration signatures) — single eval metrics never
override generation quality.

## Training infrastructure notes

- fp16 + GradScaler on V100 with `init_scale=1024` (sharpened density models
  overflow fp16 backward at the default 65536 — 2026-09-30 incident).
- Block-level shard rotation (resident 4 shards + background preload) — kind
  to shared-cluster IO; full preload variant exists for IO-fast clusters.
- Checkpoint discipline: rolling 3 full states + permanent milestones every
  10k steps + an explicit pre-decay plateau checkpoint + best.pt.
- Graceful interrupt (STOP file / SIGTERM) + wall-out auto-resume + spike
  guard (10x running median) + stall watchdog (500 consecutive skips).

## Repository map

- `bltz/` — library (segment_v2 / shards / verify / models / trainer).
- `scripts/` — training (train_ae, train_bltz_v2) + evaluation
  (snap_decode, pushforward_bpb, diag_* family).
- `tests/` — script-style tests, run individually (17, all green).
- `slurm/` — cluster job templates; `configs/` — YAML recipes.
- `docs/` — the technical archive (Chinese); `docs/37` is the entry point.

License: MIT. All technical decisions: @ryan-he-47; implementation: Kimi K3
@ Moonshot AI (via OhMyOpenCode orchestration).
