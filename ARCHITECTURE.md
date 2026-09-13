# bltz — Architecture & Design Notes

> English-language entry point for external readers. The full technical archive
> (design decisions, POC diagnostics, literature review) lives under `docs/`
> (in Chinese). Status: **archived** (research concluded 2026-09-14).

## Motivation

Large-vocabulary tokenizers are a fixed tax: every new language, domain, or
fine-tuning setup has to pay it again. bltz (*byte-aware learnable tokenizer*)
asks whether a language model can keep the proven token-LM paradigm while
replacing its fixed I/O with **learnable byte-level I/O**:

- no vocabulary table, no learned tokenizer (BPE or entropy model) anywhere;
- the architecture must **not** depend on high-quality segmentation to work —
  segmentation is a prior that lightens the architecture's job, not a crutch.

## Architecture

```
bytes ──► Set-Transformer patch encoder ──► causal transformer backbone
        (variable-length patches,          (patch-level autoregression,
         unordered set encoding)            Llama-style: RMSNorm/RoPE/SwiGLU)
                                          │
                                          ▼
                     conditional neural-field head  (h, Δ) → byte
                     (one MLP conditioned on backbone state h and the
                      relative byte offset Δ; predicts 256-way byte logits)
```

1. **Deterministic semantic segmentation (D3)** — cheap rule-based word-ish
   patches (+ random split/merge augmentation), no learned segmenter.
2. **Local patch encoder (D2)** — Set Transformer sees only the current patch's
   bytes; all cross-patch context belongs to the backbone.
3. **Multi-future-byte training (MTP)** — for every patch position, the head is
   queried at every relative offset Δ covering up to 3 future patches (≤48
   bytes), with geometric per-patch decay λ ∈ {1, 0.5, 0.25}. Purely parallel
   marginal decoding (D1): no feedback of decoded bytes.
4. **Inference (D11)** — commit predicted bytes up to and including the first
   space-like byte (natural word-boundary stop); a high-entropy interrupt only
   serves as a fallback for pathological uncertainty. (Key insight: word
   boundaries are *low*-entropy — a BLT-style "stop when uncertain" rule can
   never land on them.)

## What the POC showed (137M params, 20k steps, 0.93B bytes; `docs/10`)

- The model learns: boundary knowledge (space predicted at 0.79 confidence at
  word ends; confident mid-word continuation), a smooth learned Δ distance
  field, and word-shaped commits under the D11 rule.
- Training was stable end-to-end on a single V100 (fp16).
- **Measurement-first lesson**: an apparent multi-task gradient conflict
  (full-parameter cosine −0.40/−0.62) dissolved under the corrected protocol
  (hidden-state gradients + module decomposition + combined-update alignment +
  random control + trajectory): the negative cosines live entirely in the
  Δ-readout layers — the trunk was never torn (`docs/10` §3.2b).

## Why it is archived (technical, not emotional)

A 2026-09 literature sweep (`docs/13`–`docs/16`) found all three headline
claims independently covered: vocab-free hierarchical byte I/O +
robustness/transferability (**HAT**, Aleph Alpha, up to 70B, weights
released) and parallel local byte decoding (**Fast BLT-D/DV**, Meta;
**DMBP**). What remains is formulation-level novelty (the (h,Δ) field head,
the boundary-stop + deterministic-segmentation combination, training-dynamics
diagnostics), which does not justify continued investment. The POC evidence
says the architecture *works* — the niche, not the idea, ran out.

## Repository layout

- `bltz/` — model, trainer, inference, cache layers (9 green tests in `tests/`)
- `scripts/` — training, cache builders, `diag_poc.py` (7-view qualitative
  diagnostic suite), `plot_train.py`
- `slurm/` — cluster job templates (cache build / bench / training / probes)
- `docs/` — full archive: design decisions D1–D11 (`docs/01`), code
  architecture & ops manual (`docs/07`), POC diagnostics (`docs/10`),
  literature verdict (`docs/13`–`docs/16`), closure notes
  (`docs/17`–`docs/19`) — **in Chinese**
- `reference_projects/` — paper text dumps (BLT / Fast BLT / HAT / DMBP)

## License

MIT (see `LICENSE`).
