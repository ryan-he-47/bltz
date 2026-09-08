# 04 — Stage 0/1 冒烟报告(2026-09-08 晚)

> 范围:管线搭建 + 机制冒烟。**不含**全量数据缓存与 124M 正式训练(等用户放行)。

## 1. 已建管线

```
bytefield/
  config.py     YAML + --set a.b=value(浮点带小数点)
  segment.py    固定语义分割器(D3:词核+尾随归附+超长 UTF-8 安全细分+随机双增强)
  data.py       文本 → patch 序列 → batch 张量(byte_ids/pad_mask/flat/ends/patch_pos)
  models/
    encoder.py  Set Transformer(byte emb ⊕ 局部 PE → ISAB → PMA → latent)
    backbone.py Llama 式因果 transformer(RMSNorm/RoPE/SwiGLU,SDPA is_causal)
    head.py     条件解码头 (h, Δ) → 256 logits(Δ 永远相对偏移)
    model.py    ByteFieldLM 接线
  objectives.py MTP 损失(patch 间几何衰减 λ^(k−1),查询分块防显存爆)
  trainer.py    AdamW + WSD + bf16 autocast + spike guard(max(10×median, 2000))
  infer.py      熵停止变长生成循环(朴素 O(T²) 重算,KV cache 留 v2)
configs/        default.yaml(twin 目标) / smoke.yaml(微型)
tests/          test_segment.py / test_model.py(脚本式)
scripts/        smoke_overfit.py / smoke_tiny_train.py
```

## 2. 测试结果(全过)

- **test_segment.py**:roundtrip(拼接==原文)、UTF-8 安全、≤L_max、确定性、
  双增强有效性;样例:英文 `["Hello, ", "world!"]`、CJK `["你","好,","世","界"]`、
  缩写 `["don't ", "stop"]`;真实段落统计 avg 5.15B/patch(p95=11)。
- **test_model.py**:h 形状正确;**patch 级无泄漏**(改 patch j+1 的字节,h_j 不变;
  首个差异 patch 之前的 h 全等)——因果性硬指标;**Δ 条件化**(同 h 不同 Δ → 不同
  logits,排除 v6 静态扇式作弊);三模块梯度全通;随机初始化 loss 5.5469 ≈ ln 256。

## 3. 冒烟 1:单 batch 过拟合

loss **5.5516 → 0.0106**(300 步,18s)。MTP 损失实现正确。

## 4. 冒烟 2:真实数据短跑 + 推理循环

- 数据:流式取 FineWeb-Edu sample-10BT 前 1500 篇(7.25 MB,14s;香港直连)。
  建成 18704 条序列(S=64 patch)。
- 训练(5.66M 微型模型,300 步):loss 5.5475 → ~3.07(平台期符合预期——
  几百步只学到了边际字节统计);49 ms/step,峰值 VRAM 315 MiB,0 次 spike skip。
- 推理循环(θ_stop 扫描):机制全部正确——
  - θ ∈ {0.5, 1.0, 2.0}:首个 H(Δ)≈3.3 > θ → 每步只提交 1 字节(保底进展生效);
  - θ = ∞:每步顶满 m_max=16 字节提交;**θ_stop 作为速度/质量旋钮功能成立**。
  - 当前模型只输出 `t`/空格(欠训练的边际统计),属预期,非 bug。

## 5. 机制验证结论

D1(纯并行边际解码)、D4(patch 先验衰减 MTP)、D5(自身熵停止)的**机制**
全部按设计工作;H1-H5 的**能力**验证待 124M 规模。

## 6. 已知边界(v2 待办)

- 序列跨文档边界 + 纯因果掩码(v1 简化);v2 加文档边界 block-causal(BLT 对齐)。
- 推理为 O(T²) 朴素重算;v2 加 KV cache。
- 全量数据缓存(~30GB→字节 shard)未跑;`data.py` 目前只有内存批构建,
  需要加落盘缓存层供正式训练用。
- D-1 逐 Δ 诊断脚本(diag_delta.py)未写——Stage 1 正式跑时必备(当前冒烟已能
  从 infer.generate 的 entropies 输出看到逐 Δ 熵,接口已就绪)。

## 7. 下一步(等用户放行)

1. FineWeb-Edu 10B 全量下载 + 字节化缓存(detached 后台跑)。
2. 124M twin 臂正式训练(configs/default.yaml)+ D-1~D-6 诊断。
3. token 基线(GPT-2 tokenizer + 同骨干)对照。
