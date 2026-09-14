# 19 — 变长训练打包调研(BLT / HAT / DMBP + 通用技术)

> 触发:2026-09-14,用户提出当前 patch 打包方式(固定 16 字节 patch、空位 padding、
> 每样本固定 512 patch)显存利用低效,想要"按字节长度打包 batch"或其它提高利用率
> 的方法,并点名调研 BLT / HAT / DMBP 的可参考代码。
> 结论摘要见 §5;本文只做调研与方案提案,**未改动任何代码**,待用户拍板。

---

## 1. 我们现状的准确定位(代码事实)

- **数据路径无 DataLoader**:`scripts/train_bltz.py:33-38` 随机取 `cfg.train.batch` 个全局序列号,
  `ShardReader.make_batch(idx, S=512, augment=True)`(`bltz/shards.py:178-205`)→
  `tensorize_units`(`bltz/data.py:29-58`)→ `collate_sequences`(`bltz/data.py:61-63`,纯 `stack`)。
- **固定画布**:`tensorize_units` 分配 `byte_ids (S, l_max)` 并用 `PAD_ID=256` 填满
  (`bltz/data.py:42`)、`pad_mask (S, l_max)` True=padding(`:44-45`)、`flat/patch_pos (S*l_max)`
  零填充(`:47-50`)。`S = data.n_patches = 512`,`L = segment.l_max = 16`(`configs/default.yaml:5,9`)。
- **patch 长度本来就是变的(1..16 字节)**:`l_max=16` 是**上限**不是固定值;
  `segment.py:42-58` 只负责把超长单元切成 ≤16。`configs/baseline.yaml:4` 自记平均 ≈ **5.2 字节/patch**。
  ⇒ **encoder 字节槽位里约 2/3 是 padding**(16/5.2 ≈ 3.1x)。
- **padding 只在字节层**:encoder 把 `pad_mask` 当 `key_padding_mask` 用(`bltz/models/encoder.py:108-112`),
  但**对 16 个槽位全部做前向**(`pos = arange(L)`,`:104`);backbone 只看 S=512 个 patch latent,
  纯因果、无 padding mask(`backbone.py:61`);loss 无 `ignore_index`,靠 `valid`/权重(`objectives.py:40-53`)
  与 `flat_len` 夹取(`:41`)。
- **算力量级**:encoder 输入 ≈ B·S·L = 8·512·16 = **65536** 槽位;backbone ≈ B·S = 4096。
  ⇒ **encoder 的序列长度是 backbone 的 16 倍,且其中 ~2/3 是 padding** —— 这才是主要浪费点。
- **已有可复用的长度张量**:`ends (B,S)`(各 patch 的排他式字节终点)、`flat_len (B,)`、
  `patch_pos (B,S*L)`、`pad_mask`。**per-patch `lens` 在 `tensorize_units` 内部被算出来后丢弃**,
  未进 batch dict(可由 `ends.diff()` 还原)。
- **文档边界**:磁盘上有 `unit_flag.npy`(doc-start 标志,`shards.py:5,63-64`),但训练**从未使用**
  (v1 简化:序列可跨文档,无边界 mask)。
- **缓存格式**:`bytes.npy + unit_len.npy + unit_flag.npy + meta.json`,**不存字节偏移**;
  偏移在读时通过稀疏 checkpoint 重建(`shards.py:107-135`)。
  ⇒ **reader/打包层面的改动不需要迁移缓存**;仅当改 `l_max` 才必须重建缓存。
- **耦合**:`k_max = n_patches_ahead(3) × l_max(16) = 48`(`configs/default.yaml:29-32`),
  同时是 head 的 Δ 嵌入表大小(`bltz/models/head.py:38,91`)。patch 上限一旦变化,`k_max` 必须同步。

**结论:用户说的"按字节长度打包"在本架构里的等价物不是"改变 512 这个数",而是
"把字节轴解开(padding 去掉)"——即把每行按真实 patch 长度拼成一条扁平字节流 +
每个 patch 的累积边界,让 encoder 只算真实字节。**

## 2. 三篇工作的代码可用性 + 可借鉴机制

### 2.1 BLT(`facebookresearch/blt`,**CC-BY-NC-4.0 非商用**,commit `9774ed4`)
- **数据路径**:`bytelatent/data/iterators/sequence_iterator.py` —— 把文档**拼接成一条连续字节流**,
  再按固定 **patch 数**切段(`output_seq_len`)⇒ 一行内可含多篇文档、一个文档也可跨行。
  `packing_iterator.py` —— 一个 batch 内**按 batch 最长字节动态 padding**(`tok_seq_len = max(len)-1`),
  再 `truncate_batch` 截到 `max_encoder_seq_length`(默认 12288 字节)或 `pad_to_max_length`。
  ⇒ **BLT 固定的是 patch 数,让字节长度变**;这正是"按字节打包"的一种形态。
- **注意力**:`bytelatent/model/utils.py` 的 `tokens_to_seqlen(tokens, eos_id)` 从 EOS 推出各文档长度,
  再 `xformers.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(...)`(local 模型另加
  `make_local_attention(sliding_window)`)⇒ **块对角因果,文档间不互相注意**。
  注意:`attn_impl="flex_attention"` 分支退化成纯 causal(不做文档隔离)——用 xformers 才有效。
- **位置**:RoPE 按行内 0..T-1 连续,**不在文档边界重置**。
- **loss**:`compute_loss(p, y, mask, scale)` 用 `mask` 加权归一化;无 `ignore_index`;
  padding/截断/不足一批的位置 mask=False。
- **已知坑**:issue #36「patcher 不支持 sequence packing」(patch 边界不感知文档边界)、
  issue #27(首 patch 恒长 1)、PR #65(mask/最后一批修复)。

### 2.2 HAT(arXiv 2501.10322,Aleph Alpha)——**与我们的字节编码器最像**
- **训练代码未发布**(已核:HF 上 `model.py` 里 `loss = None`,纯推理;`source/` 只是 logo;
  官方 `scaling` 训练框架仓库里也没有 HAT 模块)。
- **已发布代码**:
  1. **HF 模型仓库**(`Aleph-Alpha/tfree-hat-pretrained-7b-base` 等,Open Aleph License):
     `model.py` 的输入/批处理是**扁平字节 + 每词累积长度**:
     `input_ids` 形状 `[batch, total_bytes]`,`cumulative_seq_lengths_per_word`(int32 词边界),
     另带 `byte_position_ids` / `word_position_ids`;词内编码用 `flash_attn_varlen_func(cu_seqlens=词边界)`。
     **这就是我们要的形状**:`cumulative_seq_lengths_per_word` ≡ 我们的 per-patch 累积边界。
  2. **vLLM fork**(`Aleph-Alpha/vllm`,分支 `hat_v1`):批量推理实现,
     `vllm/v1/hat/hat_model_runner.py`(`_prepare_inputs`:把多个请求的 token 扁平拼接 + 每请求 cu_seqlens)、
     `hat_manager.py`(分阶段调度、chunked prefill、双 KV cache)、`hat_splitter.py`。
  3. **hat-splitter**(`Aleph-Alpha-Research/hat-splitter`,Rust + Python,Open Aleph License):
     分词规则(`split_with_limit`)。
- **论文里的训练批处理口径**(§4,原文):"batches of **1024 × 16,384 bytes**, packing together
  documents of varying lengths with appropriate **attention mask reset**" ⇒ 文档打包 + 边界重置注意力。
  后续 2603.15953:SFT/DPO 用 packed(约 256 序列/batch);预训练 3,500 词/≤28,000 字节,
  长上下文 32,768 词/≤262,144 字节。
- **对我们的价值**:HAT 的「词 = 一段变长字节,编码成一个 latent」与我们的
  「patch = 一段变长字节,编码成一个 latent」**完全同构**。它的扁平字节 + 词累积边界 + varlen 注意力
  就是 §5 方案 T1 的直接参考实现。

### 2.3 DMBP / FlexiTokens(arXiv 2608.15454;代码 `skai-research/lca-multibyte`,**无 LICENSE 文件**)
- 代码**已发布**(1 commit,2026-08-18);FlexiTokens 亦有 `skai-research/flexitokens`(**同样无 LICENSE**)。
- **打包方式对我们的启示有限**:预训练是经典 `group_texts` —— 拼接后按固定 **4096 字节**切块、
  **无 padding**、drop 余数;batch 32 + grad accum 4。**变长分段发生在模型内部**(边界预测器),
  不在 batch 布局层。
- 唯一可借鉴:`src/model/shortening.py` 的 `downsample/upsample` —— 用 `scatter_add` 把变长分段压到
  `(B, S_max, D)`(S_max = 批内最大分段数),再 `einsum` 单热矩阵把分段向量**复制回**各字节位置
  (DMBP 论文称之为 "duplicated upsampled inputs");LCA mask 为 `prev_group_self`。
- SFT 用 `dynamic_padding_data_collator`(按批内最长右 padding);推理循环**只支持 batch=1**。

## 3. 通用技术菜单(不限于这三篇)

| 方法 | 代表实现 | 机制 | 代价 / 注意 |
|---|---|---|---|
| BOS 对齐 best-fit | `karpathy/nanochat` `dataloader.py` `BestFit` | 文档首尾对齐进固定 T,100% 利用率 | **不改注意力**;代价是裁剪 ~35% token |
| 拼接(wrapped) | TRL `pack_dataset` 的 `_pack_wrapped` | 直接首尾相接 | 跨文档污染(无 mask) |
| FFD 装箱 | `imoneoi/multipack`、Axolotl `multipack` sampler | 一维装箱,利用率 >99%(vs interleaved ~75%) | 需配 `cu_seqlens` + varlen 注意力 |
| BFD 段树 | TRL `_pack_bfd`(来自 "Fewer Truncations",2404.10830) | 装箱 + 不跨文档截断 | 同上 |
| 块对角 mask | BLT `BlockDiagonalCausalMask.from_seqlens`、xformers `BlockDiagonalMask` | 扁平拼接 + 块对角,无 padding 且不污染 | 需重写注意力调用 |
| position_ids 重置 | HF `DataCollatorWithFlattening`(`separator_id`)+ `prepare_fa2_from_position_ids` | 由 position_ids 反推 `cu_seqlens`,padding-free | 模型需支持 position_ids |
| 显式 packed 数据集 | torchtune `PackedDataset`(`input_pos` 重置 + block causal mask)、mosaicml `llm-foundry` `BinPackCollator` | 生产级打包 + 文档 mask | 依赖各自训练栈 |
| 动态批/序列打包 | NVIDIA NeMo-RL `SequencePackingArgs`(FFD) | 2–3x 吞吐 | 需 loss wrapper |
| 论文证据 | Kosec 2021 "Packing: Towards 2x NLP BERT Acceleration"(2107.02027) | 消 padding,~2x 加速 | 评测集 padding 可达 50–89% |

## 4. 硬件红线(V100 sm70 —— 直接决定选型)

- **flash-attn 不支持 V100(sm70)** ⇒ HAT/Axolotl/TRL 那套 `flash_attn_varlen_func` 方案**在我们集群不可用**。
- **可用**:`xformers.ops.memory_efficient_attention`(CUTLASS 后端,P100+ 即 sm70 可用,
  fp16)+ `fmha.attn_bias.BlockDiagonalMask / BlockDiagonalCausalMask.from_seqlens`
  —— **这正是 BLT 训练用的路径,也是我们能照搬的那条**。
- PyTorch 原生 SDPA 的 memory-efficient(CUTLASS)后端在 V100 可用;flex_attention 需另行确认 sm70 支持。
- 本机 4060(sm89)不构成约束;方案必须按集群 V100 设计。

## 5. 对 bltz 的落地方案(提案,待拍板)

- **T1(推荐,收益最大)—— 解开 patch 内 padding**:
  把每行字节拼成扁平流 `bytes (B, T_real)` + `patch_offsets (B, S+1)`(= 现有 `ends`),
  encoder 用**块对角注意力**(xformers `BlockDiagonalMask.from_seqlens(patch_lengths)`)让每个 patch 自成一块,
  段池化(scatter/segmented PMA)出 patch latent。
  预期:encoder 计算量 ≈ 降到 **1/3**(16→5.2 字节/patch);`k_max`/Δ/loss **不受影响**
  (loss 本就从 `flat`+`ends` 取目标)。
  工作量:中(重写 encoder 的注意力与池化为块对角/分段);风险:中(需保证与原实现数值一致)。
  参考:§2.2 HAT(HF `model.py` + vLLM `hat_model_runner.py`)、§2.1 BLT `tokens_to_seqlen`+`BlockDiagonalCausalMask`。
- **T2 —— 变长 patch 数 / 文档打包**:行内改按**字节预算**装(可含多篇文档),块对角因果在文档边界隔断
  (启用已存的 `unit_flag`)。收益:提高有效上下文/利用率;代价:collate 与 loss 的 `(B,S)` 矩形假设要改。
  参考:BLT `SequenceIterator`+`PackingIterator`、nanochat `BestFit`、TRL `pack_dataset`。
- **T3(低成本,小时级)—— 长度分桶/动态批**:按每行真实字节数分桶、或按字节预算定 batch size,
  减少批内尾部 padding。不需要改注意力。
- **T4 —— 文档边界感知**:用 `unit_flag` 做块对角因果(质量向,非显存向),可与 T2 合并。

**建议顺序**:T3(先量出真实浪费)→ T1(主收益)→ T4/T2(按需)。

## 6. 待用户拍板

1. ~~是否推进 T1(改 encoder 注意力为块对角/分段),还是先做 T3 观察收益?~~
   **已闭环(2026-09-14 晚):T3 已测(docs/24 §5);T1 先记低优先级,**
   **重测(encoder 占前向 76-81%)后重审立即开工;实施为 T1' 长度分组**
   **重打包(docs/25,数值逐元等价、零新依赖),非本档的 varlen 方案;**
   **T3.3 分桶被 T1' 吸收,取消。**
2. 是否允许改动 `bltz/models/encoder.py`(架构级)——已由用户签核(T1' 开工)。
3. 参考代码仅阅读机制,还是允许移植(注意 BLT = CC-BY-NC-4.0 非商用、DMBP 仓库无 LICENSE)。

## 7. 参考索引

- 仓库:`facebookresearch/blt`(CC-BY-NC-4.0,`bytelatent/data/iterators/packing_iterator.py`、
  `sequence_iterator.py`、`bytelatent/model/utils.py`)、`Aleph-Alpha/tfree-hat-*`(HF)、
  `Aleph-Alpha/vllm@hat_v1`、`Aleph-Alpha-Research/hat-splitter`、`skai-research/lca-multibyte`(无 license)、
  `skai-research/flexitokens`、`imoneoi/multipack`、`karpathy/nanochat`、`pytorch/torchtune`、
  `mosaicml/llm-foundry`、`huggingface/transformers`(`DataCollatorWithFlattening`)。
- 论文:2501.10322(HAT)、2603.15953(HAT 后继)、2608.15454(DMBP)、2507.12720(FlexiTokens)、
  2404.10830(Fewer Truncations)、2107.02027(Packing 2x BERT)。
- 本仓代码定位:`bltz/data.py:29-63`、`bltz/shards.py:107-224`、`bltz/models/encoder.py:93-113`、
  `bltz/objectives.py:35-53`、`configs/default.yaml:5,9,29-32`。
