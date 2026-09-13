# 14 — 范式脉络与学术定位:压缩单元骨干 + 局部字节解码

> **性质**:专项调研档案(2026-09-14)。触发事件:用户发现 arXiv 2501.10322v2(HAT,Aleph Alpha)与本项目重合度高于 BLT。
> **来源**:①内部基线(`docs/03`、`docs/13`、`docs/blt_research/*`、`docs/01`);②四路外部调查(范式普查 / 并行解码与多字节预测谱系 / HAT 外部足迹 / HAT 团队 Outlook 审计);③主会话一手核验(HAT 全文 1721 行、后继论文 2603.15953 与第三方 DMBP 2608.15454 摘要)。
> **可信度**:普查/谱系表来自 librarian 报告,逐行附来源;主会话抽验关键行。标 **[未核]** 者为报告自述未验证项。表内来源列取 arXiv ID(完整 URL 见文末原始工件)。
> **关联**:`docs/15`(HAT 深挖,含 Outlook 审计 §9)、`docs/13`(MTP 冲突先例)、`docs/03`(2026-09-08 文献调研)。

---

## 0. 执行摘要

1. **用户的判断成立**:BLT 与 HAT 的局部解码器都是**自回归**;BLT 原文连"parallel"这个词都没出现过(全文 0 次)。
2. **但"并行解码无人涉足"已被证伪**:截至 2026-08,已出现四条并行/近并行路线——
   - 块扩散 + AR 验证:**Fast BLT-D/DV**(Meta,2605.08044,2026-05);
   - 掩码式段对齐并行多字节:**DMBP**(2608.15454,2026-08,第三方);
   - 层级潜空间扩散:**Zonkey**(2601.21768,2026-01);
   - 投机式多字节:**EvaByte 8 头**、**MTPC**(2511.11346)草案 + 验证。
3. **交叉缺口已闭合**:`Fast BLT-D` 与 `DMBP` 正是"层级/压缩单元骨干 × 并行字节生成"的组合(详见 §4)。
4. **HAT 团队自己没有兑现 Outlook 三项**(MTP 多头 / 扩散解码 / 更深层级);其正牌后继是规模化与 Llama 转换工程(§5)。
5. **我们的可辩护区隔必须收窄**为五"无":**单次前向的纯边际并行、无扩散迭代、无验证器、无额外头/参数、无外部分割器**,配 **单一共享条件头 (h,Δ) + 自熵停止**(§6)。
6. HAT 的最大价值是**方法与叙事**(鲁棒性评测、跨语言续训、bpb/词级准确率口径、compute-matching、Unicode splitter、[W]/[S] 停止体系),而非新的架构空间——见 `docs/15` §6。

---

## 1. 范式普查:"压缩单元 + 局部字节/字符解码"(2018→2026)

> 证据基础:BLT + Fast BLT 全文(本地)、arXiv abs/HTML、Semantic Scholar 引用 API(BLT 最新 100 条 + HAT 全部 24 条)、Crossref(venue)、GitHub/HF API。OpenAlex 引用数据经查已过期(BLT ACL 记录仅 14 cites)且调查中预算耗尽。

| 论文/模型 | 会议+年份 | 单元与边界决策 | 编码侧 | 局部解码:单元内 AR vs 并行 + 停止 | 训练目标 | 规模 | 代码/权重 |
|---|---|---|---|---|---|---|---|
| **Charformer** | ICLR 2022 | 软潜块 GBST(梯度打分,无硬边界) | 字节 encoder + GBST 下采样 | 标准 T5 编解码;**无逐单元局部解码器**;AR | T5 span 去噪 | 小 | 代码 google-research/*charformer*;无权重 |
| **CANINE** | TACL 2022 | 字符 → 学习式跨步下采样;无 decoder | 字符 encoder + 下采样 | **无**(仅 encoder 掩码 LM,不可生成) | 掩码字符 LM | encoder-only | 无官方代码/权重 |
| **ByT5** | TACL 2022 | **无 — 单元=byte**(平坦) | 字节 T5 encoder | 平坦 **AR** 字节 T5 decoder;EOS 停 | T5 span 去噪 | small→XXL | 代码 google-research/byt5;HF 权重 |
| **MegaByte** | NeurIPS 2023 | **固定 patch**(P=4/6/8) | patch embedder=字节嵌入拼接(无 encoder 网络) | **patch 内逐字节 AR**;固定长度停 | 逐字节 CE | 320M/1.4B 实验 | 无官方代码;非官方 lucidrains 复现 |
| **SpaceByte** | NeurIPS 2024 | **space-like 字节**成界(启发式外部规则) | 局部 transformer 前置;全局块只在 space-like 后 | **AR**;下一 space-like/EOS 停 | 逐字节 CE | ≤1B/80B bytes | 代码 kjslag/spacebyte;无权重 |
| **MambaByte** | COLM 2024 | **无 — 单元=byte**(各向同性 SSM) | 字节 Mamba | 平坦 **AR**;可选投机(子词起草+字节验证) | 逐字节 CE | ~1B 级 | 代码 jxiw/MambaByte;无权重 |
| **MrT5** | 预印本(2025-04 v3) | **仅编码侧**:学习删除门合并/删字节 token | ByT5 式 encoder + 删除门 | **无局部 decoder**(标准字节 T5 AR decoder) | ByT5 span 去噪续训 | ByT5-base/large | 代码 jkallini/mrt5;未发布权重 |
| **Grapheme-LLaMA** | COLING 2025 | **无 — 单元=字素/字符**(平坦) | 字符级 Llama | 平坦 **AR** 逐字符 | 因果字符 LM | baby-llama 级 | 论文未给代码/权重 |
| **BLT** | ACL 2025 | **外置熵模型驱动的变长 patch**:小字节 LM 预测下字节熵,超阈 θ_g 开新 patch(外部、非端到端) | 轻量局部 encoder(transformer+cross-attn,hash n-gram)→patch latent | **patch 内逐字节 AR**;熵阈开新 patch;EOS 止 | 逐字节 CE(原文无 MTP 表述) | 400M→8B;发布 1B/3B/7B;≤8T bytes | 代码 facebookresearch/blt;权重 gated |
| **HAT** | 预印本 2025-01 | **词(空白符切分)**——外部、非学习;空白附于前词 | 词内双向字符 transformer;[W] 位取词嵌入 | **词内逐字符 AR 循环**;预测 **[W]** 停;文档 [S] 止 | 逐字符 CE + [W] 终止符 | 至 **7B** | 论文未给代码/权重(后续见 §5) |
| **EvaByte** | 博客/模型卡 2025-01 | **无 — 平坦字节**(显式拒绝 BLT 式 patch) | 平坦字节 LM + EVA 分块线性注意力 | 默认 **AR**;可选 `multi_byte_generate`=**并行自投机**(8 个 MTP 头,Medusa 式树) | **8 头 MTP** 平均 CE | **6.5B**,1.5T bytes | 代码 OpenEvaByte/evabyte;HF EvaByte |
| **H-Net** | 预印本 2025-07 | **学习式动态分块**(路由预测边界 + 平滑/STE/比例损失,端到端) | Mamba-2 encoder;主干 Transformer(可递归嵌套) | **逐字节 AR**(递归 DC 状态);无 EOW | 逐字节 CE + 比例损失 | >1B | 代码 goombalab/hnet;HF cartesia-ai |
| **AU-Net(Bytes to Ideas)** | 预印本 2025-06 | **多尺度池化**:字节→词→词对→4 词;右插稳定 | 字节层 + 深层池化(UNet) | **推理 AR**(最细字节层每步激活,深层低频) | 逐字节 CE + 深层下一词预测 | small/mid | 未找到代码 |
| **FlexiTokens** | ACL Findings 2026 | **学习式边界预测器**→变长段 | 字节层级 LM(encoder+边界预测器) | 基础版 **AR 逐字节**(2026 DMBP 在其上加并行头) | 逐字节 CE + 边界损失 | 可变 | 代码 skai-research/flexitokens |
| **Bolmo** | 预印本 2025-12(v2 2026-02) | **无 — 平坦字节**(子词模型字节化) | 字节化 transformer | 平坦 **AR** | 两阶段转换 + 续训 | 家族 | 自称全开源 [未核] |
| **MTPC** | 预印本 2025-11(v2 2026-06) | 无新单元——改装字节 LM(EvaByte/Llama3.2-3B-byte) | —(验证器不变) | **概率电路并行多字节草案** + 投机验证(窗口至 16B);非局部解码器 | MTP 头=概率电路 | EvaByte 6.5B 等 | 代码未捕获 [未核] |
| **Zonkey** | 预印本 2026-01 | **可微分分词器**(Segment Splitter 学 BOS 决策) | 字符 encoder + 层级 | **并行/迭代扩散**(潜空间去噪);概率衰减定长 | 扩散去噪 | 小(维基) | 自称开源,URL 未捕获 [未核] |
| **Proxy Compression** | 预印本 2026-02 | **仅训练期外部压缩器**;推理还原字节接口 | 平坦字节 LM,双视图训练 | 推理期平坦 **AR** | 逐字节 CE + 代理对齐 | 多算力档 | 代码 LZhengisme/proxy-compression |
| **ByteFlow Net** | ICLR 2026 | **码率 Top-K 分块**(在线信息论) | 局部 encoder(SWA+Canon)→全局 | **AR** 字节;无 EOW | 逐字节 CE | 0.6B/1.3B | 未找到代码 |
| **Fast BLT(BLT-D/S/DV)** | 预印本 2026-05 | 同 BLT 熵分割(外部、阈值) | 同 BLT 局部 encoder + 全局 | **并行局部解码**:块扩散逐步解掩码(置信 α/熵界 γ);BLT-S 自投机;BLT-DV 扩散草案 + **AR 验证** | 逐字节 CE + **块扩散辅助损失** | 1B/3B 实验 | 论文未声明代码/权重 [未核] |
| **ATDC** | 预印本 2026-05 | 课程控制的动态分块(目标压缩比) | 层级 encoder[未核] | 继承层级基座,AR 字节[未核] | 逐字节 CE + 压缩比控制 | FineWeb-Edu 100B | 未找到代码 |
| **Scratchpad Patching** | 预印本 2026-05 | patch 式字节 LM;**patch 内熵触发瞬时 scratchpad** 刷新陈旧上下文 | (patch 架构) | **AR/因果**(显式保因果,非并行解码) | 逐字节 CE | 16B/patch;KV 降 3–4× | 未找到代码 |
| **DMBP(LCA-MBP)** | 预印本 2026-08 | 基于 **FlexiTokens** 学习段;段=多字节生成单元 | FxT 字节 encoder + 边界预测器 | **段内并行**:单 MBP 头以 Latent-Causal-Attention 掩码一次预测整段字节;P≥τ 接受(下一字节头恒接受) | 逐字节 CE + MBP 损失(单头、零额外参数) | d_model=1024 / 373M 参数 / 50B bytes | 代码待发布 skai-research/lca-multibyte |
| **When Tokenizers Fail** | EMNLP 2026 | 层级字节 chunk,**词对齐目标**(chunk 对齐 + POS 监督边界) | 字节嵌入由冻结子词 LM 初始化 | AR 字节(继承)[未核] | chunk 对齐 + POS + LM | 六语言 | 未找到代码 |
| **Disentangling LM and Boundaries** | 预印本 2026-08 | **立场文,无模型**:论"下一字节分布与边界分布可分离" | — | — | — | — | — |

**引用挖出但不属于本范式**(已核查排除):NCP-ArchPreview(潜概念,2609.10715)、ReinPatch(时序 patch,2603.26097)、BitLM(词级位扩散,2605.11577)、EntropyMoE、ReconSpan、SUNTA(视频)、GeneZip/dnaHNet/MergeDNA(DNA)、"Dynamic tokenization in the Transformer era"(综述,无 arXiv ID)、"The Efficiency Gap in Byte Modeling"(2605.12928)、"Efficient Pre-Training with Token Superposition"(2605.06546)。

**Venue 备注**:MegaByte=NeurIPS 2023、SpaceByte=NeurIPS 2024、BLT=ACL 2025、Charformer=ICLR 2022、ByT5=TACL 2022、MambaByte=COLM 2024、Grapheme-LLaMA=COLING 2025、FlexiTokens=ACL Findings 2026、ByteFlow=ICLR 2026、When Tokenizers Fail=EMNLP 2026(Crossref/arXiv 确认);**HAT、MrT5、H-Net、Fast BLT、ATDC、Scratchpad、DMBP、MTPC、Zonkey、Proxy Compression、Bolmo、AU-Net = 预印本,未找到同行评审 venue**。CANINE 经复核为 **TACL 2022**(DOI 10.1162/tacl_a_00448)。

---

## 2. 解码器类型裁决

### 2.1 明确"并行/近并行"局部解码(6 条)

1. **Fast BLT / BLT-D** —— 已核(全文)。"the local decoder can generate a fixed-size block of future bytes **in parallel**";半自回归块扩散,每步解多个掩码;BLT-S 越界 AR 起草;BLT-DV 用 AR 复核。**是经典 BLT/MegaByte/SpaceByte/HAT/H-Net 一脉中唯一带真正并行局部解码器的成员**。
2. **DMBP(LCA-MBP,2608.15454)** —— 已核(全文)。单 MBP 头在 Latent-Causal-Attention 掩码下一次预测 FlexiTokens 段的全部字节;置信阈 τ 接受;"parallel byte prediction without violating causality"。
3. **MTPC(2511.11346)** —— 已核。概率电路 MTP 并行起草多字节窗口(至 16),再投机验证;不是局部解码器,而是改装到 EvaByte/Llama-byte 上的并行多字节生成机制。
4. **EvaByte `multi_byte_generate`** —— 已核(博客+模型卡)。默认 AR;可选**并行自投机**多字节(8 MTP 头,Medusa 式);"greedy 下通常与 AR 结果一致"。
5. **Zonkey(2601.21768)** —— 已核(摘要)。层级**扩散**生成(潜空间并行/迭代去噪),非逐单元字节解码器;序列按概率衰减而非 EOS。
6. *(邻近,词级)* **BitLM(2605.11577)** —— 位级连续扩散头,块内并行去噪多个 token,块间因果。

### 2.2 明确"自回归"局部解码(已核)

- **MegaByte**:"a small local model **autoregressively** predicts each patch byte-by-byte"(原文)。
- **SpaceByte**:局部 transformer "**autoregressively** outputs byte-level logits";全文无 "parallel" 讨论。
- **BLT**:局部解码器"**autoregressively** decodes";全文 v1 **"parallel" 0 次**——既不用也不讨论/拒绝。
- **HAT**:"we run an **autoregressive loop** of the decoder module";[W] 停;其 future work 只谈词嵌入缓存/投机解码,**不谈**并行局部解码。
- **H-Net**:"H-Net generates raw bytes … **autoregressively**"(递归 DC 状态)。
- **FlexiTokens**:基础 FxT 为逐字节 AR(2026 MBP 论文把并行作为扩展加入)。
- **AU-Net**:"At inference, generation is **autoregressive**…"。
- **ByteFlow**:decoder 输出 p(x_{t+1}|x_{1:t})。
- **EvaByte(默认)、MrT5、ByT5、MambaByte、Grapheme-LLaMA**:平坦 AR 字节/字符(MambaByte 的投机是"子词起草+字节验证",仍是 AR 接受)。

### 2.3 谁讨论/回避了并行

- **没有任何范式成员明确"拒绝"并行**;最强的表述是把逐字节 AR 当作**待解决的瓶颈**(Fast BLT:"slow, byte-by-byte autoregressive generation";DMBP:"generating one byte at a time remains a bottleneck";MTPC:"generate tokens one at a time within the window, increasing latency")。
- **BLT 原文完全没提**(已核负结果)——并行局部解码只出现在 2026 的后续工作里。
- ⚠️ **MegaByte 的 "parallelism in decoding" 措辞指 patch/全局层级的并行**(各序列位置的 patch 表示并行产生),patch 内字节仍 AR——**不可误读**为并行局部解码。
- **Scratchpad Patching** 显式保因果(其干预是上下文刷新,非并行解码)。
- **[未核]**:是否有 2026 论文明确论证"不应"做并行局部解码——在 124 篇 citing 的标题/摘要筛选中未发现该表述;全文核查仅限上列成员。

---

## 3. 并行解码与多字节预测谱系

### 3.0 输入清单的 ID 更正(已核)

| 输入 | 状态 | 更正 |
|---|---|---|
| Hydra "2402.05132" | ✗ 错 ID | **Hydra = 2402.05109**;2402.05132 是 TexShape(句嵌入) |
| LLaDA "2502.11089" | ✗ 错 ID | **LLaDA = 2502.09992**;2502.11089 是 Native Sparse Attention |
| "Medusa-2" | ✗ 非独立论文 | 是 **Medusa 内部的训练制度**:"Medusa-1: 冻结骨干上微调;Medusa-2: 与骨干联合微调"(2401.10774 摘要) |
| EAGLE 系 | ✗ 不全 | EAGLE=2401.15077、EAGLE-2=2406.16858、**EAGLE-3=2503.01840** |
| OCC 2605.28184 | ✗ 误读 | OCC = **MTP+RL 训练的最优系数标定**,**不是**解码方法 |
| Jacobi"2410.18160" | ✗ 错 ID | 2410.18160 = Future Token Prediction;**Jacobi 解码 = 2305.10427** |

### 3.1 A1 — token 级多 token 预测(MTP)

| 工作(arXiv) | 层 | 未来数 | 损失设计 | 推理用法 | 真并行发射? |
|---|---|---|---|---|---|
| Gloeckle 2404.19737 | token | n(头条 n=4) | 共享主干上 n 个**独立头**,逐头 CE 辅助 | 头部当草案;"4-token 预测推理最高 3× 快" | 否(草案+验证) |
| DeepSeek-V3 2412.19437 | token | 1 额外 token/模块(D=1,可链) | 位移 CE;**共享嵌入+共享输出头** | 投机解码;**第二 token 接受 85–90%,1.8× TPS** | 否 |
| Medusa 2401.10774 | token | K 头(≤5)+树 | CE;Medusa-1 冻结 / Medusa-2 联合 | 树注意力投机验证;2.2×/2.3–3.6× | 否 |
| Hydra 2402.05109 | token | K 个顺序**依赖**头 | teacher forcing + Hydra++ 配方 | 投机;较 Medusa 1.31×,较 AR 2.70× | 否 |
| EAGLE 2401.15077 | token(特征级) | 特征上的 AR 草案 | 特征回归 + token CE | 投机 | 否 |
| EAGLE-2 2406.16858 | token | 动态草案树 | — | 上下文感知树投机 | 否 |
| EAGLE-3 2503.01840 | token | 多层融合草案 | "抛弃特征预测…直接 token 预测…训练时测试" | 投机;最高 **6.5×** | 否 |
| Gumiho 2503.10135 | token | 串行+并行混合头 | 优先早期 token | 投机 | 否 |
| Jakiro 2502.06282 | token | MoE 解耦多头/多候选 | MoE 草案头 | 投机树 | 否 |
| MuToR 2505.10518 | token | 寄存器 token 扩视野 | 寄存器交错,各预测一个未来目标;**几乎零参数、不改架构** | 训练目标;**无推理加速主张** | 否(仅训练) |
| Future Token Prediction 2410.18160 | token | 多未来语义状态向量 | 语义状态向量目标 | 训练目标 | 否 |
| AdaMTP 2608.00434 | token | **自适应变深** | 熵分割 → "动态掩码 MTP 损失,抑制跨边界的预测损失" | 声称任务+推理加速 | 否(隐含验证) |
| OCC 2605.28184 | token | (RL 中的 MTP 头) | MTP+RLVR 梯度系数标定 | **仅训练** | 否 |
| LSE-MTP 2604.06155 | token | (多步) | 锚定真值隐状态轨迹;ACL 2026 | 训练/表征(世界模型) | 否 |
| MARS 2604.07023 | token | 多 token/遍 | 轻量微调、零新参数 | 直接多 token 生成;"Unlike speculative decoding…or Medusa" | **是(自称)** |
| PIPO 2605.27255 | token | 成对(潜压缩器+MTP 头) | 镜像压缩/预测;**免昂贵验证** | 直接 | **是(自称)** |
| EntMTP 2606.27550 | token | **动态**(熵导向深度) | 静态树→熵自适应深度 | 自投机 | 否 |
| CLP 2606.10935 | token | 搭配长自适应 | "Backbone-as-Architect";头-骨干竞争 | "Zero-Loss 自适应多 token 推理"(自称) | **是(自称)** |
| LoopMTP 2608.03624 | token | 每循环的潜 MTP 引导 | 连接 looped transformer 迭代 | 训练(推理) | 否 |
| ESP 2603.17942 | token | Top-K 树,免训练 | 嵌入空间探针 + 剪枝 | "预测并行验证" | 否 |
| MTP-SD 2602.06019 | token | 多 token 模型 | 在线自蒸馏 AR→独立 MTP | "无需任何辅助验证器即可部署" | **是(自称)** |
| MRP 2605.18817 | token | 多 token/去噪步 | "单次骨干前向内的依赖感知多 token 去噪" | dLLM 去噪 | 是(扩散内) |
| P-MTP 2606.24447 | token(视觉语言) | 渐进深度 | 文档解析渐进 MTP | 加速 | 否 |
| 规划分析 2604.11912 / 综述 2509.24435 | token | — | MTP vs NTP 的规划分析 / NTP 替代方案综述 | — | — |

### 3.2 A2 — 字节/位级多未来预测(**关键表**)

| 工作(arXiv) | 层 | 未来数 | 损失设计 | 推理用法 | 真并行发射? |
|---|---|---|---|---|---|
| **MBP 2608.15454** "Dynamic Multi-Byte Prediction With Hierarchical LMs" | **字节,层级段** | 变长窗口,**对齐潜段**(实验 6 候选) | 逐字节 CE + 多项 MBP 项;**单个 LCA 掩码头**(非逐 token 头) | **置信阈接受**(P≥τ,左→右截断);平均接受 3.05/6,15% 整窗;作草案+外部验证器时 **2.1–2.3×** | **是,带门控**(默认无 AR 验证) |
| **MTPC 2511.11346** | token(施于**字节 LLM**) | 窗口 n;**联合分布**(电路) | 联合 MTP 似然(推广 HMM/张量网络);批评独立头 | **从联合并行采样**;配合投机解码"贪心输出与 AR 逐位相同" | 是(联合采样)/验证下精确 |
| **EvaByte** | **字节,平坦** | **n=8 字节窗口** | 多字节头;"实现改编自 Medusa" | `multi_byte_generate()`;"贪心下通常与 `generate()` 相同";解码最高 2× 快 | 否(Medusa 式草案+验证) |
| **Fast BLT / BLT-D 2605.08044** | **字节,层级 patch** | 每扩散步一块掩码字节 | "标准逐字节损失 + **块扩散辅助目标**";BLT-DV 加 AR 验证 | "每解码步**并行生成多个字节**";带宽降 >50%,最高 92%(有质量退化);BLT-DV 恢复 | **是**(近似;BLT-DV 验证) |
| BitLM 2605.11577 | token→**二进制码**,位级连续扩散 | 多 token 单元 | token=fixed-length 二进码 + 连续扩散 | 多 token 并行生成 | 是 |
| MDM-Prime-v2 2603.16077 | token→二进制子 token | — | 掩码扩散 LM;"索引打乱"+"二进制编码" | 并行去噪(dLLM) | 是 |
| Efficiency Gap 2605.12928 | **字节,平坦** | — | 字节 AR vs **字节 MDM** 的算力匹配缩放 | 并行、非序列生成 | 是(研究:字节 MDM 惩罚比 AR 更重) |
| Zonkey 2601.21768 | **字符→词层级**,连续 | — | 可微分 Splitter/Stitcher + 扩散;"同一层级的所有向量**并行**生成" | 逐层并行 | 是(仅定性证据) |
| MambaByte 2401.13660 | 字节 | — | AR 字节 SSM(无分词基线) | AR | 否 |
| AR U-Nets 2506.14761 | 字节→词层级 | 深层"预测更远的未来" | 学习 token 上的 AR UNet | AR(层级,非并行) | 否 |
| Byte-at-a-Time Sampling 2506.14123 | 字节(采样) | — | 把 token LM 转字节采样 | 字节级解码 | 否 |

### 3.3 B1 — 经典并行与投机解码

| 工作(arXiv) | 机制 | 并行输出? | 精确性 |
|---|---|---|---|
| Stern 2018 1811.03115 | 块并行预测 + "回退到最长已验证前缀" | 否(前缀) | 近似精确(有验证) |
| Leviathan 2022 2211.17192 | 投机解码 | 否 | **"输出不变"** |
| Chen 2023 2302.01318 | 投机采样 | 否 | 保目标分布 |
| Santilli 2023(Jacobi)2305.10427 | "把贪心 AR 解码重述为**雅可比/高斯-赛德尔不动点迭代**" | 迭代并行 | 收敛到贪心 AR 输出 |
| Lookahead 2402.02057 | "**精确、并行**解码;无辅助模型/数据存储" | 否(并行分支) | 精确 |
| Medusa/EAGLE/Hydra | 草案头/特征 AR/依赖头 | 否 | 验证下精确 |
| 半自回归 NMT 1808.08583 | 分块并行 | 块级 | 近似 |
| Gu 2017 NAR 1711.02281 | 完全并行输出;延迟降一个数量级;BLEU 代价 ~2.0 | **是** | 近似 |
| Mask-Predict 1904.09324 | 先全预测,再迭代重掩低置信 | **是**(迭代) | 近似 |
| Insertion 1902.03249 / Levenshtein 1905.11006 / Fast Structured 1910.11555 | 插入/编辑/结构化 NAR | 是 | 近似 |
| MaskGIT 2202.04200 | 迭代并行解掩码(视觉) | 是 | 近似(MDM 解码的概念祖先) |

### 3.4 B2 — 扩散/流语言模型(并行文本生成)

| 工作(arXiv) | 机制 |
|---|---|
| Diffusion-LM 2205.14217 | 连续扩散 LM(可控性;并行细化) |
| D3PM 2107.03006 | 离散去噪扩散(奠基) |
| Plaid 2305.18619 / SEDD 2310.16834 | 似然/分数熵离散扩散 |
| **LLaDA 2502.09992** | 从零训练的掩码扩散;"预测被掩 token…似然下界";8B 与 LLaMA3 8B 相当 |
| Dream 7B 2508.15487 | "迭代去噪并行细化序列";任意顺序/填充 |
| Fast-dLLM 2505.22618 | "块级近似 KV 缓存…并行解码"(免训练加速) |
| Seed Diffusion 2508.02193 | 离散扩散;"非序列、并行生成";H20 上 2,146 tok/s |
| Block Diffusion(BD3-LM)2503.09573 | AR↔扩散插值;灵活长度 + KV 缓存 + 并行采样(半 AR 块标准) |
| MDM-Prime-v2 2603.16077 / Zonkey 2601.21768 | 二进制编码+索引打乱 / 层级可微分分词扩散 |
| Mercury(Inception)/Gemini Diffusion(Google) | 商业 dLLM;仅博客级主张 [未核] |

### 3.5 B3 — 并行**字节/位**级生成(关键子问题的直接答案)

| 工作 | 并行字节/位生成? | 证据 |
|---|---|---|
| **Fast BLT(BLT-D)** 2605.08044 | ✔ 每扩散步并行发多字节,**在 BLT patch 层级上** | "generates multiple bytes in parallel per decoding step";"combines BLT's hierarchical latent tokenization with block-wise discrete diffusion" |
| **MBP** 2608.15454 | ✔ 学习段层级字节 LM 上的多字节发射 | "generates multiple bytes in parallel";LCA 掩码"enables parallel byte prediction without violating causality" |
| **MTPC** 2511.11346 | ✔(字节级目标:EvaByte、Llama-3.2-3B-Byte) | "joint distribution over the token window that can be sampled from in parallel" |
| Efficiency Gap 2605.12928 | ✔ 平坦字节 MDM(无层级) | 字节建模 + 掩码扩散缩放研究 |
| BitLM 2605.11577 / Analog Bits 2208.04202 | ✔ 位级扩散 | 位级连续扩散 |
| EvaByte(博客/仓库) | △ 每步多字节,但 **Medusa 式草案+验证**,平坦字节 | "implementation adapted from Medusa" |
| MambaByte 2401.13660 | ✗ AR 字节基线 | "trained autoregressively on byte sequences" |

---

## 4. 交叉裁决:并行字节解码 × 层级骨干 —— **缺口已闭合**

**结论:截至 2026-05/08,该组合已存在。我们的 novelty 主张必须改写。**

1. **Fast BLT / BLT-D(2605.08044,2026-05)** —— BLT 动态熵 patch + **字节块离散扩散**;"fully compatible with BLT's hierarchical architecture";BLT-S 自投机、BLT-DV(扩散草案+AR 验证)。**架构层面就是同一组合。**
2. **MBP(2608.15454,2026-08)** —— 层级字节 LM(FlexiTokens 式学习边界,与 H-Net 同族)+ **单个 LCA 掩码解码头并行预测整个潜段的字节**,变长窗口、置信阈接受。原文:*"A single decoder head equipped with the LCA mask predicts all bytes within a segment in parallel, conditioned only on previous segments. This removes the per-token parameter overhead of multi-head MTP and replaces fixed-offset prediction with variable-length, segment-aligned prediction."*
3. **MTPC(2511.11346)** —— 概率电路联合多字节;改装 EvaByte 6.5B 与 Llama-3.2-3B-Byte;投机下精确。
4. **Zonkey(2601.21768)** —— 层级(可微分字符→词)扩散,逐层并行(仅定性)。
5. **Efficiency Gap(2605.12928)** —— 平坦字节 MDM 缩放不如 AR;主张字节域需要 *"alternative structural biases"* ——**恰为我们的层级结构提供了动机**。

> **追加(2026-09-14)**:DMBP 全文精读、bltz↔DMBP 解码范式对比(架构品味 + 应用优劣势)与 bltz/HAT/DMBP 三方相似度分析见 `docs/16-dmbp-decoder-comparison.md`。

**前期工作未覆盖、可作残余新颖性的部分:**

- **条件神经场查询解码器 `(h, Δ) → byte`,以相对偏移查询并行发射整个 patch 的字节。** MBP 用 LCA 掩码 transformer 头;BLT-D 用扩散;MTPC 用概率电路。**未检索到"查询解码器/神经场 + Δ 相对偏移"的同类formulation。**
- **space-like 词界自然停止**作为推理终止规则(vs BLT 熵 patcher、FxT/H-Net 学边界、MBP 置信阈)。
- **MTP + 并行发射 + 层级 + 相对 Δ 查询解码**作为**一个系统**,且**不含**草案/验证遍历、不含扩散去噪器(MBP 最接近,但它仍靠置信门控,层级也是边界预测器而非 Set-Transformer patch)。

**建议的收窄主张**:不再说 *"首次在层级骨干上做并行字节解码"*(那是 Fast BLT/MBP 的地盘),而是 *"基于学习 patch 的条件神经场查询解码器实现并行多未来字节解码,并以词界自然停止收束"*——并在 Related Work 正面引用 Fast BLT、MBP、MTPC、EvaByte 作为最近邻。

**缺口检索证据(使否定性主张可审计)**:
- arXiv 标题检索 `multi-byte`(2024-01→2026-09):**5 条**,仅 MBP/MTPC 相关;
- `byte` × `multi-token` 全字段(2024-06→2026-09):**38 条**,相关=MBP、MTPC、AR U-Nets、byte-sampling、MDM-Prime-v2;
- `byte-level non-autoregressive generation`: **2 条**(Efficiency Gap;XRayEmb 2021);
- `byte` × `diffusion`(2025-01→2026-09):**11 条**,相关=Efficiency Gap、Fast BLT、MDM-Prime-v2、Zonkey。
- 注意:关键词检索会漏掉"不用这些术语却实现了并行字节解码"的论文;2026 进展快,此结论有时效性。

---

## 5. HAT 团队 Outlook 兑现审计(用户专项,2026-09-14)

**问题**:HAT(Aleph Alpha)团队自己在 1.5 年里有没有实现其 Outlook(①MTP 多头;②文本扩散当解码器;③更深层级)?

**结论:三项均未由团队实现**;正牌后继是规模化/工程化,第三方已占位 (①)+并行。

| Outlook 项(HAT v2 §5 原文) | 团队自己 | 第三方 |
|---|---|---|
| "multi-token prediction … with multiple output heads" | ✗ 未做 | DMBP 2608.15454(段对齐并行多字节、单 LCA 头)、MTPC 2511.11346、EvaByte 8 头 |
| "a text diffusion model … could be used as a decoder" | ✗ 未做 | Fast BLT-D/DV 2605.08044、Zonkey 2601.21768 |
| "additional levels of hierarchy, such as sentences or paragraphs" | ✗ 未做(仍列 future work) | 未检索到实现 |

**证据**:
- HAT 本体:仅 v1(2025-01-17)→v2(2025-01-20),**无 v3+**;两版 PDF 同为 1,550 KB(v2 为标题多余波浪号微修)。S2 他引 24 条(调查时);OpenAlex 该 DOI 记录 cited_by=0 且引用解析为 0(不可信;期间预算耗尽);未找到同行评审 venue。
- **正牌后继:arXiv 2603.15953《A Family of LLMs Liberated from Static Vocabularies》(2026-03-16,Aleph Alpha 36 人,含四位 HAT 作者)**。摘要原文仍为 "based on the hierarchical autoregressive transformer (HAT) architecture";backbone "a classical autoregressive transformer";decoder cross-attend 后 "converted back into bytes"。内容:7B 从零训(近 4T 词)+ **Llama 3.1 8B/70B 转 HAT**(编/解码器从头训、复用骨干)+ EN/DE SFT/DPO + HF 发布(含 200 个预训练 ckpt)。关键词检索:`diffusion / multi-token / MTP / parallel / multiple heads / deeper hierarchy` **全 0 命中**;唯一 future work="其它领域(编程语言)"。
- **SOMBRERO(2601.22805,2026-01-30,Neitemeier/Balles 等)**:层级模型边界放置研究;future work 明确含 "extending the analysis to **deeper hierarchies**"。
- 其它同作者论文(GermanWeb 2505.00022、LIME 2512.07522、Good Pretraining Bad SFT 2609.08966 等)与 Outlook 无关。
- 发布物:HF `Aleph-Alpha/tfree-hat-pretrained-7b-base`(Open Aleph License,非商用;encoder 119M + backbone 6.98B + decoder 94M = 7.19B;~4T 词;200 中间 ckpt)、`llama-3_1-8b-tfree-hat-{base,sft,dpo}`、`llama-3_1-70b-tfree-hat-sft`、`tfree-research-*`、`umup-research-*`、`sp-baseline-research-*`;另有 `Aleph-Alpha-Research/Qwen3-1.7B-Hatified-v0-A..G`。
- 官方 GitHub(Aleph-Alpha)无 HAT 训练代码;第三方小复现 `FBR65/no_tokenizer`(1 star);据 HN 转引路透,Cohere 于 2026-04 收购 Aleph Alpha(未一手核)。

**含义**:窗口仍在,但已非空地(§4);团队两年留白说明该方向对"规模化/落地"优先级低——**既是机会,也是"难/低收益"的信号**。

---

## 6. 重新定位建议

### 6.1 已不成立的说法

- ✗ "没人把并行解码用在字节/层级 LM 上" —— 被 Fast BLT-D / MBP / Zonkey 证伪。
- ✗ "BLT 的组件(局部 AR 解码器/外部熵模型)是必要的" —— 保留为我们的实验主张,但需承认 HAT 团队自己也没动这些组件(他们只换规模)。

### 6.2 Novelty 风险表(论文写作必须处理)

| 我们的设计点 | 最近的占位 | 建议区隔措辞 |
|---|---|---|
| **D1 纯并行解码** | Fast BLT-D(扩散+验证)、DMBP(掩码+置信门)、EvaByte/MTPC(草案+验证) | "单次前向的纯边际并行,无扩散迭代/验证器/额外头";主张=并行是默认且唯一路径,而非加在 AR 之上的加速件 |
| **D2 patch 局部编码器** | HAT 词内双向 encoder(已规模化到 70B) | 措辞:局部编码器方向 HAT 已证可行;我们差异在**无序 Set Transformer + 与分词正确性解耦** |
| **D5 模型自分割** | H-Net(端到端学切分)、FlexiTokens/DMBP(学边界)、AdaMTP(熵分割) | 我们**不训练分割器**,而是"目标模型自产边界信号"(space-like 停 + 熵兜底) |
| **MTP 训练目标** | `docs/13` 全套 + AdaMTP/OCC + DMBP | 已在 `docs/13` 区隔(token vs byte;多头/掩码 vs 单 (h,Δ) 头);补 DMBP 引用 |
| **自熵停止** | Fast BLT-S、AdaMTP、Scratchpad Patching | 需精确引用,**不可自称首创**;差异:词界优先、熵仅兜底(D11) |

### 6.3 叙事顺序建议

1. 先立"**字节 I/O 的泛化/鲁棒/可微调**"—— HAT 已给出强 priors 与现成评测模板(鲁棒性扰动、跨语言续训、bpb);
2. 再立"**并行解码的效率主张**"—— 需 D1/D-6 数据,并正面区隔 Fast BLT-D/DMBP;
3. 最后才谈"**挑战 BLT 组件的必要性**"—— 需承认并行一侧已有人先走。

⚠️ 与用户立场一致性:`docs/blt_research/ByteField用户立场归档.md` 的"组合式/复古/工程优先"口径不变;但"直接挑战 BLT 的实验项"需补一条事实——**并行解码方向已不再是空地**。

### 6.4 待用户拍板

- 是否把本档结论回写 `docs/01` §10(差异化列表)与 `docs/09` §6(定位备忘);
- 是否调整 D1 的实验叙事顺序(先效率论证 vs 先质量诊断);
- 是否把 DMBP/Fast BLT-D 列入"必须正面区隔"清单并各做一次针对性对照实验。

---

## 7. 缺口与后续动作

- **检索缺口**:引用挖掘只扫了 BLT 最新 100 条 + HAT 全部 24 条;OpenAlex 预算耗尽;S2 覆盖滞后(部分 2026 新作不在其 citation 图中)。结论有时效性。
- **可深挖**:Fast BLT 全文精读(本地已有 `2605.08044v1.txt`)、DMBP 全文(需下载)、H-Net 推理细节、Efficiency Gap 的字节 MDM 代价数据。
- **未核项**:MambaByte 精确规模;多处代码/权重是否存在;Fast BLT 的带宽数字为估算非实测;MARS/CLP/PIPO/自蒸馏的"无验证器/零损失"均为作者主张。

---

## 附:来源与原始工件

- **四份外部调查报告**(opencode 临时工件,可能被清理):
  - 范式普查:`C:\Users\he\.local\share\opencode\tool-output\tool_09bc04d3c001SVleQ7R7xFBu1a`
  - 并行解码谱系:`...\tool_09bc04e2a0017bdJZ3S6k8uMIy`
  - HAT 外部足迹:`...\tool_09bc04e92001ofjrGGK3OeV67W`
  - HAT Outlook 审计:`...\tool_09bc04e25001lO1lwjZMV79LGP`
  - 最终段落抽取稿:`...\tool_09bc2d600001ltmwuZj1nd95Qq`
- **一手核验**:`docs/15`(HAT 全文 1721 行)、arXiv 2603.15953 摘要、arXiv 2608.15454 摘要。
- **本地文本版**:`reference_projects/2412.09871v1.txt`(BLT)、`2605.08044v1.txt`(Fast BLT)、`2501.10322v2.txt`(HAT)。

(完)
