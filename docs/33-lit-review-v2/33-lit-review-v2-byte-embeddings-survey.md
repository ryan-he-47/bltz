# 33 补 — 字节感知/可学习嵌入层专项调研(2026-09-21 第二波)

> **性质**:第一波调研(docs/33-lit-review-v2/ 主报告)的补充。触发:用户指出
> "字节感知的 LM 嵌入层 / 基于字节的可学习嵌入层"口径未覆盖,并点名
> Kronecker Embeddings。
> **方法**:三路并行外部检索(Kronecker 专项 / 字节嵌入层普查 / 重建塑形+冻结
> AE 交叉方向),全部事实来自本轮实际抓取页面,附 URL;否定性结论标
> UNVERIFIED-as-absence。不修改任何既有文档。
> **关联**:主报告 `33-main-synthesis.md`;深挖报告 `33-lit-review-v2-deep-dives.md`;
> 重建塑形嵌入分报告 `33-lit-review-v2-reconstruction-frozen-ae.md`。

## 0. 结论先行

1. **Kronecker Embeddings 已定位:arXiv:2605.29459**(2026-05-28,Rohan Shravan
   / The School of AI)。字节×位置的 Kronecker 积构造确定性 codec + 单个可学习线性
   投影,替代 `nn.Embedding` 查表。与 v2 输入侧同口号、同"字节×位置"基元、同
   ≤32 字节尺度——但它是**固定 codec(非学习)、挂在 BPE 分词器之上、无重建目标、
   输出侧空白**,深挖后撞车程度降为**低-中**(详见深挖报告)。其 §8.5 的
   tied-head Kronecker decoding 设想(输出侧单高斯 NLL + 一次性解出全部字节)与
   v2 的 MDN 头+AE 解码器概念同构,**对方仅为未测试假设,v2 是已实现强化版**。
2. **"字节序列 → 注意力池化 → 单向量 → 冻结供骨干"的完整组合在公开文献中无先例**
   (UNVERIFIED-as-absence)。Set Transformer(ISAB+PMA)用于字节/字符编码无署名
   先例——空白地带,既是新颖性机会也缺现成超参先例。
3. **LTLM 族(Bolmo 的命名)的池化手段只有三种**:选择式(H-Net/Bolmo/AU-Net
   取末字节或界处向量)、均值(FlexiTokens/H-Net++)、cross-attention(BLT 均值
   初始化 query、TFree-HAT 学习 query)。**全部端到端 LM 训练,无一使用重建目标
   或冻结输入编码器**——v2 的 AE 路线在该族内是孤例。
4. **输入侧高相似三强**(进入深挖):Kronecker Embeddings(2605.29459)、
   Bolmo(2512.15586)、TFree-HAT(2603.15953)。
5. Kronecker 论文在 124M/2.5B tokens 上证明"字节组合嵌入替代查表不掉点反略优"
   (val loss -2.5%,三种子)——**v2 输入侧合法性的独立第三方证据**(注意其
   tying 混杂,见深挖报告 §五)。

## 1. Kronecker Embeddings 专项(详)

- **机制**:κ(b)=(1/√L)·vec(Σ_p c_{b_p} ⊗ p_p),字节 one-hot(256)⊗ 位置
  one-hot(d_p=16/32)→ D=4096/8192 维稀疏 codec,逐 token z-norm 后过**唯一**
  可学习层 `Linear(D, d_model, bias=False)`(无 GELU/隐藏层/残差)。GPT-2 124M
  设定输入侧参数 38.6M→3.1M(-91%)。 [arXiv HTML](https://arxiv.org/html/2605.29459v1 "citation"), [GitHub](https://github.com/theschoolofai/kronecker-embeddings "citation")
- **明确自述**:"not a tokenizer paper / not a character-level model / not
  tokenization-free"——仍挂标准 BPE,序列长度不变;支持 forced-OOV(任意 ≤d_p
  字节串推理时编码)。 [arXiv](https://arxiv.org/abs/2605.29459 "citation")
- **实验**:跨模型探针(135M-671B,训练嵌入聚类印刷变体而非词形亲属,Kronecker
  跳出该聚类);nanoGPT 124M × FineWeb-Edu 2.5B × 三种子:val loss -2.5±0.2%、
  样本效率 1.43×;typo 探针 top-1 保持 55.5% vs BPE 47.3%;范数稳定 ~1.0。
- **自认 tradeoff**:字节近/语义远词在嵌入层聚一起(compute/commute、
  nation/notion),消歧推给注意力层——与 v2 已观测的"λ 是字形空间、语义住在 h"
  互为印证。
- **同名辨析**:word2ket(ICLR 2020,arXiv:1911.04975)、KroneckerBERT
  (2109.06243)、Shapeshifter(NeurIPS 2021)、PHM/Compacter 均为"Kronecker 积
  **压缩**词表嵌入/权重"的老谱系,与字节无关;2605.29459 是"字节×位置
  Kronecker 积**构造**嵌入"的唯一出处。 [word2ket](https://arxiv.org/abs/1911.04975 "citation")
- **社区反响**:Semantic Scholar 被引 0;唯一引用方是作者自己的后续
  LightningLM(2606.07404,120B MoE);训练 fork 未公开,124M 主结果当前不可
  复现。 [2606.07404](https://arxiv.org/pdf/2606.07404 "citation")

## 2. 分解式/组合式嵌入层谱系(均不撞,一句话存档)

ALBERT factorized embedding(1909.11942,低秩压缩查表)、TT embeddings
(EMNLP-Findings 2020,1901.10787)、QR/Compositional(KDD 2020,1909.02107,
推荐系统)、Hash Embeddings(NeurIPS 2017)、Deep Compositional Code
(ICLR 2018,1711.01068)、DHE(KDD 2021,2010.10784,"无表嵌入"口号相同但
推荐系统域)、MorphTE(NeurIPS 2022,2210.15379,词素组合——组合式输入嵌入
哲学最近邻,低-中)。**2024-2026 该方向唯一重量级新作即 Kronecker Embeddings。**

## 3. 字节组合成嵌入的机制普查

| 工作 | 机制 | 与 v2 输入侧关系 | 撞车 |
|---|---|---|---|
| charCNN(Kim 2016)/ELMo/fastText/CHARAGRAM | 字符 CNN/n-gram 求和成词向量 | 鼻祖级,注意力池化的前声 | 低 |
| Mimick(1707.06961) | 字符 BiLSTM 模仿词向量,OOV 冻结复用 | "离线训字符→向量映射再冻结"的最直接祖先,但拟合现成嵌入而非重建 | 中 |
| Char2Subword(EMNLP 2020) | 字符 transformer 学习重建子词嵌入表,训练后替换查表 | 训练范式同族(模仿非重建) | 中 |
| BLT hash n-gram(2412.09871) | n∈{3..8} 字节 n-gram RollPolyHash 查表求和 + cross-attn 池化(均值初始化 query) | 池化机制近;上下文相关、无重建、无冻结 | 中 |
| T-FREE(2406.19223,EMNLP 2024) | 空格预分词 + 字符 trigram 哈希行求和成词向量 | 同为字符代数组合替代查表;固定哈希 vs 可学习瓶颈 | 中 |
| MegaByte(2305.07185) | patch=字节嵌入无损拼接(维数随长度涨) | 最朴素组合;v2 是池化压缩到定维 | 低-中 |
| CANINE(2103.06874)/Charformer GBST(2106.12672) | 卷积下采样 / 可学习软切块+块内加权池化 | 无词表输入编码器口号重叠;GBST 软池化与 PMA 同族 | 中 |
| Perceiver IO(2107.14795) | 字节嵌入 cross-attn 进 256 latent(全局,非 per-patch) | 直系祖先但聚合成全局 latent 序列,MLM 训练 | 中 |
| **Set Transformer 用于字节编码** | **无先例**(阴性结果,UNVERIFIED-as-absence) | v2 是署名谱系空白 | — |

## 4. LTLM 族输入侧逐点名(Bolmo 框架:字节→局部编码器→池化→骨干)

| 模型 | 池化方式 | 训练 | 撞车 |
|---|---|---|---|
| H-Net(2507.07955) | 选择式(界处字节向量)+平滑映射 | 端到端 | 中 |
| H-Net++(2508.05628) | 均值池化+Gumbel 边界 | 端到端 | 中 |
| **Bolmo(2512.15586)** | 选择式(patch 末字节) | Stage-1 蒸馏预训练局部编码器,Stage-2 解冻 | **高**(深挖见专文) |
| AU-Net(2506.14761) | 选择式(词界处保留单向量) | 端到端 | 中 |
| FlexiTokens(2507.12720) | 均值池化 | 端到端 | 中 |
| **TFree-HAT(2603.15953)** | **学习 latent query cross-attn(PMA 同型)** | 端到端 | **高**(深挖见专文) |
| BLT(2412.09871) | cross-attn,均值初始化 query | 端到端 | 中 |
| MrT5(2410.20771) | 无池化(delete gate) | 端到端 | 低 |

## 5. 重建塑形嵌入 + 冻结供下游(交叉方向,详见分报告)

分报告 `33-lit-review-v2-reconstruction-frozen-ae.md` 要点:文本域冻结 AE→LM
的先例全部是上下文压缩(ICAE 2307.06945、500xCompressor 2408.03094、Kuratov
2502.13063——单 4096 维向量无损装 1568 token,容量上限存在性证明);连续非量化
冻结 AE 的硬先例在图像/语音(LDM、GIVT、MAR、NaturalSpeech 2);**"短字节串→
≤64 维→精确重建"除 CALM 外无文本域先例**;SMILES-VAE 的 latent 死区→非法串
失败模式与 v1 幽灵词同型(v2 的分量锚定结构性排除,可作卖点);Mezentsev &
Oseledets(EMNLP 2025)发现重建最优嵌入解空间不平滑——v2 分量内只取 μ_k+
回投影正是对冲。 [ICAE](https://arxiv.org/abs/2307.06945 "citation"), [Kuratov](https://arxiv.org/abs/2502.13063 "citation")

## 6. 撞车等级汇总(输入侧口径)

| 等级 | 工作 |
|---|---|
| 致命 | 无 |
| 高(深挖) | Kronecker Embeddings(2605.29459)、Bolmo(2512.15586)、TFree-HAT(2603.15953) |
| 中 | BLT hash n-gram、T-FREE、CANINE、Charformer、Perceiver IO、Mimick、Char2Subword、H-Net/H-Net++、AU-Net、FlexiTokens、ByteFlow(2603.03583) |
| 低-中 | MegaByte、MorphTE、SpaceByte |
| 低/无 | word2ket 系、TT/ALBERT/QR/DHE/Bloom、fastText 系、MambaByte、MrT5、MBLM、EvaByte |

## 7. 对 v2 定位的新增含义(相对主报告的增量)

1. **输入侧差异化卖点依然成立且更硬**:现有工作中没有第二个"可学习、重建塑形、
   上下文无关、极低维(32/48)、梯度硬切断"的字节串编码器。
2. **必引清单新增**:Kronecker Embeddings(2605.29459,related work 第一顺位划界)、
   Bolmo(2512.15586)、TFree-HAT(2603.15953)、word2ket(1911.04975,防混淆)、
   Mimick/Char2Subword(离线训练字符→向量映射的祖先)、MorphTE。
3. **可借力的新证据**:Kronecker 的跨模型探针(印刷变体聚类病理)支持 v2 的
   "BPE 嵌入几何被共现污染"叙事;其 124M 三种子正收益为字节组合嵌入的合法性背书;
   TFree-HAT 证明 PMA 同型池化可扩展到 70B 且"无新闻"(无需特殊稳定化)。
4. **新风险**:Bolmo Appendix E 的 rank 分析(子词嵌入本质高秩)会被评审引用质疑
   v2 的 32/48 维瓶颈——v2 需以自己的几何套件+重建 EM 实证回应(学习压缩 ≠ 线性
   截秩);HAT 系经字节 AR 链式法则也获得词级联合分布——v1 幽灵词叙事对 HAT 系
   的打击面要收窄到"词级采样控制/温度"与"词内串行代价"(详见深挖报告 §TFree-HAT)。
