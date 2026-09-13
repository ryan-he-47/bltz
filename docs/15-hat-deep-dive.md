# 15 — 新论文深挖:HAT(Hierarchical Autoregressive Transformers)

> **性质**:调研档案(2026-09-14)。用户提供 PDF(`reference_projects/2501.10322v2.pdf`);文本版已提取到同目录(`2501.10322v2.txt`,24 页 / 1721 行)。**本文所有行号引用(L###)均为该 .txt 行号**,发表引用前请对 PDF 复核(PDF 提取对变音字符/公式符号有损伤,如 `Björn` 被提取为 `Bj¨orn`)。
> **来源可信度**:全文由主会话 agent 逐行通读,关键引文逐行核对;无二手转述。
> **身份**:Pit Neitemeier, Björn Deiseroth, Constantin Eichenberg, Lukas Balles(Aleph Alpha Research, Heidelberg)。arXiv 2501.10322v2 [cs.CL],2025-01-20(v1 2025-01-17 / v2 2025-01-20;venue/他引/复现与团队后续见 §9)。
> **一句话**:字符级 encoder(词内)→ 词级 backbone → 字符级 decoder(词内**自回归**循环,[W] 停止)。与 bltz"压缩单元骨干 + 局部字节解码"同范式,重合度高于 BLT;而其全文 **0 次提及 BLT**。

## 0. 与 bltz 对照速览

| 环节 | HAT | bltz(我们) | 分野 |
|---|---|---|---|
| 分割 | 空白符切分(唯一非学习步骤,L174-177);附录 C 换 Unicode UAX#29+标点 | D3 确定性语义分割 / D5 目标模型自分割 | 都主张"分割可换";HAT 实测换规则 ~1k 步适应 |
| 编码 | 词内**双向**小 transformer,[W] 位取词嵌入(L182, L199-205) | Set Transformer patch 编码(D2,局部视野) | 同为"局部编码器"路线 |
| 骨干 | Llama 风格词级因果 transformer | patch 级标准 transformer | 同为"压缩单元骨干" |
| 解码 | 词内**因果** transformer,逐字符 AR 循环(L184, L253-259) | 条件神经场 (h,Δ)→byte,**并行**多未来字节 | **核心分野:AR vs 并行(D1 实验本体)** |
| 训练目标 | 逐字符 CE + [W] 终止符(L231-247) | MTP(k≤3,几何权重) | HAT 无 MTP;其 Outlook 点名 MTP/扩散解码为未来方向(L792-798) |
| 停止 | 预测 [W] 即停(L258-259);文档尾 [S] | space-like 词界自然停止 + 熵中断兜底(D11) | 均为"模型自产边界信号" |
| 推理形态 | 嵌套串行:词间串行 + 词内逐字符串行 | 目标:patch 内并行解码(待实验) | 我们的效率主张正面对照点 |

## 1. 身份与主张

- **Headline**(Abstract,L15-27):轻量字符级 encoder + 词级 backbone + 紧凑字符级 decoder;保留 word-level 压缩收益、无固定词表;7B 规模下打平 subword 模型;**对输入扰动显著更稳**;跨语言续训"almost twice as fast"、目标语更强、旧知识保留更多。
- **贡献**(L145-160):① 重访/精炼层次架构 + 成本分析 + 架构 sweep;② 算力匹配对照实验(含 MegaByte),scaling 到 7B;③ 鲁棒性;④ OOD/新语言的可微调性(续训快 2×)。

## 2. 模型架构(§2.1,L178-247)

- **三组件**(L181-188):encoder = "a **bidirectional** transformer operating on the character embeddings **within each word**";backbone = "a **causal** transformer operating on word embeddings";decoder = "a **causal** transformer with a language modelling head, operating on character level and outputting next-character prediction logits"。
- 字符嵌入 C、投影 W_E(D×d)、W_D(d×D);d < D(字符级维度小)。
- 每词 prepend 特殊符 [W](Following Devlin/BERT 做法,L190);词嵌入 e_i = encoder 在 [W] 位置的输出(L199-205);骨干输出经 W_D 得 p_i"predictive word embedding"(L207-216)。
- **训练**(L218-227):decoder 输入 = p_i 拼接**下一词**的字符嵌入(shift-by-one),输出 next-char logits;loss(L231-247)= 对下一词逐字符 CE + 末位目标 [W](词终止符;文档尾省略)。
- 单位:UTF-8 bytes,V_B=256;利用 UTF-8 未用字节值承载特殊符,词表保持 256(L171-172, L248-250)。
- **推理**(§2.2,L253-259):嵌套循环——先 encoder+backbone 得预测词嵌入;再 "run an **autoregressive loop** of the decoder module" 逐字符物化下一词;**预测 [W] = 词完成**,追加进输入、重复。图 1 另有 "Word Recursion / Character Recursion" 机制(L124-131:补全输入末尾不完整词后,完整词进入 encoder 递归推进;图示细节在文本提取中不可复原)。
- **参数量**(Table 2,L537-551):1B→backbone 18L/1.1B + en/de 6h3L/23M;3B→28L/4.3B + 24M;7B→36L/9.2B + 55M(baseline 分别 16L/1.1B、24L/3.1B、32L/7.0B)。
- **成本**(§2.3 + 附录 A.2/A.3/A.5):Eq.7/11 给精确 FLOPs 公式;词比 subword token 粗(S_W≈0.69·S_T,由 BPW 5.97 vs BPT 4.12 得出,Fig.2);compute-matching 偏差 <5%(A.3);step 时长(256×H100, batch 1024×16384B):1B 0.9s/0.9s,3B 2.3s/1.6s,7B 4.5s/4.7s(Table 4);3B 时两个字符模块算力 ≈ baseline 的 LM head 成本(L293-295);KV cache 减少 6-13%(A.5)。

## 3. 分词方法与解码停止(点名关注项)

### 3.1 分割规则

- "split the text at Unicode whitespace characters, which are appended to the previous word";"**the splitting rule is the only non-trainable processing step** in our method";且 "our hierarchical architecture is **agnostic to the type of splitting rule**"(L172-177)。
- 附录 C:Unicode splitter = Unicode Standard Annex #29 词边界(`uniseg` 包)+ 标点切分 + 合并前导空白/尾随标点(L1568-1584);Table 5:3B 上多数任务优于空白切分(MMLU 31.0 vs 28.7,LBD OAI 70.9 vs 70.7 等)。

### 3.2 停止机制

- 词级:decoder "When a word is completed, as indicated by the **prediction of a [W] token**"(L258-259)。
- 文档级:文档尾 dummy word(单 [S] 特殊符),训练时不预测其 [W];推理时 [S] 作 termination token(L1116-1120)。

### 3.3 中文实验(附录 C.3)

- SkyPile 续训(3B,5k 步):bpb **0.80 vs baseline 0.94**(Table 6);baseline 因分词器不匹配几乎全程字节回退(BPT≈1.02,而 Unicode 切分 BPW=4.29),同步骤算力 2.3×(L1621-1625)。
- **换分割规则实验**:英文阶段用空白切分、中文阶段换 Unicode——约 **1000 步**即追上全程 Unicode 版本(L1626-1634, Fig.10)。作者解读:"the backbone learns language-agnostic representations, while the encoder/decoder components can effectively map new byte sequences into this established embedding space"——**与我们"可学习 I/O 层"主张正面共鸣**。

### 3.4 推理优化(已提出、未实现)

- 高频词 embedding 查表(O(1) 替换 encoder 前向,设想覆盖 ≥95% 词)+ 存储预测词嵌入,匹配时 "could be used for **speculative decoding** or even be accepted as is if the match is 'good enough'"(L1246-1253)。
- KV cache 比 baseline 减 6-13%,OOD 场景更多(L1255-1269)。

## 4. 训练经验(§4 + 附录 B)

### 4.1 数据与规模

- 主实验 DCLM-Baseline(英文 curated);早期 sweep 用 Fineweb;续训用 Occiglot-DE(德语)、SkyPile(中文)。
- 72k steps ≈ **1.2T bytes**;doc ≤16,384 bytes(≈4k tokens / 2.7k words);batch=1024 序列;**dataloading 按 byte 对齐**保证两架构见同一数据(L429-438)。

### 4.2 超参

- AdamW β1=0.9 β2=0.95 ε=1e-8 wd=0.1;warmup 500;cosine→10%(L440-444)。
- LR 启发式:lr32=3e-4(32 头模型),lr(H)=32/H·lr32(逆宽度);自述 "not tuned... thus there might be room for further improvement"(B.1,L1274-1279)。

### 4.3 架构 sweep(§4.1 + B.3)

- en/de 同构(Le=Ld, He=Hd);aspect ratio 1:1 / 1.5:1 / 2:1 差异不显著→取 **2:1**(L1320-1328)。
- **核心权衡句**:字节准确率 favour 大 char 模块,词准确率 favour 大骨干;"byte accuracy can be improved by merely making the **decoder better at completing words given the first few characters**, which does not improve word accuracy"(L507-511)。采用**词准确率**为导航指标。
- 最终:(6h,3L)@1B/3B、(8h,4L)@7B;enc/dec 层数均分最优(L513-514, L1329-1334)。

### 4.4 主结果(L576-648)

- 全部 scale 与 tokenizer baseline 打平(Word Acc:35.5/35.3、37.8/37.7、39.0/39.2)。
- 亮点:Lambada(含 OAI-cloze)最高 **+68% relative**(7B,43.1 vs 25.6)。
- 对 MegaByte(1B)全面超越;同用 8-byte 分割的 hier 变体也比 MegaByte 好(byte acc +2.6ppt),但仍不如空白切分→ "a semantically meaningful splitting is a valuable inductive bias"(L637-648)。

### 4.5 鲁棒性(§4.4)

- 扰动:每词 10% 字符 permute/replace/delete + ALL CAPS;hierarchical **全面更稳**,ALL CAPS 下 baseline 跌幅 3×(L649-688)。
- 例子(Table 3):扰动 "brwon" 下模型仍输出 "fox jumps over the lazy dog";baseline 输出无关内容。

### 4.6 续训/适配(§4.5 + C.3/C.4)

- 德语续训:同 FLOP 预算下 **1.9× 快**、德语评分更高、英语保持更好(L739-771)。
- OOD 不训练直接推理 bpb 打平(Table 7);FLOP ratio(baseline/hier):Pile of Law 1.30 / SkyPile 2.32 / OpenWebMath 1.28 / GitHub Code 1.72(Table 8)。

### 4.7 评测口径(对字节模型非常实用)

- **词级准确率**定义(Eq.15-16, B.2):按词聚合——整词每字符预测都对才算该词正确;为 byte-vs-subword 跨粒度公平对比设计(L1281-1317)。
- **bpb** 定义(L1603-1609):按 UTF-8 字节数归一化,N 为模型处理元素数(hier=字节数;token 模型=token 数)。

## 5. 对 BLT 的提及与比较:**零**

- 全文检索 `BLT` / `Byte Latent` / `Pagnoni`:**0 命中**(仅 T-FREE 及其引用出现)。
- Related work(§3,L363-419)覆盖:字符/词级建模史、T-FREE(自家)、Sun et al. 2023(hierarchical MLM,~100M)、MegaByte(固定 patch、无 encoder、320M)、Thawani et al. 2023(每词 4 个 [W]、77M、未算力匹配)、Slagle 2024(=SpaceByte,滑窗 byte 层)、Charformer、Perceiver AR。
- **结论**:两篇高度重叠的工作(BLT 2024-12, HAT v2 2025-01-20)**互不引用**;HAT 的 byte-level 对照选的是 MegaByte。引用网络与领域分割的含义见 `docs/14`。

## 6. 值得参考/可直接借用的清单(按优先级)

1. **鲁棒性扰动评测套件**(10% permute/replace/delete + ALL CAPS;§4.4/B.5)——便宜、直击"字节 I/O 泛化"长板;建议复刻进我们的终评协议。
2. **跨域/跨语言续训协议**(德语 20k 步;中文 SkyPile 5k 步 + bpb;换分割规则适应实验)——"更易微调/可学习 I/O"主张的现成实验模板(其 1k 步换规则结论尤其可引用)。
3. **byte-vs-token 公平口径**:词级准确率(Eq.16)+ bpb 定义(L1603-1609)。
4. **compute-matching 方法论**(Eq.10-11 + 10k 文档统计近似,偏差<5%;A.2/A.3)+ 算力占比结论(3B 时 char 模块 ≈ baseline LM head)。
5. **enc/dec 尺寸配方**:6h3L≈23-24M(≤3B)/ 8h4L≈55M(7B);aspect 2:1;层数均分;小 char 模块+大骨干(以词准确率为准绳)。
6. **[W]/[S] 标记体系**:prepend token 取嵌入、loss 含终止符、推理见 [W] 即停——与我们 space-like 停止互为对照与备选方案。
7. **Unicode splitter(UAX#29/uniseg)**(L1568-1584):无空格语言直接可用;且实测"换分割规则 ~1k 步适应"。
8. **推理效率构想**:高频词 embedding 查表 + speculative decoding(未实现,L1246-1253);KV cache 减少 6-13%。
9. **训练配置细节**:lr(H)=3e-4·32/H、β2=0.95、warmup 500 + cosine→10%、按 byte 对齐的数据加载。
10. **Word Recursion**(L124-131):输入尾部不完整词的推理处理方式——"模型自产边界信号"设计空间里的另一种解法。

## 7. 局限与开放缺口(我们的机会)

- 作者承认:空白分割不适于表意文字(中文)/代码/数学(L783-788);compute-matched 下参数更多(推理 KV cache 部分抵消);已用 Unicode splitter 缓解。
- **Outlook 原文**(L792-798):"...the small character vocabulary, may facilitate **multi-token prediction (Gloeckle et al., 2024) with multiple output heads**. Further, with few characters per word, a **text diffusion model (e.g., Li et al., 2022) could be used as a decoder**."
  → **HAT 作者亲自把"MTP 多头"与"扩散式(并行)解码"留作未来工作,且自己没有做、没有对照。**
- 我们视角的缺口:词内 AR 循环 = 每词 len(word) 次串行解码步;无 MTP;分割仍是外部规则(非学习、非模型自产);无"并行解码 vs AR"消融。
- 与 D1(纯并行解码)的关系:我们的实验正是他们展望中的方向,但主张更激进——训练期多未来字节目标 + 推理期单次并行前向(而非扩散多轮迭代);且我们的查询解码头 (h,Δ) 无需"每词重复前向 encoder+backbone 以检查完整词"。

## 8. 数字速查

| 项 | 数值 | 位置 |
|---|---|---|
| 训练数据 | DCLM-Baseline,72k steps ≈1.2T bytes,doc≤16,384B | L429-438 |
| 规模配置 | 1B: 18L/1.1B+23M;3B: 28L/4.3B+24M;7B: 36L/9.2B+55M | Table 2, L537-551 |
| Step 时长(256×H100) | 0.9/0.9s;2.3/1.6s;4.5/4.7s(hier/baseline) | Table 4, L1192-1203 |
| Word Acc | 35.5/35.3;37.8/37.7;39.0/39.2 | Table 1, L596-623 |
| Lambada(OAI-cloze) | 7B: 43.1 vs 25.6(+68% rel) | Table 1, L619-623 |
| 德语续训 | 1.9× 训练加速 | L769-771 |
| 中文(SkyPile)bpb | 0.80 vs 0.94;算力 2.3×;换规则 ~1000 步适应 | Table 6, L1612-1634 |
| OOD FLOP 比(baseline/hier) | 1.28–2.32 | Table 8, L1708-1720 |
| KV cache | 减 6-13% | L1255-1269 |
| 词/token 压缩比 | S_W≈0.69·S_T(英文);中文 BPW 4.29 vs BPT 1.02 | Fig.2, L317-361 |

## 9. Outlook 兑现审计(2026-09-14 更新)

**问题**:HAT 团队(Aleph Alpha)自己在约 1.5 年里,有没有实现本文 Outlook 的三项?
**结论:三项均未由团队实现**;正牌后继是规模化/工程化路线;第三方已占位其中 (i) 与并行解码方向。

**9.1 HAT 本体版本**:仅 v1(2025-01-17)→ v2(2025-01-20),**无 v3+**;两版 PDF 同为 1,550 KB(v2 为标题多余波浪号微修)。S2 他引 24 条(调查时);OpenAlex 该 DOI 记录 cited_by=0 且引用解析为 0(不可信,调查中预算耗尽);未找到同行评审 venue。

**9.2 团队 2025–2026 相关论文**

| 论文 | arXiv | 日期 | 与 HAT 的关系 |
|---|---|---|---|
| A Family of LLMs Liberated from Static Vocabularies(36 人,含四位 HAT 作者) | 2603.15953 | 2026-03-16 | **正牌后继**:同 HAT 架构(encoder→backbone→decoder),7B 从零训(近 4T 词)+ Llama-3.1 8B/70B 转换 + EN/DE SFT/DPO;HF 发布含 200 ckpt。摘要原文仍是 "a classical autoregressive transformer" backbone、decoder cross-attend 后 "converted back into bytes"——**未实现 MTP/扩散/更深层级**;唯一 future work = 其它领域 |
| SOMBRERO | 2601.22805 | 2026-01-30 | 同团队(Neitemeier/Balles 等)层级模型边界放置研究;future work 明确含 "extending the analysis to **deeper hierarchies**" |
| GermanWeb / LIME / Good-Pretraining-Bad-SFT 等 | 2505.00022 / 2512.07522 / 2609.08966 | 2025–2026 | 同作者,与本 Outlook 无关 |

**9.3 三项兑现状态**

| Outlook 项(HAT v2 §5 原文) | 团队自己 | 第三方 |
|---|---|---|
| "multi-token prediction … with multiple output heads" | ✗ 未做 | **DMBP 2608.15454**(层级 LM 段对齐并行多字节,单 LCA 掩码头)、MTPC 2511.11346、EvaByte 8 头 |
| "a text diffusion model … could be used as a decoder" | ✗ 未做 | **Fast BLT-D/DV 2605.08044**(块扩散+验证)、Zonkey 2601.21768 |
| "additional levels of hierarchy, such as sentences or paragraphs" | ✗ 未做(仍列 future work) | 未检索到实现 |

**9.4 发布物与背景**:HF `Aleph-Alpha/tfree-hat-pretrained-7b-base`(Open Aleph License,非商用;encoder 119M + backbone 6.98B + decoder 94M = 7.19B;~4T 词;200 中间检查点)、`llama-3_1-8b-tfree-hat-{base,sft,dpo}`、`llama-3_1-70b-tfree-hat-sft`、`tfree-research-*`、`umup-research-*`、`sp-baseline-research-*`;另有 `Aleph-Alpha-Research/Qwen3-1.7B-Hatified-v0-A..G`。官方 GitHub(Aleph-Alpha)无 HAT 训练代码;第三方小复现 `FBR65/no_tokenizer`(1 star);据 HN 转引路透,Cohere 于 2026-04 收购 Aleph Alpha(未一手核)。

**9.5 对定位的含义**:并行解码窗口仍在,但已非空地(见 `docs/14` §4:Fast BLT-D、DMBP 已占位);团队 1.5 年留白说明该方向对其"规模化/落地"路线优先级低——既是机会,也是"难或低收益"的信号。

## 附:引用与工具

- 文本版:`reference_projects/2501.10322v2.txt`(2026-09-14 由 cose2 python/fitz 提取,24 页;行号即此文件)。
- 同族对照文本版:BLT `2412.09871v1.txt`、Fast BLT `2605.08044v1.txt`(同目录)。
- 关联档案:`docs/14`(范式脉络与重新定位)、`docs/13`(MTP 冲突先例)、`docs/03`(2026-09-08 文献调研)、`docs/blt_research/ByteField用户立场归档.md`(定位立场)。

(完)
