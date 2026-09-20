# v2 两组件文献定位报告（2026-09-21，文献调研员）

调研对象：v2 架构的两个组件——(A) 增强 BPE 随机分割（双分词器投票 + 概率切 + 离线预烘焙）；
(B) 任意字节串稠密嵌入 AE（Set Transformer 编码 ≤32 字节串 → 32-48 维，重建塑形，零次桶 EM 93.6%）。
事实来源均为本次实际抓取的页面；抓不到的标注 UNVERIFIED。

---

## 组件 A：增强 BPE 随机分割

### 谱线 1：随机/正则化分词（subword regularization 家族）

| 论文 | 方法一句话 | 与 v2 组件 A 的异同 | 撞车等级 |
|---|---|---|---|
| Kudo 2018, Subword Regularization（ACL 2018）https://aclanthology.org/P18-1007/ | 用 unigram LM 诱导的分割分布，训练期每次采样不同分割，把分割歧义当正则化噪声 | 同：都是"一条文本多种分割"增强嵌入/模型见过的字符串多样性。异：采样分布由单 tokenizer 的概率模型诱导（偏向高频分割）；v2 是双 tokenizer 投票 + 独立抛硬币，分布形态完全不同；v2 目的是塑形 AE 嵌入空间而非提升 LM 鲁棒性 | 中 |
| BPE-dropout（Provilkov et al. 2020, ACL）https://aclanthology.org/2020.acl-main.170/ | 在 BPE 合并过程中以概率 p 随机跳过合并，同一词产生多种分割，推理回归确定性 BPE | 同：随机化分割、单遍掷骰思想。异：随机性注入点在合并过程（分布有偏，见下）；v2 在字符间隙级独立伯努利，且以另一套分词器的意见为锚 | 中 |
| MaxMatch-Dropout（Hiraoka 2022）（转引自 https://arxiv.org/html/2409.06216v1 与 ETH 论文） | 最长匹配分词时随机丢弃最长匹配、退到次长 | 机制类似（对候选边界做随机否决），但仍是单 tokenizer 内部扰动 | 低 |
| Distributional Properties of Subword Regularization（ETH Research Collection，作者未在抓取片段显示，UNVERIFIED 作者）https://www.research-collection.ethz.ch/server/api/core/bitstreams/90a23562-cdd1-4890-901e-60af6ad2a5b4/content | 证明 BPE/MaxMatch-dropout 的分割分布天然有偏（Lemma 3.1：不存在使分布均匀的 p） | 对 v2 是**反面激励**：dropout 系方法分布不可控；v2 "一致必切/不一致 p=0.5" 的分布在两个 tokenizer 分割集合的交集上有锚点、争议边界上等权，更透明可控 | 低（佐证而非撞车） |
| Multi-view Subword Regularization（Wang, Ruder, Neubig 2021, NAACL）https://aclanthology.org/people/g/graham-neubig/ （摘要页确认） | 同一 tokenizer 的确定性分割 + 概率分割双视图，强制两种输入下预测一致性（KL），XTREME +2.5 | **最接近"双分割"概念的先例**。异：两个视图来自同一 tokenizer 的确定/随机模式，不是两个独立 tokenizer 的投票；且是训练期一致性正则，不是边界合并规则 | 中偏高（必须引用并在文中划清界线） |
| StochasTok（Sims et al. 2025, arXiv 2506.01687）https://arxiv.org/html/2506.01687v2 | 先用确定性 tokenizer 分一次，再离线做随机"扩展"（把 token 随机拆成词表内的更短对），得到大量重分割副本；语言游戏任务 25%→97%，且能在没见过的分词方式下 grok 加法 | **组件 A 的最强撞车点**：同样是离线一次性随机化、同样产出多粒度分割、同样主张"多见分割→字符级理解"。异：单 tokenizer 词表内的随机拆分 vs 双 tokenizer 投票；下游是标准 token LM vs 字符串 AE；v2 的 unit 可跨 BPE 词表（规则侧切出的整词不必在 BPE 词表中） | **高** |
| Saleva & Lignos 2023, "What changes when you randomly choose BPE merge operations? Not much."（转引自 https://arxiv.org/html/2402.18376v1 参考文献） | 随机化 BPE 合并本身几乎不改变什么 | 对 v2 是警讯类证据：随机化如果只在 BPE 框架内打转，收益有限；v2 的规则侧预分词引入了 BPE 框架外的切割模式，可引此作动机 | 低 |
| Hiraoka et al. 2021, Joint optimization of tokenization and downstream model（转引自 https://arxiv.org/html/2402.14614v2 参考文献） | 分词与下游模型联合优化 | 与 v2 无关方向（v2 分词器冻结、预烘焙） | 无 |
| Progressive and Consistent Subword Regularization（NLPCC 2024，转引自 https://dl.acm.org/doi/abs/10.1007/978-981-97-9437-9_25） | 渐进+一致性子词正则化 | 谱系延续，机制无新意 | 无 |

### 谱线 2：多 tokenizer 集成/投票分割

| 来源 | 内容 | 与 v2 的异同 | 撞车等级 |
|---|---|---|---|
| Mixture of Tokenizers（snimu 博客提案，2024-09-03，非正式）https://snimu.github.io/2024/09/03/mixture-of-tokenizers.html | 用字符级 + 子词级（甚至多个语言专用子词）tokenizer 并行提供多视角嵌入 | **概念上最接近"多 tokenizer"**：多个 tokenizer 的"多视野"。异：是嵌入层融合而非分割投票；非正式博客，无实验定论 | 中（概念邻居，必须提及） |
| Auxiliary Subword Segmentations as Related Languages（EAMT 2022）https://protonish.github.io/assets/pdf/multisub_eamt2022.pdf | 把同一文本的多套子词分割当"亲属语言"做多语言式迁移 | 用多套分割增强训练，但各分割独立处理、不投票合并 | 低-中 |
| Takase et al. 2022（转引自 https://arxiv.org/html/2409.06216v1 与 https://www.emergentmind.com/topics/subword-regularization） | 推理期对同一模型喂多个分割候选做集成（K+1 分割边缘化），低资源 +0.2-0.3 BLEU | 分割多样性用于集成而非数据生成 | 低 |
| GAC 跨 tokenizer 模型集成（转引自 aau.dk 硕士论文 https://projekter.aau.dk/projekter/files/785150816/Masters.pdf） | 多模型不同词表下通过词表映射聚合下一 token 概率 | 模型级集成，与分割机制无关 | 无 |
| Cross-Tokenizer LLM Distillation through a Byte-Level Interface（arXiv 2604.07466, 2026）https://arxiv.org/html/2604.07466v2 | 用字节级模型作为接口在不同 tokenizer 的 LLM 间蒸馏 | 佐证"字节层是多 tokenizer 的天然公共接口"——与 v2 用字节串 AE 替代词表的思想同向 | 低（同向佐证） |

**结论**：检索范围内**未检到"两个独立 tokenizer 对边界投票（一致必切/分歧概率切）"的先例**（否定性结论无法被检索完全证明，标注 UNVERIFIED-as-absence）。最近的邻居是 MVR（双视图一致性）和 Mixture of Tokenizers（多 tokenizer 嵌入融合）。

### 谱线 3：离线预烘焙 vs 在线采样

- **StochasTok 是最直接的离线先例**：论文明确说"tokenize the data only once and then cheaply 'detokenize' or expand the dataset for various levels of stochasticity"（先确定性分一次，再离线随机扩展），且明确宣称相对 subword regularization 的优势就是不用重复跑分词（https://arxiv.org/html/2506.01687v2；OpenReview PDF https://openreview.net/pdf/98a0455e39c46f67bc7035b9e61c30c8fd1eb717.pdf 同段确认）。**v2 的 pre-baked 缓存与此同构**——撞车等级：高（离线随机化先例存在），但"单遍训练下与在线随机等价"这一论证未在该文中形式化讨论（UNVERIFIED：未检到专门讨论离线/在线随机分割等价性的论文）。
- 一般数据增强文献中 online vs offline 的等价性讨论散见（如 https://trepo.tuni.fi/bitstream/10024/229398/1/Towards_Length_Versatile_and_Noise_Robust.pdf 明确"离线增强的样本数 = 训练样本数、online 为 steps×batch"），但这是 CV/信号领域口径，非分词专门讨论。
- Deterministic Reversible Data Augmentation for NMT（arXiv 2406.02517）https://arxiv.org/html/2406.02517v2 ——确定性可逆增广，精神相反（去掉随机性）但属同一问题域，可作对照引用。

### 谱线 4：2025-2026 tokenizer-free / dynamic tokenization 盘点

| 工作 | 一句话 | 与 v2 关系 | 撞车等级 |
|---|---|---|---|
| H-Net（Hwang et al. 2025, arXiv 2507.07955）https://arxiv.org/abs/2507.07955 | 端到端动态分块层级网络，字节级单阶段即在 >1B 规模追平 BPE Transformer；分块自动收敛到 4.5-5 字节/块 | v2 是"离线 tokenizer + 嵌入空间 AR"，H-Net 是"在线学习分块"——路线对立但目标相同（消灭固定词表）；H-Net 的字节鲁棒性结果是 v2 的对照锚 | 中（路线竞品） |
| H-Net++（arXiv 2508.05628）https://arxiv.org/pdf/2508.05628 | H-Net 在波斯语等富形态语言上的扩展（上下文感知路由+文档级超先验） | 同上 | 低 |
| dnaHNet（arXiv 2602.10603）https://arxiv.org/pdf/2602.10603 | H-Net 迁移到基因组序列，动态分块跨模态有效性 | 佐证动态分块泛化性 | 无 |
| BLT（Pagnoni et al. 2024/2025, arXiv 2412.09871）+ Fast BLT（ICML 2026 接收，见 https://juliekallini.com/ 新闻栏） | 熵驱动的动态字节 patch；Fast BLT 修复推理期重熵计算 | v1 已引用；v2 的"边界由外部规则+概率决定、嵌入由 AE 重建塑形"与 BLT 的"边界由熵决定、patch 嵌入上下文相关"互补 | 中 |
| MrT5（Kallini et al., ICLR 2025, arXiv 2410.20771）https://huggingface.co/papers/2410.20771 | ByT5 编码器内学习删除门动态合并字节 | 已知存档 | 无 |
| FlexiTokens（Owodunni et al. 2025，转引自 https://arxiv.org/html/2606.20993 与 2601.22805） | 学字节级边界形成变长段，强调分布漂移下分割自适应 | 与 v2 的"分割多样性"动机相邻 | 低 |
| MAGNET（Ahia et al. 2024，转引同上） | 脚本特定的边界预测器，跨书写系统等粒度 | 低 | 低 |
| SuperBPE（Liu et al. 2025, arXiv 2503.13423）https://arxiv.org/abs/2503.13423 | 预分词课程：先学词内子词再学跨空格"超词"，-33% token 数、+4.0% 下游 | **与 v2 规则预分词的思想相邻**：都用 space 作为分割先验；SuperBPE 最终跨空格、v2 以 space 为必切锚。规则预分词 + BPE 的组合先例存在（但非随机化） | 中 |
| T-FREE（Deiseroth et al. 2024, arXiv 2406.19223, EMNLP 2024）https://aclanthology.org/2024.emnlp-main.1217/ | 字符三元组哈希稀疏激活直接嵌入整词，无词表，嵌入层 -85% 参数 | 属组件 B 谱线（见下） | — |
| Bolmo（Minixhofer et al. 2025，转引自 https://www.arxiv.org/pdf/2601.22805） | 把预训练子词 LM 蒸馏改造成字节级 | 改造路线，与 v2 从头训练无关 | 无 |
| AU-Net（Videau et al. 2025，转引自 https://arxiv.org/html/2604.07466v2） | 固定规则多尺度池化字节 | 低 | 低 |

---

## 组件 B：任意字符串的稠密嵌入

### 谱线 5："任意字符串 → 固定维向量"文献

| 工作 | 方法一句话 | 与 v2 AE 的异同 | 撞车等级 |
|---|---|---|---|
| Kim et al. 2016, Character-Aware Neural Language Models（AAAI 2016, arXiv 1508.06615）https://roomylee.github.io/character-aware-lm/ | charCNN + highway 把任意词编码为向量喂词级 LM | 同：开词表、从字符组合出词向量。异：无重建约束，嵌入形状由下游 LM 目标间接塑形；不保证可逆/可解码 | 低 |
| char2vec（Cao & Rei 2016）、Mimick（Pinter et al. 2017）（转引自 https://link.springer.com/article/10.1007/s13278-021-00777-5） | 双向 LSTM 组合字符为词向量；Mimick 训练字符模型"模仿"已有词向量以覆盖 OOV | Mimick 与 v2 最接近的**老**先例：目标就是给训练中没见过的词产向量。异：模仿的是别人训练好的 embedding（语义空间），不是重建塑形；无可解码性 | 低-中 |
| fastText（Bojanowski et al. 2017） | 子词 n-gram 均值组合任意词向量 | 组合式但无神经编码器、无重建 | 低 |
| CANINE（Clark et al. 2022）、ByT5（Xue et al. 2022） | 无词表字符/字节级 Transformer | 编码的是上下文表示不是独立字符串向量 | 低 |
| T-FREE（arXiv 2406.19223）https://arxiv.org/abs/2406.19223 | 哈希字符三元组稀疏激活模式 = 词嵌入，跨语言 fertility 恒定 | 同：任意字符串→嵌入、无词表。异：固定哈希结构（不可学习几何）、嵌入极高维稀疏、不追求重建 | 中 |
| BLT hash n-gram embeddings + local encoder（arXiv 2412.09871 §3.2.1）https://arxiv.org/html/2412.09871v1 | 字节 n-gram 哈希嵌入 + cross-attention 把 patch 字节聚成 patch 表示 | **机制最近邻**：都是"把变长字节段压成向量"。异：BLT 的 patch 表示是上下文相关的中间态，不是独立可解码的字符串嵌入；无重建目标、无零次泛化评估 | 中（必须引用划界） |
| HashFormers（EMNLP 2022）https://aclanthology.org/2022.emnlp-main.536.pdf | 哈希 + 微型码本把无限词表压进小嵌入矩阵 | 嵌入压缩路线，无重建 | 低 |
| Backpack LM（Hewitt et al. 2023；中文版 arXiv 2310.12751）https://arxiv.org/abs/2310.12751 | 字符 sense 向量对数加性组合成词义；中文 Backpack 词表示优于 Transformer 字符嵌入 | 组合式语义嵌入，无重建、非瓶颈向量 | 低 |
| ZeTT（Minixhofer et al. 2024, arXiv 2405.07883，转引自 https://futureagi.com/blog/what-is-tokenization-llms-2026/） | 超网络为任意新 tokenizer 的 token 生成嵌入（零次 tokenizer 迁移） | 同：给任意字符串产嵌入且测零次。异：学的是"模仿原嵌入"，且评测是下游迁移而非重建 | 中（零次字符串嵌入的最接近先例之一） |

### 谱线 6："未见字符串的重建/泛化"作为指标

- **未检到以"训练中从未出现的字节串的 exact-match 重建率"为 headline 指标的 NLP 先例**（UNVERIFIED-as-absence）。最近的类比：
  - **vec2text / 嵌入反演**（Morris et al. 2023，The Gradient 综述 https://thegradient.pub/text-embedding-inversion/）：证明长度 ≤32 的文本大多可从嵌入完美反演恢复——**间接支持 v2 的 32 字节上限 × 低维嵌入的信息论可行性**，但方向相反（反演攻击 vs 主动设计可解码嵌入），且反演对象是训练域内嵌入模型。撞车等级：低，但**必须引用**（32 这个数字撞得很巧）。
  - **汉字笔画自编码器零次识别**（MDPI Applied Sciences 2023）https://www.mdpi.com/2076-3417/13/3/1750 ：SAE 预训练后可重建训练中没见过的汉字笔画序列并用于零次识别——**"重建未见结构"作为零次能力的先例**，视觉领域。撞车等级：低（跨模态类比）。
  - **CoSE 的 D 维消融**（本地存档 cose_paper_full_text.txt；https://arxiv.org/pdf/2006.09930 Table 2）：D 越大重建越好但预测与 SC 越差——v2 32-48 维选择的直接先例，已是内部引用。
  - **StochasTok 的跨分词泛化**（"tested with questions tokenized with methods not seen during training"，arXiv 2506.01687 §5）：未见分割上的泛化测试，与 v2 零次桶精神相通但测的是 LM 行为而非字符串重建。

### 谱线 7：Set Transformer / Perceiver 做字符串/变长序列编码（2024-2026 新增）

| 工作 | 内容 | 撞车等级 |
|---|---|---|
| CoSE（NeurIPS 2020，本地存档）https://proceedings.neurips.cc/paper/2020/file/723e8f97fde15f7a8d5ff8d558ea3f16-Paper.pdf | 变长笔画序列→固定维 λ，Transformer 编码器 + GMM 解码器，SC 指标 | 已吸收（v2 祖先）；注意 CoSE 自己消融发现**把笔画当集合（去位置编码）显著变差**（§7.1 变体 3），佐证 v2 保留 RoPE 的决定 |
| Perceiver / Perceiver IO（2021，字节数组直接 cross-attend）https://towardsdatascience.com/from-set-transformer-to-perceiver-sampler-2f18e741d242/ | latent 查询 cross-attention 压任意长字节数组 | 老先例已存档 |
| ASPIRE（2025，表格开放世界建模，转引自 https://arxiv.org/html/2603.13308v1） | Set Transformer（ISAB×2，32 inducing points）+ **GMM 头** 做连续目标 | 低：结构巧合地与 v2 "Set Transformer + MDN/GMM" 同形，但域完全不同（表格） |
| CASE（2026，推荐系统）https://arxiv.org/html/2604.06718v3 | ISAB 编码无序物品集合 | 无（域外） |
| TabICL（2025）https://www.emergentmind.com/topics/tabicl-base | ISAB 做列级超网络 | 无（域外） |
| BLT local encoder（2412.09871 §3.2） | max-pool 初始化查询 + cross-attention 聚 patch 字节 ≈ Perceiver 式字符串压缩 | 中（见谱线 5） |

**结论**：2024-2026 未检到用 Set Transformer 对**任意字节串**做重建塑形嵌入的新工作；该组合（ISAB 编码 + 极小瓶颈 + 重建 CE + EOS 行）在检索范围内保持新颖（UNVERIFIED-as-absence）。

---

## 总结

### 组件 A 新颖度判断：**中高**
- 随机分割正则化谱系拥挤（Kudo 2018 → BPE-dropout → MaxMatch-Dropout → MVR → StochasTok），"用多样分割增强训练"不是新主张。
- 但三个具体设计点叠加后未见先例：①**双 tokenizer 投票**（一致必切/分歧 p=0.5，分布锚定在两个分割集合的交集上，规避了 dropout 系的分布有偏问题）；②**规则侧引入 BPE 框架外的整词切割**（跨词表字符串多样性，恰好补 Saleva & Lignos 2023 指出的"BPE 内部随机化收益有限"）；③**喂给重建塑形 AE 而非 token LM**（目的不同：嵌入空间的字符串覆盖而非 LM 鲁棒性）。
- 离线预烘焙有 StochasTok 直接先例（撞车高），需在文中显式区分并引用。

### 组件 B 新颖度判断：**高**
- "任意字符串→向量"本身谱系很长（Kim 2016 → Mimick → CANINE → T-FREE → ZeTT），但全部是下游目标塑形或模仿已有嵌入。
- **重建塑形 + 极小瓶颈（32-48 维）+ 零次桶 EM 作为 headline 指标**的组合未检到先例；最近邻 CoSE（已吸收）在笔画域，ZeTT 测零次但不重建，vec2text 证明 32 长度级文本可逆但属反演攻击。
- Set Transformer × 字符串编码 × GMM/MDN 头的组合在 2024-2026 文献中未见（ASPIRE 是结构巧合的域外邻居）。

### 必须引用清单（按优先级）
1. **StochasTok**（arXiv 2506.01687）— 组件 A 最近撞车点，离线随机分割 + 多粒度混合 + 未见分词泛化。必须正面区分。
2. **Kudo 2018**（ACL, aclanthology.org/P18-1007）与 **BPE-dropout**（ACL 2020, aclanthology.org/2020.acl-main.170）— 随机分词始祖。
3. **Multi-view Subword Regularization**（Wang et al. 2021, NAACL）— "双分割视图"最近邻。
4. **Distributional Properties of Subword Regularization**（ETH，URL 见谱线 1）— dropout 分布有偏证明，v2 设计动机。
5. **Saleva & Lignos 2023**（"Not much"）— BPE 内部随机化收益有限的反面证据，支撑规则侧设计。
6. **CoSE**（NeurIPS 2020）— 组件 B 祖先（已存档）。
7. **BLT**（arXiv 2412.09871）— patch 字节压缩最近机制邻居 + hash n-gram。
8. **vec2text / The Gradient 反演综述** — 32 字节 × 低维可逆性的间接证据。
9. **ZeTT**（arXiv 2405.07883）— 零次字符串嵌入最近先例。
10. **H-Net**（arXiv 2507.07955）与 **SuperBPE**（arXiv 2503.13423）— 动态分块竞品路线 + space 预分词先例。
11. **Mixture of Tokenizers**（snimu 博客 2024）— 多 tokenizer 概念邻居（非正式来源，注明为博客）。
12. **T-FREE**（arXiv 2406.19223）— 无词表任意字符串嵌入对照。
