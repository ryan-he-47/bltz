# 33 补二 — 高相似工作深挖:Kronecker Embeddings / Bolmo / TFree-HAT(2026-09-21)

> **性质**:第二波调研的 Phase 2。对字节嵌入层专项(见
> `33-lit-review-v2-byte-embeddings-survey.md`)筛出的三篇高相似工作逐篇全文深挖,
> 统一按"创新点 / 参考价值 / 撞车部分 / 撞车程度"四段式。
> 全部事实来自本轮实际抓取的 arXiv HTML 全文/官方仓库/模型卡,附 URL;
> 抓不到的标 UNVERIFIED。不修改任何既有文档。

---

# 一、Kronecker Embeddings(arXiv:2605.29459)

Shravan(The School of AI)独作,2026-05-28,28 页 16 表;参考实现 Apache 2.0
(PyPI `kronecker-embeddings`),**但 124M 训练 fork 标注 forthcoming,主结果当前
不可复现**。 [arXiv HTML](https://arxiv.org/html/2605.29459v1 "citation"), [GitHub](https://github.com/theschoolofai/kronecker-embeddings "citation")

## 1. 创新点(作者五条贡献的核实)

1. **方法本体**:字节 one-hot ⊗ 位置 one-hot 的确定性 codec(D=4096/8192)+ 逐
   token z-norm + 唯一可学习层 `Linear(D, d_model, bias=False)`。支撑充分(定义性
   贡献)。
2. **跨模型探针**(135M-671B×6 模型):训练后嵌入聚类"印刷变体"而非词形亲属
   (DeepSeek-V3-Base 的 `run` top-5 邻居全是自身大小写/空格变体);Kronecker 在
   嵌入层即跳出(loose morph@K 0.92 vs BPE 0.54)。**全文最扎实的贡献**;作者诚实
   声明两臂严格词形检索都 <30%。
3. **受控训练对比**(nanoGPT 124M × FineWeb-Edu 2.5B × 三种子):val loss
   -2.5±0.2%(0.083±0.007 nats),18/18 格子全胜,样本效率 1.43×。**重大混杂:
   对照臂是 tied-BPE 而非 untied-BPE——优势里未知比例来自 untying 本身**,作者
   承认是最尖锐审稿意见但未量化。
4. **typo 鲁棒探针**(110 对×11 类):top-1 保持 55.5% vs 47.3%(+8.2pp)、
   KL -7.6%;但**单种子**,作者自降证据等级为 Moderate。
5. **范数稳定 + gpu_dynamic 部署**:投影 std 稳定 ~1.0(BPE 嵌入漂移 +30%);
   部署开销 0.01-0.24% 来自内部 9B/120B 配置,**不可复现**,自评 Weak。

证据强度呈阶梯:探针(强)> 三种子对比(强但混杂)> typo(中)> 部署数字(弱)。
作者的证据分级自认(§8.1)少见地诚实。

## 2. 参考价值(对 v2 逐项)

| 对方资产 | 对 v2 适用性 |
|---|---|
| typo 鲁棒探针协议(110 对×11 类×4 度量:top-1 保持/KL/隐态 cos/Δlogp) | **高价值可直接移植**:v2 的 typo 簇增强需要行为级评估;v2 可多种子直接超越其单种子弱点 |
| 印刷变体聚类探针 + canonical_form 清洗函数 | 中高:可作 v2 λ 空间几何新探针——学习出来的 λ 应介于"纯 byte-locality"与"纯共现"之间,**测出来本身是卖点** |
| "消歧推给第一层注意力"的诚实写法(compute/commute cos 0.86) | **高价值修辞模板**:v2 的 32/48 维 λ 必有字形碰撞,审稿人必问;"两种方法都靠下游消歧,只是归纳偏置不同"可直接化用 |
| 诚实参数记账(承认 124M 净 +3.1M) | 方法论标杆:v2 报参数账也应给"输入侧/输出侧/总量"三栏 |
| forced-OOV 编码 + "124M 模型用不好该能力"的定性发现 | 预警:**输入侧能编码 ≠ 模型会利用**;v2 零次桶 EM 93.6% 是编码侧,骨干利用侧需单独验证 |
| token-bucket NLL 分析(对方 future work) | v2 已做(频率分层 val/零次桶)——**v2 领先点**,写作时对照 |

## 3. 撞车部分(逐维)

分词器依赖(对方强依赖 BPE,自述 not tokenization-free)、编码器哲学(固定 codec
vs 可学习 AE)、训练目标(标准 CE vs 重建+MDN NLL)、表征维度(4096→768 vs
32/48 瓶颈)、输出侧(标准 softmax vs GMM+AE 解码)、卖点(参数效率 vs
vocab-free/词级温度)——**全部正交**。轻度撞点仅:≤32 字节上限的数字共识、
字节×位置分解基元(双方都拒绝 bag-of-bytes)。

**§8.5 tied-head Kronecker decoding(唯一实质撞点)**:对方两条**明确标注未测试**
的假设——Hypothesis A:输出头预测 D 维向量匹配 codec,反解 Kronecker 结构一次性
并行解出 d_p 个字节分布;Hypothesis B:预测 codec 空间的**单**对角高斯(μ,σ²),
NLL 训练,"in the spirit of VAEs"。
**v2 = Hypothesis B 的已实现增强版**:学习的目标空间(AE λ 而非手工 codec)、
K=64 混合(单高斯无法表示多模态下一词分布——v1 的 cat/dog/dot 在输出侧同样
致命,写作时可点名)、词级 π-温度(对方全文无采样温度讨论)、EOS 终止(对方靠
空槽位)、60k POC+1B 长训实证。且 v1 的 σ 坍缩事故(574046)恰好实证了对方
担心的 "training stability is unknown"。

## 4. 撞车程度:**低-中**

主体不撞(全栈正交);§8.5 仅为假设且是单高斯,审稿人最多要求引用,不构成
scoop。related work 建议主动对位:"concurrent work hypothesizes (without experimental
validation) a single-Gaussian output head over a fixed codec space; bltz instantiates and
strengthens this direction — learned latent space, K-component mixture, word-level
temperature, explicit EOS — trained and evaluated at scale"。不要踩的坑:别称对方为
byte-level LM(其 §1.5 明确反驳);别把 91-94% 输入侧削减当全模型削减引用。

## 5. 附:社区反响

Semantic Scholar 被引 0;唯一引用方是作者自己的 LightningLM(2606.07404);
第三方仅聚合器级报道。 [2606.07404](https://arxiv.org/pdf/2606.07404 "citation")

---

# 二、Bolmo(Ai2,arXiv:2512.15586v2)

2025-12-17(v2 2026-02);全开源(Apache-2.0 代码 + HF 权重 + 数据)。
byteify 路线:把 OLMo 2/3 用 <1% 预训练预算(49.1B tokens)改造成字节级 LTLM,
Bolmo-7B 接近 OLMo-3-7B、字符任务大幅反超。 [arXiv HTML](https://arxiv.org/html/2512.15586v2 "citation"), [Ai2 Blog](https://allenai.org/blog/bolmo "citation"), [bolmo-core](https://github.com/allenai/bolmo-core "citation")

## 1. 创新点

- **残差子词后缀嵌入**:字节嵌入 += "以当前字节结尾的最长子词"的查表向量——
  廉价稀疏激活扩容,作者自述非必要(加大局部编码器可替代)。
- **非因果边界预测器(最大架构改动)**:用未来 1 字节判边界(投影余弦距离)。
  动机精彩:subword 分词器本身就是非因果的("_Wor" 在 "_Hello_Wor!" 中是 token、
  在 "_Hello_World!" 中不是),因果边界预测有原理性表达力缺口。消融:因果边界
  全面崩盘(42.5 vs 非因果 55.7)。
- **边界符号融合**:词表 256→512(每字节一个"后随边界"变体),输出侧边界预测
  零序列开销。
- **选择式池化 + 同维局部/全局**:取 patch 末字节向量;刻意取消 H-Net 的线性
  上投影——Appendix E rank 分析证明线性上投影硬性截秩。
- **mLSTM 局部编码器**:选型按 wallclock 而非 FLOPs(实测两者 R² 仅 0.63-0.66),
  据此批评 BLT/H-Net 的 FLOP-matching 口径。
- **两阶段 byteify**:Stage-1(冻全局,9.8B tokens)四项损失,关键是 `L_E`:
  **池化表示先过全局模型前 n=4 冻结层再与"子词嵌入过同样 4 层"做 L2**(n=4 显著
  优于 n=0;灵感来自 model stitching——表示相似不保证在后续层传播相似);
  `L_D,Distill` 为 patch 级似然蒸馏(有输出边界预测时是精确目标)。Stage-2 全解冻。
- 输出侧:输出嵌入矩阵整体丢弃,换随机初始化 512 维字节 head,行为靠蒸馏恢复。
- 零成本后训练迁移:Task Arithmetic 把 RL checkpoint 权重差加到全局层
  (IFEval 31.1%→67.4%)。

## 2. 参考价值(对 v2)

- **(a) "过冻结骨干层再对齐"损失**:原则是"表示等价性要在下游若干层非线性变换
  后判定"。对 v2 的评估:**不建议照搬进 AE 训练**(v2 无现成强骨干可锚;POC 分锅
  探针已判 AE 重建质量非瓶颈;会破坏 detach 纯净性)——但诊断方法论免费收获:
  未来诊断嵌入空间时,"过几层再比对"比嵌入空间直接 cos/L2 更有判别力。
- **(b) Appendix E rank 分析(双向弹药)**:子词嵌入本质高秩、线性上投影硬截秩。
  反方:评审会用它质疑 v2 的 32/48 维瓶颈——需几何套件+重建 EM 回应(**学习压缩
  ≠ 线性截秩**);正方:v2 的 48→2048→768 非线性适配器恰是 Appendix E 给出的
  出路。
- **(c) 残差嵌入精神**:可借为 byte n-gram hash embedding 叠加在 v2 的 192d byte
  emb 上(纯字节、稀疏激活、低成本可消融)——候选,非必需。
- **(d) 压缩率经验**:Bolmo ~4.4 B/patch、~6.6 才推理反超;per-example BPE merge
  监督优于熵/CE merge——间接支持 v2 的增强 BPE 选择;因果边界崩盘案例强化
  "学习边界难、离线边界稳",支持 v2 预烘焙决策。

## 3. 撞车部分(逐维)

切分(学习非因果边界 vs 离线增强 BPE)、池化(选择式零参数 vs PMA)、patch 表示
(4096 同维无损 vs 32/48 有损瓶颈)、编码器目标(蒸馏对齐 vs 重建)、冻结对象
(冻骨干 vs 冻 AE)、输出头(512-way 字节 softmax vs GMM 整串)——**全部不撞**。
v2 独占:词级温度(Bolmo Bit 5 明列未研究)、多字节联合预测(Bit 3 未做)。

## 4. 撞车程度:**低(技术)/ 中(叙事)**

技术内容几乎不撞;交锋在叙事层——Bolmo 占据"字节级首次打平 SOTA subword"高地,
评审必问"为什么不直接 byteify"。弹药(全部 Bolmo 自供):① 转换有损,附录 A 的
OLMo 3 CT 对照显示相当差距来自 continued training 本身的破坏性;② 训练期并不
vocab-free(子词边界监督+保留子词嵌入表);③ Bit 0 承认未评估非因果边界在从头
训练是否有效——byteify 与从头 vocab-free 是两个问题(作者自己定位
"complementary");④ v2 独占能力恰是其空白(词级温度、patch 级联合分布);
⑤ 推理仍逐字节(~125 vs ~150 bytes/s),batch 适配未解。

## 5. 附:严谨度与开源

附录 A 的 CT 对照(解耦 byteify 损耗与续训损耗)是反事实设计的标杆;CUTE 78.6 的
成色需打折(训练混入 0.04% 合成字符数据,但 CT 对照吃同样数据仍输,结论站得住)。
开源完整度高(脚本/权重/数据/uv.lock 全)。

---

# 三、TFree-HAT(arXiv:2603.15953,HAT 系 70B 扩展)

2026-03,Aleph Alpha;HAT 原文(2501.10322)**已被 ICLR 2025 接收**。
[TFree-HAT HTML](https://arxiv.org/html/2603.15953v1 "citation"), [HAT HTML](https://arxiv.org/html/2501.10322 "citation"), [HF 模型卡](https://huggingface.co/Aleph-Alpha/tfree-hat-pretrained-7b-base "citation")

## 0. 重要更正(对 v1 时代认知)

HAT 原文的词编码器**不是**学习 query cross-attention——是 `[W]` 前缀符 + 词内双向
transformer + `[W]` 位读出(BERT 式)。**学习 latent query 的 cross-attn 池化是
TFree-HAT 才引入的**(编码器改为全字节流因果+滑窗 768 transformer,连接层每词一个
可学习 latent 向量 cross-attend 词内字节块)——这才是与 v2 PMA 同型的机制。
v1 时代文档(docs/14/15)若按旧理解引用需修正。

## 1. 创新点(相对 HAT 的改动 + 声称贡献核实)

- 输入侧换为因果滑窗字节 transformer + **学习 query cross-attn 池化**(PMA 同型);
- 解码器条件化加深:每个 block 都 cross-attn 到骨干的 next-word 预测(而非只在词首
  拼接一次);
- 切分升级 UAX#29 + camelCase/数学符号规则,Rust 实现开源(`hat-splitter`);
- 损失不变(逐字节 CE);
- 声称逐条核实:① 从零 7B 用 Llama-3.1 约 1/3 数据预算打平/互有胜负(MMLU 0.665
  vs 0.670,GSM8K 0.612 vs 0.566;TriviaQA 落后)——**有支撑**;② HATification
  替换 Llama tokenizer,8B 总参数 -10.4%,非骨干参数 13%→<3%——**有支撑**;
  ③ 全 pipeline+权重公开(200 个中间 ckpt,超 Pythia 密度)——属实,但许可证
  仅非商业研究;④ vLLM PR 未合并;70B SFT 弱于 Llama-3.3-70B-Instruct,作者自称
  "experimental release"。
- 70B 工程:FlashAttention 反向对"8 万条长度 1 序列"非法访问(改 kernel);放弃
  context parallelism;从零 7B 加 QK-norm+softcapping@100;HATification 骨干冻前
  2000 步再 1/10 LR 解冻(曲线在 ~2000 步有"bump"标志过渡到层级行为)。

## 2. 参考价值(对 v2)

- **(a) PMA 同型池化的 70B 可行性背书(最有力)**:学习 query 池化在 68.45B
  骨干+477M 编码器配置下稳定训练;从零 7B 跑 1M 步/4T 词;**训练稳定性归因于
  QK-norm+softcapping,池化本身"无新闻"**——无需特殊处理即是最强背书。注意差异:
  他们证明的是"池化机制可扩展",不是"32 维瓶颈可扩展"(其词嵌入 4096 宽)。
- **(b) 反面对照:v1 分锅结论的外部印证**:HAT 原文架构 sweep 发现字节准确率
  偏好大编解码器、词准确率偏好大骨干,两者讲述相反故事;字节准确率可被"补全词"
  刷高而不提升词级质量——与 v2 POC 的"解码器无辜、误差在骨干"互为印证,支持 v2
  用 EM(词级)而非字节级指标验收 AE。 [HAT §4.1](https://arxiv.org/html/2501.10322 "citation")
- **(c) 可直接沿用的评测口径**:词级准确率的公式化定义(HAT App B.2,与 v2 的
  EM 等价,可作口径出处);压缩率 bytes per backbone position(TFree-HAT 英文
  4.7-5.5,v2 均长 3.47 偏短一档,可对标);弃用 bpb 跨架构对比的论证可引为 v2
  评测辩护;词内 10% 字符扰动鲁棒协议与 v2 typo 增强同族。
- **(d) 停止符体系对比**:HAT 的 `[W]`/`[S]` 在字节解码器词表内(256 中的未用
  值),词界检测与字节生成耦合在同一 softmax,字节级温度会同时扰动"是否停";v2
  的 EOS 在 AE 解码侧第 257 行,与骨干/MDN 侧的词级温度**解耦**——设计差异点。

## 3. 撞车部分(逐维)

| 维度 | TFree-HAT | bltz v2 | 撞车度 |
|---|---|---|---|
| 输入池化 | 学习 latent query cross-attn(PMA 同型) | ISAB×2+PMA | **高**(核心机制同型;内部结构与维度不同) |
| 编码器目标 | 端到端 CE,无独立目标 | 重建目标独立训练 | 不撞(本质分叉) |
| 冻结 | 全程联合(HATification 仅冻 2k 步) | AE 冻结+λ detach 硬切断 | 不撞,方向相反 |
| 输出解码 | 词内逐字节 AR+cross-attn+`[W]` | GMM 一次性分量+AE 解码 | 低-中(一次性 vs 自回归是原理分野) |
| 词级温度 | **无**(全文 temperature/sampling 0 命中;温度只能作用在字节 logits) | π 温度=严格词级温度 | **不撞,v2 独家** |
| 词级联合分布 | **有**——词内字节 AR 链式法则 ⇒ P(word)=∏P(byte\|prefix) 良定义 | MDN 分量即词,天然词级 | **部分撞**(见下) |
| 切分 | UAX#29 确定性 | 增强 BPE×规则,随机化 | 低-中 |

**"部分撞"的含义(对 v2 叙事的重要修正)**:HAT 系通过词内字节 AR 链式法则已经
获得良定义的词级联合分布——**v1 幽灵词病理(cat/dog/dot)对 HAT 系的打击面要
收窄**:HAT 没有幽灵词问题,它缺的是**词级采样控制**(温度/top-p 只能作用在字节
logits,v1 病理的"温度语义断裂"这一支在 HAT 依然存在)与**一次性出整词**的效率。
v2 的病理叙事应精确表述为:并行/独立解码系(v1、DMBP、BLT-D 草案)有幽灵词;
词内 AR 系(HAT、BLT、Bolmo)有联合分布但无词级采样控制且有词内串行代价;
**v2 是唯一同时拿到"词级联合 + 词级温度 + 一次性生成"的**。

## 4. 撞车程度:**中**

概念纲领(变长字节块→学习池化→单元级骨干→字节侧解码)高度重合,且对方已以生产
质量做到 70B/4T 词——**v2 卖点若停留在"字节聚成词单元的层级 LM"会被直接覆盖**。
三条命脉不撞:AE 重建+冻结+硬切断 vs 端到端;MDN 一次性分量 vs 词内字节 AR;
词级温度独家。
**两条路的根本优劣**:HAT 路=词嵌入无瓶颈(4096 宽)、联合优化无信息瓶颈、词级
联合经链式免费获得,代价是词内串行(官方承认 batch 吞吐低于 FLOP 对齐基线)、
采样控制只能字节级、单元表示无显式几何;v2 路=一次性出整词、严格词级温度、
λ 空间显式几何(min-σ 重排等免费推理增益)、I/O 与骨干解耦(AE 可独立升级),
代价是 32/48 维硬瓶颈、两阶段 detach 切断协同(POC"误差在骨干")、规模化零证据
而对手已有 70B。

## 5. 附:团队动向

HAT 未被弃反转正(ICLR 2025 接收;2025-01 HATified Llama→2025-08 从零 7B→
2026-03 70B 技术报告,持续加码);社区采用度低(旗舰模型月下载 371 次;许可证
仅非商业研究是硬门槛);权重+200 中间 ckpt+hat-splitter+eval framework 可用——
**200 个高密度中间 ckpt 是研究学习动态的独有资产**。 [Aleph Alpha 博客](https://aleph-alpha.com/en/blog/introducing-tfree-hat-7b-tokenizer-free-models-achieving-top-tier-multilingual-performance/)

---

# 四、三篇深挖对 v2 定位的综合修正(相对主报告的增量)

1. **撞车矩阵更新**:Kronecker Embeddings 从初判"高"降为**低-中**(主体全正交,
   §8.5 仅假设);Bolmo 低(技术)/中(叙事);TFree-HAT **中**(纲领重合+70B
   实证)。三篇合计不改变主报告§0 的总裁决(无致命撞车、CALM 仍是最近邻)。
2. **v1 幽灵词叙事的精确化**:经 TFree-HAT 深挖确认——词内 AR 系(HAT/BLT/Bolmo)
   经链式法则已有词级联合分布,幽灵词叙事只打击并行/独立解码系(v1、DMBP 草案头、
   Fast BLT-D);对词内 AR 系的攻击点应改为"词级采样控制不可能 + 词内串行代价"。
   写作时必须做这层区分,否则评审(读过 HAT 的)会立刻反驳。
3. **v2 的独占三元组**(三篇深挖后确认无人同时具备):**词级联合分布 + 词级温度
   + 一次性整词生成**。CALM 有前两个的雏形但 likelihood-free;HAT 有第一个;
   Kronecker §8.5 只是假设。
4. **可白拿的方法论资产**:Kronecker 的 typo 探针协议与印刷变体探针(移植到 λ
   空间);Bolmo Appendix E 的 rank 论证(攻防两用)与 CT 对照设计(反事实标杆);
   HAT 的词级准确率公式(=v2 EM 的口径出处)、压缩率口径、字符扰动协议、
   "池化机制 70B 无新闻"背书、200 个中间 ckpt。
5. **必引清单再增**:2605.29459(Kronecker)、2512.15586(Bolmo)、2603.15953
   (TFree-HAT)、2501.10322(HAT,ICLR 2025 口径更新)、1911.04975(word2ket,
   防 Kronecker 混淆)、Mimick(1707.06961)、Char2Subword、MorphTE(2210.15379)。
