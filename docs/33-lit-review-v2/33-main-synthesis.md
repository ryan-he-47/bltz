# 33 — v2 架构文献调研总报告:撞车评估与生态位判定(2026-09-21)

> **性质**:v2 转向(docs/31/32)后的第一轮外部调研。触发:用户要求重新评估
> v2 架构(AE+MDN+增强 BPE+EOS)的学术生态位。
> **方法**:四路并行外部检索(连续潜空间文本 LM / MDN 谱系 / 随机分词与字符串
> 嵌入 / 字节级赛道 2026 盘点)+ 本地 CoSE 全文精读 + v2 代码核对
> (`bltz/models/model_v2.py`、`mdn.py`、`autoencoder.py`)。全部事实来自本轮实际
> 抓取的页面,附 URL;抓不到的标 UNVERIFIED。不修改任何既有文档。
> **分报告**:`33-lit-review-v2-mdn-head.md`(输出头谱系+CoSE 精读)、
> `33-lit-review-v2-components-20260921.md`(分词与字符串嵌入两组件)。

## 0. 结论先行

1. **无致命撞车,但骨架不新。** "AE 嵌入变长单元 + 潜空间自回归 + GMM 预测头 +
   梯度解耦"四件套,CoSE(NeurIPS 2020,sketch 域)全有;"GMM 头替代 softmax
   预测连续单元"是 GIVT(ECCV 2024,图像域)的原话机制。v2 是把这两套机制首次
   组合到**文本/字节域**。
2. **范式旗已被 CALM 先立**(arXiv:2510.27688,2025-10-31,腾讯微信 AI+清华):
   "AE 压缩文本块到连续向量 + 冻结 AE + 潜空间自回归"的完整骨架已发表。v2 与它
   五个设计点全部不同(§2),其中**显式似然 vs likelihood-free** 是最锋利的区隔。
3. **三个真新贡献尖点**(无人占位):① 词级 π-温度的精确免费实现(CALM 的
   future work 第一条就是要轻量温度方案,v2 是直接回答);② 分量=确定字节串的
   可解释 GMM(μ_k 经冻结解码器映射为完整词);③ min-σ/解码器置信推理期免费
   重排(无直接先例)。
4. **赛道活跃且升温,窗口期约 6-12 个月**:潜空间子方向 2025-10 以来平均每月
   有新预印本(CALM→Projected AR→STAR-LDM→TextLDM→NCP-ArchPreview 8.9B)。
   建议尽快成稿,并把 CALM 作为主对照。
5. **项目有研究价值,但卖点必须重排**:不能卖骨架(已被 CoSE+GIVT+CALM 三面
   夹击),要卖"显式密度的回归 + vocab-free + 变长 + 词级温度 + 幽灵词病理的
   原理性根治"。

## 1. v2 架构指纹(代码核对版)

```
字节串 ──增强BPE(双分词器投票,p=0.5,预烘焙)──> patch 序列(均长 3.47B)
patch ──ByteStringEncoder(ISAB×2+inducing8+PMA(1),byte192+pos64 拼接)──> λ∈R^32
       ──ByteStringDecoder(FiLM 条件 (λ,Δ)→257 类 softmax,256 字节+EOS)──
       AE 只受重建梯度,无 KL;训练后冻结
λ(detached)──LN→Linear(32→2048)→GELU→Linear(2048→768)──> 骨干 768×12 RoPE
骨干 ──MDNHead(768→1024→SiLU→1024→SiLU→K(1+2d),K=64 对角 GMM,σ=ELU+1+0.05)
采样:π^(1/τ) 采分量 → 取 μ_k → 解码至 EOS →(可选)回投影
```

已验证的实证锚:AE 零次桶 EM 93.6%(真字符串级泛化);POC 60k NLL -32.8、
零泛化间隙;argmax EM 32.7%、分量覆盖 60.9%;min-σ 重排 +3pp EM;幽灵词病理
在原理层根治(docs/32 §S3)。

## 2. 最近邻逐篇:CALM 与其军团

### 2.1 CALM(撞车等级:高,主对照)

arXiv:2510.27688(2025-10-31;投 ICLR 2026,**收录结果 UNVERIFIED**——OpenReview
标 under review,Semantic Scholar 2026-09 仍只标 arXiv)。AE(VAE,β=0.001+KL
clip+双 dropout)把**固定 K=4 个 token** 压成 128 维向量(重建 >99.9%),骨干做
next-vector 预测,生成头为 Energy Transformer(energy score,隐式,likelihood-free,
~10% 参数),评估用自造 BrierLM,温度采样用拒绝采样+Bernoulli factory。371M 打平
281M 离散基线,训练 FLOPs -44%/推理 -34%;**K=1 反而不如离散基线**——连续预测在
粒度过细时变难,正面佐证 v2 的词/子词粒度。 [arXiv](https://arxiv.org/abs/2510.27688 "citation"), [HTML 全文](https://arxiv.org/html/2510.27688v1 "citation"), [代码](https://github.com/shaochenze/calm "citation")

与 v2 的五点差异(按防御力排序):

| # | 差异点 | CALM | v2 | 备注 |
|---|---|---|---|---|
| 1 | **生成头** | Energy Transformer,隐式,无似然 | MDN/GMM,显式 NLL 可训可评 | CALM 整套 BrierLM/拒绝采样工具链都是为绕开"没有显式似然" |
| 2 | **温度** | chunk 级温度需拒绝采样+Bernoulli factory,两端代价爆炸(原文自认) | π^(1/τ) 一次 softmax,精确免费 | CALM §8 future work 第一条=找轻量温度方案;**v2 措辞应为"廉价精确的显式词/块级温度",不能说"首次可行"** |
| 3 | **单元** | 定长 K-token,依赖 Llama tokenizer 与词表 | 变长字节 patch,词界对齐,vocab-free | v2 是 BLT 谱系正统延续;CALM 不在该谱系 |
| 4 | **骨干输入** | 实测直接喂 latent 退化,被迫落回离散 token 重嵌入 | 直接消费 detach 的 λ,POC 已证可训 | **对 CALM 经验结论的反例,独立论点** |
| 5 | **AE 正则** | 报告称纯重建 AE 的 latent "practically impossible",故上 VAE+KL | 纯重建无 KL + 梯度硬切断,32 维走通 | 第二条反例;CoSE D 维消融支持低维 |

### 2.2 潜空间范式军团(2024-12 → 2026-09,全部本轮实抓)

| 论文 | 时间 | 一句话 | 撞车 |
|---|---|---|---|
| LCM(Meta) [arXiv](https://arxiv.org/abs/2412.08821 "citation") | 2024-12 | SONAR 句嵌入 AR;MSE 差、扩散最好、仍输同级 Llama | 中低(句级语义嵌入 vs 词级表面形) |
| CoCoMix [arXiv](https://arxiv.org/abs/2502.08524 "citation") | 2025-02 | 连续概念混入隐状态,输出仍 token softmax | 低 |
| SONAR-LLM [arXiv](https://arxiv.org/html/2508.05305v1 "citation") | 2025-08 | SONAR 空间思考,去掉 LCM 扩散采样 | 低 |
| GQ-VAE [arXiv](https://arxiv.org/abs/2512.21913 "citation") | 2025-12 | 变长**离散**量化 token 化器,drop-in 替 BPE | 低(离散码本,互为对照) |
| Projected Autoregression [arXiv](https://arxiv.org/abs/2601.04854 "citation") | 2026-01 | token 级连续预测(NCE+回归),最近邻投影提交 | 中(同家族,组件无重叠) |
| STAR-LDM [arXiv](https://arxiv.org/abs/2602.20528 "citation") | 2026-02 | 潜扩散规划+暂停嵌入 AR | 低 |
| Latent Lookahead [arXiv](https://arxiv.org/abs/2603.20219 "citation") | 2026-03 | 潜空间前瞻仅作训练策略 | 低 |
| TextLDM [arXiv](https://arxiv.org/abs/2605.07748 "citation") | 2026-05 | Text VAE + 非自回归潜扩散,追平 GPT-2 同级 | 低 |
| ReconSpan [arXiv](https://arxiv.org/abs/2608.12756 "citation") | 2026-08 | **重建准则定切分边界**,连续潜 token;无 LM 无 MDN | 中(思路同源,可引为佐证) |
| NCP-ArchPreview [arXiv](https://arxiv.org/abs/2609.10715 "citation") | 2026-09 | PQ 量化 concept 词表,NCP+NTP 联合,8.9B/5.73T,51.3% 数据追平 OLMo-3-7B | 中低(离散 VQ+保留 NTP;评审会问区隔) |
| HiLP [arXiv](https://arxiv.org/pdf/2608.05806 "citation") | 2026 | 潜预测仅辅助训练;§5.2 已把 LCM/CALM/Coconut 并列为谱系 | 低(佐证谱系已成名) |
| CLaM-TTS [arXiv](https://arxiv.org/abs/2404.02781 "citation") | 2024-04 | **语音域:GMM 头预测连续 latent + EOS 终止符** | 中(**组件级先例**:削弱"MDN 头+EOS"本身的新颖性主张,必引) |

音频域另有 CLEAR(2508.19098)与一篇 K=1024 GMM 头音频 LM(OpenReview)——
"GMM 头+连续 latent"在音频/语音已是小生态,文本域仍是空白。 [CLEAR](https://arxiv.org/html/2508.19098v1 "citation")

### 2.3 骨架祖先:CoSE 与 GIVT

- **CoSE**(NeurIPS 2020,arXiv:2006.09930):四件套全有(sketch 域);梯度解耦
  "略好"是 v2 硬切断的出处;D=8 预测最好/D=32 重建最好(**重建好≠预测好**,v2
  压 32 维的直接先例);VAE/KL 全面有害;无 EOS(t 进度参数化,v2 的 EOS 是真实
  差异点)。被引 ~31-54 次,**全部在 sketch/HCI 域,无语言建模继承者**。
  [arXiv](https://arxiv.org/abs/2006.09930 "citation"), 精读全文见分报告
  `33-lit-review-v2-mdn-head.md` §0
- **GIVT**(ECCV 2024,arXiv:2312.02116):softmax→GMM 替换的原话机制(图像域);
  k=1→16 平台化、对角≈全协方差(支持 v2 对角选择);其连续温度=σ 缩放,**不调
  π**——与 v2 机制正交。 [arXiv HTML](https://arxiv.org/html/2312.02116v4 "citation")
- **JetFormer**(arXiv:2411.19722):即便图文统一模型,**文本仍保留离散 softmax**,
  GMM(1024 分量)只给图像——文本域无 GMM 头先例的最强证据;1024→1 分量消融
  显示分量数与 recall 挂钩(支持 K=64 的覆盖度动机)。 [arXiv HTML](https://arxiv.org/html/2411.19722v1 "citation")
- **KerBS**(NeurIPS 2019):文本输出层尝试 Gaussian 型建模,明言"高维高斯数值
  不稳定"退回核函数——佐证 GMM 头应作用在低维 AE 空间(v2 的 32-48 维)而非全
  词表嵌入空间。 [NeurIPS PDF](http://papers.neurips.cc/paper/9415-kernelized-bayesian-softmax-for-text-generation.pdf "citation")

## 3. 撞车矩阵(各主张 × 最近先例)

| v2 主张 | 最近先例 | 等级 | 判定 |
|---|---|---|---|
| AE 嵌入+潜空间 AR+GMM 头+梯度解耦(骨架) | CoSE(sketch)/ CALM(文本) | **高** | 组合不新;迁到文本字节域是首例 |
| GMM 头替代 softmax 预测连续单元 | GIVT(图像)/ CLaM-TTS(语音) | **高(组件)** | 文本域首例,但必须引 GIVT/CLaM-TTS |
| 字节串→低维连续 λ 的文本 LM | CALM(K-token)/ LCM(句) | **高(范式)** | 范式旗已被 CALM 立走;粒度+可逆性+EOS 是差异 |
| 词级温度(只调 π,取 μ_k) | CALM=昂贵拒绝采样;GIVT=σ 缩放 | **低** | **新且免费**,正面回答 CALM 的 open problem;"word-level temperature"无命名先例(UNVERIFIED-as-absence) |
| min-σ / 解码器置信重排 | σ=aleatoric 公共知识;无 rerank 先例 | **低** | 基本为新(+3pp EM 已实测) |
| 分量=确定字节串的可解释 GMM | GIVT/JetFormer 分量无语义对齐 | **低** | 新;解码器把 μ_k 映射为离散语义实体 |
| 幽灵词病理的原理性根治 | 无直接对应;DA-DLM/DiLaDiff 在扩散域表述 multimodality | **低** | 论证需自建;"phantom word"术语在字节级文献不存在,有原创命名空间 |
| 增强 BPE 双分词器投票分割 | **StochasTok(2506.01687,离线随机重分割,高)**;MVR 2021(双视图一致性,中偏高) | **高(组件)** | 双 tokenizer 投票+交集锚定无先例(UNVERIFIED-as-absence);离线预烘焙有 StochasTok 先例 |
| 任意字符串零次重建指标 | vec2text(嵌入可反演);ZeTT(零次迁移不重建) | **低** | 以"未见串 EM 重建率"为 headline 指标无 NLP 先例(UNVERIFIED-as-absence) |

## 4. 赛道盘点:字节级/token-free 2026

**活跃且有空位,不是红海。** 标志性成果已出现:BLT 8B FLOP 对齐追平 Llama-3 8B;
Bolmo-7B 接近 OLMo-3-7B 且字符任务 +20pp;H-Net 算力对齐超 BPE Transformer
(ICLR 2026);NCP 8.9B 用 51% 数据追平 OLMo-3-7B。 [BLT](https://arxiv.org/abs/2412.09871 "citation"), [Bolmo](https://arxiv.org/abs/2512.15586 "citation"), [H-Net ICLR 2026](https://iclr.cc/virtual/2026/poster/10008794 "citation"), [NCP](https://arxiv.org/abs/2609.10715 "citation")

社区 2026 下半年四个流向:① 并行/快速解码(Fast BLT→ICML 2026、DMBP、字节
扩散);② 端到端切分的分析(H-Net、SSLM 对比 2608.17325);③ **潜空间化(最新
最拥挤:CALM、NCP、ReconSpan、GQ-VAE)**;④ token→byte 蒸馏转换(Bolmo、
Byte-Prefix Marginalization 2607.22334)。

Venue 证据:token-free/字节级论文 2025-2026 持续被全部顶会主会接收(ICLR 2026:
H-Net;ICML 2026:Fast BLT;NeurIPS 2025:AU-Net;ACL 2025:BLT)。 [Fast BLT ICML 公告](https://x.com/JulieKallini/status/2053853543552217478 "citation")

**两条可为我所用的外部论据**:
- The Efficiency Gap in Byte Modeling(Google/Rush 组,2605.12928):字节建模代价对
  扩散比对 AR 更大("context fragility")——支持 v2 选 AR+MDN 而非扩散头。
  [arXiv](https://arxiv.org/abs/2605.12928 "citation")
- Beyond Perplexity: UTF-8 Validity(2606.14122):字节级模型会产非法 UTF-8——v2
  的"分量=确定字节串"天然规避,可引作动机。 [arXiv](https://arxiv.org/abs/2606.14122 "citation")

## 5. 问题陈述的文献接引(写作弹药)

- **幽灵词病理**:字节级文献无 "phantom word" 术语;现行表述在扩散/NAT 文献——
  DA-DLM(2609.15070)明写"条件独立丢弃 token 间依赖、损害连贯性,与 NAT 的
  multimodality 问题同源";DiLaDiff(2605.23605)给 factorized posterior ≠ true
  joint 的形式化。**v1 的 "cat/dog/dot" 论证可对接这套话语,且有原创命名空间。**
  [DA-DLM](https://arxiv.org/abs/2609.15070 "citation"), [DiLaDiff](https://bytez.com/docs/arxiv/2605.23605/paper "citation")
- **温度粒度**:ByteSampler(2506.14123)附录 F 明确论述"温度/top-p 作用在字节
  粒度 ≠ token 粒度",并给精确(token 级先变换再聚合)程序——v2 反用:粒度是模型
  原生属性,v2 把生成粒度抬回词/片段级。 [ByteSampler](https://arxiv.org/html/2506.14123v3 "citation")
- **DMBP 的正面对照义务**:DMBP(2608.15454)声称其并行多字节"minimal performance
  impact"——但他们的评测是下游任务,**未测联合分布性质**;v2 的反驳点不是字面值
  损失而是联合分布/词级温度的原理性缺失。 [arXiv](https://arxiv.org/abs/2608.15454 "citation")

## 6. 研究价值判定与叙事建议

**判定:有研究价值,窗口期收窄(约 6-12 个月)。** 潜空间子方向正在快速拥挤
(NCP 9 月已出 8.9B),但"字节级+变长分割+冻结 AE 连续低维潜空间+显式密度头+
学习终止"这一具体组合目前无人占据。

**叙事主轴(推荐)**:**"Likelihood is back:连续潜空间 LM 不必 likelihood-free。"**
CALM 为丢掉的显式概率付出了一整个工具链(能量头、BrierLM、拒绝采样温度);v2 的
MDN 头让 NLL 评估、词级温度、min-σ 置信重排全部免费且精确。配两条反理论据
(CALM 说直接喂 latent 退化、纯重建 AE 不可用——v2 POC 两条都走了通)。

**投稿判断**:127M POC + FineWeb-Edu 10BT 的量级,适合"分析密度高的小模型论文"
路线(参考 ReconSpan/ByteSampler 的接收模式),COLM/EMNLP/ACL 主会现实;冲
ICLR/ICML 需要算力对齐 scaling 对照(H-Net/Fast BLT 的评审口味)。

**三大风险**:
1. CALM 若被 ICLR 2026 接收并铺规模,占位压力骤增(收录状态 UNVERIFIED,投稿前
   必须再查 OpenReview);
2. Bolmo 的 byteify 路线是"实用派"强敌——评审会问"为什么不直接转换现成 LLM",
   需准备"从头 vocab-free 的表示学习价值"答辩;
3. MDN 训练病理有系统文献(Makansi et al., CVPR 2019:模式坍缩、σ 退化、高维数值
   不稳)——与 v2 的 σ 坍缩死锁(sigma_floor=0.05)直接呼应,**必引**;否则评审会
   认为你对 MDN 陷阱无知。 [arXiv](https://arxiv.org/html/1906.03631v2 "citation")

## 7. 必引清单汇总(缺一即危险)

**骨架/范式**:CoSE(2006.09930)、**CALM(2510.27688)**、LCM(2412.08821)、
GIVT(2312.02116)、JetFormer(2411.19722)、MAR(2406.11838)、CLaM-TTS
(2404.02781)、ReconSpan(2608.12756)、NCP(2609.10715)。
**MDN 谱系**:Bishop 1994、Graves 2013(1308.0850)、Makansi CVPR 2019
(1906.03631)、MoS(Yang 2018)、KerBS(NeurIPS 2019)。
**分词/字符串嵌入**:StochasTok(2506.01687)、Kudo 2018、BPE-dropout(Provilkov
2020)、MVR(Wang/Ruder/Neubig 2021)、Saleva & Lignos 2023、SuperBPE
(2503.13423)、vec2text、ZeTT(2405.07883)、T-FREE(2406.19223)、Mixture of
Tokenizers(博客,注明非正式)。
**问题陈述**:DA-DLM(2609.15070)、DiLaDiff(2605.23605)、ByteSampler
(2506.14123)、UTF-8 Validity(2606.14122)、DMBP(2608.15454)、Efficiency Gap
(2605.12928)。
**字节级谱系**(承接 v1 调研):BLT、HAT、H-Net、Fast BLT、Bolmo(2512.15586)。

## 8. UNVERIFIED 待办(写作前必须补)

1. **CALM 的 ICLR 2026 最终收录结果**(OpenReview decision 页有反爬,本轮未读到);
2. CoSE 的 Semantic Scholar citation 逐条清单(API 429 限流);
3. GIVT 参考文献 34/35/64/65/38 对应的"连续词嵌入预测"支系原文;
4. "word-level temperature" 是否被更早工作命名(本轮无命中,标 UNVERIFIED-as-absence);
5. CALM 骨干参数档位与下游 benchmark 具体数字(抓取页面截断);
6. multi-token prediction(Gloeckle et al. 2024)系文献本轮未抓,幽灵词论证写作前需补;
7. 两处否定性结论(无双 tokenizer 投票分割先例、无零次串重建指标先例)均为
   UNVERIFIED-as-absence——否定性结论永远无法由检索彻底证明,写作时用 "to our
   knowledge" 措辞。

## 9. 对项目决策的回灌(非架构建议,仅供用户参考)

1. **1B 长训(575524)的叙事价值上升**:CALM 对照实验的最佳筹码是"同数据同算力
   下的 NLL/EM 对比 + 词级温度演示",1B 跑出健康曲线后这套实验即可做。
2. **min-σ 重排值得单独写一节**:无先例 + 已实测 +3pp EM,是低成本高辨识度的
   贡献点。
3. **AE 三件套(零次桶/distinct 曲线/几何)恰好对齐文献空白**:零次串重建无人做
   过 headline 指标,v2 的 93.6% 可以直接立为新基准口径。
4. **警惕 DMBP 类"并行无损"口径**:v2 需要一组联合分布性质的实证(如分支点承诺
   度、词级温度下的 distinct-n 扫描)作为对 DMBP 的正面回应,这正是 v1 时代已规划
   的诊断套件(v2 版)的用武之地。
