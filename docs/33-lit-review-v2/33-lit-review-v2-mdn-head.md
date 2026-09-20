# 33 — v2 输出头设计的文献定位调研(MDN/GMM 头)

日期:2026-09-21。调研方式:本地 CoSE 全文(主文+附录)+ WebSearch/FetchURL 实抓页面。
所有事实均来自本轮实际抓取的页面并附 URL;抓不到的标 UNVERIFIED。

问题:v2 架构(字节串 AE → 冻结 32/48 维 λ → 因果 transformer 在 λ 上自回归 →
K=64 对角协方差 GMM 头,采样只对 π 加温度、取 μ_k 整串解码;σ 作置信度做 min-σ
重排)在文献中处于什么位置?

---

## 0. CoSE 精读(直接灵感来源,本地全文)

来源:`reference_projects/cose_paper_full_text.txt`、`cose_supplementary_full_text.txt`;
arXiv:2006.09930,NeurIPS 2020。https://arxiv.org/abs/2006.09930

**方法要点**
- 自编码器:变长 stroke(2D 点序列)→ 定长 λ∈R^D。encoder = 6 层 transformer
  (d_model=64,带时序位置编码+causal mask;实验发现序列式优于集合式)。decoder =
  4×512 MLP,以曲线参数 t∈[0,1] 与 λ 联合为条件,对每个 t 输出 **20 分量 GMM**
  (2D 坐标,对角协方差),NLL 训练(不用 Chamfer;附录 §10 说明 GMM-NLL 优于 MSE:
  重建噪声显著更低)。
- 关系模型 R_θ:置换不变 transformer(故意不加位置编码),两支独立 transformer 分别
  预测下一 stroke 的起点(2D GMM)与嵌入 λ(D 维 GMM),**各 10 个分量**。
- **梯度解耦**:三个子模块并行训练,但"关系模型的梯度不回传到 stroke embedding
  模型"性能略好(主文 §3.3 末)——这就是 v2 的"梯度与骨干硬切断"的直接出处。
- 无 EOS:曲线参数化 t∈[0,1],t=1 即笔画终点,靠采样 t 的密度控制长度(supp §5)。
  **v2 在此点与 CoSE 不同:文本需要显式终止,v2 用 EOS 行。**
- 维度消融(主文 Tab.2):D=8 预测最好(SC 0.361),D=32 重建最好但预测与 SC 变差
  ——**CoSE 明确发现"重建好 ≠ 预测好",低维更利于预测**。v2 的 32 维选择已在 CoSE
  的退化区间边缘,这是需要正视的已知张力(CoSE 的 D=32 仍在跑且 SC 0.314,并非失败,
  且文本词汇量远大于 sketch 形状类别数,不能直接类比)。
- VAE/KL 正则化伤害重建、预测与 SC(CoSE-VAE 行)——支持 v2 用确定性 AE。
- 局限(supp §11):过长复杂笔画 AE 抓不住;嵌入空间未显式正则化 → 关系模型会预测到
  低密度区(对应 v2 的"分量落在无数据区/幽灵"风险,CoSE 没有解法);预测步数多了
  会重复画箭头(模式坍缩的一种表现)。

**被引情况**:Semantic Scholar 页显示 31-35 次引用,Google Scholar/arXiv 页显示
49-54 次;后续工作集中在 sketch/HCI 领域(SketchINR、Draw This First、continuous-time
neural sketch 等),**未发现任何语言建模方向的继承者**。
https://www.semanticscholar.org/paper/CoSE%3A-Compositional-Stroke-Embeddings-Aksan-Deselaers/5645ad7da64b0a072d16b8027de3089974b7ad66 ,
https://openreview.net/forum?id=JYszswBj7BM ,
https://arxiv.org/abs/2006.09930
(S2 API 本轮 429 限流,citation 数以搜索快照为准,标 UNVERIFIED 精确值。)

**与 v2 的关系**:结构性祖先——"AE 嵌入变长单元 + latent 上关系模型 + GMM 预测 +
梯度解耦"四件套 CoSE 全有。撞车等级:**高**(必须正面引用并区隔;区隔点见 §6)。

---

## 1. MDN 谱系与"GMM 头替代 softmax 做文本预测"

### Bishop 1994(必引,零撞车)
Mixture Density Networks 原始技术报告:网络输出 GMM 的 π(softmax 归一)、μ(线性)、
σ(exp 激活保证正),NLL 端到端训练。
https://www.emergentmind.com/topics/mixture-density-network ,
https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/

### Graves 2013(必引,低撞车)
arXiv:1308.0850。LSTM + MDN:手写分支输出混合二维高斯(笔位移)+ Bernoulli(抬笔);
文本分支仍是字符级 softmax。**MDN 用于连续坐标,从未用于文本 token 预测。**
https://arxiv.org/abs/1308.0850

### MoS / KerBS:softmax 表达力线(中低撞车,值得引一段)
- MoS(Yang et al. 2018):mixture of softmaxes 增强离散 softmax 表达力,仍是离散词表。
- KerBS(Zhao et al., NeurIPS 2019):贝叶斯组合多 sense 嵌入 + 可学核函数替代
  内积;**论文明说"高维空间高斯建模数值不稳定"因此改用核函数**——这是"有人在文本
  生成输出层尝试过 Gaussian 型方差建模并退回"的直接证据,可用于论证为什么 GMM 头
  不该作用在全词表嵌入空间、而该作用在低维 AE 空间(v2 的 32-48 维)。
  http://papers.neurips.cc/paper/9415-kernelized-bayesian-softmax-for-text-generation.pdf

### GIVT 引用的"continuous outputs in NLP"一支(UNVERIFIED 细节)
GIVT 相关工作 §2 指出:机器翻译中有一支"用词嵌入上的连续分布预测 token、greedy
嵌入查找解码"的工作(其文献 34/35/64/65/38),但消费与预测的都是**固定有限**嵌入集合,
且不做多样性采样。本轮未逐篇展开,标 UNVERIFIED;写作时建议顺着 GIVT 参考文献补。
https://arxiv.org/html/2312.02116v4

### 结论
**未找到任何人用 GMM 头替代 softmax 做词表外文本生成。** 最接近的是 GIVT(图像域,
见 §2)与 KerBS(文本域但退回核函数)。v2 若落成,是文本域首例。

---

## 2. 连续 token 预测头方法学(2024-2026 五条线对比)

CALM 的 ICLR 投稿版(OpenReview)自己给出了一段谱系总结,与本调研结论一致:
GIVT(GMM 头)→ MAR(diffusion 头,更强表达力但迭代采样慢)→ Energy Transformer
(energy score,单步生成,CALM 采用)。
https://openreview.net/pdf?id=Ce6Mep9oie

| 路线 | 代表作 | 头 | 域 | 与 v2 关系 | 撞车 |
|---|---|---|---|---|---|
| GMM 头 | GIVT(ECCV 2024)arXiv:2312.02116 | k=16 对角 GMM,2kd+k 参数/token,β-VAE 连续 latent | 图像 | **机制最近**:softmax→GMM 的替换就是 GIVT 的原话 | **高** |
| GMM 头规模化 | JetFormer(2024)arXiv:2411.19722 | 1024 分量 GMM(图像 soft token);**文本仍用离散 softmax** | 图文统一 | 证据:即便在统一模型里,也没人把 GMM 头用于文本 | 中 |
| diffusion 头 | MAR(NeurIPS 2024)arXiv:2406.11838 | per-token diffusion loss(小 MLP 去噪器以骨干输出为条件) | 图像 | 连续 token 的主流替代路线 | 中 |
| energy score 头 | CALM(2025-10)arXiv:2510.27688 | Energy Transformer 隐式生成头,likelihood-free | **文本** | **范式最近**:K=4 token 块→128 维向量→向量级 AR | **高** |
| 量化回离散 | LCM 的 Quant-LCM(Meta,2024-12)arXiv:2412.08821 | 量化 SONAR 空间 + CE | 文本(句级) | 句级嵌入 AR;同文对比了 MSE/diffusion/quant,diffusion 最好,无 GMM | 中 |

补充细节:
- **GIVT**:k 从 1→16 显著降 FID 并平台化;全协方差只带来 ~3% 微弱提升,对角协方差
  更划算(支持 v2 对角选择);π 用 softmax、σ 用 softplus+ε 下界。
  https://arxiv.org/html/2312.02116v4
- **JetFormer**:GMM 分量数 1024,GMM 参数 269-404M;1024→1 分量消融"性能温和下降
  但 recall 显著掉"——分量数与覆盖度的关系有实证支撑。
  https://arxiv.org/html/2411.19722v1
- **CALM**:AE 把 K=4 个 token 压成 l=128 维向量(重建>99.9%,VAE 弱正则 β=0.001+
  KL clip + dropout 0.15);371M 模型打平 281M 离散基线且训练 FLOPs -44%/推理 -34%;
  K=1 时反而不如离散基线——**连续预测在粒度过细时变难**,对 v2 的"词/子词级粒度"
  是正面佐证(v2 单元均长 3.47B,介于 CALM K=1 与 K=4 之间偏词级)。
  https://arxiv.org/html/2510.27688v1 , https://github.com/shaochenze/calm
- **LCM**:SONAR 句嵌入上自回归;MSE 基线不足(语义平均问题),diffusion 最好,
  Quant-LCM 互信息更高但语义略糊。https://arxiv.org/html/2412.08821v2

---

## 3. "词/段级温度"概念

- **CALM(2025)已占据"chunk 级温度"概念**:因其隐式头无显式似然,温度采样只能走
  拒绝采样——T=1/n 时抽 n 个样本、全同才接受,分布恰为 P(x)^n,精确但昂贵;论文
  future work 第一条就是"找更轻量的多样性-保真控制"。
  https://arxiv.org/html/2510.27688v1 (§5.1, §8)
- **GIVT 的连续域温度 = variance scaling**(σ×t,t∈[0.9,1.0]),作用在高斯宽度上,
  **不是作用在分量权重 π 上**。https://arxiv.org/html/2312.02116v4 (§3.4)
- token 级动态温度(EDT、AdapT 等)是另一根轴(逐步自适应调温),与"温度粒度"无关。
  https://arxiv.org/html/2403.14541v1
- 未找到把"词级温度"作为命名概念提出的工作(UNVERIFIED:可能存在更早期小众工作)。

**结论**:v2 的"温度只加在 π 上、分量内取 μ_k"= 把整词离散选择问题重新变回一个
K 类 categorical,温度定义精确且免费——这与 GIVT(variance scaling)和 CALM(拒绝
采样)都不同,**是对 CALM 明确留下的 open problem 的直接回答**。这是 v2 最锋利的
区隔点之一,写作时应与 CALM §8 的 future work 对照引用。撞车等级:**低**(概念空间
未被占,但 CALM 的存在使"词/块级温度"不能作为独立卖点,只能作为"显式密度带来的
免费性质")。

---

## 4. "σ 作置信度/rerank"先例

- σ=aleatoric uncertainty 是 MDN 文献的标准解读;"按 likelihood 过滤预测以提取高置信
  输出"在 MDN 应用综述中明确提及。
  https://www.emergentmind.com/topics/mixture-density-network (§7)
- **Makansi et al., CVPR 2019**(arXiv:1906.03631)系统研究 MDN 训练病理:联合优化
  不稳定、高维数值退化、**模式坍缩(mode collapse)**、σ 需要上界(sigmoid 挤压)防
  退化;oracle error = 取离 GT 最近的 hypothesis。与 v2 的 σ 坍缩死锁经历(sigma_floor
  0.05)直接呼应——**这是 v2 必须引用的"MDN 已知陷阱"文献**。
  https://arxiv.org/html/1906.03631v2
- GIVT Fig.7(右):解码后期预测 σ 系统性下降、"后期预测更确定"——σ 被用作确定性
  信号可视化,但未用于 rerank。https://arxiv.org/html/2312.02116v4
- 轨迹预测领域标准做法是取 max-π 分量或 oracle 选择;**未找到"按 min-σ 对候选做
  rerank"的直接先例**(UNVERIFIED:检索范围限于本轮抓取)。

**结论**:σ-as-uncertainty 是公共知识;min-σ rerank(尤其跨到"解码器置信"解读)无
直接先例,撞车等级:**低**。写作时引用 MDN 不确定性文献 + Makansi 病理 + GIVT 的 σ
行为观察即可。

---

## 5. CoSE 的"AE+MDN+梯度解耦"组合继承者

如 §0 所述:CoSE 被引 ~31-54 次,全部在 sketch/diagram/HCI 领域(SketchINR、
Draw This First、neural sketch representation 等);**没有发现任何工作把该组合搬到文本
或通用序列建模**。Semantic Scholar citations API 本轮被 429 限流,逐条 citation 清单
标 UNVERIFIED,但搜索快照(arxiv "cited by 54"、OpenReview "cited by 49"、S2 页
31/35)一致显示规模小且领域集中。
https://www.semanticscholar.org/paper/CoSE%3A-Compositional-Stroke-Embeddings-Aksan-Deselaers/5645ad7da64b0a072d16b8027de3089974b7ad66

---

## 6. 总结:v2 的新颖性判定

### 6.1 各主张的谱系定位

| v2 主张 | 最接近先例 | 新颖性 |
|---|---|---|
| AE 嵌入 + latent 自回归 + GMM 头 + 梯度解耦 | **CoSE**(四件套全有,sketch 域) | 组合不新;**迁到文本/字节域是首例** |
| GMM 头替代 softmax 预测连续单元 | **GIVT**(图像,机制逐点相同) | 机制不新;文本域首例 |
| 字节串→低维连续 λ 的文本 LM | **CALM**(K-token 块) / LCM(句) | 范式不新(CALM 已立旗);词/子词边界对齐 + 字节级可逆 + EOS 是差异 |
| 词级温度(只调 π,取 μ_k) | GIVT=σ 缩放;CALM=昂贵拒绝采样 | **新且免费**,正面回答 CALM 的 open problem |
| min-σ rerank | MDN 不确定性文献(概念);无直接 rerank 先例 | 基本为新 |
| GMM 根治并行解码幽灵词 | 无直接对应(v1 D1 病理的针对性解法);CALM/MAR 单向量一块隐含回避了该问题 | 论证需自建,可引 CALM/MAR 作对照 |

### 6.2 撞车总评

- **架构骨架**:CoSE + GIVT 之间,无致命撞车,但两篇叠加后"骨架新颖性"所剩无几——
  v2 论文的卖点不能放在骨架上。
- **真正的新贡献点**(按锋利度排序):
  1. **词级温度的精确免费实现**(π-temperature)——对照 CALM 的拒绝采样开销与
     GIVT 的 variance scaling;
  2. **分量=确定字节串的可解释 GMM**:每个 μ_k 经冻结解码器映射为完整词,分量是
     离散语义实体而非模糊模式——GIVT/JetFormer 的分量无此语义对齐;
  3. **min-σ / 解码器置信重排**(+3pp EM,免费);
  4. 字节级模型上根治幽灵词的机制论证(需自建理论/实证,无现成引用)。
- **必须直面引用并区隔的清单**(缺一即危险):
  Bishop 1994;Graves 2013;**CoSE(Aksan et al., NeurIPS 2020)**;**GIVT(Tschannen
  et al., ECCV 2024)**;**CALM(Shao et al., 2025)**;JetFormer(2024);MAR(Li et al.,
  NeurIPS 2024);LCM(Meta, 2024);Makansi et al. CVPR 2019(MDN 病理,呼应 σ 坍缩);
  MoS(Yang 2018)/ KerBS(NeurIPS 2019)(softmax 表达力线 + "高维高斯不稳定"论据)。
- **辅助论据**(可为我所用):CoSE 的 D 维消融(重建好≠预测好,低维利于预测)支持
  v2 压到 32 维;CALM 的 K=1 退化支持 v2 选词/子词粒度;JetFormer 的 1024→1 分量
  消融支持 K=64 的覆盖度动机;GIVT 的对角≈全协方差消融支持对角选择。
- UNVERIFIED 待办:① Semantic Scholar 上 CoSE citation 逐条清单(API 429);② GIVT
  参考文献 34/35/64/65/38 对应的"连续词嵌入预测"一支原文;③ "word-level
  temperature"是否被更早工作命名;④ multi-token prediction(Gloeckle et al. 2024)系
  文献本轮未抓取,幽灵词论证写作时需补。
