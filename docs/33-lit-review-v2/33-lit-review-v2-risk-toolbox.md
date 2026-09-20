# 33 补三 — v2 架构风险审查与"风险-应对"工具箱(2026-09-21)

> **性质**:第三轮调研,目的从"生态位/撞车"转为"风险审查与修复弹药"。
> **输入**:① 两轮文献调研(本目录全部前置文档);② 项目实测症状(docs/31 §4
> 风险登记、docs/32 §S3 首读:σ 坍缩死锁、排序弱于覆盖、分量利用率低、均值陷阱
> 碎片、零泛化间隙、复合误差、snap 无效等);③ 四路专项外部检索(MDN/GMM 训练、
> 嵌入几何与流形、LM 训练学与解码、NAT 病理与分词风险)。
> **组织方式**:每条 = 风险(症状+根因假说)→ 学界应对(附 URL)→ v2 适用性
> (可直接用/需改造/不适用+原因)。末尾有优先级总表与"不要做的事"负面清单。
> 不修改任何既有文档;文档仅供用户参考,是否采纳由用户拍板。

## 0. 风险全景(七区 25 条)

```
A. MDN/GMM 头        A1 σ坍缩  A2 排序弱于覆盖  A3 分量利用率  A4 均值陷阱  A5 候选选择
B. AE 嵌入空间       B1 死区/离流形  B2 各向异性监控  B3 Zipf容量挤压  B4 漂移/snap无效  B5 低维瓶颈
C. 骨干训练学        C1 欠拟合拐点判据  C2 exposure bias  C3 复合误差  C4 密度目标守卫
D. 生成与解码        D1 多样性-质量权衡  D2 EOS/长度学习
E. 分词与数据        E1 随机碎切垃圾串  E2 坏分割闸门  E3 预烘焙等价性  E4 词界对齐取舍
F. 评估口径          F1 NLL呈现  F2 teacher-forced CE伪影  F3 幽灵词量化
G. 防御性写作        G1 NAT谱系对照  G2 分量内独立的残余风险表述
```

---

## A. MDN/GMM 输出头

### A1. σ 坍缩死锁(574046 已发生;sigma_floor=0.05 是现状)

**症状/根因**:分量一开始拟合 σ 冲下限,NLL 梯度按 1/σ、1/σ² 放大爆炸,按 CE
尺度标定的守卫全场拦截→死锁。这是 MDN 固有力学,不是实现 bug——RL 域十年来
公认"K 个学习 σ 的分量"难训,主流做法是绕开而非硬刚。

**学界应对**:
- **log-σ 参数化 + 双侧 clamp/tanh 软映射**(SAC/PPO 系十年标准:`log_std =
  clamp(log_std, -5, 2)`;对数空间学习+有界区间=σ 冲下限时参数梯度天然有界,
  且无硬 clamp 的梯度死亡)。 [CleanRL SAC](https://docs.cleanrl.dev/rl-algorithms/sac/ "citation")
- **softplus + ε 下界**(GIVT 的实证选择,σ 下界 ε=1e-5,并明言下界是为防"真值
  落入低密度区时损失爆炸")。 [GIVT App A.1](https://arxiv.org/html/2312.02116v2 "citation")
- **自然梯度 EM 预处理(ngem,2602.10602)**:梯度按 σ² 缩放,高方差区大步、
  低方差区保守——1/σ² 爆炸的直接反药;且能逃出"covering"局部极小(一个分量
  盖住多个真值簇)。10× 收敛加速。实现成本极低(三行 custom JVP)。
  [arXiv HTML](https://arxiv.org/html/2602.10602v1 "citation")
- **σ 课程**(Hjorth & Nabney 1999,经 Makansi 转述):先学 μ(σ 固定)、再学 σ、
  最后联合。注意 Makansi 报告即便如此仍 mode collapse——缓解非根治。
  [Makansi CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf "citation")
- **逃生门:固定 σ**(CLaM-TTS):σ 从 AE 侧统计冻结,只学 π+μ,砍掉全部 σ
  病理;代价=min-σ 重排失效。 [CLaM-TTS §4.2](https://arxiv.org/html/2404.02781v1 "citation")

**v2 适用性**:**P0**——`σ = exp(clamp(logσ_raw, log σ_min, log σ_max))` 替换
ELU+1+floor,数行改动;ngem 梯度预处理 <20 行。另:GIVT 明确警告 GMM NLL 的
初值/终值/降幅与 CE 完全不同(Appendix B.1),**守卫阈值必须按 GMM NLL 尺度
重建基线**——574046 事故的文献级印证。

### A2. 排序弱于覆盖(覆盖率 60.9% vs argmax 32.7% vs top5-π 49.5%)

**根因**:π 只从 NLL 的 responsibility 获得间接梯度,没有被显式训练去"认出赢家"。
轨迹预测域有一模一样的症状与成熟疗法。

**学界应对**:
- **赢家分类损失(轨迹域标准配置)**:MTR/Wayformer 给每个 hypothesis 单独训练
  score 头,CE 指向赢家;WTA 系损失只管 μ/σ 覆盖,π 排序由单独 CE 负责。
  [aWTA §IV-A](https://arxiv.org/html/2409.11172 "citation")
- **软赢家目标**(Kreutz et al. ICLR 2024):目标 π_k ∝ exp(−s·误差_k),离真值
  越近分量分越多概率,非硬 one-hot。 [OpenReview](https://openreview.net/pdf?id=C6EHiLaiBT "citation")
- **CLaM-TTS 变分下界**:q(k|·) ∝ exp(−KL(目标分量))——π 信号来自结构化的
  "哪个分量最近"。 [§4.2](https://arxiv.org/html/2404.02781v1 "citation")
- **推理侧 MBR/centroid 选择**:候选集效用矩阵选中心候选,O(K²) 在 K=64 免费。
  [MBR finetuning](https://arxiv.org/pdf/2309.10966v6.pdf "citation")

**v2 适用性**:**P0,首选**——给 π 加辅助 CE,目标 = softmax(−NLL_k/T)
(stop-grad),<50 行,直接攻"π 排序嫩";与 A3 的熵项可共用目标分布。

### A3. 分量利用率低(有效分量 4.4/64,未坍缩)

**学界应对**:MoE 负载均衡损失移植(Switch:f·P 点积,α≈0.01)、Loss-Free
Balancing(在线调 π logits 的 per-component bias,免梯度干扰)、π 熵/Dirichlet
正则、aWTA 退火天然提升利用率(早期全分量都在动)、"宁多勿少+推理期剪枝"。
[MoE 综述 §5.2](https://www.arxiv.org/pdf/2510.23027 "citation"), [aWTA Fig.2](https://arxiv.org/html/2409.11172 "citation")

**v2 适用性**:**优先级低**——症状是"没用满但未坍缩",瓶颈不在分量数;且 GIVT
默认 k=1 也能出 SOTA 级 FID,**K=64 只用 4.4 个未必是病,可能 K 本来就过大**。
留作利用率继续恶化时的工具。 [GIVT §4.1](https://arxiv.org/html/2312.02116v2 "citation")

### A4. 多模态位置的均值陷阱(argmax μ 落簇间 → 碎片)

**根因**(双层):π 排序嫩(A2)+ covering 局部极小(Chen et al. 2024,经 ngem
引述:GMM NLL 局部极小只有 covering 与 stacking 两类,covering 的 μ 恰在簇间)。

**学界应对**:
- **确定性退火(aWTA/EWTA 谱系)**:责任权重 q ∝ exp(−ℓ/T(t)),T 指数衰减;
  早期全体分量向条件均值收敛,降温相变逐次分裂出模态;minFDE −23%,所需假设数
  64→6,免 NMS。 [aWTA 全文](https://arxiv.org/html/2409.11172 "citation")
- **推理侧别用 argmax-π**(轨迹域共识:WTA 系 π 分数不可靠,普遍 NMS/聚类后选;
  v2 观测的"升温浮出真词"与此吻合)。
- **密度峰值选择**(升级 min-σ):选使 π_k·N(μ_k;μ_k,σ_k) 最大的分量——密度
  峰值而非权重峰值,一行代码可测。 [ngem §3.4](https://arxiv.org/html/2602.10602v1 "citation")

**v2 适用性**:**P1 高价值**——aWTA 退火不改架构,只差一个温度调度,同时治
A2/A3/A4 三条;密度峰值选择是纯推理侧实验。

### A5. min-σ(+3pp)之外的候选选择

**学界应对**:MBR-centroid(解码串间 edit similarity 或 AE 解码器重构建置信做效用);
GIVT 工具箱(σ 温度与 π 温度解耦、top-k over 分量、DB-CFG);CLaM 配方
(top-p=0.5 分量截断 + σ 温度校准到验证集经验标准差)。
[GIVT §3.4](https://arxiv.org/html/2312.02116v3 "citation"), [CLaM §5](https://arxiv.org/html/2404.02781v1 "citation")

**v2 适用性**:全部**纯推理侧、无需重训**;推荐顺序:密度峰值 → MBR-centroid →
σ 温度扫描;CLaM 的"σ 温度校准到验证集统计"可直接抄。

---

## B. AE 嵌入空间

### B1. 死区/离流形(CALM 断言纯重建空间 "practically impossible")

**症状**:重建只约束流形上的点;骨干预测 λ̂ 落无数据区→碎片。实测抗噪
0.07σ→99.5%/0.15σ→86%(尚可非免疫)。CALM 的修复是 VAE 化+KL clip+双 dropout
——我们不想引 KL,**替代方案存在**。

**学界应对**:
- **Latent mixup(Khan & Storkey, AISTATS 2023)**:**不用 KL**,mixup 训练填充
  潜空间空区,提升零密度区鲁棒性且不牺牲重建——**本轮找到的最强 KL-free 死区
  修复**,实现成本极低。 [arXiv](https://arxiv.org/html/2208.03923v3 "citation")
- **Latent dropout**(CALM 同款,z-dropout 0.15 迫使编码冗余化)。
- **输入腐化 DAE 的流形解释**(Alain & Bengio 2014):输入加噪学重建 → 重建
  残差收敛到 σ²·score 向量场(指向高密度方向)=**"自动拉回流形"的数学保证**。
  **v2 已拍板的字节腐化式(typo 簇增强)正是其离散版,有理论背书**。
  [JMLR](https://jmlr.org/papers/v15/alain14a.html "citation")
- **几何诊断**:pullback metric G(z)=J^T J 特征谱(Nazari 2023 的 GAE 正则、
  Arvanitidis 2018 的谱诊断)——32 维下 32×32 矩阵算得起,先作诊断。
  [GAE](https://arxiv.org/pdf/2306.17638v1 "citation")

**v2 适用性**:mixup 与 z-dropout **可直接用**;字节腐化已在工具箱(docs/32)
且有 Alain-Bengio 理论背书;CAE/GAE 正则需改造(梯度成本翻 2-3 倍)。

### B2. 各向异性监控(SC 0.384 / PR 23.9/32 / aniso +0.405)

**关键认知**:各向异性≠伤害——Machina & Mercer(NAACL 2024)证明它不是
transformer 固有性质;Ait-Saada(ACL 2023)证明聚类任务上不必然有害。
**不要对各向异性绝对值设阈值;干预信号锚在功能性指标(val EM 分层、抗噪曲线、
骨干 NLL)恶化上**。 [Ait-Saada](https://cnrs.hal.science/hal-04471739/file/ACL_2023_ait-saada.pdf "citation")

**学界应对**(监控组合):平均成对余弦(Ethayarajh 口径)+ PR/effective rank +
IsoScore + **Zipf 加权均值/协方差**——Yokoi(NeurIPS 2024)指出标准 whitening
隐含均匀频率假设、与 Zipf 现实错配,频率加权 whitening 全面超基线,**与 v2 的
Zipf 场景直接对口**。 [Zipfian whitening](https://arxiv.org/pdf/2411.00680 "citation"), [Mu & Viswanath](https://openreview.net/pdf?id=HkuGJ3kCb "citation")

**v2 适用性**:监控指标可直接加;whitening/ABTT 是后处理,对"冻结 AE+骨干 AR"
管线意味着同步变换骨干输入,只能作 A/B 臂,不建议默认。

### B3. Zipf 容量挤压(32 维装 ~76M 去重串)

**根因**(双叠加):SSL 意义的维度坍缩(奇异值谱快衰)+ Zipf 下高频串挤占容量。
外部印证:LLM token 嵌入内禀维度研究显示**高频 token 早期聚进低维子空间、低频
散落外围**,Zipf 被点名为影响因子。 [arXiv:2503.02142](https://arxiv.org/html/2503.02142v1 "citation")

**学界应对**:
- **VICReg 双项(最高优先级)**:逐维方差 hinge + 协方差去相关,无负样本无双塔,
  **可直接叠加到重建损失上**;SSL 消融显示不加方差项会坍缩、加过头丢信息(存在
  最优权重区间)。 [arXiv](https://arxiv.org/pdf/2105.04906 "citation")
- 奇异值谱全谱监控(Jing et al. ICLR 2022)。 [arXiv](https://arxiv.org/html/2110.09348v3 "citation")
- **FRAGE**(NeurIPS 2018):对抗判别器让嵌入空间无法区分高频/低频——与重建
  目标有张力(频率信息对重建有用),只在稀有串重建显著差时启用。
  [arXiv](https://arxiv.org/pdf/1809.06858 "citation")
- 长尾"特征云/几何迁移"(Liu 2020 等):给稀有模式人造邻域——**v2 的 typo 簇
  增强正是其字符串版,方向有学术先例背书**。 [feature cloud](https://arxiv.org/pdf/2002.10826v2 "citation")

**v2 适用性**:VICReg 双项 P1;FRAGE 与"去重词库"(docs/32 工具箱第 1 条)
哲学相反(一个抹平频率痕迹、一个重采样),**二选一,数据侧方案更便宜且不动训练
动力学,优先**。

### B4. 误差累积与 snap 无效(实测 0.334≈0.336)

**学界对 snap 无效的三层解释**(重要,可停止再调 snap):
1. **纯重建 AE 没有"拉回"保证**——Alain-Bengio 定理只对 DAE/CAE 成立,纯 AE
   的 enc∘dec 不逼近正交投影到流形。
2. **snap 只能纠正法向误差,纠正不了切向误差**——流体力学潜空间降阶模型的
   rollout 误差分析(2608.07189)给出惊人精确的对应结论:**97-98% 的 rollout
   误差方差是相位/切向漂移而非幅值/法向误差**,"网络学到了吸引子的几何,只是
   走的速率不对";且**训练期注噪不能消除漂移**(负结果,与 v2"去噪解码维持
   全关"拍板方向一致)。 [arXiv:2608.07189](https://arxiv.org/html/2608.07189v1 "citation")
3. **单步重嵌入不够,要迭代**——Vec2Text:单步"假设→重嵌入"几乎无效,~50 轮
   迭代修正才到 92% EM。 [Vec2Text](https://arxiv.org/pdf/2310.06816 "citation")

**学界应对**:LVDM 的 **conditional latent perturbation**(训练时条件用加噪
latent——scheduled sampling 的潜空间版);CALM 的 discrete feedback loop(实测
直接喂 latent 退化,解码回 token 再压缩输入更优);DAgger/DaD(见 C2);多步
pushforward 损失(见 C3)。 [LVDM](https://www.alphaxiv.org/abs/2211.13221 "citation"), [CALM §3.3.3](https://arxiv.org/html/2510.27688v1 "citation")

**v2 适用性**:① **验证先行**:测 λ̂−λ 在解码器 Jacobian 法向/切向的分解比例,
法向占比低则 snap 无解是定数,不必再调;② 训练期 λ 噪声注入(σ 取骨干实测预测
误差尺度 0.05-0.15σ 分档)一行级实现,与 detach 管线兼容;③ snap 可改造为
"迭代 2-3 轮 enc∘dec"或直接放弃,预算投给训练期鲁棒化。

### B5. 低维瓶颈 vs 可预测性张力(32 维在 CoSE 退化区间边缘)

**外部定量证据链**(视觉 tokenizer 文献反复验证同一现象):维数↑→重建↑但生成↓
(Yao et al. CVPR 2025;根因=从零学习无约束高维潜空间的固有难度,且潜空间分布
Gini 越高 gFID 越差);最优压缩率依赖第二阶段容量("When Worse is Better");
CALM 实测 **l=10 维即可 K=4 token 重建 >99.9%**——低维可行性证据强。
[Yao/VA-VAE](https://arxiv.org/html/2501.01423v2 "citation"), [When Worse is Better](https://arxiv.org/pdf/2412.16326 "citation")

**v2 适用性**:**定性背书 + 决策规则**——"低维本身不是病灶,无正则才是"(CALM
原话:纯重建 l=10 空间 impossible,修复靠平滑手段而非加维)。**若 32 维读数差,
优先加平滑/对齐正则(B1/B2),而不是先升维**;Gini/uniformity 可补入 AE 验收
指标。另:Bolmo Appendix E 的 rank 论据(子词嵌入本质高秩)是评审会引的反方弹药,
回应口径=**学习压缩 ≠ 线性截秩**(Bolmo 自己批评的是线性上投影)。

---

## C. 骨干训练学

### C1. "堆训练量 vs 架构有病"的拐点判据(60k 零泛化间隙,NLL 仍上探)

**学界应对**:
- **整曲线 scaling-law 拟合外推(首选)**:Tissue et al.(NeurIPS 2025)——CE
  全程服从 `L(s)=L₀+A·S₁^(−α)−C·S₂`(S₁=LR 曲线下面积、S₂=退火面积),拟合
  1-2 条短曲线可预测任意调度任意步,R²>0.998;**发散/崩坍曲线拟合不上——拟合
  优度本身就是优化健康度判据**。 [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/830b1abc6d2da85f23d41169fa44d185-Paper-Conference.pdf "citation")
- **梯度噪声尺度 GNS**:可作辅助,但 OLMo 实测其与临界 batch size 定量不符——
  弱指标,别单独依赖。 [GNS](https://arxiv.org/abs/1812.06162 "citation"), [OLMo CBS](https://arxiv.org/abs/2505.23971 "citation")
- **loss-to-loss 外推**:train↔test 间移位幂律,判断继续降训练 NLL 是否还改善
  留出。 [Kempner](http://kempnerinstitute.harvard.edu/research/deeper-learning/loss-to-loss-prediction/ "citation")

**v2 适用性**:**P1,脚本级**——把 60k POC 的完整 WSD 曲线按上式拟合,能拟合
并外推 1B 终值则"堆训练量"有定量依据;拟合崩了反而是架构信号。注意需改造验证
(文献基于 CE,MDN NLL 可负,L₀ 失去熵下界语义,幂律形式大概率仍适用)。

### C2. Exposure bias(teacher forcing vs 推理自喂 λ̂)

**学界应对与警示**:
- **Scheduled sampling 的目标不一致性(Huszár 2015,务必读)**:SS 是 improper
  scoring rule,收敛到最优时模型学会忽略前缀、只输出边际;后续机制研究(LRP)
  证实它靠"降低前缀依赖"起效,副作用是前缀本来就对时恶化。
  [arXiv:1511.05101](https://arxiv.org/abs/1511.05101 "citation"), [机制研究](https://arxiv.org/pdf/2109.06308 "citation")
- **DaD / DAgger**(Venkatraman AAAI 2015,专为连续状态时间序列):rollout 模型、
  把预测点"指回"真实轨迹下一状态、聚合重训——**保留 proper 目标**,只是前缀
  分布覆盖了推理分布。 [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9590 "citation")
- **反方(先测再上药)**:exposure bias 在域内影响有限(Wang & Sennrich ACL
  2020);自回归模型有自我恢复能力(He et al. EMNLP 2021)。
  [ACL 2020](https://aclanthology.org/2020.acl-main.326/ "citation")

**v2 适用性**:v2 的抗噪探针已表明骨干有恢复力——**先量化"rollout vs
teacher-forced NLL 差距随步数增长曲线",增长平缓就别动训练流程**。若要做,做
DaD 式 λ 前缀混合(低比例、保真值地板),连续 λ 空间无离散化断裂,比 token 版
SS 天然干净。

### C3. 复合误差(长生成被拖累)

**学界应对**:度量先行(教师强制 vs rollout 的偏差增长率);**多步 pushforward
损失**(训练时 2-4 步滚动展开惩罚累计 NLL)——**连续 λ 空间可微、无离散化障碍,
这是离散 token 时代做不到的事**,127M 规模可负担小步数版本;输入端纯高斯噪声
注入的负结果(B4-2)提示别指望它治本。

### C4. 密度目标的训练守卫(按 CE 尺度标定失效过一次)

**学界应对**:
- **双通道守卫**:loss 与 grad_norm 各自的 running median 双通道,**MDN 头与骨干
  分开监控**(MDN 头 spike 是 σ 坍缩前兆,骨干 spike 是另一回事);良性/恶性
  spike 区分(ZClip)——恶性签名=σ 接近 floor 且 NLL 突降(更负)。
  [ZClip](https://arxiv.org/pdf/2504.02507 "citation")
- **π logits 的 z-loss**(ST-MoE router z-loss 移植,K=64 softmax 在 fp16 V100
  下防 logit 漂移,零质量代价,几乎免费)。 [ST-MoE](https://arxiv.org/pdf/2202.08906 "citation")
- LLM spike 谱系:GLM-130B(fp16 spike 随训练变频繁;OPT 的"手动跳过+回滚"正是
  v2 spike skip 的直系祖宗;根治=DeepNorm+EGS)。 [GLM-130B](https://arxiv.org/html/2210.02414v2 "citation")

**v2 适用性**:双通道守卫 + π z-loss 均为代码小改,P1。sigma_floor=0.05 的做法
有文献背书(variance clipping 是标准做法)。

---

## D. 生成与解码

### D1. 多样性-质量权衡(distinct 低、回声循环;τ=1.3 浮出真词)

**学界应对**:
- **π-温度有先例但语义不同(呈现措辞关键)**:Graves 2013 手写合成引入偏置 b
  同调 π 与 σ;IMPS(NIME 2019)明确区分"π-temperature"(分量选择锐度)与
  "σ-temperature"(高斯宽度)。v2 分量内只取 μ_k,**采样多样性完全集中在词级、
  组内零随机**——写作时应表述为"分量选择温度",引 MDN 采样文献。
  [Graves](https://arxiv.org/html/1308.0850v5 "citation"), [IMPS](https://www.nime.org/proceedings/2019/nime2019_paper050.pdf "citation")
- **distinct 低的对策可直接移植**:top-p on π(K=64 上做 nucleus 完全自然)、
  典型采样(按 π 熵做局部典型集,K=64 计算免费)、unlikelihood(惩罚已出现词
  的分量)、Mirostat 式在线控制。
  [typical sampling, TACL 2023](https://aclanthology.org/2023.tacl-1.7/ "citation")
- **粒度换算**:Phan et al. 的 token↔字节概率精确换算思想,可把 patch 级指标
  折算字节级与基线公平比较。 [OpenReview](https://openreview.net/forum?id=zGej22CBnS "citation")

### D2. EOS/长度学习是否学到位

**学界应对**:
- **长度 oracle 诊断(强烈推荐)**:模型自停 vs oracle 停(逐位置算重建 EM 取最高
  点)对照——oracle-EM ≫ 自停-EM 且系统性偏后=欠学习(该停不停),偏前=过学习。
  [Newman et al. 2020](https://www.alphaxiv.org/ko/abs/2010.07174 "citation")
- **长度分布校准**:生成长度分布 vs 训练 patch 长度分布直方图/KL。
- **EOS 加权**:CE 中给 EOS 行加权(W=10 显著改善长度遵循不损质量)——v2 的
  257 维 softmax 上一行代码级改动。 [arXiv:2506.05017](https://arxiv.org/html/2506.05017v1 "citation")
- EOS 坍缩病历(学生模型 EOS 概率 0.8→~0、长度爆炸):监控 EOS 概率轨迹本身
  就是早期预警。 [2609.20511](https://www.alphaxiv.org/abs/2609.20511 "citation")

**v2 适用性**:两个诊断脚本级,P1;EOS 加权留作欠学习确诊后的修复。

---

## E. 分词与数据

### E1. 随机碎切的垃圾串占容量

**学界证据**:
- **阴性锚**:Saleva & Lignos 2023——BPE 内部随机化对下游影响甚微("not much");
  随机化本身不是大风险源,但也别指望免费收益。 [ACL 2023](https://aclanthology.org/2023.insights-1.7/ "citation")
- **收益机制**:Cognetta & Zouhar(EMNLP 2024)——BPE-/MaxMatch-dropout 的诱导
  分布严重偏斜,收益来自"覆盖多样切分"而非噪声本身;**v2 应补做双投票随机切
  诱导的每词切分分布熵(偏斜度)分析**。 [ETH 全文](https://www.research-collection.ethz.ch/server/api/core/bitstreams/90a23562-cdd1-4890-901e-60af6ad2a5b4/content "citation")
- **垃圾容量命名先例**:Bostrom & Durrett 2020 的 junk tokens(低频中间态占词表);
  Magikarp 欠训练 token glitch。 [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.414.pdf "citation")
- StochasTok 正面对照(随机切分无损基准 + 字符理解大涨,超参一个数量级鲁棒)。
  [arXiv](https://arxiv.org/html/2506.01687v1 "citation")

**v2 适用性**:**放大器警示**——token LM 里垃圾串只占 embedding 行,v2 里垃圾
串占 **GMM 分量(K=64 是稀缺资源)**,风险量级高于文献场景;PickyBPE 式频率阈值
可移植为"低频串剔除"。防御引用时注意 Saleva/StochasTok 都在 token LM 场景,
须限定。

### E2. 坏分割闸门(不在 UTF-8 内部断 + 最小长度)

**学界证据**:**UTF-8 有效性是独立于困惑度的滞后能力**——PPL 在 2.1B token
收敛、UTF-8 有效性要到 4.2B(2× 滞后);"字节长度曝光"是独立难度轴(unseen
4 字节字符部分有效率仅 86.97%)。v2 的闸门直接删掉了这个能力负担——防御性写作
应引此作设计理由。 [arXiv:2606.14122](https://arxiv.org/abs/2606.14122 "citation")
另:碎片化计算代价量化(序列 3.4× → 16.5× 减速,+47.1% BPC)。
[arXiv:2602.11174](https://arxiv.org/pdf/2602.11174 "citation")

**v2 适用性**:可直接用;建议 val 报告**加字节长度分层**(CJK 3 字节串在 λ 空间
的重建难度单独报),补频率分层之外的第二轴。

### E3. 预烘焙随机化的等价性(目前只有口头论证)

**学界证据**:Kudo 2018 原文的近似论证(k=1 每曝光一采,迭代数足够时是好近似)
是最接近的原始依据;赫尔辛基噪声管线的反向陈述(多 epoch 小数据时在线重采样是
关键——反推单遍下离线预烘焙与在线等价)。**形式化等价证明未检到
(UNVERIFIED-as-absence)——可作为一句话命题自证**(两者对每条样本诱导相同边际
切分分布,单遍下训练损失期望相同,差异仅 mini-batch 级二阶效应)。
[Kudo 2018](https://aclanthology.org/P18-1007.pdf "citation"), [HELDA](https://helda.helsinki.fi/server/api/core/bitstreams/8d2b6982-32f1-4ba0-9ce8-d906c4424452/content "citation")

**v2 适用性**:可直接用;**"epoch 数=1"应写成 46GB 缓存的显式使用前提**——
若未来跑 >1 epoch,等价性失效,需在线重切或多份预烘焙轮换。另注意预训练随机化
+微调确定性的 mismatch 风险(2605.13436),下游评测时用哪种切分要写清。
[arXiv:2605.13436](https://arxiv.org/html/2605.13436v1 "citation")

### E4. 词界对齐 vs 跨词单元的取舍

**学界证据**:Nous Research 1.7B 受控实验(2604.27263)——**词起点边界是可移除
后仍持续改善的真归纳偏置**(v2 的最强外部支撑);终点边界先验泄漏未来信息
(与 v1 Δ 距离场退化教训呼应)。SuperBPE——跨词单元("of the"类)有真实收益
(8B 平均 +4.0%)。 [arXiv:2604.27263](https://arxiv.org/abs/2604.27263 "citation"), [SuperBPE](https://arxiv.org/abs/2503.13423 "citation")

**v2 适用性**:双向弹药——起点边界偏置是 v2 patch 设计的背书;**锁死词界
(patch 不跨词)是已知牺牲,写作时主动认领并引 SuperBPE 作 trade-off 讨论**,
别等评审指出。

---

## F. 评估口径

### F1. NLL 呈现(-32.8 是密度,可负,直接报会误导)

**学界口径**:
- **Theis et al.(ICLR 2016)经典警示**:平均对数似然、Parzen 估计、样本质量
  三者基本独立——任何单一密度数字不构成质量声明。 [arXiv](https://arxiv.org/pdf/1511.01844 "citation")
- **bits/dim 类比归一化**(PixelCNN 谱系);Nalisnick 的高维似然反直觉(OOD
  数据拿更高似然)。 [Nalisnick](https://arxiv.org/html/1810.09136v3 "citation")
- **BPB 是字节级社区标准**:LM Eval Harness 正式定义 bits-per-byte;MambaByte
  附录 E 给完整换算(bits/word=bits/subword=bits/byte 守恒)。
  [LM Harness](https://arxiv.org/pdf/2405.14782 "citation"), [MambaByte App E](https://arxiv.org/html/2401.13660v2 "citation")

**v2 适用性(P0,纯口径)**:① 标注"nats per patch,32 维对角高斯混合密度,
非离散概率";② 辅助给 bits/维(-32.8/32≈-1.03 nats/维,负值合法,类比 flow);
③ **主指标换 BPB 当量:NLL/patch ÷ 3.47 ÷ ln2**,可与 MambaByte/BLT 公开 BPB
直接对表;④ 不报 exp(NLL) 这类无意义换算。

### F2. teacher-forced 字节 CE 伪影(答错时置信地错)

Theis 教训直接适用——CE/似然与生成质量独立。**CE 仅作解码器诊断;对外主指标 =
EM / 分量覆盖率 / distinct / 重排后 EM**(重排对应学界 MBR/verifier 解码传统,
可作"候选生成+验证器选择"框架引用)。

### F3. 幽灵词量化(v1 只有定性判死)

NAT 的重复率度量不适用字节串(合法串大量重复字节);可移植的是"外部 LM PPL 量
流畅度"思路(纯 AR 字节 LM 给 v2 采样串打分,对比分量内独立解码 vs 逐字节 AR
重打分的 NLL 差)。**v2 原生度量建议:采样串的词典外率(AE 词库即闭集,直接
算)——文献无现成先例(UNVERIFIED-as-absence),可作贡献点**。
[Revisiting NAT at Scale](https://aclanthology.org/2023.findings-acl.763.pdf "citation")

---

## G. 防御性写作(评审问题的预置答案)

### G1. NAT 解法谱系 vs v2 对照表

| 解法路线 | 代表 | v2 对应物/关系 |
|---|---|---|
| fertility 潜变量(模式承诺先行) | Gu et al. 2018 [arXiv](https://arxiv.org/abs/1711.02281 "citation") | π_k 选分量 = 强化到逐分量确定的 fertility;**已内置** |
| 潜变量自编码 | LatentTransformer/FlowSeq/LaNMT [综述](https://arxiv.org/pdf/2106.08122.pdf "citation") | 冻结 AE + λ 空间同构;**已内置** |
| 知识蒸馏(数据侧去多模态) | Zhou et al. 2020 | 无教师;AE 重建目标天然单模态,免费替代;不适用且不需要 |
| Glancing 课程 | GLAT/latent-GLAT [arXiv](https://arxiv.org/html/2204.02030v1 "citation") | 词间依赖已由 λ 空间 AR 全保;词内 glancing 留作 AE 解码加固备选 |
| CTC/DAG 结构化解码 | DAT/DA-DLM [arXiv](https://arxiv.org/html/2609.15070v1 "citation") | 逐 Δ 链上 K×K 转移+Viterbi 可行(L≤32,O(LK²) 可接受);**fallback 设计** |
| 半自回归/迭代精炼 | SAT/CMLM | v2 定位即 SAT 谱系(词间 AR+词内并行);**已内置** |
| 句级潜变量扩散 | DiLaDiff [arXiv](https://arxiv.org/abs/2605.23605 "citation") | **与 v2 最直系并行工作(AE 潜空间+潜先验+并行离散解码),必引+区分**(v2=分量承诺+逐Δ查询+EOS;它=扩散先验+掩码扩散解码) |

一句话定位:**v2 = "潜变量承诺(加强到逐分量确定)× 半自回归(词间 AR)"的
谱系交点**。

### G2. 分量内独立的残余风险(书面表述建议)

表达力论证链:Bishop 1994/Li & Barron 2000——GMM 万能逼近,对角协方差不是表达
力瓶颈,瓶颈在 K 与优化;MoS 理论(softmax bottleneck)为"多混合优于单峰"提供
几何解释。 [Rothfuss best practices](https://www.researchgate.net/publication/331519536_Conditional_Density_Estimation_with_Neural_Networks_Best_Practices_and_Benchmarks "citation"), [MoS](https://arxiv.org/html/1711.03953v4 "citation")
**残余风险表述**:分量 μ_k 锁定确定串后 Δ 间已无多模态(设计意图);但若 σ_k
未坍缩到承诺串,分量内 Δ 独立会重现 v1 病理的**弱化版**(错别字级混合)——与
POC 观测的失败模式(edit≤2 占 53%)吻合,写作时主动认领并指向 A2/A4 的排序与
退火改进。

---

## H. 优先级总表(施工排序建议,供用户拍板)

| 优先级 | 措施 | 治 | 成本 |
|---|---|---|---|
| **P0** | log-σ + clamp/tanh 双侧界替换 ELU+1+floor | A1 | 数行 |
| **P0** | π 辅助软赢家 CE(softmax(−NLL_k/T),stop-grad) | A2/A4 | <50 行 |
| **P0** | NLL 呈现改 BPB 当量(÷3.47÷ln2)+ bits/维辅助 | F1 | 纯口径 |
| **P1** | ngem 式 σ² 梯度预处理 | A1/A4 | <20 行 |
| **P1** | aWTA 温度退火(责任权重软→硬) | A2/A3/A4 | 小 |
| **P1** | WSD 全曲线 scaling 拟合外推 1B 终值 | C1 | 脚本级 |
| **P1** | 长度 oracle + 长度分布校准(EOS 诊断) | D2 | 脚本级 |
| **P1** | spike 守卫双通道(loss+gn,MDN 头/骨干分监控)+ π z-loss | C4 | 代码小改 |
| **P1** | VICReg 双项叠加 AE 重建损失(防 Zipf 压坍有效秩) | B3 | 小 |
| **P1** | 推理侧扫描:密度峰值选择 → MBR-centroid → σ 温度 | A4/A5 | 纯推理 |
| **P2** | latent mixup / z-dropout(AE 死区 KL-free 修复) | B1 | 小 |
| **P2** | 训练期 λ 噪声注入(LVDM 式,σ 分档 0.05-0.15) | B4/C2 | 一行级 |
| **P2** | rollout-vs-TF 偏差增长率量化(再定要不要 DaD) | C2/C3 | 一次对照 |
| **P2** | val 加字节长度分层(E2);切分分布熵分析(E1) | E1/E2 | 脚本级 |
| **P2** | π 上 top-p/typical/unlikelihood 治 distinct 与回声 | D1 | 推理脚本 |
| **P3** | σ 课程(先固定 σ 学 μ/π) | A1 | 小 |
| **P3** | 多步 pushforward 损失(2-4 步可微展开) | C3 | 训练流程改动 |
| **P3** | 逃生门评估:CLaM 式固定 σ > DMOL > flow 头 | A 全部 | 中-大 |

## I. 负面清单(学界已证的坑,别踩)

1. **别对输入端纯高斯噪声注入抱治本期望**——潜空间 rollout 文献负结果
   (2608.07189),与项目"去噪解码全关"拍板一致。
2. **别用 scheduled sampling 而不读 Huszár**——目标不一致(improper scoring
   rule);要用就压低混合比例、保真值地板,或改 DaD 式。
3. **别对各向异性绝对值设阈值**——各向异性≠伤害(NAACL 2024/ACL 2023 反例);
   干预信号锚功能性指标。
4. **别指望 snap(单步 enc∘dec)修漂移**——纯重建 AE 无拉回保证 + 切向误差
   投影无效 + 单步重嵌入无效(Vec2Text);要迭代或放弃。
5. **别把 GNS 当硬判据**;别把任何单一密度数字当质量声明(Theis)。
6. **32 维读数差时别先升维**——视觉文献反复证明升维恶化可预测性;先加平滑/
   对齐正则(CALM:"低维不是病灶,无正则才是")。
7. **别把预烘焙等价性外推到多 epoch**——缓存显式前提 epoch=1。
8. **FRAGE 与去重词库二选一**(哲学相反);数据侧优先。

## J. UNVERIFIED 清单

Hjorth & Nabney 1999 原文(经 Makansi 转述);Rothfuss 2019 细节(经 Wiley 转述);
Dirichlet/L2 π 正则原始论文(经 Emergent Mind 转述);"字节级并行解码
out-of-lexicon rate"先例(不存在,可作贡献点);"预烘焙≡在线"形式化证明
(不存在,可自证);DCD/CoDD/DEMASK/DAPD 四篇仅经 DA-DLM 转引;Professor
Forcing 原文;scaling 拟合公式对 MDN NLL(可负)的适用性需自验。
