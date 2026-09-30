# 38 — 评估协议参考(独立完整版,2026-09-29)

> 本文是**自包含的评估协议手册**:合并 docs/34(协议清单)、docs/36(word-AE
> 协议)与历次修正案,可脱离历史文档独立执行。判据阈值 = 当前基线的
> "不应明显劣于";跨会话/跨模型迁移以本文为准。

## 0. 方法论铁律(违反即无效,优先级最高)

1. **OOD 禁令**:模型对比必须在对齐其训练分布的数据上,或在**双方训练
   分布的中立交集**上测。OOD 样式上的差距是尺子假象(教训:旧 AE 在
   尾随空格样本上 EM 21-43%,中立纯词上 99.89%)。**修正案(09-29 第二起)**:
   探针/评测集的**来源切分器**也要对齐——diag_ae.py 曾用 legacy
   segment_v2 重切探针池(BPE 前导空格风格),对 word 臂构成 OOD,
   产生 last-chunk 0.663 vs 0.912 假回归;已修为直采缓存单位。
2. **密度泛函硬规则**(AGENTS 硬性规则):解码/读出只认 **snap argmax 与
   GMM mode**;分量均值当词已废除(dec 仅作诊断)。GMM 分量是密度基函数
   而非语义单元,合作叠峰时模式可不在任何 μ_k 上。
3. **bpb 是跨模型/跨切分唯一硬通货**:NLL/PPL/EM 依赖 unit 尺(切分方案
   改变即换尺),只有按字节归一的 bpb 可比。
4. **自由生成质量以目检为准**:对单一参考续写的 EM 是有偏指标(好文本≠
   轨迹吻合),只用于退化/崩溃检测,不用于质量评分。

## 1. 训练动力学(回答:这个 run 是否健康)

| 项 | 怎么跑 | 量什么 / 判据 |
|---|---|---|
| loss 曲线分段 | `scripts/plot_train.py <log> out.png` + 肉眼 | 段值/末值/衰减形态;稳定磨 vs 平台 vs 异常;退火尾段应加速下探 |
| spike/gn | train.log `skips` 字段 | 跳步率 ~1/10³ 步量级正常;持续 gn=NaN/Inf 流=死(589611 事故);500 连跳触发停滞看门狗属预期 |
| LR 调度 | 训练脚本打印 lr vs 配方 | WSD/CosSched 按配置走;续训 peak ≤ 上一 run 结束 LR(house rule) |
| ckpt 卫生 | ls ckpt 目录 | 滚动保留(disk quota 规约);里程碑是否所需;resume 可用性 |

## 2. AE 侧(回答:嵌入空间本身好不好)

工具:`diag_ae_v2.py`(三件套)、`diag_ae.py`(几何)、`diag_ae_typo.py`
(typo 设计意图)、`diag_v2_variants.py`(变体间距)。评测集用目标单位
分布的 pkl(`dump_eval_pkl.py <cache> <pkl>`),**对比双方同尺**。

| 项 | 量什么 / 判据(对照当前基线 docs/36) |
|---|---|
| 频率分层 val | zero/low/mid/high 桶 CE/bacc/EM;**zero 桶=真字符串级泛化**(word-AE 基线 99.29%) |
| 几何 | 各向异性(<0.4)/参与比(>25/48)/kNN-CC/**SC 轮廓系数**(CoSE:与预测质量 Pearson 0.92) |
| distinct/Heaps | 有效字符串量 V∝n^α(α~0.7);量级复现即健康 |
| 变体分辨率 | 自然变体(大小写/前后空格/屈折)EM ≥0.97;非自然形态纠错式坍缩为已知特性 |
| 变体方向性 | 变体位移对均值方向的对齐(基线:大写轴 0.315 方差解释、空格轴 0.383 对齐)>0.3 即存在系统轴 |
| typo 设计意图 | **保留率**(忠实重建 typo 目标,基线 97.65%;typo 是要忠实的样本不是噪声)/ σ_typo / 云半径 / snap 回源率(≈0)/ C1/C2 碰撞余量 / 噪声分层(0.5σ exact ≥0.8) |
| 编辑距离桶 | edit=1/2 对的 λ 距离 p10/p50(基线最近 edit≤2 p10 0.228) |
| 长度-精度曲线 | **按长度桶均匀采样**(防稀疏掩盖);EM+**len-match**;基线 L≤11 ≥99%,L12+ 陡降;>14B 为已知弱区 |
| 失败模式 | 数千例:长度/大小写/尾随空格/字节替换,word|symbol 分桶,落盘供目检;case≈0、纯空格=0、space+sub 最大为基线形态 |

## 3. 骨干侧(回答:预测力强不强)

工具:`diag_v2.py <v2> <ae> --units-pkl`。注意 **NLL 跨 λ 空间不可比**
(32d/48d、不同 AE 各成一界)。

| 项 | 量什么 / 判据 |
|---|---|
| held-out NLL | 同空间纵向比较(末值+衰减尾探底性) |
| 三读出 EM+编辑分布 | argmax-π / min-σ / conf-rerank;med/NED/prefix/edit≤k;失败模式(错字级 vs 乱码级) |
| top-k 覆盖 + **排序 gap** | top5 含真值;gap=覆盖−argmax=排序能力短板 |
| π 统计 | 熵/有效分量(基线 5-8/64)/top1 share;1=坍缩、64=均匀病 |
| persistence 排除 | "预测=上一 unit"基线 EM 应显著低于模型 |

## 4. 读出/解码专项(回答:密度能不能变成词)

工具:`snap_decode.py <v2> <ae> --units-pkl --ppl --gen [--mode]`;
**`pushforward_bpb.py`(推前字节 bpb,2026-09-30 用户设计,免表旁证尺)**。
表 = 协议表(§0 修正案:top-500k 缓存高频,BPE 臂并 Qwen 词表;覆盖报告必带)。

| 项 | 量什么 / 判据 |
|---|---|
| 四读出 EM | dec(诊断)/cos/gmm/**mode**;密度泛函应稳定压过分量读出(四连验证) |
| 校准 PPL/bpb(snap) | **温度校准必做**;协议表 500k;选择税一次/单位(=token LM 语义) |
| **表规格(2026-09-30 修正案,硬约束)** | snap-PPL/bpb 读数是表大小函数(粗表虚高:20.5k→2.19/120k→2.65/500k→2.77 收敛)。协议表=该臂缓存 top-500k 高频单位(BPE 风格臂另并 Qwen 词表),`--cache` 全 shard 扫频构建;报告必带表大小/覆盖率/OOV/校准 T;120k 仅冒烟档不作结论 |
| **推前字节 bpb** | P(b\|ctx,Δ)=∫GMM·P_dec 字节边际;免表/免校准/免 OOV;**残余偏差:不收单位内前缀条件(并行解码器本性),利短单位臂**;串级联合版已废(并行解码器精确整串=边际乘积,沉底无判别力) |
| coverage | 表外真值比例(协议表下 BPE≈1.0;word 500k≈0.994) |

## 5. 生成专项(回答:自由文本什么水平)

- **反馈铁律**:下一步输入只喂流形点(mode/snap 向量);裸 μ_k 非流形,
  喂入即乱码(三连验证);mode/embed 与 mode/reencode 逐字节恒等。
- 采样:多 prompt × 多 τ 全量日志落盘;**mode 确定性无 τ**,配温度是
  待试变体。
- 目检三签名:语法骨架 / 吸引子形态(词/句/符号级)/ 退化形态(数字、
  符号 run)。60k 期句级吸引子属能力深度,非读出病。
- 生成解剖工具(Temp probe_feedback.py):五方案 × 两反馈矩阵。
- **坑**:生成循环的骨干输入必须 UNSHIFTED(cat(bos, 全部已知 token)),
  backbone_input() 是教师强制 shift 语义,生成误用即差一位复读
  (2026-09-29 事故)。

## 6. 语义结构(回答:h/λ 里有没有语义)

工具:`diag_v2_semantic.py`。NMI(h 簇 vs 类型) < NMI(λ 簇 vs 类型) =
分工健康(h 松=语义签);簇内编辑 h > λ;120 例倾倒肉眼分型。

## 7. 工具清单(CLI)

| 脚本 | 用途 |
|---|---|
| scripts/dump_eval_pkl.py <cache> <pkl> | 评测 pkl(登录节点 CPU 可跑) |
| scripts/diag_ae_v2.py <ae> --units-pkl | AE 三件套 |
| scripts/diag_ae.py <ae> --cache | 几何套件(--cache 必带) |
| scripts/diag_ae_typo.py <ae...> --cache | typo 设计意图(多 ckpt 同台) |
| scripts/diag_v2_variants.py <ae> --units-pkl | 编辑桶变体间距 |
| scripts/diag_v2.py <v2> <ae> --units-pkl | 骨干+三读出+生成 |
| scripts/snap_decode.py <v2> <ae> --units-pkl --ppl --gen | snap/mode 四读出+校准 PPL+生成 |
| scripts/diag_v2_semantic.py <v2> <ae> --units-pkl | 语义结构 |
| scripts/probe_feedback.py <v2> <ae> --units-pkl | 反馈探针矩阵 |
| scripts/plot_train.py <log> <png> | 训练曲线 |

输出一律落 checkpoints/*.log;**诊断脚本必须带加载 canary**(打印 ckpt
step + 权重范数,防随机骨干事故重演)。
