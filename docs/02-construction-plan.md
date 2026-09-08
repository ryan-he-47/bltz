# 02 — 施工计划

> **状态:待用户评审。** 阶段划分、验收门、算力估算均基于 `01-design.md` 的决策记录;
> 所有 **[提议]** 项待拍板。每阶段结束写报告归档 `docs/`(docs-first 纪律)。

## 0. 阶段总览

```
Stage 0  骨架+数据管线        本机,无 GPU 重活          ~1-2 天
Stage 1  124M 单臂调通        本机 4060 (bf16)          ~2-4 天
Stage 2  twin 对照 (token 基线) 本机或集群               ~1-2 天
Stage 3  熵分割臂 (含熵模型自训) 本机+集群               ~2-3 天
Stage 4  1B 概念验证          集群 V100 (fp16)          ~5-10 天 [预算待拍板]
```

原则:每一阶段都有明确的**验收门 (Gate)**;不过门不进下一阶段;任何阶段失败
按 §6 失败预案处理,不许硬闯。

## 1. Stage 0 — 项目骨架 + 数据管线(本机)

**产出物**:
- 包结构(包名 `bytefield`,暂定待拍板):
```
bytefield/
  config.py        YAML + --set a.b=value 覆盖(浮点带小数点)
  segment.py       固定语义分割器(design §5.1)
  data.py          FineWeb-Edu streaming → 字节+边界掩码 → shard 缓存
  models/          encoder.py / backbone.py / head.py / model.py
  objectives.py    MTP 损失(design §6)
  trainer.py       Noam/WSD + spike guard + ckpt 纪律
  infer.py         推理循环(design §7)
configs/           default.yaml 权威配置
scripts/           diag_* / viz_* / 标定脚本
tests/             脚本式测试(无 pytest)
```
- 数据:FineWeb-Edu `sample-10BT`(HF 不 gated,ODC-By,~9.67M 文档 / ~9.95B GPT-2
  tokens ≈ ~44B 字节)streaming 下载 → 分割 → 分 shard 缓存到 `data/`(gitignored)。
  划分 train/val(dev 用官方 val 或自留 0.1%)。
- 分割器测试(`tests/test_segment.py`):拼接 roundtrip == 原文;UTF-8 边界安全
  (多字节字符不切断);长度分布统计(avg / p50 / p95 / max);CJK / emoji / 代码 /
  数学样本目检;随机双增强的分布影响报告。

**Gate G0**:分割器测试全过;缓存就绪;**长度分布报告交给用户确认**后再进 Stage 1。

## 2. Stage 1 — 124M 单臂管线调通(本机 4060,bf16)

**产出物**:
- 全部模型代码 + 单 batch 过拟合 smoke(必须能把一个小 batch 的 loss 打到接近 0)。
- 124M 固定语义分割臂训练:骨干 d=768/L=12,**[提议]** 2~5B 字节
  (估算:2B 字节 ≈ 0.33B patch ≈ 6·0.11e9·0.33e9 ≈ 2.2e17 FLOPs,4060 有效
  ~15 TFLOPs → ~4 h/臂)。
- 诊断 D-1~D-6 全跑(逐 Δ 曲线 / 边界一致性 / 梯度余弦 / patch 长度直方图 /
  BPB / θ_stop 扫描生成)。
- 实验日志 `docs/04-stage1-log.md`(继承 simple_point_cloud 的 EXPERIMENT_LOG 体例)。

**Gate G1**:BPB 正常收敛(无 NaN/平台期异常);**D-1 逐 Δ 单调劣化**(一票否决项);
推理循环在多个 θ_stop 下工作且质量↔速度单调;生成样本在训练分布上可读。
不过门 → 按风险登记册 R1~R7 对应退路处理,仍不过 → 停下来问用户。

## 3. Stage 2 — twin 对照(本机或集群)

- 基线:GPT-2 tokenizer(50,257,公开免申请;FineWeb-Edu 生态的标准选择)+
  同骨干配方的 Llama 式 124M(d=768/L=12,RMSNorm/RoPE/SwiGLU,与本架构骨干逐点一致),
  同数据、同字节预算。
- 核算:FLOPs/byte 按 BLT §4.5 公式(基线计入 embedding 查表 0-FLOP 与
  de-embedding 2·h·V);BPB 按 design §9 D-5 协议(双方都不见当前单元内前文,公平)。
- **Gate G2**:BPB 对比 + NFE/字节对比 + 完整报告 → **用户拍板 go/no-go** 进入熵臂与 1B。

## 4. Stage 3 — 熵分割臂(本机 + 可上集群)

1. 自训熵模型 **[提议]** d=512/L=14/heads=8/滑窗 512(~45M;BLT 消融:>50M 收益递减),
   数据 = FineWeb-Edu 2~4B 字节(估算 ~15h @4060;可上集群),local-block-causal,
   换行重置上下文,LR 4e−4 cosine。
2. 熵分割:近似单调约束 H(t)−H(t−1)>θ_r;**θ_r 标定到与固定分割臂相同的平均 patch
   长度**(否则两臂不可比——这是消融公平性的关键)。
3. 熵臂 124M 训练,与 Stage 1 臂除分割外**全同配置**。
4. **纪律:Stage 1 冻结时打 git tag;熵臂从同一 tag 分叉,只改分割相关配置。**
   (消除"配置漂移":第二臂开跑晚,若代码已迭代,两臂比较即被污染。)

**Gate G3**:两臂 BPB / 诊断 / 效率对比报告 → **用户选型**主分割策略进 1B。

## 5. Stage 4 — 1B 概念验证(集群 V100,fp16+GradScaler)

- 骨干 d=2048/L=25(对齐 BLT-1B global 的 1B);条件头 6×3072(~110M);
  编码器 ~12M。全模型 ~1.15B。
- 数据:FineWeb-Edu sample-10BT 全量(~44B 字节)。
- 算力估算:约 7e9 patch,6ND ≈ 6×1.1e9×7e9 ≈ 4.6e19 FLOPs;V100 fp16 有效
  ~50 TFLOPs → **单卡 ~10 天**。备选:截到 20B 字节(~5 天)。**[预算待拍板]**
- 工程:fp16+GradScaler(sm70 无 bf16);sbatch 模板带 `#SBATCH --exclude=gpu-v100s-06`;
  ckpt 每 20k 步归档 + head_latest 式权重档;训练日志 jsonl 自 logging;
  断点续训(完整 optimizer/RNG 状态)。
- 监控交接:训练在轨确认后由用户盯盘(沿用 MoB_Head "我盯不消耗token" 模式)。

**Gate G4(概念验证验收)**:留出集 BPB 达到 twin 阶段外推的合理水平;CUTE 风格
字符任务显著强于同规模 token 基线(定性复现 BLT 的字符优势);多 θ_stop 生成样本可读;
NFE/字节与带宽/字节核算成表。**写总报告 `docs/05-final-report.md`,用户宣布概念验证
成败。**

## 6. 失败预案(继承两个参考项目的纪律)

- loss spike:spike guard(max(10×median, 2000))自动跳过;真崩溃 → kill,从 best.pt
  以 lr_factor≈0.2 暖启(simple_point_cloud 实测流程),不许续着烂状态硬跑。
- 收敛异常:先跑诊断脚本定位(D-1~D-6),不许"应该能行"。
- 同一问题 3 次修复失败:回退到最近良好状态,记录已试方案,咨询 Oracle,
  仍无解则问用户。
- 集群作业:登录节点只跑秒级只读命令;VPN 掉线=停手待命,不探测重试。

## 7. 资源清单(用户兼任资源收集员)

| 资源 | 状态 | 获取 |
|---|---|---|
| FineWeb-Edu sample-10BT | HF 公开,ODC-By | 我可直下(streaming),无需用户操作 |
| GPT-2 tokenizer | HF 公开 | 直下 |
| BLT 熵模型权重 | HF gated,申请被拒 | **不需要**——自训(design §5 熵模型臂) |
| 集群时间 | burgundy V100 | Stage 4 时用户确认配额 |

## 8. 时间线(粗估,不含排队)

Stage 0-3 总计约 1~1.5 周(本机为主);Stage 4 约 5~10 天(集群单卡)。
关键路径:Stage 0 分割器 → Stage 1 单臂收敛 → G1 逐 Δ 诊断。
