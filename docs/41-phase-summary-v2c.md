# 41 — v2 阶段总结(第三辑:0.5B Chinchilla 里程碑,2026-10-09)

> **压缩上下文入口**:跨会话恢复读本文 + `AGENTS.md`;评估协议→
> `docs/38`,实验编年→`docs/39`(#37-#40),下一步→`docs/40`,架构→
> `docs/07`(现行版);历史辑:`docs/37`(第二辑)/`docs/35`(第一辑)。
> 本辑覆盖 2026-09-30 晚 ~ 10-09:word 臂 0.5B 两段式 Chinchilla 训练
> (5B→10B units)、边界掩码重大修复、xformers 优化、墙外自动续投零
> 人工、p1/p2 验收、数据账本与阶段口径裁定。

## 1. 0.5B Chinchilla 里程碑(核心结果)

**运行配置**:word 臂 0.5B(d1024/L28/H16,S=4096,per-GPU batch 24,
2×V100 DDP,fp16 init_scale 1024,grad_ckpt)+ word 三件套(确定性切词 +
在线核验重分词 + 块级轮换)+ xformers 文档分块掩码。两段式:

- **p1**:0→25432 步(5B units,退火尾→0;09-30 发射→10-05 验收,含
  OOM 修复与掩码改造停-续);
- **p2**:20004→50863 步(至累计 **10B units**,平台→退火;10-05→10-09,
  93.5h,跨 3 天墙一次,**wall-out 自动续投零人工**)。

**动力学**:loss(rolling-500)热身后 ≈-25 → **-41.2**,全程无平台,
终点仍在下探(-0.26 nats/1k 步)。曲线:`viz/word_05b_p1_curve.png` /
`viz/word_05b_p2_curve.png` / **`viz/word_05b_p1p2_curve.png`(合并)**。

**验收**(docs/38 全套五段电池;产物 `checkpoints/word_05b_p2_accept.log`,
p1 对照 `checkpoints/word_05b_p1_accept.log`):

| 指标 | word-1b (120M) | word-long (120M) | p1 (5B u) | **p2 (10B u)** |
|---|---|---|---|---|
| snap-500k bpb @T=10 | 2.766 | 2.747 | 2.694 | **2.539** |
| 推前 byteCE(免表) | 5.47 | 5.565 | 4.968 | **4.891** |
| held-out NLL | -34.1 | -34.79 | -35.67 | **-40.69** |
| mode EM | 23.0% | 24.1% | 23.6% | **26.3%** |
| EM(argmax/conf) | — | 21.0/21.8% | 21.3/22.0% | **24.4/24.5%** |
| top5 覆盖 | 35.1% | 36.4% | 36.3% | **40.4%** |
| NMI(h)/NMI(λ) | 0.259/0.257 | ≈平 | 0.254/0.257 | **0.271/0.257** |

横向锚(BPE 切分口径只以 bpb 对比,详 docs/39 #32/#18):BPE-60k bpb
2.425、lin-2B 2.70。——**0.5B@10B units 能力全面大幅上移且未饱和;
生成侧:modes-peak/τ 采样流畅度与主题黏合出现段落级 discourse 结构
提升**(生成判据按 10-09 口径裁定,见 §6)。

**数据账本(10-09 核)**:word 缓存 **9.178B units**(FineWeb-10B 全量
word 化;14 shards,46.0GB 文本);累计曝光 ≈14.4B units ≈ **1.57 遍**
(本 run 11.07B + word-1b 1.25B + word-long 2.06B);**全新余量≈0**;
+5B→2.1 遍 / +10B→2.7 遍(采样=shard 置换+随机,遍数=曝光口径)。

## 2. 实验记录(本辑编年;详情 docs/39 #37-#40)

- **#37 word 0.5B 发射(09-30)**:两段设计(50863 步=10B units;p1=5B
  验货段);OOM 根因=单 shard 4.85GB(含 len/flag),`resident 2`+mem 64G
  修复;QOS 定式=2 卡×3 天墙。
- **#38 边界掩码(10-03,用户发现重大设计失误)**:首版训练无文档边界
  处理→文档分块注意力(块对角 causal)+边界 loss 掩码;零缓存重建
  (unit_flag 写侧已有,读侧在线传播);同期事故:AE `best.pt` 被清缓存
  误删→本机副本恢复(sha256 逐位一致)。
- **#38 补记(xformers 效率优化,10-03 晚,用户指令)**:BlockDiagonal
  CausalMask 集成——dense 18.1→**xf 10.96 s/步**(反快于无掩码 12.86);
  V100 基准 64ms vs dense 267ms;停-续换装首战(STOP 优雅存退 ≤1 步)。
- **#39 p1 验收(10-05)**:对 word-long 全面小幅上移;事故:`last.pt`
  保存带 `module.core.` 前缀(评测直载失败)→规范化副本评测(权重
  逐位不变)+ trainer 修复。
- **#40 p2 验收(10-09)**:能力大幅上移(见 §1);wall-out 自动续投
  首次实战(605592 跑满墙→STOP saved_step 43740→自投 612122→完成,
  段间 ≤1 步);事故:§1 采样轨迹第 25 代数值发散(NaN)撞 torch CUDA
  multinomial device assert(TensorCompare.cu `input[0] != 0`)→ diag_v2
  isfinite 守卫(docs/38 入册);口径裁定(见 §6)。

## 3. 代码改动(09-30 ~ 10-09)

| commit | 内容 |
|---|---|
| 0381d56 | **边界掩码实施**:`bltz/masking.py`(新建;块对角 causal fp16 bias+边界权重)、`shards.py` sequence_units_flags/sequence_flags、`verify.py` process_batch 传播 doc_starts、`backbone.py`/`model_v2.py` attn_bias 接入、`objectives_v2.py` 掩码损失、`train_bltz_v2.py` `_LossModule`+build_batch doc_start、`trainer.py` BLTZ_MEM、`tests/test_boundary.py` |
| b0899cc | `scripts/bench_attn_mask.py`(注意力后端基准) |
| ad45ace | **xformers 集成**:`make_attn_bias` 工厂(auto/xformers/dense)、backbone 扁平 xf 路径(`apply_rope_flat`)、`model_v2.attn_mask_mode`、objectives 走工厂 |
| fb234e8 | trainer `last.pt` 保存补 `_state_model` 剥离(module.core. 事故) |
| 0fdc59b | `diag_v2.py` 生成循环 isfinite 守卫;docs/38 数值守卫条款 |
| 文档链 | c5ca52a/55fe6d8/5b2c724/ab42242/e2aa6d3/bdb15aa/a2f3454/fb234e8/5d2927b/0fdc59b/ca2f97b/e12b6f0/3f8e339(**口径裁定+数据账本**);sbatch 链 43f7798(3 天墙)/b4378db(p2 清单+wall-out 自动续投) |

关键机制速查(现行代码):
- **训练掩码**:`model.attn_doc_mask`(auto/xformers/dense;集群 auto=xf,
  本机无 xf 自动回退 dense);边界 loss 权重;零缓存重建(unit_flag);
- **ckpt 格式**:所有保存点(`save_full`/`last.pt`/`best.pt`)model 键经
  `_state_model` 剥离,直载 `BltzLMv2`;
- **wall-out 自动续投**(p2 sbatch 内):墙前 300s 看门狗 touch STOP
  (即存即退)→ wait 后"非零退出不重投 / 未完成且近墙才自投下一段";
- **测试**:18/18 全绿(含 test_boundary)。

## 4. 资产与悬置清单

- **ckpts**(`$SCRATCH/ckpts/`):`word_05b_p1/`(ckpt_pre_decay=step
  20003,平台分支点)、`word_05b_p2/`(pre_decay=step 45429、milestones
  25k/30k/35k/40k/45k/50k、last/best;键格式已干净)、
  `ae_48_256_word_full/best.pt`;本地:`checkpoints/word_05b_p1_last_norm.pt`、
  `checkpoints/word_05b_p2_last.pt`。
- **验收/曲线产物**:`checkpoints/word_05b_p{1,2}_accept.log`、
  `viz/word_05b_p{1,2}_curve.png`、`viz/word_05b_p1p2_curve.png`。
- **悬置**:①评测工具(snap/pushforward)掩码同口径未实施;②gen_long
  modes-peak 分支 isfinite 守卫待加;③SIGTERM 优雅存退在 torchrun 下
  不可靠(用 STOP 文件,已验证);④分隔符方案(U+2403 侦察完毕,未采用)。
- **集群依赖**:xformers 0.0.29.post1(conda cose2);QOS 2 卡×3 天。

## 5. 下一阶段(用户拍板 10-09):10B 量级新数据续训 → 指令微调

**目标链**:①用 **10B 量级新数据**(全新语料,避免重复)从 0.5B 里程碑
**续训**;②随后做**指令微调(SFT)**。

**准备清单(施工前,逐项待办/拍板)**:

- [ ] **新数据**(用户提供):来源/获取方式与规模(建议:更广 FineWeb
      切片 / FineWeb-Edu / 组合;目标 ~10B tokens 级);
- [ ] **word 缓存重建**:`segment_word`+核验管线(写侧含 unit_flag);
      参照上次成本:9.2B units ≈ 46GB 文本 + ~60GB 数组,构建数小时
      级;**先核 scratch 空间**(300GB 配额,现用 ~139G+ckpt 增长);
- [ ] **续训配方(待拍板)**:分支点(p2 `ckpt_pre_decay` step 45429
      平台点 vs 终态+rewarmup)、新旧数据配比、总预算、WSD 形状、
      LR peak(家规:≤ 父 run 平台 LR 2e-4);
- [ ] **SFT 设计(待拍板)**:数据(用户提供)、训练接口(unit 级目标
      的改造 vs 其它)、评估口径(生成目检 + 任务指标)。

**不变约束**:用户对一切技术决策最终拍板;用户兼任资源收集员(数据/
权重);历史 sbatch 归档不动;p2 系列产物保持只读。

## 6. 本辑教训与口径增量

1. **训练-评测口径必须同构**:文档边界掩码缺失=重大设计失误(模型曾
   学会跨文档横向抄写);掩码类改造需同步评估评测侧。
2. **ckpt 存读对称**:新增模型包装(DDP/_LossModule)必须全量检查保存
   点(`module.core.` 事故);统一走 `_state_model`。
3. **自由生成会数值发散**:采样工具一律加 isfinite 守卫;NaN 采样行撞
   torch CUDA multinomial 的 device assert(TensorCompare.cu)。
4. **wall-out 自动续投配方**(已验证,可复用):墙前 300s STOP 优雅停→
   近墙判定自投;段间损失 ≤1 步;手动 STOP 不误触发。
5. **停-续用 STOP 文件,不靠信号**:scancel/SIGTERM 在 torchrun 下丢存。
6. **口径裁定(10-09,用户)**:贪婪/确定性读出的循环是小模型普遍
   现象,**禁止当能力否决项**(AGENTS 硬性规则);生成评估只认正向
   维度(骨架/黏合/流畅/长程连贯)。
7. **数据账本意识**:大 run 前算语料总账与曝光遍数(本辑:9.178B
   units 缓存 / 1.57 遍)。
