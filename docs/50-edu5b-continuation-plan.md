# 50. FineWeb-Edu 单语料续训 5B units —— 病因定位实验施工计划(已拍板,执行中)

> 起草 2026-10-27。用户口径:"所有指征显示本次续训成果不达预期,预计的主题一致性、
> 事实记忆都没有出现,继续深究价值不大;为了排查病因,定位是改换数据集的训练动力学
> 问题还是模型本身的能力上限问题,从 p2 分支一个续训,只使用 FineWeb-Edu 数据集,
> 先安排约 5B units 训练量,可以考虑使用 CPT 段新增的 FineWeb-Edu 数据;预期续训
> loss 轨迹平滑接续 p2 稳定段末尾的 loss 轨迹;本次续训不考虑扩充数据集、代码/数学
> 语料退火,只关心在单一高质量语料(FineWeb-Edu)上,续训是否带来能力增量。"
>
> 本文是**施工计划(2026-10-27 已拍板,执行中;拍板结果见 §8)**。执行者按 §9 的 checklist 逐步作业;所有路径/
> 命令/判据自包含,可跨会话恢复。相关前置文档:[docs/43](docs/43-continued-training-plan.md)
> (CPT S 档配方)、[docs/38](docs/38-eval-reference.md)(评估协议)、
> [docs/49](docs/49-mc-scoring-unit-space.md)(现行 MC 口径)、docs/39 #43/#46(验收/口径历史)。

---

## 0. 实验命题与设计

### 0.1 要区分的两个假设

- **H1(训练动力学/数据切换问题)**:CPT 的"无能力增量"来自数据切换配方本身——
  60% DCLM + 40% Edu 混合、全新数据域、peak 2e-4 re-warm、以及末段代码/数学退火,
  让模型主要在"适配新分布",没有转化为通用能力。
- **H2(模型能力上限)**:0.5B / 20B units 尺度上,任何单一高质量语料的继续预训练
  都不会再带来可测的能力增量(与 docs/39 #35 的"120M 尺度能力墙"同构)。

**判别实验**:从 **p2 稳定段末尾同一个分支点**续训,**只用 FineWeb-Edu(单一高质量
语料)**,约 **5B units**,不换新数据、不混 DCLM、不做代码/数学退火;与 CPT 在
**相同分支、相同 batch/模型配置、相同 new-data 步数(≈2.5 万步)** 上做对照。
若 edu-only 仍无能力增量、且与 CPT 同算力对照无差异 → 支持 H2;若 edu-only 出现
能力增量(哪怕温和),而 CPT 同算力下没有 → 支持 H1(混合/退火/切换动力学吃掉了增益)。

### 0.2 与 CPT S 档的关键对照(同一分支,控制变量)

| 维度 | CPT S 档(已完成) | 本实验(edu5b) |
|---|---|---|
| 分支点 | p2 `ckpt_pre_decay.pt` step 45429(weight-only) | **相同** |
| 模型/batch/优化器/掩码/fp16 | d1024·L28·H16,S=4096,2×24×4096=196,608 units/step,xformers 掩码,fp16 init_scale 1024,Adam β2 0.95 | **相同** |
| 数据 | 60% DCLM + 40% FineWeb-Edu(去重),零回放 | **100% FineWeb-Edu(同一份 edu 缓存)** |
| 新增量 | +50,863 步 = +10.0B units | **+25,432 步 = +5.000B units** |
| LR | peak 2e-4 平台;末段 5,087 步退火到 0 | peak 2e-4 平台(**分支点 LR 原值,无 re-warm 跳变**);末段 **2,543 步**退火到 0(10% of new data,同 CPT 形状) |
| 退火段数据 | web 88.1 / code 4.9 / math 7.1 | **同 Edu(单语料,无代码/数学)** |
| 对照价值 | — | 与 **CPT 稳定段 s0070000**(=+24,571 步,几乎同算力)可直接对比"单语料 vs 混合" |

> 说明:CPT 也是从 p2 `ckpt_pre_decay` 分支,因此本实验是**唯一变量=数据配方**的
> 同算力对照(差异仅在 5B vs 10B 总预算;同算力格子用 CPT s0070000 补齐)。

---

## 1. 数据核算(已核实,2026-10-27)

- **缓存位置**:`$SCRATCH/data/cpt_cache_edu` = CPT 段新增的 FineWeb-Edu 缓存
  (已按 id 剔除 FineWeb `sample/10BT` 交集,见 docs/43 §1.1–1.2;
  源数据集 `HuggingFaceFW/fineweb-edu`)。
- **规模(实测 meta)**:7 个 shard,合计 **5,348,918,922 units = 5.349B units**
  (bytes/unit ≈ 4.956,文本约 26.5GB);各 shard 0.758–0.771B units。
- **本实验消耗**:5.000B units = 缓存的 **93.5%**,单遍、不重复(与 CPT 零回放一致)。
- **数据序**:`data.rotate_shards: {resident: 2, every: 2000}`(与 CPT 相同);
  每 2000 步 ≈ 393M units ≈ 半个 shard 换块;7 shard 轮换,粗粒度块随机。
- **缓存无需重建**:直接 `--set data.cache_dir=$SCRATCH/data/cpt_cache_edu`。

> **口径澄清(必读)**:训练 loss 是**数据分布的熵函数**,Edu 与 p2 旧数据
> (FineWeb-10B)不同分布,loss 绝对值不可直接对表。所谓"平滑接续 p2 稳定段末尾"
> 的可检验定义见 §6.2:①分支 weight-only 恢复引入的 re-warm 不产生损失尖峰/崩溃;
> ②同一 ckpt 在同一数据流上的首步 loss 可复现(冒烟 = 正式首步);③在旧分布上的
> held-out NLL / snap bpb 轨迹连续(遗忘红线不触发)。

---

## 2. 训练配方(精确参数)

| 项 | 值 |
|---|---|
| 分支 ckpt | `$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt`(step **45429**,weight-only 1.93GB) |
| 数据 | `$SCRATCH/data/cpt_cache_edu` |
| 新增步数 | **25,432** 步(= 5,000,134,656 units = 5.000B) |
| 绝对终点 | `train.steps = 70861`(45429 + 25432) |
| 全局 batch | 2 GPU × 24 seq × 4096 units = **196,608 units/步** |
| LR | peak **2.0e-4**(= 分支点 plateau LR,**无新 warmup**;`warmup=2000` 对 start_step>warmup 不生效) |
| re-warm | `train.rewarmup=500`(weight-only 恢复 Adam 重置的既有安全措施,CPT 同款;LR 前 500 步线性 0→2e-4) |
| 衰减尾 | 末 **2,543 步**(=10% of new data)线性退到 0:`decay_frac = 2543/70861 = 0.035887`,`final_frac = 0.0`;衰减起点 step **68,318** |
| 优化器/数值 | Adam β2=0.95;fp16 `init_scale=1024.0`,`growth_interval=500`;`grad_ckpt=true`;`attn_doc_mask=xformers` |
| 存档 | `ckpt_every=250`,`ckpt_keep=3`,`milestone_every=5000`(weight-only),主动 `ckpt_pre_decay`(衰减起点) |
| 输出目录 | `$SCRATCH/ckpts/word_edu5b` |
| 日志 | `$SCRATCH/ckpts/word_edu5b/train.log` + `.../bltz/logs/word_edu5b_%j.out` |

**步数口径(2026-10-19 事故教训)**:`train.steps` 是**绝对终点**,不是新增量;
本实验起点 45429 → 终点 **70861**,任何改预算都必须重算终点与 decay_frac。

### 2.1 墙钟预算

- CPT 实测 10.5–11.3 s/步 → 25,432 步 ≈ **74–80h**;QOS 墙 3 天(72h):
  **预计 2 段墙**(第一段约 22,900 步 ≈ 72h,第二段剩余 ~2,500 步 ≈ 8h)。
- 沿用 p2/CPT 验证过的 **wall-out 自动续投**(墙前 300s watchdog touch STOP →
  近墙自投下一段;手动 STOP 不触发;非零崩溃不重投)。
- 落盘时间:预计 **约 3.5 天**(含排队)。

### 2.2 scratch 预算(2026-10-27 实测 + 拍板处理)

- **现状**(用户自行清理部分 ckpt 后实测):`$SCRATCH/bltz` = **238GiB**
  (ckpts 76GiB + data 161GiB + hf_cache 0.22GiB + venv_lmeval 0.12GiB);
  soft quota 300GB。
- 本实验 ckpt 估算:滚动 full 3×5.36GiB + last/best/pre_decay 3×1.79GiB +
  5 个里程碑(50k/55k/60k/65k/70k)×1.79GiB ≈ **30GiB**
  → 238 + 30 = **268GiB**,在 soft quota 内(余量 ~32GiB)。
- **拍板结果(2026-10-27)**:不执行 prune 作业(用户已自行清理,空间已够);
  hf_cache 仅 0.22GiB,清理无意义 → 不动。训练期间若撞 quota 告警,处置见 §7。
- **保留资产(评估/对照用,勿删)**:p2 `ckpt_pre_decay.pt` / `last.pt` / `best.pt`、
  CPT `word_cpt_s_stable/ckpt_s0070000.pt`(同算力对照)、`word_cpt_s_anneal/last.pt`、
  AE `ae_48_256_word_full/best.pt`、全部 data cache。

---

## 3. 训练脚本

已提交(2026-10-27;冒烟结果见 §3.1,六项拍板见 §8):

- `slurm/bltz_word_edu5b.sbatch` —— 正式训练(单 phase + wall 自续投)
- `slurm/bltz_word_edu5b_smoke.sbatch` —— 50 步冒烟(独立 ckpt 目录)

冒烟验收项(全部通过才发正式):
1. 第 1 步 loss 与"p2 分支 ckpt 在该数据流上的 loss"一致(冒烟本身给出基准值;
   冒烟用 `log_every=10`,因此日志含 step 45430/45440/45450 的首步与前期 loss,
   正式 run 前 50 步应逐位复现冒烟在 45450 处的 loss 及后续同名可见步);
2. 无 NaN/Inf、无 spike_skip;步速 ≈10.5–11.5 s/步;显存峰值 ≤26GB;
3. `train.log` 起始 step=45429、lr 首步 ≈ 2e-4×(1/500);
4. 衰减起点/终点/decay_frac 在日志头部可核对。

---

### 3.1 冒烟结果(2026-10-27,已通过;验收 §3 四项)

- 作业 **693723**(stingy `gpu-v100s-01`),起点 step 45430(= 分支 45429 + 1),
  共 49 步,日志步:`45430 −40.0819 / 45440 −39.7836 / 45450 −41.1368 /
  45460 −40.582 / 45470 −39.0945 / 45478 −39.8632`,**0 spike_skip、无 NaN**,
  ~11.5 s/步,产物 best/ckpt_full/last 齐全(已删,一次性)。
- **正式 run 的比对基准**:step 45450 的 loss 应为 **−41.1368**(±0.01);
  若正式 run 首步即 NaN → 按 §7.1 处理(先换节点,不改配置)。
- 首投 693674 的 NaN 事故见 §7.1。

---

## 4. 训练期监控(稀疏,不上登录节点跑任务)

- 判 run 活性**只看 `train.log` 步号**(不看 squeue/sacct)。
- 建议检查点:起飞后 30min、每 6–12h 一次(用 `ssh ... tail -n 20 train.log`);
  关键字段:step / loss / gn / skips / lr / sec。
- 异常处理:
  - NaN/连续 skip:trainer 有 stall watchdog(500 连跳自动停);出现即分析,
    **不要**自动重投。
  - 手动停:在 ckpt 目录 `touch STOP`(优雅存退,250 步粒度);不要用 scancel。
  - 墙:auto-resubmit 已内置;若作业 FAILED 且无 last.pt,按 docs/39 #42 补记
    的判据检查,必要时手动重投同一 sbatch。
- 里程碑 weight-only 落盘后,可在**训练结束后**统一评估(训练中 2 GPU 全占,
  QOS 不允许并发 eval)。

---

## 5. 评估计划(训练前 baseline + 训练后 endpoint + 同算力对照)

### 5.1 评估矩阵(新 MC 口径 docs/49 + 内部协议 docs/38)

| ckpt | 角色 | MC 8 项(acc_unit 等) | 内部四段(snap/NLL/EM/NMI/gen) | edu NLL | 旧分布 NLL/snap |
|---|---|---|---|---|---|
| `word_05b_p2/ckpt_pre_decay.pt` | 分支 baseline | ✅ | ✅ | ✅ | ✅ |
| `word_cpt_s_stable/ckpt_s0070000.pt` | 同算力混合对照(+24,571 步) | ✅ | ⬜(可选) | ✅ | ✅ |
| `word_edu5b/last.pt` | 本实验 endpoint(+25,432 步) | ✅ | ✅ | ✅ | ✅ |
| edu5b 里程碑 s0050000/55000/60000/65000 | 轨迹(可选) | — | — | ✅ | ✅(NLL 曲线) |

- 新 MC 电池工具:`scripts/mc_eval.py`(8 项 0-shot 全量,`--pmi` 可开);
  两 ckpt 双卡并行 ≈ 1.5–2h/对。
- 内部四段:`scripts/run_accept.py <ckpt> <ae> --tag ...`(4 段:diag_v2 / snap-500k /
  semantic / gen_accept;本机验收同款)。
- edu held-out NLL:`scripts/dump_eval_pkl.py $SCRATCH/data/cpt_cache_edu
  $SCRATCH/data/edu_eval.pkl`(CPU,几分钟)→ `diag_v2` 加 `--units-pkl .../edu_eval.pkl`。
- 旧分布:pkl 用现有 `data/old_word_eval.pkl`(188.6M units 旧缓存采样)与
  `data/fineweb_word_local` 协议表。
- 生成人工审阅:8 prompt × 6 读出,p2 分支 vs edu5b endpoint 并排
  (`scripts/gen_accept.py` 两跑 + `make_gen_samples.py` 改双 ckpt)。

### 5.2 预注册判据(训练前定死,避免事后解读)

**训练健康(硬闸)**
1. 无 NaN/连续 skip/stall;skip 率 ≈ 1.5‰(CPT 水平)量级;
2. 冒烟前 50 步 = 正式前 50 步(逐位或 |Δloss|<0.01);首步 loss 无 >1.0 nats
   相对冒烟基准的跳变;
3. 衰减段正常到底(step 70,861,lr→0),无墙丢失。

**遗忘红线(硬闸)**
4. 旧分布 snap-500k bpb 相对 p2 分支退化 **≤+0.02**(CPT 同款红线);
   旧分布 held-out NLL 作为连续量报告(不作硬闸,CPT 时 +0.25)。

**能力增量(主问题;预注册)**
5. 主指标:8 项 MC `acc_unit` 均值,`Δ_edu = edu5b − p2_pre_decay`。
   - **H2 支持**:`Δ_edu ≤ +1.0pp` **且** `Δ_CPT(同算力 s0070000) ≤ +1.0pp`
     (两者都在噪声内)→ 0.5B 尺度能力上限;
   - **H1 支持**:`Δ_edu ≥ +2.0pp` **且** `Δ_edu − Δ_CPT ≥ +1.0pp`
     → 单语料续训能带来增量,而混合/退火配方吃掉了它;
   - 其余为**不确定**(需报告置信区间与逐任务表,不强行结论)。
6. 次要:edu held-out NLL 下降幅度(预期明显,只证明"拟合",不作能力判据);
   生成目检只认正向维度(语法骨架/主题黏合/流畅度/长程连贯;循环不否决);
   若批准**事实 cloze 探针**:固定 20–40 条常识/事实补全(如 "The capital of
   France is"),统计 mode/nucleus 生成的首个实词命中率 + 候选似然 margin
   (正确 vs 错误续写,unit-space 打分),p2 分支 vs endpoint 对比。
7. 逐任务表 + stderr + 长度诊断(docs/49 §5)必须完整落盘,不允许只报均值。

### 5.3 评估作业安排(2026-10-27 拍板:评估预算延后,训练结束后统一执行)

> **拍板**:用户指示"评估预算等训完再说"。因此 **E0 训练前 baseline 不在起飞前跑**;
> 训练结束后用一个 GPU 窗口统一评三个 ckpt(p2 分支 / CPT s0070000 / edu5b endpoint)。
> 前提:以上 ckpt 训练期间不得删除(已列入 §2.2 保留资产;prune 不执行)。

- **E0+E1(训练后,2 GPU 作业,~4–6h)**:
  - GPU0:`mc_eval.py` → 依次 p2 `ckpt_pre_decay`、CPT `ckpt_s0070000`、edu5b `last.pt`
    (8 项全量,~50–70min/个)
  - GPU1:`run_accept.py` → 同上三个 ckpt 的内部四段 + edu NLL
    (CPU 端先 `dump_eval_pkl.py $SCRATCH/data/cpt_cache_edu $SCRATCH/data/edu_eval.pkl`)
  - 结果:`$SCRATCH/mc_eval/mc_eval_{p2pre,cpts7,edu5b}.json`、
    `checkpoints/word_edu5b_accept.log`
- **E2(训练后,生成对比)**:p2 pre_decay vs edu5b 的 8 prompt × 6 读出并排文档
  (产 `checkpoints/word_edu5b_gen_samples.md`,供用户目检)。
- **不做**:事实 cloze 探针(2026-10-27 拍板:不加)。

> 说明:用户已取消 CPT vs p2 的**验收重测**;本计划中的 MC 基线/对照仅服务于本
> 实验的 H1/H2 判别(分支基线 + 同算力对照),不重开 CPT 验收结论。

---

## 6. 预期与判读

### 6.1 预期

- **in-domain(edu held-out NLL)**:应明显下降(5B 新 token 的拟合),但**不能**
  直接当代能力增量(CPT 的教训:新分布 -3.85 nats 中通用能力仅 -0.25)。
- **训练 loss 轨迹**:Edu 分布的 loss 水平与 p2 旧数据不同,首步 loss 由冒烟给出;
  预期训练中平滑下降、无 re-warm 尖峰。
- **能力**:H2 先验概率较高(与 docs/39 #35 一致);本实验给出可判读的一次性证据。

### 6.2 "平滑接续 p2 稳定段末尾"的操作化定义(避免误读)

1. **无优化器冲击**:分支为 weight-only(Adam 重置),`rewarmup=500` 下首步
   LR≈4e-7,首步 loss 必须等于"同一 ckpt、同一数据流、无更新"的 loss(冒烟测得);
2. **可复现**:正式 run 前 50 步 loss 与冒烟前 50 步一致(|Δ|<0.01);
3. **无尖峰**:前 1,000 步内 rolling-50 loss 不超过冒烟首步 loss +1.0 nats;
4. **旧分布连续**:旧分布 held-out NLL / snap bpb 的里程碑轨迹连续(遗忘红线以内)。
   注意:Edu 训练 loss 与 p2 旧数据 loss **不可直接对表**(不同分布熵),报告时注明。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 队列拥挤(2 GPU × 3 天,QOS 2 卡) | 起飞前 `sinfo` 体检;若 gpu_v100s 无空闲,等窗口或与用户确认 stingy 备选(单卡/更慢) |
| scratch 超 soft quota | 实测 238GiB + ~30GiB = 268GiB < 300GB(2026-10-27);若训练期间撞告警,优先删 2026-09 旧消融 ckpt(`v2_*`,14.8GB,已在 prune 清单)或 CPT stable 旧里程碑 |
| 墙杀/重投 | 内置 auto-resubmit + STOP;日志判据只看 train.log |
| Adam 重置导致早期 loss 波动 | rewarmup=500(CPT 验证);前 1,000 步重点监控,异常即停 |
| 单语料过拟合/退火不足 | 预注册判据只认 **旧分布遗忘红线 + MC acc_unit**;in-domain NLL 不作能力证据 |
| 评估 GPU 被训练占用 | baseline eval 必须在训练前完成;post eval 等作业结束 |
| 与 CPT 对照混淆(5B vs 10B) | 同算力对照用 CPT s0070000(+24,571 步),不用 CPT 终态 |
| 数据序未复现 | 冒烟=正式前 50 步逐位对比(同 seed/同 config);已验证数据流确定(见 §7.1) |
| 启动期 NaN(节点/GPU 故障) | 已发生一次并定位为节点瞬时故障(§7.1);判据:同命令行重跑首步 loss 是否逐位复现;起步即 NaN 先换节点,勿改模型/数据 |

### 7.1 启动期 NaN 事故与定位(2026-10-27,已闭环;docs/39 #47)

- **现象**:冒烟首投 **693674**(stingy `gpu-v100s-01`)step 45430 正常
  (loss −40.0819 / gn 19.7),**从 45431 起 grad 全 NaN 并持续**;已 scancel。
- **定位(三轮探针 `slurm/bltz_edu5b_nanprobe.sbatch`)**:
  1. round-1:改用**全 preload reader(无 rotate)**跑 edu / main / p2data 三缓存各
     8 步 → **全部健康**(故非 edu 缓存问题;注意 preload 与 rotate 数据流不同,
     首步 loss 不同属正常);
  2. round-2:用**冒烟原命令行(含 ShardRotator)**在同一节点重跑 → step 45430
     loss **−40.0819 逐位复现**(证明数据序确定、命令行等价),45431–45437 **全部正常**;
  3. → **判定为节点/GPU 瞬时硬件故障**,与数据/配置无关;冒烟重投 **693723** 通过。
- **运维口径**:`gpu-v100s-01` 只挂 stingy(不在生产 gpu_v100s 的 02–07),有 GPU
  故障史;生产 run 一律走 gpu_v100s。**若某 run 起步 2–3 步内 grad 全 NaN:先换节点
  重投,不要改模型/数据/超参**;判定方法 = 同命令行重跑首步 loss 是否逐位复现
  (可给 sbatch 加 `echo host=$(hostname)`/`nvidia-smi -L` 留证,已加入本实验两个 sbatch)。

---

## 8. 拍板结果(2026-10-27,全部落定)

| # | 项 | 拍板 | 影响/落地 |
|---|---|---|---|
| 1 | LR 尾 | **WSD 末 2,543 步线性退到 0**(`decay_frac=0.035887`,`final_frac=0.0`) | 与 CPT 同形;§2 参数不变 |
| 2 | prune | **不执行 prune 作业**;用户已自行清理,实测 238GiB + 本实验 ~30GiB = 268GiB < 300GB;hf_cache 仅 0.22GiB 无需清 | §2.2 已更新;无 102.7GB 回收 |
| 3 | 同算力对照 | **纳入 CPT `ckpt_s0070000`**(+24,571 步混合数据) | 评估矩阵含该 ckpt;列入保留资产 |
| 4 | 事实 cloze 探针 | **不加** | E2 只做生成并排 |
| 5 | 分支点 | **`word_05b_p2/ckpt_pre_decay.pt` @ step 45429**(weight-only) | §2 不变 |
| 6 | 评估预算 | **延后到训练结束再定**("等训完再说") | E0 不在起飞前跑;E1/E2 训练后统一执行(§5.3) |

**执行状态(2026-10-27 04:55)**:✅ 冒烟通过(job **693723**,stingy `gpu-v100s-01`:
step 45430–45478 共 49 步、0 skip、无 NaN,loss −40.08/−39.78/−41.14/−40.58/−39.09/−39.86,
~11.5 s/步);✅ 正式训练 **693756** 已提交 gpu_v100s(2026-10-27 04:5x,排队)。
启动期事故与定位见 §7.1(docs/39 #47);正式 run 第一个日志步 step 45450 的 loss
应 = 冒烟的 **−41.1368**(±0.01,用于确认数据流一致)。

---

## 9. 执行 checklist(拍板后按序)

| # | 动作 | 命令/产物 | 状态 |
|---|---|---|---|
| 0 | 读 §8 拍板项并确认 | 用户 2026-10-27 已拍板(§8) | ✅ 已完成 |
| 1 | 起飞前体检 | `sinfo`/`squeue`;分支 ckpt、对照 ckpt、edu cache 实测核对 | ✅ 2026-10-27 03:34 通过(队列空;ckpt/cache 全在;meta=5,348,918,922 units) |
| 2 | 生成 edu eval pkl | 并入训练后 E0+E1 窗口(§5.3) | ⏭ 延后(拍板 6) |
| 3 | E0 pre-eval | **推迟**:训练后与 edu5b 一起评三个 ckpt(§5.3) | ⏭ 延后(拍板 6) |
| 4 | prune | **取消**:空间已够(238→268GiB < 300GB) | ⛔ 不执行(拍板 2) |
| 5 | 冒烟 | `sbatch slurm/bltz_word_edu5b_smoke.sbatch`;核对 §3 四项 | ✅ 693723 通过(2026-10-27 04:47);首投 693674 因节点瞬时故障 NaN,见 §7.1 |
| 6 | 正式训练 | `sbatch slurm/bltz_word_edu5b.sbatch`;记下 job id;按 §4 稀疏监控 | 🔄 **693756** 已提交,排队中(2026-10-27 04:5x) |
| 7 | 训练完成核对 | `tail train.log` 到 step 70,861、lr→0;ckpt 完整性 | ⏳ |
| 8 | E1 post-eval | `slurm/bltz_edu5b_eval.sbatch`(三 ckpt 统一评)→ `mc_eval_*.json` + accept log | ⏳ |
| 9 | E2 生成对比 | `checkpoints/word_edu5b_gen_samples.md` | ⏳ |
| 10 | 写实验记录 | docs/39 新条目(编号接 #46 后);更新 docs/40/AGENTS;提交 | ⏳ |

**监控/恢复**:任何会话恢复先读本文件 + AGENTS 当前状态;`ssh` 只看 `train.log`;
停 run 用 `touch $SCRATCH/ckpts/word_edu5b/STOP`;墙自续投看 `logs/word_edu5b_%j.out`
的 `[wall]` 行;若作业 FAILED 且无 last.pt,按 docs/39 #42 补记判据人工重投。

---

## 10. 产物清单(计划态)

- 本文件 `docs/50-edu5b-continuation-plan.md`
- `slurm/bltz_word_edu5b.sbatch`(正式训练)· `slurm/bltz_word_edu5b_smoke.sbatch`(冒烟)
  · `slurm/bltz_edu5b_eval.sbatch`(训练后三 ckpt 统一评估;替代原 pre/post 两个旧文件)
- (执行阶段产出)`$SCRATCH/ckpts/word_edu5b/`、`$SCRATCH/mc_eval/mc_eval_{p2pre,cpts7,edu5b}.{json,md}`、
  `$SCRATCH/data/edu_eval.pkl`、`checkpoints/word_{p2pre,cpts7,edu5b}_accept.log`、
  `checkpoints/word_edu5b_gen_samples.md`、docs/39 新条目
