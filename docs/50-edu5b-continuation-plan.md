# 50. FineWeb-Edu 单语料续训 5B units —— 病因定位实验施工计划(待拍板)

> 起草 2026-10-27。用户口径:"所有指征显示本次续训成果不达预期,预计的主题一致性、
> 事实记忆都没有出现,继续深究价值不大;为了排查病因,定位是改换数据集的训练动力学
> 问题还是模型本身的能力上限问题,从 p2 分支一个续训,只使用 FineWeb-Edu 数据集,
> 先安排约 5B units 训练量,可以考虑使用 CPT 段新增的 FineWeb-Edu 数据;预期续训
> loss 轨迹平滑接续 p2 稳定段末尾的 loss 轨迹;本次续训不考虑扩充数据集、代码/数学
> 语料退火,只关心在单一高质量语料(FineWeb-Edu)上,续训是否带来能力增量。"
>
> 本文是**施工计划(未执行,待拍板)**。执行者按 §9 的 checklist 逐步作业;所有路径/
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

### 2.2 scratch 预算(必须先腾空间)

- 现状:`$SCRATCH/bltz` = **276GB**(data 162G + ckpts 114G),soft quota 300GB。
- 本实验 ckpt 估算:滚动 full 3×5.75GB + last/best/pre_decay 3×1.93GB +
  5 个里程碑(50k/55k/60k/65k/70k)×1.93GB ≈ **33GB** → 276+33=309GB 超 soft。
- **前置动作(需批准)**:执行既有 prune 清单
  `sbatch slurm/bltz_prune_ckpts.sbatch --apply`,回收 **102.7GB(95.6GiB)**
  (清单见 `scripts/prune_ckpts.py`;保留 p2 last/best/pre_decay、CPT last/best、
  AE、全部数据缓存)。回收后 173GB + 33GB ≈ **206GB**,安全。
- ⚠️ **顺序约束**:prune 会删除 `ckpts/word_cpt_s_stable/ckpt_s0070000.pt`
  (同算力对照 ckpt)。**必须先跑 §5 的 pre-eval(含 CPT s0070000),再 prune**;
  或临时从清单剔除该文件(1.93GB)。

---

## 3. 训练脚本

已生成(未提交作业,待拍板):

- `slurm/bltz_word_edu5b.sbatch` —— 正式训练(单 phase + wall 自续投)
- `slurm/bltz_word_edu5b_smoke.sbatch` —— 50 步冒烟(独立 ckpt 目录)

冒烟验收项(全部通过才发正式):
1. 第 1 步 loss 与"p2 分支 ckpt 在该数据流上的 loss"一致(冒烟本身给出基准值,
   正式 run 前 50 步应逐位复现冒烟前 50 步 loss);
2. 无 NaN/Inf、无 spike_skip;步速 ≈10.5–11.5 s/步;显存峰值 ≤26GB;
3. `train.log` 起始 step=45429、lr 首步 ≈ 2e-4×(1/500);
4. 衰减起点/终点/decay_frac 在日志头部可核对。

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

### 5.3 评估作业安排

- **E0(训练前,释放 GPU 窗口)**:2 GPU 作业
  - GPU0:`mc_eval.py` → p2 `ckpt_pre_decay`(8 项全量)
  - GPU1:`mc_eval.py` → CPT `ckpt_s0070000`
  - 同一作业 CPU 端先 `dump_eval_pkl` 生成 `edu_eval.pkl`
  - 约 2–2.5h;结果 `$SCRATCH/mc_eval/mc_eval_{p2pre,cpts7}.json`
- **E0b(训练前,可选)**:单卡 `run_accept.py` p2 pre_decay 四段(≈40–60min)
- **E1(训练后,2 GPU)**:
  - GPU0:`mc_eval.py` → `word_edu5b/last.pt`
  - GPU1:`run_accept.py` → `word_edu5b/last.pt`(四段 + edu NLL)
  - 附加:edu5b 里程碑旧分布 NLL 曲线(单卡顺序,~5min/个)
- **E2(训练后,生成对比)**:p2 pre_decay vs edu5b 的 8 prompt × 6 读出并排文档
  (产 `checkpoints/word_edu5b_gen_samples.md`,供用户目检);事实 cloze 探针
  (若批准)。

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
| scratch 超 soft quota | 先 prune(--apply,102.7GB);若未批,先用 `ckpt_keep=2`+`milestone_every=10000` 压缩到 ~20GB 方案(需改 §2 参数) |
| 墙杀/重投 | 内置 auto-resubmit + STOP;日志判据只看 train.log |
| Adam 重置导致早期 loss 波动 | rewarmup=500(CPT 验证);前 1,000 步重点监控,异常即停 |
| 单语料过拟合/退火不足 | 预注册判据只认 **旧分布遗忘红线 + MC acc_unit**;in-domain NLL 不作能力证据 |
| 评估 GPU 被训练占用 | baseline eval 必须在训练前完成;post eval 等作业结束 |
| 与 CPT 对照混淆(5B vs 10B) | 同算力对照用 CPT s0070000(+24,571 步),不用 CPT 终态 |
| 数据序未复现 | 冒烟=正式前 50 步逐位对比(同 seed/同 config) |

---

## 8. 待拍板项(执行前逐项确认)

1. **LR 尾**:WSD 末 2,543 步(10% of new data)线性退到 0(**推荐**,与 CPT 同形)→
   还是纯 plateau 到 5B 结束(对比更"纯",但 endpoint 未退火)?
2. **prune --apply**:批准执行 102.7GB 回收(否则按 §7 的 ckpt 压缩备选)?
3. **同算力对照**:是否纳入 CPT `s0070000`(需在 prune 前评估;推荐)?
4. **事实 cloze 探针**:是否加(20–40 条,GPU 成本可忽略;直接回应"事实记忆"关切)?
5. **分支点确认**:`ckpt_pre_decay@45429`(推荐,平滑接续/与 CPT 同分支)vs p2 `last@50863`(需 re-warm 到 2e-4,loss 会跳)。
6. **评估预算**:E0+E1 合计约 4–6 GPU·小时(两个窗口),是否按 §5.3 全部执行?

---

## 9. 执行 checklist(拍板后按序)

| # | 动作 | 命令/产物 | 预估 |
|---|---|---|---|
| 0 | 读 §8 拍板项并确认 | 用户回复 | — |
| 1 | 起飞前体检 | `sinfo -N -o '%N|%C|%O|%G|%t'`;`squeue -u yihe47` | 1min |
| 2 | 生成 edu eval pkl | `sbatch` tiny: `scripts/dump_eval_pkl.py data/cpt_cache_edu data/edu_eval.pkl` → `$SCRATCH/data/edu_eval.pkl` | ~5min |
| 3 | E0 pre-eval(CPT s0070000 必须在 prune 前) | `sbatch slurm/bltz_edu5b_pre_eval.sbatch` → `$SCRATCH/mc_eval/mc_eval_p2pre.json`,`mc_eval_cpts7.json` | ~2.5h |
| 4 | prune | `sbatch slurm/bltz_prune_ckpts.sbatch --apply`;核对 `du -sh $SCRATCH/bltz` ≈173G | ~10min |
| 5 | 冒烟 | `sbatch slurm/bltz_word_edu5b_smoke.sbatch`;核对 §3 四项 | ~30min |
| 6 | 正式训练 | `sbatch slurm/bltz_word_edu5b.sbatch`;记下 job id;按 §4 稀疏监控 | ~80h/2 墙 |
| 7 | 训练完成核对 | `tail train.log` 到 step 70,861、lr→0;ckpt 完整性 | 5min |
| 8 | E1 post-eval | `sbatch slurm/bltz_edu5b_post_eval.sbatch` → `mc_eval_edu5b.json` + `word_edu5b_accept.log` | ~2.5h |
| 9 | E2 生成对比 + cloze(若批) | `checkpoints/word_edu5b_gen_samples.md`;cloze JSON | ~1h |
| 10 | 写实验记录 | docs/39 新条目(编号接 #46 后);更新 docs/40/AGENTS;提交 | — |

**监控/恢复**:任何会话恢复先读本文件 + AGENTS 当前状态;`ssh` 只看 `train.log`;
停 run 用 `touch $SCRATCH/ckpts/word_edu5b/STOP`;墙自续投看 `logs/word_edu5b_%j.out`
的 `[wall]` 行;若作业 FAILED 且无 last.pt,按 docs/39 #42 补记判据人工重投。

---

## 10. 产物清单(计划态)

- 本文件 `docs/50-edu5b-continuation-plan.md`
- `slurm/bltz_word_edu5b.sbatch`(正式训练)· `slurm/bltz_word_edu5b_smoke.sbatch`(冒烟)
- (执行阶段产出)`$SCRATCH/ckpts/word_edu5b/`、`$SCRATCH/mc_eval/*.json`、
  `$SCRATCH/data/edu_eval.pkl`、`checkpoints/word_edu5b_accept.log`、
  `checkpoints/word_edu5b_gen_samples.md`、docs/39 新条目
