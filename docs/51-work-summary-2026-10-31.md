# 51. 本轮工作总结与归档(截至 2026-10-31;生成机制改造前收官)

> **归档说明**:2026-10-31 用户决定「先收官,做完整的文档归档,给下一个主意腾地方」——
> 下一轮计划把**生成机制从 GMM 概率建模改为潜空间扩散**(较大的架构改动),本轮只做
> 工作总结与归档,不做新设计。本文是 **v2 word 臂 0.5B 这一轮的收尾档案**,目标:
> 上下文全失也能凭本文件 + AGENTS.md 恢复状态、找到资产、对齐口径、看清决策史。
>
> 阅读顺序建议:**本文 §0–§2** → `AGENTS.md` 当前状态 → 需要细节时按 §9 索引跳转。

---

## 0. 一句话状态

v2 word 臂 0.5B 线已完成三段预训练:预训练 p2(**10B units**)→ CPT S 档(**20B units**,
step 96,292)→ edu5b 单语料续训(**25B units**,step 70,861);评测口径已于 2026-10-27
从 pushforward 切换为 **unit-space MC**(docs/49)。2026-10-31 用户查看本机生成样例后判断
本轮成果不达预期(原话「看起来是烂完了」),决定**终止本线**并把生成机制改为潜空间扩散;
本轮以归档收尾。

**当前无在跑训练/评测作业**(集群队列 2026-10-31 清点为空;本机无后台任务)。

---

## 1. 时间线(本轮,v2 word 臂 0.5B)

| 日期 | 里程碑 | 出处 |
|---|---|---|
| 2026-09-17 | v2 方向拍板:AE 重建塑形 + 冻结梯度 + 骨干自回归 + GMM/MDN 头 + 增强 BPE + EOS | docs/31、docs/32 |
| 09-18/19 | AE 基座定为 48-256(后升级 ae_48_256_word_full);POC 通过 | docs/32、docs/36 |
| 09-29 | **三废除**:非确定性切词 / 非线性 GMM 头 / 分量-词对应解码 | docs/39 #33 前记 |
| 09-30 | 四裁决:BPE 臂永久关闭、生成质量是唯一质量标准、learnable 输入臂判死、word 长训 112k | docs/39 #33 |
| 10-05 | p1 0.5B 验收(5B units,NLL −35.67,bpb 2.694) | docs/39 #38/#39 |
| 10-09 | p2 0.5B 验收(10B units,NLL −40.69,bpb 2.539) | docs/39 #40、docs/41 |
| 10-18 | CPT S 配方八项拍板(60% DCLM + 40% FineWeb-Edu,+10B units,退火尾) | docs/43 |
| 10-26 | CPT S 完成+验收(step 96,292);SFT 方案成文;用户指示暂停推进 | docs/39 #43/#44/#45、docs/48 |
| 10-27 | pushforward 退役 + MC 打分改 unit-space(docs/49);edu5b 病因实验拍板并发射 | docs/39 #46、docs/49、docs/50 |
| 10-30 | edu5b 完成(step 70,861);三 ckpt 训后评估(MC + 内部协议) | docs/50 §5.4、docs/39 #48 |
| 10-31 | ARC-Easy 31%→27% 口径核对;**本机生成样例**;用户决定收官 + 下轮改潜空间扩散;本文归档 | docs/50 §5.4(g)(h)、docs/39 #48/#49 |

---

## 2. 现行架构(v2,本轮终止点)

- **切词**:`bltz/segment*` — `segment_word`(尾随空格 word 单元)+ Qwen3.5 BPE 兜底
  (超长/不一致时概率重切);在线 `bltz/verify.py` `VerifyResplitter` 用冻结 AE 自检
  (熵阈值 0.2 bits、字符地板 ≤3B),失败单元 BPE 重切后再核验。
- **表示**:冻结 AE `ae_48_256_word_full`(48 维,reconstruction-shaped,梯度与骨干硬切断)
  —— `bltz/models/autoencoder.py`;AE 资产 = `$SCRATCH/ckpts/ae_48_256_word_full/best.pt`。
- **缓存**:cache v2(`bytes.npy / unit_flag.npy / unit_len.npy / meta.json`),块级轮换
  `ShardRotator`(`bltz/shards.py`,resident K + 后台预载换块)。
- **骨干**:adapter(48→2048→1024) + transformer(d1024 / L28 / H16 / S=4096,
  xformers block-diagonal doc mask,回退 dense),grad_ckpt 开;入口
  `scripts/train_bltz_v2.py` + `bltz/trainer.py`。
- **输出头**:linear GMM/MDN,K=64,diag σ,σ_floor=0.05;训练目标 MDN NLL
  (`bltz/objectives.py`)。
- **读出(密度泛函,硬性规则)**:snap(protocol top-500k 表 argmax/温度采样)、
  modes-peak(局部众数枚举 + 峰顶海拔加权 + 核采样)、mode(全局 MAP);
  **严禁把分量均值当词用**(docs/31 后记)。
- **训练配方**:WSD;fp16+GradScaler(init_scale=1024,growth 500);Adam β2=0.95;
  spike_skip=max(10×median,2000);ckpt 滚动 keep 3 + milestone weight-only;
  `train.steps` 为绝对终点;weight-only 重启 `rewarmup=500`;wall-out 自动续投
  (墙前 300s watchdog touch STOP → 近墙自投下一段);优雅停用 `STOP` 文件。
- **评测(现行口径)**:`scripts/mc_eval.py`(8 项 MC unit-space `acc_unit` 主指标)、
  `scripts/validate_mc_scoring.py`(G1–G6 闸门)、`scripts/diag_v2*.py`(内部协议)、
  `scripts/snap_decode.py`(protocol-500k bpb)、`scripts/gen_accept.py`(生成电池)、
  `scripts/run_accept.py`(四段驱动,现含 4 段;pushforward 段已删)、
  `scripts/lmeval_bltz.py` + 集群 `venv_lmeval`(lm-eval 0.4.13 适配器)。

> 下一轮若改潜空间扩散:以上「输出头 + 密度读出 + 以 GMM 密度为基础的评分」将替换/重定义,
> 而切词、AE 潜空间(48d)、缓存、训练器、集群与评测基建可沿用 —— 见 §8。

---

## 3. 资产清单(2026-10-31 实测)

### 3.1 集群 ckpt(`/gpfs1/scratch/yihe47/bltz/ckpts/`,总 79.9GB)

| 目录 | 文件(用途) |
|---|---|
| `ae_48_256_word_full/` | `best.pt` 12MB(冻结 AE,评测/生成共用) |
| `word_05b_p1/` | `last.pt`(p1 终态)、`best.pt`、`ckpt_pre_decay.pt`、`ckpt_s0010000.pt`、`ckpt_s0020000.pt` |
| `word_05b_p2/` | `last.pt`(p2 终态 step 50,863)、`best.pt`、**`ckpt_pre_decay.pt`(step 45,429 = CPT/edu5b 分支点)**、`ckpt_full.pt`、`ckpt_s0035000.pt`、`ckpt_s0050000.pt` |
| `word_cpt_s_stable/` | `last.pt`、`best.pt`、`ckpt_full.pt`、`ckpt_s0060000/70000/80000/90000.pt`(**s0070000 = edu5b 同算力对照**) |
| `word_cpt_s_anneal/` | `last.pt`(**CPT S 终态 step 96,292**)、`best.pt`、`ckpt_pre_decay.pt`(step 91,205) |
| `word_edu5b/` | `last.pt`(**edu5b 终态 step 70,861**)、`best.pt`、`ckpt_pre_decay.pt`(step 68,318)、`ckpt_s0050000/55000/60000/65000/70000.pt`、`ckpt_full.pt(.1/.2)`、`train.log`、`arc_verify.log` |
| `v2_bpe_60k/ v2_lin_2b/ v2_word_1b/ v2_word_1b_learnin2/ v2_word_long/` | 各 `best.pt`(及 bpe 的 `last.pt`),旧消融臂,结果均在 docs/39;总约 2.5GB |

**本地工作区 `checkpoints/`(评测基线/离线核对用)**:`word_05b_p1_last.pt`、
`word_05b_p1_last_norm.pt`、`word_05b_p2_last.pt`、`word_cpt_s_last.pt`、
`word_cpt_s_s95000.pt`、`word_cpt_s_stable_last.pt`、`word_edu5b_last.pt`(2026-10-31 从集群拉取)、
`ae_48_256_word_full/`、`protocol_table_500k.pt`;以及 `word_*_accept.log`、
`mc_eval_*.{json,md}`、`word_edu5b_gen_samples.md`、`gen_edu5b/{p2_last,edu5b}.log`。

### 3.2 集群数据/缓存(`$SCRATCH/data/`,总 166.5GB)

| 路径 | 规模 | 说明 |
|---|---|---|
| `fineweb10b_v2_word/` | 61.4GB | 词臂主暖启动缓存(p1/p2 训练数据;`eval_sample.pkl` 也在此) |
| `cpt_cache_dclm/` | 48.3GB | DCLM 缓存(71 shards,CPT 用) |
| `cpt_cache_edu/` | 35.5GB | FineWeb-Edu 缓存(7 shards,**5,348,918,922 units**,26.51GB 文本;edu5b 全部数据) |
| `cpt_cache_anneal_pool/` | 17.7GB | 代码/数学退火池(CPT 退火段) |
| `cpt_cache_main/`、`cpt_cache_anneal/` | ~1MB | 符号链接聚合目录(76 / 15 shards) |
| `fineweb_word_local/` | 1.27GB | 旧词缓存(188.6M units 采样 + shard;run_accept 协议 pkl/cache) |
| `sft/` | 926MB | smol-smoltalk parquet(484,570 对话,SFT 备料) |
| `tinystories_parquet/`、`raw/`、`cpt_ids/` | 945MB / 478MB / 115MB | 历史/辅助数据 |
| `old_word_eval.pkl`、`cpt_main_eval.pkl`、`cpt_anneal_eval.pkl`、`edu_eval.pkl` | 各 ~5MB | 各分布的 eval 采样 pkl(diag_v2 用) |

`$SCRATCH/protocol/protocol_table_500k.pt`(209MB)= p2/CPT/edu5b 验收同款 top-500k 协议表
(2026-10-30 从本机上传)。

**集群空间**:ckpts 79.9GB + data 166.5GB + protocol 0.2GB ≈ **247GB / 300GB 软配额**。

### 3.3 评测产物

- `$SCRATCH/mc_eval/mc_eval_{p2pre,cpts7,edu5b}.{json,md}`(各 ~30MB,含逐题 raw 分数);
- `$SCRATCH/mc_eval/arc_verify/mc_eval_{cptlast_300,cptlast_full,edu5b_300}.{json,md}`;
- `$SCRATCH/ckpts/word_edu5b/arc_verify.log`(31%→27% 口径核对原始日志,含逐 run SUMMARY);
- 集群日志:`/gpfs1/home/yihe47/bltz/logs/`(`word_edu5b_*.out`、`edu5b_eval_*.out`、
  `edu5b_arcverify_*.out` 等);
- 验收/内部协议日志:集群 `repo/checkpoints/word_{p2pre,cpts7,edu5b}_accept.log`;
  历史 p1/p2/CPT 验收日志在本机 `checkpoints/word_*_accept.log`。

### 3.4 环境

- 本机:`E:\MiniConda\envs\cose2\python.exe`(torch 2.11 cu130,RTX 4060 Laptop 8GB);
  工作区 `E:\Trash_things\char_lm`。
- 集群:`burgundy.hpc.cityu.edu.hk`(SLURM,用户 yihe47,`$SCRATCH=/gpfs1/scratch/yihe47/bltz`,
  repo `/gpfs1/home/yihe47/bltz/repo`);V100 32GB ×2(QOS 2 卡/3 天墙);
  训练 env = conda `cose2`;评测 env = `$SCRATCH/venv_lmeval`(lm-eval 0.4.13,
  `--system-site-packages`);HF 缓存 `$SCRATCH/hf_cache`;Qwen3.5 tokenizer
  `$SCRATCH/qwen35/models/Qwen3.5-2B`。

---

## 4. 本轮评测原始数据(只列数字与文件;判断留用户)

### 4.1 MC 8 项 unit-space `acc_unit`(docs/49;全量 items)

三个 ckpt:`p2pre` = p2 `ckpt_pre_decay`(step 45,429)、`cpts7` = CPT `s0070000`(+24,571 步)、
`edu5b` = edu5b `last.pt`(+25,432 步)。文件:`$SCRATCH/mc_eval/mc_eval_{p2pre,cpts7,edu5b}.json`。

| task | n | p2pre | cpts7 | edu5b |
|---|---|---|---|---|
| arc_easy | 2376 | 27.86% | 27.19% | 27.95% |
| arc_challenge | 1172 | 22.44% | 23.04% | 23.29% |
| hellaswag | 10042 | 24.64% | 24.82% | 25.32% |
| piqa | 1838 | 51.09% | 52.18% | 52.12% |
| openbookqa | 500 | 24.60% | 23.00% | 24.00% |
| boolq | 3270 | 62.17% | 62.17% | 62.14% |
| commonsense_qa | 1221 | 19.57% | 18.92% | 21.05% |
| winogrande | 1267 | invalid | invalid | invalid |

- 7 项有效均值:`acc_unit` p2pre **33.20%** / cpts7 **33.04%** / edu5b **33.70%**
  (JSON `summary` 字段是含 winogrande 恒 0 的 8 项口径 29.05/28.91/29.48,勿直接引用)。
- 配对同题逐项(McNemar 式,n=20,419):edu5b−p2pre = **+0.56pp,95%CI [+0.08, +1.04]**;
  cpts7−p2pre = **+0.06pp,95%CI [−0.42, +0.55]**。
- 逐任务 Δ 见 `scripts/report_edu5b.py` 输出。

### 4.2 内部协议(旧分布,同 p2/CPT 验收协议;`run_accept`)

| 指标 | p2pre | cpts7 | edu5b |
|---|---|---|---|
| held-out NLL(旧分布) | −38.895 | −40.060 | −42.543 |
| top5 覆盖 | 38.65% | 38.66% | 41.70% |
| argmax_pi EM | 22.87% | 22.64% | 25.33% |
| snap-500k T=10 bpb | 2.5866 | 2.5646 | 2.4792 |
| NMI(h) | 0.258 | 0.263 | 0.272 |

- 旧分布 bpb 参照:p2 终态验收 = **2.5390**(docs/39 #41)。
- edu 域内 held-out NLL:edu5b **−45.395** / cpts7 −41.899 / p2pre −40.526。
- edu5b 里程碑旧分布 NLL(50k→70k):−41.02 → −41.74 → −42.38 → −42.92 → −44.47
  (该 pkl ≠ 协议 pkl,只作轨迹)。
- 注意:内部四段的 **section 3(semantic)在集群 run_accept 中因缺 matplotlib 于画图处崩溃**
  (NMI 数值已算出;section 4 生成未执行)——生成样例改在本机用 `gen_accept.py` 直接跑,见 §4.4。

### 4.3 ARC-Easy "31% → 27%" 口径核对(2026-10-31)

- 2026-10-27 pilot(新口径首次读数)`acc_unit 31.00%` 的测量设置 = **CPT 终端 ckpt
  (`word_cpt_s_anneal/last.pt`)+ `--limit 300`(ARC-Easy 前 300 题)+ 本机 4060**。
- 集群同代码复跑(job 711594,`slurm/bltz_edu5b_verify_arc.sbatch`):同 ckpt + limit 300 =
  **31.00%**(acc_raw 30.67 / acc_char 30.33 / pmi 26.67,与 pilot 逐位一致);同 ckpt
  **全量 2376 题 = 26.39%**(raw 26.77 / char 27.36 / pmi 27.48)。
- 前 300 题 vs 全量(从全量 JSON 复算):p2pre 35.00/27.86、cpts7 30.67/27.19、
  edu5b 31.33/27.95;edu5b `--limit 300` 直跑 31.33%(与复算一致)。
- 代码版本:`scripts/mc_eval.py` md5 本机=集群 `dc976a87486fff6cfcb6c77e417289c8`,
  `scripts/lmeval_bltz.py` `a24e66d01cf632eb2e591b993e0e992a`;`mc_eval.py` 自 2026-10-27
  `8c65b34`(PMI 修复)后无改动。
- 产物:`$SCRATCH/mc_eval/arc_verify/`、`$SCRATCH/ckpts/word_edu5b/arc_verify.log`。

### 4.4 生成样例(2026-10-31,本机 RTX 4060;未用集群)

- 对比:**p2 终点** `checkpoints/word_05b_p2_last.pt`(step 50,863)vs **edu5b 终点**
  `checkpoints/word_edu5b_last.pt`(step 70,861;从集群拉取 1.9GB)。
- 协议:`scripts/gen_accept.py`(docs/38 §5),8 prompt × 6 读出 × 48 units,同脚本同种子;
  读出 = mode / modes-peak T=0.6,1.0 / snap argmax / snap t=0.8,1.0(全为密度泛函)。
  prompt 0–3 = 评测序列前 16 units;prompt 4–7 = 经典开头句。
- 产物:`checkpoints/word_edu5b_gen_samples.md`(并排全文,549 行,由 `scripts/gen_compare.py`
  生成)、原始日志 `checkpoints/gen_edu5b/{p2_last,edu5b}.log`。
- 用户 2026-10-31 查看后判断本轮成果不达预期(「看起来是烂完了」)→ 决定收官、下轮改生成机制。

---

## 5. 用户拍板决策史(本轮,按日期)

| 日期 | 决策 | 出处 |
|---|---|---|
| 2026-09-13 | D10 POC 方法论:看趋势/定性分析/快速迭代;弱参考用同级现成模型 | docs/39 |
| 2026-09-17 | v2 方向:AE 重建塑形 + 梯度硬切断 + 骨干自回归 + MDN/GMM 头 + 增强 BPE + EOS | docs/31/32 |
| 2026-09-19 | AE 验收三件套(几何/频率分层 val/distinct 曲线);全量 AE 先用 32 维(后升级 48) | docs/32/36 |
| 2026-09-24 | v1 全部封存;typo-2B 配方定为默认 | docs/39 |
| 2026-09-27 | 显存余量规约:V100 峰值 ≤26GB(余量 6GB) | AGENTS |
| 2026-09-29 | **三废除**:非确定性切词、非线性 GMM 头、分量-词对应解码 | docs/39 #33 |
| 2026-09-30 | 四裁决:BPE 臂永久关闭;生成质量是唯一质量标准;learnable 输入臂判死;word 长训吃甜点 | docs/39 #33 |
| 2026-10-09 | p2 验收通过 → +10B units 续训 → SFT(docs/41 §5) | docs/39 #40/41 |
| 2026-10-18 | CPT S 配方八项(60/40、零回放、peak 2e-4、退火 88/5/7 等);资源获取分工;集群纪律 | docs/43 |
| 2026-10-19 | 步数口径:train.steps 为绝对终点(事故修复) | docs/39 #43 |
| 2026-10-26 | CPT S 验收;SFT 方案成文(docs/48,6 项待定);指示暂停推进 | docs/39 #43/#44/#45、docs/48 |
| 2026-10-27 | pushforward 退役 + MC 打分改 unit-space;edu5b 病因实验六项拍板 | docs/39 #46、docs/49、docs/50 §8 |
| 2026-10-31 | **收官**:本轮成果判断不达预期,停止本线;下一轮把生成机制从 GMM 概率建模改为**潜空间扩散**(用户口述,未设计) | 本文 |

---

## 6. 已关闭方向(勿重启)

- v1 架构与边界探针(`boundary_2b` 停,不续);v1 全部封存(2026-09-24)。
- **纯 BPE 臂**(2026-09-30 永久关闭,使命已达成)。
- **learnable 输入臂**(2026-09-30 判死;冻结编码器不是瓶颈,预测任务只用 ~9 维)。
- **非线性 GMM 头**(mdn_depth=0 默认;深度阶梯反证)。
- **分量-词对应解码 / 分量均值当词**(三废除;密度泛函读出为唯一口径)。
- **非确定性切词**(纯 BPE bpb −15% 实证;word 级确定性 + BPE 兜底)。
- **pushforward 系列打分**(2026-10-27 退役:raw 总分被候选字节长度支配;
  MC 改为嵌入空间 GMM 密度打分,docs/49)。
- **单位内分步解码**(已关闭)。
- 本轮之后:GMM 概率建模作为生成机制将退役(用户 2026-10-31 决定改潜空间扩散;
  新方案待设计);M 档继续加数据、SFT 均**未施工**(SFT 方案 docs/48 保留备查)。

---

## 7. 未决技术问题(事实清单,供下一轮设计参考)

1. **生成机制改造**(下一轮主线):GMM 概率建模 → 潜空间扩散。涉及替换/重定义:
   输出头与训练目标、解码/读出(snap/modes-peak 依赖表与 GMM 密度)、
   MC 评分(docs/49 的 `S_raw` 以 GMM 对数密度为基础)、生成电池 gen_accept 的读出栈。
2. **0.5B 能力表现**:本轮 MC 数据(§4.1)与生成样例(§4.4)是现状记录;
   本轮终止的直接原因 = 用户对生成样例的判断,不是某个预注册指标。
3. **评测 harness 缺陷**:
   - `winogrande` invalid:lm-eval 0.4.13 该任务是「choice 作 context + 共享续写」格式
     (gold 下标在 `doc_to_text`=0/1,`doc_to_target` 返回续写文本),`mc_eval._gold_index`
     取不到下标 → 三个模型恒 0%。修复点 `scripts/mc_eval.py:_gold_index`;
   - 集群 conda env 缺 `matplotlib` → `diag_v2_semantic` 画图处崩溃,导致 run_accept
     section 4 未执行;
   - 口径统一:MC 全量 vs `--limit 300` 子集差异已量化(§4.3),后续报告需显式标注。
4. **SFT**(docs/48)未施工:方案/模板/cache 预算/评测基建已备料,6 项待拍板。
5. **资产清理**(未执行):`scripts/prune_ckpts.py` 清单(2026-10-26 起草)含 v2_* 旧消融、
   已完成 run 的滚动 full ckpt 等,预计可回收 ~100GB;当前 247GB/300GB,未触发告警。

---

## 8. 下一轮方向(用户意图记录,本轮不做设计)

用户 2026-10-31:「我打算把生成机制从 gmm 概率建模改成潜空间扩散是个比较大的架构改动,
这一轮只做工作总结归档,然后我要压缩上下文看」。

**与本轮资产的接口(事实性)**:

- 可沿用:切词(`segment_word` + verify)、冻结 AE 48d 潜空间、cache v2 与既有数据缓存、
  训练器骨架(WSD/wall 续投/ckpt 轮转/spike 守卫)、集群脚本与评测基建
  (diag_v2 / snap 表 / gen_accept / mc_eval / lmeval 适配器)、SFT 备料(docs/48)。
- 需替换/重新设计:GMM/MDN 头与 MDN NLL 目标、密度泛函读出(snap/modes-peak 工具链)、
  unit-space MC 评分(GMM 密度基础)、`docs/31/32` 中「MDN/GMM 头」相关设计前提。
- 数据与 ckpt 资产(§3)是只读基线;新的架构设计应另立文档(建议接 docs/52)。

---

## 9. 索引

### 9.1 docs/(按用途)

| 文件 | 用途 |
|---|---|
| 01/02/07/08/09 | v1 设计/施工/代码架构/runbook/路线(历史) |
| 03 | 文献调研 |
| 04/05/06/10-30 | v1/v2 各阶段实验与诊断记录(历史,含 #28 阶段总结、#29 节点诊断、#31/32 v2 设计、#34 评测协议、#35 v2 阶段总结、#36 AE 协议、#37 v2b 总结) |
| **38** | **评估协议权威**(四段/内部协议/生成) |
| **39** | **实验日志**(#1–#48 全部实验与事故;查阅用) |
| 40 | 下步计划(本轮 SFT/M 档决策点未推进) |
| 41 | 0.5B 里程碑总结(p1/p2) |
| 43 | CPT S 档施工方案+验收 |
| 44/45/46 | 外部调研(未跟踪,见 §9.3) |
| 48 | SFT 施工方案(待拍板,未施工) |
| 49 | MC 打分 unit-space(现行口径+闸门+决策记录) |
| 50 | edu5b 病因实验施工+结果(§5.4) |
| **51** | **本文:本轮总结与归档** |
| 42/ | 外部调研目录(未跟踪) |

### 9.2 关键代码/脚本

| 类别 | 文件 |
|---|---|
| 数据/切词 | `bltz/segment_word*`、`bltz/verify.py`、`scripts/build_cache_v2.py`、`scripts/assemble_cpt_caches.py`、`scripts/build_token_cache.py` |
| 模型/训练 | `bltz/models/{autoencoder,model_v2}.py`、`bltz/{mdn,objectives,trainer,shards}.py`、`scripts/train_ae.py`、`scripts/train_bltz_v2.py`、`configs/*.yaml` |
| 评测(现行) | `scripts/mc_eval.py`、`scripts/validate_mc_scoring.py`、`scripts/report_edu5b.py`、`scripts/diag_v2*.py`、`scripts/snap_decode.py`、`scripts/gen_accept.py`、`scripts/gen_compare.py`、`scripts/run_accept.py`、`scripts/lmeval_bltz.py`、`scripts/lmeval_compare.py`、`scripts/dump_eval_pkl.py` |
| 集群脚本 | `slurm/bltz_word_edu5b{,_smoke}.sbatch`、`slurm/bltz_edu5b_eval{_mc,_accept}.sbatch`、`slurm/bltz_edu5b_verify_arc.sbatch`、`slurm/bltz_word_cpt_s.sbatch`、`slurm/bltz_word_05b_p{1,2}.sbatch`、`slurm/bltz_lmeval{,_setup}.sbatch` |
| 维护 | `scripts/prune_ckpts.py`、`scripts/plot_*.py`、`tests/test_*.py`(19 个,脚本式) |

### 9.3 归档时的仓库/工作区状态

- **git**:本轮归档提交 = 本文件所在提交(`git log --oneline` 顶部);此前 HEAD 为
  `f065235`(docs/50 记录本机生成样例)。全部代码/文档已推送到远端。
- **集群 repo**:`/gpfs1/home/yihe47/bltz/repo`,归档时停在 `b6e1ca9`(ARC 核对);下次使用前
  `git pull --ff-only`。仓库内有未跟踪软链 `data -> $SCRATCH/data` 与
  `slurm/bltz_lmeval_ref.sbatch`(非正式脚本)。
- **工作区未跟踪文件**(刻意未提交,保留本地):`.tmp/`(临时脚本/日志)、`cs_staff.html`、
  `docs/42-external-survey-202610/`、`docs/44-fyp-co-supervisor-survey.md`、
  `docs/45-lm-architecture-representation-survey.md`、`docs/46-zhisong-zhang-mphil-playbook.md`。
- **本地 checkpoints/、data/、runs/、viz/** 均 gitignored;关键 ckpt 清单见 §3.1。

---

## 10. 恢复工作指引(下一轮用)

1. 读 `AGENTS.md`(唯一指令源,含硬性规则/环境/纪律)→ 本文 → 需要细节按 §9 跳转。
2. 环境检查(登录节点秒级只读):`squeue -u yihe47`;`sinfo -N -o '%N|%C|%O|%G|%t'`;
   `du -sm $SCRATCH/bltz/{ckpts,data}`。
3. 集群 repo 同步:`cd /gpfs1/home/yihe47/bltz/repo && git pull --ff-only`。
4. 评测/生成资产就绪检查:`$SCRATCH/data/fineweb_word_local/`、
   `$SCRATCH/protocol/protocol_table_500k.pt`、`$SCRATCH/data/{old_word_eval,edu_eval}.pkl`。
5. 常用命令:
   - 训练:`sbatch slurm/bltz_word_edu5b.sbatch`(模板参考;新架构需新 sbatch)
   - MC 评测:`$SCRATCH/venv_lmeval/bin/python scripts/mc_eval.py --ckpt ... --ae ... --tok ...`
   - 内部协议:`python scripts/run_accept.py <ckpt> <ae> --tag ...`
   - 生成:`python scripts/gen_accept.py <ckpt> <ae> --cache data/fineweb_word_local
     --units-pkl data/fineweb_word_local/eval_sample.pkl
     --table-cache checkpoints/protocol_table_500k.pt --steps 48`
   - 并排生成表:`python scripts/gen_compare.py --log A=... --log B=... --out ...`
   - 汇总报告:`python scripts/report_edu5b.py --mc ... --accept ... --diag ...`
6. 停/续纪律:改任何训练行为前先读 `AGENTS.md` 实验纪律;停 run 用 STOP 文件,
   判活性只看 `train.log` 步号;集群登录节点只跑秒级只读命令。
