# 48 — SFT 施工方案(smol-smoltalk;2026-10-26 起草,待用户审阅拍板)

> 前置:S 档 CPT 已完成并验收(`docs/39` #43 / `docs/43` §4.1;终态
> `word_cpt_s_anneal/last.pt` step 96292)。本文是 docs/43 §5「SFT 阶段」的
> 施工细化,依据用户 10-09/10-18 已拍板口径:纯通用 SFT=smol-smoltalk 全量、
> Llama-3 chat 式模板 `<|user|>…<|assistant|>…<EOS>`、loss 只算 assistant 段
> +EOS、prompt mask、标记走 BPE 兜底进 unit 流、序列预算 4096 units。
> **本文档通过前不动码。** 数据测量(§1)已于 2026-10-26 完成(集群 tiny job
> 692634,`scripts/probe_sft_data.py`,结果 `$SCRATCH/data/sft_probe.json`)。

## 0. 目标与验秤口径

- **目标**:纯通用指令微调,让 0.5B word 臂模型获得指令跟随格式;benchmark
  验秤看「字节能力是否经格式激发涌现」(用户 10-09 口径),不是刷分。
- **不掺字符合成数据**(docs/43 §5 已拍板;视 benchmark 结果再议)。
- **基线**:CPT 终态(step 96292)= SFT 的初始化;SFT 后必须同时报告
  ①SFT 收益(Tier 2)②旧分布遗忘(旧 eval_sample snap bpb/NLL,docs/38)③生成目检。

## 1. 数据盘点(已实测,2026-10-26)

`$SCRATCH/data/sft/smol-smoltalk/data/`:5 个 parquet(4 train + 1 test,共
**926MB**),schema=`messages: list<struct<content,role>>` + `source`。

| 项 | 数值 |
|---|---|
| 对话数 | **484,570**(train 460,341 / test 24,229) |
| 角色分布(抽样 1536 行) | user 3445 / assistant 3444 / **system 261(≈4%)** |
| 轮数分布(抽样) | 2 轮 25%、3 轮 16%、6 轮 58%、长轮(≥7)极小 |
| 渲染后字符数(5000 抽样) | mean 3912 / p50 3654 / p90 6637 / p99 13941 / max 62708 |
| **word-unit 数**(segment_word) | **mean 817** / p50 826 / p90 1261 / p99 2867 / max 19242 |
| **assistant 目标占比** | **73.1%**(5000 行抽样)/ 81.2%(前 100 行,长回答源偏置)——源异质,构建期报告全量实测值;只影响 loss 权重尺度不影响口径 |
| 截断率(>4096 units) | **0.88%**(44/5000) |
| **全量估算** | **≈396M units / epoch** → **2015 步/epoch**(196,608 units/步,与 CPT 同 batch 口径) |
| **每 epoch 墙钟** | **≈6.3h**(11.3 s/step 实测) |

标记切词实测(segment_word):`<|user|>`→`[<|, user, |>]`(3 units),
`<|assistant|>`/`<|end|>` 同;符号 run 会与相邻符号合并——**已用真实数据
实测并定模板**:轮间不加分隔符(见 §2.1/§2.3,probe 692652)。

## 2. 模板与切词

### 2.1 模板(chat 式;system 角色保留)

```
<|system|>{system}
<|user|>{user}
<|assistant|>{assistant}<|end|>
<|user|>{user2}
<|assistant|>{assistant2}<|end|>
...
```
- 无 system 时直接从 `<|user|>` 开始(smol-smoltalk 96% 无 system)。
- **轮间分隔:实测选「无分隔符」**(2026-10-26 实测,`scripts/probe_sft_template.py`,
  tiny job 692652,100 条真实对话/79,732 units):`\n` 或空格分隔会让跨轮符号
  合并成 `|>\n<|` / `|> <|`(130 例,0.16%),**这类 unit 的 AE 重建 flag 率
  7.87%**;无分隔符时合并形为 `|><|`,AE 可稳定重建 → **marker flag 率降到
  1.43%**(剩余失败来自代码围栏等正文符号融合)。三种变体的文本/切词总量/
  cross-turn 合并数完全相同,唯一差别就是合并 unit 形态 → **采用无分隔符**。
- **target 判定按字节区间**:assistant 内容 + 紧随的 `<|end|>`=1(计 loss);
  system/user 内容与所有 marker=0(prompt mask)。单位归属按「unit 首字节
  落在哪个区间」判定(与 `scripts/lmeval_bltz.py` 的 context/continuation
  切分同法,已验证稳健)。
- **EOS 口径**:word 流没有单元级 EOS;`<|end|>` 即"回答结束"标记,由模型学
  预测(3 个 unit:`<|`,`end`,`|>` 序列)。生成侧 harness 用 stop 序列
  `<|end|>` 截断。

### 2.2 切词(训练同源,离线烘焙)

- 用 **`segment_word`**(word 臂训练同源;**禁用 legacy `segment_v2`**,docs/38
  §0 OOD 铁律,2026-10-26 教训:segment_v2 会把 `b' village'` 变成 AE OOD
  的 `b'Ovillage'`)。
- **离线烘焙 online verify**:用 `VerifyResplitter`(ae_48_256_word_full,ent_thresh
  0.2 / floor_bytes 3)在 cache 构建期做一次核验重切(确定性,与训练侧在线核验
  等价)→ 训练 `train.verify.enabled=false`,免去 loss-mask 在线传播的代码改动,
  训练路径与 pre-training 完全一致。构建期报告每个 parquet 的 flag 率。
- **marker OOD 探针(2026-10-26 已实测,`scripts/probe_sft_template.py`)**:
  100 条真实对话(79,732 units)上,正文单元 AE flag 率 **0.59%**(高于预训练
  0.044%——SFT 内容含指令/代码/数学,属预期,在线 verify 会 BPE 重切);
  marker 相关单元 flag 率(无分隔符模板)**1.43%**,主要剩余形态是
  `\n```<|`/`|>``` ` 这类"正文符号+marker"融合;纯 marker 单元
  (`<|`/`user`/`|>`/`|><|`)均可稳定重建。**结论:无需改 marker 方案,
  不需要白名单**;构建期继续报告全量 flag 率。

### 2.3 已知边界效应(2026-10-26 已实测,记录,不阻塞)

- 跨轮符号合并:无分隔符模板下为 `|><|`(130/79,732 units = **0.16%**),
  AE 可稳定重建(EM/entropy 通过)→ verify 不改动,unit 粒度粗但文本不变、
  stop 仍可命中;占比低,不再需要备选分隔词。
- 正文与 marker 融合的形态(如 `.\n```<|`、`|>``` `)占比 <0.1%,由 verify
  按常规 BPE 重切处理。

## 3. 代码施工清单

### 3.1 数据侧

1. `scripts/build_sft_cache.py`(**新**):parquet → 模板渲染(记录每段字节 span)
   → `segment_word` → `VerifyResplitter` → 按 shard 写
   `unit_len.npy / unit_flag.npy(bit0=对话首 unit) / unit_loss.npy(0/1,新) /
   bytes.npy / meta.json`;心跳(进度+units/s+ETA);每 shard ≈50M units;
   预计输出 ≈4GB(unit bytes≈1.9GB + len/flag/loss 各≈396MB)。
   落盘 `$SCRATCH/data/sft_cache`。
2. 单测 `tests/test_sft_cache.py`:模板 span→mask 正确性(构造小样本逐 unit 核对)、
   mask 总量(全量构建期实测报告;抽样 see §1)、切词守恒(拼接=原文)、
   marker 单元快照。

### 3.2 训练侧(最小改动,向后兼容)

3. `bltz/shards.py`:`_load_shard` 可选加载 `unit_loss.npy`;新增
   `sequence_loss_mask(global_idx,S)`;`sequence_units_flags` 保持不变(旧缓存
   无该文件 → loss_mask=None)。
4. `scripts/train_bltz_v2.py::build_batch`:verifier is None 且 reader 有 loss
   mask 时写 `d["loss_mask"]`(bool)。
5. `bltz/objectives_v2.py::mdn_nll_loss`:若 `batch.get("loss_mask")` 非空:
   `w = boundary_weight(ds) * loss_mask` 后再归一化(现有 `(nll*w).sum()/w.sum()`
   语义不变;prompt 位置同时自动剔除)。无 mask 时行为逐位不变(单测对拍)。
6. `bltz/trainer.py`:新增 `train.init_ckpt`(weights-only 初始化,optimizer/
   schedule/RNG 全新,start_step=0);用于从 CPT 终态 ckpt 起跑,规避
   「weight-only resume 会继承 step 计数器」的 10-19 教训;与 `train.resume`
   互斥(同时给则报错)。
7. `configs/v2_word_sft.yaml`(新)与 `slurm/bltz_word_sft.sbatch`(新):
   - `data.cache_dir=$SCRATCH/data/sft_cache`、`n_patches=4096`、
   `train.verify.enabled=false`、`train.init_ckpt=$SCRATCH/ckpts/word_cpt_s_anneal/last.pt`;
   - 2×V100 fp16(init_scale 1024/growth 500),batch 24/卡(=196,608 units/步),
     grad_ckpt, attn_doc_mask=xformers, ckpt_every 250/keep 3, milestone 1000;
   - LR/WSD:**peak 5e-5**(CPT 平台 peak 2e-4 的 25%——SFT 常规区间;
     **注意家规「weight-only 重启 peak ≤ 上 run 结束 LR」字面上会让 peak≤4e-8,
     显然不适用跨阶段 SFT;本方案按"新阶段"处理,此点列为待拍板项**)、
     warmup 200、decay_frac 0.2、final_frac 0.0(单段跑完,含 wall-out 自续投);
   - 步数=epochs×2015(建议 1 epoch;2 epochs 备选)。
8. 单测 `tests/test_sft_mask.py` + `tests/test_init_ckpt.py`:loss mask 只作用于
   target、全 0 mask 退化行为、init_ckpt 后 step=0/optimizer 全新/权重逐位一致。

## 4. 冒烟与验收

### 4.1 冒烟(发射前,stingy 1 卡)

- 3 步 production 形状:loss 首步值合理(≈CPT 续训 loss 量级)、mask 覆盖率
  报告≈73%、每步 ≈11s、峰值 ≤26G、ckpt/存退/rotate 正常;
- 生成 1 条:prompt `<|user|>Say hi.<|assistant|>`,确认能生成文本并以
  `<|end|>` 停止(stop 序列接入 harness 前先用 `gen_accept` 框架手测)。

### 4.2 Tier 2 验收(docs/43 §5;SFT 后)

| 项 | 工具/口径 | 判据 |
|---|---|---|
| **IFEval** | lm-eval `ifeval`(strict/loose,prompt 走 generate_until;stop=`<|end|>`)| **门槛项**:>0 且 >p2 基线(不设硬阈值,看格式涌现) |
| 通用回归 MMLU | lm-eval `mmlu`(5-shot,acc_char;字节口径注记 docs/43 §4)| 对照 p2 的 8 项 0-shot(均 32.6%,≈随机)看是否上移 |
| GSM8K | lm-eval `gsm8k`(5-shot,flexible/strict)| 趋势信号(0.5B 预期低) |
| 字节任务 instruct | CUTE/CharBench(few-shot prompt,自写 YAML 或独立脚本)| 「格式激发」核心证据 |
| 遗忘/回归 | 旧评估 pkl diag_v2 NLL/snap bpb + `checkpoints/word_cpt_s_accept.log` 口径 | snap bpb 退化 ≤0.02(同 docs/43 §6) |
| 生成目检 | 8 prompt×6 读出(modes-peak 正向维度)| 格式合规、`<|end|>` 停止、无退化 |

- 适配器现状:`scripts/lmeval_bltz.py` 已实现 `loglikelihood`/`generate_until`
  (modes-peak 采样),**SFT 前需加**:`generate_until` 的 stop 序列透传
  (`<|end|>`)与 chat 模板注入(把 harness 的 context 包成
  `<|user|>…<|assistant|>`);~0.5 天。
- **任务可用性(2026-10-26 在集群 venv 实测)**:lm-eval 0.4.13 已内置
  `ifeval`、`mmlu`、`mmlu_pro`、`gsm8k`、`gsm8k_cot`、`minerva_math`、
  `lambada_openai` 及 8 项 MC 任务(共 14,689 个具名 task handle),无需自写
  YAML(仅 CUTE/CharBench 需另接)。
- 预算:IFEval ~1-2h(541 prompts,生成式);MMLU 5-shot ~3-4h(14k×4 选,
  可用 limit 先跑 2k 作趋势);GSM8K ~1-2h;CUTE/CharBench ~0.5 天开发。

### 4.3 预算汇总

| 阶段 | 内容 | 墙钟 |
|---|---|---|
| cache 构建 | 396M units 切词+verify(CPU+1 GPU) | **≈2.5-3.5h**(实测:切词 198 convs/s→全量单进程 40.7min/4-worker 10min;AE verify **45k units/s on V100**→≈2.5h;+写盘) |
| 冒烟 | 3 步 | ~15 min |
| SFT 训练 | 1 epoch(2015 步) | **≈6.5-7h**;2 epochs ≈13h(单段 3 天墙内,无需跨墙) |
| Tier 2 验收 | IFEval+MMLU(limit/全量)+GSM8K+旧分布回归 | 半天~1 天(含开发) |
| **磁盘** | SFT cache ≈4GB + 滚动 ckpt ≈20-25GB | **需先释放 ≥85GB(见 §4.4)** |

### 4.4 磁盘/配额(2026-10-26 实测,**SFT 前必须先清理**)

- 集群个人 scratch **软配额 300GiB**;实测用量 **355.8GiB(已超 55.8GiB)**:
  `bltz` 276GB(ckpts 114G + data 162G)+ `cose` 54G(其它项目,勿动)+
  `qwen35` 27G(训练必需 tokenizer)+ 杂项。硬限 1TiB 尚未触发强制性限制,
  但按项目规约(300GB)必须先清理再开新阶段。
- SFT 需额外 ~30GB(cache≈4GB + 滚动 ckpt≈20-25GB)→ 建议先释放 ≥85GB。
- **已备好裁剪清单**(2026-10-26 dry-run job 692688 验证):`scripts/prune_ckpts.py`
  + `slurm/bltz_prune_ckpts.sbatch`(默认 dry-run,`--apply` 才删;只按显式清单
  删,不 glob)。**可释放 ≈103GB**:已完成 run 的滚动全状态 ckpt(`ckpt_full*`)
  + 已完成 run 的里程碑 + 2026-09 已关闭消融臂目录(p1/p2 的 pre_decay 中 p1 的
  分支点也被 p2 取代,可删)。
- **保留**:p2 `last/best/pre_decay`(旧分布基线+S 档分支点)、CPT stable
  `last/best`、anneal `last/best/pre_decay/s0095000`、AE best、**全部 data
  缓存(活资产)**、SFT 数据。裁剪后 `bltz` ≈173GB,总 scratch ≈254GiB < 300。
- **执行需用户批准**:`sbatch slurm/bltz_prune_ckpts.sbatch --apply`(当前未执行)。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| marker 切词 OOD / AE 重建差 | cache 构建期 marker 探针(§2.2),备选 marker 方案 |
| 相邻符号合并 unit(§2.3) | 构建报告占比;备选轮间分隔词 |
| prompt mask 泄漏(把 user 当 target) | 单测逐 unit 核对 + 训练首步 mask 覆盖率断言(73%±3) |
| 遗忘(纯 SFT 无 replay) | 旧分布 snap bpb 回归;>0.02 回报用户;备选加 5-10% CPT 数据 replay(需用户拍板) |
| 过拟合(smol-smoltalk 全量 1 epoch) | 1 epoch 优先;2 epochs 需看 test split 与 benchmark |
| 生成 `<|end|>` 学不会 | 冒烟阶段显式检查;必要时加权 end marker 的 loss(备选) |
| harness chat 格式错配 | 适配器加模板注入开关;IFEval 前先用 20 条手测 |

## 6. 待用户拍板项

1. **本方案是否通过**(§2 模板/§3 施工/§4 验收口径);
2. **epochs**:1(推荐,6.5h)或 2(13h);
3. **LR peak**:5e-5(推荐)或 1e-4 / 1e-5;若守家规字面(≤上 run 结束 4e-8)
   则等于几乎不训练,不建议;
4. **system 角色**:保留 `<|system|>` marker(推荐)或并入首条 user;
5. **MMLU 5-shot 范围**:全量(~4h)或 limit 2000 趋势版(~0.5h,推荐先跑);
6. 是否允许在 SFT 数据中混 5-10% CPT 数据防遗忘(**默认零 replay,不混**);
7. **是否批准 scratch ckpt 裁剪**(§4.4,释放 ≈103GB,**SFT 的硬前置**;
   只删已完成 run 的滚动/里程碑 ckpt 与 2026-09 已关闭消融臂,保留
   p2/CPT 关键 ckpt 与全部缓存;`--apply` 未执行)。

## 7. 执行序(拍板后)

1. ☐ 用户拍板本方案;
2. ☐ 写 `build_sft_cache.py` + 单测 → tiny/stingy job 建 cache(报告 marker
   flag 率/mask 占比/总 units);
3. ☐ 改 shards/objectives/trainer + `configs/v2_word_sft.yaml` + sbatch + 单测;
4. ☐ 冒烟(stingy 3 步 + 生成停止检查);
5. ☐ 发射 SFT(1 epoch ≈6.5h,2 卡单段);
6. ☐ Tier 2 验收 + 旧分布回归 → 回报用户,决定是否进 M 档/规模档。

## 附:复现命令

```bash
# 数据测量(已完成,job 692634)
sbatch slurm/bltz_sft_probe.sbatch
# 结果:$SCRATCH/data/sft_probe.json + logs/sft_probe_692634.out

# 模板/marker 验证(已完成,job 692652;三变体 × AE verify)
sbatch slurm/bltz_sft_template.sbatch
# 结果:$SCRATCH/data/sft_template_probe.json + logs/sft_template_692652.out

# 构建预算实测(2026-10-26):
#   切词吞吐(CPU,登录节点秒级)= scripts/probe_sft_build_speed.py
#     -> 198 convs/s = 0.156M units/s;全量 40.7min 单进程 / ~10min 4-worker
#   AE verify 吞吐(GPU,stingy 1 卡)= scripts/probe_sft_verify_speed.py
#     -> 45k units/s on V100 => 396M units ≈ 2.5h
sbatch slurm/bltz_sft_verify_speed.sbatch
# 结果:logs/sft_vspeed_692711.out
```

