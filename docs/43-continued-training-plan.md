# 43 — 10B 新数据续训施工方案(2026-10-18 拍板落纸)

> **性质**:0.5B Chinchilla 里程碑(docs/41)之后、已拍板方向「10B 量级新数据
> 续训 → SFT」的施工方案与决策记录。调研依据 = `docs/42-external-survey-202610/`
> (A 算力数据 / B 后训练 / C benchmark)。本文是续训阶段的唯一施工入口;
> 评估协议权威在 docs/38,数据账本在 docs/41 §1。
> 上游拍板链:10-09 定方向(docs/41 §5)→ **10-18 定配方(本文 §0)**。

## 0. 决策记录(2026-10-18,用户拍板,八项)

| # | 决策点 | 拍板 | 备注/风险 |
|---|---|---|---|
| 1 | **续训档位** | **S:+10B units(~4 天)** | 先看效果再决定是否续投 M/L;2 个 72h job 自动续投 |
| 2 | **续训 LR peak** | **2e-4(=父 run 平台 LR,100%)** | 用户口径:「从上一轮稳定段末尾续跑」——WSD 框架下等于数据替换的稳定段延续。⚠️ 与 CPT 文献(rewarm 10-50%)相左,Ibrahim 2024:peak 越高上游遗忘越多;**缓解=旧分布 snap-500k bpb 差分监控**(现有协议覆盖,§6) |
| 3 | **分支点** | `word_05b_p2/ckpt_pre_decay.pt`(step 45429,稳定段末尾,weight-only) | WSD 标准分支点;re-warm 后重新安排完整稳定段+新退火尾 |
| 4 | **主力语料** | **60% DCLM-baseline + 40% FineWeb-Edu** | SmolLM2 消融组合;Edu 必须按 id 剔除已用 sample-10BT 交集(§1.2) |
| 5 | **旧数据回放** | **不回放(零回放)** | 用户理由:新数据 40% 为 FineWeb-Edu,属同族续训,漂移小;文献亦允许(同域零回放可接受) |
| 6 | **退火段混入** | 末 ~10% 步数混:**代码 10% + 数学 5% + Dolmino 高质量 web 补足**(10-18 AE 探针后修订,§3.3/§3.4) | AE 探针:code EM 96.3% 可混;math EM 86.0%(LaTeX 括号/反斜杠 14% unit 损坏)→ 数学占比压到 5% |
| 7 | **SFT 配方** | **先纯通用 SFT(smol-smoltalk 全量 ~460k)**,上 benchmark 验秤看字节能力是否涌现;不行再考虑字节任务特化(字符合成) | 字符合成数据暂缓,不作为首发组分 |
| 8 | **评测基建** | lm-eval-harness 接入服务于全链路验收:「S 档续训+通用 SFT 跑通 = 全链路首个模型」 | 续训 4 天窗口内并行搭建(1-2 天工作量),续训验收前就绪 |

## 1. 数据准备(全部公开非 gated,**我直接在集群下载**,2026-10-18 分工口径:
能自己搞到的不甩给用户;落 scratch;下载走 SLURM job/后台,不占登录节点)

### 1.1 下载清单

| 用途 | 数据集 | HF 路径 | 需要量 | 预计体积 | 落盘 |
|---|---|---|---|---|---|
| 主力 60% | DCLM-baseline | `mlfoundations/dclm-baseline-1.0` | ~7.5B tokens 切片 | ~30GB 文本 | `$SCRATCH/data/cpt_dclm/` |
| 主力 40% | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ~5B tokens 切片 | ~20GB 文本 | `$SCRATCH/data/cpt_fweb_edu/` |
| 退火段 | FineMath(4+) | `HuggingFaceTB/finemath` | ~0.25GB 文本(数学 5% 配比) | ~2GB | `$SCRATCH/data/cpt_anneal/` |
| 退火段 | Stack-Edu | `HuggingFaceTB/stack-edu` | ~0.5GB 文本(代码 10% 配比) | ~1-2GB | 同上 |
| 退火段 | Dolmino 高质量 web | `allenai/dolmino-mix-1124` | dclm/pes2o/wiki/stackexchange 各 ~1.3GB | ~5GB | 同上 |
| SFT(后阶段) | smol-smoltalk | `HuggingFaceTB/smol-smoltalk` | 全量 ~460k 样本 | ~2GB | `$SCRATCH/data/sft/` |

- **格式**:parquet,必须带 `text` 列(缓存构建管线 `build_cache_v2.py` 直接
  按文件分 shard 消费);**FineWeb-Edu 还需保留 `id` 列**(去重用,§1.2)。
- 切片方式不限(按 parquet 文件顺序取够量即可);退火段三类可各自单独目录,
  便于独立成 shard 控制混合比。
- **scratch 预算**:现用 ~139G + 本次新数据文本 ~55G + 新缓存数组 ~60G ≈
  255G,贴近 300G 配额;**下载前先 `du -sh $SCRATCH` 核账**,必要时清理
  (注意:旧 word 缓存 `fineweb10b_v2_word` 是活资产,不回放也不删——SFT
  与对照实验仍用)。

### 1.2 FineWeb-Edu 去重(硬性)

已用数据 = FineWeb `sample/10BT` 的 14 个 parquet(000_00000..013_00000,
源文件应仍在 `$SCRATCH/data/` 原下载处;若不在,从 HF 重下这 14 个文件即可,
每个 ~3GB)。施工侧负责:
1. 从 14 个 parquet 提取全部 `id` 成集合(~15M 行);
2. 构建 FineWeb-Edu 缓存前按 `id` 过滤剔除交集(预期 ~9% 命中);
3. DCLM 与 FineWeb 管线独立、文档 id 体系不同,残余内容级重叠在
   Muennighoff 框架下无害,不做处理(docs/42 A §2.0)。

## 2. 缓存重建(word 化)

- 管线:`scripts/build_cache_v2.py`,mode=word(segment_word + l_max=32,
  Qwen3.5 BPE 兜底),与现行缓存同构;DCLM/Edu/退火各源独立 shard 组,
  便于 ShardRotator 控制配比与退火段切换。
- 参照成本(docs/41 §5):10B units ≈ 46GB 文本 → ~60GB 数组,构建数小时级;
  退火段数据量小,分钟级。
- 构建完成核查:units 总量 ≥10B(不足则补切片)、unit_flag 写出、抽样
  verify 重建 sanity。

## 3. 续训配方(S 档定稿)

### 3.1 拓扑与调度

- **基座**:`ckpt_pre_decay.pt`(word_05b_p2,step 45429),weight-only 加载
  + `train.rewarmup=500`(Adam 防护,家规)。
- **调度**:WSD——rewarmup 500 → **peak 2e-4 平台段** → 末 ~10% 步数退火到 0
  (`final_frac=0.0`)。
- **步数/预算**:全局 batch 48×4096 units ≈ 196.6k units/步(沿用 p2:
  per-GPU batch 24,S=4096,2×V100 DDP);10B units ≈ **~50863 步**,
  ~11 s/步 → 全程 ≈ 93.5h ≈ 4 天(2 个 72h job,wall-out 自动续投配方
  沿用 p2 验证版,docs/41 §6.4)。
- **退火尾**:decay_frac ≈ 0.10(末 ~5100 步 ≈ 1B units),混入配比按
  §3.4 定稿(代码 10% + 数学 5% + Dolmino web 85%)。
- **fp16 纪律不变**:init_scale=1024 + growth_interval=500;V100 禁 bf16。
- **数据序**:ShardRotator 块级轮换(resident=2,every=2000,mem 64G;
  p1 实测口径)。新语料 60:40 通过 shard 数量比例落实。
- **模型/掩码**:d1024/L28/H16,grad_ckpt=true,attn_doc_mask=xformers
  (集群 auto=xf);ae_48_256_word_full 冻结;verify 在线核验回退照旧。

### 3.2 与 p2 的差异点(仅两处)

1. **数据**:全部换新(零回放),60% DCLM + 40% FineWeb-Edu(去重后);
2. **LR 语义**:同数值 2e-4,但含义从「同 run 平台延续」变为「新数据 CPT
   的 re-warm 目标」——风险监控见 §6。

### 3.3 退火段数学/代码混入的前置验证(硬门前置)——**已执行(2026-10-18)**

AE 是 FineWeb 网页文本训的,数学(LaTeX)与代码 latent 分布可能 OOD。
**探针**:`scripts/probe_ae_ood.py`(本机,AE best@105000;各领域文本经
segment_word 重切后逐 unit 评测:teacher-forced CE/bacc + 贪婪解码 EM):

| 集合 | n | 均长(B) | CE | byte-acc | 贪婪 EM |
|---|---|---|---|---|---|
| cache_val(FineWeb 对照) | 800 | 5.12 | 0.0016 | 0.9994 | **0.9988** |
| en_prose(英文散文) | 800 | 4.75 | 0.0041 | 0.9986 | **0.9988** |
| **math(LaTeX)** | 798 | 2.85 | 0.5394 | 0.9372 | **0.8596** |
| **code(Py/JS/C/shell 混合)** | 568 | 3.99 | 0.0867 | 0.9905 | **0.9630** |

**判读**:code 轻度 OOD(失败模式 `=>`/长连字符/docstring 引号);math
显著 OOD 但非塌方——失败高度集中于 LaTeX 承重字符(`{`、`}`、`\command`、
`_{`),14% unit latent 损坏且恰为数学骨架字符。

### 3.4 退火段配比定稿(2026-10-18,用户拍板)

- **代码 10%**(Stack-Edu:Python/JavaScript/Markdown 切片);
- **数学 5%**(FineMath-4plus 切片)——按探针结果压比例:全退火段损坏
  unit ≈0.7%,可接受;
- **其余 ~85% 给 Dolmino 高质量 web**:dclm/pes2o/wiki/stackexchange 切片
  (跳过 dolmino-math——FineMath 已覆盖;跳过 flan——指令格式留给 SFT);
- 退火段总长 = 末 ~10% 步数 ≈ 5100 步 ≈ 1B units;各源独立 shard 组,
  ShardRotator 按比例驻留;
- 若日后做 M/L 档或重训 AE,再议提高数学占比。

### 3.5 训练实施:两阶段单 sbatch(2026-10-18 定稿)

- **`slurm/bltz_word_cpt_s.sbatch`**:一份 sbatch 内两次 trainer 调用实现
  中途换数据——**phase A**(`word_cpt_s_stable`,0→45776 步,decay_frac=0
  纯平台,LR 2e-4,cache=cpt_cache_main,weight-only 自 ckpt_pre_decay 加载
  +rewarmup 500);**phase B**(`word_cpt_s_anneal`,全态自 A 的 ckpt_full
  45776 接续→50863,decay_frac=0.1 退到 0,cache=cpt_cache_anneal)。
  总计 50863 步×196.6k units=10.0B units,退火尾=10.0% 步数。
  wall-out 自动续投沿用 p2 配方(墙前 300s STOP→近墙自投本文件;
  phase 完成判定=各自 ckpt_dir 的 last.pt)。预计 ~100h(2 段 72h 墙内)。
- **冒烟**:`slurm/bltz_word_cpt_smoke.sbatch`(stingy 分区,双缓存各 3 步
  production 形状:A=weight-only 分支加载,B=全态接续+anneal 缓存),
  通过后才允许发射正式 run。
- 退火段数据通过缓存组装即完成配比(docs/43 §2 symlink 组装),训练侧
  零改动。

## 4. 验收计划(续训段)

按 docs/42 C 报告 Tier 1 电池,分两层:

1. **内部协议(现有,必跑)**:docs/38 全套五段——snap-500k bpb(旧分布,
   兼作遗忘监控)、held-out NLL、byteCE、mode EM、top5、NMI;生成目检
   (modes-peak/τ 采样,正向维度口径,禁循环否决项)。
2. **lm-eval-harness(新基建,并行搭建)**:自定义 LM 子类
   (loglikelihood/loglikelihood_rolling/generate_until),0.5B 档标准电池 =
   OLMES 八件套(acc_norm,与 bpb 同构)+ MMLU 5-shot(acc_char 辅口径);
   字节主场 CUTE/CharBench 留待 SFT 后(生成式)。基建 1-2 天,续训 4 天
   窗口内完成,验收前就绪。对拍校验:先跑 GPT-2 小模型核 acc_norm 口径。

**判据口径**:与 p2 终态对表(bpb 2.539/NLL -40.69/EM 26.3%);文献预期
+10B 全新数据有清晰可见的下探;遗忘红线=旧分布 snap bpb 退化 >0.02 时
回报用户。

## 5. SFT 阶段(续训验收后启动)

- **配方(已拍板)**:纯通用 SFT = smol-smoltalk 全量 ~460k(Apache 2.0,
  0.5B 档唯一公开证据配方);不掺字符合成(缓议,视 benchmark 结果再议)。
- **模板**:EvaByte 先例 Llama-3 chat 式,`<|user|>…<|assistant|>…<EOS>`,
  loss 只算 assistant 段+EOS,prompt mask;标记走 BPE 兜底进 unit 流。
- **序列预算**:4096 units ≈ 4-5k BPE tokens 当量,smol-smoltalk 截断率低。
- **验收**:Tier 2 电池——IFEval(程序化免费)门槛 + 通用回归(MMLU/GSM8K)
  + 字节任务 instruct 版(CUTE/CharBench)看「字节能力是否经格式激发涌现」
  (用户验秤口径);MT-Bench/AlpacaEval 二选一视裁判预算。
- 详细施工方案在续训验收后单独成文(本文不展开)。

## 6. 风险与监控

| 风险 | 缓解 |
|---|---|
| LR 2e-4 re-warm 偏高(文献 10-50%),旧分布遗忘 | 旧分布 snap-500k bpb 逐里程碑差分;退化 >0.02 回报用户 |
| AE 对数学/代码 OOD | §3.3 硬门前置验证,不通过则退火混入降级 |
| scratch 配额顶穿(300G) | 下载前核账;ckpt 滚动 keep=3 + 里程碑 weight-only |
| 数据卫生 | FineWeb-Edu 强制 id 去重(§1.2);DCLM 残余重叠无害不处理 |
| wall-out | 沿用 p2 自动续投配方(墙前 300s STOP→近墙自投),零人工已验证 |

## 7. 执行序(当前 todo)

1. ☑ **我在集群直接下载 §1.1 数据**(job 657764 = 主力+退火+SFT;
   657809 = dolmino 后继;均 tiny 分区,不占登录节点);
2. ☑ AE 数学/代码重建探针(本机,10-18 完成,§3.3);
3. ☑ FineWeb-Edu id 去重(fw10bt id 集合已提取 14.59M;下载管线内联过滤,
   实测保留率 99.94% ≈ 理论 0.067% 随机抽样重叠);
4. ☐ 缓存重建(job 657832,afterok 链;main 60:40 + anneal 85:10:5 symlink
   组装;逐源建缓存后删原始 parquet 保配额);
5. ☐ 双缓存冒烟(blz_word_cpt_smoke.sbatch)→ **正式 run 发射
   (bltz_word_cpt_s.sbatch,§3.5 两阶段单 sbatch)**;
6. ☐ lm-eval-harness 接入(与 4-5 并行,另立项);
7. ☐ 续训验收(Tier 1)→ 对表 → 决定是否续投 M 档;
8. ☐ SFT 施工方案成文 → 数据到位 → SFT → Tier 2 验收。
