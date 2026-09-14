# 24 — T3 施工单:打包浪费量化(定 T1 前)

> 来源:`docs/23-batch-packing-survey.md` §5 T3。用户 2026-09-14 拍板:**先 T3 量浪费,再定 T1**。
> 纪律:**本批只做测量**——不改 `bltz/`、`configs/`、训练流程;只新增诊断脚本 + 本档回填结果。
> 交付后按 §4 决策规则自动落到「上 T1」或「收回/先分桶」。

## 0. 本批只回答三个问题

- **Q1** 真实 patch 长度分布是什么?16 字节画布里有多少是 `PAD_ID`?(不能只信 `configs/baseline.yaml:4` 的 ~5.2 字节)
- **Q2** 一个训练步的耗时/显存里,encoder / backbone / head 各占多少?
- **Q3** 若做 T1(解开 patch 内 padding),理论上限是多少?值不值得?

## 1. T3.1 — patch 长度与 padding 审计(本机,分钟级)

新增 `scripts/diag_packing.py`(纯诊断、无副作用):

- 采样:N=256 行(可 `--set`),`ShardReader.make_batch(idx, S, augment=True, p_split=0.1, p_merge=0.1, rng=rng)`;
  再跑一遍 `augment=False` 作对照。
- 长度来源:`ends.diff()`(现有 batch 张量;如需亦可从 `pad_mask` 反推),断言 `Σ len == flat_len`。
- 输出:
  - patch 字节数 `mean / p50 / p90 / p99 / max`;
  - 1..16 的直方图;
  - **padding ratio = 1 − Σlen / (S·L)**;
  - 每行总字节数分布(检验行间方差);
  - 可选:按首字节粗分 latin / CJK 两组的平均 patch 长度。
- 判据:数字自洽;augment 前后对比说明 `p_split/p_merge` 对密度的影响。
- 产物:结果写回本档 §5;可选直方图存 `viz/`(gitignore)。

## 2. T3.2 — 模块耗时/显存审计(本机 4060,分钟级)

复用并小幅扩展 `scripts/bench_parts.py`(已有 encoder/backbone/head/full 计时)+ `scripts/prof_step.py`:

- 增加**分量显存**:`torch.cuda.max_memory_allocated()` 在 encoder / backbone / head 各自区间前后取差;
- 增加**有效槽位**统计:`n_valid = int((~pad_mask).sum())` vs `B·S·L = train.batch·n_patches·l_max`;
- 输出 encoder 冗余比 `(B·S·L)/n_valid`;
- 判据:三件套(ms / peak MB / 有效槽位)齐全;与 `docs/06`、`docs/07` 的历史数字不矛盾。
- **注意**:本机 4060 bf16 ≠ 集群 V100 fp16 —— 只用于**比例**,不外推绝对值。

## 3. T3.3 — 长度分桶/动态批收益预演(可选,opt-in)

- 不触默认路径;新增 batch_fn 变体(按每行真实字节数分桶采样),或给 `diag_packing.py` 加 `--bucket`;
- 测:批内 padding 方差下降幅度、step time 变化;
- 判据:只报告,不切默认。

## 4. 决策规则(测完照此判)

| 条件 | 动作 |
|---|---|
| padding ratio ≥ 50% **且** encoder 时间或显存占比 ≥ 30% | **上 T1**(出 T1 施工单) |
| encoder 占比 < 15%,或 padding ratio < 30% | 收回 T1,维持现状 |
| 中间区 | 先做 T3.3(分桶),再复评 |

## 5. 结果(已填,2026-09-14 晚,主会话施工;**Q2 经用户指正重测**)

- **Q1 patch 长度分布 / padding ratio**(diag_packing.py,dryrun 缓存 256 行):
  patch 字节数 mean 6.05 / p50 5 / p90 10 / p99 14 / max 16,直方图峰值在
  3-4 字节(词+空格形态);**(S,16) 画布 padding ratio 62.2%**(augment 前后
  62.2%→62.6%,增强不显著改变密度);行总字节 3098±8%(2411..3557)——
  行间密度方差温和,不是显存炸点主因(film OOM 主因是头的逐查询图胖,
  已修:head_grad_ckpt,docs/22 §4)。
- **Q2 encoder / backbone / head**(bench_parts.py,4060 bf16,**取比例**):
  第一版 batch 8 撞本机 8GB 卡的 WDDM 共享内存抖动区(step 峰值 17GB>8GB,
  绝对值全失真——docs/06 同款坑);**重测 batch 2/4(用户指正,干净区间)**:
  fwd ms = enc **209/237** / bb 34/42 / head 16/31 → **encoder 占前向时间
  76-81%**(字节轴画布 = 骨干序列长 ×16 且仅 36.5% 有效);峰值 MiB(batch 2)
  enc 1662 → bb 2504 → head 3085 → step 5344。
- **Q3 T1 理论上限(修正)**:encoder 字节轴计算 ÷2.64 → 折合全步约
  **-20~25%**(fwd 占比口径),比抖动态旧估(4-5%)大一个量级;显存侧
  canvas 激活 ÷2.64,峰值随真实字节数缩放。
- **结论**:按 §4 规则(padding≥50% 且 encoder 占比≥30%)应上 T1;
  **但用户 2026-09-14 晚拍板:T1 记低优先级,T3.3 分桶为下一步规划**
  (规则让位于拍板;新测量如实记录,若日后优先级重议,数字在此)。
- **教训**:本机审计**只用小 batch 取比例**,不复刻集群 batch(8GB 卡的
  WDDM 抖动区让一切绝对值失真)。

## 6. 交付物与验收

- 新脚本 `scripts/diag_packing.py`(Q1,只读);`scripts/bench_parts.py`
  扩展分量显存 + 有效槽位(Q2);
- 本档 §5 已填数;
- 本次未改 `bltz/` 数据路径与 `configs/`(T3 纪律);bltz/ 内的改动仅限上一轮
  的 FiLM 头基建与 head_grad_ckpt(独立事项,docs/22)。

## 6. 交付物与验收

- 新脚本 `scripts/diag_packing.py`,可直接 `python -u scripts/diag_packing.py` 运行;
- 本档 §5 填数;
- **未改动** `bltz/`、`configs/`、`tests/`;
- 不提交(等用户/主会话)。

## 7. 资源与风险

- 本机需可读的 `data/` 缓存。若本机无缓存:退化为用 `bltz.data.build_batch` 的小样本合成测量,
  并在结果里**明确标注**该退化(不与真实分布混淆);或用 `--set data.cache_dir=<可用路径>`。
- 本机 8GB 显存,取 `train.batch` 小值(如 2–4)即可,比例不受影响。

## 8. 参考

- `docs/23-batch-packing-survey.md` §1(现状代码定位)、§5(方案 T1/T2/T3);
- `scripts/bench_parts.py`、`scripts/prof_step.py`、`scripts/diag_poc.py`;
- `bltz/data.py:29-63`(`tensorize_units`/`collate_sequences`)、`bltz/shards.py:178-205`(`make_batch`)。
