# 27 — 2B patch 边界探针长 run(2026-09-15,用户拍板)

> **拍板**(用户):① encoder 是字符级能力的视野核心,**冻结不再动**
> (保守砍 inducing8/heads4 保持,不再加码);② FiLM 升默认头
> (词界略优 + 更现代久经考验,赌后期;concat 留作 head_type 选项);
> ③ T1' 时代首波长 run:**当前 encoder + FiLM 头,2B patch 预算**,
> 正式开探模型能力边界;batch 16(显存效率优化后优先)。

## 1. 预算与配置

| 项 | 值 | 备注 |
|---|---|---|
| 数据预算 | **2.0e9 patches** ≈ 12.1B 字节 | 60k run(0.49B)的 4.1×;占缓存 26%,无重复 |
| steps | 244000(= 2e9 / (16×512)) | WSD warmup 1000 + decay_frac 0.4(衰减自 146k) |
| batch | 16 | grouped encoder + FiLM + head_grad_ckpt 后的优先档 |
| 时长 | ~80h(1.15-1.2s/step 估) | 5 天墙内,一把完 |
| 配方 | LR 4e-4 / fp16 / Adam(0.9,0.95)/ spike_skip 10x | 与 60k 一致 |
| 里程碑 | 5000 步一份(weight-only) | ≈ 49 份 × 0.55GB ≈ 27GB scratch,配额内 |

- ckpt:`$SCRATCH/ckpts/boundary_2b/`;final 归档 `~/bltz/ckpt_final/boundary_2b/`;
  日志 `~/bltz/logs/train2b_<jobid>.out`。
- 观察点:里程碑可分段做 diag_poc + 轨迹;收尾按 docs/19 流程
  (曲线 / 诊断对照 60k 与 film@20k / gcos 轨迹)。

## 2. 待回填(跑完后)

- loss 终值与曲线形态(对照 60k 的 2.56);
- 步速(grouped encoder 的真实集群收益);
- 诊断对照(docs/20 基线);FiLM 在后期是否兑现用户之赌(docs/22 §4.2)。

## 3. 拍板回写

- `configs/default.yaml`:`head_type: film` + `loss.head_grad_ckpt: true`
  (FiLM 默认头的 batch16 显存安全配套;concat 运行显式 `--set` 关闭即可)。
- encoder 冻结:旋钮清单(docs/21)encoder 侧全部关闭,不再提案。

## 4. 中途干预记录(2026-09-16,用户诊断+拍板)

- **现象**:2B run 在 36k 后 loss 停于 2.57-2.62(与 60k run 衰减起点同位;
  60k 靠 decay 吃到 2.60→2.556,本 run 原计划 146k 才衰减)——
  **稳定段 LR 4e-4 成为长程约束**(两次 run 一致证明"恒定慢磨、衰减兑现")。
- **干预**(decay-on-demand,WSD 机制的正确用法):scancel 原 job(556282)
  → `decay_frac: 0.4 → 0.84`(decay_start 39040)→ 重提交(job 560530,
  从 ckpt_full@72250 续)。**形状选择:线性立即衰减**——余弦开头太平
  (前 ~20k LR 几乎不降,对"现在就卡"是负作用);开方衰减需新代码,暂缓。
- **生效验证**:resume 首行 LR = 3.42e-04(从 4.00e-04 按 t=0.162 精确下落),
  其后按 4e-4→4e-5 线性衰减至 244k(剩余 ~171k 步)。
- **观察协议**:看 2-4k 步内 loss 斜率是否响应;若低 LR 区(146k+)仍不兑现,
  形状(cosine/sqrt)与数据量再议。scancel 时 ckpt 已在(step 72249),
  损失 ≤50 步。

### 4.2 二次干预:暂停主 run + 10k 退火探底(2026-09-16,用户拍板)

- **现象**:第一次干预后 LR 已按长衰减下行,但 ~140k 步时 loss 仍 ~2.53-2.55。
  **用户按河谷理论拍板:暂停主 run,拿检查点跑 10k 退火探底**——退火终值 =
  当前态的盆地底部估计。
- **执行**:scancel 主 run(560530,ckpt@140749 保留,主 run 可随时续);
  退火 job **565575**(`slurm/bltz_anneal.sbatch`):冻结 ckpt@140750,
  **动态算 resume 步与主 run 在该步的 WSD LR(2.213e-4)**,10k 步线性退火
  至 ~2.2e-5,peak=当前 LR 无跳变(house rule 合规),独立目录
  `ckpts/anneal_140k/`。
- **生效验证**:resume 首行 LR 2.21e-4,900 步内已降至 2.06e-4。
- **判读规则**(探底结果出来照此议):退火终值 ≪ 2.53 → 平台是 LR 锁的,
  主 run 续跑长衰减即可;退火终值 ≈ 2.53 不动 → 盆地底部到了,约束在容量/
  数据/目标,2B 长衰减的边际价值存疑,转议架构/数据方向。
