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
