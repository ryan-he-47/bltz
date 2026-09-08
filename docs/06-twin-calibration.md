# 06 — twin 规模本地标定报告(2026-09-08 深夜)

> 问题:twin 配置(138.5M,S=512)在 8GB 4060 上装不装得下、跑多快?
> 方法:OOM 逐级探测 + torch.profiler 内核级诊断。**所有数字为实测。**
> 产物脚本:`scripts/bench_twin.py` / `bench_parts.py` / `prof_backbone.py` / `prof_step.py`。

## 1. 显存标定(S=512,138.5M)

| batch | 结果 |
|---|---|
| 32 | **OOM**(峰值约 11GB:激活 ~9.5GB + AdamW 状态 ~1.6GB) |
| 16 | 能分配但**分配器抖动**(WDDM 下 ~7GB 临界,步速崩到 >30s) |
| **8** | **稳妥(~4GB),已写入 default.yaml** |

escape hatch:`model.grad_ckpt: true` 已接线(骨干逐块 checkpoint),
需要更大 batch 时可用,默认关。

## 2. 速度诊断(完整因果链)

初始实测 ~15s/步(batch 8)——病态。profiler 逐层排查:

1. **根因一(已修复):RoPE 的交错 cat。** 旧 `apply_rope` 用 GPT-J 交错配对
   `stack([o1,o2],-1)`,stride-2 非合并访存,单次 104ms,**占骨干 CUDA 时间
   67.7%**(2.50s/3.70s)。修复为 GPT-NeoX 半劈连续 cat + bf16 直算:
   骨干前向 **3686ms → 1117ms(3.3×)**。约定数学等价,从头训练自洽即可。
2. **排除限频**:计算期间 2670MHz / 99% util / 27.8W(210MHz 读数是空闲态)。
3. **修复后再 profile 整步**:10.86s CUDA,**纯 gemm-bound**(矩阵乘 7.1s),
   无第二病理点。细分:编码器大矩阵(65536×512×2048)跑 4-5 TFLOPS(健康);
   骨干/头的瘦矩阵(4096×768、71729×1536)只有 1-1.5 TFLOPS——
   **这是 4060 笔记本对瘦 gemm 的真实速度,不是 bug。**

## 3. 标定结论

- **稳态步速 ≈ 11-15s**(batch 8,数据/优化器开销计入)。
- **20k 步 ≈ 61-83 小时 ≈ 2.5-3.5 天**(本机连续)。
- 参数量 138.5M(超 124M 目标 12%:SwiGLU 三矩阵 FFN 使骨干为 113M 而非
  预估的 85M;不影响 twin 对照,基线骨干同配方)。

## 4. 给用户的选项(预算决策,待拍板)

| 选项 | 时长 | 代价 |
|---|---|---|
| A. 本机 20k 步全量 | ~2.5-3.5 天 | 风扇连转数日 |
| B. 本机砍步数(8-10k) | ~1-1.5 天 | 预算缩水(可能不够看出趋势) |
| C. S=256(20k 步) | ~30-40h | 上下文减半(~1.3KB);基线同步匹配;§12 已授权序列长标定 |
| **D. twin 上集群 V100(推荐)** | ~1 天以内(估) | 32GB 可开 batch 32;fp16+GradScaler(MoB 既有配方);本机解放做诊断 |

我的建议:**D**——twin 双臂(ByteField + token 基线)都上 burgundy,
本机只留诊断与小规模迭代;V100 无 bf16 需切 fp16。**fp16+GradScaler 分支
已实现并本机验证**(2026-09-08:trainer `train.bf16=false` 即走 fp16+scaler,
过拟合冒烟 5.55→0.042 收敛,spike guard 正常;上集群时只需
`--set train.bf16=false`)。

## 5. 顺带修复与新增(本 commit)

- `scripts/build_cache.py` 重写为两阶段并行版(hf_hub_download + 每 parquet
  一 shard,multiprocessing,断点按文件续跑);单线程 75 docs/s → 全量从
  ~36h 降到 ~4h(8 workers 估)。`tests/test_cache_builder.py` 全过。
- `configs/default.yaml`:batch 8(标定)+ `model.grad_ckpt` 选项。
- `scripts/prof_backbone.py` / `prof_step.py`:profiler 诊断脚本(可复用)。
