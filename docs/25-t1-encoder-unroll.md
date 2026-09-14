# 25 — T1' 落地:encoder 长度分组重打包(2026-09-14 深夜)

> **拍板**(用户):修正测量(docs/24 §5 重测)显示 encoder 占前向 76-81%,
> T1 从"低优先级"重审为立即开工;实施采用 T1'(长度分组重打包),取代调研
> 中的 varlen 方案(docs/23 §5 T1)与 T3.3 分桶(被 T1' 吸收,不再做)。
> **灵感与实现致谢:Kimi K3 @ Moonshot AI**——T1' 的提出("注意力只在 patch
> 内 ⇒ 按真实长度分组 + 贴身画布,逐元等价,无需 varlen 核")与施工。

## 1. 机制

encoder 的注意力**从来只在 patch 内部**(跨 patch 上下文归骨干)。因此
(S, L=16) 固定画布上 ~62% 的 padding 槽位是纯浪费。T1':
`byte_ids/pad_mask` → 逐 patch 真实长度 `lens` → 收集全部真实槽位
(长度分组稳定 argsort)→ 每组用贴身 `(N_l, l)` 画布跑同一套
ISAB/PMA(无 padding,不需要任何 mask)→ 散射回 patch 位。
零新依赖(纯 torch 索引)、V100 安全、对外接口不变。

## 2. 数值等价

`tests/test_enc_grouped.py`:同权重双路径对拍,随机变长批 max diff ~1e-6
(浮点归约顺序噪声);全满长批、确定性、零长 patch 退化回退 `_forward_canvas`
均覆盖。**断言成立的根本**:padding 槽位在原路径中被 key_padding_mask
逐算子排除,移除它们不改任何实算。

开关:`model.enc_grouped`(默认 true,configs/default.yaml)。

## 3. 实测(batch 4,4060,同机对拍)

| 路径 | encoder fwd | 全步 fwd+bwd | 步峰值 |
|---|---|---|---|
| canvas | 237 ms | 3635 ms | 9287 MiB |
| **grouped** | **135 ms(-43%)** | **1802 ms(-50%)** | 7832 MiB(-16%) |

(绝对值受 8GB 卡约束,只看对拍比例;集群 batch 16 的真实占比以后续
train.log 步速为准。)

## 4. 对胶片记录的影响

- 在跑的 film 臂(job 555728)用的是其启动时拉取的旧代码(canvas 路径),
  结果与 60k 基线同口径可比,不受影响;**下一次集群 run 起自动走 grouped**。
- 冒烟过拟合(grouped)PASS;`test_model` 回归 PASS;测试总数 11 全绿。
- 吸收事项:T3.3 分桶不做(padding 消失);docs/23 §5 的 xformers/varlen
  T1 方案不采用(T1' 同等收益、零依赖、精确等价)。
