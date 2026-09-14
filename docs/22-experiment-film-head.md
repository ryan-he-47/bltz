# 22 — 实验:FiLM 条件头 A/B(2026-09-14 晚)

> **拍板**(用户):① encoder 保守效率砍 inducing 16→8、enc_heads 8→4(入
> `configs/default.yaml`,后续所有 run 生效);② 条件头试 FiLM 条件化
> (替代输入端拼接),做 A/B 对照。本档 = 设计/基建/协议/结果位。

## 1. 动机与适配性判断(因地制宜)

现头:`concat([h, delta_emb(Δ)])` 进 4 层 MLP——Δ 作为**内容**从输入端喂入,
靠网络自己把条件信号搬运到深层。本场景中 Δ 的语义是"**以何种方式读 h**"
(读多远),不是内容;FiLM 让 Δ 在多层做**乘性调制**(γ⊙x+β),条件路径更短、
更直接。头是全模型 FLOPs 最重的模块(~60%/patch),条件化质量直接决定
MTP 与生成质量。simple_point_cloud 有同族尝试,模式在模板内已验证。

## 2. 实现与基建(bltz/models/head.py::FiLMByteHead)

- **结构**:delta_emb(48×64,沿用)→ in_proj(768→1536)→ 2×FiLMBlock →
  1×plain Linear → out(1536→256);总隐藏层数 4,与 concat 头对齐。
  FiLM 注入前两层("解码器骨干前几层",用户指定)。参数量 ~9.06M
  (concat 8.76M,+3.4%)。
- **FiLMBlock**:RMSNorm → Linear → γ⊙(·)+β → SiLU;**γ=1+γ_raw,
  β=β_raw,conditioner(64→2×hidden)零初始化**——初始化时每个块严格等价
  普通块、对所有 Δ 输出一致(identity start,无 FiLM 引入的训练震荡)。
  此性质由 `tests/test_film_head.py` 断言(身份起步 / 条件梯度非零 /
  BltzLM(film) 前向 / fp16 有限性)。
- **数值稳定(fp16)**:RMSNorm 前置使被调制激活始终单位尺度;冒烟验证
  bf16/fp16 全程收敛 5.55→0.00(scripts/smoke_overfit.py 双路径 PASS)。
- **开关**:`model.head_type: concat|film`(默认 concat;对照臂不受影响)。

## 3. 对照协议(2026-09-14 晚修订,用户拍板:省算力,只跑 film 单臂)

| 臂 | encoder | head | 其它 |
|---|---|---|---|
| **film**(slurm/bltz_ab_film.sbatch) | inducing8/heads4(新) | FiLM | 与 60k 同配方(20k,~14.5h) |
| ~~concat 对照~~ | 不跑 | — | **拿 60k 初版(inducing16/heads8 + concat)当对照** |

- **锚点**:① 60k 曲线在 20k 处 loss 2.65(初版 encoder + concat 头);
  ② docs/20 的 60k 诊断基准(diag_poc 并排对照)。
- **如实声明的混杂**:单臂下,encoder 砍法(inducing16→8、heads8→4)与
  FiLM 头两处变化**打包**与初版对比,无法归因到单一变化;接受此局限
  (用户拍板,省一臂算力)。若结果异常需归因,再补 concat 臂。
- 判读:不预设门槛(D10);loss 曲线 + 诊断并排,如实报告。

## 4. 结果位(待 A/B 跑完回填)

(待补:双臂 20k loss 终值与曲线形态;diag_poc 对照;结论与下一步)

## 5. 测试清单更新

`tests/test_film_head.py` 加入套件(共 10 个)。
