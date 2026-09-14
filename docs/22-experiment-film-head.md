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

(待补:20k loss 终值与曲线形态;diag_poc 对照;结论与下一步)

### 4.1 OOM 事故与修复(2026-09-14 深夜,含机制解释)

**事故**:film 臂 555577 在 step ~1000 CUDA OOM(26GB 已分配 + 5GB 碎片
保留,调制行申请 452MB 失败)。

**逐位置查询机制**:`mtp_loss` 对每个 patch 位 j、每个相对偏移 Δ(1..48,
未来 3 patch 内有效者)各查一次头 `head(h_j, Δ)`;每步有效查询数 ≈ Σ_j K_j,
随 batch 字节密度浮动(均值 ~7-13 万 @batch16,行间密度方差 ±8-15%)。
前向按 `loss.query_chunk`(200k)分块,但**分块只约束前向峰值;反向的
autograd 图驻留 = 全部查询 × 每查询图大小**,与分块无关(此前的理解有误)。

**为什么 film 会炸**:concat 头每查询驻留 = 4 层 fp16 激活,瘦;FiLM 头
每查询驻留 = RMSNorm 的 fp32 中间态(4B/元)+ 调制器输出 (N,3072) +
调制链多个逐元素中间量,胖 2-3×。叠加密 batch 与碎片,翻过 32GB。
(encoder 画布的 62% padding 浪费是另一码事,见 docs/24。)

**修复**(commit `9129f40`,纯工程、不动训练目标):
1. **头梯度检查点**(`loss.head_grad_ckpt=true`):头前向按 chunk 包
   `torch.utils.checkpoint`——驻留只保留 chunk 输入(h_q、Δ),反向按 chunk
   重算前向。稠密微基准(batch8、全 16 字节 patch、196k 查询):
   **峰值 21GB → 11GB**。
2. **F.rms_norm 融合核 + dtype 对齐**(weight 转 x.dtype):不再显式驻留
   fp32 中间态,且走融合实现(修前的 dtype 不匹配告警让 norm 走慢速回退,
   也是 film 偏慢的来源之一)。
3. **`expandable_segments=True`**(film sbatch 环境变量)压碎片。

**对训练目标的影响:零。** 检查点重算的前向与原前向是同一函数同一输入,
梯度在浮点非结合律误差级内不变;微基准两路径 loss 逐位一致
(5.5566=5.5566)。代价 = 反向多一次头前向重算(步时开销百分之几)。
冒烟过拟合(film+ckpt)bf16/fp16 双 PASS,10 测试全绿。

### 4.2 结果(已填,2026-09-15 验收;对照 = 60k 初版 concat,以及 docs/10 的 20k concat)

**曲线**:film@20k final loss **2.6602**(60k 初版在 20k = 2.65-2.66)——
**打平**;gn ~0.1、skips 4,全程健康。步速 1.19s/step(vs 初版 1.09,
+9%:FiLM 计算+grad-ckpt 重算的代价)。参数量 138.8M(+0.3M)。

**诊断并排(diag_poc,film@19920)**:

| 指标 | 60k concat | 20k concat(docs/10) | **film@20k** | 读法 |
|---|---|---|---|---|
| loss@20k | 2.66 | 2.66 | **2.66** | 平 |
| edelta 均值 CE | 2.732 | 2.807 | 2.804 | 平(同预算) |
| Δ=1 词尾/非词尾 | 1.52/2.17 | 1.62/2.29 | 1.65/2.29 | 平 |
| 词界 space 置信 | 0.76-0.80 | 0.79 | **0.84** | FiLM 唯一的正向差异 |
| ∇_h(gcos_ext) | +0.116/022/090 | — | +0.133/033/107 | 同良性,略更对齐 |
| T3 comb k1/k2/k3 | +0.82/29/-0.09 | — | +0.76/21/-0.05 | 同构 |
| demb 相邻/远端 | +0.36/-0.22@20k | 同 | +0.362/-0.051 | 同水平(60k 的 0.71 是步数差异) |
| 生成 | 真短语@60k | 伪词+重复 | 伪词+重复(同 20k 水平) | 平 |
| 提交跨度 | 5.0 | 3.5 | 4.6-4.8 | 平,词界对齐率 1.00 |

**判定**:**FiLM ≈ concat,同预算全方位打平**,唯一可测正向差异是词界
space 置信略高(0.84 vs 0.79);代价 = +3.4% 头参数、+9% 步时、
需要 grad-ckpt 才能安全跑。**建议:默认头保持 concat**,FiLM 作为
`head_type` 选项保留(基建已验证稳定);若未来在更大预算下重议,
以本档数字为锚。混杂声明(docs/22 §3):encoder 砍法与 FiLM 打包
对照初版——打平意味着两者在 20k 尺度上都**没有造成可测劣化**。

**顺带修复**(验收中发现):diag_poc 的模块分组曾硬编码 `head.mlp`,
FiLM 头无此属性即崩;已改为按 head_type 动态分组(head.body =
in_proj+films+rest),并经历一次"PowerShell 手术毁文件"事故后从
git 恢复重施(教训:+1 条,同文件 edit 与脚本执行禁止并行/粗暴替换)。

## 5. 测试清单更新

`tests/test_film_head.py` 加入套件(共 10 个)。
