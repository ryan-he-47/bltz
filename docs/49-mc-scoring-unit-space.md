# 49. 多选评测的嵌入空间打分方案(unit-space MC scoring)

> 状态:**用户已拍板并实现**(2026-10-27)。本文取代 pushforward 系列打分
> (lm-eval 适配器 `pushforward_joint` / `byte_marginal`、验收电池 §3 与
> `scripts/pushforward_bpb.py`)。实现改动见 §6/§8,验证闸门结果见 docs/39 #46;
> 全量电池重跑(cluster)需另行批准。
>
> 用户 2026-10-27 口径(本文依据):"pushforward 是当初为了尝试和字节语言模型
> 比较性能才实现的打分手段,不是衡量模型性能的主要手段……彻底移除;自回归骨干
> 是以 unit 为单位、在嵌入空间给出概率分布的,自然应该在嵌入空间打分,直接拿
> 候选嵌入在 GMM 分布上打分;关于候选的 unit 长度与分数的不均衡问题,标准 LM
> 怎么处理不同长度的 token 序列,我们就怎么处理 unit 序列。"
>
> **拍板记录(2026-10-27)**:① 主指标 = `acc_unit`(按 unit 数归一);
> ② 同时跑 `acc_unit_pmi`;③ `pushforward_bpb.py` **归档**(`archive/`);
> ④ 历史文档按"保留数字 + 废止注记"处理;⑤ **立即执行**(改代码 + 本机
> 验证闸门 G1–G6;集群全量另行批准)。

---

## 0. 范围

| 类别 | 处理 |
|---|---|
| `scripts/lmeval_bltz.py` 的 `pushforward_joint` / `byte_marginal` 两种 score_mode、`_chain_pf_joint`、`_chain_byte_logmarg`、`PF_MC` | **删除** |
| `scripts/pushforward_bpb.py`(免表字节 bpb 旁证尺) | **退役删除**(git 历史保留) |
| `scripts/run_accept.py` 验收 §3(pushforward byteCE) | **删除并重编号** |
| lm-eval 适配器打分口径 | 改为**unit 空间 GMM 对数似然**单一实现(§2–§5) |
| 新增 `scripts/mc_eval.py`(MC 指标驱动) | 新建(§6.2) |
| snap bpb / 内部 held-out NLL / EM / NMI / 生成读出 | **不动**(不依赖 pushforward) |
| 历史文档中的 pushforward 数字 | 保留 + 加“2026-10-27 废止”注记(§8) |

---

## 1. 记号与模型事实

- 文本经确定性切词器 `T(·)`(segment_word + VerifyResplitter,训练同管线)得到
  **unit 序列**;unit = 字节串,长度 1..l_max(32B)。`T` 是纯函数、字节守恒。
- 冻结 AE 编码器 `E: bytes → R^48`:候选 unit 的嵌入 `λ(u) = E(u)`,不进梯度。
- 骨干:`h_i = Backbone(Adapter(λ(u_1..u_{i-1}), BOS))`,h_i 用于预测第 i+1 个 unit。
- GMM 头:对每个位置给出
  `π(h_i) ∈ Δ^{K−1}`, `μ_k(h_i) ∈ R^48`, `σ_k(h_i) ∈ R^48`, K=64,对角协方差;
  `p_θ(z | h_i) = Σ_{k=1}^{K} π_k N(z; μ_k, diag(σ_k²))`。
- 训练目标(教师强制):`L = Σ_i −log p_θ(λ(u_{i+1}) | h_i)`。**这就是模型的
  原生似然**;评测打分必须使用同一个量,不需要任何解码器/蒙特卡洛。

---

## 2. 打分定义(核心)

### 2.1 单 unit 条件对数似然

对上下文 chain 之后的下一个候选 unit `v`:

```
log p(v | ctx) := log Σ_k π_k(h) · N( λ(v); μ_k(h), diag σ_k(h)² )        (1)
               = − MDNHead.nll(h, λ(v))            # 与训练 loss 同一个函数
```

其中 `h` 是 chain 中 `v` 的前一个位置给出的骨干状态;`λ(v)` 由冻结 AE 编码。

### 2.2 候选序列分数(teacher forcing)

上下文 `c` 的 units `u_1..u_m = T(c)`;候选续写 `y` 的 units `v_1..v_n = T(ŷ)`
(`ŷ` = 候选 `choice` 文本;harness 会前置分隔符空格,按 §3.2 判界,分隔符
不进入候选项)。把两条 unit 序列拼成
`w = u_1..u_m, v_1..v_n`,**一次前向**得到每位置 NLL:

```
nll_i = − log p_θ( λ(w_i) | w_<i )                      (2)
S_raw(c, y) = Σ_{j=1..n} nll_{m+j}                      (3)    # 只计候选项
```

`S_raw` 是**候选 unit 序列的总对数似然**(nats);它是标准 LM 中
`Σ log p(token)` 的 unit 版。

### 2.3 为什么这是自然口径

- 骨干的分布对象就是**下一个 unit 的嵌入**;`p_θ(λ|h)` 是模型自己的似然,
  与训练 loss 完全同构,没有额外的近似/解码器/重要性采样。
- 候选嵌入由冻结 AE 直接给出(确定性),**不需要**从 GMM 采样、不需要解码器
  推前、没有 EOS/整串概率/蒙特卡洛方差。
- 与标准 LM 的对应是逐项一一对应的(附录 A 表)。

---

## 3. 切词与边界协议(对齐标准 LM)

### 3.1 标准 LM 的协议(事实,lm-eval 0.4.13 `lm_eval/api/model.py:_encode_pair`)

对 causal LM,lm-eval 的 `loglikelihood(ctx, cont)` 是:

1. 把 context **末尾的空白**移到 continuation 开头(保持词边界分词);
2. `whole_enc = tok(ctx + cont)`、`ctx_enc = tok(ctx)`;
3. continuation 的词元 = `whole_enc[len(ctx_enc):]`(整段编码后按 context 的
   token 数切),score = `Σ_{t ∈ cont} log p(t | 前文)`。

也就是说:**causal LM 是整段联合 tokenize、再切出候选词元**,候选的首个词元
通常自带分隔符空格(BPE 的 `Ġ`,如 `" choice"`)。候选的“项数”= 落在候选里的
词元数;分隔符算进首个词元,是分词器习惯,不是多一项。

### 3.2 bltz 的对应协议(joint unitize + 字节边界归属)

word 臂的 unit 约定与 BPE 相反:**空格尾随前一个 unit**(`b'word '`),
所以 lm-eval 的“把末尾空白移到 continuation”对 unit 流的等价结果天然成立:
分隔符空格并进 context 的最后一个 unit。对应协议是:

```
byte_boundary = len(ctx_minus_trailing_spaces)   # 先把 ctx 末尾空白移给 cont
full_units    = T(ctx' + cont')                  # segment_word + VerifyResplitter
ctx_units     = [u for u in full_units if u 的首字节 <  byte_boundary]
cont_units    = [u for u in full_units if u 的首字节 >= byte_boundary]
score         = Σ_{u ∈ cont_units} log p(λ(u) | 前文)      # 只计候选项
```

- 跨缝的那个 unit(典型:context 末尾的 `":"` + 分隔符空格 → `b': '`)归
  **context**,不计分;第一个计分 unit = 候选的第一个完整词。这与 BPE 把
  `" choice"` 算作 continuation 首个词元是同一件事——两种约定都只把
  **候选的词序列**计入项数(分隔符本身不是一项),只是空格归属的 tokenizer
  习惯不同。
- 例子(`arc_easy`):`ctx = "Question: ...\nAnswer:"`,`cont = " a liquid"`:
  `T(ctx+cont) = [..., b'Answer', b': ', b'a ', b'liquid']`,
  `cont_units = [b'a ', b'liquid']` → n=2;与 HF 的 `[" a"," liquid"]` 同样 2 项。
- **等价性校验(实现时断言)**:`n_cont_units` 必须等于“把候选文本单独切词
  得到的 unit 数”(对 MC 的常规格式成立);若不等(候选不以分隔符开头、
  跨缝吞掉首词等异常),记 warning 并退化为“候选单独切词 + 直接接在
  context units 之后”的兜底协议(此时 context 末 unit 不重切)。
- context 超长(S=4096 units)时左截断 context、保住候选(与现适配器一致)。
- `T` 确定性、字节守恒;VerifyResplitter 的 flag 统计照旧记录。
- **joint 切词本身保留**(这与标准 causal LM “整段 tokenize 再切”一致);
  被废止的只是 pushforward 打分,不是切词管线。

### 3.3 与旧实现的唯一区别

旧适配器用同一套 joint 切词,但打分走 pushforward(整串概率)。新方案**只换
打分函数**(§2),切词/边界协议与旧实现一致(§3.2),生成/训练管线完全不动。

### 3.4 退化与断言

| 情形 | 处理 |
|---|---|
| `n_cont_units = 0`(候选被跨缝 unit 全吞/全空白) | 走 §3.2 兜底协议(候选单独切);仍为 0 → 该候选记 `−inf`,记 warning |
| chain 超过 S=4096 units | 左截断 context(候选永远保留) |
| 候选 unit 内字节 > l_max | `T` 已按 BPE/字符 floor 切好,不会出现 |
| 平局(多候选同分) | 取**最小选项下标**;报告平局率(标准 `np.argmax` 行为) |
| AE 编码到异常 unit | 仍然得到 λ;记录 verify/异常计数,不特殊处理 |

---

## 4. 长度不均衡:标准 LM 怎么处理,我们就怎么处理

### 4.1 标准 LM 的三种处理(事实)

1. **`acc`**:`argmax S_raw`,原始总分。长候选天然吃亏(项数多),这是
   lm-eval 默认主指标,也是所有公开排行榜 `acc` 的定义。
2. **`acc_norm`**:`argmax S_raw / len(choice)`,长度归一。**lm-eval 0.4.13 的
   实现细节**:`completion_len = float(len(choice))` 是**字符数**(不是 token 数,
   `lm_eval/api/task.py:1490-1570` 与 `:1715`),是对“长度偏置”的工程折中。
3. **`acc_mutual_info`**(可选):`argmax [log p(choice|ctx) − log p(choice)]`,
   用无条件分扣除候选先验(长度/词频/位置),标准的小模型去偏做法。

### 4.2 我们的主口径:按项数(unit 数)归一

对候选 `y` 的 `n` 个 unit 项:

```
S_unit(c, y) = S_raw(c, y) / n        (4)
acc_unit     = argmax_y S_unit(c, y)  (5)
```

理由(与 token LM 严格同构):

- 式 (3) 是 **n 项之和**;项数只由 unit 数决定,与字节数无关。序列变长 = 项数
  变多,与标准 LM 的 token 数完全相同。
- 一个 8 字节的长 unit 是**一项**,正如一个 8 字节的长 token(“internationalization”)
  在 token LM 里也是**一项**——标准 LM 不会因为 token 长就给它加项,也不会在
  `acc_norm` 里除以 token 字节数;我们同理不按字节归一。
- unit 的**项内**差异(长词密度低/短词密度高)是模型自身的先验,不是长度项;
  如同 token LM 里低频 token 概率低是模型先验。若对候选嵌入再按字节除,等于
  把“词长先验”重复惩罚,且 latent nats/byte 不是 bpb,**不能**当字节似然用。

### 4.3 报告的全部口径(同一次前向,零额外成本)

| 名称 | 定义 | 角色 |
|---|---|---|
| `acc_unit` | `argmax S_raw / n_units` | **主指标**(内部 p2/CPT 对比、Tier 1/Tier 2 验收) |
| `acc_raw` | `argmax S_raw` | 标准 `acc` 同义,跨论文可比 |
| `acc_char` | `argmax S_raw / n_chars` | 与 lm-eval `acc_norm` 公开数字对齐(注意其实现是字符数) |
| `acc_unit_pmi` | `argmax [(S_raw − S_uncond)/n_units]` | 可选去偏(标准 `acc_mutual_info` 的 unit 版);需额外一次空 context 前向 |
| `S_raw/n_bytes` | 诊断量 | 只叫“latent nats/byte”,**不叫 bpb**,不作主指标 |

- `S_uncond` 定义:context 为空(仅 BOS),对候选 units 逐项求 log p 再求和;
  `S_raw − S_uncond` = 上下文带来的互信息。
- 每任务报告:`n`、四个 acc、对应 stderr、平局率、`n_units` 分布、
  长度相关诊断(见 §5)。
- **主结论只引用 `acc_unit`**;`acc_raw`/`acc_char` 用于与文献数字对齐,
  并明确标注口径。

### 4.4 与文献比较的口径规则

- 对标 Pythia/Qwen 等公开数字:对方是 lm-eval 的 `acc` / `acc_norm`
  (raw / 字符归一)。我们同时给 `acc_raw`(同定义)与 `acc_char`(同公式),
  不把 `acc_unit` 硬套到公开表上;跨模型结论用“区间 + 口径标注”。
- 内部 p2 vs CPT 对比:唯一主指标 `acc_unit`。

---

## 5. 指标定义与产物格式

```json
{
  "task": "arc_easy", "n": 2376,
  "scores": {"acc_unit": 0.0, "acc_raw": 0.0, "acc_char": 0.0, "acc_unit_pmi": 0.0},
  "stderr": {"acc_unit": 0.0, "...": 0.0},
  "diag": {
    "tie_rate": 0.0,
    "n_units_mean": 0.0, "n_units_p10": 0, "n_units_p90": 0,
    "pearson_Sraw_vs_n_units": 0.0,
    "pearson_Sunit_vs_n_units": 0.0,
    "pearson_Sraw_vs_n_chars": 0.0,
    "argmax_eq_shortest": 0.0, "argmax_eq_longest": 0.0
  },
  "items": [{"gold": 0, "pred_unit": 0, "pred_raw": 0, "S_raw": [..], "S_unit": [..]}]
}
```

判读规则(验收时强制):
- `acc_unit` 是唯一主指标;`acc_raw`/`acc_char` 只作对照。
- 若 `|pearson_Sunit_vs_n_units| > 0.3`,该任务结果标“长度残留偏置”,需先查
  原因(切词/归一化 bug)再引用。
- 平局率 > 5% 的任务要在结论中注明。

---

## 6. 实现方案

### 6.1 适配器简化(`scripts/lmeval_bltz.py`)

删除:

- `score_mode` 参数、`--score-mode` CLI 与三个分支;
- `_chain_byte_logmarg`、`_chain_pf_joint`、`PF_MC = 8`;
- 模块 docstring 中 pushforward/acc_norm 注记,改为 unit 空间口径说明。

保留/改造:

- `_segment`(segment_word + VerifyResplitter)保留;新增
  `_split_units(ctx, cont) -> (ctx_units, cont_units)`(§3.2 的 joint 切词 +
  字节边界归属,含兜底协议);
- `_chain_nll` 保留(与训练同函数;现在它是**唯一**打分函数);
- `loglikelihood(requests)`:对每条请求算 `S_raw`(式 3),返回 `(S_raw, is_greedy)`;
  `is_greedy` 仍用 `head.mode` + AE 解码判定,不变;
- `loglikelihood_rolling`:不变(滑窗 NLL;与打分无关);
- `generate_until`:不变(modes-peak;不依赖 pushforward)。

`loglikelihood` 伪代码:

```python
for (ctx, cont) in requests:
    ctx_units, cont_units = self._split_units(ctx, cont)  # §3.2 joint + 边界归属
    units = ctx_units + cont_units
    nll   = self._chain_nll(units)            # (S,) fp32, teacher forcing
    ll    = -float(nll[len(ctx_units):].sum())  # 式(3): 只计候选项
    greedy = self._greedy_hits(units, len(ctx_units))
    out.append((ll, greedy))
```

### 6.2 新驱动 `scripts/mc_eval.py`

职责:用 lm-eval 的任务定义/数据集/0-shot prompt 构造请求(��证与文献同
prompt),但**指标层由我们实现**(式 4–5 与 §5),输出 JSON + markdown 表。

```python
tasks = get_task_dict(TASKS)                      # lm_eval 0.4.13
for name, task in tasks.items():
    for doc in task.eval_docs:                     # 与 harness 同一 split
        ctx = task.doc_to_text(doc)                # 0-shot(与 harness 同)
        choices = task.doc_to_choice(doc)
        gold = task.doc_to_target(doc)             # 索引/字符串 → 统一成索引
        reqs = [Req(ctx, " " + c) for c in choices]   # harness 同格式
        lls  = lm.loglikelihood(reqs)              # 每条 = S_raw
        # n_units / n_chars 由 _split_units 确定性离线复算(与打分同一函数)
        # 计算 acc_unit / acc_raw / acc_char / (可选)acc_unit_pmi
```

- 可选 `--pmi`:对同一候选再算 `S_uncond`(ctx="")。
- 输出:`checkpoints/mc_eval_<tag>.json` + `.md`;带心跳/ETA(AGENTS 纪律)。
- 与 harness 的关系:prompt/数据/split 与 `simple_evaluate` 完全一致;
  `acc_raw` 必须与 harness 原生 `acc` 逐位一致(作为集成测试 G6)。
  保留 harness 仅作“原始 acc 交叉验证”,指标层以本驱动为准。

### 6.3 批量与性能

- 同一题的 K 个候选可**批成一次前向**(B=K,causal + padding,逐请求结果相等);
  context 重复但批量摊薄。
- 纯前向,无 MC:预计 V100 上 8 任务全量 ≈ 1–1.5 h/ckpt(旧 pushforward 电池
  为 2.4 h/ckpt);两 ckpt 双卡并行 ≈ 1.5 h。
- 本机 RTX 4060 用 200 题子集验证闸门 ≈ 10–15 min。

### 6.4 测试(`tests/test_mc_scoring.py`)

- 切词分开性:`ctx=..., choices` 的 `n_units` 与 `T(choice)` 一致;
- 分隔符不参与:`n_units` 不含 delimiter;
- 归一化算术:`S_unit == S_raw/n`,平局→最小下标;
- 与训练 NLL 同函数:同一 unit 链的 `-nll` 与直接 `head.nll` 逐位一致;
- 长度诊断:合成等长度候选下 `acc_unit == acc_raw`。

---

## 7. 验证闸门(本机,先于任何集群重跑)

**状态:6/6 PASS(2026-10-27,本机 RTX 4060,738s;日志
`checkpoints/word_cpt_s_mc_validate.log`;结果详 docs/39 #46)。**

| 闸门 | 内容 | 判据 | 结果 |
|---|---|---|---|
| G1 | 打分函数复现训练口径 | `eval_sample.pkl` 的 `distinct_seq_units[:32]`(diag_v2 口径,跳 BOS)平均 NLL 与 docs/39 #43 的 **−40.94** 差 < 0.05 | **−40.942,Δ=0.002** |
| G2 | 确定性 | 两次运行结果逐位一致(无随机量) | PASS |
| G3 | 边界/归一单测 | §6.4 全过 | PASS |
| G4 | 常识 sanity 对 | 10 条“显然常识对”在 `acc_raw` 与 `acc_unit` 下均 ≥ 8/10 | **8/10,8/10** |
| G5 | 长度偏置(2026-10-27 修订,理由见 #46) | ARC-Easy 300 题:`acc_unit` 高于「永远选最短」基线 ≥4pp,且 `argmax==shortest` ≤ 25%+2σ;(题内 corr 降为诊断量) | **31.0% vs 22.0%(+9pp);shortest 24.0%** |
| G6 | harness 集成 | 同环境 200 题:本驱动 `acc_raw` 与 `simple_evaluate` 原生 `acc` 逐位一致 | **0.315000 == 0.315000** |

- **闸门修订记录**:G5 初版判据 `|题内 corr(S_unit,n_units)|<0.3` 无实证依据,
  且把“候选越长→后段越易预测→平均每项似然越高”的模型性质误判为长度偏置
  (实测题内 r=0.467,而旧 raw 口径跨题 r=0.727);改为与「最短基线」比较 +
  argmax=最短率上限,corr 仅作诊断。G1 初版误用 `val_units`(2048 units,未跳
  BOS)→ −43.8,已改回 diag_v2 口径。G4 初版候选串多加一个前导空格造出双空格
  unit(训练分布外)→ 6/10,已修。
- 参考(旧口径 pilot,n=60):joint 切词下 `unit_nll` raw 30.0%、`per-unit`
  26.7%(随机 25%)、`per-byte` 33.3%——**不能假设换口径就一定出现显著信号**;
  新口径 pilot(ARC-Easy 300 题)`acc_unit` 31.0%(见 #46),全量 8 项仍需
  集群重测才能回答“CPT 是否有下游增益”。
- 闸门失败一律先修实现,不允许带着失败闸门跑全量。

---

## 8. 迁移与废止清单

### 8.1 代码/脚本

| 文件 | 动作 |
|---|---|
| `scripts/lmeval_bltz.py` | 删 pushforward 两模式/PF_MC/相关 docstring;新口径(§6.1) |
| `scripts/pushforward_bpb.py` | 删除(退役) |
| `scripts/run_accept.py` | 删 §3;重编号(1 diag / 2 snap / 3 semantic / 4 gen);默认 `--sections 1,2,3,4` |
| `slurm/bltz_lmeval.sbatch` | 去掉 `--score-mode pushforward_joint` |
| `slurm/bltz_lmeval_setup.sbatch` | 去掉 `--score-mode unit_nll` |
| `scripts/lmeval_compare.py` | 适配新指标 JSON(或由 `mc_eval.py` 取代) |
| `scripts/lmeval_bltz_sanity*.py`、`scripts/commonsense_probe.py`、`scripts/mc_score_diag.py` | 这些是诊断/一次性脚本:改造或删除(建议删除,git 历史保留) |
| `scripts/mc_eval.py` | 新增(§6.2) |
| `tests/test_mc_scoring.py` | 新增(§6.4) |

### 8.2 文档

- `docs/07` §代码架构:删除 `pushforward_bpb.py` 条目;lm-eval 适配器条目改为
  unit-space 口径。
- `docs/38` §4:删除“推前字节 bpb”行;新增“MC 打分(unit-space)”协议小节;
  §7 工具清单同步。
- `docs/39`:#32(pushforward 设计)、#43(验收表 pushforward byteCE 行、lm-eval
  电池行)、#45:保留数字 + **加“2026-10-27 废止/重测待定”注记**。
- `docs/43` §4 口径注记:主口径从 `pushforward_joint+acc` 改为
  `unit-space + acc_unit`;§4.1 表的 byteCE 行标注废止。
- `docs/44` 提及 pushforward 主口径处:加注记。
- `docs/40`/`docs/41` 的历史引用:加注记或保持(仅状态性文字)。
- `AGENTS.md` 当前状态:从验收描述中移除 pushforward byteCE,记录本次废止决定;
  新增“MC 主口径 = unit-space `acc_unit`”。
- 本文(`docs/49`)为现行方案入口。

**历史注记策略(建议)**:历史实验日志不删数字(可追溯),只加一行废止注记;
当前协议/状态文档(38/43/AGENTS)删除该指标为“现行口径”。

---

## 9. 风险(拍板项已决,2026-10-27)

1. **嵌入空间密度的可比性**:GMM 密度是 48 维潜空间密度,不是字符串概率;
   因此它**不能**跨模型/跨 AE 比较绝对数值,也不能换算 bpb——这是本次口径的
   自觉边界。跨模型只比任务准确率(`acc_raw`/`acc_char`)。
2. **unit 字节长度先验**:长/低频 unit 密度更低,这是模型先验(与 token LM 的
   低频 token 低概率同类),不做字节归一。若全量结果仍无信号,再考虑
   PMI/去偏(§4.3)而非回到字节口径。
3. **数字切分**:每个数字字符单独成 unit(训练约定),数字答案会比 token LM
   多出若干项;为保持训练/评测同分布,**不**在评测侧改切词。
4. **已拍板(2026-10-27)**:主指标 `acc_unit`;同时跑 `acc_unit_pmi`;
   `pushforward_bpb.py` 归档 `archive/`;历史文档"保留数字 + 废止注记";
   立即执行(代码 + 本机闸门),集群全量另行批准。

---

## 10. 执行序(状态)

1. ☑ 拍板 §9;本文定稿(2026-10-27)。
2. ☑ 改代码:删 pushforward、适配器新口径、新增 `mc_eval.py` /
   `validate_mc_scoring.py` / `tests/test_mc_scoring.py`、sbatch 去 flag、
   `pushforward_bpb.py` 归档、`run_accept.py` 删 §3 并重编号 4 段。
3. ☑/☐ 本机验证闸门 G1–G6:结果与修正记录见 docs/39 #46
   (首次运行暴露 G1 口径错误——应取 `distinct_seq_units[:32]` 且跳过
   BOS 位置;G4 双空格 unit 构造错误;G5 应测**题内**长度偏置——已修正后重跑)。
4. ☐ (需批准)集群双卡重跑 8 项电池(p2 + CPT,`slurm/bltz_mc_eval.sbatch`);
   与 Pythia/Qwen 公开区间的对比表同步更新。
5. ☑ 更新文档(§8.2)+ AGENTS.md。
6. ☐ 用新口径重新回答“CPT 是否带来下游增益 / 是否具备 SFT 前提”。

---

## 附录 A:标准 token LM ↔ bltz unit LM 对照

| 概念 | 标准 token LM(lm-eval) | bltz word-unit LM |
|---|---|---|
| 序列单位 | subword token id | 字节串 unit(segment_word+verify) |
| 条件分布 | softmax over vocab | GMM density over R^48(λ 空间) |
| 单 term 对数似然 | `log softmax[t_i]` | `log Σ_k π_k N(λ(u_i); μ_k, σ_k²)` |
| 序列分数 | `Σ_i log p(t_i)` | `Σ_i log p(λ(u_i))` |
| 切词边界 | 整段 tokenize 后按 context 词元数切出 continuation;context 末尾空白移入 continuation | 整段 unitize 后按字节边界归属;分隔符空白并进 context 末 unit(**同一语义**) |
| `acc` | `argmax` raw 总分 | `acc_raw`(同定义) |
| 长度归一 | `acc_norm = argmax S/字符数`(实现细节) | `acc_unit = argmax S/n_units`(主);`acc_char` 供对表 |
| 去偏 | `acc_mutual_info` | `acc_unit_pmi`(可选) |
| 跨模型尺度 | per-token 概率有限可比 | 只比任务准确率;不作 bpb 换算 |
| 字节 bpb | — | pushforward(本次退役,曾被用作与字节 LM 的旁证尺) |

## 附录 B:公式速查

```
(1) p_θ(z | h)        = Σ_k π_k(h) · N(z; μ_k(h), diag σ_k(h)²)
(2) nll_i             = − log p_θ( λ(w_i) | w_<i )        # 训练同函数
(3) S_raw(c, y)       = Σ_{j=1..n} nll_{m+j}              # 候选项求和
(4) S_unit(c, y)      = S_raw / n
(5) acc_unit          = argmax_y S_unit                   # 主指标
(6) acc_char          = argmax_y S_raw / len_chars(y)     # 对表用
(7) acc_unit_pmi      = argmax_y (S_raw − S_uncond)/n     # 可选
    S_uncond          = 同一候选在空 context(仅 BOS)下的 Σ log p
```
