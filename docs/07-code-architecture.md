# 07 — 代码架构(现行版,2026-09-29 重写)

> 角色:**当前代码结构的唯一权威参考**。v1 设计思想见 docs/01(历史),
> v2 架构设计见 docs/31,变更史见 docs/37 §2,评估工具面见 docs/38 §7。
> 本文末尾附 v1 架构历史注记。

## 1. 模块地图(bltz/)

| 模块 | 职责 | 现行要点 |
|---|---|---|
| `config.py` | YAML + `--set a.b=value` 覆盖 | 浮点覆盖须带小数点(`1.0e-8`) |
| `segment.py` | v1 语义切分器 | **历史**(v1 臂专用,新工作不用) |
| **`segment_v2.py`** | v2 三种切分 | `segment_word`(现行主轴:词/数/符分类+尾随空格并前+数字独立+BPE 兜底,确定性)/ `segment_v2(+pure_bpe)`(纯 BPE 消融臂)/ 增合 BPE(已废,p_disagree 随机分支) |
| `shards.py` | 缓存读写 | ShardWriter/Reader;bytes/unit_len/unit_flag npy;稀疏 offset 采样;make_batch(augment=False——v2 预烘焙) |
| `typo.py` | typo 簇增强 | 五操作 40/20/15/15/10、two_edit_p、长度闸门、频率节流 p_eff、force_op |
| `models/autoencoder.py` | **ByteStringAE**:字节串↔λ | Encoder:byte+pos 拼接 → ISAB(inducing=8)或**标准自注意力(inducing=0,现行长训)** → PMA 池化 → out_proj→d_emb(48)。Decoder:FiLM 条件 (λ,Δ)→257(256 字节+EOS 行);Δ≤l_max 查字节,Δ=l_max+1 查 EOS。decode() 贪心 EOS 截断。现行长训 dec_hidden=384 |
| `models/mdn.py` | **MDN/GMM 头** | **纯线性 d_in→K(1+2d)**(`mdn_depth=0` 默认;缓冲层/SwiGLU 已废,depth 1/2 留档);对角协方差,σ=elu(o)+1+σ_floor(0.05);nll(fp32 logsumexp)/sample(τ 只调 π,取 μ_k)/`mode`(GMM MAP,均值漂移定点迭代——**正统读出**) |
| `models/bytelayer.py` | 可学习输入(Beta) | ByteLayerEncoder:标准自注意力版;AttnPool(**pre-norm 种子注意力**,589611 事故修复;旧 PMA 裸 q/kv 训练必溢出) |
| `models/backbone.py` | 骨干 | pre-norm RMSNorm/RoPE(半劈)/SwiGLU/SDPA;grad_ckpt 可选 |
| `models/model_v2.py` | **BltzLMv2** | 冻结 AE 出 λ(唯一目标)+ `input_mode: frozen`(LN+MLP adapter,默认)/learnable(Beta);`adapter_linear` 消融开关;BOS 参数;encode_units **8192 行分块**(kernel launch 事故修复);`backbone_input`=教师强制 shift(cat(bos, λ[:-1]))——**生成循环禁用,见 §5** |
| `models/head.py` + `models.py` | v1 FiLM 头/BltzLM | **历史**(v1) |
| `objectives_v2.py` | mdn_nll_loss | teacher forcing,mean NLL |
| `trainer.py` | 共享基建 | `WSD` / `CosSched`(warmup→cosine peak..floor);spike 守卫(10×median 或 2000);**停滞看门狗**(500 连跳保存退出);ckpt 轮转 N 份(ckpt_keep);`_state_model` 剥 DDP/包装层(ckpt 格式与单卡一致);is_rank0 守卫;优雅中断(SIGTERM/STOP 文件);里程碑 weight-only |

## 2. 数据结构

- **v2 缓存**:shard-XXXXX/{bytes.npy(u8), unit_len.npy(u8), unit_flag.npy(doc 首单位=1), meta.json(切分器指纹)}. 单位=字节串 ≤l_max(32),单位间无 padding,S=512/4096 切序列。
- **batch 字典**:byte_ids (B,S,L)、pad_mask (B,S,L) True=pad。
- **ckpt**:{"model": raw BltzLMv2/ByteStringAE state_dict(无包装前缀), "cfg": 全配置, "step", [opt/scaler/med_hist/...]}。AE best.pt 另含 val_loss。
- **评测 pkl**(`dump_eval_pkl.py`):freq_units(131k)/val_units(2k)/probe_pool/distinct_seq_units(1000×S)/n_seq/total_units。

## 3. 训练入口

| 入口 | 用途 |
|---|---|
| `scripts/train_bltz_v2.py` | **v2 骨干**(唯一现役);torchrun DDP 可选(WORLD_SIZE>1;每 rank 异种子 7919×r;found_inf 经 allreduce 天然共识);fp16+GradScaler |
| `scripts/train_ae.py` | AE 训练;pool/stream 双模式;typo 在线增强(val 干净);`train.sched: wsd\|cosine`;fp16+scaler+守卫;**自动断点续训**(last.pt 含 opt/scaler) |
| `scripts/build_cache_v2.py` | 缓存构建;`seg.mode: enhanced\|pure_bpe\|word`;only_files/max_docs 断点;**workers≤4**(mp.Pool OOM 挂死教训) |
| `scripts/train_bltz.py` 等 | v1(历史) |

## 4. 配置(configs/)

`v2.yaml`(骨干:mdn_depth 0/d_emb 48/sigma_floor 0.05/n_comp 64;可被
`--set model.input_mode=learnable` 等覆盖)、`ae.yaml`(48-256 基座,
inducing 0/dec 384 经 --set)、`cache_v2.yaml`、`default.yaml`(v1)。
续训 house rule:weight-only 分叉 peak ≤ 上一 run 结束 LR。

## 5. 生成循环铁律

`backbone_input()` 是**教师强制 shift**(cat(bos, λ_0..λ_{S-2})). 自回归
生成必须用 UNSHIFTED 前缀:cat(bos, 全部已知单位的 λ)取末行 h,或等价
的 `cur+[dummy]` 技巧。误用=差一位,输出逐词复读(签名:TheThe/andand)。
反馈只喂**流形点**(mode/snap 向量):裸 μ_k 喂骨干即乱码(mode/embed 与
mode/reencode 对流形点逐字节恒等)。

## 6. 数值稳定要点

fp16 autocast + GradScaler + unscale 后 gn 熔断 + clip(1.0)+ 停滞看门狗。
**任何进注意力的模块必须 pre-norm**(PMA/MAB 裸 q/kv 是推理部件:训练态
残差流无界增长→fp16 反向 65504 悬崖,589611;AttnPool/TransformerEncoder
norm_first 均已合规)。梯度检查点(model.grad_ckpt)不改数学。

## 7. 附:v1 架构历史注记(2026-09-08 ~ 09-16,已废)

v1 = BltzLM:v1 语义切分 → Set Transformer(诱导点 ISAB,长度分组重打包
T1')→ 113M 骨干 → FiLM 条件头 (h,Δ)→256 逐字节 MTP(k_max、词界停止
D11)。已验证事实:FiLM≈concat 同预算;H1(条件独立并行解码)判失败
(cat/dog/dot 幽灵词)成为 v2 转向起点;encoder 冻结时代结论与 gcos
诊断见 docs/20/28。v1 代码保留(v1 消融可复现),但**新工作一律 v2**。
