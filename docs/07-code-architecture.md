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
- **读取模式**(2026-09-29/30):`ShardReader(preload=True)`=全量入 RAM
  (GPFS mmap 页错误曾是 batch 相位 1.6s 主因);`ShardRotator`=块级轮换
  (驻留 K shard+后台预载换下一块,数据序=粗粒度块随机,下一次训练起
  启用,`data.rotate_shards:{resident,every}`)。GPFS 上的 mmap 随机读
  是历史坑,新代码路径默认避开。
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

## 8. 增量补记(2026-09-30,word 臂时代新机制)

**数据/IO 层**:
- `ShardReader(preload=True)`:全量入 RAM(GPFS mmap 页错误曾是 batch 相位
  1.6s 主因;节点内存 603G,preload 需 mem≥96G)。
- `ShardRotator`(shards.py,**块级轮换**,09-30 拍板"服务器积德"):驻留
  K shard+后台线程预载下一块+shard 级随机排列;mem 96G→64G;数据序=
  粗粒度块随机;配置 `data.rotate_shards:{resident,every}`,与 preload
  互斥;换块在 `batch_fn` 里按调用数触发,`_off_ckpt` 与 shards 同步弹出。
  测试 `test_shard_rotate.py`。
- **在线核验回退** `bltz/verify.py`(`VerifyResplitter`,train.verify.*):
  EM+熵(>0.2 bits)双判据→BPE 重切→字符安全 ≤3B 地板;distinct 级 dict
  缓存(occurrence 重复 ~133×,稳态 GPU 成本~0);**`process_batch` 批级
  合并**(每窗一次 GPU 往返→每批一次,慢速事故修复);超 l_max 单位自动
  旗标;字节守恒(`fit_window` 截尾到 S);评测侧同款对齐(推前脚本内)。
  测试 `test_verify_fallback.py`。

**训练层**(trainer.py):
- `BLTZ_PROF=1` 相位计时(env 门控):batch/fwd/bwd/gn/opt/io 分解。
- GradScaler 参数化:`train.fp16_init_scale`(默认 65536,**正式 run 一律
  1024**——锐化密度模型 ≥8192 必溢出)/`fp16_growth_interval`(500)。
- `train.rewarmup`:weight-only 重启后 LR 线性回升(防 fresh-Adam 冲击)。
- `ckpt_pre_decay.pt`:退火起点主动存档(weight-only,分支点不靠里程碑
  对齐);里程碑 `milestone_every`(**预算口径 10000**,滚动全态
  ckpt_keep=3 + best.pt(log 边界节流)兜底;~32GB/段/臂)。
- grad_norm 改 `torch._foreach_norm` 单同步(原 ~200 次 .item()/步)。

**读出层**(mdn.py):
- `mode()`:单全局 MAP(均值漂移,生成冠军读出)。
- `modes()`:**多局部众数枚举**(全分量起点并行 mean-shift+聚簇),配
  **峰顶海拔加权**+核采样的免表词级温度采样(候选权重=该点密度归一化;
  盆地质量加权已废——与海拔/合法性反相关,corr −0.29)。

**评估层**(scripts/):
- `pushforward_bpb.py`(推前字节 CE,免表旁证尺):字节边际
  P(b|ctx,Δ)=Σ_k π_k·mean_m P_dec(b|λ,Δ),纯字节无 EOS(v3);串级联合版
  废弃(并行解码器整串联合=边际乘积沉底);CLI: `[tag] [v2] [ae] [mode]`。
  评测流必须过训练同款核验+回退(铁律 1 评测侧)。
- `snap_decode.py`:`--no-qwen`(word 臂表=纯 cache-top)+ `--cache`(全
  shard 扫频建协议表 top-500k+覆盖曲线)。
- 生成铁律不变(UNSHIFTED 前缀);生成采样:mode(确定性)/modes-peak
  (免表温度)/snap τ(表约束)。
