# 37 — v2 阶段总结(第二辑,2026-09-29;压缩上下文入口)

> **跨会话恢复从本文 + AGENTS.md 读起**;协议→docs/38,实验史→docs/39,
> 下步计划→docs/40,架构→docs/07(现行版)。本文覆盖 09-22 至 09-29
> 的读出革命、输入侧重构、切分之战与 word-AE 定稿。

## 1. POC 时代三个设计的废除(用户拍板,2026-09-29)

| 废除 | 原设计 | 废除理由(证据) |
|---|---|---|
| **非确定性切词** | 增强 BPE(规则∪BPE,分歧 p=0.5 随机切) | "整词 vs 子词"在每个位置是不可预测分支,GMM 被迫用容量覆盖同义重复表示(pure-BPE 60k:bpb **2.425 vs 2.857**,~15%,已在判据跑兑现) |
| **非线性 GMM 头** | 两层缓冲 MLP(1024×2)隔离骨干与 GMM 参数 | 深度阶梯单调反证(TinyStories 六臂:2 层 46.0% → 1 层 47.1% → **0 层 47.6%**);linear 头 0.03M 参数横扫 NLL/EM/PPL/bpb;FineWeb 双档(60k+2B)复验一致胜,缓冲层=纯负资产 |
| **分量-词对应解码** | argmax-π 分量均值当词,π_k≈P(词) | **GMM 分量是密度基函数而非语义单元**(用户洞察):多分量合作叠峰,模式可不在任何 μ_k 上;"分量=确定字节串"作废,解码只走**密度泛函 snap argmax / GMM mode**(AGENTS 硬性规则;mode 三连夺冠,最新 26.1% vs dec 22.0%) |

**新决策(同日拍板)**:原生训练的切分主轴 = **word 级确定性切词 + BPE
fallback**(Unicode 词/数/符分类、尾随空格并前、数字独立、超长 BPE 兜底,
见 §3);**纯 BPE 作消融对照臂**;GMM 头 = 纯线性;读出 = snap/mode。

## 2. 架构现状摘要(详见 docs/07 现行版)

```
输入: word 臂切词单位 (B,S,L<=32 bytes)
  ├─ 冻结 AE encoder -> λ(48, detach) -> LN+MLP adapter -> 骨干     [默认]
  └─ 可学习输入(Beta,严谨对照待做:同结构,48 瓶颈换 LN)         
骨干: 768x12(113M;0.5B 起 1024x28)/pre-norm RMSNorm/RoPE/SwiGLU
MDN 头: 纯线性 768->K(1+2d)=6208 (K=64,d=48,sigma_floor=0.05)   [mdn_depth=0]
目标: 冻结 typo-AE 的 λ(GMM 密度拟合;分量=密度基)
读出: snap(词表密度吸附)/mode(GMM MAP,均值漂移);词级温度在表分布
训练: WSD 或 CosSched;fp16+GradScaler+gn 熔断+停滞看门狗;DDP 可选
```

变更史要点:autoencoder.py inducing=0→标准自注意力、dec 384;mdn.py
linear 头默认+mode();bytelayer.py AttnPool(pre-norm,589611 事故修复);
trainer.py CosSched/_state_model/is_rank0/ckpt 轮转 N 份/看门狗;train_ae.py
fp16+scaler+守卫+断点续训;snap_decode.py --ppl/--mode(校准+四读出)。

## 3. word 臂切词规格(现行主轴,segment_word)

1. Unicode 三分类:字母=词、数字=数、其余=符号;
2. **数字永远单独成 patch**(数字串压缩性最差、整串预测难);
3. 字母连续段=整词 patch(含 CJK 段);符号连续段=纯净分隔符 patch;
4. **恰好一个空格的 patch 并入前一 patch,除非前一是数字**(尾随空格
   语义:'Stock ' 与[', ',': ', '? ']同构;AE 为此重训——21%→99.6% 教训);
5. 超 32B 的 patch 用 Qwen BPE 兜底切开;
6. 完全确定性,同文恒切。统计:3.03B/patch(word 4.88/symbol 1.18/
   digit 1.0),BPE 兜底率 0.01/doc。

## 4. 悬置信息与资产清单

- **在途**:593264→593265 word-AE 2.5B 长训(docs/36 判据);
- **可续 ckpt**:scale_05b 半程(~46% 单遍,旧输入结构,处置见 docs/40-4);
  v2_learnin_60k_crash1(事故留档);v2_bpe_60k / v2_lin_60k / v2_typo_2b /
  ts_* 六臂 / v2_lin_2b(best FineWeb 模型)均在集群 $SCRATCH/ckpts/;
- **缓存**:集群 data/fineweb10b_v2_bpe(纯 BPE)/ fineweb10b_v2_word
  (word 臂,建)/ tinystories_v2;**混合切分缓存已删**(本地 E:\
  data\fineweb10b_v2 有 56 文件 md5 对账备份 67GB);本地 data/
  fineweb_word_local(188.6M 单位)/ word_parquet / fineweb_edu_blob;
- **评测 pkl**:bpe_eval.pkl / ts_eval.pkl / diag_sample.pkl(Temp);
  word 缓存 pkl 在集群 eval_sample.pkl;
- **失败案例库**:checkpoints/ae_word_failures.txt(5813 例);
- **微调候选**(用户资源,另一项目文件夹):llama3.2-1B / smollm2-1B /
  qwen3.5-2B;
- **HF token**:集群 $SCRATCH/hf_cache/token(600,工作区禁存);
- **集群 QOS**:gpu_v100s 2 卡 ≤3 天/job;SSH 必须 yihe47@。

## 5. 经验教训总表(压缩版;详史 docs/32/39)

1. **OOD 测试禁令**:对比实验必须对齐训练分布或中立交集,否则是尺子
   假象(旧 AE 21% vs 99.6% 的翻案);
2. **密度泛函硬规则**:解码只走 snap/mode;分量均值当词=范畴错误;
3. **bpb 硬通货**:跨切分/跨模型唯一可比尺;NLL/PPL/EM 各有各的 unit 尺;
4. **自由生成 EM 有偏**:质量判据=目检三签名(语法/吸引子/退化);
5. **诊断脚本加载 canary**:漏 load_state_dict 会把随机骨干当模型测
   (两轮结论作废事故);
6. **生成循环 UNSHIFTED 前缀**:backbone_input 是教师强制 shift,误用=
   差一位逐词复读(逐字节恒等复读=签名);
7. **NLL≠词级任务**:密度 NLL 的白捡通道(σ)与词级质量解耦,原生 NLL
   不作质量代理;
8. **长任务心跳/断点**:每分钟级循环必打速率+ETA;3 天级跑必带断点续训;
9. **mp.Pool OOM 挂死**:worker 内存超限被杀→Pool 永久等待(状态仍显示
   RUNNING,只看日志步号);word 模式 workers=4 防;
10. **PMA/MAB 裸 q/kv 是推理部件**:训练态 fp16 反向必溢出(589611 悬崖),
   训练用一律 pre-norm;
11. **余量规约**:LM 训练显存留 6GB/20%(旧 ×2 规约是草图模型遗留);
   batch=16 是 v1 遗产,新跑先摸底;
12. **typo 增强=忠实度不是鲁棒性**:评估 typo 要用"重建 typo 目标"口径;
13. **PS 工程坑**:双引号吃 $()/管道(远端命令拆独立 ssh);Set-Content
   UTF8 加 BOM 毁 sbatch;squeue 格式串用单引号;
14. **校准 T=10 五连稳定**:密度→词级的量化锐度是结构属性;
15. **数据×架构交互**:尾随空格样式需 AE 同步重训,单位风格与 AE 训练
   分布强绑定;AE 换单位分布必重训(37 §1 废除 1 的配套)。
