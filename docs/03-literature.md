# 03 — 外部文献与代码调研(2026-09-08)

> librarian 调研结果的归档版。全部事实经直接抓取/克隆验证;无法验证者标注 **UNVERIFIED**。
> BLT 官方 repo 检视于 commit `9774ed4f`(main HEAD,2026-09-08)。

## 1. BLT 官方代码与权重(facebookresearch/blt)

- **License: CC-BY-NC-4.0(代码)**,权重为 FAIR non-commercial research license。
- **熵模型训练代码完整包含**:`bytelatent/configs/entropy_model.yaml` = 260 词表
  (256 字节+4 特殊符)字节 LM,dim 768 / 14 层 / 12 头 / max_seqlen 8192 /
  滑窗 512 / local_block_causal,`train_entropy_model: true`,`train.py` L261-264 有分支;
  离线熵预计算在 `bytelatent/preprocess/preprocess_entropies.py`、`parallel_entropies.py`。
  → **不碰 gated 权重也能自训熵模型**,配置即论文配方(发布权重 99.5M BF16,
  与 config 一致;论文 §4.2 文字写 hidden 512,与发布 config dim 768 有出入,
  我们的消融臂按 ~50M 规格自训,见 design §12)。
- **Patcher 代码同时含两种准则**(`bytelatent/data/patcher.py`):
  绝对阈值(默认 θ=1.335442 nats,自然对数)与近似单调约束
  (`patch_start_mask_from_entropy_with_monotonicity`:H_i − H_{i−1} > t 即开新 patch);
  `PatcherArgs` 默认 `monotonicity: False`;论文大 run 用单调约束+换行重置(§4.4,
  防"entropy drift")。
- **官方推理需要熵模型全程在环**:`generate_blt.py` L191-192 要求
  `realtime_patching=True`,`hf.py` L178-180 在生成时加载熵模型上设备。
  → 本项目"模型自身熵停止"替代的正是这个部件。
- **权重 gated(manual 审批),实测匿名下载 401**(huggingface.co 与 hf-mirror.com 均然,
  镜像无法绕过);ModelScope 未发现镜像。社区存在未授权再分发(itazap/blt-1b-hf 等,
  合法性存疑,**不使用**)。
- 官方数据脚本原生支持 `fineweb_edu_10bt`(`setup/download_prepare_hf_data.py`)。

## 2. 社区复现(按参考价值排序)

| Repo | 完整性 | License |
|---|---|---|
| [sssssaud/blt-llm](https://github.com/sssssaud/blt-llm)(权重:[blt-llm-tinystories-55m](https://huggingface.co/sssssaud/blt-llm-tinystories-55m)) | **全管线含熵模型训练**;55M @ TinyStories → 0.71 BPB(θ=1.09,~4.4 B/patch) | MIT |
| [ianbarber/ttblt](https://github.com/ianbarber/ttblt) | TorchTune 版;自训 14M 熵模型;**诚实负面结果**(363k 指令样本学不出局部编/解码器)+ 记录了一个 batch 依赖阈值的 patching bug | Apache-2.0 |
| [bowang-lab/dnaBLT](https://github.com/bowang-lab/dnaBLT) | 官方代码的 DNA 适配 fork(全量保留) | BSD-3 |
| [sagarsrc/exp-byte-latent-transformer](https://github.com/sagarsrc/exp-byte-latent-transformer) | 仅 patching 研究(n-gram 模型):全局阈值 vs 单调 vs 二阶导 | — |

无高星"从零完整复现";无任何 repo 做了"熵分割 + 多字节条件解码"的组合。

## 3. Medusa-2 配方与 MTP 不稳定性文献(H3 的对照组)

**Medusa-2 精确配方**(Cai et al., ICML 2024,[arXiv:2401.10774](https://arxiv.org/html/2401.10774v2);
代码 [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa),Apache-2.0):
联合损失 `L = L_LM + λ0·L_Medusa`,逐头 λ_k = 0.8^k;**头部 warmup**(原话:训练初期
头损失大、梯度大会扭曲骨干);自蒸馏 = KL 到"adapter-off"的原模型(LoRA 省显存技巧);
typical acceptance 阈值 `min(ε, δ·exp(−H))`(熵感知,非拒绝采样)。

**联合 MTP 训练不稳定性的后续证据(2024-2026)**:
- **AdaMTP([arXiv:2608.00434](https://arxiv.org/html/2608.00434),2026,最近邻)**:
  固定视野 MTP 跨高熵语义边界注入冲突梯度,头数增加准确率单调恶化
  (GSM8K: n=2 11.60 → n=6 9.68,低于纯 NTP 11.30);解法 = 熵导向可变深度 +
  掩码 MTP 损失 + 推理期"相邻 token 熵增量超阈即停"。**与本项目 D4/D5 方向一致,
  论文写作必须正面引用并区隔(token 级 vs 字节级;多头/掩码 vs 单条件头/几何衰减)。**
- **OCC([arXiv:2605.28184](https://arxiv.org/html/2605.28184v1),2026)**:联合 MTP 效应
  分解为一阶梯度相关项+恒负二阶扰动项;veRL/slime 框架默认对 MTP 做梯度 detach。
- **Gloeckle et al.([arXiv:2404.19737](https://arxiv.org/abs/2404.19737),Meta FAIR)**:
  从零训练的 token 级并行 MTP 在 7-13B 稳定有效(n=4 最佳;HumanEval +12%,MBPP +17%);
  逐头串行 fwd/bwd 把 logit 显存 O(nV+d) 降到 O(V+d)。**与 AdaMTP 互为边界证据:
  从零预训练 vs 后训练联合微调,不稳定性表现不同。**
- **DeepSeek-V3 MTP([arXiv:2412.19437](https://arxiv.org/html/2412.19437v1) eq.21-25)**:
  顺序式 MTP 模块(RMSNorm 拼接+投影+transformer block),嵌入/输出头共享;D=1。
- Gumiho([2503.10135](https://arxiv.org/pdf/2503.10135)):**早期 draft token 比远期重要**
  → 支持 patch 间几何衰减(D4)。Jakiro([2502.06282](https://arxiv.org/html/2502.06282)):
  同一表示采出的候选彼此相关 → MoE 解耦头(我们用地更深的单头+Δ 条件应对)。

## 4. 字节/字符级追平 token 级的证据链

- **Grapheme-LLaMA**(Bunzeck et al.,[arXiv:2410.01487](https://arxiv.org/abs/2410.01487),
  COLING 2025;[BabyLM 2024 论文](https://aclanthology.org/2024.conll-babylm.5/)):
  ~360 词表 15M vs BabyLlama 58M/16k BPE:BLiMP 71.7% vs 73.1%,词汇判断 99% vs 69%。
  同作者的 lexdec 字符模型系列([bbunzeck/lexdec-medium-char](https://huggingface.co/bbunzeck/lexdec-medium-char),
  用户指示)说明该生态活跃。
- **MambaByte**([arXiv:2401.13660](https://arxiv.org/abs/2401.13660),ICLR 2024):
  token-free SSM,固定参数+算力下追平甚至超过子词 transformer;子词起草+字节验证的
  投机解码 2.6×。
- **EvaByte**(HKU+SambaNova 2025,[blog](https://hkunlp.github.io/blog/2025/evabyte/) /
  [code](https://github.com/OpenEvaByte/evabyte) / [weights](https://huggingface.co/EvaByte/EvaByte)):
  6.5B,1.5T 字节,自称 5× 数据效率、2× 解码速度、含 multibyte 预测头。
  **注意:无可索引的 arXiv 论文(UNVERIFIED),blog 为一手来源。**
- **MrT5**([arXiv:2410.20771](https://arxiv.org/abs/2410.20771),ICLR 2025):编码器第 3 层
  学习式删除门,字节序列压缩至 75% 而精度≈ByT5。
- **H-Net**([arXiv:2507.07955](https://arxiv.org/abs/2507.07955)):端到端学习动态切分
  (无外部 patcher),单阶段字节级即胜算力匹配的 BPE transformer,多阶段匹配 2× 大的
  token 模型。
- 辅助:SpaceByte([2404.14408](https://arxiv.org/abs/2404.14408),词边界分割的关键证据);
  Perceiver IO 字节级 masked LM 追平 BERT([2107.14795](https://arxiv.org/abs/2107.14795));
  2026 对照研究([2604.27263](https://arxiv.org/html/2604.27263v2)):把 token 边界先验喂给
  字节模型即可复现子词优势的大半——**边界放置是关键杠杆**。

## 5. FineWeb-Edu(HuggingFaceFW/fineweb-edu)

- 配置:`default` / 按 dump / 嵌套采样 **`sample-10BT` ⊂ 100BT ⊂ 350BT**(GPT-2 tokens 计);
  10BT ≈ 9.67M 文档 ≈ 9.95B tokens(train 9.85B + val 0.10B,
  [tokenized 镜像](https://huggingface.co/datasets/minhnguyent546/fineweb-edu-10BT-for-gpt2))。
- **ODC-By 1.0 + CommonCrawl ToU;不 gated**;parquet,streaming 可下。
- BLT 官方数据脚本支持 `fineweb_edu_10bt`(§1)。

## 6. 三个细分机制的先行工作

**(a) 神经场/查询条件解码用于语言建模:直接先例=无(空白地带)。** 最近邻:
Perceiver IO 的 query 解码(序列输出的 query 含位置编码,做过字节级 MLM);
Perceiver AR([2202.07765](https://arxiv.org/abs/2202.07765),causal cross-attn 到 latent,
字节级 PG-19 SOTA);Set Transformer PMA seed queries;LH-NeF
([2606.08204](https://arxiv.org/html/2606.08204),FiLM 条件化相对坐标→场值,跨模态最近邻)。

**(b) 熵/置信度阈值的可变长解码:文献丰富。** Medusa typical acceptance;
AdaEDL([PMLR v262](https://proceedings.mlr.press/v262/agrawal24a.html),熵下界<阈即停);
SVIP([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.844/),√H>h 停);
SpecDec++([2405.19715](https://arxiv.org/html/2405.19715),最优停止=阈值策略);
HSDDW(NAACL 2025 findings);TapOut([2511.02017](https://arxiv.org/pdf/2511.02017),
停止策略 bandit,含干净的停止条件对照表);AdaMTP 熵增量停止(§3)。

**(c) 模型自定界(生成时自己产出分割信号):** H-Net(输入侧学习切分,最强先例);
ByteFlow Net([2603.03583](https://arxiv.org/pdf/2603.03583),coding-rate 自分割);
**BLT 论文 §8 自点 future work:"learning the patching model in an end-to-end fashion"**;
MrT5(学习式删除);AdaMTP(输入侧熵边界);Scratchpad Patching
([2605.09630](https://www.alphaxiv.org/abs/2605.09630),2026,熵触发 patch 内瞬时计算,
推理期算力分配的直接证据)。

## 7. Set Transformer 用于文本/字节的先例与 PMA 警告

- **无知名的 Set Transformer(ISAB/PMA)自回归字节/字符编码器先例**(arXiv/OpenAlex/web 均无所获)。
- 结构性事实(Lee et al.,[arXiv:1810.00825](https://arxiv.org/abs/1810.00825)):
  MAB/SAB/ISAB/PMA 定义上**不含位置编码**;整体可证置换不变 → 顺序必须作为输入特征
  注入(我们的"拼接局部位置编码"即标准解法;Perceiver IO 把位置放在 query 侧);
  **PMA 有翻车记录**:ModelNet40 上 ISAB(16)+mean pooling(89.15)胜过完整
  Set Transformer 的 PMA(86.62),作者原话"suffered from an optimization issue";
  PMA k>1 时须接 SAB;ISAB 是 O(n·m) 低秩瓶颈,容量随 inducing points m 增长。
- 社区共识([lucidrains/isab-pytorch#1](https://github.com/lucidrains/isab-pytorch/issues/1)):
  ISAB/PMA 只该用于真正无序的数据;有序序列必须保留位置嵌入。
- → 对应我们的 D8(PMA 先行,mean 消融)与 D2(局部 PE 拼接),风险已登记 R2。
