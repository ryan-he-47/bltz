# AGENTS.md

项目长期约定。新会话先读本文件 + `docs/31-design-v2-ae-mdn.md`(v2 设计,
当前主方向)+ `docs/32-construction-plan-v2.md`(施工计划)。历史入口:
`docs/08-runbook-stage1.md`(v1 起跑)、`docs/01-design.md`(v1 设计)、
`docs/28-phase-summary.md`(v1 阶段性总结)。

## 当前状态(2026-09-17,v2 转向)

- **2026-09-17 用户裁决:H1 不成立,v1 架构存在原理性缺陷**——条件独立
  并行解码(D1)无法表示多模态联合分布("cat/dog/dot" 问题:逐 Δ 边际
  训练 + 独立解码 = 边际乘积当联合,幽灵词无法排除,词级温度不可能),
  可能解释收敛缓慢。**v2 方向拍板**(吸收 CoSE,`docs/31`):字节串自编码器
  (重建塑形,~48 维,梯度与骨干硬切断)+ 骨干在嵌入空间自回归 + MDN/GMM
  头(分量=确定字节串,词级温度成立)+ 增强 BPE 分词(Qwen3.5 BPE ×
  规则预分词,一致必切/不一致概率切)+ EOS 终止符(softmax 第 257 行,
  取代 CoSE 归一化进度与 v1 词界停止)。**施工计划 docs/32(AE 本机先行 →
  缓存 v2 → 骨干+MDN → frozen 臂 POC,joint 降级备选);问询已答并回填
  (预烘焙分割/p=0.5/原始向量+LN、AdaLN 思路内化于 film 与 MDN 零初始化/
  分量内只取 μ_k/frozen 先行/S0 六组网格从 32-256 起跑、保留 patch 内位置
  拼接编码),待用户审阅 docs/31+32 后动工,未动工。**审阅已通过
  (2026-09-17)**:补充两条直接默认——骨干输入宽非线性适配器
  (48→2048→768,非线性升维再降维,不设对照);去噪解码维持全关,
  需要时优先字节腐化式(输入丢字节输出补全)而非 λ 高斯噪声。
  **S0 已完成(2026-09-18)**:48-256 定为 AE 基座(val EM 97.4%,网格与
  几何见 docs/32 §S0);GMM 定稿对角协方差 + 两层非线性缓冲(768→1024→
  1024→K(1+2d));64 维组弃跑;POC 直接 60k(WSD 可早停)。**S1 烟雾验收
  已过(573155;unit 均长 3.47B,直方图入 docs/32 §S1),正式缓存已建成
  (573322:~12.6B units / ~46GB / 3.85h,CACHE_V2_READY),AE 全量重训
  573889 运行中**(sys.path 引导事故 573323 已修复 489d6c1;起步即
  em 0.94@1.9k 步),**60k POC 573890 挂依赖接力**;S2 代码已绿(mdn.py/
  model_v2.py/objectives_v2.py/train_bltz_v2.py + test_v2 过;本机冒烟
  NLL 82.6→56.1)。测试 13 个全绿。
  **HF token 在 $SCRATCH/hf_cache/token(600,工作区禁存)**。
  **AE 验收口径(2026-09-19 拍板)**:POC 起跑前三件套——几何套件/
  频率分层 val(零次桶=真字符串级泛化)/distinct 串量曲线;烂则
  scancel POC。**AE 补救工具箱四件入 docs/32**(去重词库/typo 簇增强/
  合成词探针/随机切分回归待定)。**同日拍板:全量 AE 先用 32 维**
  (ae.yaml 未改回 48 的配置事故意外产出 32 维 best@99000,顺势试水,
  CoSE 低维先例支持;目录已改 ae_32_256_full,POC 用 d_emb=32;48 留作
  升级臂;模型构造期 d_emb 断言已入码防再撞)。**验收已过,POC 574099
  运行中(2026-09-19)**:三件套通过(零次桶 EM 93.6%/SC 0.384/distinct
  ~76M);首发 574046 撞 MDN σ 坍缩死锁(根因入 docs/32 §S3),sigma_floor
  定 0.05 修复后重跑;当前 750 步 NLL -25.7、skips=6、~0.40s/step
  (无逐Δ查询网格,比 v1 快 2.7×,60k 全程 ≈7h)。
- v1 全部存档不动:`boundary_2b` 主 run 暂停态(ckpt@~140749 在 scratch
  可续,是否续待 v2 首读后再议);v1 诊断数字(盆地底 2.50、k1-mix 2.173
  等)留作对照锚。

- **项目已复工(用户拍板,2026-09-14 晚):封存撤销,继续实验。** 范式轴复核
  (`docs/16` §5)修正了当日上午的功能轴裁决:单一共享 `(h,Δ)` 偏移坐标头 +
  稠密多视野监督在字节级 MTP 生态中未被实现。**复工入口/新主张口径/第一批
  任务(M1-M5,测量优先)见 `docs/18-restart-handoff.md`**;封存期文档
  (`docs/17`、`docs/17b`)留档不溯改。
- **60k 长 run(547403)已完成且终报归档 `docs/20`**(最终 loss 2.56,20k
  对照 2.66;全程健康未收敛;gcos 全段轨迹判"冲突从未存在于表示层";
  对 M1-M5 的输入见 docs/20 §4)。
- **2026-09-15 GPU 节点诊断(`docs/29`)**:`gpu-v100s-06` 系**驱动层整机卡死**
  (117 个 D 状态僵尸、load 120/配额 8),非坏卡;`slurmd` 正常所以 Slurm 仍接活
  但落上即永久挂死——只有重启能解,**`--exclude=gpu-v100s-06` 保留不动**;
  起飞前体检用 `sinfo` 的 O/A 负载比(docs/29 §5,已入 docs/08)。
- **当前方向**:参数量精调大方向(纯效率 vs 重分配)已由 docs/21 §4 拍板收口
  (encoder 冻结);长程实验与诊断口径见 docs/28 §4 建议,待用户定夺。
- **FiLM 条件头实验已验收**(docs/22 §4.2):同预算与 concat **全方位打平**
  (loss 2.66=2.66,唯一正向差异=词界 space 置信 0.84 vs 0.79;代价 +3.4%
  头参数/+9% 步时/需 grad-ckpt);**默认头保持 concat**,FiLM 留作
  head_type 选项。OOM 修复(head_grad_ckpt,21→11GB)与 gcos 动态模块分组
  (按 head_type)已入码。
- **2026-09-15 拍板:encoder 冻结**(字符级能力视野核心,不再动;inducing8/
  heads4 保持);**FiLM 定为常驻默认头**(2026-09-16 拍板:以后所有 run 一律
  film;"赌后期"系非正式口头表述,不计入实验计划;default.yaml
  `head_type: film` + `head_grad_ckpt: true`);**T1' 时代首波 = 2B patch
  边界探针长 run(docs/27)**:244k 步 ≈ 12.1B 字节,batch 16,~80h。
  **当前:2B 主 run 暂停**(ckpt@~140749 在 scratch 可续);**退火探底已完成**
  (565575;终值 2.52,平滑下降趋势真实且未走平——盆地底 ≤2.52;"振荡变大"
  经量化证伪是采样密度错觉;退火权重诊断全场最佳,docs/28 §5)。
  **阶段性总结在 `docs/28`**(压缩上下文后的入口:AGENTS + docs/28 + docs/27)。
- **2026-09-16 晚拍板(复工后首批决策,详见 docs/27 §5)**:① FiLM 常驻默认头
  (口径修正,见上);② **25k 长退火探针**:同一起点 ckpt@140750,LR 2.21e-4
  线性退到 **0**(final_frac=0.0),排除"冷却不到位"(`slurm/bltz_anneal25k.sbatch`);
  ③ **60k MTP 深度对照**:`n_patches_ahead` 3→1(k_max 48→16),LR 调度同 60k,
  验证"子词级 MTP 损害小模型学习"的外部结论是否适用于本模型(docs/30)。
  **两实验已完成(2026-09-17)**:25k 退火到 0 → 真盆地底 ≈2.50(冷却不到位
  仅占 ~0.02);mtp1@60k 同口径(k1-mix)2.173-2.177,**0.49B 数据追平 k=3 的
  148.7k 退火权重(1.15B,2.3×)**——深 MTP 稀释近场容量方向成立,代价是
  Δ>16 能力缺失与 demb 距离场退化(0.272 vs 0.807);下一步(k=1 常驻/k=2 臂/
  维持 k=3)待用户拍板,详见 docs/30 §4。
- **T1' 已落地(docs/25):encoder 长度分组重打包**——长度分组贴身画布、
  数值逐元等价(test_enc_grouped 对拍 max diff ~1e-6)、纯 torch 零新依赖,
  本机对拍**全步 -50%**、峰值 -16%;T3.3 分桶被其吸收。打包拍板史:
  T3 实测 docs/24 §5(画布 padding 62%,encoder 占前向 76-81%)。
  测试 11 个全绿。
- **集群纪律不变**:home 只放源码/配置/日志/final+best ckpt;数据/缓存/中间
  ckpt 走 scratch(配额 300GB;**缓存是活资产,不删**);登录节点秒级只读;
  不动其它 job(mob_race 等);VPN 掉线=停手待命。
- 更名史:ByteField → bltz(2026-09-12);开源准备完成(MIT,英文入口
  `ARCHITECTURE.md`)。集群 burgundy.hpc.cityu.edu.hk:22。**home
  只放源码/配置/日志/final+best ckpt,数据/缓存/中间 ckpt 全走 scratch(配额
  300GB);不动其它在跑 job(mob_race 等)与项目文件夹;登录节点只跑秒级
  只读命令。** 优雅中断(`<ckpt_dir>\STOP` / Ctrl+C,即存即退,ckpt_every=250)
  + 里程碑快照(milestone_every=2000,weight-only 0.55GB/份,WSD 分支点)。
- **更名(2026-09-12,用户拍板)**:原名 ByteField → **bltz**
  (byte-aware learnable tokenizer,致敬 BLT)。包 `bltz/`、模型类 `BltzLM`、
  入口 `scripts/train_bltz.py`;历史报告(docs/04/05/06)与用户立场归档
  (`docs/blt_research/ByteField用户立场归档.md`)保留旧名不溯改。
- **入口文档**:`docs/01` 设计思想 / `docs/02` 施工计划 / `docs/07` 代码架构 /
  `docs/08` 起跑 runbook / `docs/09` 长期路线;冒烟与诊断报告在 `docs/04`、`docs/05`、`docs/06`。
- commit 史:`706b5a3` 设计文档 → `03504a8` 立场回写+定名 → `b76aacb` 管线+冒烟 →
  `611366a` 缓存层+D-1v2+token基线 → `954a642` 补漏测试 → `1f73c5d` RoPE修复+标定
  → `c9d8edc` fp16 分支 → `9c147c0` 文档日+基线管线+续训rotation →
  `4500e0b` GradScaler新API → `7139054` 优雅中断+ckpt加密 → `8ed22ca`
  更名bltz → `0557b36` 集群脚本 → `cb0479c` tiny分区+实录 → `fec5058` 缓存OOM修复
  → `c1addc5`+`c2c6ffd` workers4+构建心跳 → `f441bc1` 空提交事故恢复 →
  `31ac21f` ShardReader稀疏检查点 → `26a5666` POC策略变更 → `b3c81b2` 训练batch16。
- **警告:旧 smoke ckpt(checkpoints/smoke/)与新 RoPE 约定(半劈)不兼容**,
  仅作历史 artifact(docs/07 §4.1)。
- 测试 9 个,全绿:test_segment / test_model / test_cache / test_token_lm /
  test_shard_train / test_cache_builder / test_baseline_pipeline / test_resume /
  test_interrupt(更名后复跑全绿)。

## 项目一句话

bltz(byte-aware learnable tokenizer,原名 ByteField,致敬 BLT):词表 free 的
字节级语言模型——Set Transformer
把变长字节 patch 聚成 latent,标准 transformer 骨干做 patch 级自回归,条件神经场风格的
查询解码头 `(h, Δ) → byte` 做多未来字节预测(MTP),推理时在 space-like 词界
自然停止、熵中断仅作兜底(D11,2026-09-13)。定位:学院派 LLM 与语言表征维新派的和事佬——token LM 范式 +
字节级可学习 I/O(详见 `docs/blt_research/ByteField用户立场归档.md`)。

## 硬性规则(violations are blocking)

- **用户对每个技术决策有最终拍板权。** 不许擅自改架构、超参、数据集、训练流程。
  到达决策点时用 `question` 工具提问,给选项+工作量估计+推荐。
- **用户兼任资源收集员。** 需要而我拿不到的资源(权重、数据集、包、repo),
  指示用户下载,不要绕过缺失硬干。
- **Docs-first:任何开发前先读 docs/ 相关文档;任何开发完成后必须更新 docs/。**
  所有计划、报告、设计文档归档为 `docs/` 下的 Markdown。
- 本文件是唯一指令源;决策做出后及时更新本文件与 docs/01 的决策记录。

## 环境

- 本机 conda 环境:`cose2`,`E:\MiniConda\envs\cose2\python.exe`(conda 不在 PATH,
  直接调用该 python)。torch 2.11 cu130,RTX 4060 Laptop 8GB(sm89,支持 bf16)。
- **pip 用默认 PyPI 源(机器在香港,清华源约定已废止)。**
- 集群(学校超算,继承 MoB_Head 约定):入口 `burgundy.hpc.cityu.edu.hk:22`,SLURM;
  V100 32GB(sm70,**不支持 bf16**,训练一律 fp16+GradScaler);
  **sbatch 模板必须带 `#SBATCH --exclude=gpu-v100s-06`**(该节点坏);
  **06 的真实病因已于 2026-09-15 确诊(`docs/28`):NVIDIA 驱动层整机卡死**——不是坏卡、
  不是硬件死;`slurmd` 正常响应,所以 Slurm 报 `MIXED` 并持续接活,每个落到它上面的 job
  都会永久挂死。**只有重启能解**,在运维重启前必须继续排除。
  (另:`gpu-v100s-05` 自 2026-09-04 起 `DOWN+DRAIN`,反而没在现役 exclude 名单里。)
  **起飞前体检(秒级只读)**:`sinfo -N -o '%N|%C|%O|%G|%t'`——看 `O`(负载)是否远大于
  `A`(已分配 CPU);健康节点比值 ≤1,06 是 15×(踩坑史与探针用法见 `docs/28` §3)。
  登录节点只跑秒级只读命令;校园 VPN 会周期性掉线,SSH 超时=停手待命,不要探测重试。
  **scratch 个人配额只有 300GB**(787T 是全集群,勿误判):hf_cache 已清理,
  里程碑 weight-only 化,长 run 前算清 ckpt 预算(docs/08 §3.2)。
- 重型训练上集群;本机 8GB 只做原型与小规模(≤124M 级)。

## 实验纪律(继承 simple_point_cloud / MoB_Head 血泪史)

- 改训练行为必须跑诊断脚本验证,不许"应该能行"。
- **条件解码头的查询永远是相对偏移 Δ,绝不喂绝对位置**(v6 静态扇作弊教训:
  绝对位置会被解码器忽视,error 随 horizon 平坦 = 作弊签名)。
- ~~逐 Δ 诊断硬性验收~~(**2026-09-13 用户拍板废止**:D-1 判据武断、不贴合
  模型特性;POC 诊断一律定性描述、不设通过门槛,套件见 docs/08 §4)。
- **判 run 是否在推进,只看 `train.log` 步号,不看 `squeue`/`sacct` 状态**——挂死的
  job 与健康长跑在作业系统里完全一样(2026-09-15 教训:某 job 报 `RUNNING` 13h 实为
  卡死的 `python`;一个跑满 5 天 `TIMEOUT` 的 job 实为挂死 5 天;见 `docs/28` §4)。
- **诊断坏节点有代价**:挂死会留下 `SIGKILL` 也杀不掉的 D 状态进程,**每探一次都在给
  该节点加永久僵尸**;只在必要时探、探完 scancel。另:`timeout` 杀不掉 D 状态,所以
  兜住挂死调用必须走"后台化 + 固定窗口回收"设计,不能靠 `timeout`(两个坑见 `docs/28` §3)。
- 训练守卫:spike_skip = max(10×running median, 2000);Adam β2=0.95;不许擅自改。
- LR 调度 house rule:可续训/探底 run 默认 **WSD**;weight-only 续训重启的 peak
  不得超过上一 run 的结束 LR。
- **POC 阶段方法论(2026-09-13 用户拍板,D10)**:看趋势、做定性分析、快速迭代;
  弱参考用同级现成模型;严格 baseline 与 D-5 对照协议缓办。
- **不跨场景套经验**:MoB 蒸馏解码头的曲线形态/节奏结论 ≠ 从头预训练 LM;
  曲线预期按本 run 实测建立(2026-09-13 用户指正,MoB 教训仅限工程纪律)。
- 测试为脚本式:逐文件 `python tests/test_xxx.py` 直接运行,无 pytest。
- YAML 配置 + `--set a.b=value` 覆盖;浮点覆盖必须带小数点(`1.0e-8`)。
- git 只提交代码+文档;`data/`、`checkpoints/`、`runs/`、`viz/` gitignored。
- 长任务:`python -u` + Start-Process detached,重定向到 checkpoints/*.log,
  用 Get-Content -Tail 轮询。

## 参考资源

- `reference_projects/simple_point_cloud/` — 架构模板(Set Transformer / 条件解码 /
  训练器 / 实验日志方法论)。
- `reference_projects/MoB_Head/` — 问题定义(固定词表的不便与昂贵)与工程约定来源。
- `reference_projects/2412.09871v1.txt` — BLT 原文文本版(PDF 提取)。
- `reference_projects/2605.08044v1.txt` — Fast BLT(BLT-D/S/DV)文本版。
- `docs/blt_research/ByteField用户立场归档.md` — 用户立场(定位/生态位/设计哲学)。
- 外部文献调研报告(2026-09-08):见 `docs/03-literature.md`。
