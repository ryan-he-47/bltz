# AGENTS.md

项目长期约定。新会话先读本文件 + `docs/08-runbook-stage1.md`(起跑)或
`docs/01-design.md`(设计)。

## 当前状态(2026-09-09)

- **阶段:设计与管线完成,Stage 1/2 就绪待发。** 明日(用户醒后)跑全量缓存 +
  twin 双臂,待用户拍板:预算选项 A/B/C/D(docs/06 §4)、磁盘 ~80GB、缓存下载放行。
- **入口文档**:`docs/01` 设计思想 / `docs/02` 施工计划 / `docs/07` 代码架构 /
  `docs/08` 明日 runbook / `docs/09` 长期路线;冒烟与诊断报告在 `docs/04`、`docs/05`、`docs/06`。
- commit 史:`706b5a3` 设计文档 → `03504a8` 立场回写+定名 → `b76aacb` 管线+冒烟 →
  `611366a` 缓存层+D-1v2+token基线 → `954a642` 补漏测试 → `1f73c5d` RoPE修复+标定
  → `c9d8edc` fp16 分支。
- **警告:旧 smoke ckpt(checkpoints/smoke/)与新 RoPE 约定(半劈)不兼容**,
  仅作历史 artifact(docs/07 §4.1)。
- 测试 8 个,全绿:test_segment / test_model / test_cache / test_token_lm /
  test_shard_train / test_cache_builder / test_baseline_pipeline / test_resume。

## 项目一句话

ByteField(名称已由用户立场文件采用):词表 free 的字节级语言模型——Set Transformer
把变长字节 patch 聚成 latent,标准 transformer 骨干做 patch 级自回归,条件神经场风格的
查询解码头 `(h, Δ) → byte` 做多未来字节预测(MTP),推理时以解码器自身 softmax 熵
决定变长停止。定位:学院派 LLM 与语言表征维新派的和事佬——token LM 范式 +
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
  登录节点只跑秒级只读命令;校园 VPN 会周期性掉线,SSH 超时=停手待命,不要探测重试。
- 重型训练上集群;本机 8GB 只做原型与小规模(≤124M 级)。

## 实验纪律(继承 simple_point_cloud / MoB_Head 血泪史)

- 改训练行为必须跑诊断脚本验证,不许"应该能行"。
- **条件解码头的查询永远是相对偏移 Δ,绝不喂绝对位置**(v6 静态扇作弊教训:
  绝对位置会被解码器忽视,error 随 horizon 平坦 = 作弊签名)。
- 逐 Δ 诊断是硬性验收项(v2 分层判据,见 `docs/05-diag-d1-v2.md`):跨未来
  patch 的预测力(acc)必须衰减;逐 Δ 熵极差≈0 = 头忽视 Δ = 作弊。
- 训练守卫:spike_skip = max(10×running median, 2000);Adam β2=0.95;不许擅自改。
- LR 调度 house rule:可续训/探底 run 默认 **WSD**;weight-only 续训重启的 peak
  不得超过上一 run 的结束 LR。
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
