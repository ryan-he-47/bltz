# AGENTS.md

项目长期约定。新会话先读本文件 + `docs/01-design.md` + `docs/02-construction-plan.md`。

## 项目一句话

ByteField(暂定名,待用户拍板):词表 free 的字节级语言模型——Set Transformer 把变长
字节 patch 聚成 latent,标准 transformer 骨干做 patch 级自回归,条件神经场风格的
查询解码头 `(h, Δ) → byte` 做多未来字节预测(MTP),推理时以解码器自身 softmax 熵
决定变长停止,实现输入侧受控分割 / 输出侧模型自定界的对称架构。

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
- 逐 Δ 诊断是硬性验收项:accuracy/entropy 必须随 Δ 单调劣化。
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
- 外部文献调研报告(2026-09-08):见 `docs/03-literature.md`。
