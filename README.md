# bltz

*byte-aware learnable tokenizer — a vocab-free byte-level language model:
Set-Transformer patch encoder → causal transformer backbone → conditional
neural-field head `(h, Δ) → byte` (multi-future-byte prediction), with
word-boundary natural stopping at inference.* **Status: archived**
(research concluded 2026-09-14; the architecture works at POC scale, the
novelty niche closed — see `docs/17`). External readers: start with
[`ARCHITECTURE.md`](ARCHITECTURE.md) (English). License: MIT.

词表 free 的字节级语言模型——**学院派 LLM 与语言表征维新派之间的和事佬**:
把分词算法打包出的 token-ish patch 用小 encoder(Set Transformer)编成向量,
标准 transformer 骨干在向量上做自回归,条件神经场风格的查询解码头
`(h, Δ) → byte` 把嵌入向量解回一小段字节语义单元,推理时在 space-like
词界自然停止、熵中断仅作兜底(D11)。

本质:**传统 token LM 的范式 + 字节级分辨率的可学习 I/O**——把定死的大嵌入
矩阵换成一套可学习、可泛化、更易微调的嵌入 i/o。组合式创新,新瓶装旧酒,
不追求颠覆;长板主张是 vocab-free 的泛化性与推理效率,其余皆为 trade-off。
高风险设计点(D1 纯并行解码 / D2 patch 局部编码器 / D5 模型自分割)是实验本体:
本架构能 work 的前提与 BLT 的若干研究结论相矛盾——若我们 work,那些组件的
必要性即被证伪。

**状态:已封存(2026-09-14,归档)**——收尾说明见 `docs/17-archive-closure.md`。入口文档:

| 文档 | 内容 |
|---|---|
| `docs/01-design.md` | 设计思想:决策记录、定位、架构规格、训练目标、推理循环、诊断清单、风险登记册 |
| `docs/02-construction-plan.md` | 施工计划:Stage 0-4、验收门、失败预案、资源清单 |
| `docs/03-literature.md` | 外部文献与代码调研(2026-09-08,含全部 URL) |
| `docs/04-stage0-1-smoke.md` | 管线冒烟报告(过拟合/真实短跑/θ_stop 机制) |
| `docs/05-diag-d1-v2.md` | D-1 诊断判据修订(v2 分层)与首读 |
| `docs/06-twin-calibration.md` | twin 本地标定(batch 8 / RoPE 修复 / gemm-bound / 预算选项) |
| `docs/07-code-architecture.md` | 代码架构与运维手册(数据结构/命令/坑位/性能事实) |
| `docs/08-runbook-stage1.md` | 明日 runbook:全量缓存 + twin 双臂(无损起跑指南) |
| `docs/09-roadmap.md` | 长期路线图:Stage 3/4 规格、开放超参、论文定位备忘 |
| `docs/10-poc-diag-20k.md` | POC 诊断档案:首个 20k 模型七视角定性结果、D11 停止规则、开放问题(2026-09-13) |
| `docs/11-gcos-measure-fix.md` | gcos 测量修正施工单(T1-T5,2026-09-13) |
| `docs/12-gcos-second-opinion.md` | gcos 负余弦:独立第二意见(整理稿) |
| `docs/13-mtp-conflict-prior-art.md` | MTP/多视界梯度冲突先例档案(2026-09-13) |
| `docs/14-paradigm-lineage-and-positioning.md` | 范式普查(24 模型)、并行解码谱系与学术定位裁决(2026-09-14) |
| `docs/15-hat-deep-dive.md` | HAT 深挖(逐行)+ Outlook 兑现审计(2026-09-14) |
| `docs/16-dmbp-decoder-comparison.md` | DMBP 精读与解码范式对比(bltz/HAT/DMBP,2026-09-14) |
| `docs/17-archive-closure.md` | 项目封存说明:结论、资产索引、遗留事项(2026-09-14) |
| `docs/18-retrospective.md` | 设计回顾与封存说明(技术版):动机/实测/方法学教训/残余问题 |
| `docs/19-60k-wrapup-guide.md` | 60k 长 run 收尾手册(待通知后执行:曲线分析+诊断+轨迹+清理) |
| `docs/blt_research/ByteField用户立场归档.md` | 用户立场:工作定位、学术生态位、设计哲学 |
| `reference_projects/` | 参考:BLT / Fast BLT / HAT / DMBP 论文文本版、simple_point_cloud(架构模板)、MoB_Head(问题定义) |

参考论文:Pagnoni et al., *Byte Latent Transformer* (arXiv:2412.09871);
Kallini et al., *Fast Byte Latent Transformer* (arXiv:2605.08044)。

注:`docs/` 技术文档为中文;`ARCHITECTURE.md` 为英文入口。
代码以 MIT 协议开源(见 `LICENSE`)。
