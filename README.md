# bltz

词表 free 的字节级语言模型——**学院派 LLM 与语言表征维新派之间的和事佬**:
把分词算法打包出的 token-ish patch 用小 encoder(Set Transformer)编成向量,
标准 transformer 骨干在向量上做自回归,条件神经场风格的查询解码头
`(h, Δ) → byte` 把嵌入向量解回一小段字节语义单元,推理时以解码器自身
softmax 熵决定变长停止。

本质:**传统 token LM 的范式 + 字节级分辨率的可学习 I/O**——把定死的大嵌入
矩阵换成一套可学习、可泛化、更易微调的嵌入 i/o。组合式创新,新瓶装旧酒,
不追求颠覆;长板主张是 vocab-free 的泛化性与推理效率,其余皆为 trade-off。
高风险设计点(D1 纯并行解码 / D2 patch 局部编码器 / D5 模型自分割)是实验本体:
本架构能 work 的前提与 BLT 的若干研究结论相矛盾——若我们 work,那些组件的
必要性即被证伪。

**状态:管线施工与冒烟阶段。** 入口文档:

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
| `docs/blt_research/ByteField用户立场归档.md` | 用户立场:工作定位、学术生态位、设计哲学 |
| `reference_projects/` | 参考:BLT / Fast BLT 论文文本版、simple_point_cloud(架构模板)、MoB_Head(问题定义) |

参考论文:Pagnoni et al., *Byte Latent Transformer* (arXiv:2412.09871);
Kallini et al., *Fast Byte Latent Transformer* (arXiv:2605.08044)。
