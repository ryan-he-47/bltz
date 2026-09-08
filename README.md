# ByteField(暂定名,待拍板)

词表 free 的字节级语言模型:Set Transformer 把变长字节 patch 聚成 latent,
transformer 骨干做 patch 级自回归,条件神经场风格的查询解码头 `(h, Δ) → byte`
做多未来字节预测(MTP),推理时以解码器自身 softmax 熵决定变长停止——
输入侧受控分割 / 输出侧模型自定界的对称架构。

**状态:设计评审阶段。** 入口文档:

| 文档 | 内容 |
|---|---|
| `docs/01-design.md` | 设计思想:决策记录、架构规格、训练目标、推理循环、诊断清单、风险登记册 |
| `docs/02-construction-plan.md` | 施工计划:Stage 0-4、验收门、失败预案、资源清单 |
| `docs/03-literature.md` | 外部文献与代码调研(2026-09-08,含全部 URL) |
| `reference_projects/` | 参考:BLT / Fast BLT 论文文本版、simple_point_cloud(架构模板)、MoB_Head(问题定义) |

参考论文:Pagnoni et al., *Byte Latent Transformer* (arXiv:2412.09871);
Kallini et al., *Fast Byte Latent Transformer* (arXiv:2605.08044)。
