# archive/ — 退役脚本(2026-10-27)

用户 2026-10-27 拍板(docs/49):**pushforward 系列打分全部退役** —— 它只是
当初为了与字节语言模型比较 bpb 才实现的旁证尺,不是衡量模型性能的主要手段。
现行 MC 打分口径 = unit 嵌入空间 GMM 对数似然(scripts/lmeval_bltz.py +
scripts/mc_eval.py)。

| 文件 | 退役原因 |
|---|---|
| `pushforward_bpb.py` | 推前字节 bpb 旁证尺;2026-10-27 退役(docs/49)。脚本保留仅作历史追溯,不再属于 pipeline |

2026-10-27 同时删除的还有一批 untracked 的一次性诊断脚本(`lmeval_bltz_debug.py`、
`lmeval_bltz_sanity*.py`、`commonsense_probe.py`、`mc_score_diag.py`)——它们的
结论已固化进 docs/39 #46 与 docs/49,且依赖已删除的旧 `score_mode` API,不再保留。
