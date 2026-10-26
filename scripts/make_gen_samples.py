"""Build a human-reviewable generation-samples sheet from the acceptance logs.

Sources (all local, produced 2026-10-26):
  checkpoints/word_cpt_s_accept.log            CPT S terminal §5 (8 prompts x 6 readouts)
  checkpoints/word_p2_accept_gen_rerun.log     p2 baseline, SAME script & seed (48 steps)
  checkpoints/word_05b_p2_accept.log           p2 original acceptance log (64 steps, old script)
  checkpoints/word_cpt_s_accept.log §1         diag_v2 component-sampling generation

Writes checkpoints/word_cpt_s_gen_samples.md (side-by-side CPT vs p2).

  python scripts/make_gen_samples.py
"""
from __future__ import annotations

import re
from pathlib import Path

CPT = "checkpoints/word_cpt_s_accept.log"
P2_RERUN = "checkpoints/word_p2_accept_gen_rerun.log"
P2_ORIG = "checkpoints/word_05b_p2_accept.log"
OUT = "checkpoints/word_cpt_s_gen_samples.md"

READOUT_ORDER = ["mode", "modes-peak T=0.6", "modes-peak T=1.0",
                 "snap argmax", "snap t=0.8", "snap t=1.0"]


def read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        txt = raw.decode("utf-16", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    return txt.replace("\r\n", "\n").replace("\r", "\n")


def _clean(text: str) -> str:
    """Drop trailing log-marker lines that leak into a generation block."""
    bad = ("[gen]", "WORD_05B", "[diag_v2]", "[section", "[verify", "DONE", "=====", "=== ")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(bad)]
    return "\n".join(lines).strip("\n")


def parse_prompts(path: str, start_marker: str | None = None):
    """-> {prompt_idx: {"prompt": bytes-ish str, "readouts": {label: text}}}"""
    txt = read_text(path)
    if start_marker:
        i = txt.find(start_marker)
        if i >= 0:
            txt = txt[i:]
    blocks: dict[int, dict] = {}
    cur_p, cur_r = None, None
    for line in txt.splitlines():
        m = re.match(r"==================== prompt (\d+): (.*?) ==========", line)
        if m:
            cur_p = int(m.group(1))
            blocks[cur_p] = {"prompt": m.group(2), "readouts": {}}
            cur_r = None
            continue
        m = re.match(r"--- (.+?) ---$", line)
        if m:
            cur_r = m.group(1).strip()
            blocks[cur_p]["readouts"].setdefault(cur_r, [])
            continue
        if cur_p is not None and cur_r is not None:
            blocks[cur_p]["readouts"][cur_r].append(line)
    out = {}
    for p, b in blocks.items():
        out[p] = {"prompt": b["prompt"],
                  "readouts": {k: _clean("\n".join(v)) for k, v in b["readouts"].items()}}
    return out


def parse_diag_gen(path: str):
    """diag_v2 §1 generation block -> {tau: text}."""
    txt = read_text(path)
    out = {}
    pat = re.compile(
        r"--- tau=([0-9.]+) top3-conf-rerank \(.*?\) ---\n(.*?)(?=\n--- tau=|\n=== |\Z)",
        re.S)
    for m in pat.finditer(txt):
        out[f"tau={m.group(1)}"] = _clean(m.group(2))
    return out


cpt = parse_prompts(CPT, "=== 5. extended generation")
p2 = parse_prompts(P2_RERUN)
p2o = parse_prompts(P2_ORIG, "=== 5. extended generation")
cpt_diag = parse_diag_gen(CPT)
p2_diag = parse_diag_gen(P2_ORIG)

lines: list[str] = []
w = lines.append

w("# CPT S 档验收 · 生成样例(人工审阅版)")
w("")
w("- **模型**:`checkpoints/word_cpt_s_last.pt`(S 档 CPT 终态,step 96292)")
w("- **协议**:docs/38 §5 扩展生成——8 prompt × 6 读出 × 48 units")
w("  (mode / modes-peak T=0.6 / modes-peak T=1.0 / snap argmax / snap t=0.8 / snap t=1.0)")
w("- **对照**:p2 基线用**同脚本同种子**重跑(`scripts/gen_accept.py`,48 units),")
w("  原始日志:CPT=`checkpoints/word_cpt_s_accept.log` §5;p2=`checkpoints/word_p2_accept_gen_rerun.log`")
w("- **判定口径**:生成只看正向维度(语法骨架/主题黏合/流畅度/长程连贯);")
w("  确定性读出的吸引子循环按 2026-10-09 裁定**不作能力否决项**。")
w("- prompt 0-3 = 评测序列前 16 units;prompt 4-7 = 常用开头句。")
w("")
w("---")
w("")
for p in sorted(cpt):
    pr = cpt[p]["prompt"]
    w(f"## prompt {p}: {pr}")
    w("")
    for label in READOUT_ORDER:
        a = cpt[p]["readouts"].get(label, "")
        b = p2.get(p, {}).get("readouts", {}).get(label, "")
        w(f"### {label}")
        w("")
        w("**CPT S**")
        w("```")
        w(a if a else "(empty)")
        w("```")
        w("**p2(同脚本重跑)**")
        w("```")
        w(b if b else "(missing)")
        w("```")
        w("")

w("---")
w("")
w("# 附录 A:diag_v2 §1 生成(分量采样 + conf-rerank 重投影;非密度读出主口径)")
w("")
w("CPT S:")
w("")
w("```")
for k, v in cpt_diag.items():
    w(f"--- {k} ---")
    w(v)
    w("")
w("```")
w("")
w("p2(其原始验收日志 §1):")
w("")
w("```")
for k, v in p2_diag.items():
    w(f"--- {k} ---")
    w(v)
    w("")
w("```")
w("")
w("# 附录 B:旧脚本 p2 原始 §5(64 units,供参考,不作跨脚本比较)")
w("")
for p in sorted(p2o):
    w(f"## [old-script] prompt {p}: {p2o[p]['prompt']}")
    w("")
    for label in READOUT_ORDER:
        t = p2o[p]["readouts"].get(label, "")
        if t:
            w(f"**{label}**")
            w("```")
            w(t)
            w("```")
    w("")

Path(OUT).write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}: {len(lines)} lines, prompts={len(cpt)}")
