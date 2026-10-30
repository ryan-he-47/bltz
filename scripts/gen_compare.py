"""Side-by-side generation-samples sheet from gen_accept.py logs (any number of models).

Parses the 8-prompt x 6-readout extended generation log format of
scripts/gen_accept.py and writes a markdown sheet with one column-block per model.

  python scripts/gen_compare.py \
      --log p2=checkpoints/gen_edu5b/p2_last.log \
      --log edu5b=checkpoints/gen_edu5b/edu5b.log \
      --out checkpoints/word_edu5b_gen_samples.md \
      --title "edu5b 生成样例(p2 终点 vs edu5b 终点)" \
      --note "同脚本同种子;48 units;8 prompt x 6 读出"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

READOUT_ORDER = ["mode", "modes-peak T=0.6", "modes-peak T=1.0",
                 "snap argmax", "snap t=0.8", "snap t=1.0"]
BAD_PREFIX = ("[gen]", "[diag_v2]", "[section", "[verify", "DONE", "=====",
              "=== ", "Traceback", "File \"", "  ")


def read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(BAD_PREFIX)]
    return "\n".join(lines).strip()


def parse(path: str) -> dict[int, dict]:
    blocks: dict[int, dict] = {}
    cur_p = cur_r = None
    for line in read_text(path).replace("\r\n", "\n").splitlines():
        m = re.match(r"==================== prompt (\d+): (.*?) ==========", line)
        if m:
            cur_p = int(m.group(1))
            blocks[cur_p] = {"prompt": m.group(2), "readouts": {}}
            cur_r = None
            continue
        m = re.match(r"--- (.+?) ---$", line)
        if m:
            cur_r = m.group(1).strip()
            blocks[cur_p]["readouts"][cur_r] = []
            continue
        if cur_p is not None and cur_r is not None:
            blocks[cur_p]["readouts"][cur_r].append(line)
    return {p: {"prompt": b["prompt"],
                "readouts": {k: _clean("\n".join(v)) for k, v in b["readouts"].items()}}
            for p, b in blocks.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True,
                    help="tag=path (可重复;第一个是最左边一列)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="生成样例并排")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    models = []
    for spec in args.log:
        tag, path = spec.split("=", 1)
        models.append((tag, parse(path)))
    prompts = sorted(models[0][1])
    lines: list[str] = []
    w = lines.append
    w(f"# {args.title}")
    w("")
    if args.note:
        w(f"- {args.note}")
    w(f"- 模型:" + "、".join(f"`{t}`" for t, _ in models))
    w("")
    for p in prompts:
        pr = models[0][1][p]["prompt"]
        w(f"## prompt {p}: {pr}")
        w("")
        for label in READOUT_ORDER:
            w(f"### {label}")
            w("")
            for tag, data in models:
                txt = data.get(p, {}).get("readouts", {}).get(label, "")
                w(f"**{tag}**")
                w("```")
                w(txt if txt else "(empty)")
                w("```")
            w("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: {len(lines)} lines, prompts={len(prompts)}, models={[t for t, _ in models]}")


if __name__ == "__main__":
    main()
