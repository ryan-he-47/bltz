"""Build the p2 vs CPT S lm-eval comparison table from the two saved result JSONs."""
import json
from pathlib import Path


def load(p):
    txt = Path(p).read_text(encoding="utf-8", errors="replace")
    start = txt.index("{")
    return json.loads(txt[start:])


p2 = load("checkpoints/lmeval_p2_692113_results.json")
cpt = load("checkpoints/lmeval_cpt_s_692113_results.json")
order = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande",
         "openbookqa", "boolq", "commonsense_qa"]
print(f"{'task':16s} {'n':>6s} {'p2 acc':>8s} {'cpt acc':>8s} {'dpp':>7s} "
      f"{'p2 an':>8s} {'cpt an':>8s}")
accs_p, accs_c, ans_p, ans_c = [], [], [], []
for t in order:
    a, b = p2[t], cpt[t]
    n = a["sample_len"]
    ap, ac = a["acc,none"], b["acc,none"]
    d = (ac - ap) * 100
    accs_p.append(ap)
    accs_c.append(ac)
    if "acc_norm,none" in a:
        np_, nc = a["acc_norm,none"], b["acc_norm,none"]
        ans_p.append(np_)
        ans_c.append(nc)
        print(f"{t:16s} {n:6d} {ap*100:7.2f}% {ac*100:7.2f}% {d:+6.2f} "
              f"{np_*100:7.2f}% {nc*100:7.2f}%")
    else:
        print(f"{t:16s} {n:6d} {ap*100:7.2f}% {ac*100:7.2f}% {d:+6.2f} "
              f"{'-':>8s} {'-':>8s}")
print(f"\nmean acc (8 tasks): p2 {sum(accs_p)/len(accs_p)*100:.2f}% "
      f"cpt {sum(accs_c)/len(accs_c)*100:.2f}% "
      f"(delta {(sum(accs_c)-sum(accs_p))/len(accs_c)*100:+.2f}pp)")
print(f"mean acc_norm (6 tasks): p2 {sum(ans_p)/len(ans_p)*100:.2f}% "
      f"cpt {sum(ans_c)/len(ans_c)*100:.2f}% "
      f"(delta {(sum(ans_c)-sum(ans_p))/len(ans_c)*100:+.2f}pp)")
