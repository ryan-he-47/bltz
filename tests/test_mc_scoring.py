"""docs/49 unit-space MC scoring tests.

Covers: ctx/cont 切分(docs/49 §3.2:分隔符不计分、候选单独切词一致性、兜底)、
归一化指标小工具(§4:平局取最小下标/pearson/stderr)、以及(有 GPU+ckpt 时)
适配器 raw 分数与手算 `-Σ chain_nll[候选项]` 的一致性。

Run:  python tests/test_mc_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bltz.segment_v2 import segment_word  # noqa: E402
from lmeval_bltz import BltzEvalLM  # noqa: E402
from mc_eval import _argmax, _gold_index, _pearson, _stderr  # noqa: E402

TOK = str(ROOT / "data" / "qwen35_tokenizer")
CKPT = ROOT / "checkpoints" / "word_cpt_s_last.pt"
AE = ROOT / "checkpoints" / "ae_48_256_word_full" / "best.pt"


class _Req:
    def __init__(self, *a):
        self.args = a


def _bare_adapter() -> BltzEvalLM:
    """只给 _split_units/_segment 用的裸对象(不加载模型,纯 CPU)。"""
    lm = BltzEvalLM.__new__(BltzEvalLM)
    lm.tok_dir = TOK
    lm.l_max = 32
    lm.verify = None
    return lm


def test_split_boundary() -> None:
    lm = _bare_adapter()
    ctx = "Question: What is the state of water at 0C?\nAnswer:"
    cont = " a liquid"
    cu, vu = lm._split_units(ctx, cont)
    # 候选 unit = 候选文本单独切词的结果(分隔符空格不计入候选项)
    assert vu == segment_word(b"a liquid", TOK, l_max=32), vu
    assert all(u.strip() for u in vu), vu
    assert not any(u == b" " for u in vu)
    # 跨缝 unit(含分隔符空格)归 context
    assert cu and cu[0] == segment_word(ctx.encode(), TOK, l_max=32)[0]

    # context 末尾空白: 移到 continuation(与 lm-eval _encode_pair 同约定)
    cu2, vu2 = lm._split_units("Hello ", "world")
    assert vu2 == [b"world"], vu2
    assert cu2 == [b"Hello "], cu2

    # csqa 单字母候选: 1 项
    cu3, vu3 = lm._split_units("Question: x\nA. a\nB. b\nAnswer:", " A")
    assert vu3 == [b"A"], vu3

    # 兜底: 跨缝把候选吞掉(无分隔符) -> 候选单独切词
    cu4, vu4 = lm._split_units("Answ", "er")
    assert vu4 == [b"er"], vu4
    assert cu4 == [b"Answ"], cu4
    print("[PASS] split boundary")


def test_metric_helpers() -> None:
    # 平局取最小下标(np.argmax 语义)
    assert _argmax([1.0, 1.0, 2.0]) == 2
    assert _argmax([2.0, 2.0, 1.0]) == 0
    # pearson 退化输入
    assert _pearson([1.0, 2.0], [1.0, 2.0]) != _pearson([1.0, 2.0], [1.0, 2.0])  # nan
    assert abs(_pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-9
    # binomial stderr
    assert abs(_stderr(0.25, 100) - 0.04330127) < 1e-6

    class _T:
        def __init__(self, target):
            self._target = target

        def doc_to_target(self, doc):
            return self._target

    assert _gold_index(_T(2), {}, ["a", "b", "c"]) == 2
    assert _gold_index(_T("1"), {}, ["a", "b", "c"]) == 1
    assert _gold_index(_T("C"), {}, ["A", "B", "C", "D"]) == 2
    print("[PASS] metric helpers")


def test_adapter_consistency() -> None:
    import torch

    if not torch.cuda.is_available() or not CKPT.exists() or not AE.exists():
        print("[SKIP] adapter consistency (no CUDA/ckpt)")
        return
    lm = BltzEvalLM(pretrained=str(CKPT), ae=str(AE), tok_dir=TOK, device="cuda")
    lm.heartbeat = 10_000
    lm.collect_meta = True
    ctx, cont = "Question: What is the state of water at 0C?\nAnswer:", " a liquid"
    ll, _greedy = lm.loglikelihood([_Req(ctx, cont)])[0]
    cu, vu = lm._split_units(ctx, cont)
    manual = -float(lm._chain_nll(cu + vu)[len(cu):].sum())
    assert abs(ll - manual) < 1e-5, (ll, manual)
    assert lm.last_meta[-1]["n_units"] == len(vu)
    # 空 context 的 PMI 分支可运行
    uncond = lm.unconditional_loglikelihood(vu)
    assert uncond == uncond  # not NaN
    print(f"[PASS] adapter consistency (S_raw={ll:.3f}, n_units={len(vu)}, "
          f"S_uncond={uncond:.3f})")


if __name__ == "__main__":
    test_split_boundary()
    test_metric_helpers()
    test_adapter_consistency()
    print("[test_mc_scoring] ALL PASS")
