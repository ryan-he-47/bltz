"""bltz × lm-evaluation-harness adapter(docs/49,2026-10-27 口径修订).

模型: bltz 0.5B word 臂(冻结 AE ae_48_256_word_full + 骨干 + linear GMM 头)。

打分口径(2026-10-27 用户拍板,docs/49,取代旧 pushforward 口径):
- 骨干以 unit 为单位、在嵌入空间给出概率分布(K=64 对角 GMM over λ∈R^48),
  因此打分直接在嵌入空间做:候选 unit 的分数 = GMM 在该 unit 的冻结 AE 嵌入
  处的对数密度(与训练 MDN NLL 是同一个函数,确定性,无解码器/蒙特卡洛)。
- continuation 分数 S_raw = Σ_{候选项} log p(λ(u_i) | 前文)(teacher forcing)。
  长度不均衡按标准 token LM 的处理方式:主指标
  acc_unit = argmax S_raw / n_units(项数 = unit 数,对应 token LM 的 token 数);
  另报 acc_raw(标准 acc)与 acc_char(S_raw / 字符数,对齐 lm-eval acc_norm
  的字符归一实现)。归一化指标由 scripts/mc_eval.py 计算,本适配器只忠实返回
  S_raw(保持 LM API 语义)。
- pushforward(整串字节概率)系列打分已于 2026-10-27 全部退役 —— 它只是当初
  与字节 LM 比较 bpb 的旁证尺,不是性能主口径;脚本已归档 archive/。
- 切词:ctx+cont 整段 segment_word(word 臂空格尾随约定)+ VerifyResplitter,
  与训练同管线;跨缝 unit 归 context 不计分 —— 等价于标准 causal LM 的
  "整段 tokenize 后按 context 长度切出 continuation",分隔符不构成一项。

LM API:
- loglikelihood(ctx, cont) -> (S_raw nats, is_greedy)。continuation 超长时
  右截断;context 超长时左截断保住 continuation(与 harness 默认截断策略同向)。
- loglikelihood_rolling:滑窗整文 NLL(S=4096 units;首窗全计,后续窗重叠
  半窗,每个 unit 恰计一次且 context ≥ S/2;属有文档的口径选择)。
- generate_until: modes-peak 采样(modes() 枚举局部众数 + 峰顶海拔
  log_dens 加权 + 核采样 0.95,docs/39 #36 翻案)+ isfinite 守卫(发散即停
  该请求并记录,不伪造样本)+ stop 序列 + max_gen_units 上限(适配器逐 unit
  重算 context 是 O(n^2),IFEval 类长生成必须截断预算)。

本机用法(8GB GPU, batch 1 串行):
  # 冒烟自检(不跑 harness):验证加载/切词/打分/生成
  E:\\MiniConda\\envs\\cose2\\python.exe scripts/lmeval_bltz.py --selftest
  # 8 项 MC 电池(新口径:acc_unit/acc_raw/acc_char/PMI):
  E:\\MiniConda\\envs\\cose2\\python.exe scripts/mc_eval.py --ckpt ... --tag cpt_s
  # harness 原生 acc 交叉验证(raw 口径):
  python -m lm_eval --model bltz --model_args pretrained=<ckpt> --tasks hellaswag

集群用法:ckpt 换 $SCRATCH 路径,batch_size 可开 4-8;MC 指标用
scripts/mc_eval.py(见 slurm/bltz_mc_eval.sbatch)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.segment_v2 import segment_word
from bltz.verify import VerifyResplitter
from train_ae import tensorize

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

DEV = "cuda"
CHAT_USER = "<|user|>"
CHAT_ASSISTANT = "<|assistant|>"
CHAT_END = "<|end|>"


@register_model("bltz")
class BltzEvalLM(LM):
    """bltz v2 word-arm LM for lm-evaluation-harness (see module docstring)."""

    def __init__(
        self,
        pretrained: str = "checkpoints/word_05b_p2_last.pt",
        ae: str = "checkpoints/ae_48_256_word_full/best.pt",
        tok_dir: str = "data/qwen35_tokenizer",
        device: str = "cuda",
        verify: bool = True,
        temperature: float = 1.0,
        chat_template: bool = False,
        max_gen_units: int = 256,
        max_length: int = 0,          # 0 -> 取 ckpt 的 data.n_patches
        heartbeat: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.dev = device
        self.temperature = float(temperature)
        self.chat_template = bool(chat_template)
        self.max_gen_units = max(1, int(max_gen_units))
        self.heartbeat = max(1, int(heartbeat))
        # 供 mc_eval.py 读取的逐请求元数据(docs/49 §5):n_units/n_chars/n_bytes。
        self.collect_meta = False
        self.last_meta: list[dict] = []

        ae_state = torch.load(ae, map_location="cpu", weights_only=False)
        self.ae = ByteStringAE(Cfg(ae_state["cfg"]))
        self.ae.load_state_dict(ae_state["model"])
        st = torch.load(pretrained, map_location="cpu", weights_only=False)
        self.model = BltzLMv2(Cfg(st["cfg"]), self.ae).to(device).eval()
        self.model.load_state_dict(st["model"])
        self.l_max = self.model.l_max
        cfg_max = int(Cfg(st["cfg"]).data.n_patches)
        self.max_len = int(max_length) if max_length else cfg_max
        self.tok_dir = tok_dir
        self.verify = (
            VerifyResplitter(self.ae, tok_dir, ent_thresh=0.2, floor_bytes=3,
                             device=device, log_every=0)
            if verify else None
        )
        print(f"[bltz-eval] ckpt={pretrained} step={st.get('step')} "
              f"d_model={Cfg(st['cfg']).model.d_model} S={self.max_len} "
              f"verify={verify is not None} dev={device} "
              f"chat_template={self.chat_template} "
              f"max_gen_units={self.max_gen_units}", flush=True)

    # ---- chat 模板(SFT 口径;docs/48 §2.1) --------------------------------
    def _chat_prompt(self, context: str) -> str:
        """`<|user|>{context}<|assistant|>`(轮间无分隔符;默认关闭时不动)。"""
        return f"{CHAT_USER}{context}{CHAT_ASSISTANT}"

    # ---- 切词(训练同管线: segment_word + 在线核验重切) -------------------
    def _segment(self, text: str) -> list[bytes]:
        units = segment_word(text.encode("utf-8"), self.tok_dir, l_max=self.l_max)
        if self.verify is not None:
            units = self.verify.process(units)
        return [u[: self.l_max] for u in units]

    # ---- ctx/cont 切分(docs/49 §3.2: joint unitize + 字节边界归属) ---------
    def _split_units(self, context: str, continuation: str) -> tuple[list[bytes], list[bytes]]:
        """把 (context, continuation) 切成 (ctx_units, cont_units),只对后者计分。

        对齐标准 causal LM 的 `_encode_pair` 约定 (lm_eval/api/model.py):
        - context 末尾空白先移到 continuation(保持词边界);
        - 整段 ctx+cont 一起 unitize(训练同分布),按**字节边界**归属:
          首字节 < boundary 的 unit 归 context(跨缝 unit 含分隔符空格 → context,
          不计分);首字节 >= boundary 的 unit 才是候选词项。
        - 兜底:若跨缝 unit 吞掉了候选(无分隔符等异常),退化到候选单独切词,
          ctx 单独切词,然后拼接,并打印 warning。
        """
        ctx = context
        trail = ctx[len(ctx.rstrip()):] if ctx else ""
        if trail:
            ctx = ctx[: len(ctx) - len(trail)]
            cont = trail + continuation
        else:
            cont = continuation
        full = self._segment(ctx + cont)
        bnd = len(ctx.encode("utf-8"))
        split = len(full)
        acc = 0
        for k, u in enumerate(full):
            if acc >= bnd:
                split = k
                break
            acc += len(u)
        cont_units = full[split:]
        expected = len(self._segment(cont.lstrip())) if cont.strip() else 0
        if not cont_units or len(cont_units) != expected:
            # 异常(候选被跨缝 unit 吞掉/切分数不一致): 兜底协议
            print(f"[bltz-eval][warn] fallback separate unitize: ctx={ctx[-40:]!r} "
                  f"cont={cont[-40:]!r} joint={len(cont_units)} expected={expected}",
                  flush=True)
            cont_units = self._segment(cont.lstrip()) if cont.strip() else []
            full = self._segment(ctx) + cont_units
            split = len(full) - len(cont_units)
        return full[:split], full[split:]

    # ---- 核心: unit 链逐位 NLL (nats; 位置 i = p(u_i | u_<i), u_<0=BOS) ---
    @torch.no_grad()
    def _chain_nll(self, units: list[bytes]) -> torch.Tensor:
        bids, lens, pm = tensorize(units, self.l_max, self.dev)
        lam = self.model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
        h = self.model.backbone(
            self.model.backbone_input(bids.unsqueeze(0), pm.unsqueeze(0)))
        return self.model.head.nll(h, lam)[0].float()  # (S,)

    @torch.no_grad()
    def _greedy_hits(self, units: list[bytes], start: int) -> bool:
        """is_greedy: continuation(自 start 起)每个 unit 都等于密度泛函贪婪读数
        (GMM mode -> AE decode)。教师强迫逐位判定, 与 HF 路径的逐 token argmax
        语义对齐。"""
        bids, lens, pm = tensorize(units, self.l_max, self.dev)
        h = self.model.backbone(
            self.model.backbone_input(bids.unsqueeze(0), pm.unsqueeze(0)))
        modes = self.model.head.mode(h[0, start:])          # (P, d_emb)
        dec = self.ae.decode(modes)
        for u, d in zip(units[start:], dec):
            if bytes(d) != u:
                return False
        return True

    # ---- LM API ------------------------------------------------------------
    @torch.no_grad()
    def _ll_one(self, context: str, continuation: str) -> tuple[float, bool, dict]:
        """单条 (ctx, cont) -> (S_raw nats, is_greedy, meta)。

        docs/49 §2-§3: 只对候选词项求 Σ log p(λ(u_i) | 前文);归一化指标
        (acc_unit/acc_char/PMI)在 scripts/mc_eval.py 计算,这里保持 LM API
        的 raw 语义(harness 原生 acc 即 acc_raw,可交叉验证)。
        """
        ctx_units, cont_units = self._split_units(context, continuation)
        # context 左截断 / continuation 右截断: 总长 <= max_len, 保住候选
        keep = self.max_len - len(cont_units)
        if keep < 0:
            cont_units = cont_units[: self.max_len]
            keep = 0
        ctx_units = ctx_units[-keep:] if keep else []
        units = ctx_units + cont_units
        if not cont_units:
            return float("-inf"), False, {"n_units": 0}
        nll = self._chain_nll(units)
        ll = -float(nll[len(ctx_units):].sum())   # 式(3): 只对候选项求和
        greedy = self._greedy_hits(units, len(ctx_units))
        return ll, greedy, {"n_units": len(cont_units)}

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        out: list[tuple[float, bool]] = []
        if self.collect_meta:
            self.last_meta = []
        t0 = time.time()
        for i, req in enumerate(requests):
            context, continuation = req.args
            if self.chat_template:  # SFT 口径:只包 context,continuation 照旧
                context = self._chat_prompt(context)
            ll, greedy, meta = self._ll_one(context, continuation)
            out.append((ll, greedy))
            if self.collect_meta:
                self.last_meta.append(meta)
            if (i + 1) % self.heartbeat == 0 or i + 1 == len(requests):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                eta = (len(requests) - i - 1) / rate / 60
                print(f"[bltz-eval] loglikelihood {i+1}/{len(requests)} "
                      f"({rate:.2f} req/s, ETA {eta:.1f} min)", flush=True)
        return out

    @torch.no_grad()
    def unconditional_loglikelihood(self, units: list[bytes]) -> float:
        """PMI 用的 S_uncond(docs/49 §4.3):同一候选 unit 序列在空 context
        (仅 BOS)下的 Σ log p(λ(u_i) | u_<i)。"""
        units = units[: self.max_len]
        if not units:
            return float("-inf")
        nll = self._chain_nll(units)
        return -float(nll.sum())

    def loglikelihood_rolling(self, requests) -> list[float]:
        out: list[float] = []
        t0 = time.time()
        S = self.max_len
        for i, req in enumerate(requests):
            (text,) = req.args
            units = self._segment(text)
            if not units:
                out.append(0.0)
                continue
            ll = 0.0
            pos = 0
            while pos < len(units):
                if pos == 0:
                    win = units[:S]
                    nll = self._chain_nll(win)
                    ll -= float(nll.sum())
                    pos = len(win)
                else:
                    ctx_n = S // 2
                    block = units[pos : pos + (S - ctx_n)]
                    win = units[pos - ctx_n : pos] + block
                    nll = self._chain_nll(win)
                    ll -= float(nll[ctx_n:].sum())
                    pos += len(block)
            out.append(ll)
            if (i + 1) % self.heartbeat == 0 or i + 1 == len(requests):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                eta = (len(requests) - i - 1) / rate / 60
                print(f"[bltz-eval] rolling {i+1}/{len(requests)} "
                      f"({rate:.2f} req/s, ETA {eta:.1f} min)", flush=True)
        return out

    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        out: list[str] = []
        t0 = time.time()
        for i, req in enumerate(requests):
            context, gen_kwargs = req.args
            until = list(gen_kwargs.get("until", []) or [])
            if self.chat_template:  # SFT 口径:包 prompt + 停止标记
                context = self._chat_prompt(context)
                if CHAT_END not in until:
                    until.append(CHAT_END)
            # 粗换算见头注;上限 --max-gen-units(适配器逐 unit 重算 context,O(n^2))
            max_units = min(int(gen_kwargs.get("max_gen_toks", 256)),
                            self.max_gen_units)
            tau = float(gen_kwargs.get("temperature", self.temperature))
            do_sample = bool(gen_kwargs.get("do_sample", True))
            units = self._segment(context)[-self.max_len :]
            texts: list[str] = []
            stopped = False
            for _ in range(max_units):
                win = units[-self.max_len :]
                bids, lens, pm = tensorize(win, self.l_max, self.dev)
                lam = self.model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
                x = torch.cat([self.model.bos.expand(1, 1, -1), lam], dim=1)
                h = self.model.backbone(self.model.adapter(x))[:, -1:]
                if not torch.isfinite(h).all():
                    print(f"[bltz-eval] gen req {i}: non-finite h at step "
                          f"{len(texts)}; diverged trajectory stopped (10-09 守卫)",
                          flush=True)
                    stopped = True
                    break
                modes, _log_mass, log_dens = self.model.head.modes(h[0, 0:])
                if not torch.isfinite(log_dens).all():
                    print(f"[bltz-eval] gen req {i}: non-finite log_dens; stop",
                          flush=True)
                    stopped = True
                    break
                w = log_dens / max(tau, 1e-6)  # (M,) 峰顶海拔加权
                if not do_sample or tau <= 0:
                    j = int(w.argmax())
                else:
                    # 核采样 0.95(docs/39 #36: 峰顶海拔加权 + nucleus)
                    p = F.softmax(w, dim=-1)
                    ps, idx = p.sort(descending=True)
                    keep = int(torch.searchsorted(torch.cumsum(ps, 0),
                                                  torch.tensor(0.95, device=self.dev),
                                                  right=True).clamp(min=1))
                    top_p = (ps[:keep] / ps[:keep].sum())
                    j = int(idx[torch.multinomial(top_p, 1)])
                dec = bytes(self.ae.decode(modes[j : j + 1])[0])
                if not dec:
                    stopped = True  # 空解码视作 EOS(word 流无 stream 级 EOS)
                    break
                s = dec.decode("utf-8", errors="replace")
                joined = "".join(texts) + s
                hit = next((u for u in until if u in joined), None)
                if hit is not None:
                    joined = joined[: joined.index(hit)]
                    texts = [joined]  # 截到 stop 序列前
                    stopped = True
                    break
                texts.append(s)
                units.append(dec)
            out.append("".join(texts))
            mark = " stop" if stopped else " maxlen"
            if (i + 1) % self.heartbeat == 0 or i + 1 == len(requests):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                eta = (len(requests) - i - 1) / rate / 60
                print(f"[bltz-eval] generate {i+1}/{len(requests)}{mark} "
                      f"({rate:.2f} req/s, ETA {eta:.1f} min)", flush=True)
        return out

    # ---- harness 需要的几个常规属性 -----------------------------------------
    @property
    def batch_size(self) -> int:
        return 1  # 请求串行处理(本机 8GB);harness 只用它做分块

    @property
    def device(self) -> str:
        return self.dev

    @property
    def max_length(self) -> int:
        return self.max_len

    @property
    def eot_token_id(self) -> int:
        return -1  # word 流无 token 表;generate_until 用 stop 序列/空解码

    @property
    def tokenizer_name(self) -> str:
        return "bltz-word-units"


def _selftest(args) -> None:
    lm = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                    device=args.device, verify=not args.noverify,
                    chat_template=args.chat_template,
                    max_gen_units=args.max_gen_units)

    class _Req:
        def __init__(self, *a):
            self.args = a

    print("\n== selftest: loglikelihood(unit-space native) ==", flush=True)
    lm.collect_meta = True
    pairs = [
        ("The capital of France is", " Paris"),
        ("The capital of France is", " London"),
        ("The cat sat on the", " mat"),
        ("The cat sat on the", " quantum"),
        ("", "Hello world"),
    ]
    for ctx, cont in pairs:
        ll, greedy = lm.loglikelihood([_Req(ctx, cont)])[0]
        n_units = lm.last_meta[-1]["n_units"]
        n_chars = max(len(cont.lstrip()), 1)
        print(f"  ctx={ctx!r:38s} cont={cont!r:12s} S_raw={ll:9.3f} nats | "
              f"n_units={n_units} S_unit={ll/max(n_units,1):8.3f} | "
              f"S_char={ll/n_chars:8.3f} | greedy={greedy}", flush=True)

    print("\n== selftest: loglikelihood_rolling ==", flush=True)
    text = "The quick brown fox jumps over the lazy dog. " * 30
    ll = lm.loglikelihood_rolling([_Req(text,)])[0]
    # 注意: 返回 unit 级 GMM 密度的 log-likelihood(latent nats, 可为正),
    # 不是 byte CE/bpb; pushforward 字节 bpb 口径已于 2026-10-27 退役(docs/49)。
    print(f"  rolling ll={ll:.2f} nats | {-ll / len(text.encode()):.3f} "
          f"latent-nats/byte(诊断,非 bpb)", flush=True)

    print("\n== selftest: unconditional (PMI 用) ==", flush=True)
    cu, vu = lm._split_units("The capital of France is", " Paris")
    print(f"  cont units={vu} | S_uncond={lm.unconditional_loglikelihood(vu):.3f}",
          flush=True)

    print("\n== selftest: generate_until ==", flush=True)
    g = lm.generate_until([_Req("The capital of France is",
                                {"until": ["\n", "."], "max_gen_toks": 24,
                                 "temperature": 0.8})])[0]
    print(f"  generated: {g!r}", flush=True)

    # ---- chat 模板机制(2026-10-26;SFT Tier 2 口径)-----------------------
    print("\n== selftest: chat template ==", flush=True)
    lm_chat = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                         device=args.device, verify=not args.noverify,
                         chat_template=True, max_gen_units=24)
    prompt = "Give one short sentence about Paris."
    print(f"  wrapped prompt: {lm_chat._chat_prompt(prompt)!r}", flush=True)
    g = lm_chat.generate_until([_Req(prompt, {"until": ["\n"], "max_gen_toks": 64,
                                              "temperature": 0.8})])[0]
    print(f"  generated: {g!r} (chat mode appends {CHAT_END!r} to stop list)",
          flush=True)
    ll, greedy = lm_chat.loglikelihood([_Req("What is the capital of France?",
                                            " Paris")])[0]
    print(f"  chat loglikelihood: ll={ll:9.3f} nats | greedy={greedy}", flush=True)
    print("\n[selftest] DONE", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console safe
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/word_05b_p2_last.pt")
    ap.add_argument("--ae", default="checkpoints/ae_48_256_word_full/best.pt")
    ap.add_argument("--tok", default="data/qwen35_tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noverify", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chat-template", action="store_true",
                    help="SFT 口径:context 包成 <|user|>…<|assistant|>,"
                         "生成自动加 <|end|> stop(docs/48 §2.1)")
    ap.add_argument("--max-gen-units", type=int, default=256,
                    help="generate_until 的 unit 数上限(默认 256;适配器逐 "
                         "unit 重算 context,O(n^2),大 max_gen_toks 必须截断)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tasks", default="hellaswag")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    if args.selftest:
        _selftest(args)
        return

    from lm_eval import simple_evaluate
    lm = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                    device=args.device, verify=not args.noverify,
                    temperature=args.temperature,
                    chat_template=args.chat_template,
                    max_gen_units=args.max_gen_units)
    t0 = time.time()
    res = simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        log_samples=False,
    )
    print(f"\n[results] ({time.time()-t0:.0f}s)")
    print(json.dumps(res["results"], indent=1, ensure_ascii=False))
    if "samples" in res:
        pass


if __name__ == "__main__":
    main()
