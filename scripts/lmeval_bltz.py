"""bltz × lm-evaluation-harness adapter(docs/42 C 报告 §3.1 / docs/43 拍板⑧,2026-10-18).

模型: bltz 0.5B word 臂(冻结 AE ae_48_256_word_full + 骨干 + linear GMM 头)。
读出硬性规则(AGENTS.md): 一律密度泛函(GMM mode / modes-peak 采样),严禁把
分量均值当词用;生成循环带 isfinite 守卫(2026-10-09 p2 事故教训)。

口径(与训练/held-out NLL 同构, docs/38):
- 文本 -> utf-8 -> bltz.segment_v2.segment_word(word 臂确定性切词,
  merge="prev" 默认) -> VerifyResplitter 在线核验重切(训练同管线,
  pushforward word 臂同款; 可用 --noverify 关);
- loglikelihood(context, continuation): context 与 continuation 分别切词后
  拼接 unit 链(与 harness HF 路径同约定——接缝不跨 unit),continuation 的
  每个 unit 的 NLL 以全部前序(含完整 context)为条件;context 超长时左截断
  保住 continuation。返回总 logprob(nats)——harness 的 acc_norm 按
  continuation 字节数归一,与本项目 bpb 口径天然同构(C §3.1),归一化由
  harness 完成,我们不做任何额外处理。
- loglikelihood_rolling: 滑窗整文 NLL(S=4096 units;首窗全计,后续窗重叠
  半窗——每个 unit 恰好计一次且 context ≥ S/2;比 harness 默认 context_len=1
  更宽松,属有文档的口径选择)。bpb = -logprob / n_bytes / ln2。
- generate_until: modes-peak 采样(modes() 枚举局部众数 + 峰顶海拔
  log_dens 加权 + 核采样 0.95,docs/39 #36 翻案)+ isfinite 守卫(发散即停
  该请求并记录,不伪造样本)+ stop 序列 + max_gen_toks 粗换算
  (1 unit ≈ 5 字节 ≈ 1 常规 token,max_units = max_gen_toks,并上限
  `--max-gen-units`,默认 256——适配器逐 unit 重算 context,长生成是 O(n²),
  IFEval 类任务必须截断预算)。unit 解码为空串视作 EOS 停止(word 流无
  stream 级 EOS 单元)。

- **chat 模板(2026-10-26 新增,--chat-template;为 SFT 后 Tier 2 备)**:
  生成侧把 harness context 包成 `<|user|>{context}<|assistant|>`,并自动追加
  stop 序列 `<|end|>`;loglikelihood(MMLU 等)只包 context、continuation 照旧
  计分(接缝不跨 unit 的约定不变)。**默认关闭** → 与 2026-10-18 行为逐位
  一致(pre-training 口径基准数字可直接复用)。模板与 SFT 训练一致
  (docs/48 §2.1:轮间无分隔符,target=assistant 内容+`<|end|>`)。

口径发现(2026-10-18 本机验证,重要): unit 级 GMM 密度 NLL 的绝对值随
unit 字节长度变化极弱(NLL 约与 unit 数成正比),而 harness acc_norm 按
continuation 字符数归一——二者相乘系统性偏向长候选(长词 latent 更尖锐,
总 NLL 高但摊到更多字符)。合成电池实测: 原始 total(acc 口径)4/5 金标
正确,acc_norm 口径只剩 1/5。因此本适配器提供两种 score_mode:
- "unit_nll"(默认,拍板处方): Σ_continuation unit 级 GMM NLL;多选题请读
  **acc(未归一)**,acc_norm 在该口径下有系统性长度偏置(见上)。
- "byte_marginal": 解码器 pushforward 字节边际(μ-plug 版 pushforward_bpb
  口径): P(b|ctx,Δ) = Σ_k softmax(π)_k · P_dec(b|μ_k,Δ),continuation 得分
  = Σ unit Σ 字节 log P_marg(不含 EOS,pushforward v3 口径)。按字节计分,
  acc/acc_norm 一致且与 byteCE/bpb 协议同族。代价: 每位置 K×(L+1) 次解码器
  查询(并行 FiLM 前向)。**局限(合成电池实测): 并行解码器在 unit 内逐 Δ
  独立取边际,丢掉 unit 内字节相关性,长候选被错误拉平。**
- "pushforward_joint"(多选题推荐): 精确字符串概率的层化 MC —
  P(u|ctx) = Σ_k π_k · mean_m Π_Δ P_dec(u[Δ]|μ_k+σ_k ε_m, Δ)(同一 λ 跨 Δ
  共享,保留 unit 内联合;含 EOS 终止事件——终止是字符串身份的一部分)。
  m=8×K=64 每位置 512 条 λ,定数种子;短 continuation 无 underflow
  (pushforward 文档的 e-40 地板只出现在长串)。acc/acc_norm 均可用。

本机用法(8GB GPU, batch 1 串行):
  # 自带入口(推荐; 内部注册 bltz 模型并调 simple_evaluate):
  E:\\MiniConda\\envs\\cose2\\python.exe scripts/lmeval_bltz.py \
      --tasks hellaswag --limit 32
  # 冒烟自检(不跑 harness, 只验证加载/切词/NLL/生成):
  E:\\MiniConda\\envs\\cose2\\python.exe scripts/lmeval_bltz.py --selftest
  # 作为 lm_eval CLI 的自定义模型(需把本文件加入 harness 进程 import 链,
  # 例如 PYTHONPATH=scripts 后 --model lmeval_bltz:BltzEvalLM 不适用本版
  # 注册时机; 推荐用上面的自带入口):
  python -m lm_eval --model bltz --model_args pretrained=<ckpt> --tasks hellaswag

集群用法(验收时): 同自带入口, ckpt 换 $SCRATCH 路径, batch_size 可开 4-8。
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
from bltz.models.autoencoder import EOS_ID, ByteStringAE
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
LOG2E = 1.4426950408889634


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
        score_mode: str = "unit_nll",  # "unit_nll" | "byte_marginal" | "pushforward_joint"
        heartbeat: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.dev = device
        self.temperature = float(temperature)
        self.chat_template = bool(chat_template)
        self.max_gen_units = max(1, int(max_gen_units))
        self.heartbeat = max(1, int(heartbeat))

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
        if score_mode not in ("unit_nll", "byte_marginal", "pushforward_joint"):
            raise ValueError(f"unknown score_mode {score_mode!r}")
        self.score_mode = score_mode
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

    # ---- 字节边际 pushforward(μ-plug; 与 pushforward_bpb 的 marginal 同族) --
    @torch.no_grad()
    def _chain_byte_logmarg(self, units: list[bytes]) -> torch.Tensor:
        """-> (S, l_max+1): 位置 i、Δ=d+1 处 P_byte(b|ctx,Δ) 的 log(π 加权、
        对 μ_k 逐分量解码后取边际; 不含 EOS, pushforward v3 口径)。"""
        bids, lens, pm = tensorize(units, self.l_max, self.dev)
        h = self.model.backbone(
            self.model.backbone_input(bids.unsqueeze(0), pm.unsqueeze(0)))
        logit_pi, mu, _sig = self.model.head.params(h)   # (1,S,K),(1,S,K,D)
        S = len(units)
        pi = F.log_softmax(logit_pi[0].float(), -1)      # (S,K)
        mu = mu[0].float()                               # (S,K,D)
        K, D = mu.shape[1], mu.shape[2]
        out = torch.full((S, self.l_max + 1), float("-inf"), device=self.dev)
        for i in range(S):
            L = min(len(units[i]), self.l_max)
            if L == 0:
                continue
            # 该位置全部 K 分量 × Δ=1..L 的并行解码器查询(一次 FiLM 前向)
            mu_q = mu[i].unsqueeze(1).expand(K, L, D).reshape(K * L, D)
            d_idx = torch.arange(L, device=self.dev).repeat(K)  # Δ 下标=d+1
            logp = F.log_softmax(self.ae.decoder(mu_q, d_idx), dim=-1)
            marg = torch.logsumexp(pi[i][:, None, None]
                                   + logp.view(K, L, 257), dim=0)  # (L,257)
            tgt = torch.tensor(list(units[i][:L]), device=self.dev)
            out[i, :L] = marg.gather(-1, tgt[:, None]).squeeze(-1)
        return out

    # ---- 字符串精确概率(层化 MC pushforward joint; 同一 λ 跨 Δ 共享) --------
    PF_MC = 8  # 每分量 λ 样本数(pushforward_bpb 同值; 每位置 K×8=512 条)

    @torch.no_grad()
    def _chain_pf_joint(self, units: list[bytes]) -> torch.Tensor:
        """-> (S,): 每位置 unit 的精确字符串 log P(u_i | ctx)(解码器
        pushforward; bytes + EOS 终止; 层化 MC: 每分量 m 条 μ_k+σ_k·ε)。"""
        M = self.PF_MC
        bids, lens, pm = tensorize(units, self.l_max, self.dev)
        h = self.model.backbone(
            self.model.backbone_input(bids.unsqueeze(0), pm.unsqueeze(0)))
        logit_pi, mu, sig = (t.float() for t in self.model.head.params(h))
        S = len(units)
        K, D = mu.shape[-2], mu.shape[-1]
        pi = F.softmax(logit_pi[0], -1)                # (S,K)
        mu, sig = mu[0], sig[0]                        # (S,K,D)
        gen = torch.Generator(device=self.dev).manual_seed(20261018)
        out = torch.zeros(S, dtype=torch.float64, device=self.dev)
        for i in range(S):
            L = min(len(units[i]), self.l_max)
            if L == 0:
                continue
            eps = torch.randn(M, K, D, generator=gen, device=self.dev)
            lam = mu[i].unsqueeze(0) + sig[i].unsqueeze(0) * eps   # (M,K,D)
            lam_f = lam.reshape(M * K, D)
            d_idx = torch.arange(L + 1, device=self.dev).repeat(M * K)
            mu_q = lam_f.repeat_interleave(L + 1, dim=0)
            logp = F.log_softmax(self.ae.decoder(mu_q, d_idx), dim=-1)
            tgt = torch.tensor(list(units[i][:L]) + [EOS_ID], device=self.dev)
            tgt = tgt.repeat(M * K)
            lp = logp.gather(-1, tgt[:, None]).squeeze(-1).view(M * K, L + 1)
            joint = lp.sum(-1).view(M, K).double()                 # (M,K)
            # P = Σ_k π_k · mean_m exp(joint); log 域数值稳定化
            jmax = joint.max(dim=0).values                          # (K,)
            p_hat = (pi[i].double()
                     * (joint - jmax.unsqueeze(0)).exp().mean(0)    # mean over M
                     * jmax.exp()).sum()
            out[i] = torch.log(p_hat.clamp_min(1e-300))
        return out

    # ---- LM API ------------------------------------------------------------
    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        out: list[tuple[float, bool]] = []
        t0 = time.time()
        for i, req in enumerate(requests):
            context, continuation = req.args
            if self.chat_template:  # SFT 口径:只包 context,continuation 照旧
                context = self._chat_prompt(context)
            # 联合切词(与训练流同分布): context+continuation 拼成整段再
            # segment_word,避免分别切词在接缝处造出 lone-space 等分布外 unit
            # (实测: " Paris" 单切出 [' ','Paris'], 孤空格 unit NLL +64.8,
            #  并污染后续; 联合切词得 ['France ', 'is ', 'Paris'] 正常水位)。
            # 边界归属: 跨缝 unit(同时含 context/continuation 字节)归
            # context 不计分——continuation 计分 unit = 首字节落在
            # continuation 内的所有 unit(其 NLL 以完整 context 为条件)。
            full_units = self._segment(context + continuation)
            bnd = len(context.encode("utf-8"))
            split = len(full_units)
            acc = 0
            for k, u in enumerate(full_units):
                if acc >= bnd:
                    split = k
                    break
                acc += len(u)
            else:
                split = len(full_units)
            cont_units = full_units[split:]
            if not cont_units:
                # continuation 字节全被跨缝 unit 吸收(极短 continuation);
                # 退化为计分最后一个 unit(否则零长答案, harness 不容)
                split = max(0, len(full_units) - 1)
                cont_units = full_units[split:]
            ctx_units = full_units[:split]
            # context 左截断: 总长 <= max_len, 保住 continuation
            keep = self.max_len - len(cont_units)
            if keep < 0:  # continuation 自身超长: 右截断其尾部(与 harness 截断
                cont_units = cont_units[: self.max_len]  # 策略一致的方向)
                keep = 0
            ctx_units = ctx_units[-keep:] if keep else []
            units = ctx_units + cont_units
            if self.score_mode == "byte_marginal":
                bm = self._chain_byte_logmarg(units)     # (S, l_max+1) log P
                ll = 0.0
                for i in range(len(ctx_units), len(units)):
                    L = min(len(units[i]), self.l_max)
                    ll += float(bm[i, :L].sum())
            elif self.score_mode == "pushforward_joint":
                pf = self._chain_pf_joint(units)         # (S,) log P(u|ctx)
                ll = float(pf[len(ctx_units):].sum())
            else:
                nll = self._chain_nll(units)
                ll = -float(nll[len(ctx_units):].sum())
            greedy = self._greedy_hits(units, len(ctx_units))
            out.append((ll, greedy))
            if (i + 1) % self.heartbeat == 0 or i + 1 == len(requests):
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                eta = (len(requests) - i - 1) / rate / 60
                print(f"[bltz-eval] loglikelihood {i+1}/{len(requests)} "
                      f"({rate:.2f} req/s, ETA {eta:.1f} min)", flush=True)
        return out

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
                    score_mode=args.score_mode,
                    chat_template=args.chat_template,
                    max_gen_units=args.max_gen_units)

    class _Req:
        def __init__(self, *a):
            self.args = a

    print("\n== selftest: loglikelihood ==", flush=True)
    pairs = [
        ("The capital of France is", " Paris"),
        ("The capital of France is", " London"),
        ("The cat sat on the", " mat"),
        ("The cat sat on the", " quantum"),
        ("", "Hello world"),
    ]
    for ctx, cont in pairs:
        ll, greedy = lm.loglikelihood([_Req(ctx, cont)])[0]
        n_bytes = len(cont.encode("utf-8"))
        print(f"  ctx={ctx!r:38s} cont={cont!r:12s} ll={ll:9.3f} nats | "
              f"bpb={-ll * LOG2E / max(n_bytes, 1):.3f} | greedy={greedy}", flush=True)

    print("\n== selftest: loglikelihood_rolling ==", flush=True)
    text = "The quick brown fox jumps over the lazy dog. " * 30
    ll = lm.loglikelihood_rolling([_Req(text,)])[0]
    # 注意: 返回的是 unit 级 GMM 密度 log-likelihood(latent nats, 可为正),
    # 不是 byte CE; 项目的 bpb 口径在 pushforward_bpb.py(解码器推出蒙特卡洛)。
    print(f"  rolling ll={ll:.2f} nats | {-ll / len(text.encode()):.3f} "
          f"latent-nats/byte(非 bpb)", flush=True)

    print("\n== selftest: generate_until ==", flush=True)
    g = lm.generate_until([_Req("The capital of France is",
                                {"until": ["\n", "."], "max_gen_toks": 24,
                                 "temperature": 0.8})])[0]
    print(f"  generated: {g!r}", flush=True)

    # ---- chat 模板机制(2026-10-26;SFT Tier 2 口径)-----------------------
    print("\n== selftest: chat template ==", flush=True)
    lm_chat = BltzEvalLM(pretrained=args.ckpt, ae=args.ae, tok_dir=args.tok,
                         device=args.device, verify=not args.noverify,
                         score_mode=args.score_mode,
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
    ap.add_argument("--score-mode", default="unit_nll",
                    choices=["unit_nll", "byte_marginal", "pushforward_joint"])
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
                    temperature=args.temperature, score_mode=args.score_mode,
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
