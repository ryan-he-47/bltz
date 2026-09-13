> **定位**:调研档案(2026-09-13 归档)。来源:本会话的外部先例检索(librarian 报告 + 二次精炼)合并整理稿。
> **内容**:多 token 预测(MTP)/多视界损失的梯度冲突先例;多任务学习"负余弦是否单独构成有害证据"的文献判定;头设计对照(独立头 vs 共享头);引用表与检索缺口。
> **说明**:引文均为检索报告转录(原始论文 HTML 提取),标注 **[R]** 者为检索报告的转述/推断而非论文原文;发表前请核对原文。原报告为临时工件,本档为长期留存版。
> **关联**:配合 `docs/11-gcos-measure-fix.md`(施工单)与 `docs/12-gcos-second-opinion.md`(本案例第二意见)阅读。

# Prior-Art Dossier: Gradient Conflict in Multi-Token Prediction (MTP) and Multi-Horizon Losses

**Provenance.** Consolidated from two session files:
- **File 1** = `tool_099d0a605001GukxBnUx4RMSxC` (194 KB, 1827 lines): full prior-art research transcript. Final report at lines 1743–1827; verbatim quote inventory at lines 1605–1636; TL;DR at lines 1556–1601.
- **File 2** = `tool_099d586db001p4T09WuwQTtwga` (80 KB, 795 lines): mining report that distilled File 1; structured answer at lines 612–795.

**Fidelity note.** Quotes are reproduced as recorded in the files (agent-extracted from paper HTML; File 1 line 1603 notes "minor spacing artifacts; I'll clean lightly but keep wording"). Ellipses inside quotes are as recorded. **`[R]` marks the report author's own statement/paraphrase, not a paper quote.** No outside knowledge is added except in the clearly marked "Citation flags" in Section 4 and item 12 of Section 5.

---

## 1. MTL literature: is a negative gradient cosine between task losses, by itself, diagnostic of harm?

**Consolidated answer as found: no.** Every primary source in the files treats negative cosine as insufficient on its own; harm depends on additional conditions (curvature, magnitude imbalance, conditioning), and cosine-based decisions have a documented counterexample.

**PCGrad (Yu et al.) — canonical definition and its own 3-condition hypothesis**
- Citation: *Gradient Surgery for Multi-Task Learning*, arXiv:2001.06782, 2020 (venue not stated in files).
- Verbatim: "We define two gradients to be conflicting if they point away from one another, i.e., have a negative cosine similarity. We hypothesize that such conflict is detrimental when a) conflicting gradients coincide with b) high positive curvature and c) a large difference in gradient magnitudes." [File 1 lines 644, 1626, 1785]
- Meaning as found [R]: negative cosine is condition (a) only; PCGrad itself does not treat it as diagnostic alone [File 1 line 644].

**PCGrad — the pathological cos = −1 edge case, which they say they never observe**
- Verbatim: "in practice, since we are using SGD ... the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be −1, thus avoiding this scenario" [File 1 line 1627]
- Verbatim: "convergence may be slow if cos(φ12) hovers near −1. However, we don't observe this in practice." [File 1 line 1627]
- Final report's elided variant: "A sub-optimal solution occurs when the cosine similarity … is exactly −1 … However, in practice, since we are using SGD … the cosine similarity between the gradients of two tasks in a minibatch is unlikely to be −1" [File 1 line 1788]

**PCGrad — they measured the frequency of positive-cosine steps (Figure 4, RL tasks)**
- Verbatim: "The solid lines show the percentage gradients with positive cosine similarity between two task gradients" [File 1 lines 1628, 1789]
- Verbatim: "condition (a) holds most of the time for both Adam and Adam combined with PCGrad when they haven't solved Task 2" [File 1 line 1628]
- Verbatim: "we empirically validate that the first three points, (i-iii), are frequently met in a neural network multi-task learning problem in Figure 4" [File 1 line 1628]
- Partial caption continuation captured mid-transcript (extraction cut off): "the percentage gradients with positive cosine similarity between two task gradients, while the dotted lines and dashed lines show the percentage of iterations in which condition (a) and the implication of condition (b)..." [File 1 line 1368]
- Meaning as found [R]: negative-cosine steps are common and not automatically fatal; no numeric percentage is recorded in the files [File 1 line 1789].

**Aligned-MTL (Senushkin et al.) — condition-number critique**
- Citation: *Independent Component Alignment for Multi-Task Learning*, CVPR 2023, arXiv:2305.19000, DOI 10.1109/cvpr52729.2023.01923. (Early file notes mis-ID it as arXiv:2306.07180; the report itself corrects this at File 1 line 732.)
- Verbatim: "Similarly, gradient conflicts can be estimated as a cosine distance between vectors (PCGrad Def. 1, CAGrad). However, each of these metrics describes a specific characteristic of a linear system of gradients, and cannot provide a comprehensive assessment if taken separately." [File 1 line 1710; shorter fragment at line 1631]
- Verbatim (abstract): "we propose using a condition number of a linear system of gradients as a stability criterion" [File 1 line 1631]
- Report paraphrase [R, not a quote]: "the stability of MTL training is determined not by gradient conflict but by the condition number of the gradient matrix"; they "use the condition number of the gradient matrix as the stability criterion (conflict + dominance together)" [File 1 lines 243, 1793].

**Fifty et al. (TAG) — proven counterexample where cosine-based grouping is inferior**
- Citation: *Efficiently Identifying Task Groupings for Multi-Task Learning*, NeurIPS 2021, arXiv:2109.04617.
- Verbatim: "A counterexample on a quadratic loss function where the task grouping based on inter-task similarity results in an inferior performance." [File 1 lines 1632, 1796]
- They benchmark "grouping tasks by maximizing inter-task cosine similarity between pairs of gradients (CS)" against their actual-effect method (TAG) [File 1 lines 1632, 1796]. Practical implication [R]: cosine clustering is a heuristic, not ground truth [File 1 line 1796].

**Sener & Koltun — weighted-sum validity and the MGDA common-descent test**
- Citation: *Multi-Task Learning as Multi-Objective Optimization*, arXiv:1810.04650, 2018 (venue: File 1 line 1496 says ECCV 2018; see Citation flags).
- Verbatim: "minimization of a weighted sum of empirical risk is only valid if tasks are not competing, which is rarely the case." [File 1 line 1629]
- Verbatim: "MTL with conflicting objectives requires modeling of the trade-off between tasks". [File 1 line 1629]
- Report's use [R]: the common-descent-direction test (MGDA) is highlighted as the decisive diagnostic — if the min-norm point in the convex hull of gradients is nonzero, a Pareto-improving direction exists even under conflict [File 1 lines 1216, 1526, 1718].

**CAGrad (Liu et al.)**
- Citation: *Conflict-Averse Gradient Descent for Multi-task Learning*, arXiv:2110.14048, 2021 (venue not stated in files).
- Verbatim: "A major challenge in optimizing a multi-task model is the conflicting gradients, where gradients of different task objectives are not well aligned so that following the average gradient direction can be detrimental to specific tasks' performance." [File 1 line 1630]
- Final report also quotes the fragment "following the average gradient direction can be detrimental to specific tasks' performance" [File 1 line 1798].

**FairGrad / Fair Resource Allocation (Ban & Ji) — status: cited, NO verbatim quote captured**
- Citation in files: *Fair Resource Allocation in Multi-Task Learning*, ICML 2024, arXiv:2402.15638 [File 1 lines 242, 671, 1497, 1658, 1798].
- **No verbatim FairGrad quote exists in either file.** Closest content is a report paraphrase: "Also 'FairGrad' abstract mentions conflicting gradients hinder fair optimization; fine as background." [R/paraphrase, File 1 line 1224] and report speculation that it "may show that 'conflict' measured by cosine is scale-invariant and that conflicts can be beneficial" [R/speculation, File 1 line 862].

**GradNorm — status: cited only, no quote**
- Citation in files: *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, arXiv:1711.02257, 2018 (Chen et al.; venue not stated in files). Report states it was not fetched ("I didn't fetch, but known") [File 1 line 1492]; report summary places it in the conflict toolbox for magnitude rebalancing [R, File 1 line 1544].

**Scale-invariance / gradient-magnitude caveats**
- PCGrad's condition (c), "a large difference in gradient magnitudes", is the in-quote magnitude caveat [File 1 line 1626].
- Report's own analytic note [R, not sourced to any paper]: "cosine is scale-invariant, so λ doesn't affect cosine. But λ affects the total gradient direction and the harm." [File 1 line 1528]
- LESS normalizes before using cosine, evidencing the practical scale issue: "we normalize the gradient features in LESS and use the cosine similarity instead of the dot product to estimate influences" [File 1 line 1634].
- Report context [R]: same-loss minibatch gradients are typically near-orthogonal (gradient-noise-scale analysis, arXiv:1812.06162); a sign-stable −0.4…−0.6 between loss components is "a strong, systematic anti-alignment, not noise", but the literature gives no "normal range" for inter-task cosines [File 1 line 1808].

---

## 2. MTP-specific evidence

**Gloeckle et al. 2024 — independent output heads on shared trunk; notable NEGATIVE finding on conflict**
- Citation: *Better & Faster Large Language Models via Multi-token Prediction*, ICML 2024, arXiv:2404.19737 (Meta FAIR).
- Verbatim: "we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk." [File 1 line 1606]
- Verbatim: "our architecture consists of a shared transformer trunk ... n independent output heads ... and a shared unembedding matrix". [File 1 line 1607]
- Sequential fwd/bwd memory trick, verbatim: "By performing the forward/backward on the heads in sequential order, we avoid materializing all unembedding layer gradients in memory simultaneously and reduce peak GPU memory usage." [File 1 line 1608]
- Head-design ablation, verbatim: "We experimented with a single linear layer without any nonlinearity as heads, amounting to linear probing of the model's residual representation"; "Replicating the unembedding matrix n times ... prohibitive for large-scale trainings." [File 1 line 1609]
- n results, verbatim: "training with 4-future tokens outperforms all the other models consistently". [File 1 line 1610] Report summary [R]: n=4 best for 7B code models [File 1 line 1754]. Loss is an unweighted sum over heads [R, File 1 line 1192].
- NEGATIVE finding [R, the report author's regex search, not a paper quote]: "I searched the full text for 'interfer|conflict' — zero hits." [File 1 line 1753] The only head-interaction issue the paper addresses is memory [R, same line].

**DeepSeek-V3 — sequential MTP modules, shared embedding + shared output head**
- Citation: *DeepSeek-V3 Technical Report*, arXiv:2412.19437, 2024.
- Verbatim: "Different from Gloeckle et al. (2024), which parallelly predicts D additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth." [File 1 line 1611]
- Verbatim: "The k-th MTP module consists of a shared embedding layer, a shared output head, a Transformer block, and a projection matrix." [File 1 line 1612]
- Verbatim (physical gradient sharing): "This arrangement enables the physical sharing of parameters and gradients, of the shared embedding and output head, between the MTP module and the main model. This physical sharing mechanism further enhances our memory efficiency." [File 1 line 1758; quote inventory line 1614 has an elided variant]
- Verbatim (λ schedule): "The MTP loss weight λ is set to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens." [File 1 line 1613]
- Verbatim: "which serves as an additional training objective"; "during inference, we can directly discard the MTP modules". [File 1 line 1615]
- Framing [R]: shared head is motivated by memory, not conflict avoidance [File 1 line 1758].

**AdaMTP — closest published match: explicit "noisy, conflicting" gradients from shared backbone representations**
- Citation: *AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction*, arXiv:2608.00434, 2026 (report attributes CUHK/Huawei; venue = arXiv preprint, Aug 2026).
- Verbatim (abstract): "Forcing the auxiliary heads to predict across high-entropy semantic boundaries injects noisy, conflicting training signals; because these heads share the backbone's latent representations, the resulting gradients backpropagate and interfere with the model's core capabilities." [File 1 lines 1616, 1762]
- Verbatim (body): "Because the auxiliary heads and the main language modeling head share the same latent representations, these noisy gradients backpropagate and cause severe representation interference, ultimately degrading the base model's core capabilities." [File 1 lines 1617, 1763]
- Verbatim (measured degradation, Llama-3.1-8B, GSM8K): "the accuracy of standard MTP declines almost monotonically as n grows, falling from 11.60 at n=2 to 9.68 at n=6 — well below the 11.30 NTP reference. This is a direct consequence of representation interference: each additional head forces the shared backbone to predict one token further ahead, crossing more high-entropy semantic boundaries and injecting proportionally more noisy gradients." [File 1 lines 1618, 1764]
- Fix [R, report paraphrase]: mask/drop horizon predictions that cross semantic boundaries, plus auxiliary-head warm-up; not separate heads [File 1 line 1765].

**How Transformers Learn to Plan via Multi-Token Prediction — positive counterpoint, "gradient decoupling"**
- Citation: COLM 2026, arXiv:2604.11912.
- Verbatim: "This behavior arises from a gradient decoupling property of MTP, which provides a cleaner training signal compared to NTP."; "the shallow loss L₂ routes gradients exclusively through Layer 1, entirely bypassing the uninitialized Layer 2." [File 1 line 1619; the quote inventory writes "L2"]
- Verbatim (shared-head variant caveat): "its theoretical setup considers a simplified shared-head variant of MTP, rather than the standard independent-head architecture introduced by Gloeckle et al. (2024)." [File 1 lines 1620, 1768]

**LSE-MTP — gradient coupling; distant-target overemphasis**
- Citation: *Toward Consistent World Models with Multi-Token Prediction*, ACL 2026 Main (aclanthology.org/2026.acl-long.618/), arXiv:2604.06155.
- Verbatim: "MTP promotes the convergence toward internal belief states by inducing representational contractivity via gradient coupling" [File 1 line 1621]
- Verbatim: "overemphasis on distant targets over local connectivity" [File 1 line 1621]; the report says this causes "structural hallucinations" [R, File 1 line 1771].

**MuToR / registers**
- Citation: *Multi-Token Prediction Needs Registers (MuToR)*, NeurIPS 2025, arXiv:2505.10518.
- Verbatim: "Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning." [File 1 line 1623]
- Verbatim: the fix keeps the model "remains aligned with the next-token pretraining objective". [File 1 line 1623]

**Pre-training curriculum**
- Citation: *Pre-Training Curriculum for Multi-Token Prediction in Language Models*, ACL 2025 (aclanthology.org/2025.acl-long.1243/); report attributes to Aynetdinov & Akbik [File 1 line 1559]. arXiv ID not confirmed in files.
- Verbatim: "prior work has shown that smaller language models (SLMs) struggle with the MTP objective." [File 1 line 1622]
- Fix [R]: forward NTP→MTP curriculum [File 1 line 1774].

**Beyond MTP / Future Summaries (FSP)**
- Citation: *Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries*, ICLR 2026, arXiv:2510.14751 (Mila/CMU/Meta).
- Verbatim: "MTP partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement." [File 1 line 1624]

**FastMTP**
- Citation: FastMTP, arXiv:2509.18362, 2025 (Tencent). Full title not captured in files.
- Verbatim: "fine-tunes a single MTP head with position-shared weights on self-distilled data." [File 1 line 1625]
- Note [R]: a parenthetical "Enhanced Multi-Token Prediction" appears at File 1 line 662 but is not established as the title [File 2 line 710].

**L-MTP — status: citation only, NO verbatim quote**
- Citation: *L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context*, arXiv:2505.17505, NeurIPS 2025 (venue "per Awesome list"). The paper was queued/fetched, but the final report records only "L-MTP: leap offsets" [File 1 lines 1570, 848]; no abstract quote appears in either file.

**Self-distillation MTP — partial coverage**
- *Multi-Token Prediction via Self-Distillation*, arXiv:2602.06019 (Maryland): fetched, but report says the abstract does not mention conflict [File 1 line 1513].
- *Self-Distillation for Multi-Token Prediction*, arXiv:2603.23911 (Tencent): not fetched [File 1 line 1513].
- The only self-distillation-adjacent item actually quoted is FastMTP (above).

**OCC — status: NOT FOUND**
- Case-sensitive search for "OCC" in File 1 returns zero matches; File 2's mining report also reports zero matches and states it is "not covered" [File 2 lines 188, 545, 721]. No OCC content exists in either file.

**Headline negative finding for MTP [R]**
- "no paper I could find reports per-horizon gradient cosines at all, so your −0.40/−0.62 numbers are novel in specificity, not in phenomenon" [File 1 line 1745]
- "I found **no** published paper measuring gradient cosine between MTP horizon losses, and none claiming that sign-stable near/far anti-alignment is catastrophic per se" [File 1 line 1827]

---

## 3. Multi-horizon forecasting / world models

**Chevillon — direct vs recursive/iterated multi-step estimation**
- Citation: *Direct Multi-Step Estimation and Forecasting*, Journal of Economic Surveys, 2007, DOI 10.1111/j.1467-6419.2007.00518.x.
- Finding [R, no verbatim quote captured]: the classical multi-horizon forecasting literature frames the problem as a direct-vs-iterated bias–variance/parameter-proliferation trade-off; "I found no gradient-cosine analysis in that literature." [File 1 lines 1591, 1814; the report did not read the full text, per caveat at line 1502]

**DreamerV2 — KL balancing quote (gradient-path splitting/reweighting precedent)**
- Citation: *Mastering Atari with Discrete World Models*, ICLR 2021, arXiv:2010.02193.
- Verbatim: "In the ELBO objective, the KL loss serves two purposes: it trains the prior toward the representations, and it regularizes the representations toward the prior. However, learning the transition function is difficult and we want to avoid regularizing the representations toward a poorly trained prior. To solve this problem, we minimize the KL loss faster with respect to the prior than the representations by using different learning rates, α = 0.8 for the prior and 1−α for the approximate posterior." [File 1 lines 1636, 1815]
- Why it matters [R]: split the gradient path (stop-gradient) and reweight — the family of λ-annealing/masking fixes; it is not diagnosed via cosine [File 1 line 1815].

**LSE-MTP cross-reference** — the closest "world model + MTP gradient coupling" analysis [R, File 1 line 1816].

**Other relevant results in the files**: DeBERTaV3 (LM-pretraining "tug-of-war"; Section 4), LESS and Fish (gradient cosine/inner product as utility; Section 1 caveats) [File 1 lines 1804–1806].

---

## 4. Citations for every quoted item

Venues are as stated in the files; "not stated in files" means the files do not record a venue. Flags for known discrepancies are clearly marked as outside knowledge.

| Item | Title (as in files) | arXiv | Venue (as in files) | Year | Flags |
|---|---|---|---|---|---|
| PCGrad | Gradient Surgery for Multi-Task Learning | 2001.06782 | not stated in files | 2020 | commonly NeurIPS 2020 [outside knowledge] |
| Aligned-MTL | Independent Component Alignment for Multi-Task Learning | 2305.19000 | CVPR 2023; DOI 10.1109/cvpr52729.2023.01923 | 2023 | early file notes mis-ID arXiv:2306.07180; self-corrected |
| Fifty/TAG | Efficiently Identifying Task Groupings for Multi-Task Learning | 2109.04617 | NeurIPS 2021 | 2021 | — |
| Sener & Koltun | Multi-Task Learning as Multi-Objective Optimization | 1810.04650 | File 1 line 1496 says ECCV 2018 | 2018 | **DISCREPANCY**: commonly cited as NeurIPS 2018 [outside knowledge]; File 2 wrongly says "venue not recorded in file" |
| CAGrad | Conflict-Averse Gradient Descent for Multi-task Learning | 2110.14048 | not stated in files | 2021 | commonly NeurIPS 2021 [outside knowledge] |
| FairGrad | Fair Resource Allocation in Multi-Task Learning | 2402.15638 | ICML 2024 | 2024 | no verbatim quote in files |
| GradNorm | GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks | 1711.02257 | not stated in files | 2018 | cited only; not fetched |
| DeBERTaV3 | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing | 2111.09543 | File 1 line 1498 says ICLR 2022 | 2021 | **DISCREPANCY**: commonly cited as ICLR 2023 [outside knowledge; File 2 flags it too] |
| LESS | LESS: Selecting Influential Data for Targeted Instruction Tuning | 2402.04333 | ICML 2024 | 2024 | — |
| Fish | Gradient Matching for Domain Generalization | 2104.09937 | ICLR 2022 | 2022 | — |
| DreamerV2 | Mastering Atari with Discrete World Models | 2010.02193 | ICLR 2021 | 2021 | — |
| Gloeckle et al. | Better & Faster Large Language Models via Multi-token Prediction | 2404.19737 | ICML 2024 (Meta FAIR) | 2024 | — |
| DeepSeek-V3 | DeepSeek-V3 Technical Report | 2412.19437 | tech report, no venue | 2024 | — |
| AdaMTP | AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction | 2608.00434 | arXiv preprint (Aug 2026); CUHK/Huawei per report | 2026 | authors not captured |
| planmtp | How Transformers Learn to Plan via Multi-Token Prediction | 2604.11912 | COLM 2026 | 2026 | authors not captured |
| LSE-MTP | Toward Consistent World Models with Multi-Token Prediction | 2604.06155 | ACL 2026 Main (2026.acl-long.618) | 2026 | authors not captured |
| MuToR | Multi-Token Prediction Needs Registers | 2505.10518 | NeurIPS 2025 | 2025 | repo github.com/nasosger/MuToR |
| Curriculum | Pre-Training Curriculum for Multi-Token Prediction in Language Models | not confirmed in files | ACL 2025 (2025.acl-long.1243) | 2025 | Aynetdinov & Akbik per report; arXiv ID unknown |
| FSP | Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries | 2510.14751 | ICLR 2026 (Mila/CMU/Meta) | 2026 | final report uses short title "Beyond MTP: Pretraining with Future Summaries" |
| FastMTP | "FastMTP" (full title not captured) | 2509.18362 | arXiv 2025 (Tencent) | 2025 | title uncertain; "Enhanced Multi-Token Prediction" not established |
| L-MTP | L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context | 2505.17505 | NeurIPS 2025 (per Awesome list) | 2025 | no quote captured |
| Self-distill (1) | Multi-Token Prediction via Self-Distillation | 2602.06019 | not stated in files | 2026 | fetched; no conflict content per report |
| Self-distill (2) | Self-Distillation for Multi-Token Prediction | 2603.23911 | not stated in files | 2026 | not fetched |
| Chevillon | Direct Multi-Step Estimation and Forecasting | DOI 10.1111/j.1467-6419.2007.00518.x | Journal of Economic Surveys (File 2 citation list; File 1 gives DOI only) | 2007 | full text not read |
| McCandlish et al. | An Empirical Model of Large-Batch Training | 1812.06162 | not stated in files | 2018 | context only; no quote |

**Unmined leads listed in the files (no quotes):** Sony, *On multi-token prediction for efficient LLM inference* (ICLR 2025 Workshop; head-sharing ablation — "could skip") [File 2 line 781]; MiMo technical report [File 2 line 781]; *Next-Latent Prediction Transformers Learn Compact World Models* (arXiv:2511.05963; listed in Awesome-MTP, not fetched) [File 1 line 1673]; *Parallel Token Prediction* (arXiv:2512.21323) [File 1 line 1665]; *Enhancing Visual Planning with Auxiliary Tasks and Multi-token Prediction* (WACV 2026) [File 1 line 1667].

---

## 5. GAPS: searched but NOT found or NOT quoted

1. **No published paper found that measures per-horizon gradient cosine in MTP.** The report's explicit negative finding; closest are AdaMTP (qualitative "noisy, conflicting" + performance evidence), LSE-MTP (coupling theory), planmtp (decoupling theory) [File 1 lines 1470, 1745, 1827].
2. **No canonical "normal range" for inter-task/inter-horizon gradient cosines published** [R, File 1 line 1808].
3. **FairGrad: no verbatim quote captured.** The scale-invariance/can-be-beneficial claim is the report's speculation, not sourced [File 1 lines 862, 1224; File 2 line 652].
4. **OCC: zero matches in either file.** Must be sourced elsewhere [File 1 grep; File 2 lines 188, 545, 721].
5. **L-MTP: no verbatim quote.** Only title/arXiv/venue and the summary "leap offsets" [File 2 line 713].
6. **Self-distillation MTP: partial.** 2602.06019 fetched (no conflict content per report); 2603.23911 not fetched; only FastMTP quoted in this family [File 1 line 1513].
7. **Head-sharing ablation leads not mined:** Sony ICLR 2025 Workshop paper; MiMo [File 2 line 781].
8. **Chevillon: no full-text read, no gradient-cosine analysis found**; the direct-vs-iterated framing is the report's characterization [File 1 lines 1502, 1814].
9. **Gloeckle "zero hits for interfer|conflict" is the report author's regex search**, not a published claim [File 1 line 1753].
10. **Search-infrastructure limitations:** Exa/Semantic Scholar/arXiv APIs were rate-limited; coverage came from OpenAlex full-text search, direct fetches, the curated Awesome-MTP index, and one DuckDuckGo pass [File 1 line 1827].
11. **Quote fidelity:** quotes are agent-extracted from stripped HTML with acknowledged minor spacing artifacts; verify against primary PDFs before publication [File 1 line 1603].
12. **Citation discrepancies to verify before archiving:** DeBERTaV3 file says ICLR 2022 vs commonly ICLR 2023; Sener & Koltun file says ECCV 2018 vs commonly NeurIPS 2018; Aligned-MTL early mis-ID 2306.07180 vs final 2305.19000; Curriculum arXiv ID unconfirmed.

---
