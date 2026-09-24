# 35 — v2 阶段性总结(2026-09-22,压缩上下文前封存)

> 用途:v2 时代(2026-09-17 转向起)收口——实验时间线与结论、代码架构变更、
> 悬在上下文未落档的信息、下一步候选。新会话从 AGENTS.md 当前状态 + 本文
> + docs/31(设计)+ docs/32(施工)+ docs/34(评估协议)读起。

## 1. 实验时间线与结论

1. **2026-09-17 转向**:H1 判失败(条件独立病理,"cat/dog/dot"),v2 拍板
   (CoSE 式:重建塑形 AE + 梯度硬切断 + MDN/GMM 骨干 + 增强 BPE + EOS
   终止符)。设计 docs/31、施工 docs/32、问询全回填、审阅通过(两补充:
   骨干输入宽非线性适配器默认;去噪解码全关、需要时优先字节腐化式)。
2. **S0(AE 规模)**:网格定 48-256 为基座(val EM 97.4%;几何健康:
   近满秩/低各向异性);GMM 定稿对角协方差+两层非线性缓冲(768→1024→
   1024→K(1+2d));64 维组弃(GMM 高维拟合难)。
3. **S1(缓存)**:增强 BPE 全量缓存建成(573322:~12.6B units/46GB/
   3.85h);均长 3.47B/unit。两起 sys.path/配置事故(573323、573890 的
   32/48 撞维)均已修复并立防(构造期 d_emb 断言)。
4. **AE 全量验收(三件套,已过)**:零次桶 EM 93.6%(真泛化)/SC 0.384
   (超 CoSE 论文最优)/distinct ~76M 串(有效样本量由分割多样性界定)。
   配置事故意外产出 **32 维**全量 AE,顺势先用(48 留升级臂)。
5. **POC 60k(574099)**:首发撞 MDN σ 坍缩死锁(根因入档),sigma_floor
   0.05 修复;NLL -32.8(衰减尾仍在上探),argmax EM 32.7%/分量覆盖
   60.9%/π 无坍缩;分锅探针判**解码器无辜、误差在骨干预测**;幽灵词
   病理在原理层根治(采样=词级承诺)。
6. **1B 长训(576234,+1B 自 50k 分叉,LR 2e-4)**:NLL -33.86(尾段仍
   在上探,零泛化间隙);EM 34.3%/覆盖 51.2%;**生成置信大幅改善**
   (-0.63→-0.16);排序 gap 未收。
7. **typo 工具箱 #2**:四臂网格(p=0/0.1/0.2/0.4)→ 我初评口径错误
   (拿抗噪 EM 测 typo 忠实度,南辕北辙,被用户纠正并记录)→ 按设计
   意图重评估**翻案**:typo 保留率单调随剂量升(0.876→0.977)、case 从
   0.583 修到 0.92+、**t01(p=0.1)全面甜点**(margin/碰撞/分离交叉)。
   全量 typo-AE(576435)+ typo-POC(576436):能力 ≈ plain@1B(同步数
   +1.5pp),失败模式一致。**真词分离度量修复**(C1 规范形队列/C2 余裕
   与碰撞)。**变体可分性排查:无架构漏洞**(大写/复数方向存在;自然
   变体分辨率 ~完美;非自然形态纠错式坍缩)。
8. **生成解剖(2026-09-22)**:周期-2 吸引子;**回投影有害**(软误差
   硬化,生存 0.25 vs 0.5-1.0);**verifier 门控后真词骨架浮现**;
   单步 EM ~34% 与 GPT-2-small 级下词 top-1 同量级——**乱码主因是
   读出,不是能力**。
9. **词表吸附解码(用户提出,已入库)**:GMM 密度在合法词(Qwen∪缓存
   top-10 万)嵌入表上离散化 = **词级分类分布**;gmm 吸附 EM 37.9%
   (>解码器 +3.3pp,>余弦 +3.2pp——σ 投票有效);**乱码模式消灭,
   首个通顺英文样本**;词级温度/top-k/top-p 全部成立。
10. **typo-2B 计算最优长跑(579081,已完成 2026-09-23,验收 09-24)**:50k
    分叉,+244140 步 ≈2.0B units,36.7h,WSD 退火至 -63.41(健康无失控)。
    **验收(docs/34 协议全过,详表入 docs/32 末节)**:snap gmm EM **0.390**
    (> plain@1B 0.379 > typo@60k 0.365),argmax 0.369/top5 0.531,π 无坍缩,
    h 空间话题簇成立;生成 τ0.6-0.8 完整真句但内容仍浅——读出层问题已
    收口,**剩余瓶颈=能力深度(数据/参数档位)**。

## 2. 代码架构变更(v2 新增,全部入 git)

| 模块 | 内容 |
|---|---|
| `bltz/segment_v2.py` | 增强 BPE 分割器(规则预分词 × Qwen3.5 BPE;一致必切/分歧 p=0.5 预烘焙按文档定种;向量化热循环 + encode_batch 批处理 499 docs/s) |
| `bltz/models/autoencoder.py` | ByteStringAE(Set Transformer 编码器+位置拼接,输出投影到 d_emb;film 条件查询解码 (λ,Δ)→257 行含 EOS;Δ=33 非法截断;贪心 decode) |
| `bltz/models/mdn.py` | MDN 头(两层非线性缓冲 768→1024→1024;对角 GMM K=64;π/μ/σ,σ=ELU+1+floor 0.05;fp32 logsumexp NLL;小初始化;分量均值采样,τ 只调 π) |
| `bltz/models/model_v2.py` | BltzLMv2(冻结 AE 硬切断 + 宽非线性适配器 LN→48→2048→GELU→768 + 可学 BOS + 113M 骨干 + MDN;λ 硬 detach;构造期 d_emb 断言防撞维) |
| `bltz/objectives_v2.py` | mdn_nll_loss(teacher forcing,fp32 NLL) |
| `bltz/typo.py` | typo 簇增强(char_units:ASCII 字节级+多字节整体;五操作 40/20/15/15/10;1-2 编辑;长度闸门;p_eff 频率节流;force_op 诊断口) |
| `scripts/train_ae.py` | AE 训练(pool/stream 双模式;val 留出;typo 批时在线增强;best-by-val_loss) |
| `scripts/train_bltz_v2.py` | 骨干训练(复用 v1 trainer:WSD/spike_skip/优雅中断/里程碑) |
| `scripts/build_cache_v2.py` | v2 缓存构建(only_files/进度条 docs/s+ETA/批处理分割;心跳规矩由此立) |
| 诊断套件 | `diag_ae.py`(几何)/ `diag_ae_v2.py`(验收三件套,--units-pkl 零额度)/ `diag_ae_typo.py`(typo 设计意图口径)/ `diag_v2.py`(能力+三读出+生成)/ `diag_v2_semantic.py`(h 聚类+案例倾倒+t-SNE)/ **`snap_decode.py`(词表吸附读出+多 prompt 多 τ 生成)** |
| slurm( v2 ) | cache_v2 / ae_full / v2_60k / v2_1b / ae_full_typo / v2_typo_60k / v2_typo_2b;(ae_full_plain48 / v2_plain48_60k 解混备件,已挂后按指示撤回,随时可再挂) |
| 测试 | 14 个全绿(v1 11 + test_ae + test_v2 + test_typo) |

## 3. 悬在上下文、未(全)落档的信息

1. **集群在途**:**无 bltz job 在跑**(typo-2B 579081 已完成验收,见 §1.10)。
   **2026-09-24 拍板关闭:v1 全部封存不再续(boundary_2b 弃);48-plain
   解混臂取消(sbatch 存档不挂);typo-2B 配方(48d typo-AE + MDN 骨干 +
   增强 BPE)定为默认配置,后续一切基于此。** **SSH 必须显式
   `yihe47@`**(本机用户名「何」会被拒,AGENTS.md 已立规)。
3. **本地工件**:Temp\opencode 下 ckpt 副本(ae32_full_best、
   ae48_typo_full_best、v2_60k_last、v2_1b_last、v2_typo60k_last、
   diag_sample.pkl、best_anneal 等 v1 旧件);`data/fineweb_v2_local`
   (本地 1.1GB v2 缓存,250k 文档)+ `data/parquet_local`(2GB parquet)
   + `data/qwen35_tokenizer`;本机 checkpoints/ae48_local_*(typo 四臂
   48 维 ckpt)与 ae_32_256/ae_48_256(S0 网格 ckpt)。
4. **教训集**(详散各文档,集中索引):评估口径必须服从设计意图
   (typo 南辕北辙,docs/32 #2);shuffle 对逐对统计不变(对照必须真
   打乱配对,probe 修正);**NLL 跨 λ 空间不可比**;**回投影有害**
   (软误差硬化);长脚本必须心跳(AGENTS 已立规);PowerShell `$()`
   与 heredoc 插值坑(sbatch 串联一律两条独立 ssh);squeue 格式化串在
   PS 双引号里会被吃,用单引号;本机 python 退出挂起僵尸定期清理。
5. **未用/可扩充资产**:vocab 表覆盖 95.6%,可扩 cache-topN 提升;
   48 维 local typo 四臂 ckpt;v1 boundary_2b 主 run 仍暂停可续
   (scratch ckpt@~140749);v1 全部诊断锚点(docs/28)。
6. **备录**:多语言(中文等)typo 模式不是简单拼写错误,多语言语料
   的增强模式以后详细商讨(docs/32 #2);typo 结构抓手为下游任务
   (辨识错别字/拼写与单词计数/合成词理解)预留,任务级评估未建
   (docs/34 §7 缺口)。
7. **HF token** 在 `$SCRATCH/hf_cache/token`(600,HF 自动读取,
   工作区禁存)。
