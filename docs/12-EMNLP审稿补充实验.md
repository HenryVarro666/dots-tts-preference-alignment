# 12 · EMNLP 审稿驱动的补充实验（resubmit 实操清单）

> **来源**：对 `EMNLP/dots-tts-emnlp.tex` 跑 `academic-paper-reviewer` 五人面板 + Devil's Advocate 的裁决（Major-Revision，Findings 天花板；as-is ~25–30% → 文字修订后 ~60–70% Findings；Main 需新实验）。文字类 overclaim 已在 `.tex` 修掉；**本文件只列需要「跑」的补充实验**。
> **本文件是这批实验的单一 source-of-truth**；跑完按「落表位置」回填 `.tex`，并在此更新状态。与 [`11-未来工作与展望.md`](11-未来工作与展望.md) 的关系：11 是全局 roadmap，12 是 EMNLP 审稿专项、更细。

## 紧急度总览（先跑哪个）

| | 实验 | 对应审稿 CRITICAL | 是否需 GPU/新数据 | 成本 | 紧急度 |
|---|---|---|---|---|---|
| **E1** | 选择伪影 null control | #3（DA 最强反驳：极值选择伪影） | 重跑 probe 出 per-candidate L_FM（**用现有候选池、无需新合成**） | 小时级 no_grad | **🔴 最高** |
| **E2** | 分块 + 最坏情况 EOS 计分 | collapse 混淆 / n=75 vs n=100 | **若 per-utterance 评测已存 → 纯分析、零 GPU** | 分钟–小时 | **🔴 高（最便宜）** |
| **E3** | 再合成方差（确认 open-SOAR） | #4（唯一正面结果可能落在噪声带） | K 次再合成 ×{base,open\_soar} | 小时级 | 🟠 中高 |
| **E4** | VoxCPM2 原生配对 | #2（复现非独立、两 ρ 依赖） | VoxCPM2 自采样 + 探针 | 半天–1 天 | 🟠 中 |
| **E5** | 多说话人 scale-up + 人类听测 | excitement / venue 封顶 | VCTK+AISHELL-3 重训 + 听测 | **1–2 GPU 周 + 听测预算** | 🟢 低（冲 Main 才需） |

> **结论**：**Findings resubmit 只需 E1 + E2 + E3**（都不大）。E4 把复现做成真独立（Findings→更强）。E5 是 Main 的门票，资源大、可缓。

---

## E1 · 选择伪影 / order-statistic null control 🔴 最高优先

- **审稿点**：Devil's Advocate 最强反驳 + methodology reviewer。settings-matched 下 winner/loser 只差 seed，reward gap 近噪声；best-vs-worst 选的是**极值抽样**，极值天然 atypical（高 `L_FM`）。⇒ 反相关 ρ=−0.216 可能是「在噪声上选极值」的 order-statistic **伪影**，而非 reward↔likelihood **耦合**。当前 `.tex` 只在 §6/Limitations 加了 caveat 承认这一点——**跑出 null control 才能反驳、把 caveat 变成正面证据**。
- **可行性**：**无需新合成**——用现有 settings-matched 候选池（每 prompt 多 seed 候选 + reward）即可；只需重跑探针**对全部候选**算 `L_FM`（no_grad 前向）。
- **协议**：
  1. **✅ 脚本已就绪**（`probe_fm_reward.py` 加了 `--emit-per-candidate`、`probe_analysis.py` 加了 `--selection-null`、并配 `run_e1.sh`；本机 py_compile / bash -n 已过）——一条命令：
     ```bash
     bash run_e1.sh 0 reward/out/rewards_fixed.jsonl   # GPU=0, settings-matched 候选池
     ```
     = ① `probe_fm_reward.py --emit-per-candidate`（对每 prompt 的**全部候选**算 K-MC 去噪 `L_FM`，落 `alignment/out/percand_lfm.csv`：`fid,k,reward,lfm,audio`；no_grad、**无需新合成**）；② `probe_analysis.py --selection-null`（读该 csv，CPU 秒级）。
  2. `--selection-null` 报三件事（裁决「伪影 vs 耦合」）：
     - **(candidate-level，最干净)**：within-prompt 去均值 corr(reward, `L_FM`) over **全部候选、零极值选择**——ρ>0（高 reward = 高 `L_FM` = 更 off-manifold）且 perm-p<0.05 ⇒ 耦合在候选级即成立、与极值选择**无关**（最强反驳）。
     - **(Null A，关键)**：用 B=1000 个随机准则替代 reward 做 best-vs-worst → null ρ 分布；ρ_real 落在 null **之外**(p<0.05) ⇒ **reward-specific、非通用极值伪影**。
     - **(Null B)**：interior pairs（2nd-best vs 2nd-worst）反相关是否存活 ⇒ 非仅极值驱动。
  3. **判据**：
     - ρ_real **落在 (b) null 分布之外**（单侧 perm-p < 0.05）⇒ 反相关是 **reward-specific**、非通用极值伪影 → **直接反驳 DA**，把 §6 caveat 改成「已排除 order-statistic 伪影」。
     - ρ_real **落在 null 内** ⇒ 诚实**收回**「intrinsic coupling」措辞、改述为「extreme-selection 效应」（论文仍成立但更弱）。
  4. **落表**：§6 头条段加一句 + 附录小表（ρ_real vs null 均值/95%）；替换 `.tex` 现有 Limitations 的「native null control left to future work」。
- **预计**：probe per-candidate 前向 ~小时级；分析 CPU 分钟级。
- **✅ 已跑（2026-06-28，结果 mixed、已诚实落 §6）**：`ρ_real=−0.217`（≈ headline −0.216，自洽性通过）。
  - **Null A ✅ p=0.004**：反相关 reward-specific、**非通用极值伪影** → 正面反驳 DA 头号拒稿点。
  - **candidate-level ❌**：within-prompt Spearman −0.008（perm-p=0.75，n.s.）→ 候选级无平滑耦合。
  - **Null B ❌**：interior pairs +0.051（翻号）→ 不在非极值对存活。
  - **裁决**：反相关真实且 reward-specific，但是 **best-vs-worst 极值对比现象**（= DPO-FM 训练 regime），**非平滑内容耦合**。§6 已据此：加 Null A(p=0.004)、把"intrinsic content-typicality coupling"收窄为"extreme-contrast, reward-specific, 限 best-vs-worst regime"，机理段同步加 scope。**一得（Null A 强反驳）一失（机理收窄）**，净诚实增强。

## E2 · 分块 + 最坏情况 EOS 计分 🔴 高（最便宜，建议先跑）

- **审稿点**：EIC + methodology + DA。dpo\_multiobj 崩坏可能是 **LJ002 块集中** + **丢弃 25 条 immediate-EOS 失败**的伪影；`n=75`（成功句）与 base 的 `n=100` **不可比**，低估或误判退化。
- **可行性**：**若逐句 WER/UTMOS 已存（评测产物带 fid）→ 纯分析、零 GPU**；若只存了 aggregate → 对现有音频重跑 ASR/UTMOS（便宜，无需重新合成）。
- **协议**：
  1. **✅ 脚本已就绪**（`eval/analyze_collapse_blocks.py` + `run_e2.sh`；本机 py_compile / bash -n 已过；**纯 CPU、无需 GPU/新合成**）——一条命令：
     ```bash
     bash run_e2.sh                                  # 默认 manifest=data/prompts/eval_big.jsonl, arms=base_nfe10 sft_nfe10 dpo_multiobj_nfe10
     ```
     读 `results/out/{arm}.metrics.jsonl`（`score_synth.py` 落盘的逐句 `{fid,wer,sim,utmos}`）+ manifest（全 100 fid）；**缺失的 fid = immediate-EOS 失败句**（无需单独记录）。
  2. 报告四件事（写回 §5.1 / Table 1）：
     - **(a) scored-only**：当前 n=75 口径。
     - **(b) worst-case n=100**：把 EOS 失败句补 `WER=1.0 / UTMOS=1.0`，得**可与 base n=100 比**的数字。
     - **(c) apples-to-apples**：`base` 限制到崩坏臂打分的**同一子集**重算（n 一致，直接回应「n=75 vs n=100」）。
     - **(d) 分块**：按 block（LJ001/LJ002…）报 WER/UTMOS/scored/EOS-fail——看崩坏跨 block 均匀还是仅 LJ002。
  > 前提：`results/out/{arm}.metrics.jsonl` 存在（评测时 `score_synth.py` 已落盘）。若只剩 aggregate summary.json、metrics.jsonl 丢了 → 对现有 `inference/out/{arm}/*.wav` 重跑 `eval/score_synth.py` 一次（便宜、无需重新合成）即可补出逐句。
  3. **判据**：worst-case n=100 仍崩 **且** 崩坏跨 block 均匀 ⇒ collapse 稳健、非 LJ002+丢句伪影（可保留「regardless of objective」但仍标 cautionary）；若只在 LJ002 ⇒ 改述为「block-specific」、并扩 n / 重排评测集。
  4. **落表**：Table 1 加 **worst-case-n100** 列或脚注；§5.1 加分块说明；按结果决定是否软化「off-manifold self-training harms regardless of objective」。
- **预计**：有逐句数据则 < 1 小时纯分析。**先跑这个（最便宜、直接补 n=75 的硬伤）。**

## E3 · 再合成方差（坐实 open-SOAR 唯一正面结果）🟠 中高

- **审稿点**：CRITICAL #4。open\_soar WER 0.094 vs base 0.112，可能落在 base 的**再合成漂移带**（已知 0.103/0.128/0.112，generate() 无 seed）内；bootstrap CI 只覆盖句子抽样、**不覆盖再合成方差**。`.tex` 已先软化为「不退化」——**跑出方差带才能（可能）升回「真增益」**。
- **可行性**：需 **K=5 次再合成** × {base, open\_soar} × 100 句（无新训练，仅推理）。
- **协议**：
  1. **✅ 脚本已就绪**（`eval/resynth_variance.py` + `run_e3.sh`；本机 py_compile / bash -n 已过）——一条命令（需 GPU）：
     ```bash
     bash run_e3.sh 0                                  # GPU=0, manifest=data/prompts/eval_big.jsonl, reps=5, nfe=10, arms=base open_soar
     ```
     编排既有 `batch_synthesize.py` + `score_synth.py`：对 base 与 open\_soar 各**独立再合成 5 次**（generate() 无 seed → 每次不同），每次算 WER/UTMOS。
  2. 报告每臂 **跨 K 次的均值 ± SD**；构造 base 的再合成带（如 mean ± 2SD）。
  3. **判据**：open\_soar 的 WER/UTMOS 均值若**清出** base 再合成带 ⇒ 升级为「真增益」（强化唯一正面结果）；否则维持当前「不退化」措辞（已诚实）。
  4. **落表**：§5.1 + Table 1 脚注加再合成 SD；据结果升级/维持 open\_soar 结论。
- **预计**：5×2×100 句推理，小时级。

## E4 · VoxCPM2 原生配对（让跨架构复现真正独立）🟠 中

- **审稿点**：CRITICAL #2（3 reviewers + DA）。当前把**同一批 dots 生成对**喂给 VoxCPM2，两 ρ 共享 reward-gap predictor、**统计依赖** → 只能说「第二个似然模型在同一批对上同意」，不能说独立复现 N=2。`.tex` 已据此降调。
- **协议**：
  1. 从 **VoxCPM2 自身** 采样候选（其 generate + 同样的 grid）、用**同一 reward** 打分、VoxCPM2 自身 best-vs-worst 配对 → 得一批 **VoxCPM2 原生偏好对**。
  2. 在这批原生对上跑 VoxCPM2 FM 探针（`probe_fm_reward_voxcpm.py`，已有）+ settings-matched 复探针。
  3. **判据**：原生对仍 ρ<0 显著 ⇒ **升回**「architecture-level independent N=2」措辞（接近 Main 的独立性论据）；否则维持「dependent check」。
- **预计**：VoxCPM2 候选合成 + 打分 + 探针，半天–1 天。

## E5 · 多说话人 scale-up + 人类听测（冲 Main，可缓）🟢 低

- **审稿点**：EIC excitement/scale 封顶、venue-fit、多 reviewer。单说话人 LJSpeech（SIM 近 ceiling、N=188、n=100）+ 无人类听测，是 excitement 低于 Main 线的**结构性原因**。
- **协议**：
  1. **多说话人/多语种**：VCTK（EN ~110 spk）+ AISHELL-3（ZH ~218 spk，走 **CER** 非 word-WER，重采样 48k）重跑 `base`/`open_soar`/`dpo_multiobj`（scaffold `configs/*_{vctk,aishell3}.yaml` 已建）→ 使 SIM 可判别、超 188 对、测跨语言。
  2. **人类听测**：≥15–20 听者 ABX/MOS，锚定崩坏臂量级 + open\_soar 增益（把「automatic-metric」升级为「perceptual」）。
- **判据**：SIM 变可判别 + 跨语言成立 + 听测确认 ⇒ 解 excitement/scale 封顶 ⇒ 够 **Main**。
- **预计**：**1–2 GPU 周 + 听测预算/平台**。**Findings 不需要；Main 必需**，列 resubmit 后或单独冲 Main。

---

## 给 Findings resubmit 的最小充分集

**E2 →（同时）E1 → E3**，三个跑完即可把 5 条 CRITICAL 中靠实验的三条（选择伪影、collapse 混淆、再合成噪声）从「文字 caveat」升级为「已排查」。E4 让跨架构变独立（更强）。E5 留给 Main。

> **维护提示**：每个 E 跑完，更新本表状态 + 回填 `.tex`（见各「落表位置」）+ 同步 [`README.md`](README.md) / [`docs/README.md`](README.md) 索引。本批实验脚本多为**扩展现有脚本**（`probe_fm_reward.py` 加 `--emit-per-candidate`、`probe_analysis.py` 加 `--selection-null`、复用 `batch_synthesize` / `probe_fm_reward_voxcpm.py`），非全新。
