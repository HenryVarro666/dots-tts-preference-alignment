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
- **✅ 已跑（2026-06-28，干净 PASS、已落 §5.1 + Limitations）**：
  - **worst-case n=100**：dpo_multiobj WER 0.890→**0.917**（补 EOS 失败后**更崩**，没藏）；UTMOS 2.719→2.289。
  - **apples-to-apples**：base 在**同 75 句** WER **0.100** vs dpo_multiobj 0.890 → **同输入 ~9× 退化** → 封掉 n=75 vs n=100。
  - **跨 block**：崩坏在 LJ001(0.84)+LJ002(0.90) 都崩（非单块）；仅 immediate-EOS 失败集中 LJ002（25 条全在）。
  - **裁决**：collapse 稳健、**非「LJ002+丢句」伪影** → §5.1 加三条 control，Limitations 把该混淆标为「已控制」，只剩 graded-vs-divergence 待定。**干净加分。**

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
- **✅ 已跑（2026-06-28，分轴结果、已落 §5.1）**：
  - **base WER 0.136 ± 0.045**（带 [0.091, 0.181]）——论文里的 0.112 是一次走运抽样；UTMOS 4.242 ± 0.041。
  - **open_soar UTMOS 4.398 ± 0.018 → 清出 base 带 [4.20, 4.28] = 真增益** ✅；**WER 0.091 ± 0.004 落在 base 带下沿 → 不主张 WER 增益**，但**稳定性碾压**（SD 0.004 vs 0.045）。
  - **崩坏臂 0.556/0.890 远超 base 最差抽样 0.181 → 崩坏结论对再合成噪声稳健。**
  - **裁决**：§5.1 已改为「UTMOS 真增益 + WER 不主张（但更稳）+ 崩坏稳健」。诚实分轴。

---

## ★1 · 剂量-反应（审稿 2.0 的因果桥，脚本已就绪）🔴 最高 ROI 的下一步

- **审稿点（claim 4/9 缺因果桥）**：X4 是 4 个离散类别点；"越 off-manifold 的 target 越毒"还是相关、非因果。把 x 轴**量化**为各 target 类型音频的均值 `L_FM`，对崩坏(WER)做单调性 ⇒ 剂量-反应。
- **✅ 脚本已就绪**（`probe_fm_reward.py` 加 `--score-audio`、`eval/dose_response.py`、`run_dose.sh`；py_compile/bash -n 过）。**大部分免费**：winners/losers/random 三点直接从 E1 的 `percand_lfm.csv` 抽；groundtruth(真录音) 一次 GPU 前向补第 4 点。一条命令：
  ```bash
  bash run_dose.sh 0
  ```
- **判据**：按 target `L_FM` 升序，崩坏 WER 单调上升 + Spearman>0 ⇒ 剂量-反应成立 → §5.1 把"monotone in off-manifold-ness"从 4 类别点升为量化曲线（claim 4 🟡→✅）；否则诚实收回单调叙事。
- **后续（拔高）**：再加 1–2 个插值臂（groundtruth↔winner 按 α 混的 target）训练，补成平滑曲线。
- **⚠️ 第一次跑（2026-06-29）= 无效 + 暴露深层隐患**：
  - **踩 pool 坑**：用了 E1 的 settings-matched `percand_lfm.csv`，但 X4 的 sft/sft_losers/sft_random 训练用的是 **grid 池**（`winners_only`←`pairs_best_worst`←`rewards.jsonl`）→ target L_FM 算错池。**已修 `run_dose.sh` 用 grid rewards 重算**，待重跑。
  - **深层隐患**：四类 target L_FM **几乎一样（0.776–0.784）且真录音最高**——违背"真录音=低 L_FM=on-manifold"。即 **L_FM 可能根本不区分 target off-manifold 度**，崩坏真正跟 **reward 极值**（winners/losers 都崩、random/real 不崩），L_FM 解释不了。与 E1（候选级耦合缺失）一致 → **L_FM 不是崩坏故事主线**。
  - **若 grid 重跑仍 flat**：诚实**降级 claim 4/9 的"off-manifold via L_FM"机制**，改述为"崩坏由 reward-extreme self-generated target 驱动、但 flow-likelihood 不解释"，并跑 ② 找真正的解释变量（reward 极值 / 生成设置）。
- **✅ v2 全量重跑（2026-06-29，188 prompt / grid 池 / 双轴）= 结论清晰、已落 §5.1+contrib+Table1**：
  - **轴 A（绝对 L_FM）= inconclusive**：四类 0.777–0.785 全在噪声内（类间差 ~0.01 < 1 个 MC 标准误）→ 绝对 L_FM 测不动 target 类型（E1 那个反相关靠的是**配对去噪**，绝对均值没法抵消噪声）。**这是"量太吵"，不是"机制错"**。
  - **轴 B（target reward 极值 |reward|）= ✅ 完美单调（Spearman +1.000）**：random(|r|=0.460,WER 0.138) < winners(0.619,0.556) < losers(0.899,0.569)；real 非极值→不崩 0.087。
  - **裁决**：崩坏**随 target 的 reward 极值单调加剧** = 可量化剂量-反应 → **claim 4 🟡→✅**。机制统一到 **reward-extreme self-generated target**（与 E1 的 extreme-contrast 同主线"极值"），**不靠绝对 L_FM**。
  - **已回填**：§5.1 X4 句、contrib 3、Table 1 caption 全改成"monotone in **reward-extremity**"（量化 + L_FM-too-noisy 注）。**E1+dose-response 这条"极值"线闭环。**

---

## ★3 · claim 5 识别 —— ❌ 不用做新实验（现有 2×2 已闭合 + 揪出论文事实错误）

- **核代码发现**：`reward_weighted` 的 `train_manifest = pairs/out/winners_only.jsonl` → **训练目标是 winner（off-manifold），不是真实录音**。论文原写"anchored to the **real** target"**是事实错误**（已修）。
- 所以现有三臂已构成 **2×2 因子识别**（共享 `sft` 这个角）：
  - `sft`(uniform, winner) 崩 0.556 vs `reward_weighted`(reward-weighted, **同** winner) 逃逸 0.092 → **目标函数是一条杆**。
  - `sft`(uniform, winner) 崩 vs `sft_groundtruth`(uniform, **真录音**) 不崩 0.087 → **目标是第二条独立杆**。
  - ⇒ **两条杆各自都能救**；崩坏需"似然/模仿型目标函数 + off-manifold 目标"**两者兼具**。`dpo_onmanifold`(似然-DPO+部分清洗) 仍崩 0.315 一致。
- **2×2 第 4 格（reward-weighted×真目标）= 冗余、最没信息量 → 不训**。`.tex` §5.1/abstract/contrib 已改成"两杆识别"诚实版，**claim 5 🟡→✅、零新训练**。

## ★2 · 第二指标交叉验证 —— ✅ 脚本就绪（band 担忧其实大部分已被现有流程控制）

- **核代码发现**：`utmos_backend.py` `librosa.load(sr=16000)` → UTMOS **本就把音频重采样到 16kHz 再打分**，所有臂口径一致 → **48k vs 16k band-mismatch 不是混淆**（不会制造相对差异）。`.tex` band 措辞已改准（"均匀保守低估、非混淆"）。
- 残留是"单一指标"风险 → ★2 改为**换第二个独立 MOS（DNSMOS）重打分臂音频**，看 open-SOAR 增益是否复现：
  ```bash
  pip install speechmos   # 第二指标后端（DNSMOS）
  bash run_e2b.sh         # 默认 base_nfe10 open_soar_nfe10；纯重打分、无新合成
  ```
  - **判据**：open_soar 第二指标仍 > base ⇒ 自然度增益**跨指标稳健、非 UTMOS 伪影** → §5.1/Limitations 加一句；否则收回。
- 脚本：`eval/rescore_metric.py` + `run_e2b.sh`（py_compile/bash -n 过）。
- **✅ 已跑（2026-06-29）**：base DNSMOS 3.287 vs open_soar 3.345（Δ+0.058，同号 > base）→ open-SOAR 自然度增益**跨指标方向一致、非 UTMOS 伪影**（量级小、无 CI，仅方向确认；配 E3 的 UTMOS 清出再合成带 = 双重确认）。§5.1 已加一句。**★2 收尾。**

> **仍未写**（拔 Main 才需）：★4 β/λ 扫（发散 vs 退化）、★5 VoxCPM2 自生成 pair（真 N=2）。

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

---

## 终检 · 全文对抗式一致性扫描（2026-06-29，实验批次全完后）

十几轮 surgical 改动后跑了一个 5 路并行 workflow（数字一致性 / 校准过度声称 / 跨节自相矛盾 / 极值主线连贯 / LaTeX+8 页），合并出 punch-list，已全部落 `.tex`（25 处编辑，括号/美元号校验平衡）。

**1 个 CRITICAL（五审稿人独立都标）**：§6 还留着禁用语 `valid likelihood surrogate`，与正文 "not a broken loss"、§3 "keep it heuristic" 直接打架 → 改为 "not a broken loss, but not a valid implicit reward either"。

**9 个 MAJOR（几乎全是 abstract/contribution/caption 的旧措辞漂移，正文 §5/§6 早已改对）**：
- `only open-SOAR improves over base` / `every arm ... collapses` / `It is the only arm that does not collapse` —— 被自己表里 `sft_groundtruth`(0.087)、`reward_weighted`(0.092)、`sft_random`(0.138) **同表证伪** → 全部 scope 到「among the alignment arms」「likelihood/imitation arm」「reward-extreme target」，并点名 `sft_groundtruth` 这个 oracle 对照也 beat base（恰好坐实 two-lever）。
- abstract 说崩坏臂 `stream normally (WER ≈ base)`，但流式 WER 0.212/0.134 ≫ base 0.087 → 改「UTMOS ≈ base；streaming WER far below offline collapse」。
- abstract / contribution(4) 的 two-lever 句把 `dpo_onmanifold`(只 mitigate) 误当 target-lever 的逃逸证据，漏点真正逃逸的 `sft_groundtruth` → 重写点名两杆杠（objective=reward_weighted、target=sft_groundtruth），`dpo_onmanifold` 归位为「needs both」的反证。
- line 142 VoxCPM2 唯一一处没带 dependency caveat（"replicates ... across N=2 implementations"）→ 加「same pairs / statistically dependent / not independent replication」。
- **`4/24 pairs, win-rate 0.33` 算术自相矛盾**（4/24=0.167）：查 `docs/02` 原始记录证实 `4/24 为正`(逐对均值)与 `win_rate 0.33`(逐次 MC draw)是**两个不同的量、都真实**——审稿建议「改 8/24」会引入假数字；正确修法是**消歧措辞**（"better mean margin on 4/24 pairs and better fit in 33% of individual draws"）。

**MINOR**：abstract ρ²≈5%→4.7%（与正文一致）；ρ=−0.217 加注「per-candidate 池重算、与头条 −0.216 一致」；"intrinsic effect"→"settings-matched (reward-specific) effect"；line 134 "monotone in off-manifold"→"tracks reward-extremity"；删 §3 重复的括号；§3 VoxCPM2 加 same-pairs caveat；§6 机理段从 "content typicality" 改为 lead "reward-extremity"；Table 1 去掉误导性 WER 0.094 加粗（非全表最低、且不主张 WER 增益）；intro "off-manifold member of each pair"→"on the largest reward-gap pairs"；`$\mathtt{}$`→`\texttt{}` 统一；`\S6`→`Section~6`。

**裁决（workflow synthesizer）**：硬数字（三张表、dose-response、E2/E3、null-control、VoxCPM2 dependent-evidence scoping）与 canonical **全部自洽**；阻塞项是「正文已改对、headline/caption 没跟上」的机械漂移，无需新实验。修完 + 一次真实 EMNLP 编译确认 ≤8 页 → Findings 有竞争力。
- ⚠️ **长度**：workflow 估「AT RISK」（abstract ~330 词、§6 偏长、若干 caveat 重复 6–7 次）；本机无法编译，已删最明显冗余（§3 重复括号），其余待 Overleaf 真实编译核页数，若超则按 synthesizer 点名的冗余处（conclusion 的 per-component 三元组、decode-path 重述）收。

## 收页 · 8 页超限修复（2026-06-29）

Overleaf 编译实测 12 页（含 refs）：**正文 §6 尾 + 整个 Conclusion 溢到第 9 页**（Conclusion 结束在 p9 右栏上 1/3，Limitations 才在其后开始）→ 正文超约 **0.67 页**。EMNLP：Limitations/Ethics/References **不计页**，故只能从正文（§1–Conclusion）砍。

砍的全是 workflow 早标的真冗余（零信息丢失，约 15 处、≈0.5–0.55 页）：
- **Conclusion 两长段合并成一段**（删掉对正文逐字复述的 per-component 三元组、Pearson 句、各系数重述）——最大单笔。
- **Related Work 的 "Delta." 段**复述 contributions(1)–(5) → 压成一句。
- **§5 开头**把 §4 的 Metrics/Statistics/CI 又抄一遍 → 删，指回 §4。
- **§6.186 机理段尾**重复 §5+Table 2 的 decode-path 流式数字 → 删数字留结论。
- **§6.184 末尾 VoxCPM2 caveat**、**§6.178 UTMOS 16k band 括注**、**§5.134 collapse-control 细节** → 各自压缩，完整版留在免费的 Limitations。
- §3.71 结果预告、§4.79 RQ 枚举、§4.87 gate、§5.138 chunk-boundary 悬空数字、§5.140 套话 → 各微调。

整篇 braces/$ 平衡、零禁用语、4 个 `\ref` 均有 `\label`。**本机装不了 LaTeX，页数须 Overleaf 真实编译核**；若仍超 ~0.1 页，备选再砍：intro 的 VoxCPM2 去重、§6.182 null-control 收紧、§2 收一句。

### 收页 round 2（2026-06-29）：Table 1 全宽→单栏 + 微调

第一轮去冗余后 Overleaf 实测 **11 页**，正文 Conclusion 结束在 **p9 左栏顶部**（仅超 ~0.16 页 / ~16 行）。

**根因（"注意表格格式"）**：**Table 1 原是 `table*`（跨双栏全宽）**，但只有 5 个窄列，内容挤中间、左右大片留白——全宽 float band 白吃一整条页宽，且把相邻栏也占成空白。

**改法**：Table 1 `table*`→单栏 `table`，`{\footnotesize\setlength{\tabcolsep}{4pt} ... }`（字号紧排限定在 tabular 内，**caption 保持正常字号**）。单栏后相邻栏让给正文 → 回收 ~0.18 页（> 0.16 溢出）。另补 intro 的 VoxCPM2 依赖句去重（与 §6.184 / Limitations 重复）做安全垫。整篇 braces/$ 平衡、3 个 `\begin{table}` 均单栏、零 `table*`。

→ 待 Overleaf 再编译确认 Conclusion 落回 p8。若仍差一线，备选：§6.182 null-control 收紧、§2 收一句、Table 2/3 也可同法紧排。

### 收页 round 3（2026-06-29，最后一行级）

Table 1 单栏 + intro 去重后再编译：**§6 整段回到 p8，Conclusion 也几乎全在 p8，只剩最后 2 行溢到 p9 顶**（~0.02 页）。最后一刀：Conclusion 内砍掉复述正文/§6 的 3 处（VoxCPM2 数字、honest-negative 括注 dpo_weronly/reward_weighted_weronly、结尾重复 headline 句、opening 微调）≈ 3 行 > 2 行溢出，留 ~1 行余量。零数字/论证改动，纯删正文里已有的重复。braces/$ 平衡。→ 再编一次应正文 ≤8 页（Conclusion 落 p8、Limitations 起 p9）。

> **✅ 已核验（2026-06-29）**：Overleaf 编译确认正文 §1–Conclusion 完整落在 **8 页内**（Conclusion 结束在 p8 右栏底，Limitations 起于 p9 顶）。EMNLP 8 页硬限制达标，收页收口。

---

## ★4 · DPO-FM β×λ_sft 超参网格（Main-tier 加分；脚手架已建，结果待出）

- **审稿点（Main track）**：论文「向 off-manifold winner 做 DPO-FM 必崩」是在单组 (β=0.1, λ_sft=0.1) 上观察的。换 DPO 强度 β、换 anchor-SFT 权重 λ_sft，崩坏是否依旧？**加大 λ_sft 能否救**？把单点崩坏升为「(β,λ) 网格稳健」的更强主张。
- **脚手架**：`eval/beta_lambda_sweep.py`（emit-configs 派生网格配置 / aggregate 拼 β×λ 网格）+ `run_x15.sh`（一条命令：派生→逐格 train+synth+score→聚合→归档）。配置驱动（β=dpo.beta、λ=dpo.lambda_sft 在 YAML 里），故先派生 `configs/_sweep/<tag>.yaml`。**默认 LoRA base 省卡**。
- **判据**：整张网格 WER 远高于 base(0.112) ⇒ 崩坏对超参稳健（§5.1 升级主张）；某格（多半大 λ_sft）WER 回落 ⇒ 诚实报「强 anchor 可缓冲」，机理不变、措辞软化。
- **命令**：`bash run_x15.sh 0`（或 `BETAS="0.1 0.5" LAMBDAS="0.0 1.0" bash run_x15.sh 0 configs/dpo_lora.yaml`）。前置：pairs/winners 已就绪（第一轮已生成）。成本：默认 9 格 LoRA ~1 天。

## ★5 · VoxCPM2 原生配对（把跨架构复现升级为真正独立的 N=2；脚手架已建）

- **审稿点（§6 复现强度封顶）**：当前 VoxCPM2 复现拿 **dots 生成的同一批 pair** 打分 → predictor(reward gap) 共享、统计依赖，只算「第二个似然模型在同样 pair 上同意」。让 **VoxCPM2 自采样候选 → 同套 reward → 原生偏好对 → 原生对上重跑探针**，则 generator/pair/reward-gap 全独立 ⇒ 独立 N=2。
- **脚手架**：`candidates/generate_candidates_voxcpm.py`（VoxCPM2 自采样，写同款 CandidateRow schema）+ `run_x16.sh`（生成→打分→建对→原生探针→分析→归档）。下游 reward/pairs/probe 按 common.py 的 fid+Row schema 零改动复用。
- **判据**：原生对上 Spearman 反相关**同号显著** ⇒ 独立复现（§6/Limitations 把「supporting, statistically-dependent」升为「independent」）；近零/正 ⇒ 维持 dependent 证据、诚实不升级。
- **命令**：`bash run_x16.sh 0 <与 dots 188 对同一 prompt 集>.jsonl 6`。前置：env/setup_env.sh 已挂 VoxCPM2 + 下权重。

### verify 闭环（2026-06-29，2 路对抗式核对真实 CLI/schema/路径）

★4 干净（integration_ok），1 个 MINOR 已修：完全崩坏的格零 wav→score_synth 不产 summary→原会静默掉成 `—`（抹掉最强崩坏信号）；改为从 `inference/out/<tag>/_synthesis_report.json` 取失败率、显式标 `FAIL`/fail=1.0 并计入裁决。

★5 抓到 **2 个真 MAJOR（不在我标注的外部 seam 内），均已修**：
1. **`pairs/build_preference_pairs.py` 硬编码 dots 候选索引**（`PATHS.candidates_dir/candidates_index.jsonl`，无 `--candidates`）→ VoxCPM 原生对会 join 到 dots 的 text/prompt（缺失则 text="" 静默污染探针）。**已加 `--candidates` 参数 + 缺失即 fail-fast**，run_x16 step 3 传 `${CAND}`。
2. **生成器加载器偏离已核验的 `voxcpm_preprocess`**（用了裸 `from voxcpm import VoxCPM` + 无 kwargs → 会去加载 denoiser 触发 modelscope、triton 依赖）。**已镜像为 `from voxcpm.core import VoxCPM` + `from_pretrained(hf_model_id=..., load_denoiser=False, optimize=False, device=...)`**。
- **seam-2（reward 中间产物）**：verify 确认 `score_{wer,sim,utmos}.py` 写固定名 `reward/out/{wer,sim,utmos}.jsonl`（无 --out），voxcpm 这遍会覆盖 dots 中间产物 → run_x16 step 2 **已自动备份/还原**，最终 `rewards_voxcpm.jsonl` 由 aggregate `--out` 隔离。

> **✅ seam 已核验（2026-06-29, GPU 机器 /data/VoxCPM）**：`generate(text, prompt_wav_path, prompt_text, inference_timesteps, cfg_value)` → 1D float32 ndarray；输出采样率 = `model.tts_model.sample_rate`（README/CLI 官方口径）；加载 `voxcpm.core.VoxCPM` + `load_denoiser=False/optimize=False`。`generate_candidates_voxcpm._voxcpm_synthesize` 已据此定死。**唯一运行前提**：`export VOXCPM_REPO=/data/VoxCPM`（对 run_x16 子进程可见）；VoxCPM 不走 pip、靠 `ensure_voxcpm_importable()` 挂 sys.path。环境须用项目 .venv（base anaconda 有 torch↔torchaudio ABI 不匹配）。

### ★5 结果（2026-06-29，GPU 实跑，**诚实负面：原生不复现**）

VoxCPM2 自采样 188 对（settings-matched：固定 steps=10/cfg=2.0、仅换 seed）→ 同套 reward → 原生偏好对 → 原生 FM margin。`results/out/x16_voxcpm_native_20260629_153551/`：

- **头条 Spearman ρ=+0.105（n.s.，95%CI[−0.053,+0.257]，perm-p=0.155）**；Pearson +0.051（n.s.）。
- 高-gap 子集也不存活：top-50% ρ=+0.163(n.s.)、top-25% ρ=−0.019(n.s.)。
- per-component：SIM_gap −0.586（但 trim-top-5% → +0.044，杠杆驱动）、WER_gap +0.109(n.s.)、UTMOS_gap −0.242(n.s.)。

**裁决（关键）**：dots 的 −0.216 落在本次 CI **之外** → 不是单纯欠功率，而是**真·非复现**。即 **VoxCPM2 自己的 reward-winner 对 VoxCPM2 自己的似然并不系统性 off-manifold**。对照：VoxCPM2 打分 **dots 的对**时是 −0.266（显著）。

⇒ 反相关是 **dots 候选生成分布**的性质（dots 的 reward-winner 是 off-manifold 的、且被第二个模型独立确认为 atypical），**不是连续-latent 的普适律**。

**对论文的影响（建议，待定）**：
- 头条（dots ρ=−0.216）+ 机理 + 2×2 建设性 **全部不动**。
- §6 的 VoxCPM2-on-dots-pairs（−0.266）保留为「第二个 scorer 确认 dots winner 确实 off-manifold」= dependent 证据（已是现框架）。
- **不升级**为 independent N=2（本来就没声称）；**新增**原生-null 作为 scope 边界：把「scope 是 continuous-latent TTS（两模型）」收窄为「在 dots.tts 上证实；可泛化的是**诊断方法**（先测 sign 再 DPO），而非该具体反相关」。
- 这其实**增强可信度**（测了自己 claim 的强版本、诚实报负面），但需改 §6/Limitations/Conclusion 的 scope 措辞——动 positioning，先与作者确认再落刀。

### ★4 状态（2026-06-29，首跑失败 → 已修 base）

- **失败根因**：`run_x15` 原默认 LoRA base（`dpo_lora.yaml`）。LoRA DPO 的 ref 前向走 `self.policy.disable_adapter()`，本机 peft 版本在该上下文调 `get_model_status()` → `No adapter layers found` 抛错。**第一轮 dpo_multiobj 用全参、从未走 LoRA 路 → bug 潜伏至今**。全参路径用独立冻结 `self.ref`、不碰 disable_adapter，验证过能跑。
- **已修**：`run_x15.sh` 默认 base 改 `configs/dpo_multiobj_fullparam.yaml`；LoRA 路留作 future-fix。全参单格较贵 → 建议「十字」5 格（BETAS 三档@λ=0.1 + LAMBDAS 三档@β=0.1）。
- **重跑命令**：`bash run_x15.sh 0`（默认即全参），或缩格 `BETAS="0.1 0.5 2.0" LAMBDAS="0.1" bash run_x15.sh 0`。
> **✅ ★5 已落 .tex（2026-06-29）**：generality 从「continuous-latent TTS（两模型）」收窄为「dots.tts + 可泛化的诊断方法（DPO 前测 sign）」；VoxCPM2-on-dots(−0.266) 保留为「第二模型确认 dots winner off-manifold」的 dependent 证据；Limitations 加原生-null 全文（ρ=+0.105 n.s.，dots −0.216 在其 CI 外）。改了 intro§37 / results§143 / §6 cross-arch / conclusion / Limitations 五处；并精简 §6 的 VoxCPM2 描述 + n=24 pilot 段抵消正文增量。over-scope 措辞（'not dots.tts-specific' / 'scope is continuous-latent TTS'）已清零、braces/$ 平衡。**待 Overleaf 重编确认仍 ≤8 页**（正文净增 ~1 行，若 Conclusion 又溢一两行，备选再压 §6.180 或 §6.186）。

### ★4 结果定性完成（2026-06-30，全参 3×3 实跑 + FAIL 溯源）

`results/out/x15_betalambda_20260629_170718/`。**先消红旗**：原始 dpo_multiobj（论文 0.890 臂）配置是 **β=0.5, λ_sft=0.2**（见 `dpo_multiobj_fullparam.yaml`），**不在网格点**（网格 λ∈{0,0.1,1.0}）。网格把它夹在 (0.5,0.1)=FAIL 与 (0.5,1.0)=好 之间 → **0.890 完全自洽，非偶然/不可复现**。

**WER 网格**（base 0.112）：
```
β\λ_sft   0.0     0.1     1.0
0.1      FAIL    0.121   0.085
0.5      FAIL    FAIL    0.124
2.0      FAIL    FAIL    0.149
```
**FAIL 溯源（GPU 实查 dpo_b0p1_l0p0）**：wav 时长 **71–78 秒**（参考句几秒）、振幅 0.17–0.22 → 模型**永不停、无限复读 = EOS/stopping 通路彻底崩**（论文 decode-path 特异的极端版）。score_synth 崩是因 Whisper 批 padding 撑不住 70s 跑飞音频（mel 7824 vs 7120）——**打分崩是崩坏的症状、非误判**。已修 `score_wer.py`（`chunk_length_s=30` + 逐条退化兜底；正常 <30s 音频零影响，不动已有 Table1 数字）。

**真实发现（须诚实落论文，软化「regardless of hyperparams」）**：
- 崩坏由 **anchor-SFT 强度 λ_sft 主控**：λ=0（无 anchor）全 β 崩；λ=1.0（强 anchor）全 β 救回 ≈base（WER 0.085–0.149、UTMOS 4.17–4.29）；边界随 β 移（β 越大越需更强 anchor）。
- **机理一致且强化**：anchor SFT = CE+EOS-on-winners，专稳 backbone 的 EOS/text 通路；弱 anchor → DPO 往 off-manifold winner 推 → EOS 通路崩（跑飞或 immediate-EOS）；强 anchor → 守住。**= EOS-pathway 崩坏 + anchor 强度作第三稳定杠杆**（与 2×2 的 objective/target 两杠杆并列）。
- ⚠️ 注意两种崩坏模式：原始 dpo_multiobj(0.5,0.2) = immediate-EOS 25% + 高 WER 0.890；★4 的 λ=0 格 = 永不停跑飞。**同为 EOS 通路失败、方向相反**（一个秒停、一个不停）。

**落论文（待重打分后）**：§5.1/附录加 β×λ 网格——「崩坏受 anchor 强度调控（λ=0 崩、λ=1 救）、定位到 EOS 通路、anchor 是第三稳定杠杆」；**收回**「regardless of hyperparameters」；并注「原始 dpo_multiobj 在 (β=0.5,λ=0.2) 崩坏区」。

**下一步（便宜、无需重训）**：修了 score_wer 后，对 5 个 FAIL 格重打分 + 重聚合 → 网格出真实 WER（跑飞臂必然 ≫1）。命令见对话。

### ★4 完整网格 + .tex 整合（2026-06-30，重打分后）

修了 score_wer 后重打分 5 个 FAIL 格 → 真实 WER（跑飞臂 8–20 = 800–2000%、UTMOS 贴地板 1.28–1.38、SIM≈0）。完整 WER 网格：
```
β\λ_sft   0.0     0.1     1.0
0.1       8.04    0.121   0.085
0.5      13.02   20.57    0.124
2.0      19.17   13.95    0.149
```
λ=1.0 全救回 ≈base，λ≤0.1 几乎全崩，**严重度随 λ_sft 单调**；dpo_multiobj(λ=0.2, WER0.890) 正落 (0.5,0.1)=20.57 与 (0.5,1.0)=0.124 之间。

**✅ 已落 .tex**（4 处）：
- §5：加一句「anchor-SFT 强度是第三杠杆、崩坏 graded in λ_sft、强 anchor 救」+ 指 Appendix。
- Limitations：把 graded-vs-divergence 的 **open question 标记为已解决**（β×λ 单调 → graded EOS-pathway destabilization、anchor 可控，非失控发散）；**收回** "harms regardless of objective" → anchor-conditional。
- **新增 Appendix A**（不计页）：β×λ 网格表 `tab:sweep` + 机理段。
- §6：删一处重复的 reward-hacking 数字抵消正文增量。
- braces/$ 平衡、6 个 `\ref` 均有 `\label`。

**净效果**：★4 不再是「漂亮正面」，而是**解决了论文一个 open caveat（graded vs divergence → graded）+ 把崩坏精确定位到 EOS 通路 + 加了 anchor 这第三杠杆**；代价是软化 "regardless of hyperparameters"（已诚实收回）。

> 待 Overleaf 重编确认 ★5+★4 合起来仍 ≤8 页（正文净增约 1 行；appendix/Limitations 免计）。FAIL 格已重打分入 results/out/，建议再跑一次 `beta_lambda_sweep --mode aggregate` 刷新归档报告。

