# 论文骨架（PAPER_SKELETON）

> 工作论文骨架：把每个 section 映射到 `docs/08` / `results/out/` 的具体表/图。空着的 `___` 跑完回填。
> **标题**：*When Likelihood Disagrees with Quality: A Cautionary Diagnosis of DPO for Continuous-Latent Flow-Matching TTS (dots.tts)*
> **定位（已重定位）**：本文**不是**「开源可用 DPO 配方」，而是一份**机理性警示研究 / 可证伪的负面结果**——在连续-latent flow-matching TTS 上，向 automatic-reward 选出的自生成 off-manifold winner 做 self-training 必退化（SFT/DPO 均然），唯 `open_soar` 改善；并给出反相关的机理诊断。统稿全文见 [`PAPER_DRAFT.md`](PAPER_DRAFT.md)。

## Abstract
连续-latent AR TTS（dots.tts）证明了「无离散 token」也能 SOTA，但把**偏好对齐/自训练**直接搬上来是否安全，缺系统证据。我们做了一次受控诊断，得到一个**警示性负面结果**：在 dots.tts 上，向自动多目标奖励（WER+SIM+UTMOS）选出的**自生成 off-manifold winner** 做 self-training **会退化模型，且与训练目标无关**——纯 SFT 与 Diffusion-DPO-on-flow-matching 都崩，崩坏强度随「往 off-manifold 拉的强度」**单调加剧**（dpo_multiobj 灾难崩坏 + 25% 生成失败（生成时子集崩，graded-vs-divergence 证据不一）；sft 崩坏；dpo_weronly ≈base）；**唯独 reward-free、训向真实 on-manifold 目标的 open-SOAR 改善**（UTMOS 4.309→4.408、WER 0.103→0.093）。机理是 **flow-matching 似然与三个 perceptual reward 轴内在反相关**（头条 Spearman ρ=−0.187；偏相关 −0.392，控住全部生成 knob 后存活 → intrinsic；per-component 三轴全负 −0.41/−0.68/−0.60）。我们定性地把病因定到 **preference-likelihood 对齐**（off-manifold winner），而**非** `L_FM` 本身——open-SOAR 用同一损失训出最佳自然度即为证；这是已知 **likelihood-vs-quality / CFG gap** 在 TTS 偏好对齐上的具体体现。经典 reward-hacking 在错配的 DPO-FM 框架内**未复现**，改在非错配的 reward-weighted 回归框架内干净复现。

## 1. Introduction
- 背景：离散 vs 连续之争、dots.tts 的连续-AR、SOAR 未开源（引 `../../学习资料/05` §3.1/§3.13）。
- **空白 → 警示**：连续-AR 的偏好对齐缺系统证据；我们本想给配方，实测发现**直觉路线会崩**，于是把贡献重定位为**诊断**。
- 贡献：①**警示性主结果**——off-manifold self-training 退化、与训练目标无关、单调加剧；②**机理诊断**——FM 似然 vs perceptual reward intrinsic 反相关（偏相关坐实），且病因在 preference-likelihood 而非 `L_FM`；③**建设性对照**——唯 open-SOAR（on-manifold 目标）有效，给出「该怎么做」的方向；④reward-hacking 的诚实裁决（DPO-FM 内未复现，reward-weighted 内干净复现）。

## 2. Related Work
偏好对齐（DPO、Diffusion-DPO[Wallace 2023]）｜likelihood-vs-quality / CFG gap（似然与感知质量背离的已知现象）｜TTS 后训练（Seed-TTS REINFORCE、Qwen3-TTS DPO+GSPO、dots.tts SOAR）｜连续-latent TTS（dots.tts、VoxCPM）。引文核对见 `../../学习资料/02`。

## 3. Method
- 3.1 dots.tts 回顾（Qwen2.5 backbone + DiT flow-matching head + AudioVAE，patch-AR）。〔camera-ready 须核实 backbone 代号〕
- 3.2 **DPO over flow-matching loss**（推导见 `docs/01-METHOD.md`）：隐式奖励 = policy/ref 的 velocity-MSE 之差；共享 `(x0,t)`；anchor SFT 正则。
- 3.3 多目标奖励与偏好对构造；候选来自 guidance×num_steps×speaker_scale 多样池（→ best-vs-worst 选出的 winner 偏 off-manifold）。
- 3.4 open-SOAR：on- vs off-trajectory 的 reward-free 自校正（训向真实 clean 音频=on-manifold）。
- 3.5 诊断框架：FM-margin × reward-gap 偏相关探针（`probe_analysis.py`）+ SFT-target 梯度对照（on/off-manifold）。

## 4. Experimental Setup
数据（VCTK/AISHELL-3/LJSpeech；第一轮 188 LJSpeech pair 训 / 100 句 held-out 评；评测 seed-tts-eval + EmergentTTS）｜模型梯队（base + 4 对齐臂 + 第二轮对照臂）｜指标（离线 WER/SIM/UTMOS + 失败率；流式 TBD）｜算力（2×H100/A6000）。对应 `PROTOCOL.md`。

## 5. Results
- **Table 1（主表，已实测）** [`docs/08`](docs/08-实验结果-Table1.md) / `results/out/main_table.md`：base **0.103/0.891/4.309**；open_soar **0.093/0.844/4.408**（唯一改善）；dpo_weronly **0.115/0.852/4.112**（≈base）；sft **0.545/0.673/3.917**（崩）；dpo_multiobj **0.752/0.365/2.841，n=75 + 25% 生成失败**（灾难）。**失败率必须与 WER 并列。** 结论：**off-manifold self-training 退化、与训练目标无关、随拉力单调加剧。**
- **§5.2 机理（已实测）** `probe_analysis.py` → [`docs/08`](docs/08-实验结果-Table1.md) §8：Spearman ρ=−0.187；偏相关 −0.392（CI[−0.548,−0.209]，intrinsic）；per-component (−WER)−0.41 / SIM −0.68 / UTMOS −0.60。**病因 = preference-likelihood 对齐（off-manifold winner），非 `L_FM`。**
- **§5.3 reward hacking（诚实）** `results/out/reward_hacking.md`：DPO-FM 框架内**未复现**（dpo_weronly WER 0.115≈base、非最低）；**改在 reward_weighted_weronly（X7）干净复现** ___。
- **Table 2（流式）** `results/out/streaming_table.md` + **Fig 1** 首包延迟：**TBD（X8，去 --skip-streaming）** ___。
- **§5.4 泛化** VoxCPM 探针（X10）___ / 多语种（X11）___。
- **Fig 2（机理图）**：单调崩坏序（拉力 → WER/UTMOS）+ FM-margin vs reward-gap 散点。

## 6. Discussion / Limitations
- **强假设守门**：`sft` 崩与小数据 SFT 过拟合/遗忘尚未干净分离（待 X4 SFT-target 对照隔离）；off-manifold 退化目前是强假设。
- `dpo_multiobj` 25% 是 immediate-EOS 的「生成时子集崩」，与 graded（ρ=−0.187）不同质——但 graded degradation vs optimization divergence **证据不一、暂不定论**（X2/X3：训练 `fm_w` 稳 ~0.74、未二次爆、margin 仍小 0.065 → 训练未发散；交叉表 34% vs 48% 偏 divergence；合判生成时子集崩），保守报告、待人工 TB-curve + 听测裁决。
- 失败聚于 `LJ002` 块须诚实归因（评测集本身 89% 为 LJ002、组内 LJ001 0/11 vs LJ002 25/89、仅轻微偏长 1.12×、无位置偏置；非均匀、非「长 prompt 触发」）｜崩坏臂 CI 极大、单说话人 LJSpeech、**无人类听测（ABX/MOS）**→ 量级仅供方向，未做听测前不称 perceptual 退化｜自动奖励噪声｜SIM 在单说话人近 ceiling，−0.68 须标可能被少数对带动、UTMOS −0.60 较实。
- **Phase B（少步蒸馏 + 量化进 8GB）降为远期 future work**：以「Phase A 出可部署配方」为前提，而 Phase A 是警示性负面结果，故不在本文范围；开放问题「对齐/自校正能否在蒸馏后存活」留作 future work。

## 7. Reproducibility
代码 Apache-2.0；第一轮 `docs/03-REPRODUCE.md`、第二轮 `docs/09-第二轮实验程序.md` 端到端命令；`docs/RUN_JOURNAL.md`（由模板填）为一手记录；Table 1 数字逐字精确、GPU 机器回填。
