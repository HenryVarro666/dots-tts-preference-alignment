# 论文骨架（PAPER_SKELETON）

> 工作论文骨架：把每个 section 映射到 `results/out/` 的具体表/图。空着的 `___` 跑完回填。
> 标题候选：*Self-Aligned Continuous-Latent TTS: An Open, Reward-Hacking-Resistant Preference Optimization Recipe for dots.tts*

## Abstract
连续-latent AR TTS（dots.tts）证明了「无离散 token」也能 SOTA，但其 SOAR 自校正未开源、且连续-AR 的偏好对齐是空白。我们给出 **DPO-on-flow-matching** 的开源配方，用自动多目标奖励（WER+SIM+UTMOS）构造偏好对，在**不牺牲 SIM/自然度**下提升鲁棒性，并**受控复现 reward hacking**（仅 WER 奖励）。增益在 **1T1A 流式**下保持。

## 1. Introduction
- 背景：离散 vs 连续之争、dots.tts 的连续-AR、SOAR 未开源（引 `../../学习资料/05` §3.1/§3.13）。
- 空白：连续-AR 的 DPO、reward hacking 缺系统证据。
- 贡献：①开源 DPO-on-FM 配方；②open-SOAR 再现；③reward-hacking 受控复现（负面结果）；④流式迁移验证。

## 2. Related Work
偏好对齐（DPO、Diffusion-DPO[Wallace 2023]）｜TTS 后训练（Seed-TTS REINFORCE、Qwen3-TTS DPO+GSPO、dots.tts SOAR）｜连续-latent TTS（dots.tts、VoxCPM）。引文核对见 `../../学习资料/02`。

## 3. Method
- 3.1 dots.tts 回顾（Qwen2.5 backbone + DiT flow-matching head + AudioVAE，patch-AR）。
- 3.2 **DPO over flow-matching loss**（推导见 `docs/01-METHOD.md`）：隐式奖励 = policy/ref 的 velocity-MSE 之差；共享 `(x0,t)`；anchor SFT 正则。
- 3.3 多目标奖励与偏好对构造。
- 3.4 open-SOAR：on- vs off-trajectory 的 reward-free 自校正。

## 4. Experimental Setup
数据（VCTK/AISHELL-3/LJSpeech；评测 seed-tts-eval + EmergentTTS）｜模型梯队｜指标（离线+流式）｜算力（2×H100/A6000）。对应 `PROTOCOL.md`。

## 5. Results
- **Table 1（主表）** `results/out/main_table.md`：treatments × {WER, hard-WER, SIM, UTMOS, EmergentTTS}。结论：`dpo_multiobj` hard-WER ___，SIM Δ ___，UTMOS ___。
- **Table 2（流式）** `results/out/streaming_table.md` + **Fig 1** 首包延迟：增益迁移 ___，延迟不变 ___。
- **Table 3（reward hacking）** `results/out/reward_hacking.md`：`dpo_weronly` WER 最低 ___ 但 UTMOS/EmergentTTS ___。**核心负面结果。**
- **Table 4（消融）**：β / 奖励组成 / 配对策略 / NFE 迁移。
- **Fig 2（机理）**：open_soar vs dpo_multiobj 在鲁棒性 vs 自然度上的分解。

## 6. Discussion / Limitations
小数据规模（公开集、claim 是 delta 非绝对 SOTA）｜自动奖励噪声｜流式仅 x-vector 克隆｜Phase B 的「对齐能否在蒸馏后存活」留作 future work。

## 7. Reproducibility
代码 Apache-2.0；`docs/03-REPRODUCE.md` 端到端命令；`docs/RUN_JOURNAL.md`（由模板填）为一手记录。
