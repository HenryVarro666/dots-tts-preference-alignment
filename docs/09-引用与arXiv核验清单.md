# 09 · 引用与 arXiv 核验清单（camera-ready / X13）

> **用途**：PAPER_DRAFT.md 目前用 `(Author, YEAR)` 行内引用、尚无 `.bib`。本表把每条引用映射到**已核验**的 arXiv id + 标题，供建 bib 用。
> **核验方式（遵守仓库铁律「写前必核」）**：`curl https://arxiv.org/abs/<id>` 取 `<title>` 与目标论文逐条比对，全部 title 命中。核验日期 **2026-06-28**（本机 macOS、无 GPU，仅做链接核验）。
> **复核命令**：`curl -L -sS "https://arxiv.org/abs/<id>" | grep -oE '<title>[^<]*</title>'`

## A. 行内 (Author, YEAR) 引用 — 全部已核验 ✓

| 引用 key | arXiv id | 标题（核验命中） | 出现处 |
|---|---|---|---|
| Rafailov et al., 2023（DPO） | **2305.18290** | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | §1, §3.2 |
| Wallace et al., 2023（Diffusion-DPO） | **2311.12908** | Diffusion Model Alignment Using Direct Preference Optimization | §1, §3.2 |
| Theis et al., 2016（likelihood≠quality） | **1511.01844** | A note on the evaluation of generative models | Abstract, §6.7 |
| Ho & Salimans, 2022（CFG） | **2207.12598** | Classifier-Free Diffusion Guidance | §6.7（off-manifold/CFG gap） |
| Anastassiou et al., 2024（Seed-TTS / seed-tts-eval） | **2406.02430** | Seed-TTS: A Family of High-Quality Versatile Speech Generation Models | §1, §4 |
| Lipman et al., 2023（Flow Matching） | **2210.02747** | Flow Matching for Generative Modeling | §2, §3 |
| Wang et al., 2023（VALL-E） | **2301.02111** | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers | §1 |
| Shen et al., 2023（NaturalSpeech 2） | **2304.09116** | NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers | §2.2 |
| Ju et al., 2024（NaturalSpeech 3） | **2403.03100** | NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models | §2.2 |
| Le et al., 2023（Voicebox） | **2306.15687** | Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale | §2.2 |
| Ouyang et al., 2022（InstructGPT/RLHF） | **2203.02155** | Training language models to follow instructions with human feedback | §1, §2 |
| Christiano et al., 2017（RLHF preferences） | **1706.03741** | Deep reinforcement learning from human preferences | §2 |
| Gao et al., 2023（reward over-optimization） | **2210.10760** | Scaling Laws for Reward Model Overoptimization | §2（PPO over-opt） |
| Borsos et al., 2023（SoundStorm） | **2305.09636** | SoundStorm: Efficient Parallel Audio Generation | §1 |
| Azar et al., 2024（IPO / preference theory） | **2310.12036** | A General Theoretical Paradigm to Understand Learning from Human Preferences | §2/§3 |
| **Chen et al., 2024a（VALL-E 2）** | **2406.05370** | VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers | §2.1 |
| **Chen et al., 2024b（F5-TTS）** | **2410.06885** | F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching | §2.2 |

## B. 评测 / 组件系统（命名引用）— 全部已核验 ✓

| 系统 | arXiv id | 标题 | 备注 |
|---|---|---|---|
| UTMOS（Saeki et al., 2022） | **2204.02152** | UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022 | 自然度自动指标 |
| ECAPA-TDNN（Desplanques et al., 2020） | **2005.07143** | ECAPA-TDNN: Emphasized Channel Attention… for Speaker Verification | 本 pilot 的 SIM 后端 |
| Whisper（Radford et al., 2023） | **2212.04356** | Robust Speech Recognition via Large-Scale Weak Supervision | WER 后端（large-v3） |
| WavLM（Chen et al., 2022） | **2110.13900** | WavLM: Large-Scale Self-Supervised Pre-Training… | 仅作 scale-up SV 协议提及（本 pilot 未用） |
| SpeechBrain（Ravanelli et al., 2021） | **2106.04624** | SpeechBrain: A General-Purpose Speech Toolkit | ECAPA 实现来源 |
| EmergentTTS-Eval（2025） | **2505.23009** | EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic… Model-as-a-Judge | §7 limitation（X9，judge API 待接） |
| GSPO（2025） | **2507.18071** | Group Sequence Policy Optimization | §1（Qwen3-TTS 的 DPO/GSPO 适配） |
| VITS（Kim et al., 2021） | **2106.06103** | Conditional VAE with Adversarial Learning for End-to-End TTS | §1 背景 |
| FastSpeech2（Ren et al., 2021） | **2006.04558** | FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | §1 背景 |

## C. 本研究的两个目标模型 — 已核验，VoxCPM 版本 **已锁定 = VoxCPM2** ✅

| 系统 | arXiv id | 标题 | 状态 |
|---|---|---|---|
| **dots.tts**（rednote-hilab） | **2606.07080** | dots.tts Technical Report | ✓ 核验命中；正文「dots.tts (…, 2026)」一致 |
| **VoxCPM2**（2B，OpenBMB）✅采用 | **2606.06928** | VoxCPM2 Technical Report | ✓ 核验命中（**2026**、2B、tokenizer-free diffusion-AR）；**X10 实跑版本** |
| ~~VoxCPM 原版（0.5B）~~ 未用 | ~~2509.24650~~ | VoxCPM: Tokenizer-Free TTS… (2025) | 不引——X10 代码硬拒 v1 |

> **✅ 已从代码定版（非作者凭记忆）**：X10 适配器**硬性要求 VoxCPM2**——
> `run_x10.sh` / `probe_fm_reward_voxcpm.py` / `setup_env.sh` 默认 `--ckpt openbmb/VoxCPM2`；
> `voxcpm_preprocess.py:222` `from voxcpm.model.voxcpm2 import VoxCPM2Model` 且 `isinstance` 断言、**显式报错拒绝 v1**（"只适配 VoxCPM2…请用 openbmb/VoxCPM2"）。
> ⇒ 引用 **2606.06928**、正文「2B / 2026」**正确**；原版 0.5B (2509.24650) **不引**（refs.bib 已删该条）。
> **一处已修正**：正文原写 VoxCPM「on a MiniCPM-4 backbone」——那是**原版 0.5B** 的架构，**VoxCPM2 报告未确认 MiniCPM-4 backbone**（仅述 hierarchical diffusion-AR / tokenizer-free / 2B）。已据此把 §2.3 与 §6.7 的 backbone 措辞改为已核验的「tokenizer-free, hierarchical diffusion-AR + own AudioVAE」，并将全文 27 处「VoxCPM」统一为 **VoxCPM2**。

## D. 非 arXiv / 无需 arXiv 的引用

| 引用 | 处理 |
|---|---|
| **Qwen3-TTS** | Qwen 团队 tech report / blog，**无独立 arXiv**。引 Qwen 官方报告 URL，或仅作系统名提及（其 DPO/GSPO 适配另引 GSPO 2507.18071）。〔待核实官方链接〕 |
| **SOAR**（dots.tts 的自纠后训练） | 正文明确「**未公开 / 无文档**」，故**不单独引**，归入 dots.tts 报告（2606.07080）。本研究的 **open-SOAR 是其开放近似**，为本文贡献、无外部引用。 |
| **REINFORCE**（Williams, 1992） | 经典期刊论文（Machine Learning, 1992），**pre-arXiv**；按期刊引，非 arXiv id。 |

## E. 可选组件引用（按是否在正文「命名为方法」决定要不要建 bib 条目）— 已核验 ✓

| 组件 | arXiv id | 标题 | 是否建议引 |
|---|---|---|---|
| DiT（Peebles & Xie, 2023） | **2212.09748** | Scalable Diffusion Models with Transformers | **建议**（dots.tts/VoxCPM 的 flow-matching head 核心、正文多处命名） |
| Qwen2.5（Qwen, 2024） | **2412.15115** | Qwen2.5 Technical Report | **建议**（dots.tts backbone） |
| PPO（Schulman et al., 2017） | **1707.06347** | Proximal Policy Optimization Algorithms | 可选（§2 仅命名一次） |
| FSQ（Mentzer et al., 2023） | **2309.15505** | Finite Scalar Quantization: VQ-VAE Made Simple | 可选（描述 VoxCPM 时命名） |

## F. Camera-ready 动作清单（X13）

1. **`.bib` 已生成 → `refs.bib`**（仓库根，33 条；title/author 直接从 arXiv meta 抓取、非凭记忆）。直接 `\bibliography{refs}` 引用即可。其中 `voxcpm2025`/`voxcpm22026` 两条**二选一保留**（见 C 节），`chen2024valle2`/`chen2024f5tts` 已拆分。Qwen3-TTS / Williams1992-REINFORCE 仍需**手动**补非-arXiv 条目。
2. **拆分 Chen 2024 引用冲突**：正文 `(Chen et al., 2024)` 现同时指 VALL-E 2 与 F5-TTS → 改为 **`Chen et al., 2024a`（VALL-E 2）/ `2024b`（F5-TTS）**，并核对两处 §2.1/§2.2 行内 key。
3. ~~定 VoxCPM 版本~~ **✅ 已完成**：从 X10 代码锁定 = **VoxCPM2 (2606.06928)**；正文 27 处统一为「VoxCPM2」、删 MiniCPM-4 backbone 臆断、refs.bib 删原版条目。
4. **Qwen3-TTS 官方链接** 〔待核实〕。
5. 全文 `(Author, YEAR)` → `\citep{key}`（投会模板）替换；删除 L7 头部注释里的 `[CIT]/[*-placeholder]` 占位说明。
