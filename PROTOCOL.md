# 实验协议（PROTOCOL）

> 这是「跑什么、和谁比、报什么、什么算赢」的单一事实源。任何一次 run 都应能对回本协议的某一行。
>
> **定位（已重定位）**：本项目最初设想为「开源可用的多目标 DPO 配方」，实测后**重定位为机理性警示研究**——核心交付是一份**可证伪的负面结果**：在连续-latent flow-matching TTS 上，向 automatic-reward 选出的自生成 **off-manifold winner** 做 self-training（SFT 或 DPO 均然）会退化模型，只随拉力单调加剧；唯 `open_soar` 改善。下文 RQ/treatments/判定准则按此重写。

## 1. 研究问题（已按警示性诊断重写）

1. **RQ1（自训练退化，头条）**：在连续-latent AR TTS 上，把模型往 automatic-reward 选出的**自生成候选**对齐/自训练，是否会**退化**？退化是否**与训练目标（SFT vs DPO）无关、只随「往 off-manifold winner 拉的强度」单调加剧**？→ **实测：是**（见 §6 Table 1）。
2. **RQ2（机理）**：退化的机理是否是 **flow-matching 似然与 perceptual reward 内在反相关**？反相关是 **intrinsic** 还是候选生成 knob（guidance/num_steps/speaker_scale）的副产品？是 `L_FM` 本身坏、还是 preference-likelihood 对齐坏（off-manifold winner）？
3. **RQ3（reward-hacking，受控）**：经典 reward-hacking（仅 WER 奖励 → WER 最低但自然度退化）能否复现？→ **在错配的 DPO-FM 框架内未复现**（两 DPO 臂都崩、WER-only 未取最低 WER）；**改在非错配的 reward-weighted 回归框架内干净复现**（X7）。
4. **RQ4（建设性）**：什么**有效**？`open_soar`（reward-free、训向真实 on-manifold 目标）为何是唯一改善臂？只用 **on-manifold winner**（X6a `dpo_onmanifold`）能否救回 DPO-FM？
5. **RQ5（泛化）**：反相关与 off-manifold 退化是否**跨架构**（VoxCPM，X10）、**跨语种**（VCTK/AISHELL-3，X11）成立？

## 2. 模型梯队（treatments）

> 第一轮 5 臂（base + 4 对齐臂）+ 第二轮新增对照臂；新臂已全部进 `ablation/run_matrix.py` 的 `TREATMENTS`。

**第一轮（已实测，Table 1）**

| 代号 | 说明 | 作用 |
|---|---|---|
| `base` | `rednote-hilab/dots.tts-base`（不动） | 下界基线 |
| `sft` | 仅用 reward-winners 继续 SFT | 原设为「控制变量」；**实测崩坏**（反转，见 §6） |
| `open_soar` | reward-free 自校正（训向真实 clean 音频=on-manifold） | **唯一改善臂** |
| `dpo_multiobj` | DPO，奖励 = WER+SIM+UTMOS | 原「头条方法」；**实测灾难崩坏 + 25% 发散** |
| `dpo_weronly` | DPO，奖励 = 仅 WER | 原「reward-hacking 对照」；**实测 ≈base、WER-only 未取最低 WER** |

**第二轮（脚手架已建，结果待出；意义见 [`docs/09`](docs/09-第二轮实验程序.md)）**

| 代号 | 说明 | 回答什么 |
|---|---|---|
| `sft_losers` / `sft_random` / `sft_groundtruth` | SFT-target 梯度（管线/步数/损失全同，唯一变量是 target 的 on/off-manifold 位置） | **X4**：隔离「off-manifold 目标退化」vs「小数据 SFT 本身不稳」 |
| `dpo_onmanifold` | 只用 likelihood-过滤后的 **on-manifold winner** 重训 DPO | **X6a**：能否救回 DPO-FM（建设性） |
| `reward_weighted` / `reward_weighted_weronly` | 朝 winner 做 reward 加权回归、**不**做 win/lose 的 `L_FM` 对比 | **X6b/X7**：在非错配框架内干净复现 reward-hacking |

> 可选对照：`dots.tts-soar`（官方权重）作为「闭源 SOAR」参照，训练程序不可复现，仅作上界参考、不参与同条件比较。

## 3. 消融（ablations）

| 维度 | 取值 | 回答什么 |
|---|---|---|
| 奖励组成 | wer / sim / utmos / equal / tuned(默认 0.4/0.3/0.3) | 哪个目标在起作用；reward-hacking 在哪发生 |
| SFT-target on/off-manifold | winners(=best，最 off) / losers(=worst) / random / groundtruth(真实录音=on) | **X4 核心**：退化是否由 target 分布驱动 |
| 候选设置匹配 | 多样池(默认) / `--fixed-setting`（固定 guidance×num_steps×speaker） | **X5**：反相关是否在 settings-matched 下仍存活（intrinsic 复证） |
| 配对策略 | best_worst(默认) / top2_bottom2 / threshold(τ) / **likelihood-filtered(on-manifold)** | 偏好信号质量 vs 数量；X6a on-manifold 过滤 |
| DPO β | 0.1 / 0.5 / 2.0 | 对齐强度 vs 漂移 |
| 微调方式 | full-param(默认 H100) / LoRA(默认 A6000) | 显存-效果权衡 |
| 训练/部署 NFE | 训练候选 NFE∈{8,10,16}，部署 NFE∈{4,8,10,16} | num_steps 是否温和混淆反相关（实测单独 −0.237 显著、被偏相关控住） |
| 对齐目标 | offline / streaming-aware(X8) | 流式专用奖励是否更利于流式 |
| 架构 / 语种 | dots.tts / **VoxCPM(X10)**；EN / **ZH·多语种(X11)** | RQ5 泛化（升架构级/多语种） |

## 4. 数据

- **训练/偏好 prompt**（公开）：VCTK-0.92(EN) + AISHELL-3(ZH) + LJSpeech-48kHz(EN)。规模默认每语 **~1–2k prompt**，每 prompt **K=4–8** 候选。第一轮 Table 1 训练用 **188 单说话人 LJSpeech** pair（小数据——见 caveat）。
- **评测集**：第一轮 Table 1 用 **100 句 LJSpeech held-out**（含 `LJ001`/`LJ002` 块）；扩展用 seed-tts-eval `test-en` / `test-zh` / **`test-zh-hard`**；表现力用 **EmergentTTS-Eval**（X9，需另克隆）。
- **训练集与评测集严格不重叠**；许可/采样率见 `data/LICENSES.md`（标〔需核实〕项跑前确认）。

## 5. 指标（每个 treatment，离线 + 流式双模）

| 指标 | 工具 | 方向 | 模式 | 状态 |
|---|---|---|---|---|
| WER (en/zh) | seed-tts-eval / score_synth（Whisper-large-v3 / Paraformer） | ↓ | 离线 + 流式 | 离线✓ |
| **生成失败率** | `_synthesis_report.json` → `aggregate_results.py` 入主表 | ↓ | 离线 + 流式 | ✓（与 WER 并列） |
| **hard-case WER** | seed-tts-eval `test-zh-hard` | ↓ | 离线 + 流式 | planned |
| SIM | seed-tts-eval / score_synth（ecapa / WavLM-SV） | ↑ | 离线 + 流式 | 离线✓ |
| UTMOS | `speechmos`（utmos22_strong） | ↑ | 离线 + 流式 | 离线✓ |
| EmergentTTS 胜率 | EmergentTTS-Eval（LLM-judge，head-to-head） | ↑ | 离线 | **未跑（X9）** |
| 首包延迟 / RTF / chunk 边界不连续度 | `DoubleStreamingSession` 计时 / `eval/streaming_metrics.py` | ↓ | 流式 | **TBD（X8，第一轮 --skip-streaming）** |

> ⚠️ 评测纪律（沿用仓库 `学习资料/03`）：统一 ASR / speaker encoder / 子集；不同来源数字不直接横比；EmergentTTS 的 LLM-judge 胜率标「非绝对分」。**无人类听测前不称 perceptual/naturalness 退化。**

## 6. 关键结果（第一轮裁决，n=100 离线，bootstrap 95% CI）

**Table 1**（offline，100 句 LJSpeech held-out，nfe=10，skip-streaming）

| 臂 | WER ↓ | SIM ↑ | UTMOS ↑ | n | 裁决 |
|---|---|---|---|---|---|
| base | 0.103 | 0.891 | 4.309 | 100 | 基线 |
| open_soar | **0.093** | 0.844 | **4.408** | 100 | **唯一改善** |
| dpo_weronly | 0.115 | 0.852 | 4.112 | 100 | ≈base |
| sft | 0.545 | 0.673 | 3.917 | 100 | **崩坏（反转）** |
| dpo_multiobj | 0.752 | 0.365 | 2.841 | **75** | **灾难 + 25% 生成失败** |

- **单调崩坏序**：拉向 on-manifold（open_soar，改善）→ 弱拉 reward-winner（dpo_weronly，≈base）→ 强拉 reward-winner（sft，崩）→ 最强拉 + 反向梯度（dpo_multiobj，灾难）。
- **机理**：头条 Spearman **ρ=−0.187**；偏相关 **−0.392**（控住全部生成 knob 后存活 → **intrinsic**）；per-component reward 对齐三轴全负 **(−WER)−0.41 / SIM −0.68 / UTMOS −0.60**（SIM/UTMOS 比 WER 强）。**`L_FM` 是有效似然代理**（open_soar 同损失训满分为证），失败在 **preference-likelihood 对齐（off-manifold winner）**=已知 likelihood-vs-quality/CFG gap。
- **caveat**：`sft` 崩与小数据 SFT 不稳混淆（待 X4 隔离）；`dpo_multiobj` 25%=发散非 graded；失败聚集 `LJ002`；无人类听测。

裁决全文见 [`docs/08-实验结果-Table1.md`](docs/08-实验结果-Table1.md)；机理见 [`docs/06-核心发现-FM损失与奖励错配.md`](docs/06-核心发现-FM损失与奖励错配.md)。

## 7. 判定准则（什么算结论成立 / 反证处理）

- **RQ1（自训练退化）**：四对齐臂的崩坏强度与「往 off-manifold winner 拉的强度」**单调对应**，且唯 `open_soar` 相对 base 净改善（UTMOS↑ 且 WER 不升）。→ **已成立**。**反证守门**：`sft` 崩须由 **X4（SFT-target 对照）** 隔离——若 `sft_random` 也崩、`sft_groundtruth` 改善，则「off-manifold 目标」为因坐实；若 random/groundtruth 都崩，则混入「小数据 SFT 不稳」，**降级表述**为强假设。
- **RQ2（intrinsic 机理）**：偏相关控住全部生成 knob 后反相关存活（−0.392，CI 不含 0）。→ **已成立**（intrinsic）。**C6**：intrinsic ≠ intrinsic-to-`L_FM`，诊断仍是 preference-likelihood/off-manifold。
- **RQ3（reward-hacking）**：DPO-FM 框架内**未复现**（如实写）；判据移到 **X7 reward_weighted_weronly**——WER-only 取得最低 WER 但 UTMOS/EmergentTTS 显著低于 multiobj，方算干净复现。
- **RQ4（建设性）**：`dpo_onmanifold`（X6a）若相对 `dpo_multiobj` 显著回升（不崩、不发散），则「off-manifold winner 是病因」得正向佐证。
- 显著性：自举置信区间（bootstrap CI）或配对检验；崩坏臂 CI 极大、单说话人、无人类听测 → **量级仅供方向**，报告 CI 而非只报均值。

## 8. 每条 claim ↔ 结果表映射（见 `PAPER_SKELETON.md`）

| Claim | 来源表/图 |
|---|---|
| RQ1 自训练退化 + 单调崩坏序 | `docs/08` Table 1 / `results/out/main_table.md`（含失败率列） |
| RQ2 intrinsic 反相关 | `alignment/probe_analysis.py` 输出（偏相关 −0.392 / per-component / 杠杆）→ `docs/08` §8 |
| RQ3 reward-hacking（未复现→X7 干净复现） | `results/out/reward_hacking.md`（reward_weighted_weronly 臂） |
| RQ4 建设性（open_soar 唯一改善 / dpo_onmanifold 救回） | open_soar vs dpo_multiobj vs dpo_onmanifold 对照行 |
| RQ5 泛化 | VoxCPM 探针（X10）/ 多语种表（X11） |
| 流式迁移 | `results/out/streaming_table.md`（X8，去 --skip-streaming） |
