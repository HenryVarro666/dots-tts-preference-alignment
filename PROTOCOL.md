# 实验协议（PROTOCOL）

> 这是「跑什么、和谁比、报什么、什么算赢」的单一事实源。任何一次 run 都应能对回本协议的某一行。

## 1. 研究问题

1. **RQ1（正向）**：在连续-latent AR TTS 上做**多目标偏好对齐**（DPO），能否在**不牺牲音色相似度/自然度**的前提下提升鲁棒性（降 hard-case WER、升 UTMOS）？
2. **RQ2（流式）**：对齐带来的增益能否**迁移到 1T1A 流式解码**，且不损首包延迟？
3. **RQ3（负向，关键）**：**只用 WER 当奖励**是否会 **reward-hack**——WER 最低但自然度/表现力退化？（受控复现 05 文档 §3.4 指出、业界只口述的现象。）
4. **RQ4（机理）**：**reward-free 自校正（open-SOAR）** 与 **外部多目标偏好（DPO）** 各自贡献多少？

## 2. 模型梯队（treatments）

| 代号 | 说明 | 作用 |
|---|---|---|
| `base` | `rednote-hilab/dots.tts-base`（不动） | 下界基线 |
| `sft` | 仅用 winners 继续 SFT | **控制变量**：排除「只是多看了好数据」 |
| `open_soar` | reward-free 自校正（我们的开源再现） | 隔离「自校正」单独效果 |
| `dpo_multiobj` | DPO，奖励 = WER+SIM+UTMOS | **头条方法** |
| `dpo_weronly` | DPO，奖励 = 仅 WER | **故意的 reward-hacking 对照臂** |

> 可选对照：`dots.tts-soar`（官方权重）作为「闭源 SOAR」参照，但其训练程序不可复现，仅作上界参考、不参与同条件比较。

## 3. 消融（ablations）

| 维度 | 取值 | 回答什么 |
|---|---|---|
| 奖励组成 | wer / sim / utmos / equal / tuned(默认 0.4/0.3/0.3) | 哪个目标在起作用；reward-hacking 在哪发生 |
| 配对策略 | best_worst(默认) / top2_bottom2 / threshold(τ) | 偏好信号质量 vs 数量 |
| DPO β | 0.1 / 0.5 / 2.0 | 对齐强度 vs 漂移 |
| 微调方式 | full-param(默认 H100) / LoRA(默认 A6000) | 显存-效果权衡 |
| 训练/部署 NFE | 训练候选 NFE∈{8,10,16}，部署 NFE∈{4,8,10,16} | 对齐能否迁移到低 NFE（接 Phase B） |
| 对齐目标 | offline / streaming-aware(stretch) | 流式专用奖励是否更利于流式 |

## 4. 数据

- **训练/偏好 prompt**（公开）：VCTK-0.92(EN) + AISHELL-3(ZH) + LJSpeech-48kHz(EN)。规模默认每语 **~1–2k prompt**，每 prompt **K=4–8** 候选。
- **评测集**：seed-tts-eval `test-en` / `test-zh` / **`test-zh-hard`**（hard-case，对接 05 文档鲁棒性轴）；表现力用 **EmergentTTS-Eval**（需另克隆）。
- **训练集与评测集严格不重叠**；许可/采样率见 `data/LICENSES.md`（标〔需核实〕项跑前确认）。

## 5. 指标（每个 treatment，离线 + 流式双模）

| 指标 | 工具 | 方向 | 模式 |
|---|---|---|---|
| WER (en/zh) | seed-tts-eval（Whisper-v3 / Paraformer） | ↓ | 离线 + 流式 |
| **hard-case WER** | seed-tts-eval `test-zh-hard` | ↓ | 离线 + 流式 |
| SIM | seed-tts-eval（WavLM-SV） | ↑ | 离线 + 流式 |
| UTMOS | `speechmos` | ↑ | 离线 + 流式 |
| EmergentTTS 胜率 | EmergentTTS-Eval（LLM-judge，head-to-head） | ↑ | 离线 |
| 首包延迟 / RTF | `DoubleStreamingSession` 计时 | ↓ | 流式 |
| chunk 边界不连续度 | `eval/streaming_metrics.py` | ↓ | 流式 |

> ⚠️ 评测纪律（沿用仓库 `学习资料/03`）：统一 ASR / speaker encoder / 子集；不同来源数字不直接横比；EmergentTTS 的 LLM-judge 胜率标注「非绝对分」。

## 6. 判定准则（什么算「A 成功」→ 触发 Phase B）

- **主判据（gate）**：`dpo_multiobj` 相对 `base` 在 **test-zh-hard WER 显著下降** 且 **SIM 退化 ≤ 0.01** 且 **UTMOS 不降**。
- **RQ3 判据**：`dpo_weronly` 取得**全场最低 WER**，但 **UTMOS 或 EmergentTTS 胜率显著低于** `dpo_multiobj`。
  - 反证处理：若 `dpo_weronly` 不退化自然度 → **多目标前提不成立**，如实写进结论（负面结果照样有价值）。
- 显著性：自举置信区间（bootstrap CI）或配对检验；样本规模小，**报告 CI 而非只报均值**。

## 7. 每条 claim ↔ 结果表映射（见 `PAPER_SKELETON.md`）

| Claim | 来源表/图 |
|---|---|
| RQ1 正向 | `results/out/main_table.md`（treatments × 指标） |
| RQ2 流式迁移 | `results/out/streaming_table.md` + 首包延迟图 |
| RQ3 reward hacking | `results/out/reward_hacking.md`（WER↓ 但 UTMOS/EmergentTTS↓ 的反差） |
| RQ4 机理拆解 | open_soar vs dpo_multiobj 对照行 |
