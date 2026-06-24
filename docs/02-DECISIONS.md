# 02 · 关键决策记录（ADR：背景 / 选择 / 理由 / 被否方案 / 影响）

> 把每个重要设计抉择固化下来，方便日后回看「当初为什么这么定」。

## ADR-1 用 flow-matching 损失当 DPO 的隐式奖励（而非自训 reward model）
- **背景**：连续-AR 无 token log-prob，标准 DPO 不可用。
- **选择**：Diffusion-DPO 思路——用 velocity-MSE 当 −log p 代理。
- **理由**：直接复用 dots.tts 既有 `_compute_fm_loss_term`，零额外网络；理论有据。
- **被否**：①自训显式 reward model 再 PPO（实现重、易不稳、reward hacking 更难控）；②只做 SFT（学不到「相对偏好」）。
- **影响**：方法轻、可复现；命门转移到「共享 (x0,t)」。

## ADR-2 四前向共享 (x0, t)（最高优先级正确性约束）
- **背景**：L_FM 依赖随机 (x0,t)，不共享则 policy/ref、win/lose 不可比。
- **选择**：`fork_rng` + 固定 RNG state 包住 FM 采样，四前向复用同一份。
- **理由**：Diffusion-DPO 同样共享时间步；不共享 → 似然比失效 → 训不出偏好。
- **被否**：各前向独立采样（错误，会引入方差淹没信号）。
- **影响**：实现需小心；列为 04-RISKS 风险②、代码注释高亮。

## ADR-3 多目标 z-norm 奖励（WER+SIM+UTMOS），并**故意保留**仅-WER 臂
- **背景**：只压 WER 易牺牲自然度/音色（Seed-TTS 口述、05 §3.4）。
- **选择**：默认多目标(0.4/0.3/0.3)，按候选池 z-norm；另设 `dpo_weronly` 对照臂。
- **理由**：多目标回答 RQ1；仅-WER 臂**主动复现 reward hacking**（RQ3），是项目最有冲击力的负面结果。
- **被否**：只报多目标好结果（缺少反例就缺说服力）。
- **影响**：奖励权重成为核心操纵变量；结果三表之一专门给 reward hacking。

## ADR-4 候选多样性靠超参网格 + 逐候选 seed（因 generate 无 seed 参数）
- **背景**：`DotsTtsRuntime.generate()` 无 seed（已读源码确认）。
- **选择**：`num_steps×guidance×speaker` 网格 + 每候选 `seed_everything`。
- **理由**：在 API 约束下务实制造可控差异。
- **被否**：改 dots.tts 源码加 seed（增加耦合与维护成本）。
- **影响**：需在 reward 阶段抽查奖励方差确认多样性足够（风险①）。

## ADR-5 流式设为一等评测轴；流式克隆仅 x-vector
- **背景**：用户重视流式；dots.tts 1T1A 流式**拒 prompt_text**（源码 :117）。
- **选择**：离线+流式双模评测；流式用 prompt-audio-only 条件。
- **理由**：要 claim「增益迁移到流式」须走真实流式路径；x-vector-only 是 dots.tts 的硬约束。
- **被否**：拿离线结果替代流式（解码 schedule 不同，不成立）。
- **影响**：流式/离线 SIM 非严格可比，报告时分开标注。

## ADR-6 full-param(H100) vs LoRA(A6000) 双轨；LoRA 时 ref=disable_adapter
- **背景**：2×A6000(48GB) 全参 DPO + 冻结 ref 偏紧。
- **选择**：默认全参(H100, ZeRO-2)；A6000 走 LoRA，ref=关掉适配器的同一模型。
- **理由**：LoRA 省掉第二份模型，显存大降；同时作 LoRA-vs-full 消融。
- **被否**：一律全参（48GB 档可能 OOM）。
- **影响**：成为可行性保障 + 一个消融维度。

## ADR-7 把对 dots.tts 内部的耦合收敛到单一 `dots_preprocess.py`
- **背景**：AudioVAE 编码、x-vector、token schedule 都依赖 dots.tts 内部。
- **选择**：所有耦合放进一个适配器；上游只认统一接口。
- **理由**：dots.tts 版本变动时只改一处；上游可独立测试。
- **影响**：`dots_preprocess.py` 的几处 TODO 是「方法落地」的接缝，需对照源码补全。

## ADR-8 A→B 分阶段、B 门控
- **背景**：用户明确「先做出 A，效果好再推进 B」。
- **选择**：A 完整脚手架先行；B 仅脚手架 + 门控判据，达标才执行。
- **理由**：避免在未验证对齐增益前就投入蒸馏/量化。
- **影响**：B 的核心问题（对齐能否在蒸馏后存活）作为 future-work 级开放问题。
