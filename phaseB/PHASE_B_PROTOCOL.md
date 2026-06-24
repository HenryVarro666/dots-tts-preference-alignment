# Phase B 协议（门控）

> **门控原则**：Phase B 仅在 Phase A **达标**后执行。先把 A 做透，再谈变快变小。

## 进入 Phase B 的判据（gate）

全部满足才进 B：
1. `dpo_multiobj` 相对 `base` 在 **test-zh-hard WER 显著下降**（bootstrap CI 不含 0）。
2. **SIM 退化 ≤ 0.01**，**UTMOS 不降**。
3. reward-hacking 对照成立（`dpo_weronly` WER 更低但 UTMOS/EmergentTTS 退化），即对齐配方确有「多目标价值」。

> 若 gate 不过：停在 A，不做 B；如实记录「对齐未达预期」也是有价值的结论。

## Phase B 步骤

1. **蒸馏**（`distill_aligned.py`）：teacher = 对齐后模型（**非 base**），用 dots.tts MeanFlow 蒸到 **NFE 1~2**。
2. **量化**（`quantize.py`）：INT8/INT4，瞄准 8GB Turing；注意 **FlashRT 需 SM80+，RTX2070 不可用**（对接 `../../04`）。
3. **8GB 矩阵**（`eval_8gb_matrix.py`）：{NFE}×{精度} 的 质量×延迟×显存 Pareto，标 `fits_8gb`。

## Phase B 的核心研究问题（陈述，不预答）

> **偏好对齐能否在少步蒸馏 + 量化后存活？**
> MeanFlow 的区间均值目标可能抹平「对齐过的速度场」；量化也可能吃掉细粒度增益。
> 判据：蒸馏/量化后，对齐相对 base 的 hard-WER/UTMOS 增益**保留比例**。这是 future-work 级别的开放问题，B 给出第一手证据。

## 交付

- `results/out/phaseB_8gb_matrix.json`：哪些配置能在 8GB 跑出可用质量。
- 结论一句话模板：「对齐后模型经 NFE=__ + INT__ 量化后，峰值显存 __GB（<8GB ✓/✗），hard-WER 相对 base 仍降 __%（对齐增益保留 __%）。」
