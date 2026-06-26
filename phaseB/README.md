> 🔻 **现状更新（重定位后）**：原计划的 A→B 主线**已废**——Phase A 的真数据是**负面结果**（DPO-FM 错配、off-manifold self-train 退化），
> 门控判据「A 的对齐增益显著为正」**前提不成立**，故 **Phase B（蒸馏+量化）降为远期 future work、未执行、不在本论文范围**。
> 本目录仅保留为脚手架占位。详见 `docs/02 ADR-8/ADR-11`、`README`、`docs/06`。

# phaseB/ — A→B 主线的「变快变小」一环（门控）

**角色**：把 Phase A 对齐后的模型蒸馏到少步(NFE 1~2) + INT8/INT4 量化，落到 **8GB Turing(RTX2070)**。

**为什么是门控**：用户定的主线是「**先把 A 做出效果，再推进 B**」。B 仅在 A 达标后跑（判据见 `PHASE_B_PROTOCOL.md`）。

**上游**：Phase A 的对齐 ckpt。
**下游**：8GB 可行性结论（对接 `../../04-Windows-RTX2070-部署指南.md`）。

| 文件 | 作用 |
|---|---|
| `distill_aligned.py` | 以对齐模型为 teacher，MeanFlow 蒸到 NFE 1~2 |
| `quantize.py` | INT8/INT4 量化（无 FlashRT，Turing SM7.5） |
| `eval_8gb_matrix.py` | {NFE}×{精度} 的 质量×延迟×显存 Pareto |
| `PHASE_B_PROTOCOL.md` | 门控判据 + 核心开放问题 |

**核心开放问题**：偏好对齐能否在少步蒸馏 + 量化后存活？（B 给第一手证据。）
