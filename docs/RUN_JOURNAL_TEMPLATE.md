# 运行日志模板（RUN_JOURNAL）

> 用法：复制本文件为 `RUN_JOURNAL.md`，在 GPU 上**逐步回填**。把「记录每一步」从设计期延续到运行期——
> 最终它就是论文复现附录与 debug 的一手材料。每步记：实际命令 / 关键数字 / 观察 / 与预期的偏差 / 处置。

## 环境
- 机器 / GPU：______（如 2×H100-80GB）
- 日期：______
- dots.tts commit / 权重版本：______
- 关键依赖版本：torch ___ transformers ___ accelerate ___ speechmos ___

## ① 数据
| 语料 | 命令 | train/eval 条数 | 采样率核实 | 观察 |
|---|---|---|---|---|
| LJSpeech | | | | |
| VCTK | | | | |
| AISHELL-3 | | | | |

## ② 候选生成
- 命令：______  K=___  网格：______
- 每 prompt 候选数：___　生成 RTF：___
- 观察：______

## ③ 奖励
- WER/SIM/UTMOS 命令与耗时：______
- **候选池平均奖励方差**：___（是否够大？不够→处置：______）
- 多目标权重：______　reward-hacking 臂权重：(1,0,0)

## ④ 偏好对
- 策略 / 对数：best_worst ___　top2_bottom2 ___　threshold(τ=__) ___
- 人工抽听 win/lose 是否合理：______

## ⑤ 对齐训练（每 treatment 一行）
| treatment | config | 步数 | 最终 loss | reward_margin | reward_acc | 显存 | 备注 |
|---|---|---|---|---|---|---|---|
| sft | | | | — | — | | |
| open_soar | | | | | | | |
| dpo_multiobj | | | | | | | |
| dpo_weronly | | | | | | | |

- 命门核查：fork_rng 共享 (x0,t) 是否生效（reward_margin 上升？）：______

## ⑥⑦ 合成 + 评测（主表，离线）
| treatment | NFE | WER↓ | SIM↑ | UTMOS↑ | EmergentTTS↑ |
|---|---|---|---|---|---|
| base | 10 | | | | |
| sft | 10 | | | | |
| open_soar | 10 | | | | |
| dpo_multiobj | 10 | | | | |
| dpo_weronly | 10 | | | | |

### 流式表
| treatment | 首包 p50(ms) | RTF | 边界不连续度 | 流式 WER | 流式 SIM |
|---|---|---|---|---|---|
| | | | | | |

## 判定（→ Phase B 门控）
- dpo_multiobj vs base：hard-WER 降？(CI) ___　SIM Δ ___　UTMOS ___
- reward hacking 成立？dpo_weronly WER 最低但 UTMOS/Emergent 退化？___
- **是否进入 Phase B**：是 / 否（理由：______）

## Phase B（若进入）
| NFE | 精度 | WER | SIM | UTMOS | 首包(ms) | 峰值显存(GB) | <8GB? | 对齐增益保留% |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## 结论一句话
______（对应 PAPER_SKELETON 的 Abstract / 各 claim）
