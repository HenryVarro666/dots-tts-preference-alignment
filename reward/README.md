# reward/ — 阶段③：多目标奖励打分

**角色**：给每个候选打三个客观分（WER↓ / SIM↑ / UTMOS↑），再 z-norm 聚合成单一奖励 `R`。

**为什么存在**：DPO 需要「谁更好」的标量信号。奖励的**组成**就是实验的操纵变量——多目标 vs 仅 WER 的对照，正是 RQ1（不伤 SIM/自然度地提鲁棒）与 RQ3（仅 WER 会 reward-hack）的核心。

**上游**：`candidates/out/candidates_index.jsonl`。
**下游**：`pairs/build_preference_pairs.py` 按 `R` 排序构造偏好对。

**产物**：`out/{wer,sim,utmos}.jsonl` → `out/rewards.jsonl`（`RewardRow`，含 z 分量便于审计）。

**关键设计**：
- **按每个 prompt 的候选池**做 z-normalize → 三种量纲可比，且奖励聚焦「同一 prompt 内的相对好坏」（DPO 要的就是相对信号）。
- WER 越低越好 → 聚合时取 `−WER_z`。
- `aggregate_reward.py` 会打印**候选池奖励方差**：太小说明候选差异不足、偏好信号弱（回到 `candidates/` 加大网格）。

**怎么用**：
```bash
python score_wer.py && python score_sim.py && python score_utmos.py
python aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3   # 多目标(默认)
python aggregate_reward.py --w-wer 1 --w-sim 0 --w-utmos 0 --out out/rewards_weronly.jsonl  # reward-hacking 臂
```
