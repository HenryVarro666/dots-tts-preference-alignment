# candidates/ — 阶段②：候选生成

**角色**：对每个 prompt 用 dots.tts 采样 **K 个有差异的候选**，供下游打分、配对。

**为什么存在**：DPO 的偏好对必须来自「同一条件下质量有高低差」的候选池。没有差异 → 没有偏好信号。

**上游**：`data/prompts/*_train.jsonl`。
**下游**：`reward/score_*.py` 对每个候选打分。

**产物**：`out/{fid}/cand_{k}.wav`（48kHz）+ `out/candidates_index.jsonl`（每行 = `CandidateRow`）。

**关键设计与坑**：
- `DotsTtsRuntime.generate()` **无 seed 参数**（已读源码 `runtime.py:692` 确认）→ 多样性靠 **超参网格** `num_steps×guidance_scale×speaker_scale` + 逐候选 `seed_everything`。
- **风险**：若网格制造的差异太小 → 候选奖励方差低 → 偏好对噪声大。**跑前用 `reward/aggregate_reward.py` 抽查每 prompt 的奖励离散度**（见 `docs/04-RISKS.md` 风险①）。

**怎么用**：
```bash
python generate_candidates.py --prompts ../data/prompts/ljspeech_train.jsonl --k 6 --limit 50
```
