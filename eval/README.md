# eval/ — 阶段⑦：评测（离线 + 流式 双模）

**角色**：把合成音频转成可比的硬指标，覆盖**可懂(WER)/相似(SIM)/自然(UTMOS)/表现力(EmergentTTS)/流式(延迟+边界)**。

**为什么存在**：所有 RQ 的证据都在这里产生。评测纪律（沿用 `../../学习资料/03`）：统一 ASR/speaker-encoder/子集，不同来源数字不混比，LLM-judge 胜率标「非绝对分」。

**上游**：`inference/`（离线 + 流式合成产物）。
**下游**：`results/aggregate_results.py`。

| 文件 | 指标 | 备注 |
|---|---|---|
| `make_seedtts_manifest.py` | — | 把评测清单转 seed-tts-eval 的 5 字段 meta |
| `run_seedtts_eval.sh` | WER + SIM | 无 sudo / 单卡适配（原脚本耦合 ByteDance 集群） |
| `run_utmos.py` | UTMOS | reward-hacking 探针（WER↓ 但自然度↓） |
| `run_emergent_tts.py` | 表现力胜率 | 需另 clone；预期对齐有表现力 tradeoff（05 line 325） |
| `streaming_metrics.py` | 首包延迟/RTF/边界不连续度 | 流式专属 |

**怎么用**：
```bash
python make_seedtts_manifest.py --manifest <eval.jsonl> --out eval_sets/zh.meta.lst
bash run_seedtts_eval.sh eval_sets/zh.meta.lst ../inference/out/dpo_multiobj zh ../.ckpt/wavlm_large_finetune.pth
python run_utmos.py --synth-dir ../inference/out/dpo_multiobj --treatment dpo_multiobj
python run_emergent_tts.py --ckpt <ckpt> --treatment dpo_multiobj
python streaming_metrics.py --streaming-dir ../inference/out_streaming/dpo_multiobj --treatment dpo_multiobj
```
