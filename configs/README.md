# configs/ — 实验配置（扩展 dots.tts 训练 schema + `dpo:` 块）

每个 YAML = 一个 treatment 或消融臂。脚本 `alignment/train_dots_tts_dpo.py` / `train_open_soar.py` 读取。

| 配置 | 对应 | 作用 |
|---|---|---|
| `dpo_multiobj_fullparam.yaml` | **头条** | 全参 DPO + 多目标奖励（2×H100 默认） |
| `dpo_weronly.yaml` | **reward-hacking 臂** | 仅 WER 奖励 → 复现 reward hacking |
| `dpo_lora.yaml` | A6000 档 | LoRA DPO（ref=disable_adapter，省显存） |
| `open_soar_selfcorrect.yaml` | 中间臂 | reward-free 自校正（SOAR 再现） |
| `sft_baseline.yaml` | 控制变量 | 仅 winners SFT（排除「只是多数据」） |
| `streaming_aware.yaml` | stretch | 前缀/边界敏感奖励 |
| `phaseB/*.yaml` | 门控 | 蒸馏 + 量化 |

**消融维度**（β / 奖励组成 / 配对策略）：用上述配置改对应字段即可，或由 `ablation/run_matrix.py` 注入覆盖。
默认超参取舍见 `../docs/02-DECISIONS.md`。
