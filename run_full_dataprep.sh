#!/usr/bin/env bash
# =============================================================================
# run_full_dataprep.sh — 全量数据准备一条龙（适合挂 tmux 后台跑）
#
# 做什么:   候选生成 → 三路打分(WER/SIM/UTMOS) → 多目标聚合 + reward-hacking 聚合
#           → 构造两套偏好对(多目标 / 仅WER)。产物即各 treatment 训练所需的输入。
# 怎么做:   串既有脚本，每步打印耗时；任一步失败即停(set -e)，便于定位。
# 为什么:   候选生成最贵(~3h/188 prompt)，做成一条龙挂后台，跑完即可直接开训各 treatment。
# 对实验意义: 一次产出「多目标」与「仅WER」两套偏好对 → 支撑 RQ1(多目标) 与 RQ3(reward-hacking 对照)。
#
# 用法:     bash run_full_dataprep.sh [PROMPTS_JSONL] [K] [WER_BATCH]
#   默认:   PROMPTS=data/prompts/dots_manifest_train.jsonl  K=6  WER_BATCH=16
# 前置:     环境就绪、dots.tts-base 权重已缓存、prompts 已生成。
# 预期结果: candidates/out/ + reward/out/{wer,sim,utmos,rewards,rewards_weronly}.jsonl
#           + pairs/out/{pairs_best_worst,pairs_weronly}.jsonl
# ⚠️ 本机(macOS/无 GPU)只 bash -n；真正运行在 GPU 机器、建议 tmux。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROMPTS="${1:-data/prompts/dots_manifest_train.jsonl}"
K="${2:-6}"
WER_BATCH="${3:-16}"

ts() { date +"%H:%M:%S"; }   # 仅打印用，与实验无关

echo "[$(ts)] [1/6] 候选生成  prompts=${PROMPTS}  K=${K}  (最耗时, ~3h/188prompt)"
python candidates/generate_candidates.py --prompts "${PROMPTS}" --k "${K}"

echo "[$(ts)] [2/6] WER 打分 (Whisper-large-v3, 批量)"
python reward/score_wer.py --batch-size "${WER_BATCH}"

echo "[$(ts)] [3/6] UTMOS 打分 (torch.hub utmos22_strong)"
python reward/score_utmos.py

echo "[$(ts)] [4/6] SIM 打分 (SpeechBrain ECAPA; 单说话人语料 SIM 区分度有限)"
python reward/score_sim.py --backend ecapa

echo "[$(ts)] [5/6] 聚合奖励: 多目标 + 仅WER(reward-hacking 对照)"
python reward/aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3 \
       --out reward/out/rewards.jsonl
python reward/aggregate_reward.py --w-wer 1 --w-sim 0 --w-utmos 0 \
       --out reward/out/rewards_weronly.jsonl

echo "[$(ts)] [6/6] 构造偏好对: 多目标 / 仅WER"
python pairs/build_preference_pairs.py --rewards reward/out/rewards.jsonl \
       --strategy best_worst --out pairs/out/pairs_best_worst.jsonl
python pairs/build_preference_pairs.py --rewards reward/out/rewards_weronly.jsonl \
       --strategy best_worst --out pairs/out/pairs_weronly.jsonl

echo "[$(ts)] ✓ 数据准备完成。下一步: 依次训练各 treatment（见 docs/03-REPRODUCE.md 第3节）"
echo "  pairs: $(wc -l < pairs/out/pairs_best_worst.jsonl) 对(多目标) / $(wc -l < pairs/out/pairs_weronly.jsonl) 对(仅WER)"
