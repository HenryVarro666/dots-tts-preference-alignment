#!/usr/bin/env bash
# run_dose.sh — ★1 剂量-反应（训练目标 off-manifold 度 → 崩坏）。
#   ⚠️ 关键: X4 的 sft/sft_losers/sft_random 臂训练用的是 **grid 池**（winners_only←pairs_best_worst←rewards.jsonl），
#            **不是 settings-matched**。故 per-candidate L_FM 必须用 grid rewards 重算（早前用 rewards_fixed 是错池）。
#   用法:   bash run_dose.sh [GPU] [REWARDS]
#           默认 GPU=0, REWARDS=reward/out/rewards.jsonl（grid，= X4 target 的池）。
#   前置:   candidates/out/candidates_index.jsonl + REWARDS + 权重 + results/out/sft*_nfe10.summary.json + dots_manifest_train.jsonl。
#   产物:   results/out/dose_response_<ts>/report.txt（+ alignment/out/percand_lfm_grid.csv、groundtruth_lfm.csv）
#   裁决:   按 target L_FM 升序 WER 单调上升 + Spearman>0 ⇒ 剂量-反应成立（claim 4 因果桥）；否则 L_FM 不解释崩坏 → 诚实降级机制。
set -euo pipefail
GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
REWARDS="${2:-reward/out/rewards.jsonl}"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="results/out/dose_response_${TS}"; mkdir -p "${OUT}"
PERCAND="alignment/out/percand_lfm_grid.csv"
GT="alignment/out/groundtruth_lfm.csv"
echo "[dose] REWARDS(grid)=${REWARDS}  -> ${OUT}"
# 1) per-candidate L_FM on the GRID pool（= X4 target 的池；no_grad、无新合成）
if [ -f "${PERCAND}" ] && [ "$(wc -l < "${PERCAND}")" -lt 200 ]; then echo "[dose] 删除残缺的 24-prompt percand，重算全量"; rm -f "${PERCAND}"; fi
if [ ! -f "${PERCAND}" ]; then
  echo "[dose] 1/3 per-candidate L_FM（grid 池，全量 --n 0）"
  python alignment/probe_fm_reward.py --emit-per-candidate --rewards "${REWARDS}" \
    --candidates candidates/out/candidates_index.jsonl --out-percand "${PERCAND}" --n 0 --k 16 \
    2>&1 | tee "${OUT}/emit_grid.log"
fi
# 2) groundtruth(真录音) L_FM（若已存则跳过）
if [ ! -f "${GT}" ]; then
  echo "[dose] 2/3 groundtruth 真录音 L_FM"
  python alignment/probe_fm_reward.py --score-audio data/prompts/dots_manifest_train.jsonl \
    --out-lfm "${GT}" --k 16 2>&1 | tee "${OUT}/score_gt.log" || echo "[dose] groundtruth 跳过 → 仅 3 点"
fi
# 3) dose-response（用 grid percand）
echo "[dose] 3/3 dose-response 分析"
python eval/dose_response.py --percand "${PERCAND}" --groundtruth-lfm "${GT}" \
  --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/run.log"
echo "[dose] ✓ -> ${OUT}/report.txt"
