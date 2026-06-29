#!/usr/bin/env bash
# run_dose.sh — ★1 剂量-反应（训练目标 off-manifold 度 → 崩坏；把「诊断↔崩坏」从相关变因果桥）。
#   做什么: ① winners/losers/random 三类 target 的均值 L_FM 从 E1 的 percand_lfm.csv 抽（免费）；
#           ② groundtruth(真录音) 的 L_FM 用 probe_fm_reward.py --score-audio 算一次（score dots_manifest_train 的 prompt_audio）；
#           ③ dose_response.py 对各 X4 臂 eval WER 做单调性/剂量-反应。
#   用法:   bash run_dose.sh [GPU]
#           默认 GPU=0；若 alignment/out/groundtruth_lfm.csv 已存在则跳过 ② 直接分析（纯 CPU）。
#   前置:   alignment/out/percand_lfm.csv（E1 产出）+ results/out/sft*_nfe10.summary.json（X4 各臂）+ 权重(② 需 GPU)。
#   产物:   results/out/dose_response_<ts>/{report.txt,run.log}
#   裁决:   report.txt 末尾——按 target L_FM 升序 WER 单调上升 ⇒ 剂量-反应成立（claim 4 因果桥）。
#   ⚠️ 本机只 bash -n / py_compile；② 需 CUDA+权重，①③ 纯 CPU。
set -euo pipefail
GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="results/out/dose_response_${TS}"; mkdir -p "${OUT}"
GT="alignment/out/groundtruth_lfm.csv"
if [ ! -f "${GT}" ]; then
  echo "[dose] groundtruth L_FM 不存在 → 对真录音算一次（probe --score-audio，需 GPU）"
  python alignment/probe_fm_reward.py --score-audio data/prompts/dots_manifest_train.jsonl \
    --out-lfm "${GT}" --k 16 2>&1 | tee "${OUT}/score_gt.log" \
    || echo "[dose] groundtruth 评分失败/跳过 → 仅用 winners/losers/random 三点"
fi
python eval/dose_response.py --percand alignment/out/percand_lfm.csv \
  --groundtruth-lfm "${GT}" --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/run.log"
echo "[dose] ✓ -> ${OUT}/report.txt  （看末尾剂量-反应裁决）"
