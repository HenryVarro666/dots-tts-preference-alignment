#!/usr/bin/env bash
# run_e3.sh — E3 再合成方差（确认 open-SOAR 唯一正面结果是否清出 base 再合成噪声带）。
#   做什么: 对 base 与 open_soar 各独立**再合成 REPS 次**（generate() 无 seed → 每次不同），每次评测 WER/UTMOS，
#           报每臂均值±SD + base 再合成带，判 open_soar 是否清出该带。
#   用法:   bash run_e3.sh [GPU] [MANIFEST] [REPS] [NFE]
#           默认 GPU=0, MANIFEST=data/prompts/eval_big.jsonl, REPS=5, NFE=10
#   前置:   base + open_soar 的对齐 ckpt + manifest + 权重；GPU。
#   产物:   results/out/e3_resynth_<ts>/{report.txt,run.log}
#   裁决:   report.txt 末尾 [E3 裁决]——open_soar 均值清出 base ±1SD 带 ⇒ 真增益；否则「不退化」。
#   ⚠️ 本机只 bash -n / py_compile；运行需 CUDA + 权重。约 REPS×2 次合成+评测，小时级。
set -euo pipefail
GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
MANIFEST="${2:-data/prompts/eval_big.jsonl}"
REPS="${3:-5}"; NFE="${4:-10}"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="results/out/e3_resynth_${TS}"; mkdir -p "${OUT}"
echo "[E3] GPU=${GPU}  MANIFEST=${MANIFEST}  REPS=${REPS}  NFE=${NFE}  arms=base open_soar  -> ${OUT}"
python eval/resynth_variance.py --manifest "${MANIFEST}" --arms base open_soar \
  --reps "${REPS}" --nfe "${NFE}" --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/run.log"
echo "[E3] ✓ -> ${OUT}/report.txt  （看末尾 [E3 裁决]）"
