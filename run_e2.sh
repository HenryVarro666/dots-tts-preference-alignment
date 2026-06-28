#!/usr/bin/env bash
# run_e2.sh — E2 崩坏臂分块 + 最坏情况 EOS 计分（CPU 纯分析、无需 GPU / 新合成）。
#   回应 EMNLP 审稿：dpo_multiobj 崩坏是否 LJ002 块集中 + 丢 25 条 EOS 失败的伪影；n=75 vs n=100 不可比。
#   用法:   bash run_e2.sh <eval-manifest.jsonl> [arm tags...]
#           默认 arms = base_nfe10 sft_nfe10 dpo_multiobj_nfe10
#   前置:   results/out/{arm}.metrics.jsonl（eval/score_synth.py 产出的逐句 wer/sim/utmos）+ 评测 manifest（全 100 fid）。
#   产物:   results/out/e2_collapse_blocks_<ts>/{report.txt,run.log}
#   裁决:   report.txt 末尾 [E2 裁决]——worst-case n=100 仍崩 + 跨 block 均匀 ⇒ 非伪影。
#   ⚠️ 本机只 bash -n / py_compile；运行只需 CPU + metrics.jsonl（无 GPU）。
set -euo pipefail
MANIFEST="${1:?用法: bash run_e2.sh <eval-manifest.jsonl> [arm tags...]}"; shift || true
ARMS=("$@"); [ "${#ARMS[@]}" -eq 0 ] && ARMS=(base_nfe10 sft_nfe10 dpo_multiobj_nfe10)
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/out/e2_collapse_blocks_${TS}"; mkdir -p "${OUT}"
echo "[E2] manifest=${MANIFEST}  arms=${ARMS[*]}  -> ${OUT}"
python eval/analyze_collapse_blocks.py --manifest "${MANIFEST}" \
  --arms "${ARMS[@]}" --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/run.log"
echo "[E2] ✓ -> ${OUT}/report.txt  （看末尾 [E2 裁决]）"
