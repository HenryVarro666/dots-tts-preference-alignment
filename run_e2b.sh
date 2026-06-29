#!/usr/bin/env bash
# run_e2b.sh — ★2 第二指标交叉验证（DNSMOS 重打分臂音频，排除 open-SOAR 增益是 UTMOS 单指标伪影）。
#   纯重打分、无新合成；需第二指标后端（pip install speechmos）。GPU 可选（DNSMOS 很轻、CPU 即可）。
#   用法: bash run_e2b.sh [arm tags...]   默认 base_nfe10 open_soar_nfe10
#   产物: results/out/e2b_metric_xcheck_<ts>/report.txt
#   裁决: open_soar 第二指标仍 > base ⇒ 自然度增益跨指标稳健、非 UTMOS 伪影。
set -euo pipefail
ARMS=("$@"); [ "${#ARMS[@]}" -eq 0 ] && ARMS=(base_nfe10 open_soar_nfe10)
TS="$(date +%Y%m%d_%H%M%S)"; OUT="results/out/e2b_metric_xcheck_${TS}"; mkdir -p "${OUT}"
echo "[E2b] arms=${ARMS[*]}  metric=dnsmos  -> ${OUT}"
python eval/rescore_metric.py --arms "${ARMS[@]}" --metric dnsmos \
  --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/run.log"
echo "[E2b] ✓ -> ${OUT}/report.txt"
