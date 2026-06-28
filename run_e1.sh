#!/usr/bin/env bash
# run_e1.sh — E1 选择伪影 / order-statistic null control（回应 EMNLP 审稿 Devil's Advocate 最强反驳）一条命令。
#   做什么: 对现有 settings-matched 候选池的**每个候选**算 K-MC 去噪 L_FM（no_grad、无需新合成）→
#           做 selection-null 分析（candidate-level 耦合 + Null A 随机准则 + Null B interior pairs）→ 归档。
#   用法:   bash run_e1.sh [GPU] [REWARDS] [N] [K]
#           默认 GPU=0, REWARDS=reward/out/rewards_fixed.jsonl(settings-matched), N=0(全部 prompt), K=16
#   前置:   candidates/out/candidates_index.jsonl + REWARDS + dots.tts-base 权重；真实运行在 GPU 机。
#   产物:   alignment/out/percand_lfm.csv + results/out/e1_selection_null_<ts>/{report.txt,emit.log,analysis.log}
#   裁决:   看 report.txt 末尾 [E1 裁决]——candidate-level corr(reward,L_FM) 同向显著 + Null A p<0.05 +
#           Null B 存活 ⇒ 反相关是 reward↔likelihood 耦合、非选择伪影 → §6 把 caveat 升级为「已排除选择伪影」。
#   ⚠️ 本机只 bash -n / py_compile；真正运行需 CUDA + 权重 + 候选池。
set -euo pipefail
GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
REWARDS="${2:-reward/out/rewards_fixed.jsonl}"
N="${3:-0}"; K="${4:-16}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/out/e1_selection_null_${TS}"; mkdir -p "${OUT}"
echo "[E1] GPU=${GPU}  REWARDS=${REWARDS}  N=${N}  K=${K}  -> ${OUT}"
echo "[E1] 1/2 per-candidate L_FM（全部候选、no_grad、无需新合成）"
python alignment/probe_fm_reward.py --emit-per-candidate \
  --rewards "${REWARDS}" --candidates candidates/out/candidates_index.jsonl \
  --n "${N}" --k "${K}" --out-percand alignment/out/percand_lfm.csv 2>&1 | tee "${OUT}/emit.log"
echo "[E1] 2/2 selection-null 分析（CPU、秒级）"
python alignment/probe_analysis.py --selection-null \
  --percand alignment/out/percand_lfm.csv --null-B 1000 \
  --out "${OUT}/report.txt" 2>&1 | tee "${OUT}/analysis.log"
cp alignment/out/percand_lfm.csv "${OUT}/" 2>/dev/null || true
echo "[E1] ✓ 归档 -> ${OUT}/  （看 report.txt 末尾 [E1 裁决]）"
