#!/usr/bin/env bash
# =============================================================================
# run_x7.sh — X7 reward-hacking 干净复现(RQ3/Table 3) 一条命令：抽→训→评→汇→归档
#
# 做什么:   跑 reward_weighted_weronly 臂——在**非 DPO 错配**框架里(纯 reward-加权回归、无 win/lose 对比)
#           只用「仅 WER」奖励，干净复现 reward-hacking：取最低 WER 但 UTMOS/EmergentTTS 退化。
# 怎么做:   1) 从 pairs_weronly.jsonl(WER-only 选的对)抽 winners_only_weronly.jsonl；2) run_train_all 训该臂；
#           3) run_matrix 离线评测；4) aggregate_results 累积 + 归档。
#   用法:   bash run_x7.sh [GPU] [MANIFEST]   # 默认 GPU=0, MANIFEST=data/prompts/eval_big.jsonl
# 为什么这么设计: 当前 Table 3 的 dpo_weronly 没取到最低 WER、hacking **未复现**(被 DPO-FM 崩坏混淆)。X7 把
#           reward-hacking 从 DPO 错配框架里剥出来干净复现 → 解 domain reviewer 的 Contribution-2 vs 4 + caveat②。
# 前置:     run_full_dataprep.sh 已产 reward/out/rewards_weronly.jsonl + pairs/out/pairs_weronly.jsonl；
#           configs/reward_weighted_weronly.yaml(读 winners_only_weronly.jsonl + rewards_weronly.jsonl) 在。
#           与 reward_weighted(X6b)**共享权重函数/τ**(RISK⑬)——对比才干净。
# 预期结果: alignment/out/reward_weighted_weronly/ + 主表累积 + 归档 results/out/x7_hacking_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑(~1 天)。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
MANIFEST="${2:-data/prompts/eval_big.jsonl}"
ARM="reward_weighted_weronly"

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x7_hacking_${STAMP}"
mkdir -p "${ARCH}" results/out

echo "[X7] GPU=${GPU}  ARM=${ARM}  MANIFEST=${MANIFEST}"

# ---- 1) 抽 WER-only winners（run_train_all 不自动产这个；config 读它）----
echo "[X7] 1/4 extract_winners(WER-only) -> pairs/out/winners_only_weronly.jsonl"
python pairs/extract_winners.py --pairs pairs/out/pairs_weronly.jsonl \
  --out pairs/out/winners_only_weronly.jsonl

# ---- 2) 训该臂 ----
echo "[X7] 2/4 训练 ${ARM}"
bash run_train_all.sh "${GPU}" "${ARM}"

# ---- 3) 离线评测（同 Table 1 口径）----
echo "[X7] 3/4 评测 -> run_matrix（offline）"
python ablation/run_matrix.py --manifest "${MANIFEST}" \
  --treatments "${ARM}" --sim-backend ecapa --skip-streaming

# ---- 4) 汇总 + 归档 ----
echo "[X7] 4/4 aggregate_results + 归档 -> ${ARCH}/"
python results/aggregate_results.py
cp -p results/out/main_table.md "${ARCH}/" 2>/dev/null || true
cp -p results/out/"${ARM}"*.summary.json "${ARCH}/" 2>/dev/null || true

echo "[X7] ✓ 完成。判读 ${ARCH}/main_table.md 的 ${ARM} 行(填 Table 3)："
echo "      取最低 WER 但 UTMOS/EmergentTTS 退化 ⇒ 非错配框架内干净复现 reward-hacking(解 caveat②)。"
