#!/usr/bin/env bash
# =============================================================================
# run_x6.sh — X6 建设性出路两臂 一条命令：(过滤)→训 dpo_onmanifold+reward_weighted→评→汇→归档
#
# 做什么:   跑两条「建设性」臂——X6a dpo_onmanifold(只用 winner 更 on-manifold 的偏好子集重训 DPO)、
#           X6b reward_weighted(winners 上非似然 reward-加权回归，无 win/lose 对比)。
# 怎么做:   1) filter_pairs_by_likelihood 产 on-manifold 子集(给 X6a)；2) run_train_all 训两臂；
#           3) run_matrix 离线评测；4) aggregate_results 累积进主表 + 归档。
#   用法:   bash run_x6.sh [GPU] [MANIFEST]   # 默认 GPU=0, MANIFEST=data/prompts/eval_big.jsonl
# 为什么这么设计: 解审稿 R3「建设性出路」——X6a 若 on-manifold pair 上 DPO 不崩(margin 起得来) ⇒ 病因在
#           off-manifold winner 而非 DPO 本身；X6b 若不崩且 ≥ open_soar ⇒ 正面回应 X5「L_FM 不能当 reward」、
#           给一条非似然但不踩错配坑的路。两臂填 Table 1。
# 前置:     run_full_dataprep.sh 完成(pairs/winners/rewards 就绪)；configs/dpo_onmanifold.yaml(读
#           pairs_onmanifold.jsonl) + configs/reward_weighted.yaml(读 winners_only.jsonl+rewards.jsonl) 在。
# 预期结果: alignment/out/{dpo_onmanifold,reward_weighted}/ + 主表累积 + 归档 results/out/x6_constructive_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑(~2–3 天)。RISK⑬：reward_weighted 的 τ/加权方案须显式(见 config)。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
MANIFEST="${2:-data/prompts/eval_big.jsonl}"
ARMS=(dpo_onmanifold reward_weighted)

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x6_constructive_${STAMP}"
mkdir -p "${ARCH}" results/out

echo "[X6] GPU=${GPU}  ARMS=${ARMS[*]}  MANIFEST=${MANIFEST}"

# ---- 1) 过滤 on-manifold 子集（给 X6a；--out 默认 pairs_onmanifold.jsonl=dpo_onmanifold.yaml 读的路径）----
echo "[X6] 1/4 filter_pairs_by_likelihood --by guidance -> pairs/out/pairs_onmanifold.jsonl"
python pairs/filter_pairs_by_likelihood.py --by guidance --thr 1.2

# ---- 2) 训两臂（run_train_all 会先 extract_winners → winners_only.jsonl 供 reward_weighted）----
echo "[X6] 2/4 训练 ${ARMS[*]}"
bash run_train_all.sh "${GPU}" "${ARMS[@]}"

# ---- 3) 离线评测（同 Table 1 口径）----
echo "[X6] 3/4 评测 -> run_matrix（offline）"
python ablation/run_matrix.py --manifest "${MANIFEST}" \
  --treatments "${ARMS[@]}" --sim-backend ecapa --skip-streaming

# ---- 4) 汇总 + 归档 ----
echo "[X6] 4/4 aggregate_results + 归档 -> ${ARCH}/"
python results/aggregate_results.py
cp -p results/out/main_table.md "${ARCH}/" 2>/dev/null || true
for a in "${ARMS[@]}"; do cp -p results/out/"${a}"*.summary.json "${ARCH}/" 2>/dev/null || true; done

echo "[X6] ✓ 完成。判读 ${ARCH}/main_table.md："
echo "      dpo_onmanifold 不崩/margin 起来 ⇒ 病因在 off-manifold winner、非 DPO 本身；"
echo "      reward_weighted 不崩且 ≥ open_soar ⇒ 非似然出路成立(正面回应 X5)。结果已归档。"
