#!/usr/bin/env bash
# =============================================================================
# run_x4.sh — X4 SFT-target 梯度对照 一条命令：训 3 臂→评测→汇总→归档
#
# 做什么:   训三个 SFT 臂(目标只变「在 reward 轴上的位置」)：sft_losers(automatic-reward 选出的最差=
#           最 off-manifold)、sft_random(居中)、sft_groundtruth(真实音频=on-manifold)，再评测进主表。
# 怎么做:   1) run_train_all.sh 训三臂(自动 extract_losers/random)；2) run_matrix 离线评测(--skip-streaming，
#           与现有 Table 1 同口径)；3) aggregate_results 累积进主表；4) 归档主表+summary。
#   用法:   bash run_x4.sh [GPU] [MANIFEST]   # 默认 GPU=0, MANIFEST=data/prompts/eval_big.jsonl
# 为什么这么设计: 钉死「sft 崩」的病因——四臂共享 pipeline/步数/损失/数据量，唯一变量是**目标流形位置**。
#           判据：winners 最崩 > random/losers 居中 > groundtruth 改善 ⇒ 退化是 **off-manifold 目标**所致、
#           **而非小数据 SFT 不稳**(解审稿 R3/R5 + caveat1)。填 Table 1 缺的对照臂。
# 前置:     run_full_dataprep.sh 已完成(pairs/winners 就绪)；configs/sft_{losers,random,groundtruth}.yaml 在；
#           data/prompts/dots_manifest_train.jsonl(groundtruth 用真实音频) + 评测清单存在。
# 预期结果: alignment/out/sft_{losers,random,groundtruth}/ + results/out/main_table.md(累积) +
#           归档 results/out/x4_sft_target_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑(3 个短训 ~1–2 天)。训练前各臂内部已过 check_fm_loss 自检。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
MANIFEST="${2:-data/prompts/eval_big.jsonl}"
ARMS=(sft_losers sft_random sft_groundtruth)

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x4_sft_target_${STAMP}"
mkdir -p "${ARCH}" results/out

echo "[X4] GPU=${GPU}  ARMS=${ARMS[*]}  MANIFEST=${MANIFEST}"

# ---- 1) 训三臂（run_train_all 会自动 extract_winners + 按需 extract_losers/random）----
echo "[X4] 1/4 训练 ${ARMS[*]}"
bash run_train_all.sh "${GPU}" "${ARMS[@]}"

# ---- 2) 离线评测（--skip-streaming，与现有 Table 1 同口径，可比）----
echo "[X4] 2/4 评测 -> run_matrix（offline）"
python ablation/run_matrix.py --manifest "${MANIFEST}" \
  --treatments "${ARMS[@]}" --sim-backend ecapa --skip-streaming

# ---- 3) 汇总进主表（读 ledger ∪ results/out/*.summary.json，累积不覆盖，B-20）----
echo "[X4] 3/4 aggregate_results -> 主表"
python results/aggregate_results.py

# ---- 4) 归档：主表 + 三臂 summary 拷进带时间戳目录 ----
echo "[X4] 4/4 归档 -> ${ARCH}/"
cp -p results/out/main_table.md "${ARCH}/" 2>/dev/null || echo "  (main_table.md 待 aggregate 产出，确认 results/out/)"
for a in "${ARMS[@]}"; do cp -p results/out/"${a}"*.summary.json "${ARCH}/" 2>/dev/null || true; done

echo "[X4] ✓ 完成。判读 ${ARCH}/main_table.md 的三新行："
echo "      sft_losers/sft_random 崩坏程度 ≥ sft_groundtruth(改善)，且按「目标越 off-manifold 越崩」单调"
echo "      ⇒ 退化是 off-manifold 目标所致、非小数据 SFT 不稳(解 caveat1)。结果已归档：${ARCH}/"
