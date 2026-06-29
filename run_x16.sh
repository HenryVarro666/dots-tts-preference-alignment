#!/usr/bin/env bash
# =============================================================================
# run_x16.sh — ★5 VoxCPM2 原生配对 一条命令：自采样→打分→建对→原生探针→分析→归档
#              （把跨架构复现从「依赖证据」升级为真正独立的 N=2）
#
# 做什么:   让 VoxCPM2 自己生成候选 → 同一套 automatic-metric reward 打分 → 构造 VoxCPM2-原生偏好对
#           → 在原生对上重跑 FM-reward 探针。generator/pair/reward-gap 全部独立于 dots ⇒ 独立复现。
# 怎么做:   1) generate_candidates_voxcpm.py 自采样 K 候选(原生 schema)；2) score_{wer,sim,utmos}.py +
#           aggregate_reward.py(--candidates voxcpm → --out rewards_voxcpm.jsonl)；3) build_preference_pairs.py
#           (--rewards rewards_voxcpm → --out pairs_voxcpm.jsonl)；4) probe_fm_reward_voxcpm.py(--pairs 原生对)；
#           5) probe_analysis.py 裁决；6) 归档。
#   用法:   bash run_x16.sh [GPU] [PROMPTS] [K]
#           # 默认 GPU=0, PROMPTS=data/prompts/dots_manifest_train.jsonl, K=6
#           # PROMPTS 建议传**与 dots 188 对同一 prompt 集**，使比较仅差「生成端架构」。
# 为什么这么设计: 下游 reward/pairs/probe 全按 common.py 的 fid+Row schema 走，原生候选写同款 schema 即可
#           零改动复用——只把 --candidates/--rewards/--pairs 指向 voxcpm 后缀文件，不碰 dots 产物。
# 判据:     原生对上 Spearman 反相关**同号显著** ⇒ generator+pair+reward-gap 全独立 ⇒ **独立 N=2 复现**
#           （§6/Limitations 把「supporting, statistically-dependent」升为「independent replication」）；
#           近零/正 ⇒ 维持现状（dependent 证据），诚实不升级。
# 前置:     env/setup_env.sh 已挂 VoxCPM2 + 下权重；PROMPTS 就绪。
# 预期结果: candidates/out/candidates_index_voxcpm.jsonl + reward/out/rewards_voxcpm.jsonl +
#           pairs/out/pairs_voxcpm.jsonl + alignment/out/probe_per_pair_voxcpm_native.csv +
#           results/out/probe_voxcpm_native_analysis.txt + 归档 results/out/x16_voxcpm_native_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑（自采样 ~N_prompt×K 次 VoxCPM2 推理 + 一遍探针，半天级）。
# ⚠️【seam-1】generate_candidates_voxcpm.py 内 VoxCPM2 generate API 须对照已装包核对（见该脚本头）。
# ⚠️【seam-2 已自动处理】verify 确认 score_{wer,sim,utmos}.py 写**固定**中间名 reward/out/{wer,sim,utmos}.jsonl
#            （无 --out）；step 2 已自动「备份 dots 中间产物→跑 voxcpm 打分→还原」，最终 rewards_voxcpm.jsonl
#            由 aggregate_reward --out 隔离，dots 的 rewards.jsonl 与中间产物均不受影响。无需手动干预。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
PROMPTS="${2:-data/prompts/dots_manifest_train.jsonl}"
K="${3:-6}"

CAND="candidates/out/candidates_index_voxcpm.jsonl"
REW="reward/out/rewards_voxcpm.jsonl"
PAIRS="pairs/out/pairs_voxcpm.jsonl"
CSV="alignment/out/probe_per_pair_voxcpm_native.csv"
RPT="results/out/probe_voxcpm_native_analysis.txt"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x16_voxcpm_native_${STAMP}"
mkdir -p "${ARCH}" results/out reward/out pairs/out alignment/out

echo "[X16] GPU=${GPU}  PROMPTS=${PROMPTS}  K=${K}"

# ---- 1) VoxCPM2 自采样候选（原生 schema）----
echo "[X16] 1/6 VoxCPM2 自采样 -> ${CAND}"
python candidates/generate_candidates_voxcpm.py --prompts "${PROMPTS}" --k "${K}" --out-index "${CAND}"

# ---- 2) 同一套 reward 打分 → 原生 rewards（aggregate --out 显式隔离 dots 的 rewards.jsonl）----
# score_*.py 写**固定**中间名 reward/out/{wer,sim,utmos}.jsonl（无 --out）；备份 dots 的、跑完还原，
# 避免 voxcpm 这遍覆盖 dots 中间产物（verify seam-2 已确认这是唯一会被 --candidates 漏掉的隔离点）。
echo "[X16] 2/6 reward 打分（WER/SIM/UTMOS）-> aggregate -> ${REW}"
for m in wer sim utmos; do [[ -f "reward/out/${m}.jsonl" ]] && cp -p "reward/out/${m}.jsonl" "reward/out/${m}.jsonl.dotsbak"; done
python reward/score_wer.py   --candidates "${CAND}"
python reward/score_sim.py   --candidates "${CAND}" --backend ecapa
python reward/score_utmos.py --candidates "${CAND}"
python reward/aggregate_reward.py --out "${REW}"
for m in wer sim utmos; do [[ -f "reward/out/${m}.jsonl.dotsbak" ]] && mv -f "reward/out/${m}.jsonl.dotsbak" "reward/out/${m}.jsonl"; done

# ---- 3) 构造 VoxCPM2-原生偏好对（best_worst，与 dots 同口径；--candidates 指向原生索引，join 原生 text/prompt）----
echo "[X16] 3/6 建偏好对 -> ${PAIRS}"
python pairs/build_preference_pairs.py --rewards "${REW}" --candidates "${CAND}" --strategy best_worst --out "${PAIRS}"

# ---- 4) 在原生对上跑探针（自带命门自检；N=原生对总数，全量）----
NPAIRS="$(wc -l < "${PAIRS}" | tr -d ' ')"
echo "[X16] 4/6 原生探针（N=${NPAIRS} 全量）-> ${CSV}"
python alignment/probe_fm_reward_voxcpm.py --ckpt openbmb/VoxCPM2 --pairs "${PAIRS}" \
  --n "${NPAIRS}" --k 16 --out-csv "${CSV}" 2>&1 | tee "alignment/out/probe_voxcpm_native_run.log"

# ---- 5) 统计裁决（双系数 CI / per-component / 高-gap 子集）----
echo "[X16] 5/6 probe_analysis -> ${RPT}"
python alignment/probe_analysis.py --csv "${CSV}" \
  --index "${CAND}" --rewards "${REW}" --out "${RPT}"

# ---- 6) 归档 ----
echo "[X16] 6/6 归档 -> ${ARCH}/"
cp -p "${CSV}" "${RPT}" "alignment/out/probe_voxcpm_native_run.log" "${ARCH}/" 2>/dev/null || true
cp -p "${REW}" "${PAIRS}" "${ARCH}/" 2>/dev/null || true

echo "[X16] ✓ 完成。裁决看 ${RPT} 的 [C4 headline] FM_margin vs reward_gap："
echo "      VoxCPM2-原生对上**同号显著反相关** ⇒ generator+pair+reward-gap 全独立 ⇒ **独立 N=2 复现**"
echo "      （§6/Limitations 升级措辞）；近零/正 ⇒ 维持 dependent 证据、诚实不升级。归档：${ARCH}/"
