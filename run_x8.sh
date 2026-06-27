#!/usr/bin/env bash
# =============================================================================
# run_x8.sh — X8 流式评测(Table 2 / RQ2) 一条命令：流式 run_matrix→汇总→归档
#
# 做什么:   对已训臂跑**流式(1T1A)** 合成+评测，填 paper Table 2 的整表占位——TTFA / RTF /
#           边界不连续 / 流式 WER / 流式 hard-WER。回答 RQ2：离线增益是否迁移到流式、不增首包延迟。
# 怎么做:   run_matrix **不加 --skip-streaming**(默认就跑流式) → aggregate_results 汇总 → 归档。
#           不训练、仅评测；复用已训 ckpt(base/open_soar/... 已在 alignment/out/)。
#   用法:   bash run_x8.sh [GPU] [MANIFEST] [treatment ...]
#           # 默认 GPU=0, MANIFEST=data/prompts/eval_big.jsonl, 臂=base open_soar dpo_multiobj dpo_weronly sft
# 为什么这么设计: Table 2 当前 5 指标 × 5 臂全 `__` 占位(此前评测用了 --skip-streaming)。gate：对齐臂
#           TTFA 须落在 base CI 内(对齐不拖慢首包)；流式 WER 应跟离线趋势。
# 前置:     待评臂 ckpt 已训好(alignment/out/<arm>/)；评测清单存在。
#           ⚠️ RISK⑫：DoubleStreaming API 随 dots.tts 版本敏感——跑前对照当前版本 example_double_streaming 核 API。
#           ⚠️ RISK④：流式 SIM 仅 x-vector，报告只比「流式 vs 流式」treatment 差。
# 预期结果: 流式 summary + results/out/(Table 2 相关) + 归档 results/out/x8_streaming_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑(~小时级，无训练)。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
MANIFEST="${2:-data/prompts/eval_big.jsonl}"
shift || true; shift || true
TREATS=("$@"); [[ ${#TREATS[@]} -eq 0 ]] && TREATS=(base open_soar dpo_multiobj dpo_weronly sft)

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x8_streaming_${STAMP}"
mkdir -p "${ARCH}" results/out

echo "[X8] GPU=${GPU}  TREATS=${TREATS[*]}  MANIFEST=${MANIFEST}"

# ---- 1) 流式评测：run_matrix 不加 --skip-streaming（默认跑流式合成+流式指标）----
echo "[X8] 1/3 流式 run_matrix（不 skip-streaming）"
python ablation/run_matrix.py --manifest "${MANIFEST}" \
  --treatments "${TREATS[@]}" --sim-backend ecapa

# ---- 2) 汇总（累积，不覆盖）----
echo "[X8] 2/3 aggregate_results"
python results/aggregate_results.py

# ---- 3) 归档：主表 + 流式 summary 拷进带时间戳目录 ----
echo "[X8] 3/3 归档 -> ${ARCH}/"
cp -p results/out/main_table.md "${ARCH}/" 2>/dev/null || true
for t in "${TREATS[@]}"; do cp -p results/out/"${t}"*stream*.summary.json results/out/"${t}"*.summary.json "${ARCH}/" 2>/dev/null || true; done

echo "[X8] ✓ 完成。填 Table 2(RQ2)：看流式 WER/SIM/UTMOS + TTFA/RTF/边界。"
echo "      gate：dpo/对齐臂 TTFA 落在 base CI 内(对齐不拖慢首包)、流式 WER 跟离线趋势。结果已归档：${ARCH}/"
