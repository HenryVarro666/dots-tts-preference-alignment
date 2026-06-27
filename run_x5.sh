#!/usr/bin/env bash
# =============================================================================
# run_x5.sh — X5 settings-matched 复探针（一条命令、安全自包含）
#
# 做什么:   生成"winner/loser 生成设置完全相同(同 guidance/num_steps/speaker、仅变 seed)"的候选 →
#           打分 → 配对 → 复探针 → 统计分析。回答：反相关在**设置完全匹配**的对上是否仍存活？
# 怎么做:   --fixed-setting 固定一组设置、只变 seed 生成；下游沿用默认管线脚本。
#           **关键安全**：generate --fixed-setting 会覆盖网格的 candidates_index.jsonl，score_* 会覆盖
#           reward/out/{wer,sim,utmos}.jsonl —— 这些 X6(filter_pairs_by_likelihood --by guidance)/X10
#           还要用。本脚本先备份这 4 个文件、用专属 --out 产 rewards_fixed/pairs_settings_matched/
#           probe_per_pair_fixed，结束(含失败/中断, trap)自动恢复网格文件 → 不丢主实验数据。
#   用法:   bash run_x5.sh [GPU] [K]      # 默认 GPU=0, K=8
# 为什么这么设计: settings-matched 是 intrinsic 的"最后一道"——偏相关(docs/02 ADR-10)控住设置后反相关
#           仍 −0.392；若在设置**完全相等**的对上也存活，则反相关确证非生成-grid 产物，强于偏相关。
# 对实验意义: 反相关存活=intrinsic 盖章(直堵 R3/DA 的"grid 制造")；消失=收回 intrinsic、改 grid effect。
# 前置:     主实验的候选/打分管线已就绪(本脚本备份它们); dots_manifest_train.jsonl 存在。
# 预期结果: alignment/out/probe_per_pair_fixed.csv + probe_analysis 输出(headline 双系数即裁决)。
# ⚠️ 本机只 bash -n；真正运行在 GPU 机器上(候选生成 ~数 GPU-h)。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
K="${2:-8}"
PROMPTS="data/prompts/dots_manifest_train.jsonl"
CDIR="candidates/out"; RDIR="reward/out"
BAK=("${CDIR}/candidates_index.jsonl" "${RDIR}/wer.jsonl" "${RDIR}/sim.jsonl" "${RDIR}/utmos.jsonl")
echo "[X5] GPU=${GPU}  K=${K}  prompts=${PROMPTS}"

# ---- 备份网格文件 + 注册自动恢复(EXIT/中断/失败都恢复) ----
echo "[X5] 备份网格 candidates_index + scores（结束自动恢复，不丢 X6/X10 用的网格数据）"
for f in "${BAK[@]}"; do [[ -f "$f" ]] && cp -p "$f" "${f}.grid-bak"; done
restore() {
  echo "[X5] 恢复网格文件..."
  for f in "${BAK[@]}"; do [[ -f "${f}.grid-bak" ]] && mv -f "${f}.grid-bak" "$f"; done
}
trap restore EXIT INT TERM

# ---- 1) 固定设置候选(winner/loser 同设置, 仅变 seed) ----
echo "[X5] 1/5 生成固定设置候选 (cand_fixed_*.wav, 覆盖 candidates_index.jsonl)"
python candidates/generate_candidates.py --fixed-setting --prompts "${PROMPTS}" --k "${K}" \
  --fix-num-steps 10 --fix-guidance 1.2 --fix-speaker 1.5

# ---- 2) 打分(默认读 candidates_index.jsonl=现固定设置; 覆盖 wer/sim/utmos.jsonl) ----
echo "[X5] 2/5 打分 WER/SIM/UTMOS"
python reward/score_wer.py --batch-size 8
python reward/score_utmos.py
python reward/score_sim.py --backend ecapa

# ---- 3) 聚合奖励 → 专属 rewards_fixed.jsonl(不碰网格 rewards.jsonl) ----
echo "[X5] 3/5 聚合多目标奖励 -> rewards_fixed.jsonl"
python reward/aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3 \
  --out "${RDIR}/rewards_fixed.jsonl"

# ---- 4) settings-matched pairs(专属文件) ----
echo "[X5] 4/5 建 settings-matched pairs -> pairs_settings_matched.jsonl"
python pairs/build_preference_pairs.py --rewards "${RDIR}/rewards_fixed.jsonl" \
  --strategy best_worst --out pairs/out/pairs_settings_matched.jsonl

# ---- 5) 复探针 + 分析(专属 csv; 此时 candidates_index 仍是固定设置, 供 analysis 的 join) ----
echo "[X5] 5/5 复探针 + 统计分析"
python alignment/probe_fm_reward.py --pairs pairs/out/pairs_settings_matched.jsonl --n 188 --k 16 \
  --out-csv alignment/out/probe_per_pair_fixed.csv
python alignment/probe_analysis.py --csv alignment/out/probe_per_pair_fixed.csv \
  --rewards "${RDIR}/rewards_fixed.jsonl"

echo "[X5] ✓ 完成。看 probe_analysis 的 [C4 headline]：FM_margin vs reward_gap 在设置匹配对上是否仍显著反相关。"
echo "     (设置匹配 ⇒ setting_gap≈0，confound/partial 段会退化/NaN，属正常；裁决看 headline 双系数。)"
echo "     网格文件已自动恢复。"
