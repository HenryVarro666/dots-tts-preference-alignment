#!/usr/bin/env bash
# =============================================================================
# run_train_all.sh — 依次训练各 treatment（适合挂 tmux；可拆到两张卡并行）
#
# 做什么:   在指定 GPU 上依次训练给定 treatment（默认全部 4 个需训练的）。
# 怎么做:   用 plain `python`(单进程，与 smoke 同款已验证路径)；GPU 用 CUDA_VISIBLE_DEVICES 选。
# 为什么:   单进程最稳；两张卡可开两个 tmux 各跑一半 treatment 实现并行(无需 DDP/accelerate 配置)。
# 对实验意义: 产出 sft/open_soar/dpo_multiobj/dpo_weronly 四个 ckpt(base 不训)，供评测对比。
#
# 用法:
#   bash run_train_all.sh [GPU] [treatment ...]
#   单卡全跑:        bash run_train_all.sh 0
#   两卡并行(开2个tmux):
#     tmux1: bash run_train_all.sh 0 dpo_multiobj open_soar
#     tmux2: bash run_train_all.sh 1 dpo_weronly  sft
# 前置:     run_full_dataprep.sh 已完成(pairs/winners 就绪)。
# 预期结果: alignment/out/{treatment}/ 下各 ckpt + metrics.jsonl + tb/。
# ⚠️ 本机只 bash -n。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}"
echo "[env] python=$(command -v python)  CUDA_VISIBLE_DEVICES=${GPU}"

TREATS=("$@")
[[ ${#TREATS[@]} -eq 0 ]] && TREATS=(dpo_multiobj dpo_weronly open_soar sft)

declare -A SCRIPT=(
  [dpo_multiobj]="alignment/train_dots_tts_dpo.py"
  [dpo_weronly]="alignment/train_dots_tts_dpo.py"
  [open_soar]="alignment/train_open_soar.py"
  [sft]="alignment/train_sft.py"
  # 第二轮 X4 SFT-target 梯度对照（都走 train_sft.py，只换 config/数据）
  [sft_losers]="alignment/train_sft.py"
  [sft_random]="alignment/train_sft.py"
  [sft_groundtruth]="alignment/train_sft.py"
  # 第二轮 X6/X7 建设性臂
  [dpo_onmanifold]="alignment/train_dots_tts_dpo.py"
  [reward_weighted]="alignment/train_reward_weighted.py"
  [reward_weighted_weronly]="alignment/train_reward_weighted.py"
)
declare -A CFG=(
  [dpo_multiobj]="configs/dpo_multiobj_fullparam.yaml"
  [dpo_weronly]="configs/dpo_weronly.yaml"
  [open_soar]="configs/open_soar_selfcorrect.yaml"
  [sft]="configs/sft_baseline.yaml"
  [sft_losers]="configs/sft_losers.yaml"
  [sft_random]="configs/sft_random.yaml"
  [sft_groundtruth]="configs/sft_groundtruth.yaml"
  [dpo_onmanifold]="configs/dpo_onmanifold.yaml"
  [reward_weighted]="configs/reward_weighted.yaml"
  [reward_weighted_weronly]="configs/reward_weighted_weronly.yaml"
)

# winners 清单(SFT 需要; 幂等)
python pairs/extract_winners.py || true
# 第二轮 SFT-target 对照臂所需清单(幂等; 仅在请求对应臂时抽)
#   sft_groundtruth 用真实音频清单 data/prompts/dots_manifest_train.jsonl，无需抽取。
if [[ " ${TREATS[*]} " == *" sft_losers "* ]]; then python pairs/extract_losers.py || true; fi
if [[ " ${TREATS[*]} " == *" sft_random "* ]]; then python pairs/extract_random.py || true; fi

for t in "${TREATS[@]}"; do
  echo "[$(date +%H:%M:%S)] ====== 训练 ${t} ======"
  python "${SCRIPT[$t]}" --config "${CFG[$t]}"
  echo "[$(date +%H:%M:%S)] ✓ ${t} 完成 -> alignment/out/${t}"
done
echo "[$(date +%H:%M:%S)] ✓ 本批 treatment 全部完成: ${TREATS[*]}"
echo "  看曲线: python alignment/plot_metrics.py --run alignment/out/<treatment>"
