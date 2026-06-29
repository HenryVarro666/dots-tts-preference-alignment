#!/usr/bin/env bash
# =============================================================================
# run_x15.sh — ★4 DPO-FM β×λ_sft 超参网格 一条命令：生成网格配置→逐格训练+合成+打分→聚合→归档
#
# 做什么:   在一张 (β, λ_sft) 网格上重训崩坏臂 dpo_multiobj，检验「崩坏对超参是否稳健」、
#           「加大 anchor-SFT 权重 λ_sft 能否救回」。把论文单点崩坏升级为网格级稳健主张。
# 怎么做:   1) beta_lambda_sweep.py --mode emit-configs 派生 configs/_sweep/<tag>.yaml + cells.tsv；
#           2) 逐格：train_dots_tts_dpo.py(--config) → batch_synthesize.py(--ckpt output_dir --treatment tag)
#              → score_synth.py(--synth-dir inference/out/tag --treatment tag)；
#           3) beta_lambda_sweep.py --mode aggregate 拼 β×λ 的 WER/UTMOS/失败率网格 + 裁决；4) 归档。
#   用法:   bash run_x15.sh [GPU] [BASE_CONFIG] [MANIFEST]
#           # 默认 GPU=0, BASE_CONFIG=configs/dpo_multiobj_fullparam.yaml(验证过), MANIFEST=data/prompts/eval_big.jsonl
#           # 自定义网格：导出 BETAS / LAMBDAS 环境变量（空格分隔）缩格控成本，如：
#           #   BETAS="0.1 0.5 2.0" LAMBDAS="0.1"   bash run_x15.sh 0   # β 轴 3 格
#           #   BETAS="0.1" LAMBDAS="0.0 0.1 1.0"   bash run_x15.sh 0   # λ 轴 3 格（共享 (0.1,0.1)）
# 为什么这么设计: 训练是配置驱动（β=dpo.beta / λ=dpo.lambda_sft 在 yaml 里），故先派生配置再逐格训；
#           默认全参 base（dpo_lora.yaml 的 LoRA ref 路径当前崩，见下方 BASE_CONFIG 注释）——全参单格较贵，
#           想省卡就用上面的「十字」缩格（β 轴 + λ 轴各一条，5 格而非 3×3=9 格）。
# 判据:     整张网格 WER 远高于 base(0.112) ⇒ 崩坏对超参稳健（强化主张）；
#           若大 λ_sft 角落 WER 回落 ⇒ 诚实报「强 anchor 可缓冲」，机理不变、措辞软化。
# 前置:     run_full_dataprep.sh 已完成（pairs/winners 就绪）；base DPO 配置在；评测清单存在。
# 预期结果: configs/_sweep/<tag>.yaml + alignment/out/<tag>/ 各 ckpt + results/out/<tag>.summary.json
#           + results/out/beta_lambda_sweep.txt(+.csv) + 归档 results/out/x15_betalambda_<时间戳>/。
# ⚠️ 本机只 bash -n；GPU 机器上跑。默认 3×3=9 格全参较贵——成本随网格大小线性，建议先用「十字」5 格
#    （BETAS 三档@λ=0.1 + LAMBDAS 三档@β=0.1）。第一轮的 dpo_multiobj 即 (β=0.1,λ=0.1) 格，可作锚点对照。
#    每格训练前 train_dots_tts_dpo.py 内部已过 check_fm_loss 自检（FM 接缝 + 共享-RNG 决定论）。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
# 默认全参 base：第一轮验证过能跑。⚠️ LoRA base(dpo_lora.yaml) 当前会崩——LoRA DPO 的 ref 走
# self.policy.disable_adapter()，本机 peft 版本在该上下文里调 get_model_status() 抛
# "No adapter layers found"（这条路第一轮用全参时从未走过、bug 潜伏至今）。LoRA 修好前用全参。
BASE_CONFIG="${2:-configs/dpo_multiobj_fullparam.yaml}"
MANIFEST="${3:-data/prompts/eval_big.jsonl}"
BETAS="${BETAS:-0.1 0.5 2.0}"
LAMBDAS="${LAMBDAS:-0.0 0.1 1.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCH="results/out/x15_betalambda_${STAMP}"
CELLS="configs/_sweep/cells.tsv"
mkdir -p "${ARCH}" results/out configs/_sweep

echo "[X15] GPU=${GPU}  base=${BASE_CONFIG}  manifest=${MANIFEST}"
echo "[X15] BETAS=[${BETAS}]  LAMBDAS=[${LAMBDAS}]"

# ---- 1) 派生网格配置 ----
echo "[X15] 1/4 生成网格配置 -> configs/_sweep/"
python eval/beta_lambda_sweep.py --mode emit-configs \
  --base-config "${BASE_CONFIG}" --betas ${BETAS} --lambdas ${LAMBDAS}

# ---- 2) 逐格：训练 -> 合成 -> 打分（单格失败不炸全程，记下继续）----
echo "[X15] 2/4 逐格 train -> synth -> score"
while IFS=$'\t' read -r TAG CFG OUTDIR; do
  [[ -z "${TAG}" ]] && continue
  echo "[X15]   == ${TAG}  (cfg=${CFG}, ckpt=${OUTDIR}) =="
  if ! python alignment/train_dots_tts_dpo.py --config "${CFG}"; then
    echo "[X15]   ⚠️ ${TAG} 训练失败，跳过该格"; continue
  fi
  python inference/batch_synthesize.py --ckpt "${OUTDIR}" --manifest "${MANIFEST}" \
    --treatment "${TAG}" --num-steps 10 || { echo "[X15]   ⚠️ ${TAG} 合成失败"; continue; }
  python eval/score_synth.py --synth-dir "inference/out/${TAG}" --manifest "${MANIFEST}" \
    --treatment "${TAG}" --sim-backend ecapa || echo "[X15]   ⚠️ ${TAG} 打分失败"
done < "${CELLS}"

# ---- 3) 聚合成 β×λ 网格 + 裁决 ----
echo "[X15] 3/4 聚合 -> results/out/beta_lambda_sweep.txt"
python eval/beta_lambda_sweep.py --mode aggregate \
  --betas ${BETAS} --lambdas ${LAMBDAS} --manifest "${MANIFEST}" \
  --out results/out/beta_lambda_sweep.txt

# ---- 4) 归档：网格报告 + 各格 summary 拷进带时间戳目录 ----
echo "[X15] 4/4 归档 -> ${ARCH}/"
cp -p results/out/beta_lambda_sweep.txt results/out/beta_lambda_sweep.csv "${ARCH}/" 2>/dev/null || true
cp -p "${CELLS}" "${ARCH}/" 2>/dev/null || true
while IFS=$'\t' read -r TAG _ _; do
  [[ -z "${TAG}" ]] && continue
  cp -p results/out/"${TAG}".summary.json "${ARCH}/" 2>/dev/null || true
done < "${CELLS}"

echo "[X15] ✓ 完成。判读 ${ARCH}/beta_lambda_sweep.txt 的 [裁决]："
echo "      整张网格 WER 远高于 base ⇒ 崩坏对 (β,λ) 稳健（§5.1 升级主张）；"
echo "      大 λ_sft 角落 WER 回落 ⇒ 诚实报『强 anchor 可缓冲』。结果已归档：${ARCH}/"
