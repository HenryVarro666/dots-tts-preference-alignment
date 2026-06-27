#!/usr/bin/env bash
# =============================================================================
# run_x14.sh — X14 L_FM 似然有效性独立锚（typicality）一条命令：抽 winners→探针→归档
#
# 做什么:   按 fid **配对**比较「真实 held-out 录音」 vs 「reward-winner 候选」的 per-sample L_FM——
#           真实 on-manifold 录音是否被模型自身似然判得**系统性更典型**(L_FM 更低)。出均值差 +
#           bootstrap 95%CI + Wilcoxon 符号秩。
# 怎么做:   1) extract_winners 出 winners_only.jsonl；2) probe_lfm_validity 复用 probe_fm_reward 的
#           「四前向共享-(x0,t)」契约(同 RNG、K-MC、no_grad)，逐样本算 L_FM 并自带裁决打印；3) 归档。
#   用法:   bash run_x14.sh [GPU] [CKPT] [N] [K]   # 默认 GPU=0, CKPT=rednote-hilab/dots.tts-base, N=188, K=16
# 为什么这么设计: 关 ADR-12 **C2(OPEN-SOAR-REFRAME)** caveat——此前「L_FM 是有效似然代理」只靠 open-SOAR
#           训到饱和(by construction 易分、不独立)。本探针给一条**不依赖 open-SOAR、不依赖 reward** 的独立锚。
#           便宜(no_grad、~0.5–1h)、不训练，性价比最高；结果全程落盘(兑现「实验结果必存」)。
# 前置:     dots.tts 已装(主管线)；第一轮产物就绪——pairs/out/pairs_best_worst.jsonl +
#           data/prompts/dots_manifest_train.jsonl(真实录音清单)。本探针在 dots.tts 上、决定论已由
#           check_fm_loss 立过(无需 X10 式新模型自检)。
# 预期结果: alignment/out/probe_lfm_validity.csv + 归档目录 results/out/x14_lfm_<时间戳>/；
#           裁决看 stdout「汇总（子协议 (b) typicality）」段(real 系统性更低 ⇒ 关 C2)。
# ⚠️ 本机只 bash -n；真正运行在 GPU 机器上(dots.tts bf16)。子协议 (a) ODE-NLL〔需核实〕默认不开。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
CKPT="${2:-rednote-hilab/dots.tts-base}"
N="${3:-188}"; K="${4:-16}"
PAIRS="pairs/out/pairs_best_worst.jsonl"
REAL="data/prompts/dots_manifest_train.jsonl"
WINNERS="pairs/out/winners_only.jsonl"

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="alignment/out/probe_lfm_validity.csv"
LOG="alignment/out/probe_lfm_validity_run.log"
ARCH="results/out/x14_lfm_${STAMP}"
mkdir -p "$(dirname "${CSV}")" "${ARCH}"

echo "[X14] GPU=${GPU}  CKPT=${CKPT}  N=${N}  K=${K}"

# ---- 1) 抽 winners 清单(配对的 winner 侧；real 侧来自 dots_manifest_train) ----
echo "[X14] 1/3 extract_winners -> ${WINNERS}"
python pairs/extract_winners.py --pairs "${PAIRS}" --out "${WINNERS}"

# ---- 2) typicality 探针：逐样本 L_FM(real vs winner) + bootstrap CI + Wilcoxon，自带裁决 ----
#       不加 --ode-nll：子协议 (a) 接口 8 处〔需核实〕，typicality(b) 已足够关 C2。
echo "[X14] 2/3 probe_lfm_validity -> ${CSV}"
python alignment/probe_lfm_validity.py --ckpt "${CKPT}" \
  --real-manifest "${REAL}" --winners "${WINNERS}" \
  --n "${N}" --k "${K}" --out-csv "${CSV}" 2>&1 | tee "${LOG}"

# ---- 3) 归档：CSV + 运行 log(含裁决)拷进带时间戳目录 ----
echo "[X14] 3/3 归档 -> ${ARCH}/"
cp -p "${CSV}" "${LOG}" "${ARCH}/"

echo "[X14] ✓ 完成。裁决看 ${LOG} 的「汇总（子协议 (b) typicality）」段："
echo "      real L_FM 系统性更低(diff=winner−real>0、CI 排 0、Wilcoxon 显著) ⇒ 关 ADR-12 C2、给 L_FM"
echo "      似然有效性一条不依赖 open-SOAR 的独立支撑；CI 含 0/反向 ⇒ 收回「有效似然代理」强表述(写 Limitations)。"
echo "      本次结果已归档：${ARCH}/"
