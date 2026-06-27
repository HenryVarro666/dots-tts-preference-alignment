#!/usr/bin/env bash
# =============================================================================
# run_x10.sh — X10 架构级二模型探针（VoxCPM）一条命令：自检→探针→分析→归档
#
# 做什么:   把同一批 dots 生成的 winner/loser 偏好音频喂进 **VoxCPM**(独立实现的连续-latent
#           diffusion-AR TTS)，测它自己的训练 FM 损失，看「reward 偏好的 winner 是否也更不像」——
#           即 dots.tts 上的 FM-reward 反相关在 VoxCPM 上是否**复现**。复现 ⇒ 现象是
#           **continuous-latent TTS 架构级**(解审稿 EIC CRITICAL #3：N=1 模型不能泛到整类)。
# 怎么做:   1) 全量探针(自带命门 gate：跑前自动校验 L_FM 有限正值 / 同 seed 决定论 / 异 seed 有变化，
#              坏接缝在此 fail-fast，不会静默出错表)；2) probe_analysis 出双系数 CI / per-component /
#              高-gap 子集；3) 把 CSV+log+报告拷进**带时间戳目录**，永不被下次覆盖。
#   用法:   bash run_x10.sh [GPU] [CKPT] [N] [K] [PAIRS]   # 默认 GPU=0, CKPT=openbmb/VoxCPM2, N=188, K=16
#           confound-free 复测(settings-matched，强于 grid)：
#             bash run_x10.sh 0 openbmb/VoxCPM2 188 32 pairs/out/pairs_settings_matched.jsonl
#           (自动用 rewards_fixed.jsonl、输出/归档加 _sm tag 不覆盖 grid 结果)
# 为什么这么设计: 只跑一次模型加载——全量探针内部已自 gate(同一 loaded model)，无需再单独 --selfcheck
#           二次加载 5GB 权重；结果全程落盘(--out-csv / --out / tee / 时间戳归档)，兑现「实验结果必存」。
# 前置:     VoxCPM 可 import(setup_env 已挂 site-packages .pth，或 export VOXCPM_REPO 指向 repo 根)；
#           有 openbmb/VoxCPM2 权重(运行时 HF 下载或本地)；第一轮产物已就绪——
#           pairs/out/pairs_best_worst.jsonl + reward/out/rewards.jsonl + candidates/out/candidates_index.jsonl。
# 预期结果: alignment/out/probe_per_pair_voxcpm.csv + results/out/probe_voxcpm_analysis.txt
#           + 归档目录 results/out/x10_voxcpm_<时间戳>/；裁决看报告 [C4 headline]。
# ⚠️ 本机只 bash -n；真正运行在 GPU 机器上(VoxCPM2 bf16，需 CUDA)。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f ".venv/bin/activate" ]]; then source ".venv/bin/activate"; fi

GPU="${1:-0}"; export CUDA_VISIBLE_DEVICES="${GPU}"
CKPT="${2:-openbmb/VoxCPM2}"
N="${3:-188}"; K="${4:-16}"
PAIRS="${5:-pairs/out/pairs_best_worst.jsonl}"   # 第5参可传 pairs_settings_matched.jsonl 做 confound-free 复测

# settings-matched 对：reward 取 X5 的 rewards_fixed.jsonl、setting_gap≡0(probe_analysis 自动跳 confound)；
# 输出/归档名带 tag，settings-matched 复测不覆盖 grid 探针结果。
if [[ "${PAIRS}" == *settings_matched* ]]; then
  TAG="voxcpm_sm"; REWARDS="reward/out/rewards_fixed.jsonl"
else
  TAG="voxcpm";    REWARDS="reward/out/rewards.jsonl"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="alignment/out/probe_per_pair_${TAG}.csv"
LOG="alignment/out/probe_${TAG}_run.log"
RPT="results/out/probe_${TAG}_analysis.txt"
ARCH="results/out/x10_${TAG}_${STAMP}"
mkdir -p "$(dirname "${CSV}")" "$(dirname "${RPT}")" "${ARCH}"

echo "[X10] GPU=${GPU}  CKPT=${CKPT}  N=${N}  K=${K}  pairs=${PAIRS}"
echo "[X10] VOXCPM_REPO=${VOXCPM_REPO:-（未设；靠 venv site-packages 的 voxcpm.pth / 参考代码 回退）}"

# ---- 1) 全量探针：内部先自检(命门 gate)再跑 K-MC；--out-csv 落盘 + tee 存 stdout ----
echo "[X10] 1/3 VoxCPM 探针（自带命门 gate）-> ${CSV}"
python alignment/probe_fm_reward_voxcpm.py --ckpt "${CKPT}" --pairs "${PAIRS}" \
  --n "${N}" --k "${K}" --out-csv "${CSV}" 2>&1 | tee "${LOG}"

# ---- 2) 统计裁决(双系数 CI / per-component / 高-gap 子集)；--out 归档报告(D-8) ----
echo "[X10] 2/3 probe_analysis -> ${RPT}"
python alignment/probe_analysis.py --csv "${CSV}" \
  --index candidates/out/candidates_index.jsonl \
  --rewards "${REWARDS}" \
  --out "${RPT}"

# ---- 3) 归档：CSV + 运行 log + 分析报告 拷进带时间戳目录，永不被下次覆盖 ----
echo "[X10] 3/3 归档 -> ${ARCH}/"
cp -p "${CSV}" "${LOG}" "${RPT}" "${ARCH}/"

echo "[X10] ✓ 完成。裁决看 ${RPT} 的 [C4 headline] FM_margin vs reward_gap："
echo "      VoxCPM 上**同号显著反相关** ⇒ 架构级（标题升 continuous-latent TTS、解 EIC CRITICAL #3）；"
echo "      近零/正相关 ⇒ 反相关是 dots.tts 实现特性、诚实收窄——先按 RISK⑩ 回查 8 接缝再下结论。"
echo "      本次结果已归档：${ARCH}/"
