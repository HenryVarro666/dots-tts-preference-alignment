#!/usr/bin/env bash
# =============================================================================
# run_seedtts_eval.sh — 无 sudo、单卡友好的 seed-tts-eval(WER+SIM) 包装
#
# 做什么:   对一个合成目录跑 seed-tts-eval 的 WER(Whisper/Paraformer) + SIM(WavLM)。
# 怎么做:   调用 seed-tts-eval 的 cal_wer.sh / cal_sim.sh；先设置 ARNOLD_WORKER_GPU、
#           去掉脚本里的 sudo 依赖(原脚本为 ByteDance 集群写，含 `sudo split`/$ARNOLD_*)。
# 为什么:   复用业界事实标准评测；但其原脚本耦合集群环境，需做单机适配。
# 对实验意义: WER/SIM 是 RQ1/RQ3 的硬指标；统一后端保证与榜单/各 treatment 可比。
#
# 前置条件: seed-tts-eval 依赖已装；WavLM ckpt 已放置；已 make_seedtts_manifest 生成 meta。
# 用法:     bash run_seedtts_eval.sh <meta.lst> <synth_dir> <lang:en|zh> <wavlm_ckpt>
# 预期结果: 终端打印平均 WER 与平均 SIM。
# ⚠️ 本机只 bash -n。
# =============================================================================
set -euo pipefail
META="${1:?meta.lst}"; SYNTH="${2:?synth_dir}"; LANG="${3:?en|zh}"; WAVLM="${4:?wavlm ckpt}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
SEED_EVAL="${REPO_ROOT}/参考代码/seed-tts-eval"

# 单卡适配：原脚本读 $ARNOLD_WORKER_GPU 决定并行进程数；这里默认 1。
export ARNOLD_WORKER_GPU="${ARNOLD_WORKER_GPU:-1}"

echo "[WER] lang=${LANG}  (为什么: 重转写比对，量可懂性)"
# 原 cal_wer.sh 内含 `sudo split`；用 sed 现场去 sudo，避免无 sudo 环境失败
sed 's/sudo split/split/g' "${SEED_EVAL}/cal_wer.sh" > "${HERE}/.cal_wer_nosudo.sh"
bash "${HERE}/.cal_wer_nosudo.sh" "${META}" "${SYNTH}" "${LANG}"

echo "[SIM] (为什么: WavLM 说话人嵌入余弦，量音色相似)"
sed 's/sudo split/split/g' "${SEED_EVAL}/cal_sim.sh" > "${HERE}/.cal_sim_nosudo.sh"
bash "${HERE}/.cal_sim_nosudo.sh" "${META}" "${SYNTH}" "${WAVLM}"

echo "✓ seed-tts-eval 完成（WER/SIM 见上）"
