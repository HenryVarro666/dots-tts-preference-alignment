#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 一键准备 GPU 机器上的实验环境
#
# 做什么:   建 venv → 装 dots.tts(-e) → 装本实验增量依赖 → 装 seed-tts-eval 依赖
#           → 下载 WavLM-SV 说话人验证 ckpt → 提示 EmergentTTS-Eval 需另 clone。
# 怎么做:   纯 shell；幂等(可重复跑)；每步前打印「在做什么、为什么」。
# 为什么:   多源依赖(dots.tts / seed-tts-eval / speechmos)易冲突，集中固化一次。
# 对实验意义: 环境可复现是「数字可信」的前提；版本飘了指标就不可比。
#
# 前置条件: NVIDIA GPU + CUDA + Python 3.10~3.12；在 dots.tts/seed-tts-eval 已 clone 的仓库内。
# 预期结果: `python -c "import dots_tts, speechmos, jiwer, peft"` 成功；WavLM ckpt 落地。
# ⚠️ 本机(macOS/无 GPU)不要运行此脚本；仅 `bash -n` 语法检查。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." 2>/dev/null && pwd || echo "${PROJECT_ROOT}")"
DEPS="${PROJECT_ROOT}/.deps"
VENV="${PROJECT_ROOT}/.venv"

# 解析依赖仓库：优先用知识库内的(../../参考代码/)，独立部署则克隆到 .deps/(代码 only, 无 LFS 权重)
resolve_or_clone() {
  local name="$1" url="$2"
  local kb="${REPO_ROOT}/参考代码/${name}"
  if [[ -d "${kb}" ]]; then echo "${kb}"; return; fi
  local dst="${DEPS}/${name}"
  if [[ ! -d "${dst}" ]]; then
    mkdir -p "${DEPS}"
    echo "  → 克隆 ${name} 到 ${dst} (code only)" 1>&2
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${url}" "${dst}" 1>&2
  fi
  echo "${dst}"
}

echo "[0/6] 准备依赖仓库 dots.tts / seed-tts-eval"
DOTS_TTS="$(resolve_or_clone dots.tts https://github.com/rednote-hilab/dots.tts)"
SEED_EVAL="$(resolve_or_clone seed-tts-eval https://github.com/BytedanceSpeech/seed-tts-eval)"
# 让 common.py 在独立部署时也能定位到它们
export DOTS_TTS_REPO="${DOTS_TTS}"
export SEED_TTS_EVAL="${SEED_EVAL}"
echo "  dots.tts     = ${DOTS_TTS}"
echo "  seed-tts-eval= ${SEED_EVAL}"

echo "[1/6] 创建并激活 venv (为什么: 隔离依赖，避免污染系统 Python)"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

echo "[2/6] 以可编辑模式安装 dots.tts (为什么: 我们要 import 其 model/runtime/training 内部模块)"
pip install -e "${DOTS_TTS}"

echo "[3/6] 安装本实验增量依赖 (speechmos/jiwer/peft/deepspeed ...)"
pip install -e "${PROJECT_ROOT}"

echo "[4/6] 安装 seed-tts-eval 依赖 (WER 用 funasr/Paraformer, SIM 用 WavLM)"
if [[ -f "${SEED_EVAL}/requirements.txt" ]]; then
  pip install -r "${SEED_EVAL}/requirements.txt"
else
  echo "  ⚠️ 未找到 ${SEED_EVAL}/requirements.txt —— 请确认 seed-tts-eval 已 clone"
fi

echo "[5/6] 下载 WavLM-SV 说话人验证 checkpoint (SIM 指标必需)"
WAVLM_DIR="${PROJECT_ROOT}/.ckpt"
mkdir -p "${WAVLM_DIR}"
WAVLM_CKPT="${WAVLM_DIR}/wavlm_large_finetune.pth"
if [[ ! -f "${WAVLM_CKPT}" ]]; then
  echo "  → 见 seed-tts-eval/README 的 Google Drive 链接，将 wavlm_large_finetune.pth 放到:"
  echo "    ${WAVLM_CKPT}"
  echo "  (无法脚本化下载 Google Drive 大文件；请手动放置后重跑本步校验)"
else
  echo "  ✓ 已存在: ${WAVLM_CKPT}"
fi

echo "[6/6] EmergentTTS-Eval 提示 (表现力评测，非必需但强烈推荐)"
EMERGENT="${PROJECT_ROOT}/.third_party/EmergentTTS-Eval"
if [[ ! -d "${EMERGENT}" ]]; then
  echo "  → 表现力评测需另 clone (arXiv 2505.23009):"
  echo "    git clone https://github.com/boson-ai/EmergentTTS-Eval-public ${EMERGENT}"
fi

cat <<EOF

=============================================================================
环境准备完成。验证:
  source ${VENV}/bin/activate
  python -c "import dots_tts, speechmos, jiwer, peft; print('ok')"
下一步: 见 docs/03-REPRODUCE.md
=============================================================================
EOF
