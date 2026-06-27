#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 一键准备 GPU 机器上的实验环境
#
# 做什么:   建 venv → 装 dots.tts(-e) → 装本实验增量依赖 → 装 seed-tts-eval 依赖
#           → 挂 VoxCPM(X10 二模型探针, 走 sys.path/.pth、不 pip 装免 torch 冲突)
#           → 下载 WavLM-SV 说话人验证 ckpt → 提示 EmergentTTS-Eval 需另 clone。
# 怎么做:   纯 shell；幂等(可重复跑)；每步前打印「在做什么、为什么」。
# 为什么:   多源依赖(dots.tts / seed-tts-eval / speechmos)易冲突，集中固化一次。
# 对实验意义: 环境可复现是「数字可信」的前提；版本飘了指标就不可比。
#
# 前置条件: NVIDIA GPU + CUDA + Python 3.10~3.12；在 dots.tts/seed-tts-eval 已 clone 的仓库内。
# 预期结果: `python -c "import dots_tts, jiwer, peft"` 成功；WavLM ckpt 落地。
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

echo "[0/6] 准备依赖仓库 dots.tts / seed-tts-eval / VoxCPM"
DOTS_TTS="$(resolve_or_clone dots.tts https://github.com/rednote-hilab/dots.tts)"
SEED_EVAL="$(resolve_or_clone seed-tts-eval https://github.com/BytedanceSpeech/seed-tts-eval)"
VOXCPM="$(resolve_or_clone VoxCPM https://github.com/OpenBMB/VoxCPM)"   # X10 架构级二模型探针（仅 X10 用）
# 让 common.py / 探针在独立部署时也能定位到它们
export DOTS_TTS_REPO="${DOTS_TTS}"
export SEED_TTS_EVAL="${SEED_EVAL}"
export VOXCPM_REPO="${VOXCPM}"          # probe_fm_reward_voxcpm 的 ensure_voxcpm_importable 优先认它
echo "  dots.tts     = ${DOTS_TTS}"
echo "  seed-tts-eval= ${SEED_EVAL}"
echo "  VoxCPM       = ${VOXCPM}"

echo "[1/6] 创建并激活 venv (为什么: 隔离依赖，避免污染系统 Python)"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

echo "[2/6] 以可编辑模式安装 dots.tts (为什么: 我们要 import 其 model/runtime/training 内部模块)"
pip install -e "${DOTS_TTS}"

echo "[3/6] 安装本实验增量依赖 (jiwer/peft/deepspeed/tensorboard ...; UTMOS 走 torch.hub)"
pip install -e "${PROJECT_ROOT}"

# 把 VoxCPM 的 src 挂进本 venv 的 site-packages（写一个 .pth），让 `import voxcpm` 在任意 shell 持久可用。
# 为什么不 `pip install -e VoxCPM`：VoxCPM 复用 dots.tts 已装好的 torch/transformers，pip 装其全部依赖易把
# torch 顶成别的版本→污染主管线；.pth 只加 import 路径、零依赖改动，与探针 ensure_voxcpm_importable 的
# sys.path 思路一致但持久（无需每次 export VOXCPM_REPO）。对实验意义：X10 一键即用、不碰主实验环境。
VOXCPM_SRC="${VOXCPM}/src"
if [[ -d "${VOXCPM_SRC}/voxcpm" ]]; then
  SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
  echo "${VOXCPM_SRC}" > "${SITE}/voxcpm.pth"
  echo "  ✓ VoxCPM 已挂到 site-packages: ${SITE}/voxcpm.pth -> ${VOXCPM_SRC}"
  if python -c "import voxcpm" 2>/dev/null; then
    echo "  ✓ import voxcpm OK"
  else
    echo "  ⚠️ import voxcpm 仍失败——多半缺 VoxCPM 的某个子依赖；X10 跑前按报错 pip install 对应包即可"
    echo "     （X10-only：不影响 dots.tts 主管线；切勿为它升级/降级 torch）"
  fi
else
  echo "  ⚠️ 未找到 ${VOXCPM_SRC}/voxcpm —— VoxCPM 仅 X10 探针需要；确认其 clone 成功（或手动 export VOXCPM_REPO 指向 repo 根）"
fi

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
  python -c "import dots_tts, jiwer, peft; print('ok')"
  python -c "import voxcpm; print('voxcpm ok')"        # X10 二模型探针（可选）
X10 跑前先过命门自检:
  python alignment/probe_fm_reward_voxcpm.py --ckpt openbmb/VoxCPM2 --pairs pairs/out/pairs_best_worst.jsonl --selfcheck
下一步: 见 docs/03-REPRODUCE.md
=============================================================================
EOF
