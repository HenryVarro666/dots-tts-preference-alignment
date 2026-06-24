#!/usr/bin/env bash
# =============================================================================
# fetch_public_corpora.sh — 下载本实验用到的公开语料
#
# 做什么:   下载 LJSpeech-48kHz(复用 dots.tts 脚本) / VCTK-0.92 / AISHELL-3 到本地。
# 怎么做:   优先复用 dots.tts 自带的 LJSpeech 下载器；VCTK/AISHELL-3 给出官方链接与
#           落地路径(部分需手动同意许可，无法纯脚本化)。
# 为什么:   实验只用公开集(用户无私有数据)；许可/采样率差异在 LICENSES.md 标注。
# 对实验意义: 数据来源透明可溯源；采样率/许可是合规与可比性的前提。
#
# 前置条件: 网络；磁盘(VCTK~数十 GB，AISHELL-3~十几 GB)。
# 预期结果: data/corpora/{ljspeech48k,vctk,aishell3}/ 落地。
# ⚠️ 本机只 `bash -n`；真正下载在 GPU 机器上。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
CORPORA="${HERE}/corpora"
mkdir -p "${CORPORA}"

echo "[LJSpeech-48kHz] 复用 dots.tts 自带下载器 (为什么: 它已对齐 48kHz 训练契约)"
DOTS_PREP="${REPO_ROOT}/参考代码/dots.tts/scripts/prepare_train_jsonl_manifest.py"
if [[ -f "${DOTS_PREP}" ]]; then
  echo "  python ${DOTS_PREP} --output-dir ${CORPORA}/ljspeech48k   # 见其 --help"
else
  echo "  ⚠️ 未找到 ${DOTS_PREP}"
fi

echo "[VCTK-0.92] CC-BY-4.0；官方: https://doi.org/10.7488/ds/2645"
echo "  → 下载 VCTK-Corpus-0.92.zip 解压到 ${CORPORA}/vctk"
echo "  ⚠️ 默认 wav48_silence_trimmed 为 FLAC；采样率/格式跑前核实 (LICENSES.md)"

echo "[AISHELL-3] Apache-2.0；官方: https://www.openslr.org/93/"
echo "  → wget https://www.openslr.org/resources/93/data_aishell3.tgz 解压到 ${CORPORA}/aishell3"
echo "  ⚠️ 原生 44.1kHz；候选生成时由 dots.tts 内部重采样到 48kHz"

echo "完成提示。下一步: python data/prepare_prompts.py --corpus <name> --root ${CORPORA}/<dir>"
