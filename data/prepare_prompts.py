"""
prepare_prompts.py — 从公开语料构造统一的 prompt 清单（zero-shot 克隆请求）

做什么:
    扫描已下载的公开语料(VCTK / AISHELL-3 / LJSpeech-48kHz)，输出统一 schema 的
    `data/prompts/{corpus}_prompts.jsonl`，字段 = PromptRow(fid,text,prompt_audio,
    prompt_text,lang,source)。同时切出「训练/偏好用」与「评测用」两份不重叠子集。
怎么做:
    每个语料一个 adapter 函数，把其原生目录结构/转写文件映射到 PromptRow；
    对非 48kHz 语料只「记录待重采样」标志，真正重采样在候选生成时由 dots.tts 内部
    high_quality_resample 处理（避免在此处引入额外音频依赖）。
为什么这么设计:
    dots.tts 的训练/推理契约是 48kHz；统一 schema 让下游(candidates/reward/pairs)
    完全不关心数据来自哪个语料 —— 解耦数据源与算法。
对实验意义:
    训练集与评测集**严格不重叠**是结论可信的底线；本脚本用确定性切分(按 speaker/hash)
    保证可复现且无泄漏。

前置条件: 已运行 data/fetch_public_corpora.sh 下载语料；许可见 LICENSES.md。
预期结果: data/prompts/ 下生成 {vctk,aishell3,ljspeech}_{train,eval}.jsonl。
⚠️ 本机只语法检查；真正扫描需要已下载的语料目录。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, PromptRow, write_jsonl  # noqa: E402


def _split(fid: str, eval_ratio: float = 0.1) -> str:
    """按 fid 的稳定哈希切 train/eval —— 确定性、可复现、无泄漏。"""
    h = int(hashlib.md5(fid.encode("utf-8")).hexdigest(), 16)
    return "eval" if (h % 1000) / 1000.0 < eval_ratio else "train"


def adapt_ljspeech(root: Path) -> list[PromptRow]:
    """LJSpeech-48kHz：单说话人 EN，48kHz，最低风险首跑数据。

    复用 dots.tts `prepare_train_jsonl_manifest.py` 的下载产物；metadata.csv 为
    `id|raw_text|normalized_text`，音频在 wavs/ 下。这里把同一条既当合成文本、又当
    参考音频(自我克隆)，用于快速打通管线。
    """
    rows: list[PromptRow] = []
    meta = root / "metadata.csv"
    if not meta.exists():
        print(f"  ⚠️ 未找到 {meta}（请先下载 LJSpeech-48kHz）")
        return rows
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        uid, text = parts[0], parts[-1].strip()
        wav = root / "wavs" / f"{uid}.wav"
        rows.append(PromptRow(
            fid=f"ljspeech-{uid}", text=text,
            prompt_audio=str(wav), prompt_text=text, lang="en", source="ljspeech",
        ))
    return rows


def adapt_vctk(root: Path) -> list[PromptRow]:
    """VCTK-0.92：多说话人 EN。⚠️ 标准发布是 48kHz 但 wav48_silence_trimmed 为 FLAC，
    采样率/格式见 LICENSES.md〔需核实〕。txt/ 下是每条转写。
    """
    rows: list[PromptRow] = []
    txt_root = root / "txt"
    if not txt_root.exists():
        print(f"  ⚠️ 未找到 {txt_root}（请先下载 VCTK-0.92）")
        return rows
    for spk_dir in sorted(txt_root.glob("p*")):
        for txt in sorted(spk_dir.glob("*.txt")):
            uid = txt.stem  # e.g. p225_001
            audio = root / "wav48_silence_trimmed" / spk_dir.name / f"{uid}_mic1.flac"
            rows.append(PromptRow(
                fid=f"vctk-{uid}", text=txt.read_text(encoding="utf-8").strip(),
                prompt_audio=str(audio), prompt_text=txt.read_text(encoding="utf-8").strip(),
                lang="en", source="vctk",
            ))
    return rows


def adapt_aishell3(root: Path) -> list[PromptRow]:
    """AISHELL-3：多说话人 ZH，44.1kHz → 候选生成时重采样到 48kHz。
    content.txt 为 `wav_name\t拼音/汉字` 形式（按版本不同，这里取汉字转写列）。
    """
    rows: list[PromptRow] = []
    content = root / "train" / "content.txt"
    if not content.exists():
        print(f"  ⚠️ 未找到 {content}（请先下载 AISHELL-3）")
        return rows
    for line in content.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        wav_name = cols[0]
        # 偶数列常为拼音、奇数为汉字；这里做稳健处理：取所有汉字字符
        hanzi = "".join(c for c in cols[1] if "一" <= c <= "鿿")
        spk = wav_name[:7]
        audio = root / "train" / "wav" / spk / wav_name
        rows.append(PromptRow(
            fid=f"aishell3-{wav_name.replace('.wav','')}", text=hanzi,
            prompt_audio=str(audio), prompt_text=hanzi, lang="zh", source="aishell3",
        ))
    return rows


ADAPTERS = {"ljspeech": adapt_ljspeech, "vctk": adapt_vctk, "aishell3": adapt_aishell3}


def main() -> None:
    ap = argparse.ArgumentParser(description="构造统一 prompt 清单")
    ap.add_argument("--corpus", choices=list(ADAPTERS), required=True)
    ap.add_argument("--root", required=True, help="该语料的下载根目录")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（0=全部），用于快速打通")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    args = ap.parse_args()

    rows = ADAPTERS[args.corpus](Path(args.root))
    if args.limit:
        rows = rows[: args.limit]
    train = [r for r in rows if _split(r.fid, args.eval_ratio) == "train"]
    ev = [r for r in rows if _split(r.fid, args.eval_ratio) == "eval"]

    out_dir = PATHS.prompts_dir
    n_tr = write_jsonl(out_dir / f"{args.corpus}_train.jsonl", train)
    n_ev = write_jsonl(out_dir / f"{args.corpus}_eval.jsonl", ev)
    print(f"✓ {args.corpus}: train={n_tr}  eval={n_ev}  -> {out_dir}")
    print("  (训练/评测按 fid 哈希切分，确定性、无泄漏)")


if __name__ == "__main__":
    main()
