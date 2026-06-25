"""
prepare_prompts.py — 从公开语料构造统一的 prompt 清单（zero-shot 克隆请求）

做什么:
    把异质公开语料统一成下游唯一认得的 schema(`PromptRow`)，并切出无泄漏的 train/eval。
    最省事的来源是 dots.tts 自带下载器产出的 `{fid,audio,text}` manifest（推荐 `--corpus dots_manifest`）。
怎么做:
    每种来源一个 adapter，签名统一 `(source: Path, lang: str) -> list[PromptRow]`：
      - dots_manifest: 直接读 dots.tts 的 manifest JSONL（audio 已是绝对路径，最稳）。
      - ljspeech:   读 metadata.csv + wavs/MossFormer2_SR_48K/<fid>.wav（48kHz 超分子目录）。
      - vctk/aishell3: 读各自原生布局。
为什么这么设计:
    dots.tts 的下载器已把 csv→jsonl 做好且写绝对路径，直接复用最不易错；
    统一 schema 让下游(candidates/reward/pairs)与数据源解耦。
对实验意义:
    训练/评测按 fid 哈希切分，确定性、无泄漏 —— 结论可信的底线。

前置条件: 已下载语料 / 已用 dots.tts 下载器生成 manifest（见 data/README.md）。
预期结果: data/prompts/{name}_{train,eval}.jsonl。
⚠️ 本机只语法检查；真正扫描需要已下载的语料/manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, PromptRow, read_jsonl, write_jsonl  # noqa: E402


def _split(fid: str, eval_ratio: float = 0.1) -> str:
    """按 fid 的稳定哈希切 train/eval —— 确定性、可复现、无泄漏。"""
    h = int(hashlib.md5(fid.encode("utf-8")).hexdigest(), 16)
    return "eval" if (h % 1000) / 1000.0 < eval_ratio else "train"


def adapt_dots_manifest(source: Path, lang: str) -> list[PromptRow]:
    """读 dots.tts `prepare_train_jsonl_manifest.py` 产出的 {fid,audio,text} manifest。

    这是**推荐路径**：audio 已是绝对路径、csv 已解析好。自我克隆(prompt=自身)用于打通管线。
    """
    rows: list[PromptRow] = []
    for r in read_jsonl(source):
        rows.append(PromptRow(
            fid=str(r["fid"]), text=r["text"],
            prompt_audio=r["audio"], prompt_text=r["text"], lang=lang, source="dots_manifest",
        ))
    return rows


def adapt_ljspeech(source: Path, lang: str) -> list[PromptRow]:
    """经典 LJSpeech 布局：metadata.csv（fid|raw|normalized）+ wavs/MossFormer2_SR_48K/<fid>.wav。

    注意: dots.tts 48kHz 重制版音频在 `wavs/MossFormer2_SR_48K/`（不是 `wavs/`）。
    若你已用 dots.tts 下载器，更推荐直接用 `--corpus dots_manifest`。
    """
    rows: list[PromptRow] = []
    meta = source / "metadata.csv"
    if not meta.exists():
        print(f"  ⚠️ 未找到 {meta}；若已用 dots.tts 下载器，请改用 --corpus dots_manifest")
        return rows
    audio_dir = source / "wavs" / "MossFormer2_SR_48K"
    if not audio_dir.is_dir():
        audio_dir = source / "wavs"  # 回退到经典布局
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        uid = parts[0].strip()
        text = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else parts[1].strip()
        wav = audio_dir / f"{uid}.wav"
        rows.append(PromptRow(fid=f"ljspeech-{uid}", text=text, prompt_audio=str(wav),
                              prompt_text=text, lang=lang, source="ljspeech"))
    return rows


def adapt_vctk(source: Path, lang: str) -> list[PromptRow]:
    """VCTK-0.92：多说话人 EN。⚠️ wav48_silence_trimmed 为 FLAC，采样率见 LICENSES.md〔需核实〕。"""
    rows: list[PromptRow] = []
    txt_root = source / "txt"
    if not txt_root.exists():
        print(f"  ⚠️ 未找到 {txt_root}（请先下载 VCTK-0.92）")
        return rows
    for spk_dir in sorted(txt_root.glob("p*")):
        for txt in sorted(spk_dir.glob("*.txt")):
            uid = txt.stem
            content = txt.read_text(encoding="utf-8").strip()
            audio = source / "wav48_silence_trimmed" / spk_dir.name / f"{uid}_mic1.flac"
            rows.append(PromptRow(fid=f"vctk-{uid}", text=content, prompt_audio=str(audio),
                                  prompt_text=content, lang=lang, source="vctk"))
    return rows


def adapt_aishell3(source: Path, lang: str) -> list[PromptRow]:
    """AISHELL-3：多说话人 ZH，44.1kHz → 候选生成时由 dots.tts 内部重采样到 48kHz。"""
    rows: list[PromptRow] = []
    content = source / "train" / "content.txt"
    if not content.exists():
        print(f"  ⚠️ 未找到 {content}（请先下载 AISHELL-3）")
        return rows
    for line in content.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        wav_name = cols[0]
        hanzi = "".join(c for c in cols[1] if "一" <= c <= "鿿")
        spk = wav_name[:7]
        audio = source / "train" / "wav" / spk / wav_name
        rows.append(PromptRow(fid=f"aishell3-{wav_name.replace('.wav','')}", text=hanzi,
                              prompt_audio=str(audio), prompt_text=hanzi, lang=lang, source="aishell3"))
    return rows


ADAPTERS = {
    "dots_manifest": adapt_dots_manifest,   # 推荐：读 dots.tts 的 {fid,audio,text} manifest
    "ljspeech": adapt_ljspeech,
    "vctk": adapt_vctk,
    "aishell3": adapt_aishell3,
}
DEFAULT_LANG = {"ljspeech": "en", "vctk": "en", "aishell3": "zh", "dots_manifest": "en"}


def main() -> None:
    ap = argparse.ArgumentParser(description="构造统一 prompt 清单")
    ap.add_argument("--corpus", choices=list(ADAPTERS), required=True)
    ap.add_argument("--source", help="语料目录；dots_manifest 时为 manifest .jsonl 文件路径")
    ap.add_argument("--root", help="(别名) 等同 --source，向后兼容")
    ap.add_argument("--manifest", help="(别名) dots_manifest 的 jsonl，等同 --source")
    ap.add_argument("--lang", default=None, help="语言代码（默认按语料推断）")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（0=全部），用于快速打通")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    args = ap.parse_args()

    source = args.source or args.manifest or args.root
    if not source:
        ap.error("必须给 --source（dots_manifest 给 manifest .jsonl，其余给语料目录）")
    lang = args.lang or DEFAULT_LANG.get(args.corpus, "en")

    rows = ADAPTERS[args.corpus](Path(source), lang)
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
