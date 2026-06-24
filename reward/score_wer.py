"""
score_wer.py — 阶段③a：候选的 WER（可懂性，越低越好）

做什么:    对每个候选音频做 ASR 重转写，与目标文本算 WER/CER，写 reward/out/wer.jsonl。
怎么做:    EN 用 Whisper-large-v3，ZH 用 Paraformer(funasr)——与 seed-tts-eval 同后端，
           复用其文本归一化(去标点/繁简)，再用 jiwer 算错误率。
为什么:    WER 是「读对了没」的客观信号，是多目标奖励的第一目标，也是 reward-hacking 臂的唯一目标。
对实验意义: 统一 ASR 后端是「数字可比」的前提；EN/ZH 分别用社区标准后端避免跨语言不可比。

前置条件: 候选已生成；funasr/whisper 已装(env/setup_env.sh)。
预期结果: reward/out/wer.jsonl，每行 {fid,k,audio,lang,wer}。
⚠️ 本机只语法检查（无 ASR 模型/无音频）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402


class _ASR:
    """惰性加载的双语 ASR 包装：EN→Whisper，ZH→Paraformer。"""

    def __init__(self) -> None:
        self._whisper = None
        self._paraformer = None

    def transcribe(self, audio: str, lang: str) -> str:
        if lang == "zh":
            if self._paraformer is None:
                from funasr import AutoModel  # type: ignore
                self._paraformer = AutoModel(model="paraformer-zh")
            res = self._paraformer.generate(input=audio)
            return res[0]["text"] if res else ""
        # 默认 EN
        if self._whisper is None:
            import whisper  # type: ignore
            self._whisper = whisper.load_model("large-v3")
        return self._whisper.transcribe(audio)["text"]


def _normalize(text: str, lang: str) -> str:
    """与 seed-tts-eval 一致的归一化：去标点、ZH 繁→简、小写。"""
    import string
    if lang == "zh":
        try:
            import zhconv  # type: ignore
            text = zhconv.convert(text, "zh-cn")
        except Exception:
            pass
        # 去中英标点
        for p in string.punctuation + "，。！？、；：「」『』（）【】…—":
            text = text.replace(p, "")
        return text.replace(" ", "")
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def main() -> None:
    ap = argparse.ArgumentParser(description="候选 WER 打分")
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    args = ap.parse_args()

    import jiwer  # type: ignore
    asr = _ASR()
    rows = []
    for c in read_jsonl(args.candidates):
        lang = c.get("lang", "en") or "en"
        hyp = _normalize(asr.transcribe(c["audio"], lang), lang)
        ref = _normalize(c["text"], lang)
        # ZH 习惯报 CER（字级），EN 报 WER（词级）；jiwer 同一接口
        err = jiwer.cer(ref, hyp) if lang == "zh" else jiwer.wer(ref, hyp)
        rows.append({"fid": c["fid"], "k": c["k"], "audio": c["audio"], "lang": lang, "wer": float(err)})
    n = write_jsonl(PATHS.rewards_dir / "wer.jsonl", rows)
    print(f"✓ WER 打分 {n} 条 -> reward/out/wer.jsonl")


if __name__ == "__main__":
    main()
