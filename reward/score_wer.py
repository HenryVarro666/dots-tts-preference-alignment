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
    """惰性加载的双语 ASR 包装：EN→Whisper(transformers pipeline)，ZH→Paraformer(funasr)。

    为什么用 transformers 的 Whisper 而非 openai-whisper:
        transformers 是 dots.tts 的硬依赖（一定已装），用它的 ASR pipeline 加载
        openai/whisper-large-v3，避免再装 openai-whisper 带来的 torch 版本冲突。
        与 seed-tts-eval 同款 large-v3 后端 → 结果可比。
    """

    def __init__(self, whisper_model: str = "openai/whisper-large-v3") -> None:
        self._whisper = None
        self._paraformer = None
        self._whisper_model = whisper_model

    def _ensure_whisper(self):
        if self._whisper is not None:
            return
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore
        use_cuda = torch.cuda.is_available()
        self._whisper = pipeline(
            "automatic-speech-recognition",
            model=self._whisper_model,
            device=0 if use_cuda else -1,
            dtype=torch.float16 if use_cuda else torch.float32,  # 新 API（torch_dtype 已弃用）
            # 不设 chunk_length_s：TTS 样本 <30s，Whisper 原生窗口即可，避免实验性分块警告
        )

    def transcribe_en_batch(self, audios: list[str], batch_size: int = 8) -> list[str]:
        """批量转写英文：一次喂一批给 pipeline（GPU 利用率远高于逐条）。"""
        self._ensure_whisper()
        outs = self._whisper(
            list(audios), batch_size=batch_size,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        if isinstance(outs, dict):  # 传单条时 pipeline 返回 dict
            outs = [outs]
        return [o["text"] if isinstance(o, dict) else str(o) for o in outs]

    def transcribe_zh(self, audio: str) -> str:
        if self._paraformer is None:
            from funasr import AutoModel  # type: ignore
            self._paraformer = AutoModel(model="paraformer-zh")
        res = self._paraformer.generate(input=audio)
        return res[0]["text"] if res else ""


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
    ap.add_argument("--batch-size", type=int, default=8, help="Whisper 批大小（GPU 利用率↑）")
    args = ap.parse_args()

    import jiwer  # type: ignore
    asr = _ASR()
    cands = list(read_jsonl(args.candidates))
    en = [c for c in cands if (c.get("lang") or "en") != "zh"]
    zh = [c for c in cands if (c.get("lang") or "en") == "zh"]

    rows = []
    # EN：批量转写（一次一批喂 Whisper，远快于逐条）
    if en:
        hyps = asr.transcribe_en_batch([c["audio"] for c in en], batch_size=args.batch_size)
        for c, hyp in zip(en, hyps):
            err = jiwer.wer(_normalize(c["text"], "en"), _normalize(hyp, "en"))
            rows.append({"fid": c["fid"], "k": c["k"], "audio": c["audio"], "lang": "en", "wer": float(err)})
    # ZH：funasr 逐条（习惯报 CER 字级）
    for c in zh:
        hyp = _normalize(asr.transcribe_zh(c["audio"]), "zh")
        err = jiwer.cer(_normalize(c["text"], "zh"), hyp)
        rows.append({"fid": c["fid"], "k": c["k"], "audio": c["audio"], "lang": "zh", "wer": float(err)})

    # 还原成与候选一致的顺序（en/zh 分组后可能乱序）
    order = {(c["fid"], c["k"]): i for i, c in enumerate(cands)}
    rows.sort(key=lambda r: order.get((r["fid"], r["k"]), 0))
    n = write_jsonl(PATHS.rewards_dir / "wer.jsonl", rows)
    print(f"✓ WER 打分 {n} 条（EN 批大小={args.batch_size}）-> reward/out/wer.jsonl")


if __name__ == "__main__":
    main()
