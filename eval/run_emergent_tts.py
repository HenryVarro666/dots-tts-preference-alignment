"""
run_emergent_tts.py — 评测：EmergentTTS-Eval 表现力胜率（LLM-judge）

做什么:    在 EmergentTTS-Eval(1645 条，6 类难例)上跑某 treatment，得 head-to-head 胜率。
怎么做:    调用外部 EmergentTTS-Eval(需另 clone)：用本项目的合成函数作为「被测系统」，
           与固定参照(gpt-4o-mini-tts)对比，由 LLM-judge(如 Gemini-2.5-Pro)打偏好。
为什么:    WER/SIM 测不出情感/副语言/句法复杂度等表现力；这是少数第三方表现力基准。
对实验意义:
    ⚠️ 关键预期(文档 05 line 325)：SOAR/对齐可能在表现力上有 tradeoff(dots.tts 自身 SOAR
    把 Emotions 72.7→63.9)。本列用于检验对齐是否以牺牲表现力换鲁棒——RQ3 的另一面。

前置条件: `git clone boson-ai/EmergentTTS-Eval-public`(arXiv 2505.23009)；judge API key。
预期结果: results/out/{treatment}.emergent.json（各类目胜率 + overall）。
⚠️ 本机只语法检查；胜率是「对固定参照的 LLM-judge 偏好」，非绝对分(报告时标注)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="EmergentTTS-Eval 表现力评测")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--emergent-root", default=str(PATHS.project_root / ".third_party" / "EmergentTTS-Eval"))
    ap.add_argument("--judge", default="gemini-2.5-pro")
    args = ap.parse_args()

    root = Path(args.emergent_root)
    if not root.exists():
        raise SystemExit(
            f"未找到 EmergentTTS-Eval: {root}\n"
            f"请先: git clone https://github.com/boson-ai/EmergentTTS-Eval-public {root}")

    sys.path.insert(0, str(root))
    # 用我们的 runtime 作为被测 TTS 后端，接入 EmergentTTS 的评测循环。
    # 具体接线依其 API（categories: Emotions/Paralinguistics/SyntacticComplexity/
    # ForeignWords/ComplexPronunciation/QuestionsErrors）。
    from common import ensure_dots_tts_importable  # noqa
    ensure_dots_tts_importable()
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore
    runtime = DotsTtsRuntime.from_pretrained(args.ckpt)

    def tts_fn(text: str, prompt_audio: str | None = None) -> str:
        res = runtime.generate(text=text, prompt_audio_path=prompt_audio,
                               prompt_text="", normalize_text=True)
        # 写临时 wav 返回路径，交给 EmergentTTS 的 judge 流程
        import soundfile as sf  # type: ignore
        tmp = root / "_synth" / f"{abs(hash(text))}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        audio = res["audio"]
        try:
            audio = audio.float().cpu().squeeze().numpy()
        except AttributeError:
            pass
        sf.write(str(tmp), audio, int(res["sample_rate"]))
        return str(tmp)

    # TODO(对照 EmergentTTS-Eval API): run_eval(tts_fn, judge=args.judge) -> per-category win-rate
    print("⚠️ 接线点：调用 EmergentTTS-Eval 的评测入口，传入 tts_fn 与 judge。")
    result = {"treatment": args.treatment, "judge": args.judge,
              "note": "win-rate vs gpt-4o-mini-tts（非绝对分）", "categories": {}, "overall": None}

    out = PATHS.results_dir / f"{args.treatment}.emergent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 模板已写 -> {out}（补全 EmergentTTS 接线后产出真实胜率）")


if __name__ == "__main__":
    main()
