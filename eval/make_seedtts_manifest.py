"""
make_seedtts_manifest.py — 为 seed-tts-eval 生成 pipe 分隔的 meta_lst

做什么:    把我们的评测清单 + 合成输出目录，转成 seed-tts-eval 认得的 5 字段 meta:
           `utt|prompt_text|prompt_wav|infer_text|gt_wav`，供 cal_wer.sh / cal_sim.sh。
怎么做:    读评测 jsonl，逐条拼字段；synth 目录里 {fid}.wav 即合成结果(utt=fid)。
为什么:    seed-tts-eval 的 get_wav_res_ref_text.py 用此格式；统一接口才能复用其 WER/SIM。
对实验意义: 让我们的输出直接对接业界事实标准评测脚本，结果与榜单可比。

前置条件: 评测 jsonl(含 text/prompt_text/prompt_audio/可选 gt_audio)。
预期结果: eval/eval_sets/{name}.meta.lst。
本机可语法检查；纯文本处理。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构造 seed-tts-eval meta_lst")
    ap.add_argument("--manifest", required=True, help="评测 jsonl")
    ap.add_argument("--out", required=True, help="输出 .meta.lst")
    args = ap.parse_args()

    lines = []
    for r in read_jsonl(args.manifest):
        gt = r.get("gt_audio", r.get("prompt_audio", ""))
        # 5 字段：utt | prompt_text | prompt_wav | infer_text | gt_wav
        lines.append("|".join([
            r["fid"], r.get("prompt_text", ""), r.get("prompt_audio", ""),
            r["text"], gt,
        ]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ {len(lines)} 行 -> {out}")


if __name__ == "__main__":
    main()
