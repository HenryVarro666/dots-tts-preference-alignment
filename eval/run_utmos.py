"""
run_utmos.py — 评测：对一个合成目录算平均 UTMOS（自然度代理）

做什么:    遍历 synth 目录所有 wav，用 speechmos UTMOS 打分，输出每条 + 平均。
怎么做:    speechmos 惰性加载；16kHz 重采样。
为什么:    UTMOS 是免人耳的自然度近似，是检测 reward-hacking(WER↓ 但自然度↓)的核心探针。
对实验意义: RQ3 的关键证据列——`dpo_weronly` 预期在这一列显著低于 `dpo_multiobj`。

前置条件: 合成目录；`pip install speechmos`。
预期结果: out/{treatment}.utmos.jsonl + 终端平均。
⚠️ 本机只语法检查；UTMOS 仅近似，不替代真人 MOS。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="UTMOS 评测")
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--treatment", required=True)
    args = ap.parse_args()

    from speechmos import utmos  # type: ignore
    import librosa  # type: ignore

    rows, scores = [], []
    for wav in sorted(Path(args.synth_dir).glob("*.wav")):
        y, _ = librosa.load(str(wav), sr=16000)
        s = utmos.run(y, sr=16000)
        val = float(s["utmos"]) if isinstance(s, dict) and "utmos" in s else float(s)
        rows.append({"fid": wav.stem, "utmos": val})
        scores.append(val)

    out = Path(__file__).parent.parent / "results" / "out" / f"{args.treatment}.utmos.jsonl"
    write_jsonl(out, rows)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"✓ [{args.treatment}] 平均 UTMOS = {avg:.3f} (n={len(scores)}) -> {out}")


if __name__ == "__main__":
    main()
