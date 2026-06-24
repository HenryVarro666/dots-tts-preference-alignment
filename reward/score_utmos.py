"""
score_utmos.py — 阶段③c：候选的 UTMOS（自然度 MOS 代理，越高越好）

做什么:    用 speechmos 的 UTMOS 给每个候选打自然度分，写 utmos.jsonl。
怎么做:    `speechmos` 包惰性加载；UTMOS 吃 16kHz，内部重采样。
为什么:    UTMOS 是免人耳的自然度近似，是多目标奖励第三项，也是**检测 reward hacking 的探针**
           ——「只压 WER」常以牺牲自然度为代价，UTMOS 能把这种退化量化出来。
对实验意义: 没有自然度信号，reward-hacking 臂的「读得清但不像人」就无法被看见。

前置条件: 候选已生成；`pip install speechmos`。
预期结果: reward/out/utmos.jsonl，每行 {fid,k,utmos}。
⚠️ 本机只语法检查。UTMOS 只是近似，不替代真人 MOS（见 ../../学习资料/03）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="候选 UTMOS 打分")
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    args = ap.parse_args()

    from speechmos import utmos  # type: ignore
    import librosa  # type: ignore

    rows = []
    for c in read_jsonl(args.candidates):
        wav, _ = librosa.load(c["audio"], sr=16000)  # UTMOS 期望 16kHz
        score = utmos.run(wav, sr=16000)
        # speechmos 返回 dict 或 float，做稳健取值
        val = score["utmos"] if isinstance(score, dict) and "utmos" in score else float(score)
        rows.append({"fid": c["fid"], "k": c["k"], "utmos": float(val)})
    n = write_jsonl(PATHS.rewards_dir / "utmos.jsonl", rows)
    print(f"✓ UTMOS 打分 {n} 条 -> reward/out/utmos.jsonl")


if __name__ == "__main__":
    main()
