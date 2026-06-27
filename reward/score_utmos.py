"""
score_utmos.py — 阶段③c：候选的 UTMOS（自然度 MOS 代理，越高越好）

做什么:    用 UTMOS(torch.hub utmos22_strong) 给每个候选打自然度分，写 utmos.jsonl。
怎么做:    UtmosScorer(torch.hub utmos22_strong) 惰性加载、自动 GPU；16kHz 输入。
为什么:    UTMOS 是免人耳的自然度近似，是多目标奖励第三项，也是**检测 reward hacking 的探针**
           ——「只压 WER」常以牺牲自然度为代价，UTMOS 能把这种退化量化出来。
对实验意义: 没有自然度信号，reward-hacking 臂的「读得清但不像人」就无法被看见。

前置条件: 候选已生成；torch + librosa（已装）；首次联网下载 utmos22_strong。
预期结果: reward/out/utmos.jsonl，每行 {fid,k,utmos}。
⚠️ 本机只语法检查。UTMOS 只是近似，不替代真人 MOS（见 ../../学习资料/03）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402
from utmos_backend import UtmosScorer  # noqa: E402

# 进度可视化（UTMOS 模型很轻，但逐条 librosa.load 在上千条时也要十几秒~分钟级，给条进度条）。
try:
    from tqdm import tqdm  # noqa: E402
except Exception:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore
        return it


def main() -> None:
    ap = argparse.ArgumentParser(description="候选 UTMOS 打分")
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    args = ap.parse_args()

    scorer = UtmosScorer()
    cands = list(read_jsonl(args.candidates))  # 物化以便 tqdm 知道总数、显示百分比/ETA
    rows = []
    for c in tqdm(cands, desc="UTMOS", unit="clip"):
        rows.append({"fid": c["fid"], "k": c["k"], "utmos": scorer.score(c["audio"])})
    n = write_jsonl(PATHS.rewards_dir / "utmos.jsonl", rows)
    print(f"✓ UTMOS 打分 {n} 条 -> reward/out/utmos.jsonl")


if __name__ == "__main__":
    main()
