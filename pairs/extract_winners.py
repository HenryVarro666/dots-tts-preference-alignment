"""
extract_winners.py — 从偏好对抽取 winners 清单（给 SFT 控制变量用）

做什么:    读 pairs_*.jsonl，把每对的 winner 音频抽成 {fid,text,audio} 清单，供 train_sft.py 监督训练。
怎么做:    audio = win_audio（SFT 学习「文本→高奖励生成」；x-vector 由该音频自身提供，无需额外 prompt）。
为什么:    SFT 控制变量需要「只看 winners」的数据，与 DPO 的(win,lose)对照，分离「更多好数据」vs「对比偏好」。
对实验意义: 支撑 RQ1 —— 若 dpo 优于 sft，才证明是偏好对齐而非数据增量在起作用。

前置条件: pairs/out/pairs_best_worst.jsonl。
预期结果: pairs/out/winners_only.jsonl，每行 {fid,text,audio}。
本机可语法检查；纯文件处理。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取 winners 清单（SFT 用）")
    ap.add_argument("--pairs", default=str(PATHS.pairs_dir / "pairs_best_worst.jsonl"))
    ap.add_argument("--out", default=str(PATHS.pairs_dir / "winners_only.jsonl"))
    args = ap.parse_args()

    rows = []
    for r in read_jsonl(args.pairs):
        rows.append({"fid": f"{r['fid']}-win", "text": r["text"],
                     "audio": r["win_audio"], "lang": r.get("lang", "")})
    n = write_jsonl(args.out, rows)
    print(f"✓ winners {n} 条 -> {args.out}")


if __name__ == "__main__":
    main()
