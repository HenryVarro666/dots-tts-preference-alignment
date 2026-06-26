"""
extract_losers.py — 从偏好对抽取 losers 清单（给 SFT-target 梯度对照臂 X4 用）

做什么:    读 pairs_*.jsonl，把每对的 loser 音频抽成 {fid,text,audio,lang} 清单，供 train_sft.py 监督训练。
怎么做:    audio = lose_audio（SFT 学习「文本→低奖励生成」；manifest schema 与 winners_only 完全同形，
           只是 fid 后缀换成 -lose、audio 换成 lose_audio）。
           命令: python pairs/extract_losers.py --pairs pairs/out/pairs_best_worst.jsonl \
                                                --out pairs/out/losers_only.jsonl
           输入: pairs/out/pairs_best_worst.jsonl(每行 PairRow，含 win_audio/lose_audio)
           输出: pairs/out/losers_only.jsonl(每行 {fid,text,audio,lang})
为什么这么设计:
    winners(X3) 已是「best 候选 SFT」；本臂换成「worst 候选 SFT」，构成 SFT-target 的梯度对照——
    训练管线、步数、损失、数据量全相同，**唯一变量是目标音频在奖励轴上的位置**（best vs worst）。
    复用 extract_winners 的同形 schema，保证两臂除 target 外完全可比（不引入 schema 差异这个混淆）。
对实验与 claim 的意义:
    与 winners/random/groundtruth 一起，构成隔离「off-manifold 目标退化」vs「小数据 SFT 本身不稳」的
    决定性梯队。预期 losers 居中（worst 候选虽差，仍是 dots.tts 自采样、留在生成流形上，
    不像 winner 那样被奖励挑成 off-manifold 极端）——若 winners 崩得比 losers 更惨，
    才能把退化归因于「目标 off-manifold」而非「SFT 数据太少」。

前置条件: pairs/out/pairs_best_worst.jsonl（由 pairs/build_preference_pairs.py 产出）。
预期结果: pairs/out/losers_only.jsonl，每行 {fid,text,audio,lang}（schema 同 winners_only）。
⚠️ 本机只语法检查；纯文件处理，不加载模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取 losers 清单（SFT-target 对照臂 X4 用）")
    ap.add_argument("--pairs", default=str(PATHS.pairs_dir / "pairs_best_worst.jsonl"))
    ap.add_argument("--out", default=str(PATHS.pairs_dir / "losers_only.jsonl"))
    args = ap.parse_args()

    rows = []
    for r in read_jsonl(args.pairs):
        rows.append({"fid": f"{r['fid']}-lose", "text": r["text"],
                     "audio": r["lose_audio"], "lang": r.get("lang", "")})
    n = write_jsonl(args.out, rows)
    print(f"✓ losers {n} 条 -> {args.out}")


if __name__ == "__main__":
    main()
