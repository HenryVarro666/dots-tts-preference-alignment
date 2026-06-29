"""
build_preference_pairs.py — 阶段④：从打分候选构造偏好对(win/lose)

做什么:    读 rewards.jsonl + candidates_index.jsonl，按策略为每个 prompt 生成偏好对，
           写 pairs/out/pairs_{strategy}.jsonl（每行 PairRow，含共享的 text+prompt 条件）。
怎么做:    三种策略——best_worst(每 prompt 1 对) / top2_bottom2(至多 4 对) / threshold(R 差>τ 的全部对)。
为什么这么设计:
    DPO 的「赢/输」必须**共享同一 text+prompt 条件**，差异只在生成质量；
    这样 DPO 的似然比才纯粹反映「模型该偏向哪个输出」。
对实验意义:
    配对策略是消融维度之一：质量(阈值过滤噪声对) vs 数量(top2_bottom2 产更多对)的权衡。

前置条件: reward/out/rewards.jsonl + candidates/out/candidates_index.jsonl。
预期结果: pairs/out/pairs_{strategy}.jsonl。
本机可语法检查；纯文件处理。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, PairRow, read_jsonl, write_jsonl  # noqa: E402


def _load(rewards_path: Path, index_path: Path):
    if not index_path.exists():
        raise SystemExit(f"候选索引不存在: {index_path}（构造偏好对需 join text/prompt 元数据；"
                         f"VoxCPM 原生池请传 --candidates candidates_index_voxcpm.jsonl）")
    cand = {(c["fid"], c["k"]): c for c in read_jsonl(index_path)}
    by_fid: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(rewards_path):
        key = (r["fid"], r["k"])
        meta = cand.get(key, {})
        by_fid[r["fid"]].append({**r, **{
            "text": meta.get("text", ""), "prompt_audio": meta.get("prompt_audio", ""),
            "prompt_text": meta.get("prompt_text", ""), "lang": meta.get("lang", r.get("lang", "")),
        }})
    return by_fid


def _mk(a: dict, b: dict, strategy: str) -> PairRow:
    """a=win, b=lose"""
    return PairRow(
        fid=a["fid"], text=a["text"], prompt_audio=a["prompt_audio"],
        prompt_text=a["prompt_text"], lang=a["lang"],
        win_audio=a["audio"], lose_audio=b["audio"],
        reward_win=a["reward"], reward_lose=b["reward"], strategy=strategy,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="构造偏好对")
    ap.add_argument("--rewards", default=str(PATHS.rewards_dir / "rewards.jsonl"))
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"),
                    help="候选索引（join text/prompt 元数据）；VoxCPM 原生池传 candidates_index_voxcpm.jsonl")
    ap.add_argument("--strategy", choices=["best_worst", "top2_bottom2", "threshold"], default="best_worst")
    ap.add_argument("--tau", type=float, default=0.5, help="threshold 策略的最小奖励差")
    ap.add_argument("--out", default=None, help="输出路径（默认 pairs/out/pairs_{strategy}.jsonl）；reward-hacking 臂用它指定独立文件")
    args = ap.parse_args()

    by_fid = _load(Path(args.rewards), Path(args.candidates))
    pairs: list[PairRow] = []
    for fid, items in by_fid.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x["reward"], reverse=True)  # 高奖励在前
        if args.strategy == "best_worst":
            pairs.append(_mk(items[0], items[-1], args.strategy))
        elif args.strategy == "top2_bottom2":
            tops, bots = items[:2], items[-2:]
            for a, b in itertools.product(tops, bots):
                if a["reward"] > b["reward"]:
                    pairs.append(_mk(a, b, args.strategy))
        else:  # threshold
            for a, b in itertools.combinations(items, 2):
                if a["reward"] - b["reward"] > args.tau:
                    pairs.append(_mk(a, b, args.strategy))

    out = Path(args.out) if args.out else PATHS.pairs_dir / f"pairs_{args.strategy}.jsonl"
    n = write_jsonl(out, pairs)
    print(f"✓ {args.strategy}: {n} 对 -> {out}  (来自 {len(by_fid)} 个 prompt)")


if __name__ == "__main__":
    main()
