"""
extract_random.py — 每 prompt 随机抽 1 个候选清单（给 SFT-target 梯度对照臂 X4 用）

做什么:    读 candidates/out/candidates_index.jsonl，对每个 prompt(fid) 用固定种子随机抽 1 个候选，
           抽成 {fid,text,audio,lang} 清单，供 train_sft.py 监督训练。
怎么做:    按 fid 聚合候选→每组内按 k 排序(去除 dict 顺序的不确定性)→用 random.Random(seed) 在排序后的
           fid 列表上逐个 sample 一条；种子固定 ⇒ 完全可复现（同 index + 同 seed ⇒ 同抽样）。
           命令: python pairs/extract_random.py --index candidates/out/candidates_index.jsonl \
                                                --out pairs/out/random_only.jsonl --seed 0
           输入: candidates/out/candidates_index.jsonl(每行 CandidateRow，含 fid/k/audio/text/lang)
           输出: pairs/out/random_only.jsonl(每行 {fid,text,audio,lang}，schema 同 winners_only)
为什么这么设计:
    winners(X3)/losers(X4) 是奖励轴的两端，random 是**奖励轴上的随机/期望点**——同样从 dots.tts 自采样
    候选里取，但不经任何奖励挑选。三者训练管线/步数/损失/数据量全同，唯一变量是 target 在奖励轴上的位置。
    固定 seed + 先排序再抽样：保证账本可逐条溯源、跨机复跑同结果（不引入抽样随机性这个混淆）。
对实验与 claim 的意义:
    random 应落在 winners 与 losers 之间、且不被奖励挑成 off-manifold 极端，是「无偏好信号的 SFT 基线」。
    若只有 winners(最 off-manifold) 崩坏、而 random/losers 不崩，就排除了「SFT 在这点数据量上本就不稳」
    的替代解释，把退化精确归因于「目标 off-manifold」——这是 X4 梯队隔离两种成因的关键一格。

前置条件: candidates/out/candidates_index.jsonl（由 candidates/generate_candidates.py 产出）。
预期结果: pairs/out/random_only.jsonl，每行 {fid,text,audio,lang}（schema 同 winners_only）。
⚠️ 本机只语法检查；纯文件处理，不加载模型。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="每 prompt 随机抽 1 候选（SFT-target 对照臂 X4 用）")
    ap.add_argument("--index", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    ap.add_argument("--out", default=str(PATHS.pairs_dir / "random_only.jsonl"))
    ap.add_argument("--seed", type=int, default=0, help="固定种子，保证可复现")
    args = ap.parse_args()

    by_fid: dict[str, list[dict]] = defaultdict(list)
    for c in read_jsonl(args.index):
        by_fid[c["fid"]].append(c)

    rng = random.Random(args.seed)
    rows = []
    # 先按 fid 排序、组内按 k 排序：消除 dict/文件顺序对抽样的影响，使 (index, seed) 唯一决定结果
    for fid in sorted(by_fid):
        cands = sorted(by_fid[fid], key=lambda x: x["k"])
        c = rng.choice(cands)
        rows.append({"fid": f"{fid}-rand", "text": c["text"],
                     "audio": c["audio"], "lang": c.get("lang", "")})
    n = write_jsonl(args.out, rows)
    print(f"✓ random {n} 条 (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
