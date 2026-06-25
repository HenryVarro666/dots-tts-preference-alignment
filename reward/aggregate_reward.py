"""
aggregate_reward.py — 阶段③d：把 WER/SIM/UTMOS 聚合成单一奖励 R

做什么:    join 三个打分文件(by fid,k)，**按每个 prompt 的候选池**做 z-normalize，
           再加权 R = w_wer·(−WER_z) + w_sim·SIM_z + w_utmos·UTMOS_z，写 rewards.jsonl。
怎么做:    纯标准库；权重从 CLI/配置读；同时打印每 prompt 的奖励方差（诊断候选多样性）。
为什么这么设计:
    三指标量纲不同(WER 百分比 / SIM 余弦 / UTMOS 1~5)，直接相加无意义；
    **按候选池 z-norm** 让它们可比，且让奖励聚焦「同一 prompt 内谁更好」(正是 DPO 需要的相对信号)。
    reward-hacking 臂只需把权重设成 (1,0,0)。
对实验意义:
    奖励权重就是「我们告诉模型什么算好」。多目标 vs 仅 WER 的对照，正是 RQ1/RQ3 的实验操纵变量。

前置条件: wer.jsonl / sim.jsonl / utmos.jsonl 都已生成。
预期结果: reward/out/rewards.jsonl（每行 RewardRow）+ 终端打印奖励离散度诊断。
⚠️ 本机可语法检查；逻辑为纯文件处理，理论上离线可跑（若有打分文件）。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, RewardRow, read_jsonl, write_jsonl, zscore  # noqa: E402


def _index(path: Path, key: str) -> dict[tuple, float]:
    """读打分文件为 {(fid,k): value}；文件不存在则返回空 dict（容错，便于跳过某指标）。"""
    if not Path(path).exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        out[(r["fid"], r["k"])] = float(r[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="多目标奖励聚合")
    ap.add_argument("--w-wer", type=float, default=0.4)
    ap.add_argument("--w-sim", type=float, default=0.3)
    ap.add_argument("--w-utmos", type=float, default=0.3)
    ap.add_argument("--out", default=str(PATHS.rewards_dir / "rewards.jsonl"))
    args = ap.parse_args()

    wer = _index(PATHS.rewards_dir / "wer.jsonl", "wer")
    sim = _index(PATHS.rewards_dir / "sim.jsonl", "sim")
    utm = _index(PATHS.rewards_dir / "utmos.jsonl", "utmos")
    if not wer:
        raise SystemExit("缺 reward/out/wer.jsonl —— 先跑 score_wer.py（WER 是必需项）")

    # 缺哪个打分文件就把它的权重强制为 0 并告警（容错：冒烟阶段常先跳过 SIM）
    w_wer, w_sim, w_utmos = args.w_wer, args.w_sim, args.w_utmos
    if not utm and w_utmos > 0:
        print("⚠️ 缺 utmos.jsonl → w-utmos 置 0（先跑 score_utmos.py 可纳入）"); w_utmos = 0.0
    if not sim and w_sim > 0:
        print("⚠️ 缺 sim.jsonl → w-sim 置 0（放好 WavLM ckpt 或用 score_sim.py --backend ecapa 后可纳入）"); w_sim = 0.0
    tot = w_wer + w_sim + w_utmos
    if tot <= 0:
        raise SystemExit("三项权重全为 0，无法聚合")

    lang = {(r["fid"], r["k"]): r.get("lang", "") for r in read_jsonl(PATHS.rewards_dir / "wer.jsonl")}
    audio = {(r["fid"], r["k"]): r["audio"] for r in read_jsonl(PATHS.rewards_dir / "wer.jsonl")}

    # 按 fid 分组（z-norm 在「同一 prompt 的候选池」内做）
    by_fid: dict[str, list[int]] = defaultdict(list)
    for (fid, k) in wer:
        by_fid[fid].append(k)

    rows: list[RewardRow] = []
    variances = []
    for fid, ks in by_fid.items():
        ks = sorted(ks)
        wer_z = zscore([wer[(fid, k)] for k in ks])
        sim_z = zscore([sim.get((fid, k), 0.0) for k in ks])
        utm_z = zscore([utm.get((fid, k), 0.0) for k in ks])
        rewards_here = []
        for i, k in enumerate(ks):
            # 注意 WER 越低越好，故取 −WER_z；权重为缺文件后修正过的有效权重
            R = w_wer * (-wer_z[i]) + w_sim * sim_z[i] + w_utmos * utm_z[i]
            rewards_here.append(R)
            rows.append(RewardRow(
                fid=fid, k=k, audio=audio[(fid, k)], lang=lang.get((fid, k), ""),
                wer=wer[(fid, k)], sim=sim.get((fid, k), 0.0), utmos=utm.get((fid, k), 0.0),
                reward=R, wer_z=-wer_z[i], sim_z=sim_z[i], utmos_z=utm_z[i],
            ))
        # 候选池奖励方差 → 诊断多样性（方差太小=偏好信号弱）
        m = sum(rewards_here) / len(rewards_here)
        variances.append(sum((x - m) ** 2 for x in rewards_here) / len(rewards_here))

    n = write_jsonl(args.out, rows)
    avg_var = sum(variances) / len(variances) if variances else 0.0
    print(f"✓ 聚合奖励 {n} 条 -> {args.out}")
    print(f"  有效权重 (wer,sim,utmos)=({w_wer},{w_sim},{w_utmos})"
          + ("" if (w_wer, w_sim, w_utmos) == (args.w_wer, args.w_sim, args.w_utmos)
             else f"  [原始 {args.w_wer},{args.w_sim},{args.w_utmos}，缺文件项已置 0]"))
    print(f"  候选池平均奖励方差 = {avg_var:.4f}  (太小→偏好信号弱，需加大候选多样性，见 docs/04-RISKS 风险①)")


if __name__ == "__main__":
    main()
