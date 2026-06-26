"""
inspect_gap.py — 诊断偏好对的「质量差」分布（回答：DPO margin 涨不动是不是数据问题）

做什么:
    读 build_preference_pairs.py 产出的 pairs_*.jsonl，统计每对的奖励差
    gap = reward_win − reward_lose 的分布（分位数 + 粗直方图 + 近似平局占比）。
怎么做:
    python pairs/inspect_gap.py [pairs/out/pairs_best_worst.jsonl ...]
    不带参数则自动扫 pairs/out/*.jsonl。逐文件汇报。
为什么这么做:
    DPO 用 FM-loss 当隐式奖励，只能分开「声学质量真有差」的 pair；若 winner/loser
    是同一句的近似平局候选（gap≈0），无论 β 多大 margin 都涨不起来（实测 run3 β=0.5：
    多数 step margin~0.01、个别真有差的 step 才炸到 0.5–0.8）。本脚本量化「近似平局占比」，
    定位瓶颈到底在「超参」还是「数据」，并指导是否要 threshold 重造 pair / 重生成更分散的候选。
对实验意义:
    若 gap 普遍小 → margin 上限受数据制约（非超参），需提高候选多样性或按 tau 阈值筛 pair；
    这是 docs/05 B-16「数据瓶颈」诊断的量化证据，也支撑论文 §6.2/§7。
本机只语法检查；真实运行在有 pairs 数据的服务器上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev


def _percentile(xs: list[float], q: float) -> float:
    """线性插值分位数（q∈[0,1]）；xs 须已排序、非空。"""
    if not xs:
        return float("nan")
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _histogram(gaps: list[float], bins: int = 10, width: int = 40) -> list[str]:
    """粗 ASCII 直方图，看分布形状（单峰近 0 = 数据瓶颈；双峰 = 部分对可用）。"""
    lo, hi = gaps[0], gaps[-1]
    if hi - lo < 1e-9:
        return [f"  所有 gap≈{lo:.3f}（完全无展开）"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for g in gaps:
        idx = min(int((g - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    out = []
    for i, c in enumerate(counts):
        left = lo + i * step
        bar = "█" * round(width * c / peak)
        out.append(f"  [{left:+.2f},{left + step:+.2f}) {bar} {c}")
    return out


def inspect(path: Path) -> None:
    gaps: list[float] = []
    n_bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        try:
            g = float(rec["reward_win"]) - float(rec["reward_lose"])
        except (KeyError, TypeError, ValueError):
            n_bad += 1
            continue
        gaps.append(g)

    print(f"\n===== {path} =====")
    if not gaps:
        print(f"  ⚠️ 无可用 gap（缺 reward_win/reward_lose？跳过 {n_bad} 行）")
        return
    gaps.sort()
    n = len(gaps)
    # 近似平局：gap 小于若干阈值的占比（z-norm 奖励下，gap<0.25 基本不可分）
    ties = {t: sum(1 for g in gaps if g < t) for t in (0.1, 0.25, 0.5, 1.0)}
    print(f"  n={n} 对  mean={mean(gaps):+.3f}  std={pstdev(gaps):.3f}")
    print(f"  分位: min={gaps[0]:+.3f}  p10={_percentile(gaps,0.10):+.3f}  "
          f"p25={_percentile(gaps,0.25):+.3f}  median={_percentile(gaps,0.50):+.3f}  "
          f"p75={_percentile(gaps,0.75):+.3f}  p90={_percentile(gaps,0.90):+.3f}  max={gaps[-1]:+.3f}")
    for t, c in ties.items():
        print(f"  gap < {t:<4}: {c:4d} 对 ({100*c/n:5.1f}%)  {'← 近似平局，DPO 分不开' if t <= 0.25 else ''}")
    print("  直方图（gap 分布形状）:")
    for row in _histogram(gaps):
        print(row)
    # 一句话诊断
    frac_tie = ties[0.25] / n
    if frac_tie > 0.6:
        print(f"  ▶ 诊断：{100*frac_tie:.0f}% 的对 gap<0.25（近似平局）→ **数据瓶颈**：")
        print("      候选池多样性不足。建议：① threshold 策略筛大 gap 对（--strategy threshold --tau）；")
        print("      ② 或重生成更分散的候选（扩大 grid / 提高 temperature / 增大 K）。")
    elif frac_tie > 0.3:
        print(f"  ▶ 诊断：{100*frac_tie:.0f}% 近似平局——部分对可用。建议 threshold 筛掉平局对再训。")
    else:
        print(f"  ▶ 诊断：仅 {100*frac_tie:.0f}% 近似平局——pair 质量差充足，margin 不动更可能是超参/优化问题。")


def main() -> None:
    args = [Path(p) for p in sys.argv[1:]]
    if not args:
        out_dir = Path(__file__).resolve().parent / "out"
        args = sorted(out_dir.glob("*.jsonl"))
        if not args:
            print(f"未找到 pairs 文件于 {out_dir}/*.jsonl；可显式传路径。")
            return
    for p in args:
        if p.exists():
            inspect(p)
        else:
            print(f"⚠️ 不存在：{p}")


if __name__ == "__main__":
    main()
