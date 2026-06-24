"""
plots.py — 出图：reward-vs-WER hacking 曲线 + 质量×延迟 Pareto

做什么:
    (1) reward-hacking 图：x=训练步/奖励权重，y1=WER↓，y2=UTMOS/EmergentTTS↓ —— 揭示「WER 降但自然度塌」。
    (2) Pareto 图：x=首包延迟/NFE，y=质量(WER 或 SIM)，标各 treatment（接 Phase B 的少步点）。
怎么做:    读 results/out/master.csv；matplotlib 出图到 results/out/*.png。
为什么:    一图胜千言——reward hacking 的「反向剪刀差」用图最直观；Pareto 是 Phase B 的桥。
对实验意义: 图 1/2 直接进论文 Results。

前置条件: results/out/master.csv（aggregate_results 产出）。
预期结果: results/out/{reward_hacking.png, pareto.png}。
⚠️ 本机只语法检查（无 matplotlib/无数据）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402


def _read_master() -> list[dict]:
    p = PATHS.results_dir / "master.csv"
    if not p.exists():
        print(f"⚠️ 缺 {p}，先跑 aggregate_results.py")
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = _read_master()
    if not rows:
        return

    # 图1：reward hacking 反向剪刀差（treatment 维）
    treats = [r["treatment"] for r in rows]
    wer = [_f(r["wer"]) for r in rows]
    utmos = [_f(r["utmos"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(range(len(treats)), [w or 0 for w in wer], alpha=0.5, label="WER↓")
    ax1.set_xticks(range(len(treats)))
    ax1.set_xticklabels(treats, rotation=30, ha="right")
    ax1.set_ylabel("WER ↓")
    ax2 = ax1.twinx()
    ax2.plot(range(len(treats)), [u or 0 for u in utmos], "o-", color="C3", label="UTMOS↑")
    ax2.set_ylabel("UTMOS ↑")
    ax1.set_title("Reward hacking: 仅WER 把 WER 压最低，UTMOS 反塌")
    fig.tight_layout()
    fig.savefig(PATHS.results_dir / "reward_hacking.png", dpi=150)

    # 图2：质量×延迟 Pareto（首包延迟 vs WER）
    fig2, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        x = _f(r.get("first_packet_ms_p50"))
        y = _f(r.get("wer"))
        if x is not None and y is not None:
            ax.scatter(x, y)
            ax.annotate(f"{r['treatment']}/NFE{r['nfe']}", (x, y), fontsize=7)
    ax.set_xlabel("首包延迟 p50 (ms) ↓")
    ax.set_ylabel("WER ↓")
    ax.set_title("质量×延迟 Pareto（接 Phase B 少步点）")
    fig2.tight_layout()
    fig2.savefig(PATHS.results_dir / "pareto.png", dpi=150)
    print("✓ 出图 -> results/out/{reward_hacking.png, pareto.png}")


if __name__ == "__main__":
    main()
