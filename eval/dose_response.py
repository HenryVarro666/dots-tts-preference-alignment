"""
dose_response.py — ★1 剂量-反应：训练目标的 off-manifold 度 → 崩坏（把「诊断 ↔ 崩坏」从相关升为因果桥）

背景（EMNLP 审稿 claim 4/9 缺因果桥）:
    X4 SFT-target 梯度有 4 臂（groundtruth/random/losers/winners），崩坏随 target 越 off-manifold 越重——但那是
    4 个离散「类别」点、off-manifold 度未量化。本脚本把 x 轴量化为「各 target 类型音频的均值 L_FM」，对崩坏
    （eval WER）做单调性/剂量-反应检验 → 若 target L_FM↑ ⇒ 崩坏↑，即「越 off-manifold 的目标越毒」的因果证据。

做什么（大部分纯分析、复用 E1 产物）:
    winners/losers/random 三类 target 的 L_FM 直接从 E1 的 percand_lfm.csv 抽（winner=每 prompt max-reward 候选、
    loser=min-reward、random=随机一条）；groundtruth(真录音) 的 L_FM 由 probe_fm_reward.py --score-audio 另算（可选第 4 点）。
    各臂崩坏 WER 读 results/out/{arm}.summary.json。

判据（回填 §5.1）:
    Spearman(target 均值 L_FM, 臂 WER) > 0 且按 L_FM 升序时 WER 单调不降 ⇒ 剂量-反应成立 → claim 4 升为因果桥；
    否则诚实收回「monotone」叙事。

用法: python eval/dose_response.py --percand alignment/out/percand_lfm.csv \
        [--groundtruth-lfm alignment/out/groundtruth_lfm.csv] --out results/out/dose_report.txt
纯 CPU、秒级（groundtruth L_FM 若需另算则一次 GPU 前向，由 run_dose.sh 代办）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _spearman(x, y):
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        for p, i in enumerate(o):
            r[i] = float(p)
        return r
    rx, ry, n = rank(x), rank(y), len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((v - mx) ** 2 for v in rx) ** 0.5
    dy = sum((v - my) ** 2 for v in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="★1 dose-response: target off-manifold 度 → 崩坏")
    ap.add_argument("--percand", default="alignment/out/percand_lfm.csv")
    ap.add_argument("--groundtruth-lfm", default=None,
                    help="probe_fm_reward.py --score-audio 产出的 {fid,lfm}（可选第 4 点：真录音）")
    ap.add_argument("--metrics-dir", default=str(PATHS.results_dir))
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    # 1) 各 target 类型的均值 L_FM（winners/losers/random 从 percand 免费抽）
    pool: dict[str, list[tuple]] = defaultdict(list)
    for r in csv.DictReader(open(args.percand, encoding="utf-8")):
        pool[r["fid"]].append((float(r["reward"]), float(r["lfm"])))
    rng = random.Random(args.seed)
    win, los, ran = [], [], []
    for fid, c in pool.items():
        if len(c) < 2:
            continue
        win.append(max(c, key=lambda x: x[0])[1])
        los.append(min(c, key=lambda x: x[0])[1])
        ran.append(rng.choice(c)[1])

    gt_lfm = None
    if args.groundtruth_lfm and Path(args.groundtruth_lfm).exists():
        gt = [float(r["lfm"]) for r in csv.DictReader(open(args.groundtruth_lfm, encoding="utf-8"))]
        gt_lfm = _mean(gt)

    # tag -> (展示名, target 均值 L_FM)
    targets = {
        "sft_groundtruth_nfe10": ("groundtruth(real)", gt_lfm),
        "sft_random_nfe10": ("random", _mean(ran)),
        "sft_losers_nfe10": ("losers", _mean(los)),
        "sft_nfe10": ("winners", _mean(win)),
    }

    emit("==== [★1 dose-response] target off-manifold 度(均值 L_FM) → 崩坏(eval WER) ====\n")
    pts = []
    for tag, (name, tl) in targets.items():
        p = Path(args.metrics_dir) / f"{tag}.summary.json"
        if not p.exists():
            emit(f"  ⚠️ 缺 {p}，跳过 {name}")
            continue
        wer = float(json.loads(p.read_text(encoding="utf-8"))["wer"])
        if tl is None or tl != tl:
            emit(f"  {name:18s} target L_FM=（缺；groundtruth 需 --groundtruth-lfm）   eval WER={wer:.3f}")
            continue
        pts.append((tl, wer, name))
        emit(f"  {name:18s} target 均值 L_FM={tl:.4f}   eval WER={wer:.3f}")

    if len(pts) >= 3:
        sp = _spearman([a for a, _, _ in pts], [b for _, b, _ in pts])
        spts = sorted(pts)  # 按 target L_FM 升序
        mono = all(spts[i][1] <= spts[i + 1][1] for i in range(len(spts) - 1))
        emit(f"\n  Spearman(target L_FM, WER) over {len(pts)} 点 = {sp:+.3f}")
        emit("  按 target L_FM 升序的 WER：" + "  ≤  ".join(f"{b:.3f}({n})" for _, b, n in spts))
        if mono and sp > 0:
            emit("  ✅ 单调上升 ⇒ 剂量-反应成立：越 off-manifold 的 target → 崩坏越重（claim 4 从相关升为因果桥）。")
        else:
            emit("  ⚠️ 非单调 / 负 ⇒ 诚实收回「monotone in off-manifold-ness」，dose-response 未成立、只是 4 点序关系。")
    else:
        emit("\n  可用点 <3（多半缺 groundtruth）→ 先跑 probe_fm_reward.py --score-audio 补 groundtruth L_FM（run_dose.sh 会代办）。")

    emit("\n==== 落表：单调成立 → §5.1 把『monotone in off-manifold-ness』从 4 类别点升为量化剂量-反应；否则收窄措辞。 ====")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[--out] 报告已归档 → {out}")


if __name__ == "__main__":
    main()
