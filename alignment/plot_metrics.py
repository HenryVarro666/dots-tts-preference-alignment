"""
plot_metrics.py — 离线把训练曲线画成 PNG（免 TensorBoard / 免端口转发）

做什么:    读某个 run 的 metrics.jsonl，画 dpo/sft 损失 + reward_acc/margin + fm_w/fm_l 三联图，存 PNG。
怎么做:    纯 matplotlib；远程服务器跑完后 scp 这张 PNG 到本地看即可。
为什么:    TensorBoard 要 SSH 端口转发才能在本地浏览器看；PNG 更省事，适合「跑完瞄一眼趋势」。
对实验意义: 直观确认 DPO 是否在学——dpo↓ / reward_acc→1 / margin↑ / fm_w<fm_l。

前置条件: 该 run 的 output_dir 下有 metrics.jsonl（训练自动写）。
预期结果: <run>/curve.png。
⚠️ 本机只语法检查。
用法: python alignment/plot_metrics.py --run alignment/out/dpo_multiobj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(run: Path) -> list[dict]:
    p = run / "metrics.jsonl"
    if not p.exists():
        raise SystemExit(f"缺 {p}（训练时自动写；确认 run 路径对）")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="画训练曲线 PNG")
    ap.add_argument("--run", required=True, help="训练 output_dir（含 metrics.jsonl）")
    ap.add_argument("--out", default=None, help="输出 PNG（默认 <run>/curve.png）")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    run = Path(args.run)
    rows = _load(run)
    steps = [r["step"] for r in rows]

    def col(name):
        return [r.get(name) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # ① 损失：dpo（应↓）+ sft
    axes[0].plot(steps, col("dpo"), label="dpo (偏好损失↓)")
    if any(v for v in col("sft")):
        axes[0].plot(steps, col("sft"), label="sft (anchor)", alpha=0.6)
    axes[0].set_title("Loss"); axes[0].set_xlabel("step"); axes[0].legend()
    # ② DPO 健康信号：reward_acc（→1）+ margin（↑）
    axes[1].plot(steps, col("reward_acc"), label="reward_acc →1", color="C2")
    ax1b = axes[1].twinx()
    ax1b.plot(steps, col("reward_margin"), label="reward_margin ↑", color="C3", alpha=0.6)
    axes[1].set_title("DPO 学习信号"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("reward_acc"); ax1b.set_ylabel("reward_margin")
    # ③ FM 损失：winner 应低于 loser
    axes[2].plot(steps, col("fm_w"), label="fm_w (winner)")
    axes[2].plot(steps, col("fm_l"), label="fm_l (loser)")
    axes[2].set_title("FM 损失：winner<loser 才对"); axes[2].set_xlabel("step"); axes[2].legend()

    fig.tight_layout()
    out = Path(args.out) if args.out else run / "curve.png"
    fig.savefig(out, dpi=140)
    print(f"✓ 曲线 -> {out}")
    print("  判读：dpo↓、reward_acc→1、reward_margin↑、fm_w<fm_l 即 DPO 在正确地学偏好。")


if __name__ == "__main__":
    main()
