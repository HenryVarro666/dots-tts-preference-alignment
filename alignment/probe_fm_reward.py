"""
probe_fm_reward.py — 决定性探针：FM 损失到底是不是「有效的隐式奖励」？

背景（为什么写它）:
    inspect_gap.py 证明 pairs_best_worst(multiobj) 的 **reward 差充足**（0% 近似平局，中位数 1.39），
    但 DPO 训练时 **FM 损失差**(reward_margin) 平均只有 ~0.01、偶尔才炸到 0.5–0.8。
    → 矛盾点收敛到一个问题：模型的 flow-matching 损失 L_FM 到底跟不跟 perceptual reward(WER/SIM/UTMOS) 走？
    单次 (x0,t) 采样的 L_FM 方差很大，单看一次分不清「噪声淹没」还是「FM 根本不跟 reward」。

做什么:
    取 N 个 pair，对 winner / loser 各算 **K 次 MC（不同 (x0,t)）平均**的 L_FM；每次 draw 内 win/lose
    共享同一 (x0,t)（与 DPO 训练同款可比性）。据此判定两件事：
      (A) 去噪后 winner 是否**稳定**低于 loser（margin = L_FM^lose − L_FM^win 的均值是否可靠为正）；
      (B) 去噪后的 FM-margin 是否与 reward 差**正相关**（Pearson/Spearman）。
怎么做:
    复用 check_fm_loss 的 DotsPreprocessor + fork_rng 机制；no_grad 纯测量、不训练。
    python alignment/probe_fm_reward.py --pairs pairs/out/pairs_best_worst.jsonl --n 24 --k 24
为什么这么设计:
    单次 L_FM 高方差 → 必须 MC 平均才能看出「真实」信号。
    每 draw 内 win/lose 共享 (x0,t) → 抵消公共噪声，直接比较二者对同一噪声的重建难度。
对实验意义（决定下一步走向）:
    • 若 (A) 多数 pair margin 稳定为正 且 (B) 正相关 → **L_FM 是有效但高噪的隐式奖励**：
        训练学不动是**方差问题** → 修法 = 训练时对 L_FM 做 MC 平均(每步多采样) / 增大 lr / 加步数。可修。
    • 若 (A) margin≈0 或 (B) 相关≈0 → **L_FM 不跟 perceptual reward**：DPO-FM 方法本身有局限，
        这是论文级的负面/机理发现（写入 §6 稳定性 / §7 Limitations / §3.2 方法假设的边界）。
本机只语法检查；真正运行在 GPU 机器上（有 pairs + 权重）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable, read_jsonl  # noqa: E402

import torch  # noqa: E402

from dots_preprocess import DotsPreprocessor  # noqa: E402
from losses_dpo import fm_loss_per_sample  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _fm_with_seed(pre: DotsPreprocessor, batch: dict, seed: int, dev) -> float:
    """在指定 seed 下前向，取 per-sample L_FM（batch=1 → 标量）。fork_rng 隔离全局状态。"""
    devs = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            loss_terms = pre.prepare_and_forward(batch)
    return float(fm_loss_per_sample(loss_terms["fm_loss"]).mean().item())


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return _pearson(rank(xs), rank(ys))


def _load_candidate_pool(rewards_path: Path, candidates_path: Path) -> dict:
    """构造每 prompt 的候选池，供 E1 选择伪影 null control。
    关键: **audio 必须取 rewards 文件自带的那个**——settings-matched 用 rewards_fixed.jsonl 时它指向
    cand_fixed_*.wav（正确的 matched 音频）；若错用 candidates_index(grid) 的 audio 会算到错误的 grid 候选
    上（曾导致 ρ_real=−0.103 ≠ headline −0.216 的自洽性失败）。text 是 per-prompt，从 candidates_index
    建 fid→text 映射（与候选级 k 对齐无关，稳健）。"""
    text_by_fid: dict[str, str] = {}
    for c in read_jsonl(candidates_path):
        text_by_fid.setdefault(c["fid"], c.get("text", ""))
    by_fid: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(rewards_path):
        by_fid[r["fid"]].append({"k": r["k"], "reward": float(r["reward"]),
                                 "audio": r.get("audio"),
                                 "text": r.get("text") or text_by_fid.get(r["fid"], "")})
    return by_fid


def _run_emit_per_candidate(pre: DotsPreprocessor, dev, args) -> None:
    """E1（选择伪影 null control 的输入端）：对每 prompt 的**全部候选**算 K-MC 去噪 L_FM，
    落 percand_lfm.csv（fid,k,reward,lfm,audio）。供 probe_analysis.py --selection-null。
    无需新合成——只对现有候选池做 no_grad 前向（与配对探针同款 fork_rng + 共享 seed 序列）。"""
    pool = _load_candidate_pool(Path(args.rewards), Path(args.candidates))
    fids = list(pool.keys())
    if args.n and args.n > 0:
        fids = fids[: args.n]
    rows_out: list[dict] = []
    for fi, fid in enumerate(fids):
        for c in pool[fid]:
            if not c["audio"]:
                continue
            batch = _to_device(pre.collate([pre.process(fid, c["text"], c["audio"])]), dev)
            draws = [_fm_with_seed(pre, batch, args.seed0 + k, dev) for k in range(args.k)]
            rows_out.append({"fid": fid, "k": c["k"], "reward": c["reward"],
                             "lfm": sum(draws) / len(draws), "audio": c["audio"]})
        print(f"  [{fi:3d}] fid={fid}  cands={len(pool[fid])}")
    out = Path(args.out_percand)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fid", "k", "reward", "lfm", "audio"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n✓ per-candidate L_FM -> {out}  "
          f"({len(rows_out)} candidates over {len(fids)} prompts; 供 probe_analysis.py --selection-null)")


def main() -> None:
    ap = argparse.ArgumentParser(description="探针：FM 损失是否为有效隐式奖励")
    ap.add_argument("--ckpt", default="rednote-hilab/dots.tts-base")
    ap.add_argument("--pairs", default="pairs/out/pairs_best_worst.jsonl")
    ap.add_argument("--n", type=int, default=24, help="抽多少个 pair")
    ap.add_argument("--k", type=int, default=24, help="每个 pair 的 MC 采样次数")
    ap.add_argument("--seed0", type=int, default=20260625)
    ap.add_argument("--out-csv", default="alignment/out/probe_per_pair.csv",
                    help="逐对结果落盘（解 reviewer 的『headline 不可独立重算』+ 供 probe_analysis.py）")
    ap.add_argument("--emit-per-candidate", action="store_true",
                    help="E1 选择伪影 null control：对每 prompt 的全部候选算 L_FM、落 --out-percand（不跑配对探针）")
    ap.add_argument("--candidates", default="candidates/out/candidates_index.jsonl",
                    help="--emit-per-candidate 用：候选索引（fid,k,audio,text）")
    ap.add_argument("--rewards", default="reward/out/rewards.jsonl",
                    help="--emit-per-candidate 用：候选奖励（settings-matched 用 reward/out/rewards_fixed.jsonl）")
    ap.add_argument("--out-percand", default="alignment/out/percand_lfm.csv")
    args = ap.parse_args()

    dev = _device()
    print(f"device={dev}  pairs={args.pairs}  N={args.n}  K={args.k}")
    pre = DotsPreprocessor.from_pretrained(args.ckpt)
    pre.model.to(dev)

    if args.emit_per_candidate:
        _run_emit_per_candidate(pre, dev, args)
        return

    rows = list(read_jsonl(Path(args.pairs)))[: args.n]
    if not rows:
        raise SystemExit(f"pairs 空: {args.pairs}")

    reward_gaps: list[float] = []      # reward_win − reward_lose（外部奖励差）
    mc_margins: list[float] = []       # K-MC 平均的 L_FM^lose − L_FM^win（去噪 FM margin）
    single_margins: list[float] = []   # 单次(k=0) 的 FM margin（量化噪声）
    win_rates: list[float] = []        # K 次中 winner 更优(L_FM^win<L_FM^lose) 的占比
    rows_out: list[dict] = []          # 逐对持久化（headline 可独立重算 + 供 probe_analysis）

    for idx, r in enumerate(rows):
        fid = str(r["fid"])
        text = r["text"]
        win_batch = _to_device(pre.collate([pre.process(fid, text, r["win_audio"])]), dev)
        lose_batch = _to_device(pre.collate([pre.process(fid, text, r["lose_audio"])]), dev)

        per_draw = []
        for k in range(args.k):
            seed = args.seed0 + k  # 每 draw 不同 (x0,t)；win/lose 用同 seed → 共享噪声
            fm_w = _fm_with_seed(pre, win_batch, seed, dev)
            fm_l = _fm_with_seed(pre, lose_batch, seed, dev)
            per_draw.append(fm_l - fm_w)        # >0 ⇒ winner 的 L_FM 更低(更优)

        mc = sum(per_draw) / len(per_draw)
        reward_gaps.append(float(r["reward_win"]) - float(r["reward_lose"]))
        mc_margins.append(mc)
        single_margins.append(per_draw[0])
        win_rates.append(sum(1 for d in per_draw if d > 0) / len(per_draw))
        rows_out.append({"fid": fid, "win_audio": r["win_audio"], "lose_audio": r["lose_audio"],
                         "reward_win": float(r["reward_win"]), "reward_lose": float(r["reward_lose"]),
                         "reward_gap": reward_gaps[-1], "single_margin": per_draw[0],
                         "mc_margin": mc, "win_rate": win_rates[-1], "k": args.k,
                         # B6 钩子：完整 K 个 draw 的 margin list，供 probe_analysis.py 做 split-half/解衰减；新增尾列、不破坏现有读者
                         "per_draw_json": json.dumps(per_draw)})
        print(f"  [{idx:2d}] reward_gap={reward_gaps[-1]:+.3f}  "
              f"FM_margin: single={per_draw[0]:+.4f}  MC(K={args.k})={mc:+.4f}  "
              f"win_rate={win_rates[-1]:.2f}")

    # 逐对结果落盘：使 ρ 可一键独立重算（解 R2 的 reproducibility CRITICAL），并供 probe_analysis.py
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n✓ 逐对结果已存 -> {out_csv}  ({len(rows_out)} 行；供 probe_analysis.py)")

    n = len(mc_margins)
    pos = sum(1 for m in mc_margins if m > 0)
    mean_mc = sum(mc_margins) / n
    # 噪声量级：单次 margin 的样本标准差
    mean_single = sum(single_margins) / n
    var_single = sum((s - mean_single) ** 2 for s in single_margins) / max(n - 1, 1)
    pear = _pearson(mc_margins, reward_gaps)
    spear = _spearman(mc_margins, reward_gaps)

    print("\n========== 汇总 ==========")
    print(f"(A) 去噪 FM margin 为正的 pair: {pos}/{n} ({100*pos/n:.0f}%)   均值={mean_mc:+.4f}")
    print(f"    单次 margin 标准差(噪声量级)={var_single**0.5:.4f}  ← 远大于 MC 均值则=高噪")
    print(f"    平均 win_rate(K 次中 winner 更优占比)={sum(win_rates)/n:.2f}")
    print(f"(B) 去噪 FM margin vs reward 差: Pearson={pear:+.3f}  Spearman={spear:+.3f}")
    print("\n=== 判定 ===")
    fixable = (pos / n >= 0.7) and (mean_mc > 0)
    correlated = (not pear != pear) and pear > 0.2  # pear==pear 排除 NaN
    if fixable and correlated:
        print("▶ L_FM 是【有效但高噪】的隐式奖励：winner 去噪后稳定更优 + 与 reward 正相关。")
        print("  训练学不动 = **方差问题**。修法：训练时对 L_FM 做 MC 平均(每步多采 x0,t)/增大 lr/加步数。")
    elif fixable and not correlated:
        print("▶ winner 去噪后稳定更优，但与 reward 差**不相关**：L_FM 只反映重建难度、不排序 perceptual 质量。")
        print("  → DPO-FM 能学到「偏好 winner」但未必对齐 WER/SIM/UTMOS；需在 §6 讨论这一错配。")
    else:
        print("▶ 去噪后 winner 也不稳定更优(或均值≈0)：L_FM 不是有效隐式奖励——方法本身的边界。")
        print("  → 论文级负面发现，写入 §3.2 假设边界 / §6 稳定性 / §7 Limitations；考虑改用别的隐式似然代理。")


if __name__ == "__main__":
    main()
