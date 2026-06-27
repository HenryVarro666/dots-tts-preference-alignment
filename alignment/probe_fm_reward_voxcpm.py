"""
probe_fm_reward_voxcpm.py — 把 FM-reward 反相关探针搬到 **VoxCPM**：做架构级泛化(解 EIC CRITICAL #3)

背景(为什么写它):
    `probe_fm_reward.py` 在 **dots.tts**(flow-matching 连续-latent)上发现：去噪后的 FM-margin 与
    perceptual reward **反相关**(Pearson≈−0.41)。审稿 EIC CRITICAL #3 指出：**N=1 个模型不能把结论
    泛化到「整类 continuous-latent TTS」**。本脚本在**第二个、独立实现的**连续-latent TTS——
    **VoxCPM(diffusion-AR)**——上重跑同一探针，看反相关是否复现。

做什么(与 probe_fm_reward.py 完全同构):
    取 N 个 pair，对 winner/loser 各算 **K 次 MC(不同 (z,t)) 平均**的 diffusion velocity-MSE
    (VoxCPM 的 `loss/diff`，= dots.tts `fm_loss` 的同构量)；每次 draw 内 win/lose **共享同一 (z,t)**。
    据此判定:
      (A) 去噪后 winner 是否**稳定**低于 loser(margin = L^lose − L^win 的均值是否可靠为正)；
      (B) 去噪 FM-margin 是否与 reward 差**(反)相关**(Pearson/Spearman)。

怎么做(含命令+输入输出):
    复用 `voxcpm_preprocess.VoxCPMPreprocessor`(对接 VoxCPM 训练侧 packer + forward) + fork_rng 机制；
    no_grad 纯测量、不训练。输入 = dots 阶段产出的 pairs(同一批偏好对，喂进 VoxCPM 编码器算它的 FM 损失)；
    输出 = 逐对 CSV(列与 `probe_per_pair.csv` **完全一致**) → 直接复用现成 `probe_analysis.py` 分析。
      python alignment/probe_fm_reward_voxcpm.py \
          --ckpt openbmb/VoxCPM2 \
          --pairs pairs/out/pairs_best_worst.jsonl --n 24 --k 24 \
          --out-csv alignment/out/probe_per_pair_voxcpm.csv
      # 然后(复用同一分析脚本)：
      python alignment/probe_analysis.py --csv alignment/out/probe_per_pair_voxcpm.csv \
          --index candidates/out/candidates_index.jsonl --rewards reward/out/rewards.jsonl

为什么这么设计(命门同样守):
    单次 L 高方差 → 必须 MC 平均才看得出真实信号；每 draw 内 win/lose 共享 (z,t) → 抵消公共噪声、
    直接比较二者对同一噪声的去噪难度(与 DPO 训练同款可比性)。在 VoxCPM 上，(z,t) 的采样发生在
    `UnifiedCFM.compute_loss` 内部，故用 fork_rng+manual_seed 包住 `forward` 来固定，win/lose 用同一
    seed → 共享随机流。

对 claim 的意义(裁决「架构级 vs dots.tts 特性」):
    • 复现(margin 不稳定为正 + 与 reward 反相关) → 结论升到 **continuous-latent TTS 架构级**：
      两个独立实现(dots=flow-matching、VoxCPM=diffusion)都呈现 FM-as-reward 的反相关 → 解 EIC #3。
    • 不复现 → 把结论**诚实收窄**为 dots.tts 的特性(写入 §6/§7：泛化边界)。
    判读细节(双系数 CI / 偏相关 / per-component / 高-gap 子集存活)交给 `probe_analysis.py` 复跑。

⚠️ 本机(macOS,无 GPU,未下载权重)只语法检查；真正运行在 GPU 机器上(有 pairs + VoxCPM2 权重)。
   依赖 voxcpm_preprocess 的全部〔需核实〕(A–G：训练侧接口可跑性 / tokenizer 约定 / 编码 SR /
   dit_mean_mode / loss-diff 语义 / 架构版本 / cfg dropout)，详见该文件头注释。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl  # noqa: E402

import torch  # noqa: E402

from voxcpm_preprocess import VoxCPMPreprocessor, diff_loss_per_sample  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _diff_with_seed(pre: VoxCPMPreprocessor, batch: dict, seed: int, dev) -> float:
    """在指定 seed 下前向，取 per-sample diffusion velocity-MSE(batch=1 → 标量)。

    fork_rng 隔离全局 RNG；manual_seed(seed) 固定 `compute_loss` 内部采样的 (z,t)(以及 cfg_mask)。
    win/lose 传入**同一 seed** → 共享随机流 = 命门。与 probe_fm_reward._fm_with_seed 一一对应。
    """
    devs = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            outputs = pre.prepare_and_forward(batch)
    return float(diff_loss_per_sample(outputs).mean().item())


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


def _selfcheck(pre: VoxCPMPreprocessor, batch: dict, seed0: int, dev) -> None:
    """X10 命门自检（跑全量前的 gate）：坏接缝不该静默出一张"看起来正常"的相关性表。
    核三件事：① **L_FM 有限正值**（接缝 E/A/权重/config 没崩，velocity-MSE 应 >0）；
    ② **同 seed 决定论**（win/lose 共享 (z,t)+cfg_mask 的前提，接缝 G）；
    ③ **异 seed 真有变化**（(z,t) 确由 seed 驱动，K-MC 才有意义）。
    判据：异 seed 变化量 var 必须 ≫ 同 seed 残差 det——一次抓住"非决定论"与"RNG 被冻住"两种坏法。"""
    import math
    a1 = _diff_with_seed(pre, batch, seed0, dev)
    a2 = _diff_with_seed(pre, batch, seed0, dev)        # 同 seed、同输入 → 理想下逐位相同
    b = _diff_with_seed(pre, batch, seed0 + 1, dev)     # 仅换 seed → 应有可观变化
    det, var = abs(a1 - a2), abs(a1 - b)
    if not all(math.isfinite(x) for x in (a1, a2, b)):
        raise SystemExit(f"[selfcheck] ✗ L_FM 非有限值 (a1={a1}, b={b}) — 接缝 E/A/权重/config 可疑")
    if not a1 > 0:
        raise SystemExit(f"[selfcheck] ✗ L_FM 非正 ({a1:.6g}) — velocity-MSE 应 >0，疑 loss/diff 取错量（接缝 E）")
    if var <= max(1e-4, 5 * det):
        raise SystemExit(
            f"[selfcheck] ✗ RNG 失效：异 seed 变化 {var:.6f} ≤ 5×同 seed 残差 {det:.6f}\n"
            f"  → 要么同 seed 不决定论（接缝 G：cfg_mask/z/t 未被 manual_seed 固定），"
            f"要么 (z,t) 没真正随 seed 变（MC 无意义）。")
    print(f"[selfcheck] ✓ L_FM 有限正值({a1:.5f}) | 同 seed 残差 {det:.2e}(应≈0) | 异 seed 变化 {var:.4f}(应≫残差)")


def main() -> None:
    ap = argparse.ArgumentParser(description="VoxCPM 探针：diffusion FM 损失是否为有效隐式奖励(架构级泛化)")
    ap.add_argument("--ckpt", default="openbmb/VoxCPM2")
    ap.add_argument("--pairs", default="pairs/out/pairs_best_worst.jsonl")
    ap.add_argument("--n", type=int, default=24, help="抽多少个 pair")
    ap.add_argument("--k", type=int, default=24, help="每个 pair 的 MC 采样次数")
    ap.add_argument("--seed0", type=int, default=20260625)
    ap.add_argument("--out-csv", default="alignment/out/probe_per_pair_voxcpm.csv",
                    help="逐对结果落盘(列同 probe_per_pair.csv → 直接喂 probe_analysis.py)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="只跑命门自检(决定论/RNG有效/L_FM有限正值)在第一对上并退出，不跑全量——冒烟/gate 用")
    args = ap.parse_args()

    dev = _device()
    print(f"[VoxCPM 探针] device={dev}  ckpt={args.ckpt}  pairs={args.pairs}  N={args.n}  K={args.k}")
    pre = VoxCPMPreprocessor.from_pretrained(args.ckpt, device=str(dev))

    rows = list(read_jsonl(Path(args.pairs)))[: args.n]
    if not rows:
        raise SystemExit(f"pairs 空: {args.pairs}")

    # ---- 命门自检：每次跑全量前都 gate 一遍（仅 3 次前向，开销可忽略）；--selfcheck 则只自检后退出 ----
    r0 = rows[0]
    wb0 = _to_device(pre.collate([pre.process(str(r0["fid"]), r0["text"], r0["win_audio"])]), dev)
    _selfcheck(pre, wb0, args.seed0, dev)
    if args.selfcheck:
        print("[selfcheck] --selfcheck 指定：自检通过、退出（未跑全量）。")
        return

    reward_gaps: list[float] = []      # reward_win − reward_lose(外部奖励差)
    mc_margins: list[float] = []       # K-MC 平均的 L^lose − L^win(去噪 FM margin)
    single_margins: list[float] = []   # 单次(k=0) 的 FM margin(量化噪声)
    win_rates: list[float] = []        # K 次中 winner 更优(L^win<L^lose) 的占比
    rows_out: list[dict] = []          # 逐对持久化(headline 可独立重算 + 供 probe_analysis)

    for idx, r in enumerate(rows):
        fid = str(r["fid"])
        text = r["text"]
        # 用同一批 dots-生成的偏好音频喂进 VoxCPM 编码器算它的 FM 损失 = 架构级横向对照。
        win_batch = _to_device(pre.collate([pre.process(fid, text, r["win_audio"])]), dev)
        lose_batch = _to_device(pre.collate([pre.process(fid, text, r["lose_audio"])]), dev)

        per_draw = []
        for k in range(args.k):
            seed = args.seed0 + k  # 每 draw 不同 (z,t)；win/lose 用同 seed → 共享噪声(命门)
            fm_w = _diff_with_seed(pre, win_batch, seed, dev)
            fm_l = _diff_with_seed(pre, lose_batch, seed, dev)
            per_draw.append(fm_l - fm_w)        # >0 ⇒ winner 的 L 更低(更优)

        mc = sum(per_draw) / len(per_draw)
        reward_gaps.append(float(r["reward_win"]) - float(r["reward_lose"]))
        mc_margins.append(mc)
        single_margins.append(per_draw[0])
        win_rates.append(sum(1 for d in per_draw if d > 0) / len(per_draw))
        # 列名与 probe_per_pair.csv 严格一致(fid/win_audio/lose_audio/reward_win/reward_lose/
        # reward_gap/single_margin/mc_margin/win_rate/k) → probe_analysis.py 无需改动即可复用。
        rows_out.append({"fid": fid, "win_audio": r["win_audio"], "lose_audio": r["lose_audio"],
                         "reward_win": float(r["reward_win"]), "reward_lose": float(r["reward_lose"]),
                         "reward_gap": reward_gaps[-1], "single_margin": per_draw[0],
                         "mc_margin": mc, "win_rate": win_rates[-1], "k": args.k})
        print(f"  [{idx:2d}] reward_gap={reward_gaps[-1]:+.3f}  "
              f"FM_margin: single={per_draw[0]:+.4f}  MC(K={args.k})={mc:+.4f}  "
              f"win_rate={win_rates[-1]:.2f}")

    # 逐对结果落盘：列与 dots 版完全一致，使 ρ 可一键独立重算并直接复用 probe_analysis.py。
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n✓ 逐对结果已存 -> {out_csv}  ({len(rows_out)} 行；列同 probe_per_pair.csv，供 probe_analysis.py)")

    n = len(mc_margins)
    pos = sum(1 for m in mc_margins if m > 0)
    mean_mc = sum(mc_margins) / n
    mean_single = sum(single_margins) / n
    var_single = sum((s - mean_single) ** 2 for s in single_margins) / max(n - 1, 1)
    pear = _pearson(mc_margins, reward_gaps)
    spear = _spearman(mc_margins, reward_gaps)

    print("\n========== 汇总 (VoxCPM / diffusion velocity-MSE) ==========")
    print(f"(A) 去噪 FM margin 为正的 pair: {pos}/{n} ({100*pos/n:.0f}%)   均值={mean_mc:+.4f}")
    print(f"    单次 margin 标准差(噪声量级)={var_single**0.5:.4f}  ← 远大于 MC 均值则=高噪")
    print(f"    平均 win_rate(K 次中 winner 更优占比)={sum(win_rates)/n:.2f}")
    print(f"(B) 去噪 FM margin vs reward 差: Pearson={pear:+.3f}  Spearman={spear:+.3f}")
    print("\n=== 架构级泛化判定(对照 dots.tts 的 Pearson≈−0.41) ===")
    # 反相关 = margin 与 reward_gap 负相关(reward 越高的 winner，FM 反而没更低)。
    anti_correlated = (pear == pear) and pear < -0.2  # pear==pear 排除 NaN
    if anti_correlated:
        print("▶ 反相关在 VoxCPM 上**复现** → FM-as-reward 的反相关是 **continuous-latent TTS 架构级**现象")
        print("  (dots=flow-matching、VoxCPM=diffusion，两套独立实现一致) → 解 EIC CRITICAL #3(N=1)。")
    elif (pear == pear) and pear > 0.2:
        print("▶ VoxCPM 上呈**正相关** → 反相关是 dots.tts 的实现特性，不跨架构 → 结论须诚实收窄。")
    else:
        print("▶ VoxCPM 上**近零相关** → FM 损失在 VoxCPM 上不排序 perceptual 质量(既不正也不反)；")
        print("  反相关未跨架构复现，结论收窄为 dots.tts 特性(仍诚实；写入泛化边界 §6/§7)。")
    print("\n→ 细化统计(双系数 95%CI / 偏相关裁 intrinsic / per-component / 高-gap 子集存活)：")
    print(f"  python alignment/probe_analysis.py --csv {out_csv} "
          f"--index candidates/out/candidates_index.jsonl --rewards reward/out/rewards.jsonl")


if __name__ == "__main__":
    main()
