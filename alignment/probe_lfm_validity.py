# SPDX-License-Identifier: Apache-2.0
"""
probe_lfm_validity.py — Part C / X14：L_FM「似然有效性」的**独立锚**（关掉 C2 / OPEN-SOAR-REFRAME）

背景（为什么写它 = 动机）:
    第一轮把「L_FM 是有效似然代理」的证据搭在 **open-SOAR 训到饱和** 上。审稿 C2 / 权威措辞
    OPEN-SOAR-REFRAME 指出：open-SOAR by construction 是「干净真实音频 vs 同一音频加噪」的偏好——
    loser 永远 off-manifold、winner 永远 on-manifold，二者**易分**，故 DPO-FM 训到满分只说明
    「机制可用 + optimizer 正常」，**不**独立证明 L_FM 忠实地排序了「典型性 / 似然」。
    本脚本提供一条**不依赖 open-SOAR、不依赖任何外部 reward** 的独立证据：直接问——
        在模型自身的 flow-matching 损失下，**真实 held-out 录音** 是否系统性比 **reward-winner 候选**
        更「典型」（L_FM 更低）？
    若是 → L_FM 把真实 on-manifold 音频判为更可能、把 reward 选出的自生成候选判为更 off-manifold，
            这与「忠实似然」的方向一致，且**完全独立于 open-SOAR 饱和** → 关掉 C2。
    （保留 OPEN-SOAR-REFRAME 的 caveat：L_FM 仍是 **似然启发代理、非忠实 NLL 上界**；本探针给方向证据、
      不给精确 −log p。真正的精确 NLL 需子协议 (a) ODE-NLL，而该接口在 dots.tts runtime 是否暴露存疑。）

做什么:
    子协议 (b) typicality（首选、无需任何新接口）——读「真实 held-out 录音 manifest」与
    「reward-winner 候选清单」，用**同一** DotsPreprocessor + fm_loss_per_sample 算 per-sample L_FM，
    按 fid **配对**比较：diff = L_FM(winner) − L_FM(real)。若 diff 系统性 > 0（winner 损失更高=更不典型），
    则示「reward-winner 在模型自身似然下 off-manifold」。出配对统计：均值差 + bootstrap CI + Wilcoxon
    符号秩检验 + diff>0 占比。
    子协议 (a) ODE-NLL（可选、`--ode-nll` 开）——若 dots.tts runtime 暴露概率流 ODE / 精确 −log p，
    则算真实录音与 winner 的精确 NLL 做同款配对比较；接口不确定处用 〔需核实〕 显式标注，不暴露则只跑 (b)。

怎么做（复用 probe_fm_reward 骨架 + 探针铁律）:
    复用 `DotsPreprocessor`(process/collate/prepare_and_forward) + `fm_loss_per_sample`；no_grad 纯测量、不训练。
    **探针铁律不变**：每个 MC draw 内，real 与 winner 共享同一 (x0,t)——用 `torch.random.fork_rng` +
    同一 `manual_seed` 包住前向（与 probe_fm_reward 的 win/lose 共享 seed 完全同款），抵消公共噪声、
    使两条 L_FM 直接可比。真实录音的 (text,audio) 取法与 sft_groundtruth(X4) 一致：
    `text = r["text"]`、`audio = r.get("prompt_audio") or r.get("audio")`（见 data_pairs.build_single_dataloader）。
        python alignment/probe_lfm_validity.py \
            --real-manifest data/prompts/dots_manifest_train.jsonl \
            --winners pairs/out/winners_only.jsonl --n 188 --k 16 \
            --out-csv alignment/out/probe_lfm_validity.csv

为什么这么设计:
    单次 (x0,t) 的 L_FM 高方差 → 必须 K 次 MC 平均才看得出真实信号；real/winner 共享 (x0,t) → 抵消公共
    噪声、直接比较二者对同一噪声的去噪难度。按 fid 配对 → 控住 text 与说话人（同一 prompt 的真实录音 vs
    其 reward-winner 生成），是比无配对更强、更省样本的对照。

对实验意义（裁决 C2 / 守 OPEN-SOAR-REFRAME 的可证伪线）:
    • 若真实录音 L_FM **系统性低于** winner（diff 均值>0、CI 排除 0、Wilcoxon 显著）→ L_FM 把真实
      on-manifold 音频判为更典型、reward-winner 判为更 off-manifold → 给「L_FM 是似然启发代理」一条
      **不依赖 open-SOAR** 的独立支撑，关掉 C2；并与「结构化显著反相关本身」(Spearman −0.187 /
      partial −0.392 + per-component 三轴一致) 互证 L_FM 非退化(H2 被否)。
    • 若真实录音 L_FM **不低于** winner（diff≤0 或 CI 含 0）→ 按可证伪线 **收回「L_FM 有效似然代理」**
      的表述：模型在自身损失下并不偏好真实录音，则不能把 L_FM 当似然方向标尺，§方法假设须降级。
    无论结论，保留「L_FM 非忠实 NLL 上界」caveat；精确 NLL 锚见子协议 (a)〔需核实〕。

⚠️ 本机（macOS / 无 GPU / 权重未下载）只 `python3 -m py_compile` 语法检查；**真实运行在 GPU 机器上**
   （需 dots.tts-base 权重 + 真实录音 manifest + winners 清单）。本文件**未在本机运行过**。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from math import erf, sqrt
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
    """在指定 seed 下前向，取 per-sample L_FM（batch=1 → 标量）。fork_rng 隔离全局状态。

    与 probe_fm_reward._fm_with_seed 完全同款——守探针铁律：real 与 winner 在同一 draw 用同一 seed →
    共享 (x0,t)，使两条 L_FM 直接可比。
    """
    devs = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            loss_terms = pre.prepare_and_forward(batch)
    return float(fm_loss_per_sample(loss_terms["fm_loss"]).mean().item())


# --------------------------------------------------------------------------- #
# 纯标准库统计（与 probe_fm_reward 同风格、不引入 numpy）：配对统计 + bootstrap + Wilcoxon。
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（输入须已排序）。q 为百分数（0–100）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _boot_ci_mean(vals: list[float], B: int = 3000, seed: int = 20260627) -> tuple[float, float]:
    """对一组（配对差）均值做 bootstrap 95% CI（重采样观测下标）。"""
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        stats.append(s)
    stats.sort()
    return _percentile(stats, 2.5), _percentile(stats, 97.5)


def _wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float, float, int]:
    """配对 Wilcoxon 符号秩检验（正态近似 + 连续性校正 + tie 校正）。

    返回 (W, z, p_two_sided, n_nonzero)。p 为双侧；方向由均值差/占比另行报告。
    纯标准库；样本小时正态近似偏保守，足以做「显著/不显著」的定性裁决。
    """
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n < 1:
        return (float("nan"), float("nan"), float("nan"), 0)
    absd = sorted(((abs(x), (1.0 if x > 0 else -1.0)) for x in d), key=lambda t: t[0])
    # 对 |diff| 相同者赋平均秩（1-based）
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1][0] == absd[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for kk in range(i, j + 1):
            ranks[kk] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    w_pos = sum(r for r, (_, s) in zip(ranks, absd) if s > 0)
    w_neg = sum(r for r, (_, s) in zip(ranks, absd) if s < 0)
    W = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return (W, float("nan"), float("nan"), n)
    z = (abs(W - mean_w) - 0.5) / sqrt(var_w)   # 连续性校正
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return (W, z, p, n)


# --------------------------------------------------------------------------- #
# manifest / winners 读取（字段取法与 data_pairs / sft_groundtruth 一致）。
# --------------------------------------------------------------------------- #
def _strip_win_suffix(fid: str) -> str:
    """把 winners_only.jsonl 的 fid（形如 "<base>-win"）归一回 base，便于与真实 manifest 的 fid 配对。

    winners_only.jsonl 用 `fid=f"{r['fid']}-win"`（见 pairs/extract_winners.py）；pairs_best_worst.jsonl
    则是裸 fid + win_audio。两者都兼容：去掉尾部 "-win"/"-w" 即得 base。
    """
    for suf in ("-win", "-w"):
        if fid.endswith(suf):
            return fid[: -len(suf)]
    return fid


def _load_real_manifest(path: Path) -> dict:
    """真实录音 manifest → {fid: (text, audio)}。

    取法对齐 sft_groundtruth(X4) / data_pairs.build_single_dataloader：
    audio = prompt_audio or audio（PromptRow 的 prompt_audio 即真实人声录音），text = r["text"]。
    """
    out: dict = {}
    for r in read_jsonl(path):
        audio = r.get("prompt_audio") or r.get("audio")
        if not audio:
            continue
        out.setdefault(str(r["fid"]), (r["text"], audio))
    return out


def _load_winners(path: Path) -> dict:
    """winners 清单 → {base_fid: (text, audio)}。兼容 winners_only.jsonl 与 pairs_best_worst.jsonl。"""
    out: dict = {}
    for r in read_jsonl(path):
        audio = r.get("win_audio") or r.get("audio")
        if not audio:
            continue
        out.setdefault(_strip_win_suffix(str(r["fid"])), (r["text"], audio))
    return out


def _mc_lfm(pre: DotsPreprocessor, fid: str, text: str, audio: str, k: int, seed0: int, dev,
            seeds: list[int]) -> tuple[float, float]:
    """对单条样本算 K 次 MC 平均的 L_FM；返回 (MC 均值, 单次 k=0 值)。seeds 由调用方给 → real/winner 共享。"""
    batch = _to_device(pre.collate([pre.process(fid, text, audio)]), dev)
    per = [_fm_with_seed(pre, batch, s, dev) for s in seeds]
    return sum(per) / len(per), per[0]


# --------------------------------------------------------------------------- #
# 子协议 (a) ODE-NLL —— 可选，8 处 〔需核实〕。dots.tts runtime 是否暴露概率流 ODE / 精确 −log p 存疑；
# 不暴露则本函数只打印「未确认 → 跳过」，主裁决回落到子协议 (b) typicality。
# --------------------------------------------------------------------------- #
def attempt_ode_nll(pre: DotsPreprocessor, paired, dev) -> bool:
    """尝试用概率流 ODE 算精确 −log p 做同款配对比较；接口未确认则安全跳过、返回 False。

    精确 change-of-variables NLL 需要逐个核实以下 dots.tts 内部接口（均 〔需核实〕）：
      〔需核实-1〕DotsTtsModel / DotsTtsRuntime 是否暴露 flow-matching 的**概率流 ODE 求解器**
                 （连续归一化流的 exact log-density，而非训练用的 velocity-MSE）。
      〔需核实-2〕velocity 场可调用签名 v(z, t, cond) 是否可在 runtime 外部独立取到。
      〔需核实-3〕latent 先验/基分布是否为标准高斯 N(0,I)（change-of-variables 的端点 log p 项）。
      〔需核实-4〕瞬时 log-density 变化所需的散度/迹估计（Hutchinson trace）是否有可挂载的 hook。
      〔需核实-5〕给定波形 → VAE encode 得 z0（latent）的入口与其确定性（是否含随机采样）。
      〔需核实-6〕text/speaker 等条件张量打包成 ODE 场 cond 的方式是否与训练前向一致。
      〔需核实-7〕积分时间约定（t:0→1 还是 1→0）及 velocity 符号，决定 NLL 累积方向。
      〔需核实-8〕NLL 落在 latent 空间还是音频空间——若要音频空间还需 VAE 解码器的 log-det Jacobian。
    以上任一不满足，精确 NLL 不可信，故默认不跑；用 `--ode-nll` 显式开启时也只在确认接口后才计算。
    """
    print("\n[子协议 (a) ODE-NLL] 〔需核实〕dots.tts runtime 的概率流 ODE / 精确 −log p 接口未在本仓核验"
          "（8 处 〔需核实〕见函数 docstring）→ 跳过，主裁决用子协议 (b) typicality。")
    # 不臆造接口：在 GPU 机器上逐条核实 attempt_ode_nll docstring 的 〔需核实-1..8〕 后再实装。
    _ = (pre, paired, dev)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="X14：L_FM 似然有效性独立锚（typicality：真实录音 vs reward-winner）")
    ap.add_argument("--ckpt", default="rednote-hilab/dots.tts-base")
    ap.add_argument("--real-manifest", default="data/prompts/dots_manifest_train.jsonl",
                    help="真实 held-out 录音清单（PromptRow；取 prompt_audio 作真实录音、text 作其文本）")
    ap.add_argument("--winners", default="pairs/out/winners_only.jsonl",
                    help="reward-winner 候选清单（winners_only.jsonl 或 pairs_best_worst.jsonl 均可）")
    ap.add_argument("--n", type=int, default=188, help="最多比较多少个配对 fid")
    ap.add_argument("--k", type=int, default=16, help="每条样本的 MC 采样次数（与 probe_fm_reward 同款）")
    ap.add_argument("--seed0", type=int, default=20260627)
    ap.add_argument("--ode-nll", action="store_true",
                    help="额外尝试子协议 (a) 精确 ODE-NLL（接口 〔需核实〕，未确认则自动跳过）")
    ap.add_argument("--out-csv", default="alignment/out/probe_lfm_validity.csv",
                    help="逐样本 L_FM 落盘（供复分析 / 独立重算配对统计）")
    args = ap.parse_args()

    dev = _device()
    print(f"device={dev}  real={args.real_manifest}  winners={args.winners}  N={args.n}  K={args.k}")
    pre = DotsPreprocessor.from_pretrained(args.ckpt)
    pre.model.to(dev)

    real_map = _load_real_manifest(Path(args.real_manifest))
    win_map = _load_winners(Path(args.winners))
    if not real_map:
        raise SystemExit(f"真实录音 manifest 空: {args.real_manifest}")
    if not win_map:
        raise SystemExit(f"winners 清单空: {args.winners}")

    paired_fids = [f for f in real_map if f in win_map]
    paired = bool(paired_fids)
    if paired:
        paired_fids = paired_fids[: args.n]
        print(f"配对模式：{len(paired_fids)} 个 fid 同时出现在真实 manifest 与 winners（共享 (x0,t) 配对比较）")
    else:
        # 无 fid 交集 → 无配对回落：各取前 N 条独立比较（失去公共噪声抵消，统计较弱，明确标注）。
        print("⚠ 无配对：真实 manifest 与 winners 无共同 fid → 回落为**无配对**均值比较"
              "（各自独立 seed、无公共噪声抵消，效能弱于配对；优先修正清单使 fid 可配对）。")

    seeds = [args.seed0 + k for k in range(args.k)]  # real/winner 同 draw 共享同一 (x0,t)

    rows_out: list[dict] = []
    lfm_real: list[float] = []
    lfm_win: list[float] = []
    diffs: list[float] = []   # winner − real（>0 ⇒ winner 损失更高=更不典型=更 off-manifold）

    if paired:
        for idx, fid in enumerate(paired_fids):
            rt, ra = real_map[fid]
            wt, wa = win_map[fid]
            mr, sr = _mc_lfm(pre, f"{fid}-real", rt, ra, args.k, args.seed0, dev, seeds)
            mw, sw = _mc_lfm(pre, f"{fid}-win", wt, wa, args.k, args.seed0, dev, seeds)
            lfm_real.append(mr)
            lfm_win.append(mw)
            diffs.append(mw - mr)
            rows_out.append({"fid": fid, "real_audio": ra, "winner_audio": wa, "text": rt,
                             "lfm_real_mc": mr, "lfm_winner_mc": mw, "lfm_diff": mw - mr,
                             "lfm_real_single": sr, "lfm_winner_single": sw, "k": args.k, "paired": 1})
            print(f"  [{idx:3d}] fid={fid}  L_FM real={mr:+.4f}  winner={mw:+.4f}  "
                  f"diff(win−real)={mw - mr:+.4f}")
    else:
        real_items = list(real_map.items())[: args.n]
        win_items = list(win_map.items())[: args.n]
        for fid, (t, a) in real_items:
            m, s = _mc_lfm(pre, f"{fid}-real", t, a, args.k, args.seed0, dev, seeds)
            lfm_real.append(m)
            rows_out.append({"fid": fid, "real_audio": a, "winner_audio": "", "text": t,
                             "lfm_real_mc": m, "lfm_winner_mc": "", "lfm_diff": "",
                             "lfm_real_single": s, "lfm_winner_single": "", "k": args.k, "paired": 0})
        for fid, (t, a) in win_items:
            m, s = _mc_lfm(pre, f"{fid}-win", t, a, args.k, args.seed0, dev, seeds)
            lfm_win.append(m)
            rows_out.append({"fid": fid, "real_audio": "", "winner_audio": a, "text": t,
                             "lfm_real_mc": "", "lfm_winner_mc": m, "lfm_diff": "",
                             "lfm_real_single": "", "lfm_winner_single": s, "k": args.k, "paired": 0})

    # ---- 落盘逐样本 L_FM（供独立重算配对统计 / 复分析） ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n✓ 逐样本 L_FM 已存 -> {out_csv}  ({len(rows_out)} 行)")

    # ---- 汇总裁决 ----
    print("\n========== 汇总（子协议 (b) typicality） ==========")
    if paired and diffs:
        n = len(diffs)
        mean_diff = sum(diffs) / n
        ci = _boot_ci_mean(diffs)
        W, z, p, n_nz = _wilcoxon_signed_rank(diffs)
        n_pos = sum(1 for d in diffs if d > 0)
        mean_real = sum(lfm_real) / n
        mean_win = sum(lfm_win) / n
        print(f"配对 n={n}  L_FM(real) 均值={mean_real:+.4f}  L_FM(winner) 均值={mean_win:+.4f}")
        print(f"diff=winner−real 均值={mean_diff:+.4f}  bootstrap 95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]")
        print(f"Wilcoxon 符号秩(双侧): W={W:.1f}  z={z:+.3f}  p={p:.4f}  (n_nonzero={n_nz})")
        print(f"diff>0（winner 更不典型）占比={n_pos}/{n}={n_pos / n:.2f}")

        sig = (ci[0] > 0) or (ci[1] < 0)
        real_lower = mean_diff > 0 and ci[0] > 0  # 真实录音 L_FM 系统性更低（更典型）
        print("\n=== 裁决（守 OPEN-SOAR-REFRAME 可证伪线）===")
        if real_lower:
            print("▶ 真实录音 L_FM **系统性低于** reward-winner（diff>0、CI 排除 0）：")
            print("  模型自身似然把真实 on-manifold 音频判为更典型、把 reward-winner 判为更 off-manifold。")
            print("  → 给「L_FM 是似然启发代理」一条**不依赖 open-SOAR 饱和**的独立支撑 = 关掉 C2；")
            print("    与「结构化显著反相关本身」(Spearman −0.187 / partial −0.392 + per-component 三轴) 互证 L_FM 非退化。")
            print("  （caveat 保留：L_FM 仍是似然启发代理、非忠实 NLL 上界；精确 NLL 见子协议 (a)〔需核实〕。）")
        elif sig and mean_diff < 0:
            print("▶ 真实录音 L_FM **反而高于** winner（diff<0、CI 排除 0）：与典型性预期相反——")
            print("  按可证伪线 **收回『L_FM 有效似然代理』**：模型在自身损失下不偏好真实录音，")
            print("  L_FM 不能当似然方向标尺，§方法假设须降级（写入 Limitations）。")
        else:
            print("▶ diff 的 CI 含 0（未显著）：真实录音 L_FM **不低于** winner 的零假设无法否定——")
            print("  按可证伪线 **收回『L_FM 有效似然代理』** 的强表述，降级为「方向未定、需更大 N / 精确 NLL」。")
    elif not paired:
        # 无配对回落：仅作均值差的弱对照（无公共噪声抵消、无 Wilcoxon），明确标注效能弱。
        if lfm_real and lfm_win:
            mean_real = sum(lfm_real) / len(lfm_real)
            mean_win = sum(lfm_win) / len(lfm_win)
            print(f"[无配对] L_FM(real) 均值={mean_real:+.4f} (n={len(lfm_real)})  "
                  f"L_FM(winner) 均值={mean_win:+.4f} (n={len(lfm_win)})  "
                  f"均值差(winner−real)={mean_win - mean_real:+.4f}")
            print("  ⚠ 无配对=无公共噪声抵消、无 Wilcoxon，仅供方向参考；裁决请用可配对清单重跑。")
        else:
            print("[无配对] real/winner 至少一侧为空，无法比较。")

    # ---- 子协议 (a) ODE-NLL（可选、接口 〔需核实〕）----
    if args.ode_nll:
        attempt_ode_nll(pre, paired_fids if paired else [], dev)


if __name__ == "__main__":
    main()
