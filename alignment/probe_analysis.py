"""
probe_analysis.py — 第一轮审稿后的统计加固：把 reviewer 最狠的几条一次算清（CPU、秒级、可反复跑）

背景（docs/07 第一轮审稿）:
    审稿人在 §6.7 的核心发现上挑出必须在投稿前处理的统计问题：
      C1/DA：confound 测错了量——测了 corr(reward_gap, setting_gap)，**该测 corr(FM_margin, setting_gap)**；
             且 §6.7 自相矛盾（guidance 既是机理又被排除）。
      C5/R1：guidance-confound「排除」是「接受零假设当等价」——需 **偏相关 + TOST**，否则不能说 intrinsic。
      C4/R1：headline 领 Pearson −0.41，但稳健 Spearman 仅 −0.187 → 需 **杠杆诊断 + 双系数 CI**。
      R1：独立性未验（188 对嵌套在多少 distinct prompt？单说话人）→ 需报 distinct-prompt 数。
      R2/R3：**per-component 分解**（FM_margin vs WER/SIM/UTMOS gap 各自），看是不是只是 FM-vs-WER。
      MINOR：binomial 检验 82/188 vs 0.5（p≈0.08，非"at chance"）。

做什么（全部基于已落盘数据，无需 GPU、无需模型）:
    读 probe_fm_reward.py 落盘的 alignment/out/probe_per_pair.csv（逐对 fid/reward/mc_margin/win_rate），
    join candidates_index.jsonl（每对的 setting-gap = setting(win)−setting(lose)），可选 join rewards.jsonl
    （per-component WER/SIM/UTMOS）。算：
      (1) headline 双系数 Pearson/Spearman + bootstrap 95% CI（C4）；
      (2) binomial 检验 n_pos vs 0.5（MINOR）；
      (3) 杠杆：删 top-5% |reward_gap| 后的 Pearson + 留一法最大影响（C4）；
      (4) **正确的 confound 测**：corr(FM_margin, 各 setting_gap)（C1/DA）；
      (5) **偏相关** corr(FM_margin, reward_gap | guidance,num_steps,speaker) + bootstrap CI + TOST（C1/C5）——
          这是"intrinsic"存亡的唯一裁决；
      (6) per-component：corr(FM_margin, WER_gap / SIM_gap / UTMOS_gap)（若 rewards.jsonl 有分量）。
      (7) **X1 按 reward_gap 分层**：按 |reward_gap| 的中位数 / 上四分位把 pair 分层，在 **高-gap(非平局)子集**
          上复用 _pearson/_spearman/_boot_ci 重算 Pearson/Spearman + 95%CI，并与全集对比——回应审稿
          「win_rate 0.47 大半是 near-tie」：若子集系数仍显著且 |系数| ≥ 全集，则反相关非 near-tie 产物。
怎么用:
    python alignment/probe_analysis.py \
        --csv alignment/out/probe_per_pair.csv \
        --index candidates/out/candidates_index.jsonl \
        --rewards reward/out/rewards.jsonl     # 可选，用于 per-component
    再加 --out alignment/out/probe_analysis_report.txt 可把以上全部 stdout 报告同时归档一份(utf-8)；
      --out 默认 None=只打 stdout、不落盘（默认行为不变）——二选一里取「保持 None、由调用方给路径」：
      传了才归档，每次分析自动落档、不靠人工粘贴；不传则与改前完全一致。
为什么这么设计:
    把"需要 GPU 的探针"与"纯统计分析"解耦：探针落盘一次（~30min/N188），本脚本可反复跑（秒级），
    审稿人要的统计加固都在这里，且结论（尤其偏相关）直接决定论文 §6.7 措辞。
对实验意义（裁决"intrinsic"）:
    • 若偏相关仍显著（CI 排除 0）→ 反相关在控住生成设置后仍存活 → "intrinsic" 站得住；
    • 若偏相关塌到不显著 → 反相关（部分）由生成设置驱动 → §6.7 必须从"intrinsic"降级为"under-powered/grid effect"。
本机只语法检查；真实运行在有 CSV+index 的服务器上。
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb
from pathlib import Path

import numpy as np  # type: ignore


# ---------- 输出收集（--out 归档用） ----------
_REPORT_LINES: list[str] = []  # 所有报告行的缓冲；main 末尾按 --out 决定是否落盘


def _emit(line: str = "") -> None:
    """既打 stdout 又收进缓冲：行为与 print(单串) 一致，额外把整行存入 _REPORT_LINES 供 --out 归档。"""
    print(line)
    _REPORT_LINES.append(line)


# ---------- join 工具（与 inspect_guidance_confound 同款键） ----------
def _key(p: str) -> str:
    p = str(p)
    return f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"


def _load_index(path: Path) -> dict:
    by_audio = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = {"guidance_scale": r.get("guidance_scale"), "num_steps": r.get("num_steps"),
             "speaker_scale": r.get("speaker_scale")}
        by_audio[_key(r["audio"])] = s
        by_audio.setdefault(os.path.basename(r["audio"]), s)
    return by_audio


def _load_rewards_components(path: Path) -> dict:
    """audio -> {wer,sim,utmos}（若 rewards.jsonl 含分量）。缺则返回空，per-component 跳过。"""
    if not path or not path.exists():
        return {}
    by_audio = {}
    has_any = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        comp = {k: r.get(k) for k in ("wer", "sim", "utmos")}
        if any(v is not None for v in comp.values()):
            has_any = True
        a = r.get("audio")
        if a:
            by_audio[_key(a)] = comp
            by_audio.setdefault(os.path.basename(a), comp)
    return by_audio if has_any else {}


def _lookup(d: dict, audio: str):
    return d.get(_key(audio)) or d.get(os.path.basename(audio))


# ---------- 统计工具 ----------
def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a):
    return np.argsort(np.argsort(np.asarray(a, float))).astype(float)


def _spearman(a, b) -> float:
    return _pearson(_rank(a), _rank(b))


def _partial_pearson(y, x, Z) -> float:
    """corr(resid(y~Z), resid(x~Z))：控住 Z（设置-gap 矩阵）后的偏相关。"""
    y, x, Z = np.asarray(y, float), np.asarray(x, float), np.asarray(Z, float)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    return _pearson(ry, rx)


def _boot_ci(fn, arrays, B=3000, seed=20260626):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(*[np.asarray(a)[idx] for a in arrays])
        if v == v:  # 排除 NaN
            stats.append(v)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(sum(pi for pi in probs if pi <= probs[k] + 1e-12))


def _sig(lo: float, hi: float) -> str:
    return "显著(CI 排除 0)" if (lo > 0 or hi < 0) else "不显著(CI 含 0)"


def main() -> None:
    ap = argparse.ArgumentParser(description="probe 统计加固（偏相关/杠杆/per-component/CI）")
    ap.add_argument("--csv", default="alignment/out/probe_per_pair.csv")
    ap.add_argument("--index", default="candidates/out/candidates_index.jsonl")
    ap.add_argument("--rewards", default="reward/out/rewards.jsonl")
    ap.add_argument("--tost", type=float, default=0.1, help="TOST 可忽略界 |ρ|<bound 视作等价")
    ap.add_argument("--out", default=None,
                    help="把全部分析报告同时落一份文件(utf-8)；默认 None=只打 stdout、不归档（默认行为不变）")
    args = ap.parse_args()

    # ---- 读 CSV ----
    rows = []
    with open(args.csv, encoding="utf-8") as f:
        import csv
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise SystemExit(f"CSV 空: {args.csv}")
    fid = [r["fid"] for r in rows]
    margin = np.array([float(r["mc_margin"]) for r in rows])
    rgap = np.array([float(r["reward_gap"]) for r in rows])
    winr = np.array([float(r["win_rate"]) for r in rows])
    n = len(rows)
    n_pos = int((margin > 0).sum())

    _emit(f"==== probe_analysis  (n={n} 对) ====\n")

    # ---- (R1) distinct-prompt（独立性） ----
    ndp = len(set(fid))
    _emit(f"[独立性] distinct fid(prompt) 数 = {ndp} / {n} 对  "
          f"({'1 对/prompt，prompt 级独立(但单说话人)' if ndp == n else f'<n：{n-ndp} 对与他对共享 prompt → 需 cluster/permutation'})\n")

    # ---- (C4) headline 双系数 + bootstrap CI ----
    pear = _pearson(margin, rgap)
    spear = _spearman(margin, rgap)
    pear_ci = _boot_ci(_pearson, (margin, rgap))
    spear_ci = _boot_ci(_spearman, (margin, rgap))
    _emit("[C4 headline] FM_margin vs reward_gap:")
    _emit(f"  Pearson  r={pear:+.3f}  (r²={pear**2:.3f})  95%CI[{pear_ci[0]:+.3f},{pear_ci[1]:+.3f}]  {_sig(*pear_ci)}")
    _emit(f"  Spearman ρ={spear:+.3f}  (ρ²={spear**2:.3f})  95%CI[{spear_ci[0]:+.3f},{spear_ci[1]:+.3f}]  {_sig(*spear_ci)}")
    _emit(f"  → 稳健量(Spearman) ρ²={spear**2:.1%} 才是该领的头条；Pearson/Spearman 比值={abs(pear/spear) if spear else float('nan'):.2f}"
          f"{'（>2 ⇒ 高杠杆点主导线性系数，C4 成立）' if spear and abs(pear/spear) > 2 else ''}\n")

    # ---- (MINOR) binomial ----
    bp = _binom_two_sided(n_pos, n)
    _emit(f"[binomial] mc_margin>0 的 {n_pos}/{n}={n_pos/n:.3f} vs 0.5 双侧 p={bp:.3f}  "
          f"平均 win_rate={winr.mean():.3f}  "
          f"→ {'与 0.5 无显著差' if bp > 0.05 else '显著偏离 0.5'}（{'不能写 at chance' if 0.01 < bp < 0.1 else ''}）\n")

    # ---- (C4) 杠杆诊断 ----
    thr = np.percentile(np.abs(rgap), 95)
    keep = np.abs(rgap) <= thr
    pear_trim = _pearson(margin[keep], rgap[keep])
    loo = [abs(pear - _pearson(np.delete(margin, i), np.delete(rgap, i))) for i in range(n)]
    imax = int(np.argmax(loo))
    _emit("[C4 杠杆]:")
    _emit(f"  删 top-5% |reward_gap|({int((~keep).sum())} 对)后 Pearson={pear_trim:+.3f}（原 {pear:+.3f}）"
          f"{' → 塌向 Spearman，头条必须换秩相关' if abs(pear_trim) < abs(spear) * 1.3 else ''}")
    _emit(f"  留一法最大单点影响：删 #{imax}(fid={fid[imax]}, reward_gap={rgap[imax]:+.2f}) 使 Pearson 变 {loo[imax]:+.3f}\n")

    # ---- (X1) 按 |reward_gap| 分层：高-gap(非平局)子集存活 ----
    # 为什么：审稿指出 win_rate≈0.47、平均 win_rate 接近 0.5，怀疑反相关大半来自 near-tie（两候选质量几乎
    #   一样、reward_gap≈0）对里 FM_margin 的随机噪声。near-tie 对的 |reward_gap| 小，故按 |reward_gap| 分层、
    #   只留高-gap（真有质量差）子集重算；若反相关在该子集仍显著且不弱于全集，则它不是 near-tie 产物。
    # 怎么做：复用本文件 _pearson/_spearman/_boot_ci，对 |reward_gap| 中位数(top-50%)与上四分位(top-25%)两档
    #   阈值各取一次高-gap 子集，与全集系数并排打印；裁决以稳健量 Spearman 为准（与 C4 一致）。
    agap = np.abs(rgap)
    _emit("[X1 高-gap 子集存活] 回应审稿『win_rate 0.47 大半 near-tie』——按 |reward_gap| 分层、只看真有质量差的非平局子集：")
    _emit(f"  全集(n={n}):                         "
          f"Pearson r={pear:+.3f} 95%CI[{pear_ci[0]:+.3f},{pear_ci[1]:+.3f}] {_sig(*pear_ci)} | "
          f"Spearman ρ={spear:+.3f} 95%CI[{spear_ci[0]:+.3f},{spear_ci[1]:+.3f}] {_sig(*spear_ci)}")
    for label, q in (("top-50%(>中位数)", 50), ("top-25%(>上四分位)", 75)):
        thr_g = float(np.percentile(agap, q))
        sub = agap >= thr_g  # 留 |reward_gap| 大者 = 非平局（真有质量差）
        ns = int(sub.sum())
        if ns < 10:
            _emit(f"  {label}(|reward_gap|≥{thr_g:.3f}): 子集 n={ns}<10，样本不足，跳过")
            continue
        m_s, r_s = margin[sub], rgap[sub]
        sp, ss = _pearson(m_s, r_s), _spearman(m_s, r_s)
        sp_ci, ss_ci = _boot_ci(_pearson, (m_s, r_s)), _boot_ci(_spearman, (m_s, r_s))
        _emit(f"  {label}(|reward_gap|≥{thr_g:.3f}, n={ns}): "
              f"Pearson r={sp:+.3f} 95%CI[{sp_ci[0]:+.3f},{sp_ci[1]:+.3f}] {_sig(*sp_ci)} | "
              f"Spearman ρ={ss:+.3f} 95%CI[{ss_ci[0]:+.3f},{ss_ci[1]:+.3f}] {_sig(*ss_ci)}")
        sig_sub = ss_ci[0] > 0 or ss_ci[1] < 0  # 子集 Spearman 是否显著（CI 排除 0）
        ge_full = abs(ss) >= abs(spear)          # 子集 |ρ| 是否不弱于全集
        if sig_sub and ge_full:
            verdict = (f"非平局子集 Spearman 仍显著且 |ρ|={abs(ss):.3f} ≥ 全集 {abs(spear):.3f} "
                       f"→ 坐实：反相关非 near-tie 产物，真有质量差时不弱反更强")
        elif sig_sub:
            verdict = (f"非平局子集 Spearman 仍显著但 |ρ|={abs(ss):.3f} < 全集 {abs(spear):.3f} "
                       f"→ 反相关存活、未被 near-tie 放大（截尾缩小了 gap 动态范围会压系数）")
        else:
            verdict = ("非平局子集 Spearman CI 含 0 → 反相关在高-gap 子集未存活，"
                       "可能是 near-tie 产物或子集样本太少，§6.7 需更大 N 复核")
        _emit(f"    → {verdict}")
    _emit()

    # ---- join 设置-gap ----
    idx = _load_index(Path(args.index))
    g = {"guidance_scale": [], "num_steps": [], "speaker_scale": []}
    ok = []
    for r in rows:
        sw, sl = _lookup(idx, r["win_audio"]), _lookup(idx, r["lose_audio"])
        if sw and sl and all(sw[k] is not None and sl[k] is not None for k in g):
            ok.append(True)
            for k in g:
                g[k].append(float(sw[k]) - float(sl[k]))
        else:
            ok.append(False)
    ok = np.array(ok)
    if ok.sum() >= 10:
        m_ok, r_ok = margin[ok], rgap[ok]
        G = np.column_stack([np.array(g[k]) for k in g])
        _emit(f"[C1/DA 正确的 confound 测] corr(FM_margin, 各 setting_gap)  (join 成功 {int(ok.sum())}/{n}):")
        for j, k in enumerate(g):
            c = _pearson(m_ok, G[:, j]); ci = _boot_ci(_pearson, (m_ok, G[:, j]))
            _emit(f"  {k:14s}: r={c:+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {_sig(*ci)}")
        _emit("  （审稿人指出：该测的是 FM_margin×setting_gap，不是 reward_gap×setting_gap）\n")

        # ---- (C1/C5) 偏相关：intrinsic 存亡裁决 ----
        pp = _partial_pearson(m_ok, r_ok, G)
        pp_ci = _boot_ci(lambda a, b, *Z: _partial_pearson(a, b, np.column_stack(Z)),
                         (m_ok, r_ok, *[G[:, j] for j in range(G.shape[1])]))
        _emit("[C1/C5 偏相关（裁决 intrinsic）] corr(FM_margin, reward_gap | guidance,num_steps,speaker):")
        _emit(f"  partial r={pp:+.3f}  95%CI[{pp_ci[0]:+.3f},{pp_ci[1]:+.3f}]  {_sig(*pp_ci)}")
        tost_pass = pp_ci[0] > -args.tost and pp_ci[1] < args.tost
        _emit(f"  原始 Pearson={_pearson(m_ok, r_ok):+.3f} → 控住设置后 {pp:+.3f}"
              f"（{'基本不变 → 反相关 intrinsic 站得住' if abs(pp) > 0.8*abs(_pearson(m_ok,r_ok)) else '明显减弱 → 部分由生成设置驱动，intrinsic 需降级'}）")
        _emit(f"  TOST(|ρ|<{args.tost}): {'通过=可忽略' if tost_pass else '不通过'}\n")
    else:
        _emit(f"[C1] 设置-gap join 成功太少({int(ok.sum())}/{n})，跳过偏相关——检查 candidates_index 路径\n")

    # ---- (R2/R3) per-component ----
    comp = _load_rewards_components(Path(args.rewards))
    if comp:
        names = ("wer", "sim", "utmos")
        cg = {k: [] for k in names}; okc = []
        for r in rows:
            cw, cl = _lookup(comp, r["win_audio"]), _lookup(comp, r["lose_audio"])
            if cw and cl and all(cw[k] is not None and cl[k] is not None for k in names):
                okc.append(True)
                for k in names:
                    cg[k].append(float(cw[k]) - float(cl[k]))
            else:
                okc.append(False)
        okc = np.array(okc)
        if okc.sum() >= 10:
            mc = margin[okc]
            _emit(f"[R2/R3 per-component] corr(FM_margin, 各分量 gap)  (join {int(okc.sum())}/{n})：")
            for k in names:
                c = _pearson(mc, np.array(cg[k])); ci = _boot_ci(_pearson, (mc, np.array(cg[k])))
                _emit(f"  {k.upper():6s}_gap: r={c:+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {_sig(*ci)}")
            _emit("  （看反相关是不是只是 FM-vs-WER；告诉社区是『弃 FM-as-reward』还是『只修 reward 组成』）\n")
    else:
        _emit("[per-component] rewards.jsonl 无 wer/sim/utmos 分量，跳过（如需：让 reward 阶段落盘分量）\n")

    _emit("==== 据此改 §6.7：偏相关存活→保留 intrinsic(但头条换 Spearman)；偏相关塌→改『under-powered/部分 grid 驱动』 ====")

    # ---- (--out) 把以上打到 stdout 的全部报告同时归档一份；默认 None 时不落盘、行为不变 ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
        print(f"\n[--out] 以上报告已同时归档 → {out_path}（{len(_REPORT_LINES)} 条 emit，utf-8）")


if __name__ == "__main__":
    main()
