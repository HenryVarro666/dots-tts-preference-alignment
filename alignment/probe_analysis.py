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
      (8) **审稿加固附加项**（B 组，全部走 _emit→--out 归档；本机只语法检查、真实运行在有数据的 GPU 服务器）：
          B1 partial **Spearman**（秩偏相关：对 y/x/各 Z 列先 _rank 再复用 _partial_pearson 的 OLS 残差化）与
             partial Pearson 并排——同符号同显著即化解「Spearman 头条 vs Pearson 偏相关」张力，秩偏相关对非线性更稳；
          B2 对决定性 partial r 加 **Bonferroni/Holm**（相关族 m≈14、α=0.05/m）重算更严 CI，判是否过校正；
          B3 对 **SIM_gap** 复用头条同款 trim-top-5%+LOO 杠杆诊断，据 trim 后是否仍显著定「保留 / 退役为带 hedge 次要佐证」；
          B4 在偏相关 Z 里加 **num_steps²**（NFE 非线性项）复跑一次，与线性设定对比（决定性关闸仍是 X5 settings-matched）；
          B6 若 CSV 含 per_draw_json：**split-half 可靠性 r_hh + 解衰减 ρ_dis=ρ/√r_hh**（测量噪声下报告 ρ 的保守下界量级）。
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
    • LINEAR caveat：偏相关用**线性 OLS 残差化**，只剔除 {guidance,num_steps,speaker} 的线性投影；num_steps(NFE)
      可能非线性作用于 velocity-MSE，故偏相关的 intrinsic 裁决「以线性设定为条件」——B4 加 NFE² 是一阶非线性探查，
      消线性+非线性+交互的**决定性非参关闸是 X5 settings-matched**（setting_gap≡0，本脚本只做线性条件下的稳健性）。
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


def _partial_spearman(y, x, Z) -> float:
    """秩偏相关：把 y/x/各 Z 列先 _rank 再复用 _partial_pearson 的同款 OLS 残差化（残差化后 _pearson）。
    与 partial Pearson 并排可化解「Spearman 头条 vs Pearson 偏相关」张力，且对(单调)非线性更稳。"""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Zr = np.column_stack([_rank(Z[:, j]) for j in range(Z.shape[1])])
    return _partial_pearson(_rank(y), _rank(x), Zr)


def _boot_ci(fn, arrays, B=3000, seed=20260626, alpha=0.05):
    """bootstrap CI。默认 alpha=0.05→分位[2.5,97.5]（与改前完全一致）；传更小 alpha 即做
    Bonferroni/Holm 家族校正后的更严 CI（B2：α=0.05/14→更宽分位、更难显著，z 1.96→2.91）。"""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(*[np.asarray(a)[idx] for a in arrays])
        if v == v:  # 排除 NaN
            stats.append(v)
    # 退化保护（X5 settings-matched 触发）：若每个 bootstrap 复本都是 NaN ⇒ stats 全空。
    #   成因：某输入列为常数——settings-matched 下 setting_gap≡0，_pearson 因 std==0 恒返回 NaN。
    #   旧码此处直接 np.percentile([])→IndexError 炸全程（见 run_x5.sh 第 5 步）。CI 此时无定义，
    #   返回 (nan, nan) 让上层退化为「NaN/不显著」而非崩溃（run_x5.sh 已声明该退化属正常）。
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(sum(pi for pi in probs if pi <= probs[k] + 1e-12))


def _sig(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:  # NaN CI（退化：常数列/全 NaN bootstrap）——既非「排除 0」也非「含 0」，须如实标注
        return "退化(NaN CI：常数列/样本不足)"
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
    n_join = int(ok.sum())
    if n_join >= 10:
        m_ok, r_ok = margin[ok], rgap[ok]
        G = np.column_stack([np.array(g[k]) for k in g])
    # X5 settings-matched 检测：winner/loser 同设置 ⇒ 每个 setting_gap 列恒为常数(≡0)。此时整段
    #   confound/偏相关在数学上退化：(a) corr(FM_margin, 常数)=NaN、bootstrap 全空（_boot_ci 已护住）；
    #   (b)「partial out 一个常数」是无操作 ⇒ 偏相关恒等于原始相关，写「控住设置后基本不变→intrinsic」是循环论证。
    #   故按 run_x5.sh 设计在此跳过本段；intrinsic 改由上方 C4 headline 在「设置完全相等」对上是否仍显著
    #   反相关来无条件裁决（强于线性偏相关，消线性+非线性+交互三类设置依赖）。
    settings_matched = n_join >= 10 and bool(np.all(G.std(axis=0) <= 1e-12))
    if settings_matched:
        _emit(f"[C1/DA confound + C1/C5 偏相关] 跳过：检出 settings-matched（X5，join {n_join}/{n}）——三个 setting_gap 全为常数 0")
        _emit("  winner/loser 同 guidance/num_steps/speaker、仅变 seed ⇒ setting_gap≡0：")
        _emit("  • confound 相关 corr(FM_margin, 常数)=NaN（无定义）；")
        _emit("  • 偏相关「控住常数设置」是无操作 ⇒ 恒等于原始相关，自证 intrinsic 属循环论证，故不报。")
        _emit("  设置控制已由『配对设计』无条件达成（强于线性偏相关）。")
        _emit("  ▶ intrinsic 裁决看上方 [C4 headline]：FM_margin vs reward_gap 在设置完全相等的对上是否仍显著反相关。\n")
    elif n_join >= 10:
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
        _emit(f"  partial Pearson  r={pp:+.3f}  95%CI[{pp_ci[0]:+.3f},{pp_ci[1]:+.3f}]  {_sig(*pp_ci)}")

        # ---- (B1) partial Spearman：秩偏相关，与 partial Pearson 并排，化解头条 vs 偏相关 的统计张力 ----
        psp = _partial_spearman(m_ok, r_ok, G)
        psp_ci = _boot_ci(lambda a, b, *Z: _partial_spearman(a, b, np.column_stack(Z)),
                          (m_ok, r_ok, *[G[:, j] for j in range(G.shape[1])]))
        same_sign = (pp < 0) == (psp < 0)
        same_sig = _sig(*pp_ci) == _sig(*psp_ci)
        _emit(f"  partial Spearman ρ={psp:+.3f}  95%CI[{psp_ci[0]:+.3f},{psp_ci[1]:+.3f}]  {_sig(*psp_ci)}")
        _emit(f"  → partial Pearson 与 partial Spearman {'同符号' if same_sign else '异符号'}、"
              f"{'同显著' if same_sig else '显著性不一致'}"
              f"{'：化解『Spearman 头条 vs Pearson 偏相关』张力，且秩偏相关对(单调)非线性更稳' if same_sign and same_sig else '：§6.7 需并报两者并解释差异'}")

        # ---- (B2) Bonferroni/Holm：对相关族(m≈14)用 α=0.05/m 重算决定性 partial r 的更严 CI ----
        # 为什么：偏相关是唯一「预设确证终点」；审稿要 FDR/Bonferroni 的合法对象本是 >100 条的消融网格，
        #   但把决定性 partial r 也过一遍最严校正、上界仍 <0，才彻底堵住「多重比较」的口（z 1.96→2.91）。
        M_FAM = 14  # 相关族近似条目数
        # α=0.05/14 把分位推到 [0.179, 99.821]%，默认 B=3000 时每侧仅 ~5 个 bootstrap 样本支撑端点、
        #   方差极大；这条是决定性裁决，故单独把 B 调大(20000)让校正后 CI 端点稳定（仅此一调用，秒级）。
        pp_ci_bonf = _boot_ci(lambda a, b, *Z: _partial_pearson(a, b, np.column_stack(Z)),
                              (m_ok, r_ok, *[G[:, j] for j in range(G.shape[1])]),
                              alpha=0.05 / M_FAM, B=20000)
        bonf_pass = pp_ci_bonf[0] > 0 or pp_ci_bonf[1] < 0
        _emit(f"  Bonferroni/{M_FAM}(α={0.05 / M_FAM:.4f}, {100 * (1 - 0.05 / M_FAM):.2f}%CI)"
              f"[{pp_ci_bonf[0]:+.3f},{pp_ci_bonf[1]:+.3f}] → "
              f"{f'partial r 过 Bonferroni/{M_FAM} 校正(CI 上界<0)' if bonf_pass else f'未过 Bonferroni/{M_FAM} 校正(CI 含 0)'}")

        tost_pass = pp_ci[0] > -args.tost and pp_ci[1] < args.tost
        _emit(f"  原始 Pearson={_pearson(m_ok, r_ok):+.3f} → 控住设置后 {pp:+.3f}"
              f"（{'基本不变 → 反相关 intrinsic 站得住' if abs(pp) > 0.8*abs(_pearson(m_ok,r_ok)) else '明显减弱 → 部分由生成设置驱动，intrinsic 需降级'}）")
        _emit(f"  TOST(|ρ|<{args.tost}): {'通过=可忽略' if tost_pass else '不通过'}")

        # ---- (B4) 非线性(NFE²)设定：把 num_steps 的二次项并入 Z 复跑偏相关 ----
        # 为什么：偏相关只做线性 OLS 残差化、只剔 num_steps 的线性投影；NFE 可能非线性作用于 velocity-MSE。
        #   加 num_steps² 吸收一阶非线性、看 intrinsic 偏相关是否仍存活。注：消线性+非线性+交互的决定性
        #   关闸是 X5 settings-matched(setting_gap≡0)，本行只是「以非线性设定为条件」的稳健性探查。
        nfe_idx = list(g).index("num_steps")
        G_nl = np.column_stack([G, G[:, nfe_idx] ** 2])
        pp_nl = _partial_pearson(m_ok, r_ok, G_nl)
        pp_nl_ci = _boot_ci(lambda a, b, *Z: _partial_pearson(a, b, np.column_stack(Z)),
                            (m_ok, r_ok, *[G_nl[:, j] for j in range(G_nl.shape[1])]))
        nl_same = (pp < 0) == (pp_nl < 0) and _sig(*pp_ci) == _sig(*pp_nl_ci)
        _emit(f"  非线性(NFE²)设定: partial r={pp_nl:+.3f} 95%CI[{pp_nl_ci[0]:+.3f},{pp_nl_ci[1]:+.3f}] {_sig(*pp_nl_ci)}"
              f"（线性 {pp:+.3f} → 加 num_steps² 后 {pp_nl:+.3f}，{'方向/显著不变' if nl_same else '方向或显著性有变化'}）")
        _emit("  注：偏相关 intrinsic 裁决以线性设定为条件；决定性的非参关闸是 X5 settings-matched(setting_gap≡0，消线性+非线性+交互)。\n")
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
            _emit("  （看反相关是不是只是 FM-vs-WER；告诉社区是『弃 FM-as-reward』还是『只修 reward 组成』）")

            # ---- (B3 / SIM-HEDGE) SIM_gap 杠杆诊断：复用头条同款 trim-top-5% + LOO（杠杆块 207-216 同逻辑）----
            # 为什么：SIM 来自单说话人 LJSpeech、近-ceiling、判别力弱，per-component 的 SIM_gap 反相关可能由
            #   少数有真实 SIM 差的对(杠杆点)带动。复用头条同款 trim+LOO，看 trim 后是否仍显著，据此决定 SIM_gap
            #   是承重证据、还是「带 hedge 的次要佐证」（承重放更具判别力的 UTMOS_gap、方向明确的 WER_gap）。
            sg = np.array(cg["sim"])
            if sg.std() > 0 and len(mc) >= 10:
                sim_r = _pearson(mc, sg)
                thr_s = np.percentile(np.abs(sg), 95)
                keep_s = np.abs(sg) <= thr_s
                sim_r_trim = _pearson(mc[keep_s], sg[keep_s])
                sim_ci_trim = _boot_ci(_pearson, (mc[keep_s], sg[keep_s]))
                loo_s = [abs(sim_r - _pearson(np.delete(mc, i), np.delete(sg, i))) for i in range(len(mc))]
                imax_s = int(np.argmax(loo_s))
                _emit("  [B3/SIM-HEDGE SIM_gap 杠杆诊断] 复用头条同款 trim-top-5% + LOO：")
                _emit(f"    全集 SIM_gap r={sim_r:+.3f}；删 top-5% |SIM_gap|({int((~keep_s).sum())} 对)后 r={sim_r_trim:+.3f} "
                      f"95%CI[{sim_ci_trim[0]:+.3f},{sim_ci_trim[1]:+.3f}] {_sig(*sim_ci_trim)}")
                _emit(f"    留一法最大单点：删 #{imax_s} 使 SIM_gap r 变 {loo_s[imax_s]:+.3f}")
                sim_trim_sig = sim_ci_trim[0] > 0 or sim_ci_trim[1] < 0
                if sim_trim_sig and abs(sim_r_trim) >= 0.5 * abs(sim_r):
                    _emit("    → trim 后仍显著且量级未塌：SIM_gap 反相关非少数杠杆点独撑，可作次要佐证保留")
                else:
                    _emit("    → trim 后塌/转不显著：单说话人 LJSpeech 近-ceiling、判别力弱的 SIM_gap 由少数杠杆对带动，"
                          "退役为带 hedge 的次要佐证；承重证据放更具判别力的 UTMOS_gap 与方向明确的 WER_gap")
            _emit()
    else:
        _emit("[per-component] rewards.jsonl 无 wer/sim/utmos 分量，跳过（如需：让 reward 阶段落盘分量）\n")

    # ---- (B6 / ATTENUATION) split-half 可靠性 + 解衰减 ρ（若 CSV 含 per_draw_json）----
    # 为什么：mc_margin 是 K=16 draw 的 MC 估计、含测量噪声；经典 regression dilution 只把 |相关| 往 0 衰减。
    #   把每对逐 draw 序列分两半各 8、各求 mc，两半 mc 跨对相关=split-half 可靠性 r_hh（Spearman-Brown 拉全长 r_full）；
    #   解衰减 ρ_dis=ρ/√r_full 给「报告 ρ 被噪声压低了多少」的量级——disattenuation 只增不减，故报告 ρ 是保守下界。
    if "per_draw_json" in rows[0]:
        halfA, halfB, n_ok_pd, n_half = [], [], 0, 0
        for r in rows:
            raw = r.get("per_draw_json")
            try:
                draws = json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                draws = None
            if isinstance(draws, list) and len(draws) >= 2 and all(isinstance(v, (int, float)) for v in draws):
                d = np.asarray(draws, float)
                h = len(d) // 2
                halfA.append(float(d[:h].mean())); halfB.append(float(d[h:2 * h].mean()))
                n_ok_pd += 1; n_half = h
        if n_ok_pd >= 10:
            r_hh = _pearson(np.array(halfA), np.array(halfB))
            r_full = (2 * r_hh) / (1 + r_hh) if r_hh == r_hh and r_hh > -1 else float("nan")
            rho_dis = spear / np.sqrt(r_full) if r_full == r_full and r_full > 0 else float("nan")
            _emit("[B6/ATTENUATION split-half 可靠性 + 解衰减] 用 per_draw_json 逐 draw 序列：")
            _emit(f"  每对分两半各 {n_half} draw，跨对 split-half r_hh={r_hh:+.3f}；"
                  f"Spearman-Brown 全长 r_full={r_full:+.3f}（可解析 n={n_ok_pd}）")
            _emit(f"  头条 Spearman ρ={spear:+.3f} → 解衰减 ρ_dis=ρ/√r_full={rho_dis:+.3f}"
                  f"（|ρ_dis|≥|ρ| ⇒ 报告 ρ 是被 K=16 测量噪声压低的保守下界，disattenuation 只增不减）\n")
        else:
            _emit(f"[B6 split-half] per_draw_json 可解析对太少({n_ok_pd})，跳过解衰减\n")
    else:
        _emit("[B6 split-half] CSV 无 per_draw_json 列，跳过（如需：让 probe_fm_reward.py 落逐 draw 序列）\n")

    _emit("==== 据此改 §6.7：偏相关存活→保留 intrinsic(但头条换 Spearman)；偏相关塌→改『under-powered/部分 grid 驱动』 ====")

    # ---- (--out) 把以上打到 stdout 的全部报告同时归档一份；默认 None 时不落盘、行为不变 ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(_REPORT_LINES) + "\n", encoding="utf-8")
        print(f"\n[--out] 以上报告已同时归档 → {out_path}（{len(_REPORT_LINES)} 条 emit，utf-8）")


if __name__ == "__main__":
    main()
