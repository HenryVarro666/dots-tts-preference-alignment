"""
diagnose_eval.py — 崩坏归因双诊断：(a)失败是否「聚集」 + (b)崩坏是「发散」还是「graded」

做什么:
    把审稿人(DA/EIC)对 dpo_multiobj「25% 崩坏」追问的两个定性问题，用**已落盘数据**一次算清：
    (a) LJ002 聚集诊断：失败是不是非均匀地堆在某些 prompt 组(LJ001/LJ002/...)、或堆在文件后段/长文本组？
        若聚集 → 崩坏不是「全局均匀退化 25%」，而是「某子集 prompt 整段崩」，结论措辞必须改。
    (b) 发散 vs graded 诊断：读训练曲线 metrics.jsonl 末段(fm_w/reward_margin 是否二次爆)
        + probe_per_pair.csv 的 reward_gap×mc_margin 交叉表，定性判 25% 是 optimization divergence
        还是 graded degradation —— 这是 DA caveat 的裁决点。
怎么做（纯标准库、CPU、秒级、可反复跑）:
    (a) 读 --eval-manifest(data/prompts/eval_big.jsonl) → 按 fid 前缀分组，统计每组
        条数 / 平均 text 长度 / 在文件中的行号区间 → join --report 的 failures(逐条 fid)
        → 算每组失败数+组内失败率 → 看失败是否集中在「靠后/长文本」组。
    (b) 读 --metrics(alignment/out/<arm>/metrics.jsonl) 末段，按分位窗看 fm_w / reward_margin
        的早→晚走势(是否持续上爬=二次爆)；读 --csv(probe_per_pair.csv) 把 reward_gap 按三分位、
        mc_margin 按符号交叉成表 → 看「外部奖励差大」的对是否对应「FM margin 也为正」。
    命令:
        python eval/diagnose_eval.py \
            --eval-manifest data/prompts/eval_big.jsonl \
            --report inference/out/dpo_multiobj_nfe10/_synthesis_report.json \
            --metrics alignment/out/dpo_multiobj/metrics.jsonl \
            --csv alignment/out/probe_per_pair.csv
    四个输入各自独立：缺哪个就跳过对应小节并提示，绝不报错（可只跑 (a) 或只跑 (b)）。
    输入: 上述四文件(均为前序阶段已落盘产物，schema 不改)；输出: 终端诊断报告(默认不落盘新文件)。
    可选 --out: 把这份终端诊断报告原样同时落一份文本文件(默认 None=只打 stdout，行为与原先一致)。
为什么这么设计:
    崩坏的「是什么」(失败率，见 results/aggregate_results 主表)已有；这里回答「为什么/怎么崩」。
    把 GPU 探针(probe_fm_reward 落盘 csv、训练落盘 metrics.jsonl、合成落盘 report)与**纯统计归因**解耦：
    探针/训练跑一次，归因可反复跑，且只读不写、不碰任何既有 JSONL/CSV schema。
对实验与 claim 的意义:
    • 失败率是崩坏硬指标，须与 WER 并列(已在主表)；本脚本进一步把它**归因**——
    • LJ002 聚集若成立 → 必须写「非均匀崩坏(某 prompt 组整段崩)」而非「均匀退化」，否则误导；
    • 区分 divergence vs graded 直接解 DA(Devil's Advocate) caveat：
        发散(fm_w 与 margin 二次爆) ⇒ 是真 active degradation / 优化失稳；
        graded(曲线平台、reward_gap×mc_margin 仍同向) ⇒ 是温和退化，不能写成「灾难性发散」。

前置条件: 至少有 eval_big.jsonl（(a) 的结构部分）；要 join 失败需某臂 _synthesis_report.json；
          (b) 需该臂 alignment/out/<arm>/metrics.jsonl 与 probe_per_pair.csv。
预期结果: 终端打印两段诊断 + 各自定性判语；无任何文件写出。
⚠️ 本机只语法检查（py_compile）；真实运行在有上述产物的服务器上。
"""

from __future__ import annotations

import argparse
import builtins
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402

_PROJ = PATHS.project_root


# --------------------------------------------------------------------------- #
# 报告 tee：所有 print 既照常打 stdout、又把同一份文本收进 _REPORT_BUF，
# 供 main() 末尾在给了 --out 时整份落盘（默认 None=不写文件，与原行为一致）。
# 用覆盖内建 print 的方式实现，既有打印点零改动即自动 tee。
# --------------------------------------------------------------------------- #
_REPORT_BUF: list[str] = []


def print(*args, **kwargs):  # noqa: A001  覆盖内建 print 以 tee 进 buffer
    builtins.print(*args, **kwargs)
    sep, end = kwargs.get("sep", " "), kwargs.get("end", "\n")
    _REPORT_BUF.append(sep.join(str(a) for a in args) + end)


# --------------------------------------------------------------------------- #
# 分组键：LJSpeech fid 形如 "LJ002-0123" → 组前缀 "LJ002"。
# 兜底：无 "LJxxx" 模式时取 "-" 前一段；再兜底用整 fid（每条自成一组，至少不崩）。
# --------------------------------------------------------------------------- #
def _group_key(fid: str) -> str:
    m = re.match(r"([A-Za-z]+\d+)", str(fid))
    if m:
        return m.group(1)
    return str(fid).split("-")[0]


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# =========================================================================== #
# 诊断 (a): LJ002 聚集 —— 失败是否非均匀地堆在某些 prompt 组 / 文件后段 / 长文本组
# =========================================================================== #
def diagnose_clustering(manifest: Path, report: Path | None) -> None:
    print("=" * 72)
    print("诊断 (a) 失败聚集：失败是均匀分布，还是堆在某 prompt 组 / 后段 / 长文本？")
    print("=" * 72)
    if not manifest.exists():
        print(f"  ⏭  跳过：评测清单不存在 {manifest}（先跑 data/build_eval_set.py）\n")
        return

    rows = list(read_jsonl(manifest))
    if not rows:
        print(f"  ⏭  跳过：评测清单为空 {manifest}\n")
        return

    # 每条记位置(行号，0-based=合成/评测的实际处理顺序)、组、text 长度
    groups: dict[str, dict] = defaultdict(lambda: {"idx": [], "tlen": [], "fids": []})
    fid_pos: dict[str, int] = {}
    for i, r in enumerate(rows):
        fid = str(r["fid"])
        g = _group_key(fid)
        groups[g]["idx"].append(i)
        groups[g]["tlen"].append(len(r.get("text", "")))
        groups[g]["fids"].append(fid)
        fid_pos[fid] = i
    n_total = len(rows)

    # join 失败（若给了某臂 report）
    failed_fids: set[str] = set()
    report_arm = ""
    if report and report.exists():
        import json
        rep = json.loads(report.read_text(encoding="utf-8"))
        report_arm = rep.get("treatment", report.parent.name)
        failed_fids = {str(f["fid"]) for f in rep.get("failures", [])}
        print(f"  join 失败来源: {report}  臂={report_arm}  "
              f"失败 {len(failed_fids)}/{rep.get('n_total', n_total)} "
              f"(failure_rate={rep.get('failure_rate')})")
    elif report:
        print(f"  ⚠ --report 不存在({report})：只出组结构，不做失败 join")
    else:
        print("  （未给 --report：只出组结构；要看失败聚集请传某臂 _synthesis_report.json）")
    print()

    # 逐组统计表，按文件中位置排序（前→后），便于看「后段是否更易失败」
    g_sorted = sorted(groups.items(), key=lambda kv: min(kv[1]["idx"]))
    hdr = f"  {'组':8s} {'条数':>4s} {'位置区间(行)':>14s} {'平均text长':>9s} {'失败数':>5s} {'组内失败率':>9s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    per_group_fail_rate = []
    for g, d in g_sorted:
        idxs = d["idx"]
        gfails = sum(1 for fid in d["fids"] if fid in failed_fids)
        frate = gfails / len(d["fids"]) if d["fids"] else 0.0
        per_group_fail_rate.append((g, frate, _avg(d["tlen"]), _avg(idxs)))
        print(f"  {g:8s} {len(idxs):>4d} {min(idxs):>6d}-{max(idxs):<7d} "
              f"{_avg(d['tlen']):>9.1f} {gfails:>5d} {frate:>8.1%}")
    print()

    if not failed_fids:
        print("  （无失败 join，聚集判定略过；上表仅描述评测集本身的组/长度/位置分布）\n")
        return

    # ---- 聚集量化：失败是否偏向「后段」与「长文本」 ----
    fail_pos = [fid_pos[f] for f in failed_fids if f in fid_pos]
    median_pos = (n_total - 1) / 2
    n_fail_latter = sum(1 for p in fail_pos if p > median_pos)
    frac_latter = n_fail_latter / len(fail_pos) if fail_pos else 0.0

    # 文本长度：失败样本 vs 全体的平均 text 长（失败是否偏长）
    all_tlen = {str(r["fid"]): len(r.get("text", "")) for r in rows}
    fail_tlen = [all_tlen[f] for f in failed_fids if f in all_tlen]
    mean_tlen_fail = _avg(fail_tlen)
    mean_tlen_all = _avg(list(all_tlen.values()))

    # 组级集中度：失败最重的组占了多少比例的总失败（赫芬达尔式直觉）
    fails_by_group: dict[str, int] = defaultdict(int)
    for f in failed_fids:
        if f in fid_pos:
            fails_by_group[_group_key(f)] += 1
    top_group, top_n = (max(fails_by_group.items(), key=lambda kv: kv[1])
                        if fails_by_group else ("-", 0))
    top_share = top_n / len(failed_fids) if failed_fids else 0.0
    n_groups_with_fail = len(fails_by_group)

    print("  [聚集判定]")
    print(f"    · 后段集中: {frac_latter:.0%} 的失败落在文件后半段(位置> 中位{median_pos:.0f})  "
          f"{'← 偏后段(长稿/连续后段更易崩)' if frac_latter >= 0.65 else '← 未明显偏后段' if frac_latter > 0.35 else '← 偏前段'}")
    print(f"    · 长文本相关: 失败样本平均 text 长={mean_tlen_fail:.1f} vs 全体={mean_tlen_all:.1f}  "
          f"({mean_tlen_fail / mean_tlen_all if mean_tlen_all else float('nan'):.2f}×)  "
          f"{'← 失败显著偏长' if mean_tlen_all and mean_tlen_fail > 1.2 * mean_tlen_all else '← 长度无明显偏向'}")
    print(f"    · 组集中: 失败分布在 {n_groups_with_fail} 个组；最重组 {top_group} 独占 {top_share:.0%} "
          f"({top_n}/{len(failed_fids)})  {'← 高度聚集于单组(典型 LJ002 聚集)' if top_share >= 0.5 else '← 跨组较分散'}")
    clustered = (frac_latter >= 0.65) or (top_share >= 0.5) or \
                (bool(mean_tlen_all) and mean_tlen_fail > 1.2 * mean_tlen_all)
    print()
    print("  → 结论(写法约束): "
          + ("**非均匀崩坏** —— 失败聚集(后段/长文本/单组)，论文必须写成『某子集 prompt 整段崩』"
             "而非『全局均匀退化 N%』，并报聚集证据。"
             if clustered else
             "失败大体均匀散布，可按『全局退化 failure_rate%』表述（仍建议附本表佐证均匀性）。"))
    print()


# =========================================================================== #
# 诊断 (b): 发散 vs graded —— 训练曲线末段 + reward_gap×mc_margin 交叉表
# =========================================================================== #
def _window_means(xs: list[float], k: int = 4) -> list[float]:
    """把序列等分成 k 段、返回各段均值（看早→晚走势，无需画图）。"""
    if len(xs) < k:
        return [_avg(xs)] if xs else []
    step = len(xs) / k
    return [_avg(xs[int(j * step):int((j + 1) * step)]) for j in range(k)]


def _ratio(late: float, early: float) -> float:
    denom = abs(early) if abs(early) > 1e-9 else 1e-9
    return late / denom


def diagnose_divergence(metrics: Path | None, csv_path: Path | None) -> None:
    print("=" * 72)
    print("诊断 (b) 发散 vs graded：训练曲线末段二次爆? + reward_gap×mc_margin 是否同向?")
    print("=" * 72)

    verdict_curve = None   # True=发散迹象 / False=平台(graded) / None=无数据
    verdict_cross = None

    # ---- (b1) 训练曲线 metrics.jsonl：fm_w / reward_margin 早→晚走势 ----
    if metrics and metrics.exists():
        recs = list(read_jsonl(metrics))
        fm_w = [float(r["fm_w"]) for r in recs if r.get("fm_w") is not None]
        rmg = [float(r["reward_margin"]) for r in recs if r.get("reward_margin") is not None]
        n = len(recs)
        print(f"  [b1 训练曲线] {metrics}  共 {n} 步")
        if len(fm_w) >= 4 and len(rmg) >= 4:
            fm_w4 = _window_means(fm_w)
            rmg4 = _window_means(rmg)
            fm_ratio = _ratio(fm_w4[-1], fm_w4[0])
            rmg_ratio = _ratio(rmg4[-1], rmg4[0])
            # 末段是否「单调上爬」(二次爆的特征：最后一窗 = 全程峰值且高于前一窗)
            fm_rising = fm_w4[-1] >= max(fm_w4[:-1]) and fm_w4[-1] > 1.05 * fm_w4[-2]
            rmg_rising = abs(rmg4[-1]) >= max(abs(x) for x in rmg4[:-1]) and abs(rmg4[-1]) > 1.05 * abs(rmg4[-2] or 1e-9)
            print(f"    fm_w  四分窗均值(早→晚): {['%.4f' % v for v in fm_w4]}  末/早={fm_ratio:.2f}×  "
                  f"{'↑末段仍在爬' if fm_rising else '~平台/回落'}")
            print(f"    margin四分窗均值(早→晚): {['%.4f' % v for v in rmg4]}  末/早={rmg_ratio:.2f}×  "
                  f"{'↑末段仍在爬' if rmg_rising else '~平台/回落'}")
            # 发散迹象：winner 的声学 FM_loss(fm_w) 末段还在显著上爬(模型把声学搞崩去换 margin)
            #          且 margin 也持续放大 ⇒ 二次爆(优化失稳)；否则视为平台=graded。
            verdict_curve = (fm_rising and fm_ratio > 1.3) and (rmg_rising or rmg_ratio > 1.5)
            print(f"    → {'⚠ 二次爆迹象：fm_w 末段上爬 + margin 持续放大 ⇒ optimization divergence' if verdict_curve else '曲线趋于平台/回落 ⇒ 更像 graded（非发散）'}")
        else:
            print(f"    ⏭ 步数太少({n})无法看走势")
    else:
        print(f"  [b1] ⏭ 跳过：metrics.jsonl 不存在({metrics})；该臂训练曲线缺失")
    print()

    # ---- (b2) probe_per_pair.csv：reward_gap(三分位) × mc_margin(符号) 交叉表 ----
    if csv_path and csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            prows = list(csv.DictReader(f))
        if prows:
            rgap = [float(r["reward_gap"]) for r in prows]
            mcm = [float(r["mc_margin"]) for r in prows]
            n = len(prows)
            # reward_gap 按三分位切「小/中/大」（reward_gap 构造上≥0，按幅度分位）
            srt = sorted(rgap)
            q1, q2 = srt[n // 3], srt[2 * n // 3]

            def _tier(g: float) -> str:
                return "小gap" if g <= q1 else ("中gap" if g <= q2 else "大gap")

            # mc_margin 按符号：>0 = FM 认同 winner 更优（同向）；<=0 = FM 不认同（异向）
            cells: dict[tuple, int] = defaultdict(int)
            tier_pos: dict[str, list] = defaultdict(list)
            for g, m in zip(rgap, mcm):
                t = _tier(g)
                cells[(t, "mc>0(同向)" if m > 0 else "mc≤0(异向)")] += 1
                tier_pos[t].append(m)

            print(f"  [b2 交叉表] reward_gap(三分位) × mc_margin(符号)   (n={n} 对, "
                  f"gap 分位 q1={q1:+.3f} q2={q2:+.3f})")
            tiers = ["小gap", "中gap", "大gap"]
            print(f"    {'':8s} {'mc>0(同向)':>12s} {'mc≤0(异向)':>12s} {'组内 mc>0 占比':>14s}")
            for t in tiers:
                pos = cells[(t, "mc>0(同向)")]
                neg = cells[(t, "mc≤0(异向)")]
                tot = pos + neg
                share = pos / tot if tot else 0.0
                print(f"    {t:8s} {pos:>12d} {neg:>12d} {share:>13.0%}")
            # 关键对比：大gap 组的「同向占比」是否明显高于小gap 组
            def _share(t):
                p = cells[(t, "mc>0(同向)")]; nn = p + cells[(t, "mc≤0(异向)")]
                return p / nn if nn else 0.0
            s_big, s_small = _share("大gap"), _share("小gap")
            print(f"    大gap 同向占比={s_big:.0%}  vs  小gap={s_small:.0%}  (差={s_big - s_small:+.0%})")
            # 同向(graded)：外部奖励差越大、FM margin 越倾向为正 ⇒ 信号连贯、温和退化(graded)。
            # 异向/无序：大gap 也未更同向 ⇒ DPO 优化的 FM 信号与外部奖励脱钩 ⇒ 偏 divergence/hacking。
            verdict_cross = not (s_big > s_small + 0.10 and s_big >= 0.5)
            print(f"    → {'大gap 更同向、信号连贯 ⇒ 倾向 graded（非发散）' if not verdict_cross else '⚠ 大gap 未更同向、FM 信号与奖励脱钩 ⇒ 倾向 divergence/hacking'}")
        else:
            print(f"  [b2] ⏭ probe_per_pair.csv 为空({csv_path})")
    else:
        print(f"  [b2] ⏭ 跳过：probe_per_pair.csv 不存在({csv_path})；先跑 alignment/probe_fm_reward.py")
    print()

    # ---- 合判：曲线 + 交叉表 两路证据 ----
    print("  [b 合判：divergence vs graded（解 DA caveat）]")
    signals = [("训练曲线", verdict_curve), ("交叉表", verdict_cross)]
    have = [(k, v) for k, v in signals if v is not None]
    if not have:
        print("    无可用证据（metrics/csv 都缺）——补齐后再判。\n")
        return
    div_votes = sum(1 for _, v in have if v)
    if div_votes == len(have):
        concl = ("两路证据均指向 **optimization divergence**：25% 崩坏是优化失稳/active degradation，"
                 "DA 关切成立，可如实写「主动退化（发散）」。")
    elif div_votes == 0:
        concl = ("两路证据均指向 **graded degradation**：曲线平台 + 奖励/FM 信号同向，"
                 "25% 是温和退化而非发散 —— 解 DA caveat：不能写成「灾难性发散」。")
    else:
        votes = "、".join(f"{k}={'发散' if v else 'graded'}" for k, v in have)
        concl = (f"两路证据不一致({votes}) —— **定性存疑**，需人工看 tb 曲线 + 抽听失败样本再裁决；"
                 "论文先按保守口径写「退化，发散性证据不一」。")
    print("    " + concl)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="崩坏归因双诊断：(a)失败聚集 + (b)发散vs graded")
    ap.add_argument("--eval-manifest", default=str(PATHS.prompts_dir / "eval_big.jsonl"),
                    help="(a) 评测清单（按 fid 前缀分组、看位置/长度）")
    ap.add_argument("--report", default=None,
                    help="(a) 某崩坏臂的 _synthesis_report.json（join failures 看聚集）；"
                         "如 inference/out/dpo_multiobj_nfe10/_synthesis_report.json")
    ap.add_argument("--metrics", default=str(_PROJ / "alignment" / "out" / "dpo_multiobj" / "metrics.jsonl"),
                    help="(b) 该臂训练曲线 metrics.jsonl（看 fm_w/reward_margin 末段是否二次爆）")
    ap.add_argument("--csv", default=str(_PROJ / "alignment" / "out" / "probe_per_pair.csv"),
                    help="(b) probe_per_pair.csv（reward_gap×mc_margin 交叉表）")
    ap.add_argument("--out", default=None,
                    help="把上面的终端诊断报告原样同时落盘到该文本文件（默认 None=只打 stdout，行为不变）")
    args = ap.parse_args()

    diagnose_clustering(Path(args.eval_manifest),
                        Path(args.report) if args.report else None)
    diagnose_divergence(Path(args.metrics) if args.metrics else None,
                        Path(args.csv) if args.csv else None)

    print("=" * 72)
    print("说明: 本脚本只读不写、纯定性归因。失败率硬指标见 results/aggregate_results 主表；")
    print("      偏相关/intrinsic 的统计裁决见 alignment/probe_analysis.py。")

    if args.out:
        Path(args.out).write_text("".join(_REPORT_BUF), encoding="utf-8")
        print(f"\n✓ 诊断报告已同时落盘 -> {args.out}")


if __name__ == "__main__":
    main()
