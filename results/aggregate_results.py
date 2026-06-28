"""
aggregate_results.py — 阶段⑨：汇总所有 run 的指标 → 主表/流式表/reward-hacking 表

做什么:
    读 ablation/ledger.jsonl 与各评测产物(WER/SIM/UTMOS/EmergentTTS/streaming)，
    汇成 master.csv + 三张 markdown 表(对应 PAPER_SKELETON 的 Table 1/2/3)，
    每个数字标注来源 run（可溯源 ledger）。
怎么做:
    纯标准库；把分散的 *.utmos.jsonl / *.streaming.json / seed-tts-eval 输出/emergent.json 按 tag 归并。
    要汇总的 tag 集合 = ledger 条目 ∪ results/out/ 下所有 *.summary.json 对应 tag(按 tag 去重、稳定顺序)——
    累积读全部 summary，避免主表被最近一次 run_matrix 的 ledger 覆盖。
    并从每个 run 的 ledger.offline_dir 读 _synthesis_report.json，把 failure_rate(生成崩坏占比)
    贯通进主表——它与 WER 同源(同一离线合成)但是另一维度的硬指标。
为什么:    把「散落的数字」收敛成「能直接放进论文的表」，且每格可追到 run。
    失败率必须与 WER 并列：崩坏臂(dpo_multiobj)可能对某些 prompt 立即 EOS、根本产不出 wav，
    这些样本不进 WER(无音频可转写)，只看 WER 会**漏掉 active degradation**——失败率把它显式化。
对实验意义:
    主表回答 RQ1；流式表回答 RQ2；reward-hacking 表回答 RQ3——三表即论文核心结果。
    多重比较校正(Part B5): 对各表「各臂 vs base」差异族加 FDR/Bonferroni 校正标记——
    **消融网格(本三表 >100 个 臂×指标 对照)才是多重比较校正的正场**；probe 的偏相关是
    单一预设确证终点(single pre-registered endpoint)，不进此校正族。

前置条件: run_matrix 跑完、各评测产物落地。
预期结果: results/out/{master.csv, main_table.md, streaming_table.md, reward_hacking.md}。
本机可语法检查；纯文件处理（有产物时可离线跑）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, file_lock  # noqa: E402

RESULTS = PATHS.results_dir

_NFE_RE = re.compile(r"_nfe(\d+)")


def _parse_nfe(tag: str) -> int | None:
    """从 tag 名 *_nfeN 解析 NFE（供 ledger 外、只有 summary.json 的臂补字段）。"""
    m = _NFE_RE.search(tag)
    return int(m.group(1)) if m else None


def _treatment_from_tag(tag: str) -> str:
    """ledger 外的臂没有 treatment，从 tag 去掉 _nfeN 后缀当作 treatment 名。"""
    return _NFE_RE.sub("", tag) or tag


def _discover_tags(ledger: list[dict]) -> tuple[list[str], dict]:
    """要汇总的 tag 集合 = ledger 条目 ∪ results/out/ 下所有 *.summary.json 对应 tag。

    为什么并集：只跟 ledger 会让主表被最近一次 run_matrix 覆盖——跑过又未重列在
    当前 ledger 的臂会消失。改为「累积读全部 summary」，跑过的每个臂永久进表。
    去重保序：先按 ledger 出现顺序，再补 results/out 里 ledger 外的 tag（按文件名排序，稳定）。
    排除 *_stream.summary.json（那是同一 tag 的流式产物，不是独立臂）。
    """
    order: list[str] = []
    seen: set[str] = set()
    ledger_by_tag: dict[str, dict] = {}
    for run in ledger:
        tag = run["tag"]
        ledger_by_tag.setdefault(tag, run)
        if tag not in seen:
            seen.add(tag); order.append(tag)
    for p in sorted(RESULTS.glob("*.summary.json")):
        if p.name.endswith("_stream.summary.json"):
            continue
        tag = p.name[: -len(".summary.json")]
        if tag not in seen:
            seen.add(tag); order.append(tag)
    return order, ledger_by_tag


def _load_json(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_synth_report(offline_dir: str | None) -> dict:
    """从某臂的离线合成目录读 batch_synthesize 落盘的 _synthesis_report.json。

    为什么走 ledger.offline_dir 而非 results/：失败率是**合成阶段**的产物（与 wav 同目录），
    不在 results/out/ 下；ledger 每行记了 offline_dir，照它取即可逐臂对上。
    无报告(未跑/旧账本无 offline_dir)→ 返回空 dict，对应表格列留空、绝不报错。
    """
    if not offline_dir:
        return {}
    p = Path(offline_dir) / "_synthesis_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _fmt(v, ci=None, nd=4):
    """数字 + 可选 CI（小样本评测应连 CI 一起看）。"""
    if v is None:
        return ""
    s = f"{v:.{nd}f}"
    if ci and ci[0] is not None:
        s += f" [{ci[0]:.{nd}f},{ci[1]:.{nd}f}]"
    return s


def _fmt_pct(v, n_failed=None, n_total=None, nd=1):
    """失败率按百分比显示（崩坏臂如 dpo_multiobj≈25% 一眼可见）；无报告则留空。

    带上 n_failed/n_total 计数（如 25.0% (25/100)），让读者知道分母——
    5/20 与 25/100 都是 25% 但置信度天差地别，光给比率会误导。
    """
    if v is None:
        return ""
    s = f"{v * 100:.{nd}f}%"
    if n_failed is not None and n_total:
        s += f" ({n_failed}/{n_total})"
    return s


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if x is None else str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


# —— Part B5：消融网格的多重比较校正（reviewers 实指的 FDR 正场）——
# 做什么:    对「各臂 vs base」×{WER,SIM,UTMOS} 这一显著性族，按 Bonferroni / Benjamini-Hochberg(FDR)
#            标注哪些差异在多重比较校正后仍成立；**只增标记列，不改任何已存 CI/指标数字**。
# 为什么:    >100-CI 的消融网格才是多重比较校正的正场——成百个「臂×指标 vs base」对照同族检验、
#            不校正会虚报显著；而 probe 的偏相关是单一预设确证终点(single pre-registered endpoint)，不在此族。
_Z95 = 1.959963984540054  # 双尾 95% 正态分位数：从 bootstrap 95% CI 反解 SE 用


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _approx_p_vs_base(point, ci, base_value):
    """近似「该臂某指标 ≠ base」的双尾 p。

    为什么近似: master 只存了 95% bootstrap CI、没存 bootstrap 重抽样，无法在更严的 Bonferroni α
    上精确重算 CI；故按高斯/Wald 把 95% CI 反解出 SE——取「朝向 base 一侧」的半宽(对非对称
    percentile CI 更忠实地对应『CI 是否排除 base』)。这是偏保守的近似、非精确重算(精确版需重跑
    bootstrap)。缺 point/ci/base 或 SE≤0 → 返回 None（不入校正族）。
    """
    if point is None or base_value is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    lo, hi = ci
    if lo is None or hi is None:
        return None
    diff = point - base_value
    if diff == 0:
        return 1.0
    half = (point - lo) if diff > 0 else (hi - point)  # base 落在 point 的哪一侧 → 取该侧半宽
    se = half / _Z95
    if se <= 0:
        return None
    z = abs(diff) / se
    return 2.0 * (1.0 - _norm_cdf(z))


def _benjamini_hochberg(pvals: dict, q: float = 0.05) -> set:
    """BH(FDR) step-up：返回在 FDR=q 下显著的 key 集合。pvals: {key: p}（已剔 None）。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    cut = 0
    for i, (_k, p) in enumerate(items, start=1):
        if p <= (i / m) * q:
            cut = i  # 最大的 rank 满足 p<=(i/m)q → 其下全显著（step-up）
    return {items[i][0] for i in range(cut)}


def _mc_cell(levels: dict) -> str:
    """主表「vs base 校正」单元：列出校正后仍显著的指标。†=Bonferroni(最严)、*=FDR、无=不显著。"""
    if levels.get("wer") == "base":
        return "—(参照)"
    name = {"wer": "WER", "sim": "SIM", "utmos": "UTMOS"}
    marks, tested = [], False
    for mk in ("wer", "sim", "utmos"):
        lvl = levels.get(mk, "")
        if lvl in ("bonf", "fdr", "ns"):
            tested = True
        if lvl == "bonf":
            marks.append(name[mk] + "†")
        elif lvl == "fdr":
            marks.append(name[mk] + "*")
    if not tested:
        return "(无CI)"
    return ", ".join(marks) if marks else "无"


def main() -> None:
    ap = argparse.ArgumentParser(description="结果汇总")
    ap.add_argument("--ledger", default=str(PATHS.project_root / "ablation" / "ledger.jsonl"))
    args = ap.parse_args()
    with file_lock():  # 串行化「读账本 + 全 summary → 写主表」整段，防并发 aggregate 互相覆盖（无需手动末尾再 aggregate）
        _aggregate(args)


def _aggregate(args) -> None:
    ledger = list(read_jsonl(args.ledger)) if Path(args.ledger).exists() else []
    tags, ledger_by_tag = _discover_tags(ledger)  # ledger 条目 ∪ results/out 全部 summary 的并集
    master_rows = []
    for tag in tags:
        run = ledger_by_tag.get(tag)
        off = _load_json(f"{tag}.summary.json")          # score_synth 离线 WER/SIM/UTMOS(+CI)
        stm = _load_json(f"{tag}_stream.summary.json")   # score_synth 流式 WER/SIM
        st = _load_json(f"{tag}.streaming.json")         # streaming_metrics 首包/边界
        em = _load_json(f"{tag}.emergent.json")          # EmergentTTS(可选)
        if run is not None:
            treatment, nfe, ckpt = run["treatment"], run["nfe"], run["ckpt"]
            rep = _load_synth_report(run.get("offline_dir"))  # 离线合成失败率（崩坏硬指标）
        else:
            # ledger 外的臂：仅有 summary.json——nfe 从 *_nfeN 解析，ckpt/offline_dir 缺则标记/留空
            treatment, nfe, ckpt = _treatment_from_tag(tag), _parse_nfe(tag), "(ledger 外)"
            rep = {}
        master_rows.append({
            "treatment": treatment, "tag": tag, "nfe": nfe, "n": off.get("n"),
            "wer": off.get("wer"), "wer_ci": off.get("wer_ci"),
            "sim": off.get("sim"), "sim_ci": off.get("sim_ci"),
            "utmos": off.get("utmos"), "utmos_ci": off.get("utmos_ci"),
            "emergent_overall": em.get("overall"),
            # 失败率三件套（与 WER 并列）：rate 进主表；n_ok/n_failed 进 master.csv 供溯源分母
            "failure_rate": rep.get("failure_rate"),
            "n_ok": rep.get("n_ok"), "n_failed": rep.get("n_failed"),
            "n_total": rep.get("n_total"),
            "stream_wer": stm.get("wer"), "stream_sim": stm.get("sim"),
            "stream_utmos": stm.get("utmos"),  # 流式自然度（score_synth 已算、此前未 surface）
            "first_packet_ms_p50": (st.get("first_packet_ms") or {}).get("p50"),
            "boundary_discont": st.get("boundary_discontinuity_p95_flux"),
            "ckpt": ckpt,  # 溯源（ledger 外的臂无 ckpt → 标 '(ledger 外)'）
        })

    # —— Part B5：消融网格多重比较校正（FDR/Bonferroni）——
    # 族 = 「各臂 vs base」×{WER,SIM,UTMOS}；base 行作参照、不入族。标记写回 master.csv + 主表新列。
    base_row = next((r for r in master_rows if r["treatment"] == "base"), None)
    metric_keys = [("wer", "wer_ci"), ("sim", "sim_ci"), ("utmos", "utmos_ci")]
    base_vals = {mk: (base_row.get(mk) if base_row else None) for mk, _ in metric_keys}
    pfamily: dict[tuple[str, str], float] = {}
    for r in master_rows:
        if base_row is None or r is base_row:
            continue
        for mk, ck in metric_keys:
            p = _approx_p_vs_base(r.get(mk), r.get(ck), base_vals.get(mk))
            if p is not None:
                pfamily[(r["tag"], mk)] = p
    m_family = len(pfamily)                                # = 参与校正的(臂,指标)对数（齐全时=非base臂数×3）
    bonf_alpha = (0.05 / m_family) if m_family else None   # Bonferroni 单比较阈
    bh_sig = _benjamini_hochberg(pfamily) if m_family else set()
    mc_by_tag: dict[str, dict[str, str]] = {}              # tag -> {metric: 校正等级} 供建表
    for r in master_rows:
        is_base = base_row is not None and r is base_row
        levels: dict[str, str] = {}
        for mk, _ck in metric_keys:
            p = pfamily.get((r["tag"], mk))
            if is_base:
                lvl = "base"
            elif p is None:
                lvl = ""                                   # 该臂缺 CI、未入族
            elif bonf_alpha is not None and p < bonf_alpha:
                lvl = "bonf"                               # 过最严的 Bonferroni
            elif (r["tag"], mk) in bh_sig:
                lvl = "fdr"                                # 仅过 BH-FDR
            else:
                lvl = "ns"                                 # 校正后不显著
            r[f"{mk}_p_vs_base"] = (round(p, 6) if p is not None else None)
            r[f"{mk}_adj"] = lvl
            levels[mk] = lvl
        r["mc_family_m"] = m_family
        mc_by_tag[r["tag"]] = levels

    RESULTS.mkdir(parents=True, exist_ok=True)
    if master_rows:
        flat = [{k: (v if not isinstance(v, (list, tuple)) else json.dumps(v)) for k, v in r.items()}
                for r in master_rows]
        with (RESULTS / "master.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)

    # Table 1 主表（离线，带 CI）。新增「失败率↓」列：与 WER 同源(同一离线合成)、互补维度——
    # WER 只覆盖产出了 wav 的样本，失败率覆盖根本没产出的样本，两者并列才看得到崩坏臂全貌。
    main_tbl = _md_table(
        ["treatment", "NFE", "n", "WER↓", "SIM↑", "UTMOS↑", "EmergentTTS↑", "失败率↓", "vs base 校正"],
        [[r["treatment"], r["nfe"], r["n"], _fmt(r["wer"], r["wer_ci"]),
          _fmt(r["sim"], r["sim_ci"]), _fmt(r["utmos"], r["utmos_ci"], 3),
          _fmt(r["emergent_overall"], None, 3),
          _fmt_pct(r["failure_rate"], r["n_failed"], r["n_total"]),
          _mc_cell(mc_by_tag[r["tag"]])] for r in master_rows])
    (RESULTS / "main_table.md").write_text(
        "# Table 1 主表（离线；[..]为 bootstrap 95% CI）\n\n" + main_tbl +
        "\n> 小样本评测请以 CI 重叠与否判断显著性，勿只看均值。\n"
        "> 失败率↓ = 生成崩坏占比(n_failed/n_total，来自各臂 inference/out/<tag>/_synthesis_report.json)；"
        "空=未跑或旧账本无 offline_dir。**须与 WER 并读**——崩坏臂(如 dpo_multiobj)对部分 prompt 立即 EOS "
        "产不出 wav，这些样本不进 WER，单看 WER 会漏判 active degradation。\n" +
        f"> vs base 校正列：对「各臂 vs base」×{{WER,SIM,UTMOS}} 显著性族做多重比较校正（m={m_family}"
        "=参与校正的(臂,指标)对数，三指标齐全时=非 base 臂数×3）。"
        "†=Bonferroni(α/m，最严)仍排除 base；*=Benjamini-Hochberg FDR(q=0.05)仍显著；"
        "无=校正后不显著；(无CI)=该臂缺 CI 未入族；—(参照)=base。"
        "p 由所存 95% bootstrap CI 朝 base 一侧高斯/Wald 反解近似（精确重算需 bootstrap 重抽样，本列偏保守、非精确）。\n"
        "> **消融网格(本三表 >100 个 臂×指标 对照)才是多重比较校正的正场**；"
        "probe 的偏相关是单一预设确证终点(single pre-registered endpoint)，不进此 FDR 族。\n", encoding="utf-8")

    # Table 2 流式表
    st_tbl = _md_table(
        ["treatment", "NFE", "流式WER↓", "流式SIM↑", "流式UTMOS↑", "首包 p50(ms)↓", "边界不连续度↓"],
        [[r["treatment"], r["nfe"], _fmt(r["stream_wer"]), _fmt(r["stream_sim"]),
          _fmt(r.get("stream_utmos"), None, 3),
          _fmt(r["first_packet_ms_p50"], None, 1), _fmt(r["boundary_discont"], None, 4)]
         for r in master_rows])
    (RESULTS / "streaming_table.md").write_text("# Table 2 流式表\n\n" + st_tbl, encoding="utf-8")

    # Table 3 reward-hacking 对照
    hack_rows = [r for r in master_rows if r["treatment"] in ("dpo_multiobj", "dpo_weronly", "base")]
    hack_tbl = _md_table(
        ["treatment", "WER↓", "UTMOS↑", "EmergentTTS↑", "vs base 校正"],
        [[r["treatment"], _fmt(r["wer"], r["wer_ci"]), _fmt(r["utmos"], r["utmos_ci"], 3),
          _fmt(r["emergent_overall"], None, 3), _mc_cell(mc_by_tag[r["tag"]])] for r in hack_rows])
    (RESULTS / "reward_hacking.md").write_text(
        "# Table 3 reward hacking 对照\n\n" + hack_tbl +
        "\n> 判读(PROTOCOL §6)：dpo_weronly 若 **WER 最低但 UTMOS/EmergentTTS 退化** = 复现 reward hacking；"
        "若仅WER未伤自然度→多目标前提不成立，如实报告。\n"
        "> vs base 校正：标记取自**全消融网格**的多重比较族(同 main_table，†=Bonferroni/*=FDR/无=校正后不显著)，"
        "非本子表内重算——FDR 族是整张消融网格、非这 3 行。\n", encoding="utf-8")

    print(f"✓ 汇总 {len(master_rows)} 个 run -> results/out/{{master.csv, main_table.md, streaming_table.md, reward_hacking.md}}")


if __name__ == "__main__":
    main()
