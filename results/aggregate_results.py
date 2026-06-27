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

前置条件: run_matrix 跑完、各评测产物落地。
预期结果: results/out/{master.csv, main_table.md, streaming_table.md, reward_hacking.md}。
本机可语法检查；纯文件处理（有产物时可离线跑）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser(description="结果汇总")
    ap.add_argument("--ledger", default=str(PATHS.project_root / "ablation" / "ledger.jsonl"))
    args = ap.parse_args()

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
            "first_packet_ms_p50": (st.get("first_packet_ms") or {}).get("p50"),
            "boundary_discont": st.get("boundary_discontinuity_p95_flux"),
            "ckpt": ckpt,  # 溯源（ledger 外的臂无 ckpt → 标 '(ledger 外)'）
        })

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
        ["treatment", "NFE", "n", "WER↓", "SIM↑", "UTMOS↑", "EmergentTTS↑", "失败率↓"],
        [[r["treatment"], r["nfe"], r["n"], _fmt(r["wer"], r["wer_ci"]),
          _fmt(r["sim"], r["sim_ci"]), _fmt(r["utmos"], r["utmos_ci"], 3),
          _fmt(r["emergent_overall"], None, 3),
          _fmt_pct(r["failure_rate"], r["n_failed"], r["n_total"])] for r in master_rows])
    (RESULTS / "main_table.md").write_text(
        "# Table 1 主表（离线；[..]为 bootstrap 95% CI）\n\n" + main_tbl +
        "\n> 小样本评测请以 CI 重叠与否判断显著性，勿只看均值。\n"
        "> 失败率↓ = 生成崩坏占比(n_failed/n_total，来自各臂 inference/out/<tag>/_synthesis_report.json)；"
        "空=未跑或旧账本无 offline_dir。**须与 WER 并读**——崩坏臂(如 dpo_multiobj)对部分 prompt 立即 EOS "
        "产不出 wav，这些样本不进 WER，单看 WER 会漏判 active degradation。\n", encoding="utf-8")

    # Table 2 流式表
    st_tbl = _md_table(
        ["treatment", "NFE", "流式WER↓", "流式SIM↑", "首包 p50(ms)↓", "边界不连续度↓"],
        [[r["treatment"], r["nfe"], _fmt(r["stream_wer"]), _fmt(r["stream_sim"]),
          _fmt(r["first_packet_ms_p50"], None, 1), _fmt(r["boundary_discont"], None, 4)]
         for r in master_rows])
    (RESULTS / "streaming_table.md").write_text("# Table 2 流式表\n\n" + st_tbl, encoding="utf-8")

    # Table 3 reward-hacking 对照
    hack_rows = [r for r in master_rows if r["treatment"] in ("dpo_multiobj", "dpo_weronly", "base")]
    hack_tbl = _md_table(
        ["treatment", "WER↓", "UTMOS↑", "EmergentTTS↑"],
        [[r["treatment"], _fmt(r["wer"], r["wer_ci"]), _fmt(r["utmos"], r["utmos_ci"], 3),
          _fmt(r["emergent_overall"], None, 3)] for r in hack_rows])
    (RESULTS / "reward_hacking.md").write_text(
        "# Table 3 reward hacking 对照\n\n" + hack_tbl +
        "\n> 判读(PROTOCOL §6)：dpo_weronly 若 **WER 最低但 UTMOS/EmergentTTS 退化** = 复现 reward hacking；"
        "若仅WER未伤自然度→多目标前提不成立，如实报告。\n", encoding="utf-8")

    print(f"✓ 汇总 {len(master_rows)} 个 run -> results/out/{{master.csv, main_table.md, streaming_table.md, reward_hacking.md}}")


if __name__ == "__main__":
    main()
