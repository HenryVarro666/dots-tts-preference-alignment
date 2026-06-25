"""
aggregate_results.py — 阶段⑨：汇总所有 run 的指标 → 主表/流式表/reward-hacking 表

做什么:
    读 ablation/ledger.jsonl 与各评测产物(WER/SIM/UTMOS/EmergentTTS/streaming)，
    汇成 master.csv + 三张 markdown 表(对应 PAPER_SKELETON 的 Table 1/2/3)，
    每个数字标注来源 run（可溯源 ledger）。
怎么做:
    纯标准库；把分散的 *.utmos.jsonl / *.streaming.json / seed-tts-eval 输出/emergent.json 按 tag 归并。
为什么:    把「散落的数字」收敛成「能直接放进论文的表」，且每格可追到 run。
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402

RESULTS = PATHS.results_dir


def _load_json(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _fmt(v, ci=None, nd=4):
    """数字 + 可选 CI（小样本评测应连 CI 一起看）。"""
    if v is None:
        return ""
    s = f"{v:.{nd}f}"
    if ci and ci[0] is not None:
        s += f" [{ci[0]:.{nd}f},{ci[1]:.{nd}f}]"
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
    master_rows = []
    for run in ledger:
        tag = run["tag"]
        off = _load_json(f"{tag}.summary.json")          # score_synth 离线 WER/SIM/UTMOS(+CI)
        stm = _load_json(f"{tag}_stream.summary.json")   # score_synth 流式 WER/SIM
        st = _load_json(f"{tag}.streaming.json")         # streaming_metrics 首包/边界
        em = _load_json(f"{tag}.emergent.json")          # EmergentTTS(可选)
        master_rows.append({
            "treatment": run["treatment"], "tag": tag, "nfe": run["nfe"], "n": off.get("n"),
            "wer": off.get("wer"), "wer_ci": off.get("wer_ci"),
            "sim": off.get("sim"), "sim_ci": off.get("sim_ci"),
            "utmos": off.get("utmos"), "utmos_ci": off.get("utmos_ci"),
            "emergent_overall": em.get("overall"),
            "stream_wer": stm.get("wer"), "stream_sim": stm.get("sim"),
            "first_packet_ms_p50": (st.get("first_packet_ms") or {}).get("p50"),
            "boundary_discont": st.get("boundary_discontinuity_p95_flux"),
            "ckpt": run["ckpt"],  # 溯源
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    if master_rows:
        flat = [{k: (v if not isinstance(v, (list, tuple)) else json.dumps(v)) for k, v in r.items()}
                for r in master_rows]
        with (RESULTS / "master.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)

    # Table 1 主表（离线，带 CI）
    main_tbl = _md_table(
        ["treatment", "NFE", "n", "WER↓", "SIM↑", "UTMOS↑", "EmergentTTS↑"],
        [[r["treatment"], r["nfe"], r["n"], _fmt(r["wer"], r["wer_ci"]),
          _fmt(r["sim"], r["sim_ci"]), _fmt(r["utmos"], r["utmos_ci"], 3),
          _fmt(r["emergent_overall"], None, 3)] for r in master_rows])
    (RESULTS / "main_table.md").write_text(
        "# Table 1 主表（离线；[..]为 bootstrap 95% CI）\n\n" + main_tbl +
        "\n> 小样本评测请以 CI 重叠与否判断显著性，勿只看均值。\n", encoding="utf-8")

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
