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


def _load_utmos(tag: str) -> float | None:
    p = RESULTS / f"{tag}.utmos.jsonl"
    if not p.exists():
        return None
    xs = [r["utmos"] for r in read_jsonl(p)]
    return sum(xs) / len(xs) if xs else None


def _load_streaming(tag: str) -> dict:
    p = RESULTS / f"{tag}.streaming.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_emergent(tag: str) -> dict:
    p = RESULTS / f"{tag}.emergent.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


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
        utmos = _load_utmos(tag)
        st = _load_streaming(tag)
        em = _load_emergent(tag)
        master_rows.append({
            "treatment": run["treatment"], "tag": tag, "nfe": run["nfe"],
            # WER/SIM 由 seed-tts-eval 终端输出，用户回填这两列（或解析其输出文件）
            "wer": run.get("wer"), "sim": run.get("sim"), "utmos": utmos,
            "emergent_overall": em.get("overall"),
            "first_packet_ms_p50": (st.get("first_packet_ms") or {}).get("p50"),
            "boundary_discont": st.get("boundary_discontinuity_p95_flux"),
            "ckpt": run["ckpt"],  # 溯源
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    # master.csv
    if master_rows:
        with (RESULTS / "master.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
            w.writeheader()
            w.writerows(master_rows)

    # Table 1 主表（离线）
    main_tbl = _md_table(
        ["treatment", "NFE", "WER↓", "SIM↑", "UTMOS↑", "EmergentTTS↑"],
        [[r["treatment"], r["nfe"], r["wer"], r["sim"], r["utmos"], r["emergent_overall"]]
         for r in master_rows])
    (RESULTS / "main_table.md").write_text("# Table 1 主表（离线）\n\n" + main_tbl, encoding="utf-8")

    # Table 2 流式表
    st_tbl = _md_table(
        ["treatment", "NFE", "首包 p50(ms)↓", "边界不连续度↓"],
        [[r["treatment"], r["nfe"], r["first_packet_ms_p50"], r["boundary_discont"]]
         for r in master_rows])
    (RESULTS / "streaming_table.md").write_text("# Table 2 流式表\n\n" + st_tbl, encoding="utf-8")

    # Table 3 reward-hacking 对照（突出 WER↓ 但 UTMOS/EmergentTTS↓ 的反差）
    hack_rows = [r for r in master_rows if r["treatment"] in ("dpo_multiobj", "dpo_weronly")]
    hack_tbl = _md_table(
        ["treatment", "WER↓", "UTMOS↑", "EmergentTTS↑", "判读"],
        [[r["treatment"], r["wer"], r["utmos"], r["emergent_overall"],
          "仅WER：看是否 WER 最低但 UTMOS/Emergent 退化"] for r in hack_rows])
    (RESULTS / "reward_hacking.md").write_text(
        "# Table 3 reward hacking 对照\n\n" + hack_tbl +
        "\n> 结论判读见 PROTOCOL.md 第 6 节；若仅WER未退化自然度→多目标前提不成立，如实报告。\n",
        encoding="utf-8")

    print(f"✓ 汇总 {len(master_rows)} 个 run -> results/out/{{master.csv,main_table.md,streaming_table.md,reward_hacking.md}}")
    print("  注: WER/SIM 两列需从 seed-tts-eval 输出回填(或扩展本脚本解析其输出文件)")


if __name__ == "__main__":
    main()
