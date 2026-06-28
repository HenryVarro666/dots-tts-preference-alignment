"""
analyze_collapse_blocks.py — E2：崩坏臂的分块 + 最坏情况 EOS 计分（回应 EMNLP 审稿 collapse 混淆 / n=75 vs n=100）

背景（审稿点）:
    EIC + methodology + Devil's Advocate 指出 dpo_multiobj 的「崩坏」可能是 (i) LJ002 块集中、(ii) 丢弃了
    25 条 immediate-EOS 失败句的伪影；且 n=75(成功句) 与 base n=100 **不可比**，会低估/误判退化。

做什么（纯 CPU 分析、无需 GPU、无需新合成）:
    读 results/out/{arm}.metrics.jsonl（score_synth.py 落盘的逐句 {fid,wer,sim,utmos}）+ 评测 manifest（全 n fid），
    对每个臂：
      (a) scored-only：只算有音频的句（= 当前 n=75 口径）；
      (b) worst-case full-n：把缺失的 EOS 失败句以**最坏情况**(WER=1.0 / UTMOS=floor)补成 full n，得可与 base 比的数字；
      (c) apples-to-apples：把对照臂(base)限制到该崩坏臂打分的**同一子集**重算，n 一致；
      (d) 分块：按 block(LJ001/LJ002…)分组报 WER/UTMOS/scored/EOS-fail，看崩坏是跨 block 均匀还是仅集中某块。

判据（写回 §5.1 / Table 1）:
    worst-case n=100 仍崩（WER ≫ base）且跨 block 均匀 ⇒ collapse 稳健、非「LJ002+丢句」伪影（可保留结论但仍标 cautionary）；
    若只在 LJ002 ⇒ 改述为 block-specific、扩 n 或重排评测集。

前置: 各臂的 results/out/{tag}.metrics.jsonl（由 eval/score_synth.py 产出）+ 评测 manifest。
用法: python eval/analyze_collapse_blocks.py --manifest <eval.jsonl> --arms base_nfe10 sft_nfe10 dpo_multiobj_nfe10 --out results/out/e2_report.txt
本机只语法检查；在有 metrics.jsonl 的机器上运行（CPU 即可）。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402


def _block(fid: str) -> str:
    """LJSpeech fid 'LJ001-0001' -> block 'LJ001'（首个 '-' 前）。无 '-' 则原样返回。"""
    return fid.rsplit("-", 1)[0] if "-" in fid else fid


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 崩坏臂分块 + 最坏情况 EOS 计分")
    ap.add_argument("--manifest", required=True, help="评测 manifest jsonl（全 n fid，含 fid）")
    ap.add_argument("--metrics-dir", default=str(PATHS.results_dir))
    ap.add_argument("--arms", nargs="+", default=["base_nfe10", "sft_nfe10", "dpo_multiobj_nfe10"],
                    help="要分析的臂 tag（对应 results/out/{tag}.metrics.jsonl）")
    ap.add_argument("--ref", default="base_nfe10", help="apples-to-apples 对照臂（在崩坏臂同子集上重算）")
    ap.add_argument("--wer-worst", type=float, default=1.0, help="EOS 失败句的最坏 WER（无音频=全删=1.0）")
    ap.add_argument("--utmos-floor", type=float, default=1.0, help="EOS 失败句的最坏 UTMOS（量表下限）")
    ap.add_argument("--out", default=None, help="报告归档路径（utf-8）；默认只打 stdout")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    all_fids = [str(r["fid"]) for r in read_jsonl(Path(args.manifest))]
    arm_m: dict[str, dict] = {}
    for arm in args.arms:
        p = Path(args.metrics_dir) / f"{arm}.metrics.jsonl"
        if not p.exists():
            emit(f"⚠️ 缺 {p}（先跑 eval/score_synth.py 产出 metrics.jsonl）")
            continue
        arm_m[arm] = {str(r["fid"]): r for r in read_jsonl(p)}
    if not arm_m:
        raise SystemExit("无任何 metrics.jsonl —— 检查 --metrics-dir / --arms")

    ref = args.ref if args.ref in arm_m else None
    emit(f"==== [E2] 崩坏臂分块 + 最坏情况 EOS  (manifest n={len(all_fids)}) ====\n")

    for arm, m in arm_m.items():
        scored = [f for f in all_fids if f in m]
        missing = [f for f in all_fids if f not in m]   # = immediate-EOS 失败句
        wer_s = [float(m[f]["wer"]) for f in scored]
        utm_s = [float(m[f]["utmos"]) for f in scored]
        emit(f"# {arm}:  scored n={len(scored)}   EOS-fail(missing) n={len(missing)}")
        emit(f"  (a) scored-only:        WER={_mean(wer_s):.3f}  UTMOS={_mean(utm_s):.3f}   (n={len(scored)}; = 当前 n={len(scored)} 口径)")
        wer_wc = wer_s + [args.wer_worst] * len(missing)
        utm_wc = utm_s + [args.utmos_floor] * len(missing)
        emit(f"  (b) worst-case n={len(all_fids)}:    WER={_mean(wer_wc):.3f}  UTMOS={_mean(utm_wc):.3f}   "
             f"(EOS 失败补 WER={args.wer_worst}/UTMOS={args.utmos_floor}; 与 base n={len(all_fids)} 可比)")
        if ref and ref != arm and missing:
            rm = arm_m[ref]
            sub = [f for f in scored if f in rm]
            emit(f"  (c) {ref} 在同 {len(sub)} 子集:  WER={_mean([float(rm[f]['wer']) for f in sub]):.3f}  "
                 f"UTMOS={_mean([float(rm[f]['utmos']) for f in sub]):.3f}   (apples-to-apples，n 一致)")
        by_w: dict[str, list] = defaultdict(list)
        by_u: dict[str, list] = defaultdict(list)
        by_fail: dict[str, int] = defaultdict(int)
        for f in all_fids:
            b = _block(f)
            if f in m:
                by_w[b].append(float(m[f]["wer"]))
                by_u[b].append(float(m[f]["utmos"]))
            else:
                by_fail[b] += 1
        emit("  (d) 分块  block / WER / UTMOS / scored / EOS-fail:")
        for b in sorted(set(list(by_w) + list(by_fail))):
            emit(f"        {b:10s} WER={_mean(by_w[b]):.3f}  UTMOS={_mean(by_u[b]):.3f}  "
                 f"n={len(by_w[b])}  fail={by_fail[b]}")
        emit()

    emit("==== [E2 裁决] worst-case n=100 仍崩（WER ≫ base）且跨 block 均匀 ⇒ collapse 稳健、非『LJ002+丢句』伪影；")
    emit("     若只集中 LJ002 ⇒ §5.1 改述 block-specific、扩 n 或重排评测集。apples-to-apples 子集让 n 可比，正面回应 n=75 vs n=100。 ====")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[--out] 报告已归档 → {out}")


if __name__ == "__main__":
    main()
