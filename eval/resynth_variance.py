"""
resynth_variance.py — E3：再合成方差（确认 open-SOAR 唯一正面结果是否清出 base 的再合成噪声带）

背景（模拟审稿 CRITICAL #4）:
    dots.tts 的 generate() 无 seed → 每次再合成的绝对指标都会漂（base WER 见过 0.103/0.128/0.112）。
    bootstrap CI 只覆盖「句子抽样」、不覆盖「再合成」方差。open_soar 的 WER 0.094 vs base 0.112 可能就落在
    base 的再合成漂移带内 → `.tex` 已先软化为「不退化」；**跑出方差带才能（可能）升回「真增益」**。

做什么（仅推理、无训练）:
    对每个臂（默认 base, open_soar）独立**再合成 K 次**（无 seed → 每次不同），每次 score 出 WER/UTMOS，
    报每臂跨 K 次的均值 ± SD；构造 base 的再合成带，判 open_soar 是否清出该带。
怎么做:
    subprocess 串既有 `inference/batch_synthesize.py` + `eval/score_synth.py`（与 ablation/run_matrix 同款编排），
    读各次 `results/out/{tag}.summary.json` 的 wer/utmos 汇总。不重复实现合成/评测。

判据（回填 §5.1 / Table 1 脚注）:
    open_soar 的 WER/UTMOS 均值**清出** base 再合成带（mean ± 1SD）⇒ 升级为「真增益」；否则维持「不退化」（已诚实）。

前置: 各臂对齐 ckpt（base 用 dots.tts-base，open_soar 用 alignment/out/open_soar）+ 评测 manifest + 权重；GPU。
用法: python eval/resynth_variance.py --manifest data/prompts/eval_big.jsonl --arms base open_soar --reps 5 --nfe 10 --out results/out/e3_report.txt
⚠️ 本机只语法检查；真实运行需 CUDA + 权重。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402

PROJ = PATHS.project_root

# 臂 → ckpt（与 ablation/run_matrix.TREATMENTS 同款；base 是基模，其余是对齐 ckpt 目录）
CKPTS = {
    "base": "rednote-hilab/dots.tts-base",
    "open_soar": str(PROJ / "alignment" / "out" / "open_soar"),
    "sft": str(PROJ / "alignment" / "out" / "sft"),
    "dpo_multiobj": str(PROJ / "alignment" / "out" / "dpo_multiobj"),
    "dpo_weronly": str(PROJ / "alignment" / "out" / "dpo_weronly"),
    "reward_weighted": str(PROJ / "alignment" / "out" / "reward_weighted"),
}


def _mean_sd(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
    return m, sd


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 再合成方差")
    ap.add_argument("--manifest", default="data/prompts/eval_big.jsonl")
    ap.add_argument("--arms", nargs="+", default=["base", "open_soar"])
    ap.add_argument("--reps", type=int, default=5, help="每臂独立再合成次数（无 seed → 每次不同）")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--sim-backend", default="ecapa")
    ap.add_argument("--ref", default="base", help="构造再合成带的对照臂")
    ap.add_argument("--out", default=None, help="报告归档路径")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令、不真正合成/评测")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    results: dict[str, dict] = {}
    for arm in args.arms:
        ckpt = CKPTS.get(arm, str(PROJ / "alignment" / "out" / arm))
        wl, ul, nl = [], [], []
        for k in range(args.reps):
            tag = f"resynth_{arm}_k{k}"
            sdir = PROJ / "inference" / "out" / tag
            cmd_syn = ["python", str(PROJ / "inference" / "batch_synthesize.py"),
                       "--ckpt", ckpt, "--manifest", args.manifest, "--treatment", tag,
                       "--num-steps", str(args.nfe), "--out", str(sdir)]
            cmd_score = ["python", str(PROJ / "eval" / "score_synth.py"),
                         "--synth-dir", str(sdir), "--manifest", args.manifest,
                         "--treatment", tag, "--sim-backend", args.sim_backend]
            print("  $", " ".join(cmd_syn))
            print("  $", " ".join(cmd_score))
            if args.dry_run:
                continue
            subprocess.run(cmd_syn, check=True)
            subprocess.run(cmd_score, check=True)
            s = json.loads((PATHS.results_dir / f"{tag}.summary.json").read_text(encoding="utf-8"))
            wl.append(float(s["wer"]))
            ul.append(float(s["utmos"]))
            nl.append(int(s["n"]))
            emit(f"  [{arm} k{k}] WER={s['wer']:.4f}  UTMOS={s['utmos']:.3f}  n={s['n']}")
        results[arm] = {"wer": wl, "utmos": ul, "n": nl}

    if args.dry_run:
        emit("(dry-run：未真正合成/评测，仅打印命令)")
    else:
        emit(f"\n==== [E3] 再合成方差  reps={args.reps}  nfe={args.nfe} ====\n")
        for arm in args.arms:
            wm, ws = _mean_sd(results[arm]["wer"])
            um, us = _mean_sd(results[arm]["utmos"])
            emit(f"  {arm:14s} WER {wm:.4f} ± {ws:.4f}   UTMOS {um:.3f} ± {us:.3f}   (n/次={results[arm]['n']})")
        if args.ref in results and len(results[args.ref]["wer"]) >= 2:
            rw_m, rw_s = _mean_sd(results[args.ref]["wer"])
            ru_m, ru_s = _mean_sd(results[args.ref]["utmos"])
            emit(f"\n  {args.ref} 再合成带(±1SD): WER [{rw_m - rw_s:.4f}, {rw_m + rw_s:.4f}]   "
                 f"UTMOS [{ru_m - ru_s:.3f}, {ru_m + ru_s:.3f}]")
            for arm in args.arms:
                if arm == args.ref:
                    continue
                am, _ = _mean_sd(results[arm]["wer"])
                aum, _ = _mean_sd(results[arm]["utmos"])
                wer_clears = am < rw_m - rw_s   # WER 更低、清出带下界
                utm_clears = aum > ru_m + ru_s  # UTMOS 更高、清出带上界
                verdict = "真增益(至少一轴清出 base 带)" if (wer_clears or utm_clears) else "维持「不退化」(未清出 base 再合成带)"
                emit(f"  {arm}: WER 均值 {am:.4f} {'清出↓' if wer_clears else '落在带内'}；"
                     f"UTMOS 均值 {aum:.3f} {'清出↑' if utm_clears else '落在带内'}  → {verdict}")
        emit("\n==== [E3 裁决] open_soar 均值清出 base 再合成带 ⇒ §5.1 升级为真增益；否则维持「不退化」(已诚实)。 ====")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[--out] 报告已归档 → {out}")


if __name__ == "__main__":
    main()
