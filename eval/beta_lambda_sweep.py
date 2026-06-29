"""
beta_lambda_sweep.py — ★4：DPO-FM 的 β × λ_sft 超参网格（崩坏是否对超参稳健 / anchor 能否救）

背景（Main-tier 审稿点：崩坏是不是单一超参的偶然？）:
    论文头条「向 reward 选出的 off-manifold winner 做 DPO-FM 必崩」是在一组固定 (β=0.1, λ_sft=0.1)
    上观察到的。Main track 审稿人会问：换 DPO 强度 β、换 anchor-SFT 权重 λ_sft，崩坏是否依旧？
    尤其——**加大 λ_sft（更强地把模型钉在 winner 上）能否救回**？这把「单点崩坏」升级为「在整张
    (β,λ) 网格上崩坏稳健」的更强主张；若某个角落（如大 λ_sft）反而救回，那本身也是有信息量的发现
    （= 退化可被强 anchor 缓冲，机理与「off-manifold 目标」一致）。本脚本只编排，不引入新损失。

做什么（两模式，纯逻辑；训练/合成/打分复用既有脚本）:
    --mode emit-configs : 读 base DPO 配置，对每个 (β,λ) 派生一份 configs/_sweep/<tag>.yaml
                          （仅覆盖 dpo.beta / dpo.lambda_sft / output_dir，其余不动），并写
                          configs/_sweep/cells.tsv（每行 `tag<TAB>config<TAB>output_dir`）供 wrapper 逐格训。
    --mode aggregate    : 读每格 results/out/<tag>.summary.json + metrics.jsonl 行数（对评测清单算
                          immediate-EOS 生成失败率），拼成 β×λ 的 WER/UTMOS/失败率网格 + 裁决，落
                          results/out/beta_lambda_sweep.txt(+ .csv)。

判据（回填 §5.1 / 附录）:
    整张网格 WER 都远高于 base(0.112) ⇒ 崩坏对 (β,λ) 稳健，**非单一超参伪影**（强化 cautionary 主张）；
    若大 λ_sft 角落 WER 回落 ⇒ 诚实报「强 anchor 可缓冲」，机理（off-manifold 目标 + 似然/模仿目标）不变。

用法:
    # 1) 生成网格配置
    python eval/beta_lambda_sweep.py --mode emit-configs \\
        --base-config configs/dpo_lora.yaml --betas 0.1 0.5 2.0 --lambdas 0.0 0.1 1.0
    # 2)（wrapper run_x15.sh 逐格训练+合成+打分）
    # 3) 聚合成网格
    python eval/beta_lambda_sweep.py --mode aggregate \\
        --betas 0.1 0.5 2.0 --lambdas 0.0 0.1 1.0 --manifest data/prompts/eval_big.jsonl \\
        --out results/out/beta_lambda_sweep.txt

前置条件: emit-configs 需 base DPO 配置在；aggregate 需各格 results/out/<tag>.summary.json 已产出。
预期结果: configs/_sweep/<tag>.yaml + cells.tsv（emit）；results/out/beta_lambda_sweep.txt(+.csv)（aggregate）。
注意: 本机(macOS/无 GPU)只语法检查；真实训练在 GPU 机器（建议 base 用 dpo_lora.yaml + 小网格，省卡）。
      emit-configs 需 pyyaml（GPU 环境已装；本机 py_compile 不执行 import，可过）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402


def _tag(beta: float, lam: float) -> str:
    """文件系统安全的格标签：0.1 -> '0p1'。"""
    b = str(beta).replace(".", "p").replace("-", "m")
    l = str(lam).replace(".", "p").replace("-", "m")
    return f"dpo_b{b}_l{l}"


def _emit_configs(args) -> None:
    import yaml  # 仅此分支需要；GPU 环境已装

    base_path = Path(args.base_config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if "dpo" not in base:
        raise SystemExit(f"{base_path} 不是 DPO 配置（缺 dpo: 段）——请用 configs/dpo_*.yaml 作 base。")

    out_dir = Path(args.config_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for beta in args.betas:
        for lam in args.lambdas:
            tag = _tag(beta, lam)
            cfg = json.loads(json.dumps(base))  # 深拷贝
            cfg["dpo"]["beta"] = float(beta)
            cfg["dpo"]["lambda_sft"] = float(lam)
            cfg["output_dir"] = f"./alignment/out/{tag}"
            cfg_path = out_dir / f"{tag}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            cells.append((tag, str(cfg_path), cfg["output_dir"]))
            print(f"  [cell] β={beta:<4} λ_sft={lam:<4} -> {cfg_path}")

    tsv = out_dir / "cells.tsv"
    tsv.write_text("".join(f"{t}\t{c}\t{o}\n" for t, c, o in cells), encoding="utf-8")
    print(f"\n[emit] {len(cells)} 格配置 -> {out_dir}/  | 清单 -> {tsv}")
    print(f"[emit] 下一步: wrapper 逐格 train(--config) -> batch_synthesize(--ckpt output_dir --treatment tag) -> score_synth")


def _aggregate(args) -> None:
    # 评测清单的全部 fid（用于算 immediate-EOS 生成失败率：清单有、metrics.jsonl 没有 = 失败句）
    total = len(list(read_jsonl(args.manifest)))
    if total == 0:
        raise SystemExit(f"评测清单空: {args.manifest}")

    def cell_stats(tag: str):
        sj = PATHS.results_dir / f"{tag}.summary.json"
        mj = PATHS.results_dir / f"{tag}.metrics.jsonl"
        if not sj.exists():
            # 完全崩坏（每句 immediate-EOS → 零 wav → score_synth 不产 summary）：从合成报告取失败率，
            # 不让这格静默掉成 '—'（那会抹掉最强崩坏信号）。
            rep = PATHS.project_root / "inference" / "out" / tag / "_synthesis_report.json"
            if rep.exists():
                fr = float(json.loads(rep.read_text(encoding="utf-8")).get("failure_rate", 1.0))
                return {"wer": None, "utmos": None, "sim": None, "scored": 0, "fail": fr, "collapsed": True}
            return None
        s = json.loads(sj.read_text(encoding="utf-8"))
        scored = sum(1 for _ in read_jsonl(mj)) if mj.exists() else int(s.get("n", 0))
        return {"wer": s.get("wer"), "utmos": s.get("utmos"), "sim": s.get("sim"),
                "scored": scored, "fail": 1.0 - scored / total, "collapsed": False}

    lines: list[str] = []

    def emit(x: str = "") -> None:
        print(x)
        lines.append(x)

    emit("==== [★4] DPO-FM β × λ_sft 网格（崩坏是否对超参稳健 / anchor 能否救）====")
    emit(f"评测清单 {args.manifest}（{total} 句）；base 参照 WER≈0.112、dpo_multiobj(β0.1,λ0.1) WER≈0.890。\n")

    grids = {"WER↓": "wer", "UTMOS↑": "utmos", "EOS失败率": "fail"}
    csv_rows = ["metric,beta,lambda,value,scored_n"]
    all_wer: list[float] = []
    for title, key in grids.items():
        emit(f"[{title}]")
        header = "  β\\λ_sft " + "".join(f"{l:>10}" for l in args.lambdas)
        emit(header)
        for beta in args.betas:
            cells_txt = []
            for lam in args.lambdas:
                st = cell_stats(_tag(beta, lam))
                if st is None:
                    cells_txt.append(f"{'—':>10}")
                    continue
                if st[key] is None:  # 完全崩坏格：无数值 WER/UTMOS，但要显式标 FAIL（非 '—'）
                    cells_txt.append(f"{('FAIL' if st.get('collapsed') else '—'):>10}")
                    continue
                v = st[key]
                cells_txt.append(f"{v:>10.3f}")
                csv_rows.append(f"{key},{beta},{lam},{v:.4f},{st['scored']}")
                if key == "wer":
                    all_wer.append(v)
            emit(f"  {beta:<7} " + "".join(cells_txt))
        emit("")

    # 完全崩坏格计数（零 wav、最强崩坏信号；不进 all_wer，须单列）
    collapsed = sum(1 for beta in args.betas for lam in args.lambdas
                    if (st := cell_stats(_tag(beta, lam))) and st.get("collapsed"))

    # 裁决
    if all_wer or collapsed:
        emit("[裁决]")
        if collapsed:
            emit(f"  {collapsed} 格**完全崩坏**（生成失败率≈1.0、零可打分 wav）—— 最强崩坏信号。")
        if all_wer:
            wmin, wmax = min(all_wer), max(all_wer)
            emit(f"  其余可打分格 WER 范围 [{wmin:.3f}, {wmax:.3f}]（base 0.112）。")
            if wmin > 0.30:
                emit("  ✅ 整张 (β,λ) 网格 WER 均远高于 base ⇒ 崩坏对超参**稳健**、非单一超参伪影")
                emit("     → §5.1/附录：把『崩坏』从单点升为『β×λ 网格稳健』，强化 cautionary 主张。")
            else:
                emit("  ⚠️ 存在 WER≤0.30 的格 ⇒ 某些 (β,λ)（多半大 λ_sft）缓冲了崩坏：诚实报『强 anchor 可缓冲』，")
                emit("     机理（off-manifold 目标 + 似然/模仿目标，二者缺一不崩）不变，但『regardless of hyperparams』须软化。")
        elif collapsed:
            emit("  ✅ 网格内可打分的格为空、全部完全崩坏 ⇒ 崩坏对超参极稳健。")
    else:
        emit("[裁决] 无可用格（先在 GPU 机器跑完各格 train+synth+score）。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".csv")).write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"\n[--out] 网格报告 -> {out}  | CSV -> {out.with_suffix('.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="★4 DPO-FM β×λ_sft 网格 sweep（编排 + 聚合）")
    ap.add_argument("--mode", choices=["emit-configs", "aggregate"], required=True)
    ap.add_argument("--base-config", default="configs/dpo_lora.yaml",
                    help="派生网格的 base DPO 配置（sweep 建议用 dpo_lora.yaml 省卡）")
    ap.add_argument("--config-out-dir", default="configs/_sweep")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.5, 2.0])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    ap.add_argument("--manifest", default="data/prompts/eval_big.jsonl",
                    help="aggregate 模式：算生成失败率用的评测清单")
    ap.add_argument("--out", default="results/out/beta_lambda_sweep.txt")
    args = ap.parse_args()

    if args.mode == "emit-configs":
        _emit_configs(args)
    else:
        _aggregate(args)


if __name__ == "__main__":
    main()
