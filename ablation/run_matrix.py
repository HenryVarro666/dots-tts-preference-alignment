"""
run_matrix.py — 阶段⑧：实验矩阵编排（treatments × eval 模式）

做什么:
    按 PROTOCOL.md 跑完整梯队(base/sft/open_soar/dpo_multiobj/dpo_weronly)与消融(β/奖励/配对/NFE)，
    每个 run 串「合成(离线+流式)→评测」，并写一份**可溯源的 run 账本(ledger.jsonl)**。
怎么做:
    用 subprocess 串既有脚本；不重复实现训练/评测，只编排 + 记录每个数字来自哪个 config/ckpt。
为什么这么设计:
    实验易乱在「这个数字到底哪个设置跑的」。账本把 (treatment, config, ckpt, manifest, NFE, 模式)
    与产物路径绑定，保证 results 阶段能逐数溯源。
对实验意义:
    可复现与可溯源是结论可信的底线；账本即论文附录的实验登记表。

前置条件: 各 treatment 的对齐 ckpt 已训练；评测集就绪。
预期结果: ablation/ledger.jsonl + 触发各 run 的合成/评测产物。
⚠️ 本机只语法检查（不真正 subprocess 跑 GPU 任务）。
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

# (treatment, ckpt 占位) —— ckpt 由 alignment 阶段产出后填入对应 output_dir
TREATMENTS = {
    "base": "rednote-hilab/dots.tts-base",
    "sft": str(PROJ / "alignment" / "out" / "sft"),
    "open_soar": str(PROJ / "alignment" / "out" / "open_soar"),
    "dpo_multiobj": str(PROJ / "alignment" / "out" / "dpo_multiobj"),
    "dpo_weronly": str(PROJ / "alignment" / "out" / "dpo_weronly"),
}


def _run(cmd: list[str], dry: bool) -> None:
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="实验矩阵编排")
    ap.add_argument("--manifest", required=True, help="评测清单 jsonl")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--sim-backend", choices=["ecapa", "wavlm"], default="ecapa")
    ap.add_argument("--nfe", type=int, nargs="+", default=[10], help="部署 NFE（迁移消融可给多个）")
    ap.add_argument("--treatments", nargs="*", default=None, help="只跑这些 treatment（默认全部）")
    ap.add_argument("--skip-streaming", action="store_true",
                    help="跳过流式合成+流式评测（离线 RQ1/RQ3 主证据先跑通；流式单独补）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令、写账本，不真正执行")
    args = ap.parse_args()
    treatments = {k: v for k, v in TREATMENTS.items() if not args.treatments or k in args.treatments}

    ledger = []
    for treat, ckpt in treatments.items():
        for nfe in args.nfe:
            tag = f"{treat}_nfe{nfe}"
            off_dir = PROJ / "inference" / "out" / tag
            st_dir = PROJ / "inference" / "out_streaming" / tag
            status = "ok"
            # 单个臂的某一步失败（如崩坏臂的合成异常、空目录打分报错）**不应连坐其它臂**：
            # 捕获该臂、记 status="failed"、继续下一个臂。整矩阵尽量跑全，失败的臂在账本里可溯源。
            # （batch_synthesize 已内部容错单条生成，这里再兜一层防 score_synth 等步骤的异常。）
            try:
                # 1) 离线合成 + 打分（RQ1/RQ3 主证据）
                _run(["python", str(PROJ / "inference" / "batch_synthesize.py"),
                      "--ckpt", ckpt, "--manifest", args.manifest, "--treatment", tag,
                      "--num-steps", str(nfe), "--out", str(off_dir)], args.dry_run)
                #    score_synth: 统一 WER/SIM/UTMOS（复用 reward 后端，不用 seed-tts-eval shell，见 docs/05 B-4/B-5）
                _run(["python", str(PROJ / "eval" / "score_synth.py"),
                      "--synth-dir", str(off_dir), "--manifest", args.manifest,
                      "--treatment", tag, "--sim-backend", args.sim_backend], args.dry_run)
                # 2) 流式合成 + 打分 + 流式指标（可 --skip-streaming 跳过）
                if not args.skip_streaming:
                    _run(["python", str(PROJ / "inference" / "batch_synthesize_streaming.py"),
                          "--ckpt", ckpt, "--manifest", args.manifest, "--treatment", tag,
                          "--num-steps", str(nfe)], args.dry_run)
                    _run(["python", str(PROJ / "eval" / "score_synth.py"),
                          "--synth-dir", str(st_dir), "--manifest", args.manifest,
                          "--treatment", f"{tag}_stream", "--sim-backend", args.sim_backend], args.dry_run)
                    _run(["python", str(PROJ / "eval" / "streaming_metrics.py"),
                          "--streaming-dir", str(st_dir), "--treatment", tag], args.dry_run)
            except subprocess.CalledProcessError as e:
                status = "failed"
                print(f"  ⚠️ [{tag}] 某步骤非零退出，跳过该臂剩余步骤，继续下一个臂: {e}", file=sys.stderr)
            ledger.append({"treatment": treat, "tag": tag, "ckpt": ckpt, "nfe": nfe,
                           "manifest": args.manifest, "offline_dir": str(off_dir),
                           "streaming_dir": str(st_dir), "status": status})

    out = Path(__file__).parent / "ledger.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ledger) + "\n", encoding="utf-8")
    print(f"✓ 账本 {len(ledger)} 行 -> {out}")
    if args.dry_run:
        print("  (dry-run：未真正执行，仅打印命令 + 写账本)")
    print("  注: EmergentTTS(表现力) 不在本编排内——需另跑 eval/run_emergent_tts.py --treatment <tag>，"
          "否则 aggregate 的 EmergentTTS 列为空（不报错）。")


if __name__ == "__main__":
    main()
