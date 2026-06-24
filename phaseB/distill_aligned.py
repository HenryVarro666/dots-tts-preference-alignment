"""
distill_aligned.py — Phase B（门控）：把对齐后模型蒸馏到少步(NFE 1~2)

做什么:    以**对齐后**的策略模型为 teacher，用 dots.tts 既有 MeanFlow 蒸馏训出少步 student。
怎么做:    薄包装 `dots.tts/scripts/train_dots_tts_meanflow.py`，只把 teacher 指向对齐 ckpt、
           目标步数设 1~2。不重写蒸馏逻辑。
为什么:    A→B 主线的「变快变小」一环；teacher 用对齐模型而非 base，才能保住对齐增益。
对实验意义:
    回答核心开放问题——**偏好对齐能否在少步蒸馏后存活？**（MeanFlow 的区间均值目标可能抹平对齐过的速度场。）

⚠️ 门控：仅当 Phase A 达标(PHASE_B_PROTOCOL.md)才跑。本机只语法检查。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B: 蒸馏对齐模型到少步")
    ap.add_argument("--config", default=str(PATHS.project_root / "configs" / "phaseB" / "meanflow_distill_aligned.yaml"))
    ap.add_argument("--teacher", required=True, help="对齐后 ckpt（dpo_multiobj 等），作 teacher")
    ap.add_argument("--target-steps", type=int, default=2, help="student NFE（1 或 2）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meanflow = PATHS.dots_tts_repo / "scripts" / "train_dots_tts_meanflow.py"
    cmd = ["accelerate", "launch", str(meanflow),
           "--config", args.config,
           # 下列覆盖项需对照 train_dots_tts_meanflow.py 的 MeanFlowSettings 字段名
           f"--teacher_ckpt={args.teacher}", f"--student_steps={args.target_steps}"]
    print("$", " ".join(cmd))
    print("  (teacher=对齐模型；验证「对齐能否在少步蒸馏后存活」)")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
