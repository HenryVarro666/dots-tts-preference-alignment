"""
quantize.py — Phase B（门控）：INT8/INT4 量化，瞄准 8GB Turing(RTX2070)

做什么:    把(蒸馏后)模型做 INT8/INT4 量化，导出可在 8GB 卡上加载的权重。
怎么做:    bitsandbytes NF4 / torchao；对 Qwen2.5 backbone 与 DiT 头分别量化策略可调。
为什么:    对接仓库 04 部署指南——8GB Turing(SM7.5) 跑不动 2B 全精度，**FlashRT 需 SM80+**，
           2070 路线 = 少步(Phase B 蒸馏) + 量化、无 FlashRT。
对实验意义:
    把「学术上变好的模型」真正落到消费级卡，是 A→B「全流程」的终点；也检验量化是否吃掉对齐增益。

⚠️ 门控 + 仅脚手架：量化细节随后端而定。本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B: 量化到 INT8/INT4")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["int8", "int4"], default="int8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_dots_tts_importable()
    print(f"⚠️ 脚手架：将 {args.ckpt} 以 {args.mode} 量化导出到 {args.out}")
    print("  路线提示（对接 ../../04-Windows-RTX2070-部署指南.md）：")
    print("  - backbone(Qwen2.5): bitsandbytes NF4(int4) 或 int8；")
    print("  - DiT 头：精度敏感，建议先保 int8 或混精；")
    print("  - 目标：峰值显存 < 8GB；无 FlashRT(Turing SM7.5)；")
    print("  - 配合 Phase B 蒸馏的少步(NFE 1~2) 才能在 2070 上达可用 RTF。")
    # TODO: 实际量化与导出（bitsandbytes / torchao），并记录峰值显存。


if __name__ == "__main__":
    main()
