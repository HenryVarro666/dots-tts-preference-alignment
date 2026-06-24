"""
eval_8gb_matrix.py — Phase B（门控）：质量×延迟×显存 Pareto + 8GB 可行性矩阵

做什么:    对 {NFE∈1,2,4,10} × {FP16,INT8,INT4} 组合，测 WER/SIM/UTMOS + 首包延迟 + RTF + 峰值显存，
           汇成 8GB 可行性矩阵（哪些格能在 8GB 跑出可用质量）。
怎么做:    复用 inference + eval 脚本逐格跑，torch.cuda.max_memory_allocated 取峰值显存。
为什么:    直接回答仓库 04 的现实命题：dots.tts 在什么配置下能塞进 8GB 且质量可接受。
对实验意义:
    Phase B 的最终交付——把「对齐(A) + 少步蒸馏 + 量化(B)」落到消费级硬件的可行性证据；
    同时检验对齐增益在量化后是否存活。

⚠️ 门控；本机只语法检查。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402

NFE = [1, 2, 4, 10]
PRECISION = ["fp16", "int8", "int4"]


def main() -> None:
    ap = argparse.ArgumentParser(description="8GB Pareto 矩阵")
    ap.add_argument("--ckpt-root", required=True, help="含各量化变体的目录")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = []
    for nfe, prec in itertools.product(NFE, PRECISION):
        cell = {
            "nfe": nfe, "precision": prec,
            # 下列由逐格 inference+eval 回填
            "wer": None, "sim": None, "utmos": None,
            "first_packet_ms": None, "rtf": None, "peak_vram_gb": None,
            "fits_8gb": None,  # peak_vram_gb < 8
        }
        print(f"  格子 NFE={nfe} prec={prec}: 待跑 inference+eval, 记录 peak_vram")
        matrix.append(cell)

    out = PATHS.results_dir / "phaseB_8gb_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 矩阵骨架 -> {out}（逐格跑后回填，标 fits_8gb）")


if __name__ == "__main__":
    main()
