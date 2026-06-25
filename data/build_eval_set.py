"""
build_eval_set.py — 从「未训练过」的 LJSpeech 切出更大的 held-out 评测集

做什么:
    从 dots.tts 下载器的源 manifest 里，取**不在训练集**的 N 条，转成统一 PromptRow 评测清单。
怎么做:
    读训练集 fids → 从源 manifest 过滤掉它们 → 取前 N → 写出 + 校验与训练集零重叠。
    源 manifest 路径不指定时自动在 data/corpora/ 下找 *manifest_train.jsonl。
为什么这么设计:
    原 held-out 评测集只有 ~12 条 → bootstrap CI 很宽、结论大概率不显著；
    扩到 ~100 条让 WER/UTMOS 的 CI 收紧，结论才有说服力。**与训练集零重叠**是泛化结论的底线。
对实验意义:
    评测集大小直接决定结论可信度（小样本必报 CI，见 results/aggregate_results 的 bootstrap CI）。

前置条件: dots.tts 下载器产出的源 manifest（含 13100 条）+ data/prompts/dots_manifest_train.jsonl。
预期结果: data/prompts/eval_big.jsonl（N 条）+ 终端校验「与训练集重叠 = 0」。
用法:
    python data/build_eval_set.py                 # 默认 100 条 -> data/prompts/eval_big.jsonl
    python data/build_eval_set.py --n 200         # 要更大就调 --n
⚠️ 纯标准库，任何 python3 都能跑（不挑 venv）。本机可语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, PromptRow, read_jsonl, write_jsonl  # noqa: E402


def _auto_find_src() -> Path | None:
    """在 data/corpora/ 下自动找 dots.tts 下载器产出的 *manifest_train.jsonl。"""
    corpora = PATHS.project_root / "data" / "corpora"
    if not corpora.exists():
        return None
    cands = sorted(corpora.rglob("*manifest_train.jsonl"))
    return cands[0] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description="构造更大的 held-out 评测集（与训练集不重叠）")
    ap.add_argument("--train", default=str(PATHS.prompts_dir / "dots_manifest_train.jsonl"),
                    help="训练集 prompt 清单（用于排除已训练 fid）")
    ap.add_argument("--src", default=None,
                    help="源 manifest（dots.tts 下载器产出，含全部 13100 条）；不填则自动查找")
    ap.add_argument("--n", type=int, default=100, help="评测条数（越大 CI 越紧，但评测合成更费 GPU）")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default=str(PATHS.prompts_dir / "eval_big.jsonl"))
    args = ap.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"缺训练集清单: {train_path}（先跑 data/prepare_prompts.py）")
    src = Path(args.src) if args.src else _auto_find_src()
    if not src or not src.exists():
        raise SystemExit("找不到源 manifest（data/corpora/**/*manifest_train.jsonl）；"
                         "用 --src 显式指定，或先跑 dots.tts 的 prepare_train_jsonl_manifest.py")

    train_fids = {str(r["fid"]) for r in read_jsonl(train_path)}
    unused = []
    for r in read_jsonl(src):
        if str(r["fid"]) not in train_fids:
            unused.append(r)
            if len(unused) >= args.n:
                break

    rows = [PromptRow(fid=str(r["fid"]), text=r["text"],
                      prompt_audio=r["audio"], prompt_text=r["text"],
                      lang=args.lang, source="ljspeech-eval") for r in unused]
    n = write_jsonl(args.out, rows)

    overlap = len({row.fid for row in rows} & train_fids)
    print(f"✓ 写出 {n} 条 -> {args.out}（源: {src.name}）")
    print(f"  与训练集重叠 = {overlap}  ({'OK' if overlap == 0 else '⚠️ 不应非0！fid 格式可能对不上'})")
    if n < args.n:
        print(f"  ⚠️ 只凑到 {n} 条（源里未训练样本不足 {args.n}）")
    if overlap != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
