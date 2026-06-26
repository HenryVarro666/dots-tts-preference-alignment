"""
filter_pairs_by_likelihood.py — 把偏好对过滤成「winner 更 on-manifold」的子集（臂 X6a 用）

做什么:
    读 pairs_best_worst.jsonl，按「winner 更贴近模型自然生成流形(on-manifold)」的判据筛掉
    一部分对，输出 pairs/out/pairs_onmanifold.jsonl（每行仍是原样 PairRow，字段不改）。
怎么做（含命令 + 输入输出）:
    支持两种判据 --by {guidance, fm_margin}：
      • guidance：join candidates/out/candidates_index.jsonl，按 win_audio 取回 winner 的
        guidance_scale，**保留 winner guidance_scale ≤ --thr(默认 1.2) 的对**（高 guidance=吐字清但
        偏离自然分布=off-manifold，先从源头剔除这类 winner）。
      • fm_margin：join alignment/out/probe_per_pair.csv，按 (fid,win_audio,lose_audio) 取回该对的
        mc_margin，**保留 mc_margin > --fm-thr(默认 0.0) 的对**（去噪 K-MC 平均后 winner 的 L_FM
        比 loser 更低=winner 对模型而言重建更易=更 on-manifold）。
    join 键复用 inspect_guidance_confound.py 的 `{父目录名}/{文件名}` 规则 + basename 兜底，抵消相对/绝对路径差异。
    命令:
        python pairs/filter_pairs_by_likelihood.py --by guidance \
            --pairs pairs/out/pairs_best_worst.jsonl \
            --index candidates/out/candidates_index.jsonl --thr 1.2
        python pairs/filter_pairs_by_likelihood.py --by fm_margin \
            --pairs pairs/out/pairs_best_worst.jsonl \
            --probe alignment/out/probe_per_pair.csv --fm-thr 0.0
    输入: pairs/out/pairs_best_worst.jsonl(每行 PairRow) + 判据对应的 candidates_index.jsonl / probe_per_pair.csv。
    输出: pairs/out/pairs_onmanifold.jsonl(每行 PairRow，schema 原样、不增删字段)。
为什么这么设计:
    probe_fm_reward 发现 DPO-FM 学不动且 FM margin 与 reward 反相关；inspect_guidance_confound 指出
    winner 系统性偏高 guidance 可能是混淆。两条线索都指向「winner 太 off-manifold，DPO 想抬它的似然
    却与模型自身分布打架」。本脚本不动训练，只在**数据侧**先把 off-manifold winner 的对剔掉，做最便宜的
    干预；两种判据分别从「生成设置」和「去噪 FM」两个角度逼近同一个 on-manifold 概念，互为旁证。
对实验与 claim 的意义:
    产出的子集喂给 dpo_onmanifold.yaml，测「只留 on-manifold winner 能否救 DPO-FM」：
      • 若 over base 改善 → 退化(部分)源于 winner off-manifold，修法=数据侧过滤(本脚本即解药)；
      • 若仍不改善 → 把负面结论收紧到「不是 winner 选得偏，是 DPO-FM 似然代理本身的边界」，
        并据此给出「该怎么做」(换隐式似然代理 / 重采 on-manifold 候选再配对)。

前置条件: pairs/out/pairs_best_worst.jsonl；guidance 判据另需 candidates_index.jsonl，
          fm_margin 判据另需先跑 probe_fm_reward.py 产出 probe_per_pair.csv。
预期结果: pairs/out/pairs_onmanifold.jsonl(原样 PairRow，行数 ≤ 输入)，并打印保留/剔除/join 失败计数。
⚠️ 本机只语法检查；纯文件处理、不加载模型，真正运行在有 pairs/index/probe 的服务器上。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, PairRow, read_jsonl, write_jsonl  # noqa: E402


def _key(p: str) -> str:
    """join 键：`{父目录名}/{文件名}`，与 inspect_guidance_confound._key 完全一致（抵消相对/绝对差异）。"""
    p = str(p)
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{os.path.basename(p)}"


def _row_to_pair(r: dict) -> PairRow:
    """把读到的 dict 原样重建成 PairRow，保证输出 schema 不漂移（只取契约字段、不夹带额外键）。"""
    return PairRow(
        fid=r["fid"], text=r["text"], prompt_audio=r["prompt_audio"],
        prompt_text=r["prompt_text"], lang=r["lang"],
        win_audio=r["win_audio"], lose_audio=r["lose_audio"],
        reward_win=r["reward_win"], reward_lose=r["reward_lose"],
        strategy=r["strategy"],
    )


def _load_guidance(index_path: Path) -> dict:
    """audio → guidance_scale。全路径键 + basename 兜底（与 inspect_guidance_confound 同款双键）。"""
    by_audio: dict = {}
    for c in read_jsonl(index_path):
        g = c.get("guidance_scale")
        by_audio[_key(c["audio"])] = g
        by_audio.setdefault(os.path.basename(c["audio"]), g)
    return by_audio


def _lookup_guidance(by_audio: dict, audio: str):
    k = _key(audio)
    if k in by_audio:
        return by_audio[k]
    return by_audio.get(os.path.basename(audio))  # None=join 失败 或 该候选本就无 guidance


def _load_fm_margins(csv_path: Path) -> dict:
    """(fid, key(win), key(lose)) → mc_margin。probe_per_pair.csv 逐对一行，键用 _key 统一路径形态。"""
    margins: dict = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (str(row["fid"]), _key(row["win_audio"]), _key(row["lose_audio"]))
            margins[key] = float(row["mc_margin"])
    return margins


def _filter_guidance(rows: list[dict], index_path: Path, thr: float):
    by_audio = _load_guidance(index_path)
    print(f"candidates_index: {len(by_audio)} 条音频 guidance 记录")
    kept, n_join_fail = [], 0
    for r in rows:
        g = _lookup_guidance(by_audio, r["win_audio"])
        if g is None:
            n_join_fail += 1
            continue
        if float(g) <= thr:               # winner 低 guidance = 更 on-manifold = 保留
            kept.append(r)
    print(f"判据=guidance  阈值 winner guidance_scale ≤ {thr}  (join 失败 {n_join_fail} 对)")
    return kept, n_join_fail


def _filter_fm_margin(rows: list[dict], probe_path: Path, fm_thr: float):
    margins = _load_fm_margins(probe_path)
    print(f"probe_per_pair: {len(margins)} 对 mc_margin 记录")
    kept, n_join_fail = [], 0
    for r in rows:
        key = (str(r["fid"]), _key(r["win_audio"]), _key(r["lose_audio"]))
        mc = margins.get(key)
        if mc is None:
            n_join_fail += 1
            continue
        if mc > fm_thr:                   # winner 去噪 L_FM 更低 = 更 on-manifold = 保留
            kept.append(r)
    print(f"判据=fm_margin  阈值 mc_margin > {fm_thr}  (join 失败 {n_join_fail} 对)")
    return kept, n_join_fail


def main() -> None:
    ap = argparse.ArgumentParser(description="过滤偏好对：只留 winner 更 on-manifold 的子集（X6a）")
    ap.add_argument("--by", choices=["guidance", "fm_margin"], required=True,
                    help="on-manifold 判据：guidance(低 CFG) 或 fm_margin(去噪 L_FM winner 更低)")
    ap.add_argument("--pairs", default=str(PATHS.pairs_dir / "pairs_best_worst.jsonl"))
    ap.add_argument("--index", default=str(PATHS.candidates_dir / "candidates_index.jsonl"),
                    help="guidance 判据用：join 取 winner 的 guidance_scale")
    ap.add_argument("--probe", default=str(PATHS.project_root / "alignment" / "out" / "probe_per_pair.csv"),
                    help="fm_margin 判据用：join 取每对的 mc_margin")
    ap.add_argument("--thr", type=float, default=1.2, help="guidance 判据：保留 winner guidance_scale ≤ thr")
    ap.add_argument("--fm-thr", type=float, default=0.0, help="fm_margin 判据：保留 mc_margin > fm_thr")
    ap.add_argument("--out", default=str(PATHS.pairs_dir / "pairs_onmanifold.jsonl"))
    args = ap.parse_args()

    rows = list(read_jsonl(Path(args.pairs)))
    if not rows:
        raise SystemExit(f"pairs 空: {args.pairs}")
    print(f"输入 {len(rows)} 对 <- {args.pairs}")

    if args.by == "guidance":
        kept, n_join_fail = _filter_guidance(rows, Path(args.index), args.thr)
    else:
        kept, n_join_fail = _filter_fm_margin(rows, Path(args.probe), args.fm_thr)

    if not kept:
        raise SystemExit(
            f"过滤后 0 对（输入 {len(rows)}，join 失败 {n_join_fail}）——检查阈值是否过严或路径键是否匹配。")

    pairs = [_row_to_pair(r) for r in kept]
    n = write_jsonl(Path(args.out), pairs)
    drop = len(rows) - len(kept)
    print(f"✓ on-manifold 子集: 保留 {n}/{len(rows)} 对（剔除 {drop}，含 join 失败 {n_join_fail}）-> {args.out}")


if __name__ == "__main__":
    main()
