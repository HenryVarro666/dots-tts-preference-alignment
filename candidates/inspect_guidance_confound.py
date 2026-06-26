"""
inspect_guidance_confound.py — 关闭 open question：FM-reward 反相关是「本质」还是「guidance 混淆」？

背景:
    probe_fm_reward.py 发现去噪 FM margin 与 reward 差**反相关**(Pearson −0.5)：reward 越高的 winner，
    L_FM 反而越高。机理猜想：高 guidance 候选吐字更清(reward 高)但偏离模型自然分布(L_FM 高)。
    若 winner **系统性**是高 guidance/高 num_steps，则那条反相关（部分）是候选生成设置的混淆，而非 FM 本质。

做什么:
    把 pairs_best_worst.jsonl 与 candidates_index.jsonl 按音频路径 join，对每个 pair 取出
    winner / loser 的生成设置(guidance_scale / num_steps / speaker_scale)，检验：
      (1) winner 的设置是否系统性高于 loser（占比 + 均值差）；
      (2) reward 差是否与「设置差(win−lose)」相关（Pearson）——若强正相关 = reward 被设置驱动 = 混淆坐实。
怎么做:
    python candidates/inspect_guidance_confound.py \
        --pairs pairs/out/pairs_best_worst.jsonl \
        --index candidates/out/candidates_index.jsonl
为什么这么设计:
    这是**纯元数据**分析(无需模型前向、秒级)，先用最便宜的观测判定混淆是否存在；
    若存在，再跑「设置匹配」的受控重测(仅变采样噪声、固定 guidance)确认 FM 反相关是否仍在。
对实验意义:
    • winner 系统性高 guidance + reward∝设置差 → 反相关是**生成混淆** → 修法在候选生成(设置匹配配对)。
    • winner 设置与 loser 无系统差异 → 反相关更可能是 **FM 本质** → docs/06 的结论可收紧。
本机只语法检查；真正运行在有 candidates_index + pairs 的服务器上。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _key(p: str) -> str:
    """join 键：优先绝对路径；兜底用 {fid}/{basename} 以抵消相对/绝对差异。"""
    p = str(p)
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{os.path.basename(p)}"


def _load_index(index_path: Path) -> dict:
    by_audio = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        settings = {
            "guidance_scale": r.get("guidance_scale"),
            "num_steps": r.get("num_steps"),
            "speaker_scale": r.get("speaker_scale"),
        }
        by_audio[_key(r["audio"])] = settings           # 全路径键
        by_audio.setdefault(os.path.basename(r["audio"]), settings)  # basename 兜底
    return by_audio


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _lookup(by_audio: dict, audio: str):
    return by_audio.get(_key(audio)) or by_audio.get(os.path.basename(audio))


def main() -> None:
    ap = argparse.ArgumentParser(description="guidance-confound 检验")
    ap.add_argument("--pairs", default="pairs/out/pairs_best_worst.jsonl")
    ap.add_argument("--index", default="candidates/out/candidates_index.jsonl")
    args = ap.parse_args()

    by_audio = _load_index(Path(args.index))
    print(f"candidates_index: {len(by_audio)} 条音频设置")

    fields = ["guidance_scale", "num_steps", "speaker_scale"]
    diffs = {f: [] for f in fields}     # 每 pair 的 setting_win − setting_lose
    win_vals = {f: [] for f in fields}
    lose_vals = {f: [] for f in fields}
    reward_gaps: list[float] = []
    n_join_fail = 0
    n = 0

    for line in Path(args.pairs).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sw = _lookup(by_audio, r["win_audio"])
        sl = _lookup(by_audio, r["lose_audio"])
        if not sw or not sl or any(sw[f] is None or sl[f] is None for f in fields):
            n_join_fail += 1
            continue
        n += 1
        reward_gaps.append(float(r["reward_win"]) - float(r["reward_lose"]))
        for f in fields:
            win_vals[f].append(float(sw[f]))
            lose_vals[f].append(float(sl[f]))
            diffs[f].append(float(sw[f]) - float(sl[f]))

    if n == 0:
        raise SystemExit(f"无成功 join 的 pair（join 失败 {n_join_fail}）——检查路径键是否匹配。")
    print(f"成功 join {n} 对（失败 {n_join_fail}）\n")

    confounded = []
    for f in fields:
        d = diffs[f]
        higher = sum(1 for x in d if x > 0)
        lower = sum(1 for x in d if x < 0)
        equal = sum(1 for x in d if x == 0)
        mean_win = sum(win_vals[f]) / n
        mean_lose = sum(lose_vals[f]) / n
        pear = _pearson(reward_gaps, d)
        print(f"== {f} ==")
        print(f"  winner 均值={mean_win:.3f}  loser 均值={mean_lose:.3f}  差={mean_win-mean_lose:+.3f}")
        print(f"  winner 更高: {higher}/{n} ({100*higher/n:.0f}%)   更低: {lower}/{n} ({100*lower/n:.0f}%)   相等: {equal}/{n}")
        print(f"  Pearson(reward 差, {f} 差) = {pear:+.3f}   {'← reward 被该设置驱动(混淆信号)' if (pear==pear and abs(pear)>0.3) else ''}")
        # 该字段构成混淆的条件：winner 系统性偏一侧 且 与 reward 相关
        if higher / n > 0.6 and pear == pear and pear > 0.3:
            confounded.append(f)

    print("\n=== 判定 ===")
    if confounded:
        print(f"▶ winner 系统性偏高且与 reward 相关的设置：{confounded}")
        print("  → guidance-confound **存在**：probe_fm_reward 的反相关(部分)由候选生成设置驱动，非纯 FM 本质。")
        print("  → 下一步：构造『设置匹配』pair（固定 guidance、仅变采样噪声/seed）重跑 probe_fm_reward；")
        print("     若反相关消失=混淆主导(修候选生成)；若仍在=FM 本质(docs/06 结论收紧)。")
    else:
        print("▶ winner/loser 的生成设置无系统性差异、且与 reward 不相关。")
        print("  → guidance-confound **不成立**：反相关更可能是 FM 本质 → docs/06 结论可据此收紧。")


if __name__ == "__main__":
    main()
