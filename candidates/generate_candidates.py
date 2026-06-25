"""
generate_candidates.py — 阶段②：用 dots.tts 对每个 prompt 采样 K 个候选

做什么:
    读 data/prompts/*.jsonl，对每条 prompt 用 dots.tts 生成 K 个**有差异**的候选音频，
    写出 48kHz wav + candidates_index.jsonl（每行一个候选 = CandidateRow）。
怎么做:
    复用 `dots_tts.runtime.DotsTtsRuntime.generate(...)`。因该 API **没有 seed 参数**，
    多样性靠 (a) 超参网格 num_steps×guidance_scale×speaker_scale，(b) 每个候选前调用
    dots.tts 的 `seed_everything(seed_k)`（模型采样的 x0 噪声仍随机）。
为什么这么设计:
    DPO 需要「同一 prompt 下质量有高低差」的候选池；若候选全一样则无法构造偏好对。
    网格 + 逐候选 seed 是在「无 seed 参数」约束下制造可控差异的务实做法（见 docs/04-RISKS 风险①）。
对实验意义:
    候选池的「奖励离散度」直接决定偏好信号质量 —— 跑前应抽查奖励方差(reward/ 阶段)。
断点续跑（重要）:
    最贵的一步(~3h/188prompt)。支持中断后续跑——已存在的 wav 直接复用、只补缺的；
    用原子写(.tmp→os.replace)避免半截 wav；每 10 prompt 写一次索引。重跑用**同一 --k**即可。
    全部命中缓存时连模型都不加载、秒完成。

前置条件: 已装环境(env/setup_env.sh)、已下载 dots.tts-base 权重、已生成 prompts。
预期结果: candidates/out/{fid}/cand_{k}.wav + candidates/out/candidates_index.jsonl。
⚠️ 本机(无 GPU/无权重)只语法检查；真正生成在 GPU 机器上。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, CandidateRow, ensure_dots_tts_importable, read_jsonl, write_jsonl  # noqa: E402

# 多样性网格（笛卡尔积后取前 K 个；可在 configs 里覆盖）
DEFAULT_GRID = {
    "num_steps": [8, 10, 16],
    "guidance_scale": [1.0, 1.2, 1.6],
    "speaker_scale": [1.2, 1.5],
}


def _grid_points(grid: dict, k: int) -> list[dict]:
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[x] for x in keys))]
    # 均匀抽 K 个组合（步长跳采，避免只取网格一角）
    if len(combos) <= k:
        return combos
    step = len(combos) / k
    return [combos[int(i * step)] for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser(description="dots.tts 候选生成")
    ap.add_argument("--prompts", required=True, help="data/prompts/*.jsonl")
    ap.add_argument("--model", default="rednote-hilab/dots.tts-base")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--precision", default="bfloat16")
    args = ap.parse_args()

    ensure_dots_tts_importable()
    try:
        from dots_tts.utils.util_module import seed_everything  # type: ignore
    except Exception:  # 路径随版本微调，兜底为 no-op
        def seed_everything(_s):  # type: ignore
            pass

    grid = _grid_points(DEFAULT_GRID, args.k)
    prompts = list(read_jsonl(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    index_path = PATHS.candidates_dir / "candidates_index.jsonl"
    runtime = None  # 延迟加载：全部已缓存时无需加载模型/占 GPU（断点续跑友好）
    rows: list[CandidateRow] = []
    n_gen, n_skip = 0, 0
    total = len(prompts)
    for pi, p in enumerate(prompts):
        fid = p["fid"]
        out_dir = PATHS.candidates_dir / fid
        out_dir.mkdir(parents=True, exist_ok=True)
        made, reused = 0, 0
        for k, gp in enumerate(grid):
            seed = 1000 + k
            wav_path = out_dir / f"cand_{k}.wav"
            row = CandidateRow(
                fid=fid, k=k, audio=str(wav_path), text=p["text"],
                prompt_audio=p["prompt_audio"], prompt_text=p["prompt_text"],
                lang=p.get("lang", ""),
                num_steps=gp["num_steps"], guidance_scale=gp["guidance_scale"],
                speaker_scale=gp["speaker_scale"], seed=seed,
            )
            # 断点续跑：已存在且非空的 wav 直接复用（grid 确定 → 元数据可重建，需同 --k）
            if wav_path.exists() and wav_path.stat().st_size > 1024:
                rows.append(row); reused += 1; n_skip += 1
                continue
            if runtime is None:  # 真有要生成的，才加载模型
                from dots_tts.runtime import DotsTtsRuntime  # type: ignore
                runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision)
            seed_everything(seed)
            result = runtime.generate(
                text=p["text"], prompt_audio_path=p["prompt_audio"],
                prompt_text=p["prompt_text"], language=p.get("lang"),
                num_steps=gp["num_steps"], guidance_scale=gp["guidance_scale"],
                speaker_scale=gp["speaker_scale"], normalize_text=True,
            )
            _write_wav_atomic(wav_path, result["audio"], int(result["sample_rate"]))
            rows.append(row); made += 1; n_gen += 1
        # 周期性写索引：中途中断也留下「截至此刻」的可用索引；重跑会自动补全
        if (pi + 1) % 10 == 0 or pi == total - 1:
            write_jsonl(index_path, rows)
        print(f"  [{pi+1}/{total}] {fid}: 新生成 {made} / 复用 {reused}")

    write_jsonl(index_path, rows)
    print(f"✓ 共 {len(rows)} 候选（新生成 {n_gen}，复用 {n_skip}）-> {index_path}")
    if n_skip and not n_gen:
        print("  (全部命中缓存：本次未加载模型，断点续跑已完成)")


def _write_wav_atomic(path: Path, audio, sample_rate: int) -> None:
    """原子写 wav：先写 .tmp 再 os.replace，避免中断时留下半截 wav 被误当「已生成」。
    soundfile 不接受 bf16，先转 float32。"""
    import os
    import soundfile as sf  # type: ignore
    arr = audio
    try:  # torch.Tensor → numpy
        arr = audio.float().cpu().squeeze().numpy()
    except AttributeError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    sf.write(str(tmp), arr, sample_rate)
    os.replace(str(tmp), str(path))


if __name__ == "__main__":
    main()
