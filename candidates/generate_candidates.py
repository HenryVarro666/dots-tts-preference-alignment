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
    # 延迟导入：本机无 torch 时仍可 py_compile（导入只在真正运行时发生）
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore
    try:
        from dots_tts.utils.util_module import seed_everything  # type: ignore
    except Exception:  # 路径随版本微调，兜底为 no-op
        def seed_everything(_s):  # type: ignore
            pass

    runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision)
    grid = _grid_points(DEFAULT_GRID, args.k)

    prompts = list(read_jsonl(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    rows: list[CandidateRow] = []
    for p in prompts:
        fid = p["fid"]
        out_dir = PATHS.candidates_dir / fid
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, gp in enumerate(grid):
            seed = 1000 + k
            seed_everything(seed)
            result = runtime.generate(
                text=p["text"],
                prompt_audio_path=p["prompt_audio"],
                prompt_text=p["prompt_text"],
                language=p.get("lang"),
                num_steps=gp["num_steps"],
                guidance_scale=gp["guidance_scale"],
                speaker_scale=gp["speaker_scale"],
                normalize_text=True,
            )
            # 输出扩展名匹配格式(.wav)，不硬编码 content-type；用 result 的真实采样率
            wav_path = out_dir / f"cand_{k}.wav"
            _write_wav(wav_path, result["audio"], int(result["sample_rate"]))
            rows.append(CandidateRow(
                fid=fid, k=k, audio=str(wav_path), text=p["text"],
                prompt_audio=p["prompt_audio"], prompt_text=p["prompt_text"],
                lang=p.get("lang", ""),
                num_steps=gp["num_steps"], guidance_scale=gp["guidance_scale"],
                speaker_scale=gp["speaker_scale"], seed=seed,
            ))
        print(f"  ✓ {fid}: {len(grid)} 候选")

    n = write_jsonl(PATHS.candidates_dir / "candidates_index.jsonl", rows)
    print(f"✓ 共 {n} 候选 -> {PATHS.candidates_dir}/candidates_index.jsonl")


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    """把 dots.tts 返回的张量写成 48kHz wav。soundfile 不接受 bf16，先转 float32。"""
    import soundfile as sf  # type: ignore
    arr = audio
    try:  # torch.Tensor → numpy
        arr = audio.float().cpu().squeeze().numpy()
    except AttributeError:
        pass
    sf.write(str(path), arr, sample_rate)


if __name__ == "__main__":
    main()
