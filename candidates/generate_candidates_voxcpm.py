"""
generate_candidates_voxcpm.py — ★5：用 VoxCPM2 **自采样** K 个候选（让跨架构复现升级为真正独立的 N=2）

背景（当前 VoxCPM2 复现是「依赖证据」，非独立 N=2）:
    论文 §6 的 VoxCPM2 复现是拿 **dots 生成的同一批 pair** 喂给 VoxCPM2 的 flow loss 打分——predictor
    (reward gap) 共享、两系数统计依赖，只能算「第二个似然模型在同样的 pair 上同意」，**不是独立重生**。
    要把它升成真正独立的架构级 N=2：让 **VoxCPM2 自己生成候选** → 同一套 automatic-metric reward 打分
    → 构造 **VoxCPM2-原生** 偏好对 → 在原生对上重跑探针。若反相关在原生对上仍显著，则 generator、
    pair、reward-gap 全部独立于 dots ⇒ 独立复现（§6 可从「supporting/dependent」升为「independent」）。

做什么:
    镜像 candidates/generate_candidates.py 的结构，但把生成后端换成 VoxCPM2：读 prompts，对每条
    prompt 用 VoxCPM2 自采样 K 个**有差异**的候选（逐 k 换随机种子，VoxCPM2 的扩散采样本就随机），
    写同款 CandidateRow schema 到 candidates/out/candidates_index_voxcpm.jsonl + wav 到 candidates/out/voxcpm/。

为什么这么设计:
    下游 reward(score_wer/sim/utmos) / pairs(build_preference_pairs) / probe(probe_fm_reward_voxcpm)
    全部按 fid 走 CandidateRow/RewardRow/PairRow——只要原生候选写同款 schema，下游脚本零改动即可复用
    （只把 --candidates/--rewards/--pairs 指向 voxcpm 后缀的文件）。这正是 common.py 单一 schema 的价值。

对实验意义:
    产出 VoxCPM2-原生候选池 = 独立 N=2 复现的第一块、也是唯一需要新代码的一块。

用法:
    python candidates/generate_candidates_voxcpm.py --prompts data/prompts/<同 dots 的 prompt 集>.jsonl \\
        --k 6 [--limit 0] [--model openbmb/VoxCPM2]
前置条件: env/setup_env.sh 已挂 VoxCPM(VoxCPM2)；已下载 VoxCPM2 权重；prompts 就绪。
预期结果: candidates/out/candidates_index_voxcpm.jsonl + candidates/out/voxcpm/{fid}_k{k}.wav。
注意:
    ⚠️ 本机(macOS/无 GPU/未下 VoxCPM2 权重)只 py_compile，不能真跑。
    ⚠️【唯一需在 GPU 机器核对的 seam】`_voxcpm_synthesize()` 里的 VoxCPM2 TTS-generate 调用按公开
       `voxcpm` 包的已知约定写（VoxCPM.from_pretrained(...).generate(text, prompt_wav_path, prompt_text, ...)）；
       VoxCPM2 的确切签名/返回（音频 ndarray vs 路径、采样率、cfg/步数参数名）须对照已装包确认后微调——
       这是本脚本唯一无法从本机代码核验的外部调用（遵循仓库「不可验证处显式标注」纪律，见 docs/05）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, CandidateRow, read_jsonl, write_jsonl  # noqa: E402

# VoxCPM 默认推理超参（用于填 CandidateRow 的 schema 占位；语义对 VoxCPM2 = 扩散步数 / CFG）
_DEFAULT_STEPS = 10      # VoxCPM2 inference timesteps（映射到 CandidateRow.num_steps）
_DEFAULT_CFG = 2.0       # VoxCPM2 cfg_value（映射到 CandidateRow.guidance_scale）
_SEED_BASE = 20260629


def _voxcpm_synthesize(model, text: str, prompt_audio: str, prompt_text: str,
                       steps: int, cfg: float):
    """
    ⚠️ SEAM（GPU 机器上核对）：调用 VoxCPM2 做一次 zero-shot 克隆合成，返回 (audio_ndarray, sample_rate)。
    下面按公开 `voxcpm` 包的约定写；VoxCPM2 实际签名可能不同，须对照已装包微调：
      - 方法名可能是 generate / inference / tts；
      - 返回可能是 ndarray、(ndarray, sr)、或写好的 wav 路径；
      - 步数/CFG 的关键字名可能是 inference_timesteps / cfg_value 等。
    本函数是本实验唯一无法从本机代码静态核验的外部调用。
    """
    wav = model.generate(
        text=text,
        prompt_wav_path=prompt_audio,
        prompt_text=prompt_text,
        inference_timesteps=steps,
        cfg_value=cfg,
    )
    # 兼容 (audio, sr) / audio 两种返回；采样率缺省按 VoxCPM2 输出 16k（写盘按实际返回为准）。
    if isinstance(wav, tuple):
        audio, sr = wav
    else:
        audio, sr = wav, 16000
    return audio, int(sr)


def main() -> None:
    ap = argparse.ArgumentParser(description="★5 VoxCPM2 自采样候选生成（独立 N=2 复现的生成端）")
    ap.add_argument("--prompts", required=True, help="data/prompts/*.jsonl（建议与 dots 188 对同一 prompt 集）")
    ap.add_argument("--model", default="openbmb/VoxCPM2")
    ap.add_argument("--k", type=int, default=6, help="每 prompt 候选数")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条 prompt（冒烟用）")
    ap.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    ap.add_argument("--cfg", type=float, default=_DEFAULT_CFG)
    ap.add_argument("--out-index", default=str(PATHS.candidates_dir / "candidates_index_voxcpm.jsonl"))
    ap.add_argument("--out-audio-dir", default=str(PATHS.candidates_dir / "voxcpm"))
    args = ap.parse_args()

    # 仅在真跑时才 import 重依赖（本机 py_compile 不触发）。
    from alignment.voxcpm_preprocess import ensure_voxcpm_importable  # noqa: E402
    import soundfile as sf  # noqa: E402
    ensure_voxcpm_importable()
    import torch  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型加载镜像已核验的 alignment/voxcpm_preprocess.from_pretrained：
    #   load_denoiser=False(免 modelscope/zipenhancer 依赖) + optimize=False(免 triton)。
    #   facade 即高层 TTS API，带 .generate()（其确切签名仍是下方 _voxcpm_synthesize 的 ⚠️ SEAM）。
    from voxcpm.core import VoxCPM  # type: ignore
    model = VoxCPM.from_pretrained(hf_model_id=args.model, load_denoiser=False, optimize=False, device=dev)

    audio_dir = Path(args.out_audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(read_jsonl(args.prompts))
    if args.limit > 0:
        prompts = prompts[: args.limit]

    rows: list[CandidateRow] = []
    n_fail = 0
    for p in prompts:
        fid = p["fid"]
        for k in range(args.k):
            seed = _SEED_BASE + k
            torch.manual_seed(seed)  # 逐 k 换种子 → K 个有差异候选（VoxCPM2 采样随机）
            wav_path = audio_dir / f"{fid}_k{k}.wav"
            try:
                audio, sr = _voxcpm_synthesize(
                    model, p["text"], p["prompt_audio"], p.get("prompt_text", ""),
                    steps=args.steps, cfg=args.cfg,
                )
                sf.write(str(wav_path), audio, sr)
            except Exception as e:  # noqa: BLE001  单候选失败不炸全程（镜像 dots 的 immediate-EOS 容错）
                n_fail += 1
                if n_fail <= 5:
                    print(f"  ⚠️ 生成失败 fid={fid} k={k}: {e}")
                continue
            rows.append(CandidateRow(
                fid=fid, k=k, audio=str(wav_path), text=p["text"],
                prompt_audio=p["prompt_audio"], prompt_text=p.get("prompt_text", ""),
                lang=p.get("lang", "en"),
                num_steps=args.steps, guidance_scale=args.cfg, speaker_scale=1.0,  # speaker_scale 对 VoxCPM2 不适用，占位
                seed=seed,
            ))

    n_written = write_jsonl(args.out_index, rows)
    print(f"[★5] VoxCPM2 候选: 写 {n_written} 条 -> {args.out_index}  (失败 {n_fail})")
    print(f"[★5] 下一步: score_wer/sim/utmos --candidates {args.out_index} -> aggregate_reward --out rewards_voxcpm.jsonl")
    print(f"        -> build_preference_pairs --rewards rewards_voxcpm.jsonl --out pairs_voxcpm.jsonl -> probe_fm_reward_voxcpm")


if __name__ == "__main__":
    main()
