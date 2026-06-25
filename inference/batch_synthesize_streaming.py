"""
batch_synthesize_streaming.py — 阶段⑥b：1T1A 流式批量合成 + 首包延迟采集

做什么:
    用 dots.tts 双流式 API 逐 text-token 合成，拼接整段 wav，并记录**首包延迟**(首个非 None chunk)、
    总耗时、RTF，写 out_streaming/{treatment}/{fid}.wav + streaming_timing.jsonl。
怎么做（严格对齐官方 scripts/example_double_streaming.py）:
    - 用 `DotsTtsRuntimeDoubleStreaming.from_pretrained(ckpt)`（不是 DotsTtsRuntime + DoubleStreamingSession）。
    - tokenize: `runtime.model.tokenizer.encode(text, add_special_tokens=False)`（注意是 runtime.model.tokenizer）。
    - 会话: `runtime.start_double_streaming(prompt_audio_path=, ode_method=, num_steps=, guidance_scale=, eos_threshold=)`。
      **流式仅 x-vector 克隆**（无 prompt_text）。
    - 逐 token `session.push_text_token(tid)` 收 chunk；`session.finish_text()` flush 尾部。
为什么这么设计:
    流式收益是一等评测轴。要 claim「对齐增益迁移到流式」，必须走**真实流式路径**重测，而非拿离线替代。
对实验意义:
    产出 RQ2 证据：增益是否在流式下保持、首包延迟是否被对齐影响(预期不变)。

前置条件: 对齐 ckpt（dots.tts from_pretrained 格式）；评测清单(需 prompt_audio + text)。
预期结果: out_streaming/{treatment}/{fid}.wav + .../streaming_timing.jsonl。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="1T1A 流式批量合成")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--speaker-scale", type=float, default=1.5)  # 与离线 batch_synthesize 接口对称
    ap.add_argument("--eos-threshold", type=float, default=0.8)
    ap.add_argument("--ode-method", default="euler")
    ap.add_argument("--normalize-text", action="store_true", default=True)
    args = ap.parse_args()

    ensure_dots_tts_importable()
    from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming  # type: ignore
    from dots_tts.utils.text import normalize_text  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    # ckpt 可为 HF repo id（base）或本地 from_pretrained 目录（trained）；该 from_pretrained 自带解析
    runtime = DotsTtsRuntimeDoubleStreaming.from_pretrained(args.ckpt)
    out_dir = Path(__file__).parent / "out_streaming" / args.treatment
    out_dir.mkdir(parents=True, exist_ok=True)

    timing = []
    for r in read_jsonl(args.manifest):
        text = normalize_text(r["text"].strip()) if args.normalize_text else r["text"].strip()
        token_ids = runtime.model.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        # 流式仅 x-vector 克隆：只给 prompt_audio，不给 prompt_text
        session = runtime.start_double_streaming(
            prompt_audio_path=r.get("prompt_audio"),
            ode_method=args.ode_method, num_steps=args.num_steps,
            guidance_scale=args.guidance_scale, speaker_scale=args.speaker_scale,
            eos_threshold=args.eos_threshold,
        )
        chunks = []
        t0 = time.perf_counter()
        first_packet = None
        for tid in token_ids:
            chunk = session.push_text_token(int(tid))
            if chunk is not None:
                if first_packet is None:
                    first_packet = time.perf_counter() - t0   # 首包延迟
                chunks.append(chunk.detach().cpu())
        for chunk in session.finish_text():
            if chunk is not None:
                chunks.append(chunk.detach().cpu())
        total = time.perf_counter() - t0

        if not chunks:
            print(f"  ⚠️ {r['fid']}: 无音频 chunk，跳过"); continue
        audio = torch.cat(chunks, dim=-1)
        wav = audio.float().squeeze().numpy()
        sf.write(str(out_dir / f"{r['fid']}.wav"), wav, runtime.sample_rate)
        dur = len(wav) / float(runtime.sample_rate)
        timing.append({"fid": r["fid"], "first_packet_s": first_packet,
                       "total_s": total, "audio_s": dur,
                       "rtf": (total / dur) if dur > 0 else None})

    n = write_jsonl(out_dir / "streaming_timing.jsonl", timing)
    print(f"✓ [{args.treatment}] 流式合成 {n} 条 -> {out_dir}")
    fp = [t["first_packet_s"] for t in timing if t["first_packet_s"] is not None]
    if fp:
        print(f"  首包延迟 中位数 = {sorted(fp)[len(fp)//2]*1000:.1f} ms")


if __name__ == "__main__":
    main()
