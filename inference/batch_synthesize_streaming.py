"""
batch_synthesize_streaming.py — 阶段⑥b：1T1A 流式批量合成 + 首包延迟采集

做什么:
    用 dots.tts 的 DoubleStreamingSession 逐 text-token 流式合成，拼接整段 wav，
    同时记录**首包延迟**(首个非 None chunk 的耗时)、总耗时、RTF，写 streaming_timing.jsonl。
怎么做:
    复用 `runtime_double_streaming.DoubleStreamingSession`(见 dots.tts example_double_streaming.py)。
    **流式克隆仅 x-vector**(prompt_text 非空会被 dots.tts 直接报错，已读源码 :117 确认) → 评测集用
    prompt-audio-only 条件。
为什么这么设计:
    用户把流式收益设为一等评测轴。要 claim「对齐增益迁移到流式」，必须在**真实流式路径**上
    重测 WER/SIM，而非拿离线结果替代——两条路径的解码 schedule 不同。
对实验意义:
    产出 RQ2 的证据：增益是否在流式下保持、首包延迟是否被对齐影响(预期不变)。

前置条件: 对齐 ckpt；评测清单(仅需 prompt_audio + text)。
预期结果: out_streaming/{treatment}/{fid}.wav + out_streaming/{treatment}/streaming_timing.jsonl。
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
    ap.add_argument("--speaker-scale", type=float, default=1.5)
    args = ap.parse_args()

    ensure_dots_tts_importable()
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore
    from dots_tts.runtime_double_streaming import DoubleStreamingSession  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    runtime = DotsTtsRuntime.from_pretrained(args.ckpt)
    tokenizer = runtime.tokenizer  # Qwen2.5 tokenizer（文本→token id）

    out_dir = Path(__file__).parent / "out_streaming" / args.treatment
    out_dir.mkdir(parents=True, exist_ok=True)

    timing = []
    for r in read_jsonl(args.manifest):
        # 流式克隆仅 x-vector：不传 prompt_text
        session = DoubleStreamingSession(
            runtime, prompt_audio_path=r["prompt_audio"],
            num_steps=args.num_steps, guidance_scale=args.guidance_scale,
            speaker_scale=args.speaker_scale,
        )
        token_ids = tokenizer.encode(r["text"])
        chunks = []
        t0 = time.perf_counter()
        first_packet = None
        for tid in token_ids:
            chunk = session.push_text_token(int(tid))
            if chunk is not None:
                if first_packet is None:
                    first_packet = time.perf_counter() - t0   # 首包延迟
                chunks.append(chunk)
        for chunk in session.finish_text():
            if chunk is not None:
                if first_packet is None:
                    first_packet = time.perf_counter() - t0
                chunks.append(chunk)
        total = time.perf_counter() - t0

        audio = torch.cat(chunks, dim=-1) if chunks else torch.zeros(1)
        wav = audio.float().cpu().squeeze().numpy()
        sf.write(str(out_dir / f"{r['fid']}.wav"), wav, runtime.sample_rate)
        dur = len(wav) / float(runtime.sample_rate)
        timing.append({"fid": r["fid"], "first_packet_s": first_packet,
                       "total_s": total, "audio_s": dur,
                       "rtf": (total / dur) if dur > 0 else None})

    n = write_jsonl(out_dir / "streaming_timing.jsonl", timing)
    fp = [t["first_packet_s"] for t in timing if t["first_packet_s"] is not None]
    print(f"✓ [{args.treatment}] 流式合成 {n} 条 -> {out_dir}")
    if fp:
        print(f"  首包延迟 中位数 = {sorted(fp)[len(fp)//2]*1000:.1f} ms")


if __name__ == "__main__":
    main()
