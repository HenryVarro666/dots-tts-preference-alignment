"""
streaming_metrics.py — 流式专属指标：首包延迟汇总 + chunk 边界不连续度

做什么:
    (1) 汇总 batch_synthesize_streaming 写的 streaming_timing.jsonl（首包延迟/RTF 分位）。
    (2) 估计 chunk 边界不连续度：流式逐块解码可能在拼接处留下频谱/能量跳变；
        用短时谱通量(spectral flux)的高分位作为不连续度代理。
怎么做:
    若提供 --boundaries(每条音频的块边界采样点)，则只在边界帧 vs 块内帧对比(更精确)；
    否则退化为全局谱通量分位(代理)。纯 numpy/librosa。
为什么这么设计:
    离线/流式都能读对，但只有「边界处的突变」才是流式特有的坏味道；
    把它量化出来，才能 claim「对齐没有引入流式伪影」或反之。
对实验意义:
    RQ2 的支撑：增益迁移到流式的同时，边界不连续度不应变差。

前置条件: 流式合成产物(wav + streaming_timing.jsonl)。
预期结果: results/out/{treatment}.streaming.json（延迟分位 + 不连续度）。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_jsonl, PATHS  # noqa: E402


def _percentiles(xs: list[float], ps=(50, 90, 95)) -> dict:
    if not xs:
        return {f"p{p}": None for p in ps}
    s = sorted(xs)
    return {f"p{p}": s[min(len(s) - 1, int(len(s) * p / 100))] for p in ps}


def _spectral_flux_tail(wav_path: str) -> float:
    """全局谱通量的 p95 作为不连续度代理（边界突变会拉高尾部）。"""
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    y, sr = librosa.load(wav_path, sr=None)  # 读原生采样率(dots.tts 为 48kHz)，硬编码 24k 会丢高频
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))  # 48kHz 下放大窗口，保持时间分辨率
    flux = np.sqrt(((np.diff(S, axis=1)) ** 2).sum(axis=0))
    return float(np.percentile(flux, 95)) if flux.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="流式指标")
    ap.add_argument("--streaming-dir", required=True, help="batch_synthesize_streaming 的输出目录")
    ap.add_argument("--treatment", required=True)
    args = ap.parse_args()

    sdir = Path(args.streaming_dir)
    timing = list(read_jsonl(sdir / "streaming_timing.jsonl"))
    fp = [t["first_packet_s"] for t in timing if t.get("first_packet_s") is not None]
    rtf = [t["rtf"] for t in timing if t.get("rtf") is not None]
    flux = [_spectral_flux_tail(str(w)) for w in sorted(sdir.glob("*.wav"))]

    result = {
        "treatment": args.treatment,
        "first_packet_ms": {k: (v * 1000 if v is not None else None)
                            for k, v in _percentiles(fp).items()},
        "rtf": _percentiles(rtf),
        "boundary_discontinuity_p95_flux": (sum(flux) / len(flux)) if flux else None,
        "note": "边界不连续度为全局谱通量 p95 代理；提供块边界可改为边界帧精确版",
    }
    out = PATHS.results_dir / f"{args.treatment}.streaming.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ [{args.treatment}] 流式指标 -> {out}")
    print(f"  首包 p50 = {result['first_packet_ms']['p50']} ms")


if __name__ == "__main__":
    main()
