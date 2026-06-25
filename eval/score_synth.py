"""
score_synth.py — 统一评测打分器：对一个合成目录算 WER/SIM/UTMOS（复用 reward 后端）

做什么:
    给定「合成目录(synth_dir，内含 {fid}.wav) + 评测清单(manifest，含 text/prompt_audio/lang)」，
    逐条算 WER(可懂)/SIM(音色相似)/UTMOS(自然度)，写 results/out/{treatment}.metrics.jsonl + 汇总。
怎么做（复用，不重造）:
    直接复用 reward 阶段**已在真机验证**的后端：
      - WER: reward.score_wer._ASR（transformers Whisper-large-v3 批量 / Paraformer），_normalize 同款归一化
      - SIM: reward.score_sim._EcapaBackend（SpeechBrain ECAPA；或 --sim-backend wavlm 对榜）vs prompt_audio
      - UTMOS: utmos_backend.UtmosScorer（torch.hub utmos22_strong）
为什么这么设计:
    评测必须与 reward 用**同一套后端**才自洽，且绕开 seed-tts-eval 的 s3prl/select.py 依赖坑（见 docs/05 B-4/B-5）。
对实验意义:
    这是各 treatment 在 held-out 评测集上的硬指标来源 —— RQ1(鲁棒/不伤 SIM)、RQ3(reward hacking) 的证据。

前置条件: 合成目录(batch_synthesize 产出)、评测清单。
预期结果: results/out/{treatment}.metrics.jsonl（每条）+ results/out/{treatment}.summary.json（均值+CI）。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "reward"))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402
from utmos_backend import UtmosScorer  # noqa: E402
from score_wer import _ASR, _normalize  # noqa: E402  复用 WER 后端
from score_sim import _EcapaBackend, _WavLMBackend, _DEFAULT_CKPT  # noqa: E402  复用 SIM 后端


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _bootstrap_ci(xs, iters=1000, alpha=0.05):
    """自举置信区间（小样本评测必报 CI，而非只报均值）。用确定性 LCG，避免依赖随机库。"""
    if len(xs) < 2:
        return (None, None)
    n = len(xs)
    seed = 12345
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            s += xs[seed % n]
        means.append(s / n)
    means.sort()
    lo = means[int(iters * alpha / 2)]
    hi = means[int(iters * (1 - alpha / 2))]
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="对合成目录算 WER/SIM/UTMOS")
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--manifest", required=True, help="评测清单 jsonl（含 fid/text/prompt_audio/lang）")
    ap.add_argument("--treatment", required=True, help="结果标签（输出文件名）")
    ap.add_argument("--sim-backend", choices=["ecapa", "wavlm"], default="ecapa")
    ap.add_argument("--wer-batch", type=int, default=16)
    args = ap.parse_args()

    import jiwer  # type: ignore

    manifest = {str(r["fid"]): r for r in read_jsonl(args.manifest)}
    synth = {p.stem: p for p in sorted(Path(args.synth_dir).glob("*.wav"))}
    fids = [f for f in manifest if f in synth]
    if not fids:
        raise SystemExit(f"{args.synth_dir} 里没有与清单匹配的 wav（合成是否成功？）")

    # --- WER（按 lang 分组，EN 批量）---
    asr = _ASR()
    en = [f for f in fids if (manifest[f].get("lang") or "en") != "zh"]
    zh = [f for f in fids if (manifest[f].get("lang") or "en") == "zh"]
    wer = {}
    if en:
        hyps = asr.transcribe_en_batch([str(synth[f]) for f in en], batch_size=args.wer_batch)
        for f, h in zip(en, hyps):
            wer[f] = float(jiwer.wer(_normalize(manifest[f]["text"], "en"), _normalize(h, "en")))
    for f in zh:
        h = _normalize(asr.transcribe_zh(str(synth[f])), "zh")
        wer[f] = float(jiwer.cer(_normalize(manifest[f]["text"], "zh"), h))

    # --- SIM（候选 vs 参考音频）---
    sim_backend = _EcapaBackend() if args.sim_backend == "ecapa" else _WavLMBackend(Path(_DEFAULT_CKPT))
    sim = {f: float(sim_backend.cosine(str(synth[f]), manifest[f]["prompt_audio"])) for f in fids}

    # --- UTMOS ---
    utm_scorer = UtmosScorer()
    utm = {f: float(utm_scorer.score(str(synth[f]))) for f in fids}

    rows = [{"fid": f, "wer": wer[f], "sim": sim[f], "utmos": utm[f]} for f in fids]
    write_jsonl(PATHS.results_dir / f"{args.treatment}.metrics.jsonl", rows)

    wer_l = [wer[f] for f in fids]
    sim_l = [sim[f] for f in fids]
    utm_l = [utm[f] for f in fids]
    summary = {
        "treatment": args.treatment, "n": len(fids),
        "wer": _mean(wer_l), "wer_ci": _bootstrap_ci(wer_l),
        "sim": _mean(sim_l), "sim_ci": _bootstrap_ci(sim_l),
        "utmos": _mean(utm_l), "utmos_ci": _bootstrap_ci(utm_l),
        "sim_backend": args.sim_backend,
    }
    (PATHS.results_dir / f"{args.treatment}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ [{args.treatment}] n={len(fids)}  WER={summary['wer']:.4f}  SIM={summary['sim']:.4f}  UTMOS={summary['utmos']:.3f}")
    print(f"  -> results/out/{args.treatment}.{{metrics.jsonl,summary.json}}")


if __name__ == "__main__":
    main()
