"""
rescore_metric.py — ★2：第二个独立自然度指标交叉验证（排除「open-SOAR 增益 / UTMOS 反相关是 UTMOS 单指标伪影」）

背景（模拟审稿 claim 9 的替代解释）:
    UTMOS 把音频重采样到 16kHz 再打分（utmos_backend.py 已 librosa.load(sr=16000)），所有臂口径一致 → 48k vs 16k
    band-mismatch **不是混淆**（不会制造相对差异）；残留限制只是 UTMOS 看不到 16–24kHz、对 48k 自然度均匀低估。
    但 UTMOS 仍是**单一指标**。换一个**独立**的 MOS 预测器（默认 DNSMOS）重打分臂音频，看 open-SOAR 的自然度
    增益是否复现 → 若复现，则结论非「UTMOS-specific 伪影」。

做什么（纯重打分、无新合成）:
    对每个臂 glob `inference/out/{arm}/*.wav`，用第二指标逐条打分，报每臂均值 + base↔open_soar 差。
判据（回填 §5.1 / Limitations）:
    open_soar 在第二指标上仍 > base ⇒ 自然度增益跨指标稳健、非 UTMOS 伪影；否则收回「自然度增益」、标为 UTMOS-specific。
用法: python eval/rescore_metric.py --arms base_nfe10 open_soar_nfe10 --metric dnsmos --out results/out/metric_xcheck.txt
前置: inference/out/{arm}/*.wav + 第二指标后端（DNSMOS：`pip install speechmos`）。
本机只语法检查；真实运行需音频 + 指标后端。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _load_metric(name: str):
    """返回 score(path)->float。默认 DNSMOS(speechmos)；可在此加别的独立 MOS 后端。"""
    if name == "dnsmos":
        try:
            from speechmos import dnsmos  # type: ignore
        except Exception as e:  # noqa: BLE001
            hint = "pip install onnxruntime" if "onnxruntime" in str(e) else "pip install speechmos onnxruntime"
            raise SystemExit(f"DNSMOS 后端加载失败（{e}）→ {hint}")

        def score(path: str) -> float:
            try:
                r = dnsmos.run(path, sr=16000)
            except Exception:  # noqa: BLE001
                import librosa  # type: ignore
                wav, _ = librosa.load(path, sr=16000)
                r = dnsmos.run(wav, sr=16000)
            if isinstance(r, dict):
                for k in ("ovrl_mos", "OVRL", "p808_mos", "mos"):
                    if k in r:
                        return float(r[k])
                return float(next(iter(r.values())))
            return float(r)
        return score
    raise SystemExit(f"未知 --metric {name}（目前支持 dnsmos；在 _load_metric 里加别的后端即可）")


def main() -> None:
    ap = argparse.ArgumentParser(description="★2 第二指标交叉验证")
    ap.add_argument("--arms", nargs="+", default=["base_nfe10", "open_soar_nfe10"],
                    help="臂 tag（对应 inference/out/{tag}/*.wav）")
    ap.add_argument("--synth-root", default=str(PATHS.project_root / "inference" / "out"))
    ap.add_argument("--metric", default="dnsmos")
    ap.add_argument("--ref", default="base_nfe10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    score = _load_metric(args.metric)
    emit(f"==== [★2] 第二指标({args.metric}) 交叉验证：臂自然度是否非 UTMOS-specific ====\n")
    means: dict[str, float] = {}
    for arm in args.arms:
        wavs = sorted((Path(args.synth_root) / arm).glob("*.wav"))
        if not wavs:
            emit(f"  ⚠️ {arm}: 无 wav（{Path(args.synth_root)/arm}）")
            continue
        vals = []
        for i, w in enumerate(wavs):
            try:
                vals.append(score(str(w)))
            except Exception as e:  # noqa: BLE001
                if i == 0:
                    emit(f"  ⚠️ {arm}: 打分异常 {e}")
        means[arm] = _mean(vals)
        emit(f"  {arm:22s} {args.metric} 均值={means[arm]:.3f}  (n={len(vals)})")

    if args.ref in means:
        emit(f"\n  对照 {args.ref}={means[args.ref]:.3f}")
        for arm, m in means.items():
            if arm == args.ref:
                continue
            d = m - means[args.ref]
            emit(f"  {arm}: Δ{args.metric}={d:+.3f} → {'仍高于 base（自然度增益跨指标稳健、非 UTMOS 伪影）' if d > 0 else '不高于 base（增益可能 UTMOS-specific，须收回）'}")
    emit("\n==== 落表：open_soar 第二指标仍>base ⇒ §5.1/Limitations 加『跨指标稳健』；否则收回自然度增益。 ====")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[--out] 报告已归档 → {out}")


if __name__ == "__main__":
    main()
