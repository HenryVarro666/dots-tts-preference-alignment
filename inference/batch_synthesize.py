"""
batch_synthesize.py — 阶段⑥a：离线批量合成（评测前置）

做什么:    用给定 checkpoint 对评测集逐条离线合成，输出 wav，供 eval/ 算 WER/SIM/UTMOS。
怎么做:    复用 `DotsTtsRuntime.generate(...)`；按 treatment 命名输出目录(便于溯源)。
为什么:    评测必须在「同一评测集、同一推理设置、只换权重」下进行，才能归因到对齐本身。
对实验意义: 离线模式是 RQ1/RQ3 的主战场；部署 NFE 作为参数暴露(接 Phase B 的 NFE 迁移消融)。

前置条件: 对齐 ckpt、评测清单(seed-tts-eval 或 data/prompts/*_eval.jsonl)。
预期结果: out/{treatment}/{fid}.wav。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="离线批量合成")
    ap.add_argument("--ckpt", required=True, help="对齐后 checkpoint 或 dots.tts-base")
    ap.add_argument("--manifest", required=True, help="评测清单 jsonl {fid,text,prompt_audio,prompt_text,lang}")
    ap.add_argument("--treatment", required=True, help="base/sft/open_soar/dpo_multiobj/dpo_weronly")
    ap.add_argument("--num-steps", type=int, default=10, help="部署 NFE（NFE 迁移消融用）")
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--speaker-scale", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ensure_dots_tts_importable()
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore
    import soundfile as sf  # type: ignore

    out_dir = Path(args.out or (Path(__file__).parent / "out" / args.treatment))
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = DotsTtsRuntime.from_pretrained(args.ckpt)

    n = 0
    for r in read_jsonl(args.manifest):
        res = runtime.generate(
            text=r["text"], prompt_audio_path=r["prompt_audio"], prompt_text=r.get("prompt_text", ""),
            language=r.get("lang"), num_steps=args.num_steps,
            guidance_scale=args.guidance_scale, speaker_scale=args.speaker_scale, normalize_text=True,
        )
        audio = res["audio"]
        try:
            audio = audio.float().cpu().squeeze().numpy()
        except AttributeError:
            pass
        sf.write(str(out_dir / f"{r['fid']}.wav"), audio, int(res["sample_rate"]))
        n += 1
    print(f"✓ [{args.treatment}] 离线合成 {n} 条 (NFE={args.num_steps}) -> {out_dir}")


if __name__ == "__main__":
    main()
