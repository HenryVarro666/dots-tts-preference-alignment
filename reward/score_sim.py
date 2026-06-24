"""
score_sim.py — 阶段③b：候选的说话人相似度 SIM（音色像不像，越高越好）

做什么:    用 WavLM-large 说话人验证模型，算「候选音频 vs 参考音频」的嵌入余弦，写 sim.jsonl。
怎么做:    复用 seed-tts-eval 内 UniSpeech 的 WavLM-SV（同后端 → 与榜单可比）；惰性加载 ckpt。
为什么:    SIM 是零样本克隆的核心质量指标，是多目标奖励第二项；也是检验「reward-hacking 是否伤音色」的关键。
对实验意义: 只优化 WER 可能牺牲 SIM；把 SIM 纳入奖励/评测才能暴露这种权衡。

前置条件: 候选已生成；WavLM-SV ckpt 已放到 .ckpt/wavlm_large_finetune.pth(env/setup_env.sh)。
预期结果: reward/out/sim.jsonl，每行 {fid,k,sim}。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402

_DEFAULT_CKPT = PATHS.project_root / ".ckpt" / "wavlm_large_finetune.pth"


class _WavLMSV:
    """惰性加载 seed-tts-eval 的 WavLM 说话人验证模型。"""

    def __init__(self, ckpt: Path) -> None:
        self.ckpt = ckpt
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        # 把 seed-tts-eval 的 UniSpeech 说话人验证目录加入路径
        sv_dir = PATHS.seed_tts_eval / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
        sys.path.insert(0, str(sv_dir))
        import torch  # type: ignore
        from models import ECAPA_TDNN_SMALL  # type: ignore  # seed-tts-eval 内的 wrapper
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state = torch.load(str(self.ckpt), map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        model.eval()
        self._model = model

    def cosine(self, wav_a: str, wav_b: str) -> float:
        self._ensure()
        import torch  # type: ignore
        import torchaudio  # type: ignore
        import torch.nn.functional as F  # type: ignore

        def emb(path: str):
            wav, sr = torchaudio.load(path)
            if sr != 16000:  # SV 模型吃 16kHz
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                return self._model(wav)

        return float(F.cosine_similarity(emb(wav_a), emb(wav_b)).item())


def main() -> None:
    ap = argparse.ArgumentParser(description="候选 SIM 打分")
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    ap.add_argument("--wavlm-ckpt", default=str(_DEFAULT_CKPT))
    args = ap.parse_args()

    sv = _WavLMSV(Path(args.wavlm_ckpt))
    rows = []
    for c in read_jsonl(args.candidates):
        sim = sv.cosine(c["audio"], c["prompt_audio"])  # 候选 vs 参考音频
        rows.append({"fid": c["fid"], "k": c["k"], "sim": sim})
    n = write_jsonl(PATHS.rewards_dir / "sim.jsonl", rows)
    print(f"✓ SIM 打分 {n} 条 -> reward/out/sim.jsonl")


if __name__ == "__main__":
    main()
