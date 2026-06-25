"""
score_sim.py — 阶段③b：候选的说话人相似度 SIM（音色像不像，越高越好）

做什么:    算「候选音频 vs 参考音频」的说话人嵌入余弦，写 sim.jsonl。
怎么做:    两种后端可选——
    - ecapa（默认，低摩擦）：SpeechBrain `spkrec-ecapa-voxceleb`，pip 装、自动下模型，
      无需手动 ckpt、无 s3prl；适合「比较各 treatment 的 SIM 差值」（我们的 claim 是 delta）。
    - wavlm（对榜用）：seed-tts-eval 的 WavLM-large SV（与 Seed-TTS-eval 榜单同源），
      需手动放 `.ckpt/wavlm_large_finetune.pth` + s3prl 等老依赖。
为什么这么设计:
    seed-tts-eval 的 WavLM 路径依赖重(s3prl/fire/旧 sklearn)且要 Google Drive ckpt，
    冒烟阶段摩擦大；ECAPA 后端让 SIM「现在就能跑出来」，对内部 treatment 对比足够。
对实验意义:
    SIM 是零样本克隆核心指标，也是检验「reward-hacking 是否伤音色」的关键；后端只要**全程一致**，
    treatment 间的 SIM 差值就可比（绝对值对榜才需 wavlm）。

前置条件: 候选已生成。ecapa 后端需 `pip install speechbrain`；wavlm 后端需 ckpt + s3prl。
预期结果: reward/out/sim.jsonl，每行 {fid,k,sim}。缺 ckpt（wavlm）时优雅跳过、不崩。
⚠️ 本机只语法检查。

【修复记录】此前用 `sys.path.insert(0, sv_dir)` 把 seed-tts-eval 目录插到 sys.path 最前，
其中的 `select.py` 顶替了标准库 `select`（torch→subprocess→import select 命中它→import matplotlib 崩）。
现：① 顶部先 import torch（标准库 select 先入缓存）；② wavlm 后端改用 `sys.path.append`（不抢占）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 顶部先导入 torch/torchaudio：让标准库 select/subprocess 先解析并缓存，
# 避免之后任何 sys.path 改动把 seed-tts-eval/.../select.py 误当标准库 select。
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl, write_jsonl  # noqa: E402

_DEFAULT_CKPT = PATHS.project_root / ".ckpt" / "wavlm_large_finetune.pth"


def _load_16k_mono(path: str) -> "torch.Tensor":
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav  # (1, T)


class _EcapaBackend:
    """SpeechBrain ECAPA-TDNN（默认，低摩擦、自动下载，无需手动 ckpt）。"""

    def __init__(self) -> None:
        self._clf = None

    def _ensure(self):
        if self._clf is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": dev},
            savedir=str(PATHS.project_root / ".cache" / "spkrec-ecapa"),
        )

    def _emb(self, path: str):
        self._ensure()
        wav = _load_16k_mono(path)
        if torch.cuda.is_available():
            wav = wav.cuda()
        with torch.no_grad():
            e = self._clf.encode_batch(wav)  # (1, 1, D)
        return e.reshape(-1)

    def cosine(self, a: str, b: str) -> float:
        return float(F.cosine_similarity(self._emb(a).unsqueeze(0), self._emb(b).unsqueeze(0)).item())


class _WavLMBackend:
    """seed-tts-eval 的 WavLM-large SV（对榜用；需 ckpt + s3prl 等老依赖）。"""

    def __init__(self, ckpt: Path) -> None:
        self.ckpt = Path(ckpt)
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        if not self.ckpt.exists():
            raise FileNotFoundError(
                f"WavLM ckpt 不存在: {self.ckpt}\n"
                f"  → 从 seed-tts-eval README 的 Google Drive 下 wavlm_large_finetune.pth 放到此处；\n"
                f"  → 或改用默认低摩擦后端：score_sim.py --backend ecapa")
        # 关键：append（不 insert(0)），避免 seed-tts-eval 目录里的 select.py 抢占标准库 select
        sv_dir = PATHS.seed_tts_eval / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
        sys.path.append(str(sv_dir))
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # type: ignore  # 真实路径含 .ecapa_tdnn
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state = torch.load(str(self.ckpt), map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model

    def _emb(self, path: str):
        self._ensure()
        wav = _load_16k_mono(path)
        if torch.cuda.is_available():
            wav = wav.cuda()
        with torch.no_grad():
            return self._model(wav).reshape(-1)

    def cosine(self, a: str, b: str) -> float:
        return float(F.cosine_similarity(self._emb(a).unsqueeze(0), self._emb(b).unsqueeze(0)).item())


def main() -> None:
    ap = argparse.ArgumentParser(description="候选 SIM 打分")
    ap.add_argument("--candidates", default=str(PATHS.candidates_dir / "candidates_index.jsonl"))
    ap.add_argument("--backend", choices=["ecapa", "wavlm"], default="ecapa",
                    help="ecapa=SpeechBrain(默认,低摩擦); wavlm=seed-tts-eval(对榜,需 ckpt)")
    ap.add_argument("--wavlm-ckpt", default=str(_DEFAULT_CKPT))
    args = ap.parse_args()

    backend = _EcapaBackend() if args.backend == "ecapa" else _WavLMBackend(Path(args.wavlm_ckpt))
    rows = []
    for c in read_jsonl(args.candidates):
        sim = backend.cosine(c["audio"], c["prompt_audio"])  # 候选 vs 参考音频
        rows.append({"fid": c["fid"], "k": c["k"], "sim": sim})
    n = write_jsonl(PATHS.rewards_dir / "sim.jsonl", rows)
    print(f"✓ SIM 打分 {n} 条（backend={args.backend}）-> reward/out/sim.jsonl")
    if args.backend == "ecapa":
        print("  注：ecapa 与 Seed-TTS-eval 榜单的 WavLM 数字不可直接横比；treatment 间差值可比。")


if __name__ == "__main__":
    main()
