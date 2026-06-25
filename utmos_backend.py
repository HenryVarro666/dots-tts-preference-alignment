"""
utmos_backend.py — UTMOS 自然度打分后端（reward 与 eval 共用，单一事实源）

做什么:    提供 UtmosScorer：输入 wav 路径 → 预测 UTMOS（自然度 MOS 代理，越高越好）。
怎么做:    用 torch.hub 加载 `tarepan/SpeechMOS` 的 `utmos22_strong`——UTMOS 的标准轻量实现，
           自包含、无需 SSL backbone，只依赖 torch + librosa（均已装）。惰性加载、可上 GPU。
为什么这么选:
    PyPI 的 `speechmos` 包是微软 DNSMOS/PLCMOS，**不含 utmos**；torch.hub 这条是社区通用的
    UTMOS 入口，零额外 pip 依赖、与多数 TTS 论文报告的 UTMOS 同源，可比性好。
对实验意义:
    UTMOS 是多目标奖励第三项，也是检测 reward-hacking（WER↓ 但自然度↓）的关键探针。

前置条件: torch + librosa；首次会从 GitHub 下载 utmos22_strong（需联网）。
预期结果: UtmosScorer().score(path) -> float（约 1~5）。
⚠️ 本机只语法检查（torch.hub 不在 import 期执行）。
"""

from __future__ import annotations


class UtmosScorer:
    """torch.hub utmos22_strong 包装；惰性加载、自动用 GPU。"""

    def __init__(self, repo: str = "tarepan/SpeechMOS", entry: str = "utmos22_strong") -> None:
        self._repo = repo
        self._entry = entry
        self._model = None
        self._cuda = False

    def _ensure(self):
        if self._model is not None:
            return
        import torch  # type: ignore
        model = torch.hub.load(self._repo, self._entry, trust_repo=True)
        model.eval()
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            model = model.cuda()
        self._model = model

    def score(self, wav_path: str) -> float:
        """对单个 wav 算 UTMOS。utmos22_strong 期望 16kHz 单声道波形 + 采样率参数。"""
        import torch  # type: ignore
        import librosa  # type: ignore
        self._ensure()
        wav, _ = librosa.load(wav_path, sr=16000)
        x = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
        if self._cuda:
            x = x.cuda()
        with torch.no_grad():
            s = self._model(x, 16000)
        return float(s.squeeze().item())
