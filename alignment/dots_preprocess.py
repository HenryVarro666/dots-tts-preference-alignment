"""
dots_preprocess.py — 唯一对接 dots.tts 内部预处理的「适配器」(seam)

做什么:
    把 (fid, text, audio) 转成 dots.tts 训练样本，并 collate 成模型可吃的 batch。
    所有与 dots.tts 内部的耦合都收敛到这一个文件。
怎么做（已对照源码核验，复用 dots.tts 真实组件）:
    - 每样本: `BasicTtsPipeline.process_sample({fid,text,audio})`
      (src/dots_tts/data/pipelines/tts_pipeline.py:69/161) —— 它做重采样/静音规整/对齐/分词/fbank，
      产出 input_ids/labels/loss_mask/sample/sample_length/num_*_tokens/fbank/fbank_length。
    - 成批: `PadCollator(tokenizer)` (src/dots_tts/data/collator.py:39) —— pad 成 batch dict。
    - 前向: `model.prepare_training_batch(batch)` 补 loss_masks，再 `model(prepared)` 得 LossTerms。
    数据超参从模型派生(与训练器 L210-215 的强校验一致)，不硬编码:
      train_audio_sample_rate = model.config.vocoder.sample_rate
      audio_samples_per_llm_token = model.hop_size * model.config.patch_size
为什么这么设计:
    复用 dots.tts 自己的 per-sample 处理 + collator，保证我们喂给模型的张量与它训练时**完全同构**；
    map-style + 确定性顺序，便于 DPO 把 win/lose 行对齐。
对实验意义:
    它产出的 batch 决定 `fm_loss` 是否正确 —— 方法成立的接缝。`check_fm_loss.py` 做自检。

前置条件: 可 import dots_tts；有 dots.tts-base 权重。
预期结果: process()→ 单样本 dict；collate()→ batch；prepare_and_forward()→ LossTerms。
⚠️ 本机只语法检查（不执行 import torch / dots_tts）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402


class DotsPreprocessor:
    """持有 model + 可独立 pickle 的 pipeline/collator（供 DataLoader worker 用）。"""

    def __init__(self, model):
        ensure_dots_tts_importable()
        from dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline  # type: ignore
        from dots_tts.data.collator import PadCollator  # type: ignore

        self.model = model
        self.tokenizer = model.tokenizer
        # 从模型派生数据契约（与 scripts/train_dots_tts.py:210-215 的强校验一致）
        sr = int(model.config.vocoder.sample_rate)
        aspt = int(model.hop_size) * int(model.config.patch_size)
        data_cfg = SimpleNamespace(train_audio_sample_rate=sr, audio_samples_per_llm_token=aspt)
        # pipeline/collator 只依赖 tokenizer + 超参 → 可 pickle，能丢进 DataLoader worker
        self.pipeline = BasicTtsPipeline(self.tokenizer, data_cfg)
        self.collator = PadCollator(self.tokenizer)
        self.sample_rate = sr
        self.audio_samples_per_llm_token = aspt

    @classmethod
    def from_pretrained(cls, ckpt: str) -> "DotsPreprocessor":
        ensure_dots_tts_importable()
        from dots_tts.models.dots_tts.model import DotsTtsModel  # type: ignore
        model = DotsTtsModel.from_pretrained(ckpt).eval()
        return cls(model)

    # --- 单样本 / 成批（这两个只用 pipeline/collator，可在 worker 进程里跑）---
    def process(self, fid: str, text: str, audio: str, source_name: str = "dpo") -> dict:
        return self.pipeline.process_sample(
            {"fid": str(fid), "text": text, "audio": audio, "source_name": source_name}
        )

    def collate(self, samples: list[dict]) -> dict:
        return self.collator(samples)

    # --- 前向（在主进程、GPU 上）---
    def prepare_and_forward(self, batch: dict):
        """batch(collate 产物) → prepare_training_batch 补 loss_masks → model(prepared) → LossTerms。"""
        prepared = self.model.prepare_training_batch(batch)
        return self.model(prepared)
