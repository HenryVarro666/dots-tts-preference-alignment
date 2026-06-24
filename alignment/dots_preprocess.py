"""
dots_preprocess.py — 唯一对接 dots.tts 内部预处理的「适配器」(seam)

做什么:
    把 (text, 音频, 参考) 转成 dots.tts 训练样本所需的张量:
    连续 latent(AudioVAE 编码)、g_cond(CAM++ x-vector)、token schedule、fm_patch_mask、
    以及可选的 ce/eos 监督项。所有与 dots.tts 内部耦合都收敛到这一个文件。
怎么做:
    复用 dots.tts 的:
      - `DotsTtsModel.from_pretrained` 取 audio_vae / speaker encoder / tokenizer；
      - AudioVAE.encode(wav) → latent；SpeakerXVectorFeatures(prompt) → g_cond；
      - 数据管线的 token 拼接(model.py 的 prepare_training_batch / data/ 的 tokenizing)。
为什么这么设计:
    把「与 dots.tts 内部的全部耦合」隔离到单一适配器 → 上游(data_pairs)无需了解内部，
    dots.tts 版本变动时只改这一处。
对实验意义:
    它产出的 `fm_patch_mask` 与 `target` 正是 DPO 损失的输入；正确性直接决定方法是否成立。

⚠️ 这是「待按 dots.tts 确切签名补全」的适配器：标了 TODO 的几处需对照
   `参考代码/dots.tts/src/dots_tts/models/dots_tts/model.py`(prepare_training_batch ~L1184,
   _compute_loss_terms ~L1156) 与 `data/` 管线补齐。本机只语法检查。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402


class DotsPreprocessor:
    def __init__(self, model):
        self.model = model
        # 取出 dots.tts 内部子模块（名称对照 modules/ 目录）
        self.audio_vae = getattr(model, "audio_vae", None) or getattr(model.core, "audio_vae", None)
        self.tokenizer = getattr(model, "tokenizer", None)

    @classmethod
    def from_pretrained(cls, ckpt: str) -> "DotsPreprocessor":
        ensure_dots_tts_importable()
        from dots_tts.models.dots_tts.model import DotsTtsModel  # type: ignore
        model = DotsTtsModel.from_pretrained(ckpt).eval()
        return cls(model)

    def prepare_sample(self, text: str, audio: str, prompt_audio: str,
                       prompt_text: str, lang: str) -> dict:
        """单条样本预处理。返回 DPO/训练前向所需的全部张量。

        TODO(对照 dots.tts):
          1. wav = load_48k(audio); latent = self.audio_vae.encode(wav)
          2. g_cond = speaker_features(prompt_audio)
          3. schedule = build_generation_schedule(tokenizer(text), latent)   # data/tokenizing.py
          4. fm_patch_mask = 音频 span 掩码                                    # model.py prepare_training_batch
        这里返回结构占位，键名与 losses_dpo / train_dots_tts_dpo 的读取一致。
        """
        import torch  # type: ignore
        # —— 占位实现：真实张量在 GPU 机器上由上述 TODO 步骤产出 ——
        return {
            "text": text, "audio": audio, "lang": lang,
            "latents": None,          # AudioVAE 编码后的连续 latent
            "g_cond": None,           # x-vector 说话人条件
            "input_ids": None,        # 文本+音频 token schedule
            "fm_patch_mask": None,    # (P,) 音频 span 掩码
            "_device_placeholder": torch.device("cpu"),
        }

    def collate(self, samples: list[dict]) -> dict:
        """把若干样本 pad 成 batch。可直接复用 dots.tts data/collator.py 的 collator；
        此处给出统一接口，内部 TODO 调用其 pad 逻辑。"""
        return {"samples": samples, "device": "cuda"}
