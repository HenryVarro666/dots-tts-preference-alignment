"""
data_pairs.py — 配对/单样本 DataLoader（把偏好对喂进 DPO 训练）

做什么:
    读 pairs/out/pairs_*.jsonl，对每对(win,lose)各做一次 dots.tts 预处理(AudioVAE 编码音频、
    Qwen2.5 分词文本、CAM++ x-vector)，collate 成 batch{win:{...}, lose:{...}}。
    也提供 build_single_dataloader 给 open-SOAR(单样本，偏轨在训练内构造)。
怎么做:
    复用 dots.tts 的数据预处理(JSONL 字段约定 {fid,text,audio} + 特殊音频 token)；
    最稳的复用方式是「产出 winner/loser 两份行对齐的清单，各跑既有 per-sample 预处理再 zip」，
    避免触碰 dots.tts 的 bucketing/streaming 内部。
为什么这么设计:
    win 与 lose 必须**共享同一 text+prompt 条件**(只差生成音频)，DPO 似然比才纯粹；
    用统一预处理保证两路样本除音频外完全一致。
对实验意义:
    数据装配的正确性决定偏好信号是否「干净」；字段错位会让 DPO 学到伪相关。

前置条件: pairs/out/pairs_*.jsonl；dots.tts 可导入。
预期结果: 返回可迭代 DataLoader；每个 batch 含 win/lose 已编码样本 + fm_patch_mask。
⚠️ 本机只语法检查（不执行 dots_tts 导入与音频编码）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, read_jsonl  # noqa: E402


def _preprocess_one(text: str, audio: str, prompt_audio: str, prompt_text: str, lang: str, dots_pre):
    """把一条 (text, 生成音频) 经 dots.tts 预处理成训练样本。

    复用点(见 dots.tts):
      - AudioVAE.encode(audio) -> 连续 latent（FM 的 target 来源）
      - CAM++/SpeakerXVectorFeatures(prompt_audio) -> g_cond
      - Qwen2.5 tokenizer + 7 个特殊音频 token 拼接 schedule
    这里用 dots_pre(注入的预处理器)封装上述步骤，返回含
    {input_ids, latents, g_cond, fm_patch_mask, ce_loss?, eos_loss?} 的 dict。
    """
    return dots_pre.prepare_sample(text=text, audio=audio,
                                   prompt_audio=prompt_audio, prompt_text=prompt_text, lang=lang)


def build_pair_dataloader(cfg: dict, accelerator):
    """配对 DataLoader：每个样本 = {win:{...}, lose:{...}}。"""
    import torch  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    from dots_preprocess import DotsPreprocessor  # type: ignore  # 见下方说明

    pairs_file = cfg["dpo"].get("pairs_file", str(PATHS.pairs_dir / "pairs_best_worst.jsonl"))
    rows = list(read_jsonl(pairs_file))
    dots_pre = DotsPreprocessor.from_pretrained(cfg["dpo"]["base_ckpt"])

    class _PairDS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            r = rows[i]
            shared = dict(text=r["text"], prompt_audio=r["prompt_audio"],
                          prompt_text=r["prompt_text"], lang=r["lang"])
            win = _preprocess_one(audio=r["win_audio"], dots_pre=dots_pre, **shared)
            lose = _preprocess_one(audio=r["lose_audio"], dots_pre=dots_pre, **shared)
            return {"win": win, "lose": lose}

    def _collate(batch):
        # 分别 pad win / lose（dots.tts 的 collator 可复用；此处占位为按键堆叠）
        return {
            "win": dots_pre.collate([b["win"] for b in batch]),
            "lose": dots_pre.collate([b["lose"] for b in batch]),
        }

    return DataLoader(_PairDS(), batch_size=cfg.get("micro_batch_size", 1),
                      shuffle=True, collate_fn=_collate, num_workers=cfg.get("num_workers", 2))


def build_single_dataloader(cfg: dict, accelerator):
    """单样本 DataLoader（open-SOAR 用；偏轨样本在训练内构造）。"""
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    from dots_preprocess import DotsPreprocessor  # type: ignore

    manifest = cfg.get("train_manifest", str(PATHS.prompts_dir / "ljspeech_train.jsonl"))
    rows = list(read_jsonl(manifest))
    dots_pre = DotsPreprocessor.from_pretrained(cfg["dpo"]["base_ckpt"])

    class _DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            r = rows[i]
            return {"sample": _preprocess_one(
                text=r["text"], audio=r["prompt_audio"], prompt_audio=r["prompt_audio"],
                prompt_text=r["prompt_text"], lang=r.get("lang", ""), dots_pre=dots_pre)}

    def _collate(batch):
        return {"sample": dots_pre.collate([b["sample"] for b in batch])}

    return DataLoader(_DS(), batch_size=cfg.get("micro_batch_size", 1),
                      shuffle=True, collate_fn=_collate, num_workers=cfg.get("num_workers", 2))
