"""
data_pairs.py — 配对/单样本 DataLoader（把偏好对喂进 DPO 训练）

做什么:
    读 pairs/out/pairs_*.jsonl，对每对 (win,lose) 各跑 dots.tts 的 per-sample 处理，
    collate 成 batch{win:{...}, lose:{...}}。也提供 build_single_dataloader 给 open-SOAR。
怎么做（复用 dots.tts 真实组件，已核验）:
    用 `BasicTtsPipeline(tokenizer, data_cfg).process_sample({fid,text,audio})` 处理单样本，
    `PadCollator(tokenizer)` 成批。这两者只依赖 tokenizer + 两个超参 → 可 pickle，能进 worker。
    数据契约(sr / audio_samples_per_llm_token)由 trainer 从模型派生后传入，保证与模型完全一致。
为什么这么设计:
    win/lose 共享同一 text，差异只在生成音频 → DPO 似然比纯粹；map-style + 行对齐，确定性。
对实验意义:
    数据装配正确性决定偏好信号是否「干净」；字段错位会让 DPO 学到伪相关。

前置条件: pairs/out/pairs_*.jsonl；trainer 暴露 tokenizer/sample_rate/aspt。
预期结果: DataLoader；每个 batch = {"win": <collated>, "lose": <collated>}（单样本时 {"sample": <collated>}）。
⚠️ 本机只语法检查。默认 num_workers=0（最稳）；>0 需 dots_tts 在 worker 可导入。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, ensure_dots_tts_importable, read_jsonl  # noqa: E402


def _make_pipeline_collator(tokenizer, sample_rate: int, aspt: int):
    ensure_dots_tts_importable()
    from dots_tts.data.pipelines.tts_pipeline import BasicTtsPipeline  # type: ignore
    from dots_tts.data.collator import PadCollator  # type: ignore
    data_cfg = SimpleNamespace(train_audio_sample_rate=int(sample_rate),
                               audio_samples_per_llm_token=int(aspt))
    return BasicTtsPipeline(tokenizer, data_cfg), PadCollator(tokenizer)


def build_pair_dataloader(cfg: dict, trainer):
    from torch.utils.data import Dataset, DataLoader  # type: ignore

    pairs_file = cfg["dpo"].get("pairs_file", str(PATHS.pairs_dir / "pairs_best_worst.jsonl"))
    rows = list(read_jsonl(pairs_file))
    pipeline, collator = _make_pipeline_collator(trainer.tokenizer, trainer.sample_rate, trainer.aspt)

    class _PairDS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            win = pipeline.process_sample({"fid": f"{r['fid']}-w", "text": r["text"],
                                           "audio": r["win_audio"], "source_name": "dpo"})
            lose = pipeline.process_sample({"fid": f"{r['fid']}-l", "text": r["text"],
                                            "audio": r["lose_audio"], "source_name": "dpo"})
            return win, lose

    def _collate(batch):
        return {"win": collator([w for w, _ in batch]),
                "lose": collator([l for _, l in batch])}

    return DataLoader(_PairDS(), batch_size=cfg.get("micro_batch_size", 1), shuffle=True,
                      collate_fn=_collate, num_workers=cfg.get("num_workers", 0))


def build_single_dataloader(cfg: dict, trainer):
    """单样本 DataLoader（open-SOAR 用；偏轨样本在训练内构造）。"""
    from torch.utils.data import Dataset, DataLoader  # type: ignore

    manifest = cfg.get("train_manifest", str(PATHS.prompts_dir / "dots_manifest_train.jsonl"))
    rows = list(read_jsonl(manifest))
    pipeline, collator = _make_pipeline_collator(trainer.tokenizer, trainer.sample_rate, trainer.aspt)

    class _DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            audio = r.get("prompt_audio") or r.get("audio")
            return pipeline.process_sample({"fid": str(r["fid"]), "text": r["text"],
                                            "audio": audio, "source_name": "soar"})

    def _collate(batch):
        return {"sample": collator(batch)}

    return DataLoader(_DS(), batch_size=cfg.get("micro_batch_size", 1), shuffle=True,
                      collate_fn=_collate, num_workers=cfg.get("num_workers", 0))
