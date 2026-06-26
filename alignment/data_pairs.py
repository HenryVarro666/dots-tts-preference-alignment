"""
data_pairs.py — 配对/单样本 DataLoader（把偏好对喂进 DPO 训练）

做什么:
    读 pairs/out/pairs_*.jsonl，对每对 (win,lose) 各跑 dots.tts 的 per-sample 处理，
    collate 成 batch{win:{...}, lose:{...}}。也提供 build_single_dataloader 给 open-SOAR、
    build_weighted_dataloader 给 reward-weighted 回归臂(X6b/X7：单样本 + 每样本 reward 权)。
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
from common import PATHS, ensure_dots_tts_importable, read_jsonl, zscore  # noqa: E402


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


def build_weighted_dataloader(cfg: dict, trainer):
    """reward-weighted 回归用单样本 DataLoader（X6b/X7）。仿 build_single_dataloader，**勿动它**。

    与 build_single 唯一区别: 额外按 **audio 路径** join reward/out/rewards*.jsonl，给每个样本带一个
    标量 reward（跨 manifest z-normalize 后），collate 进 batch["reward"]。训练侧
    (train_reward_weighted.RewardWeightedTrainer) 据此把 supervised_loss 按 reward 加权——朝高 reward
    的 winner 回归，但**不构造 (win,lose) 对、不引用 ref、不做 L_FM 似然比** → 不踩 DPO 的 FM 反相关坑。

    为何 join 键是 audio 而非 fid:
        winners_only*.jsonl 的 fid 带 "-win" 后缀，与 rewards.jsonl 的原始 fid 不一致；
        而 win_audio 的路径字符串与 rewards.jsonl 的 audio 逐字相同（同一候选 wav 由同一打分阶段产出）。
    "reward(z 后)" 的含义与为何再 z 一次:
        rewards.jsonl 的 reward 已是「候选池内 z 加权聚合」；这里对**本 manifest 命中的 reward 整体**
        再做一次 z-normalize（common.zscore）→ 零中心、单位方差。这样训练侧 exp 加权数值良态，且
        E[w]≈1 可与 sft_baseline 的有效学习率对齐（X6 uniform vs X6b weighted 的唯一差异=权重分布）。
    缺失容错:
        manifest 里若某 audio 不在 rewards 表（如误传真实录音清单），记为「命中均值」(z 后≈0→权重≈1，
        等价无偏好的 SFT)，并打印命中率；全部未命中则 fail-fast（多半是 train_manifest/rewards_file 配错）。

    输入: cfg["train_manifest"]（{fid,text,audio,lang}）+ cfg["reward_weighting"]["rewards_file"]。
    输出: DataLoader；每 batch = {"sample": <collated>, "reward": FloatTensor(B,)（已 z-norm）}。
    """
    import torch  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore

    manifest = cfg.get("train_manifest", str(PATHS.prompts_dir / "dots_manifest_train.jsonl"))
    rwcfg = cfg.get("reward_weighting", {}) or {}
    rewards_file = rwcfg.get("rewards_file", str(PATHS.rewards_dir / "rewards.jsonl"))

    rows = list(read_jsonl(manifest))
    reward_map = {r["audio"]: float(r["reward"]) for r in read_jsonl(rewards_file)}

    def _audio_of(r: dict) -> str:
        return r.get("prompt_audio") or r.get("audio")

    raw = [reward_map.get(_audio_of(r)) for r in rows]
    found = [x for x in raw if x is not None]
    if not found:
        raise SystemExit(
            f"build_weighted_dataloader: manifest({manifest}) 的 audio 无一命中 {rewards_file}——"
            "确认 train_manifest 指向带 reward 的候选清单（winners_only*.jsonl），rewards_file 对应同一套候选。")
    mean_found = sum(found) / len(found)
    raw = [x if x is not None else mean_found for x in raw]   # 未命中→均值（z 后≈0→权重≈1）
    rewards_z = zscore(raw)                                   # 「reward(z 后)」：跨 manifest 零中心/单位方差

    if getattr(trainer, "accel", None) is None or trainer.accel.is_main_process:
        print(f"[weighted-loader] reward 命中 {len(found)}/{len(rows)}（{rewards_file}）；"
              "已跨 manifest z-normalize 作样本权基。")

    pipeline, collator = _make_pipeline_collator(trainer.tokenizer, trainer.sample_rate, trainer.aspt)

    class _DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            sample = pipeline.process_sample({"fid": str(r["fid"]), "text": r["text"],
                                              "audio": _audio_of(r), "source_name": "rwr"})
            return sample, float(rewards_z[i])

    def _collate(batch):
        return {"sample": collator([s for s, _ in batch]),
                "reward": torch.tensor([w for _, w in batch], dtype=torch.float32)}

    return DataLoader(_DS(), batch_size=cfg.get("micro_batch_size", 1), shuffle=True,
                      collate_fn=_collate, num_workers=cfg.get("num_workers", 0))
