"""
train_sft.py — 控制变量臂：仅用 winners 做标准监督微调（SFT）

做什么:
    在 dots.tts-base 上，用每个 prompt 的「winner 候选」做标准 CE+FM+EOS 监督微调，产出 sft checkpoint。
怎么做（复用 DpoTrainer 基建 + dots.tts 标准损失）:
    复用 DpoTrainer 的模型加载/Accelerate/日志；但 train_step 改成单样本前向 `model(prepared)`→loss_terms→
    `supervised_loss(ce+fm+eos)`，**无 ref、无配对**。数据用 winners_only.jsonl（audio=win_audio）。
为什么这么设计:
    SFT 是关键控制变量——排除「DPO 的增益其实只是多看了好数据」这一替代解释（RQ1）。
    若 dpo_multiobj 显著优于 sft，才说明是「偏好对齐」而非「数据增量」在起作用。
对实验意义:
    base → **sft** → dpo_multiobj 三者对比，分离「更多好数据」与「对比偏好」两种贡献。

前置条件: winners_only.jsonl（由 pairs/extract_winners.py 或 run_full_dataprep 产出）。
预期结果: 在 output_dir 写出 sft checkpoint + 训练曲线(TB/metrics.jsonl)。
⚠️ 本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402

from train_dots_tts_dpo import DpoTrainer, _to_device  # noqa: E402
from losses_dpo import supervised_loss  # noqa: E402


class SftTrainer(DpoTrainer):
    """标准监督微调：单样本前向 + ce/fm/eos 加权损失（无 DPO、无 ref）。"""

    def train_step(self, batch) -> dict:
        sample = _to_device(self.policy.prepare_training_batch(batch["sample"]), self.accel.device)
        loss_terms = self.policy(sample)
        loss, parts = supervised_loss(loss_terms, self.cfg.get("loss"))

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {"loss": round(float(loss.item()), 4),
                "ce": round(parts.get("ce_loss", 0.0), 4),
                "fm": round(parts.get("fm_loss", 0.0), 4),
                "eos": round(parts.get("eos_loss", 0.0), 4)}

    def save(self, out_dir: str, step: int):
        # 复用 DpoTrainer.save_inference_ckpt：存 dots.tts from_pretrained 可加载的格式
        self.save_inference_ckpt(out_dir, {"alignment": "sft", "step": step})


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT 控制变量（仅 winners）")
    ap.add_argument("--config", required=True, help="configs/sft_baseline.yaml")
    args = ap.parse_args()
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_single_dataloader  # type: ignore

    trainer = SftTrainer(cfg)
    loader = build_single_dataloader(cfg, trainer)
    trainer.policy, trainer.opt, loader = trainer.accel.prepare(trainer.policy, trainer.opt, loader)

    max_steps = int(cfg["max_train_steps"])
    step = 0
    done = False
    while not done:
        for batch in loader:
            with trainer.accel.accumulate(trainer.policy):
                logs = trainer.train_step(batch)
            trainer.log_step(step, logs)
            if trainer.accel.is_main_process and step % cfg.get("log_interval", 10) == 0:
                print(f"step {step}: {logs}")
            step += 1
            if step % cfg.get("save_interval", 500) == 0:
                trainer.save(cfg["output_dir"], step)
            if step >= max_steps:
                done = True
                break
    trainer.save(cfg["output_dir"], step)
    trainer.close_logging()
    if trainer.accel.is_main_process:
        print(f"✓ SFT 完成，{step} 步 -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
