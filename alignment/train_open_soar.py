"""
train_open_soar.py — 核心③：open-SOAR（reward-free 自校正）开源再现

做什么:
    不依赖任何外部奖励(WER/SIM/UTMOS)，靠模型自身构造「on-trajectory(回到干净 x1) vs
    off-trajectory(扰动 rollout)」偏好对，用同一 dpo_fm_loss 管线做自校正对齐。
怎么做:
    对每条训练音频，winner = 标准 FM 目标(干净 u_t)；loser = 从扰动/高噪 x_t 起步的
    rollout 速度场。让策略偏向「能从偏轨恢复到干净轨迹」的预测 —— 即自校正。
为什么这么设计:
    dots.tts 的 SOAR 训练程序未开源（仓库只有 soar 权重，核心代码分析.md 证实 reward-free、
    只作用于 flow-matching 头）。本文件是**我们的开源再现**，用于隔离「自校正」单独的贡献，
    与「外部多目标偏好(DPO)」对照 —— 回答 RQ4。
对实验意义:
    open_soar 是 base 与 dpo_multiobj 之间的关键中间臂；若它已拿到大部分鲁棒性增益，
    说明「自校正」比「外部奖励」更本质（这正是文档 05 §3.13/§6.2 想厘清的）。

前置条件: 环境就绪、dots.tts-base、训练音频清单(可直接用 data/prompts/*_train.jsonl)。
预期结果: output_dir 写出 open-SOAR 对齐 checkpoint。
⚠️ 本机只语法检查；此为我们的再现，非原 SOAR（见 docs/01-METHOD.md 标注）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402

import torch  # noqa: E402

from train_dots_tts_dpo import DpoTrainer  # noqa: E402
from losses_dpo import dpo_fm_loss  # noqa: E402


class OpenSoarTrainer(DpoTrainer):
    """复用 DpoTrainer 的编排，但偏好对来自「干净 vs 扰动 rollout」而非外部奖励。"""

    def _make_off_trajectory(self, sample, noise_scale: float):
        """从更高噪声起点构造一个「偏轨」样本：把 FM 起点 x_t 的噪声放大。

        实现要点: 复用 dots.tts core 的 flow-matching helper 采样 x0/t，但对 loser
        注入额外噪声(noise_scale)，使其偏离干净轨迹；target 仍是干净 u_t，
        从而训练「无论起点多偏都要指回干净目标」= 自校正。
        """
        off = dict(sample)
        off["_extra_noise_scale"] = noise_scale  # core 前向读取以扰动起点（实现见 METHOD.md）
        return off

    def train_step(self, batch) -> dict:
        clean = batch["sample"]
        off = self._make_off_trajectory(clean, self.dpo.get("soar_noise_scale", 0.5))
        rng = torch.random.get_rng_state()

        p_pred_w, tgt_w, m_w = self._forward_fm(self.policy, clean, rng)
        r_pred_w, _, _ = self._ref_forward(clean, rng)
        p_pred_l, tgt_l, m_l = self._forward_fm(self.policy, off, rng)  # 注意 target 仍是干净 u_t
        r_pred_l, _, _ = self._ref_forward(off, rng)

        stats = dpo_fm_loss(p_pred_w, tgt_w, m_w, r_pred_w,
                            p_pred_l, tgt_l, m_l, r_pred_l, beta=self.beta)
        self.accel.backward(stats.loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {"loss": float(stats.loss.item()), "reward_acc": stats.reward_accuracy}


def main() -> None:
    ap = argparse.ArgumentParser(description="open-SOAR reward-free 自校正")
    ap.add_argument("--config", required=True, help="configs/open_soar_selfcorrect.yaml")
    args = ap.parse_args()
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_single_dataloader  # type: ignore

    trainer = OpenSoarTrainer(cfg)
    loader = build_single_dataloader(cfg, trainer.accel)  # 单样本(非配对)，偏轨在训练内构造
    loader, = trainer.accel.prepare(loader)

    step = 0
    for batch in loader:
        with trainer.accel.accumulate(trainer.policy):
            logs = trainer.train_step(batch)
        if trainer.accel.is_main_process and step % cfg.get("log_interval", 10) == 0:
            print(f"step {step}: {logs}")
        step += 1
        if step >= int(cfg["max_train_steps"]):
            break
    trainer.save(cfg["output_dir"], step)
    if trainer.accel.is_main_process:
        print(f"✓ open-SOAR 完成，{step} 步 -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
