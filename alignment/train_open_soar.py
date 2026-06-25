"""
train_open_soar.py — 核心③：open-SOAR（reward-free 自校正）开源再现

做什么:
    不依赖任何外部奖励(WER/SIM/UTMOS)，用「干净 vs 扰动」对比做自校正对齐：
    winner = 干净音频；loser = 同一音频加噪后的版本。DPO 让策略相对 ref 更偏好「干净重建」。
怎么做（复用主 DPO 的高层前向路径，可直接跑）:
    复用 DpoTrainer 的 `_fm_per_sample`(共享 RNG) + `dpo_loss`。loser 的 off-trajectory 由
    **在波形上加高斯噪声**得到（扰动会传到 vocoder.extract_latents → FM 目标），
    无需改 dots.tts 内部。
为什么这么设计:
    dots.tts 的原始 SOAR 训练程序未开源（仓库只有 soar 权重，核心代码分析.md 证实 reward-free、
    只作用于 flow-matching 头）。本文件是**我们的开源近似**：用「偏好干净>偏好扰动」逼近
    「自我纠正到干净轨迹」。用于隔离「自校正」单独效果，与多目标 DPO 对照（RQ4）。
对实验意义:
    open_soar 是 base 与 dpo_multiobj 之间的关键中间臂。

⚠️ 明确标注：这是 SOAR 的**近似再现**（波形扰动版），非 Boson 原始算法；本机只语法检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402

import torch  # noqa: E402

from train_dots_tts_dpo import DpoTrainer, _to_device  # noqa: E402
from losses_dpo import dpo_loss, anchor_sft_loss  # noqa: E402


class OpenSoarTrainer(DpoTrainer):
    """复用 DpoTrainer 编排；偏好对 = (干净, 加噪)，无外部奖励。"""

    def _perturb_waveform(self, batch: dict) -> dict:
        """对 collate 后 batch 的 'sample' 波形加高斯噪声，构造 off-trajectory(扰动) 样本。

        扰动经 vocoder.extract_latents 传到 FM 目标。**关键**：必须删掉预算好的 fbank/fbank_lengths，
        否则 xvector_extractor 检测到 fbank 存在就走「裁剪预算 fbank」分支、用**干净音频**的说话人条件，
        导致 loser 的 FM 目标(扰动) 与 说话人条件(干净) 不一致——删掉后 xvector 会从扰动 sample 实时重算。
        噪声强度按**逐样本** std 标定（全局 std 在批内响度差异大时不均匀）。
        """
        out = dict(batch)
        s = batch["sample"]
        scale = float(self.dpo.get("soar_noise_scale", 0.5))
        per_sample_std = s.std(dim=-1, keepdim=True).clamp_min(1e-6)
        out["sample"] = s + torch.randn_like(s) * (scale * per_sample_std)
        out.pop("fbank", None)            # 让 xvector 从扰动 sample 重算，而非用干净缓存
        out.pop("fbank_lengths", None)
        return out

    def train_step(self, batch) -> dict:
        clean_raw = batch["sample"]
        pert_raw = self._perturb_waveform(clean_raw)
        clean = _to_device(self.policy.prepare_training_batch(clean_raw), self.accel.device)
        pert = _to_device(self.policy.prepare_training_batch(pert_raw), self.accel.device)

        self._advance_global_rng()    # 每步推进，使各步噪声不同
        rng_c = self._capture_rng()
        rng_p = self._capture_rng()
        fm_c_pol, ce_w, eos_w = self._fm_per_sample(self.policy, clean, rng_c, grad=True, want_sft=True)
        fm_c_ref = self._ref_forward(clean, rng_c)
        fm_p_pol = self._fm_per_sample(self.policy, pert, rng_p, grad=True)
        fm_p_ref = self._ref_forward(pert, rng_p)

        # winner=干净, loser=扰动
        stats = dpo_loss(fm_c_pol, fm_c_ref, fm_p_pol, fm_p_ref, beta=self.beta)
        loss = stats.loss
        if self.lam_sft > 0:
            loss = loss + anchor_sft_loss(ce_term=ce_w, eos_term=eos_w, lam=self.lam_sft)

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {"loss": round(float(loss.item()), 4), "reward_acc": round(stats.reward_accuracy, 3)}

    def save(self, out_dir: str, step: int):
        # 元数据标记为 open_soar（别继承 DpoTrainer 的 "dpo"），保证 treatment 可溯源
        self.save_inference_ckpt(out_dir, {
            "alignment": "open_soar", "beta": self.beta,
            "soar_noise_scale": self.dpo.get("soar_noise_scale", 0.5), "step": step})


def main() -> None:
    ap = argparse.ArgumentParser(description="open-SOAR reward-free 自校正（近似）")
    ap.add_argument("--config", required=True, help="configs/open_soar_selfcorrect.yaml")
    args = ap.parse_args()
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_single_dataloader  # type: ignore

    trainer = OpenSoarTrainer(cfg)
    loader = build_single_dataloader(cfg, trainer)
    trainer.policy, trainer.opt, loader = trainer.accel.prepare(trainer.policy, trainer.opt, loader)

    max_steps = int(cfg["max_train_steps"])
    step = 0
    done = False
    while not done:  # 跨 epoch 续训直到 max_steps（见 train_dots_tts_dpo 同款说明）
        for batch in loader:
            with trainer.accel.accumulate(trainer.policy):
                logs = trainer.train_step(batch)
            trainer.log_step(step, logs)  # TB + metrics.jsonl（继承自 DpoTrainer）
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
        print(f"✓ open-SOAR(近似) 完成，{step} 步 -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
