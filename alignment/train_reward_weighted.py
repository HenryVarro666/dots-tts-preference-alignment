"""
train_reward_weighted.py — X6b/X7：reward-weighted 回归臂（非 DPO 对比的「建设性出路」）

做什么:
    在 dots.tts-base 上，对每个 prompt 的 winner 做标准监督前向(CE+FM+EOS)，再**按该样本的 reward
    加权** loss —— 高 reward 的 winner 梯度权重更大。产出 reward_weighted checkpoint。
    全程不构造 (win,lose) 配对、不引用 ref、不做 L_FM 似然比 —— 与 DPO 路线正交。
怎么做（复用 SftTrainer 基建 + reward 加权；含命令/输入/输出）:
    命令: accelerate launch alignment/train_reward_weighted.py --config configs/reward_weighted.yaml
          （X7 换 --config configs/reward_weighted_weronly.yaml）
    输入: configs/reward_weighted*.yaml（train_manifest=winners 清单, reward_weighting.rewards_file）
          + pairs/out/winners_only*.jsonl + reward/out/rewards*.jsonl（由 audio 路径 join）。
    输出: cfg.output_dir 下 dots.tts `from_pretrained` 可加载的 checkpoint + metrics.jsonl/TB 曲线
          + alignment_meta.json（记 scheme/temperature/rewards_file/train_manifest 供溯源）。
    实现: 继承 SftTrainer（单前向 supervised_loss、无 ref、无配对），仅把 loss 乘上由 batch["reward"]
          （build_weighted_dataloader 跨 manifest z-normalize 后的标量）算出的样本权
          w = exp(R_z/τ)/E[exp(z/τ)]（reward-weighted regression；softmax 是它在 batch>1 的批内版）。
为什么这么设计（关键）:
    连续-latent flow-matching 的 L_FM 与人类偏好**反相关**（docs §6.7：win 的 L_FM 常高于 lose），
    令 Diffusion-DPO 的「win 似然应高于 lose」假设错配。reward-weighted 回归只朝 winner 的**绝对**
    监督目标回归、用 reward 调权，**完全不依赖 win/lose 的 L_FM 相对大小** → 从结构上绕开该错配。
对实验意义（对 claim 的意义）:
    X6b 是「建设性出路」：在 §6.7 错配框架之外，给出一条仍利用 reward 信号且不崩的对齐路径，与
    sft_baseline(uniform winners) 形成「uniform vs reward-weighted」干净对照（同 winners/同步数/同损失，
    唯一变量=按 reward 加权）。
    X7 用同一引擎、把 reward 换成 WER-only（winner 也按最低 WER 选），在**非错配**框架内干净复现
    reward-hacking（读得清但自然度退化），从而把 domain reviewer 的 Contribution-2(方法) 与
    Contribution-4(reward-hacking 诊断) 解耦：hacking 是 reward 设计本身的问题，而非 DPO 错配的伪影。

前置条件: 环境就绪、dots.tts-base 权重；pairs/out/winners_only*.jsonl 与 reward/out/rewards*.jsonl
          已产出（X7 的 WER-only winners 见 config 头注的一行 extract_winners 命令）。
预期结果: 在 output_dir 写出 reward_weighted checkpoint + 训练曲线；高 reward 样本的样本权 w>1。
⚠️ 本机只语法检查（py_compile 不执行 import torch / dots_tts；真正运行需 GPU + dots.tts-base）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402

import torch  # noqa: E402

from train_sft import SftTrainer  # noqa: E402
from train_dots_tts_dpo import _to_device  # noqa: E402
from losses_dpo import supervised_loss  # noqa: E402


class RewardWeightedTrainer(SftTrainer):
    """reward-weighted 回归：单前向 supervised_loss × 样本权 w(reward)。无 ref、无配对、不碰 L_FM 似然比。"""

    def _reward_weight(self, reward: "torch.Tensor") -> "torch.Tensor":
        """把（已跨 manifest z-normalize 的）reward 标量转成非负样本权（高 reward→大权）。

        scheme（configs.reward_weighting.scheme，默认 exp）:
          - "exp": w = exp(R_z/τ) / exp(0.5/τ²)。分母是 z~N(0,1) 下 E[exp(z/τ)] 的解析值，令
            E[w]≈1 → 有效学习率与 sft_baseline 可比（X6 uniform vs X6b weighted 唯一差异=权重分布）。
            这就是 reward-weighted regression(Peters&Schaal 2007)/AWR 的逐样本权；micro_batch=1 时精确。
            τ↑=更软(趋近 uniform SFT)，τ↓=更贪婪(放大高 reward；对 X7 = 放大 reward-hacking)。
          - "softmax": w = B·softmax(R_z/τ)（批内归一、均值 1）；仅在 micro_batch_size>1 才有意义
            （batch=1 时退化为常数 1，等价无加权——故默认不用它）。
          - "none": 全 1（退化为 sft_baseline，做 sanity check / 消融用）。
        """
        rw = self.cfg.get("reward_weighting", {}) or {}
        scheme = str(rw.get("scheme", "exp"))
        temp = float(rw.get("temperature", 1.0))
        if scheme == "exp":
            w = torch.exp(reward / temp)
            if bool(rw.get("normalize", True)):
                w = w / math.exp(0.5 / (temp * temp))   # E[exp(z/τ)], z~N(0,1) → E[w]≈1
        elif scheme == "softmax":
            w = reward.numel() * torch.softmax(reward / temp, dim=0)
        elif scheme == "none":
            w = torch.ones_like(reward)
        else:
            raise ValueError(f"未知 reward_weighting.scheme: {scheme}（应为 exp/softmax/none）")
        return w

    def train_step(self, batch) -> dict:
        sample = _to_device(self.policy.prepare_training_batch(batch["sample"]), self.accel.device)
        reward = batch["reward"].to(self.accel.device).float()         # (B,) 已跨 manifest z-norm
        weight = self._reward_weight(reward).mean()                    # micro_batch=1 → 即该样本权
        loss_terms = self.policy(sample)
        base_loss, parts = supervised_loss(loss_terms, self.cfg.get("loss"))
        loss = weight * base_loss                                      # reward 加权回归（不做对比）

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        # base=未加权监督损失、w=样本权、reward=本步 z 后 reward 均值（健康信号：高 reward 步 w>1）
        return {"loss": round(float(loss.item()), 4),
                "base": round(float(base_loss.item()), 4),
                "w": round(float(weight.item()), 4),
                "reward": round(float(reward.mean().item()), 4),
                "ce": round(parts.get("ce_loss", 0.0), 4),
                "fm": round(parts.get("fm_loss", 0.0), 4),
                "eos": round(parts.get("eos_loss", 0.0), 4)}

    def save(self, out_dir: str, step: int):
        # 复用 DpoTrainer.save_inference_ckpt；元数据记 alignment=reward_weighted + 加权来源，供溯源
        rw = self.cfg.get("reward_weighting", {}) or {}
        self.save_inference_ckpt(out_dir, {
            "alignment": "reward_weighted",
            "scheme": rw.get("scheme", "exp"),
            "temperature": rw.get("temperature", 1.0),
            "normalize": rw.get("normalize", True),
            "rewards_file": rw.get("rewards_file"),
            "train_manifest": self.cfg.get("train_manifest"),
            "step": step})


def main() -> None:
    ap = argparse.ArgumentParser(description="reward-weighted 回归臂（X6b/X7；非 DPO 对比）")
    ap.add_argument("--config", required=True, help="configs/reward_weighted{,_weronly}.yaml")
    args = ap.parse_args()
    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_weighted_dataloader  # type: ignore

    trainer = RewardWeightedTrainer(cfg)
    loader = build_weighted_dataloader(cfg, trainer)
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
        print(f"✓ reward-weighted 完成，{step} 步 -> {cfg['output_dir']}")
        print(f"  曲线: tensorboard --logdir {cfg['output_dir']}/tb   或   python alignment/plot_metrics.py --run {cfg['output_dir']}")


if __name__ == "__main__":
    main()
