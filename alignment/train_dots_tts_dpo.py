"""
train_dots_tts_dpo.py — 核心②：连续-AR 偏好对齐训练（DPO over flow-matching）

做什么:
    在 dots.tts-base 上做 DPO 微调：策略模型 + 冻结参考模型，对 (win,lose) 配对前向，
    用 losses_dpo.dpo_fm_loss 反传，产出对齐后的 checkpoint。
怎么做（复用优先，fork 最小）:
    - 复用 dots.tts: `DotsTtsModel.from_pretrained`(策略+参考)、AudioVAE 编码、CAM++ x-vector、
      `_compute_fm_loss_term` 的数学(在 losses_dpo 里)、Accelerate 训练编排(仿 scripts/train_dots_tts.py)。
    - 仅新增: 冻结 ref、paired 前向、**共享 (x0,t) 采样**、DPO 损失 + anchor SFT。
为什么这么设计:
    dots.tts 已自带完整训练栈，A 不是重写模型，而是「加损失 + 配对数据 + ref」。
    最高风险点 = policy/ref/win/lose 四次 FM 前向必须共享同一 (x0,t)（见下 _forward_fm 的 fork_rng）。
对实验意义:
    这是产出所有 treatment(dpo_multiobj / dpo_weronly / ...) 的引擎；配置切换即不同实验臂。

前置条件: 环境就绪、dots.tts-base 权重、pairs/out/pairs_*.jsonl、configs/*.yaml。
预期结果: 在 output_dir 写出对齐 checkpoint(+元数据记 alignment/beta/reward_weights)。
⚠️ 本机只语法检查（py_compile 不执行 import torch / dots_tts）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS, ensure_dots_tts_importable  # noqa: E402

import torch  # noqa: E402

from losses_dpo import dpo_fm_loss, anchor_sft_loss  # noqa: E402


class DpoTrainer:
    """仿 `dots.tts/scripts/train_dots_tts.py:DotsTtsTrainingRun` 的训练编排，
    但把单模型前向换成 paired(win/lose) + 冻结 ref 的 DPO 前向。"""

    def __init__(self, cfg: dict):
        ensure_dots_tts_importable()
        from accelerate import Accelerator  # type: ignore
        from dots_tts.models.dots_tts.model import DotsTtsModel  # type: ignore

        self.cfg = cfg
        self.dpo = cfg["dpo"]
        self.accel = Accelerator(
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            mixed_precision="bf16",
        )
        base = self.dpo["base_ckpt"]
        # 策略模型（可训练；LoRA 时只训适配器）
        self.policy = DotsTtsModel.from_pretrained(base)
        if self.dpo.get("lora", {}).get("enabled"):
            from lora_adapter import wrap_lora  # type: ignore
            self.policy = wrap_lora(self.policy, self.dpo["lora"])
            self.ref = None  # LoRA: ref = 关掉适配器的同一模型，省一份显存
        else:
            # full-param: 独立冻结一份 base 作 ref
            self.ref = DotsTtsModel.from_pretrained(base).eval().requires_grad_(False)

        self.opt = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01),
        )
        self.beta = float(self.dpo.get("beta", 0.1))
        self.lam_sft = float(self.dpo.get("lambda_sft", 0.1))

    # --- 关键：共享 (x0,t) 的 FM 前向 ------------------------------------- #
    def _forward_fm(self, model, sample, rng_state):
        """对一条样本跑一次 FM 前向，返回 (pred, target, patch_mask)。

        rng_state: 预先 fork 出的随机状态；policy 与 ref、win 与 lose 必须复用同一份，
        才能保证 (x0 噪声, 时间步 t) 一致 —— 这是 Diffusion-DPO 似然比成立的前提。
        见 docs/02-DECISIONS.md「四前向共享 RNG」。
        """
        with torch.random.fork_rng(devices=[sample["device"]] if "device" in sample else None):
            torch.random.set_rng_state(rng_state)
            out = model.core(sample)  # DotsTtsForwardOutput(pred,target,...)，见 core.py
        return out.pred, out.target, sample["fm_patch_mask"]

    def _ref_forward(self, sample, rng_state):
        if self.ref is not None:
            with torch.no_grad():
                return self._forward_fm(self.ref, sample, rng_state)
        # LoRA 情形：关掉适配器 = 参考模型
        with self.policy.disable_adapter(), torch.no_grad():  # type: ignore[attr-defined]
            return self._forward_fm(self.policy, sample, rng_state)

    def train_step(self, batch) -> dict:
        # batch 含 win/lose 两路已编码样本(由 data_pairs collator 准备)
        win, lose = batch["win"], batch["lose"]
        # 为这一对采样一次共享 RNG 状态
        rng = torch.random.get_rng_state()

        p_pred_w, tgt_w, m_w = self._forward_fm(self.policy, win, rng)
        r_pred_w, _, _ = self._ref_forward(win, rng)
        p_pred_l, tgt_l, m_l = self._forward_fm(self.policy, lose, rng)
        r_pred_l, _, _ = self._ref_forward(lose, rng)

        stats = dpo_fm_loss(
            p_pred_w, tgt_w, m_w, r_pred_w,
            p_pred_l, tgt_l, m_l, r_pred_l,
            beta=self.beta,
        )
        loss = stats.loss
        # anchor SFT（winners 的 CE+EOS），防文本/停止漂移
        if self.lam_sft > 0 and "ce_loss" in win and "eos_loss" in win:
            loss = loss + anchor_sft_loss(win["ce_loss"], win["eos_loss"], self.lam_sft)

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {
            "loss": float(loss.item()), "reward_margin": stats.reward_margin,
            "reward_acc": stats.reward_accuracy,
        }

    def save(self, out_dir: str, step: int):
        meta = {
            "alignment": "dpo", "beta": self.beta,
            "reward_weights": self.dpo.get("reward_weights"),
            "pairs": self.dpo.get("pairs_file"), "step": step,
        }
        # 复用 dots.tts 的 checkpoint 保存约定；此处记录对齐元数据用于溯源
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "alignment_meta.json").write_text(
            __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.accel.save_state(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="dots.tts DPO 偏好对齐")
    ap.add_argument("--config", required=True, help="configs/dpo_*.yaml")
    args = ap.parse_args()

    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_pair_dataloader  # type: ignore

    trainer = DpoTrainer(cfg)
    loader = build_pair_dataloader(cfg, trainer.accel)
    loader, = trainer.accel.prepare(loader)

    max_steps = int(cfg["max_train_steps"])
    step = 0
    for batch in loader:
        with trainer.accel.accumulate(trainer.policy):
            logs = trainer.train_step(batch)
        if trainer.accel.is_main_process and step % cfg.get("log_interval", 10) == 0:
            print(f"step {step}: {logs}")
        if step and step % cfg.get("save_interval", 500) == 0:
            trainer.save(cfg["output_dir"], step)
        step += 1
        if step >= max_steps:
            break
    trainer.save(cfg["output_dir"], step)
    if trainer.accel.is_main_process:
        print(f"✓ DPO 完成，{step} 步 -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
