"""
train_dots_tts_dpo.py — 核心②：连续-AR 偏好对齐训练（DPO over flow-matching）

做什么:
    在 dots.tts-base 上做 DPO 微调：策略模型 + 冻结参考模型，对 (win,lose) 配对前向，
    用 losses_dpo.dpo_loss 反传，产出对齐 checkpoint。
怎么做（复用优先，已对照源码核验）:
    - 复用 dots.tts: `DotsTtsModel.from_pretrained`、`prepare_training_batch`、`model(batch)→LossTerms`，
      Accelerate 训练编排（仿 scripts/train_dots_tts.py 的 step 结构）。
    - per-sample L_FM 直接取自 `loss_terms["fm_loss"]`（不碰 DiT 内部）。
    - 仅新增: 冻结 ref（或 LoRA disable_adapter）、paired 前向、**共享 (x0,t,VAE-z)**、DPO 损失 + anchor SFT。
为什么这么设计:
    dots.tts 已自带完整训练栈，A 不是重写模型，而是「加损失 + 配对数据 + ref」。
    最高风险点 = policy/ref/win/lose 的 FM 前向必须共享同一随机状态（见 _fm_per_sample 的 fork_rng）。
对实验意义:
    产出所有 treatment(dpo_multiobj / dpo_weronly / ...) 的引擎；配置切换即不同实验臂。

前置条件: 环境就绪、dots.tts-base 权重、pairs/out/pairs_*.jsonl、configs/*.yaml。
预期结果: 在 output_dir 写出对齐 checkpoint(+元数据记 alignment/beta/reward_weights)。
⚠️ 本机只语法检查（py_compile 不执行 import torch / dots_tts）。开训前先跑 alignment/check_fm_loss.py 自检。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable  # noqa: E402

import torch  # noqa: E402

from losses_dpo import dpo_loss, fm_loss_per_sample, anchor_sft_loss  # noqa: E402


class DpoTrainer:
    """仿 dots.tts 训练编排，但把单模型前向换成 paired(win/lose) + 冻结 ref 的 DPO 前向。"""

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
        self.policy = DotsTtsModel.from_pretrained(base)
        # 在任何包装(LoRA/accelerate)前，从原始模型捕获数据契约，供 data_pairs 建 pipeline/collator
        self.tokenizer = self.policy.tokenizer
        self.sample_rate = int(self.policy.config.vocoder.sample_rate)
        self.aspt = int(self.policy.hop_size) * int(self.policy.config.patch_size)
        if self.dpo.get("lora", {}).get("enabled"):
            from lora_adapter import wrap_lora  # type: ignore
            self.policy = wrap_lora(self.policy, self.dpo["lora"])
            self.ref = None                       # LoRA: ref = disable_adapter() 的同一模型
        else:
            self.ref = DotsTtsModel.from_pretrained(base).eval().requires_grad_(False)
        self.policy.train()

        self.opt = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01),
        )
        self.beta = float(self.dpo.get("beta", 0.1))
        self.lam_sft = float(self.dpo.get("lambda_sft", 0.1))

    # --- 关键：在「共享 RNG 状态」下跑一次前向，取 per-sample L_FM ---------- #
    def _capture_rng(self):
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(self.accel.device) if torch.cuda.is_available() else None
        return cpu, cuda

    def _fm_per_sample(self, model, prepared, rng, *, grad: bool, want_sft=False):
        """在固定 RNG 状态下前向，返回 (fm (B,), 可选 (ce_term, eos_term))。

        为什么 fork_rng + set_state: dots.tts 在 model() 内部采样 (x0 噪声, 时间步 t, VAE z)。
        policy 与 ref、win 内部要可比，必须复用同一随机状态 → 同输入同 RNG ⇒ 同 (x0,t,z)。
        """
        cpu_state, cuda_state = rng
        devices = [self.accel.device] if cuda_state is not None else []
        gctx = contextlib.nullcontext() if grad else torch.no_grad()
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, self.accel.device)
            with gctx:
                loss_terms = model(prepared)
        fm = fm_loss_per_sample(loss_terms["fm_loss"])
        if want_sft:
            return fm, loss_terms.get("ce_loss"), loss_terms.get("eos_loss")
        return fm

    def _ref_forward(self, prepared, rng):
        if self.ref is not None:
            return self._fm_per_sample(self.ref, prepared, rng, grad=False)
        with self.policy.disable_adapter():  # type: ignore[attr-defined]
            return self._fm_per_sample(self.policy, prepared, rng, grad=False)

    def train_step(self, batch) -> dict:
        # batch 含 win/lose 两路已 collate 的 batch；先在 CPU 侧 prepare（不消费随机性）
        win = self.policy.prepare_training_batch(batch["win"])
        lose = self.policy.prepare_training_batch(batch["lose"])
        win = _to_device(win, self.accel.device)
        lose = _to_device(lose, self.accel.device)

        rng_w = self._capture_rng()   # win 这一对（policy/ref）共享的状态
        rng_l = self._capture_rng()   # lose 这一对共享的状态（与 win 同起点，近似共享 t）

        fm_pol_w, ce_w, eos_w = self._fm_per_sample(self.policy, win, rng_w, grad=True, want_sft=True)
        fm_ref_w = self._ref_forward(win, rng_w)
        fm_pol_l = self._fm_per_sample(self.policy, lose, rng_l, grad=True)
        fm_ref_l = self._ref_forward(lose, rng_l)

        stats = dpo_loss(fm_pol_w, fm_ref_w, fm_pol_l, fm_ref_l, beta=self.beta)
        loss = stats.loss
        if self.lam_sft > 0:
            loss = loss + anchor_sft_loss(ce_term=ce_w, eos_term=eos_w, lam=self.lam_sft)

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return {"loss": float(loss.item()), "reward_margin": stats.reward_margin,
                "reward_acc": stats.reward_accuracy,
                "fm_w": stats.fm_pol_win, "fm_l": stats.fm_pol_lose}

    def save(self, out_dir: str, step: int):
        import json
        meta = {"alignment": "dpo", "beta": self.beta,
                "reward_weights": self.dpo.get("reward_weights"),
                "pairs": self.dpo.get("pairs_file"), "step": step}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "alignment_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.accel.save_state(out_dir)


def _to_device(batch: dict, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="dots.tts DPO 偏好对齐")
    ap.add_argument("--config", required=True, help="configs/dpo_*.yaml")
    args = ap.parse_args()

    import yaml  # type: ignore
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    ensure_dots_tts_importable()
    from data_pairs import build_pair_dataloader  # type: ignore

    trainer = DpoTrainer(cfg)
    loader = build_pair_dataloader(cfg, trainer)
    trainer.policy, trainer.opt, loader = trainer.accel.prepare(trainer.policy, trainer.opt, loader)

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
