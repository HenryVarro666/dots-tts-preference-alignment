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
import json
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
        from dots_preprocess import load_dots_model  # type: ignore  # 解析 repo id→本地路径

        self.cfg = cfg
        self.dpo = cfg["dpo"]
        self.accel = Accelerator(
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            mixed_precision="bf16",
        )
        base = self.dpo["base_ckpt"]
        self.policy = load_dots_model(base)
        # 在任何包装(LoRA/accelerate)前，从原始模型捕获数据契约，供 data_pairs 建 pipeline/collator
        self.tokenizer = self.policy.tokenizer
        self.sample_rate = int(self.policy.config.vocoder.sample_rate)
        self.aspt = int(self.policy.hop_size) * int(self.policy.config.patch_size)
        if cfg.get("mode") == "sft":
            self.ref = None                       # SFT 是标准监督训练，不需要参考模型
        elif self.dpo.get("lora", {}).get("enabled"):
            from lora_adapter import wrap_lora  # type: ignore
            self.policy = wrap_lora(self.policy, self.dpo["lora"])
            self.ref = None                       # LoRA: ref = disable_adapter() 的同一模型
        else:
            # 冻结 ref 不经 accelerator.prepare（不参与 DDP），需**手动搬到设备**，
            # 否则前向时 batch 在 cuda、ref 权重在 cpu → RuntimeError(Input cuda vs weight cpu)。
            self.ref = load_dots_model(base).eval().requires_grad_(False).to(self.accel.device)
        self.policy.train()

        self.opt = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01),
        )
        self.beta = float(self.dpo.get("beta", 0.1))
        self.lam_sft = float(self.dpo.get("lambda_sft", 0.1))
        self._init_logging()

    # --- 训练曲线日志：TensorBoard（实时）+ metrics.jsonl（离线兜底）------- #
    def _init_logging(self):
        self.writer = None
        out = Path(self.cfg["output_dir"])
        self._metrics_path = out / "metrics.jsonl"
        if not self.accel.is_main_process:
            return
        out.mkdir(parents=True, exist_ok=True)
        self._metrics_path.write_text("", encoding="utf-8")  # 每次训练清空重写
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(str(out / "tb"))
        except Exception as e:  # tensorboard 缺失也不影响训练，仅退化为 jsonl
            print(f"(TensorBoard 不可用，仅写 metrics.jsonl: {e})")

    def log_step(self, step: int, logs: dict):
        if not self.accel.is_main_process:
            return
        if self.writer is not None:
            for k, v in logs.items():
                try:
                    self.writer.add_scalar(f"train/{k}", float(v), step)
                except (TypeError, ValueError):
                    pass
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **logs}, ensure_ascii=False) + "\n")

    def close_logging(self):
        if self.writer is not None:
            self.writer.close()

    # --- 关键：在「共享 RNG 状态」下跑一次前向，取 per-sample L_FM ---------- #
    def _advance_global_rng(self):
        """每个训练 step 开头推进一次全局 RNG，使各步采样到**不同**的 (x0,t,z)。

        命门细节：`_fm_per_sample` 在 `fork_rng` 内跑，退出时把全局 RNG 恢复到进入前——
        所以一个 train_step 结束后全局 RNG 不变；而 DataLoader 只在每个 epoch 开头推进一次 RNG。
        若不在此手动推进，则**同一 epoch 内所有 step 共用同一噪声**（DPO 退化为对单噪声过拟合）。
        在 capture 之前推进一次 → 步间噪声不同；步内 rng_w==rng_l 仍保证 policy/ref/win/lose 四前向共享同一 (x0,t,z)。"""
        torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1, device=self.accel.device)

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

        self._advance_global_rng()    # 每步推进，使各步噪声不同（否则同 epoch 内共用一个噪声）
        rng_w = self._capture_rng()   # win 这一对（policy/ref）共享的状态
        rng_l = self._capture_rng()   # lose 这一对共享的状态（与 win 同起点，按 Diffusion-DPO 共享 (x0,t)）

        fm_pol_w, ce_w, eos_w = self._fm_per_sample(self.policy, win, rng_w, grad=True, want_sft=True)
        fm_ref_w = self._ref_forward(win, rng_w)
        fm_pol_l = self._fm_per_sample(self.policy, lose, rng_l, grad=True)
        fm_ref_l = self._ref_forward(lose, rng_l)

        stats = dpo_loss(fm_pol_w, fm_ref_w, fm_pol_l, fm_ref_l, beta=self.beta)
        dpo_val = float(stats.loss.item())
        loss = stats.loss
        sft_val = 0.0
        if self.lam_sft > 0:
            sft = anchor_sft_loss(ce_term=ce_w, eos_term=eos_w, lam=self.lam_sft)
            loss = loss + sft
            sft_val = float(sft.item())

        self.accel.backward(loss)
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(self.policy.parameters(), self.cfg.get("grad_clip_norm", 1.0))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        # dpo=纯偏好损失(应随训练↓)、sft=anchor 正则、reward_acc/margin=DPO 健康信号
        return {"dpo": round(dpo_val, 4), "sft": round(sft_val, 4),
                "reward_acc": round(stats.reward_accuracy, 3),
                "reward_margin": round(stats.reward_margin, 5),
                "fm_w": round(stats.fm_pol_win, 4), "fm_l": round(stats.fm_pol_lose, 4)}

    def save_inference_ckpt(self, out_dir: str, meta: dict):
        """存成 dots.tts `from_pretrained` 可加载的格式（inference/eval 用）。

        关键：**不能用 accelerator.save_state**（那是 state_dict+optimizer+rng 的续训格式，
        from_pretrained 加载不了）。要用 dots.tts 自己的 model.save_pretrained，写出
        config.json/llm_config.json/tokenizer/latent_stats.pt/model|vocoder|speaker_encoder.safetensors。
        """
        if not self.accel.is_main_process:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "alignment_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        model = self.accel.unwrap_model(self.policy)
        # LoRA: 合并适配器再存（merge_and_unload 会改动模型，故 LoRA 建议只在训练结束存一次）
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        model.save_pretrained(out_dir)

    def save(self, out_dir: str, step: int):
        self.save_inference_ckpt(out_dir, {
            "alignment": "dpo", "beta": self.beta,
            "reward_weights": self.dpo.get("reward_weights"),
            "pairs": self.dpo.get("pairs_file"), "step": step})


def _to_device(obj, device):
    """递归把张量搬到设备：必须穿透嵌套 dict（如 prepare_training_batch 产出的
    loss_masks={"ce_loss":..,"fm_loss":..,"eos_loss":..}），否则 mask 留在 CPU、与 GPU pred 相乘报错。
    （当前靠 accelerate 的 prepared loader 递归搬运侥幸不崩，但这里自身也要正确，防脱离 accelerate 时炸。）"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


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
    # 跨 epoch 续训：数据集小于 max_steps 时反复过 loader，直到达到 max_steps
    # （否则只跑一个 epoch 就退出，max_train_steps 形同虚设）。
    step = 0
    done = False
    while not done:
        for batch in loader:
            with trainer.accel.accumulate(trainer.policy):
                logs = trainer.train_step(batch)
            trainer.log_step(step, logs)  # 每步写 TB + metrics.jsonl（平滑曲线）
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
        print(f"✓ DPO 完成，{step} 步 -> {cfg['output_dir']}")
        print(f"  曲线: tensorboard --logdir {cfg['output_dir']}/tb   或   python alignment/plot_metrics.py --run {cfg['output_dir']}")


if __name__ == "__main__":
    main()
