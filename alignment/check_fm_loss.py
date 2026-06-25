"""
check_fm_loss.py — 开训前自检：验证 FM 接缝 + 共享-RNG 决定论（DPO 的命门）

做什么:
    用 dots.tts-base 在一条真实样本上跑前向，验证三件事：
      (1) 接缝通：`DotsPreprocessor` 产的 batch 经 `model(prepare_training_batch(batch))` 能得到
          有限的 per-sample `fm_loss`（说明 AudioVAE 编码 / x-vector / fm_patch_mask 全接对了）。
      (2) 共享-RNG 决定论：**同一 RNG 状态**下两次前向的 fm_loss **完全一致** —— 这正是 DPO 要求
          policy/ref 在同 (x0,t,VAE-z) 下可比的前提。
      (3) 随机性存在：**不同 RNG** 下 fm_loss **不同** —— 确认 (2) 不是平凡的（模型确实在采样 x0/t）。
怎么做:
    复用 `dots_preprocess.DotsPreprocessor` + `losses_dpo.fm_loss_per_sample`，用 fork_rng+set_state
    精确控制随机状态（与 train_dots_tts_dpo._fm_per_sample 同款机制）。
为什么这么重要:
    若 (2) 不成立，DPO 的似然比 Δ=L_ref−L_pol 就被噪声淹没，训练学不出偏好。此自检过了再开训。

前置条件: 环境就绪、dots.tts-base 权重、一条可用音频（或一个 prompt manifest）。
预期结果: 打印三项 PASS/FAIL + per-sample L_FM 与 fm_loss 形状。
⚠️ 本机只语法检查；真正运行在 GPU 机器上（开训前必跑）。
用法:
    python alignment/check_fm_loss.py --ckpt rednote-hilab/dots.tts-base \
           --manifest data/prompts/dots_manifest_train.jsonl
  或  python alignment/check_fm_loss.py --ckpt ... --audio a.wav --text "hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_dots_tts_importable, read_jsonl  # noqa: E402

import torch  # noqa: E402

from dots_preprocess import DotsPreprocessor  # noqa: E402
from losses_dpo import fm_loss_per_sample  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _capture(dev):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
    return cpu, cuda


def _forward_with_rng(pre: DotsPreprocessor, batch: dict, rng, dev):
    cpu_state, cuda_state = rng
    devs = [dev] if cuda_state is not None else []
    with torch.random.fork_rng(devices=devs):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, dev)
        with torch.no_grad():
            loss_terms = pre.prepare_and_forward(batch)
    return fm_loss_per_sample(loss_terms["fm_loss"]), loss_terms["fm_loss"]


def _to_device(batch, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="FM 接缝 + 共享 RNG 自检")
    ap.add_argument("--ckpt", default="rednote-hilab/dots.tts-base")
    ap.add_argument("--manifest", default="data/prompts/dots_manifest_train.jsonl")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--text", default=None)
    args = ap.parse_args()

    dev = _device()
    print(f"device = {dev}")
    pre = DotsPreprocessor.from_pretrained(args.ckpt)
    pre.model.to(dev)

    # 取一条样本
    if args.audio and args.text:
        rec = {"fid": "check-0", "text": args.text, "audio": args.audio}
    else:
        rows = list(read_jsonl(args.manifest))
        if not rows:
            raise SystemExit(f"manifest 空: {args.manifest}（先跑 data 步），或用 --audio/--text")
        r = rows[0]
        rec = {"fid": str(r["fid"]), "text": r["text"], "audio": r.get("prompt_audio") or r["audio"]}

    sample = pre.process(rec["fid"], rec["text"], rec["audio"])
    batch = _to_device(pre.collate([sample]), dev)

    # (1) 接缝通 + (2) 同 RNG 决定论
    rngA = _capture(dev)
    fm1, term1 = _forward_with_rng(pre, batch, rngA, dev)
    fm2, _ = _forward_with_rng(pre, batch, rngA, dev)

    finite = bool(torch.isfinite(fm1).all().item())
    same = bool(torch.allclose(fm1, fm2, atol=1e-5, rtol=1e-4))

    # (3) 不同 RNG → 应不同
    torch.manual_seed(20260625)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(20260625)
    rngB = _capture(dev)
    fm3, _ = _forward_with_rng(pre, batch, rngB, dev)
    differs = not bool(torch.allclose(fm1, fm3, atol=1e-5, rtol=1e-4))

    print(f"fm_loss term shape = {tuple(term1.loss.shape)} (loss), {tuple(term1.mask.shape)} (mask)")
    print(f"per-sample L_FM    = {fm1.detach().float().cpu().tolist()}")
    print(f"[1] FM 接缝有限值     : {'PASS' if finite else 'FAIL'}")
    print(f"[2] 同 RNG 决定论(命门): {'PASS' if same else 'FAIL'}  "
          f"(fm1≈fm2 ⇒ policy/ref 在同 (x0,t) 下可比)")
    print(f"[3] 不同 RNG 有差异    : {'PASS' if differs else 'FAIL'}  "
          f"(确认确实在采样 x0/t，[2] 非平凡)")

    ok = finite and same and differs
    print("\n=== " + ("ALL PASS — 可以开训 DPO ===" if ok else "存在 FAIL — 先排查再开训 ==="))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
