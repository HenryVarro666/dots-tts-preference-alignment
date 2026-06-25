"""
losses_dpo.py — 核心①：把 DPO 适配到连续-latent flow-matching（Diffusion-DPO）

做什么:
    定义偏好对齐损失：用 dots.tts 既有的 **velocity-MSE（flow-matching 损失）当隐式 NLL 代理**，
    构造 DPO 目标 L = −log σ( β·[(L_FM^ref−L_FM^policy)_win − (L_FM^ref−L_FM^policy)_lose] )。
怎么做（复用 LossTerm，不碰 DiT 内部）:
    dots.tts 的 `model(prepared)` 返回 `LossTerms = dict[str,LossTerm]`，其中
    `loss_terms["fm_loss"]` 的 `.loss`/`.mask` 形状均为 (B, max_patch)。
    我们只对它做 per-sample 的 mask 平均，得到每条样本的 L_FM 标量，再组 DPO sigmoid。
    —— 比直接取 pred/target 张量稳健得多（无需理解 DiT 内部 patch 排布）。
为什么这么设计（关键）:
    连续-latent AR 无离散 token 的 log-prob；借鉴 Diffusion-DPO(Wallace 2023)，
    用 flow-matching 训练损失当 −log p 的代理。详见 docs/01-METHOD.md。
    正确性命门：policy/ref/win/lose 四次前向**共享同一 (x0,t,VAE-z)**
    （在 train_dots_tts_dpo.py 用 fork_rng + 同一 RNG 状态保证）。

⚠️ 本机只语法检查（py_compile 不执行 import torch）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DpoStats:
    """训练监控量：reward margin 越大、accuracy 越高 = 偏好学得越好。"""

    loss: torch.Tensor
    reward_margin: float      # E[(Δ_w − Δ_l)]，应随训练上升
    reward_accuracy: float    # P[Δ_w > Δ_l]，>0.5 表示在学对方向
    fm_pol_win: float
    fm_pol_lose: float


def fm_loss_per_sample(fm_term) -> torch.Tensor:
    """从 dots.tts 的 fm_loss LossTerm 取每条样本的 mask 平均 L_FM。

    fm_term.loss / fm_term.mask: (B, max_patch)。
    返回: (B,) —— 每条样本一个标量（越小=该样本在模型下越「可能」）。
    """
    loss = fm_term.loss
    mask = fm_term.mask.to(loss.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (loss * mask).sum(dim=1) / denom


def dpo_loss(
    fm_pol_w: torch.Tensor, fm_ref_w: torch.Tensor,
    fm_pol_l: torch.Tensor, fm_ref_l: torch.Tensor,
    beta: float = 0.1,
) -> DpoStats:
    """Diffusion-DPO over flow-matching velocity loss（输入为 per-sample L_FM 标量 (B,)）。

    注意: fm_*_w / fm_*_l 必须在共享 (x0,t,VAE-z) 下计算（见 train_dots_tts_dpo._fm_per_sample），
    否则 policy 与 ref 的 L_FM 不可比，DPO 似然比失效。
    """
    # Δ = L_FM^ref − L_FM^policy（策略把损失降得越多，似然提升越大）
    delta_w = fm_ref_w - fm_pol_w
    delta_l = fm_ref_l - fm_pol_l
    logits = beta * (delta_w - delta_l)          # 赢应被推高、输应被压低
    loss = -F.logsigmoid(logits).mean()
    return DpoStats(
        loss=loss,
        reward_margin=float((delta_w - delta_l).mean().item()),
        reward_accuracy=float((delta_w > delta_l).float().mean().item()),
        fm_pol_win=float(fm_pol_w.mean().item()),
        fm_pol_lose=float(fm_pol_l.mean().item()),
    )


def supervised_loss(loss_terms, weights=None):
    """标准监督损失（SFT 用）：ce/fm/eos 各自 mask-平均后加权求和 = dots.tts 训练目标。

    loss_terms: dots.tts `model(prepared)` 返回的 dict[str, LossTerm]。
    weights: {"ce_weight","fm_weight","eos_weight"}，缺省全 1.0。
    """
    weights = weights or {}
    w = {"ce_loss": float(weights.get("ce_weight", 1.0)),
         "fm_loss": float(weights.get("fm_weight", 1.0)),
         "eos_loss": float(weights.get("eos_weight", 1.0))}
    total = None
    parts = {}
    for name, term in loss_terms.items():
        if name not in w:
            continue
        denom = term.mask.sum().clamp_min(1.0)
        scalar = (term.loss * term.mask).sum() / denom
        parts[name] = float(scalar.item())
        contrib = w[name] * scalar
        total = contrib if total is None else total + contrib
    return total, parts


def anchor_sft_loss(fm_term=None, ce_term=None, eos_term=None, lam: float = 0.1) -> torch.Tensor:
    """winners 上的 CE+EOS anchor 正则，防文本条件/停止预测漂移。

    传入 winners 前向的 ce_loss / eos_loss LossTerm；按 Σ(loss*mask)/Σ(mask) 求标量后加权。
    为什么: 纯 DPO 只学相对偏好，可能让绝对文本对齐漂移；在赢样本上保留一点监督 SFT 是标准稳定化。
    """
    total = None
    for term in (ce_term, eos_term):
        if term is None:
            continue
        denom = term.mask.sum().clamp_min(1.0)
        scalar = (term.loss * term.mask).sum() / denom
        total = scalar if total is None else total + scalar
    if total is None:
        # 没有可用监督项时返回 0（保持计算图安全）
        import torch as _t
        return _t.zeros(())
    return lam * total
