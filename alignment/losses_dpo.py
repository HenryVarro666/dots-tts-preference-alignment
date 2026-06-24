"""
losses_dpo.py — 核心①：把 DPO 适配到连续-latent flow-matching（Diffusion-DPO）

做什么:
    定义偏好对齐的损失：用 dots.tts 既有的 **velocity-MSE（flow-matching 损失）当隐式 NLL 代理**，
    构造 DPO 目标 L = −log σ( β·[(L_FM^ref−L_FM^policy)_win − (L_FM^ref−L_FM^policy)_lose] )。
怎么做:
    复用/对齐 `dots_tts...model.py:_compute_fm_loss_term` 的数学（per-sample、mask 平均的
    ‖v_θ(x_t,t)−u_t‖²）。本文件只做「四个 FM 损失标量 → DPO 标量」的组合，不碰模型前向。
为什么这么设计（关键）:
    标准 DPO 定义在 token log-prob 上；连续-latent AR 没有离散 token 的 log-prob。
    借鉴 Diffusion-DPO(Wallace 2023)：扩散/流匹配的训练损失就是 −log p 的代理，
    因此「策略相对参考把赢样本的 FM 损失降得更多、把输样本降得更少」即偏好对齐。
    数学推导见 docs/01-METHOD.md。
对实验意义:
    这是整个项目的「方法贡献」——连续-AR 上可用的偏好优化目标；正确性命门是
    policy/ref/win/lose 四次前向**共享同一 (x0,t) 采样**（在 trainer 里用 fork_rng 保证）。

前置条件: 由 train_dots_tts_dpo.py 调用；输入是四次前向产出的 (pred,target,mask)。
预期结果: 返回可 backward 的标量 + 诊断量(reward margin / accuracy)。
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
    policy_fm_win: float
    policy_fm_lose: float


def per_sample_fm_loss(pred: torch.Tensor, target: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """每条样本、mask 平均的 velocity-MSE，复刻 dots.tts `_compute_fm_loss_term` 的数学。

    pred/target: (B, P, D) —— P=patch 数，D=latent 维；
    patch_mask:  (B, P)    —— 该 patch 是否参与 FM 损失(音频 span)。
    返回:        (B,)      —— 每条样本一个标量损失。
    """
    mse = (pred - target).pow(2).mean(dim=-1)          # (B, P) 沿 latent 维平均
    mask = patch_mask.to(mse.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)             # 防空 span 除零
    return (mse * mask).sum(dim=1) / denom             # (B,)


def dpo_fm_loss(
    pred_pol_w: torch.Tensor, tgt_w: torch.Tensor, mask_w: torch.Tensor,
    pred_ref_w: torch.Tensor,
    pred_pol_l: torch.Tensor, tgt_l: torch.Tensor, mask_l: torch.Tensor,
    pred_ref_l: torch.Tensor,
    beta: float = 0.1,
) -> DpoStats:
    """Diffusion-DPO over flow-matching velocity loss.

    注意: tgt_w/tgt_l 与 (x0,t) 必须在 policy 与 ref 间一致（由 trainer 的 fork_rng 保证），
    否则 L_FM^policy 与 L_FM^ref 不可比，DPO 似然比失效。
    """
    # 每条样本的 FM 损失（越小=该样本似然越高）
    fm_pol_w = per_sample_fm_loss(pred_pol_w, tgt_w, mask_w)
    fm_pol_l = per_sample_fm_loss(pred_pol_l, tgt_l, mask_l)
    with torch.no_grad():
        fm_ref_w = per_sample_fm_loss(pred_ref_w, tgt_w, mask_w)
        fm_ref_l = per_sample_fm_loss(pred_ref_l, tgt_l, mask_l)

    # Δ = L_FM^ref − L_FM^policy （策略把损失降得越多，Δ 越大 = 似然提升越多）
    delta_w = fm_ref_w - fm_pol_w
    delta_l = fm_ref_l - fm_pol_l
    logits = beta * (delta_w - delta_l)                # 赢应被推高、输应被压低
    loss = -F.logsigmoid(logits).mean()

    return DpoStats(
        loss=loss,
        reward_margin=float((delta_w - delta_l).mean().item()),
        reward_accuracy=float((delta_w > delta_l).float().mean().item()),
        policy_fm_win=float(fm_pol_w.mean().item()),
        policy_fm_lose=float(fm_pol_l.mean().item()),
    )


def anchor_sft_loss(ce_loss: torch.Tensor, eos_loss: torch.Tensor, lam: float) -> torch.Tensor:
    """winners 上的 CE+EOS anchor 正则，防文本条件/停止预测漂移。

    为什么: 纯 DPO 只学「相对偏好」，可能让绝对的文本对齐漂移；
    在赢样本上保留一点监督 SFT，是 DPO 的标准稳定化做法(λ 为消融旋钮)。
    """
    return lam * (ce_loss + eos_loss)
