"""
lora_adapter.py — LoRA 包装（A6000 档默认；ref = 关掉适配器的同一模型）

做什么:    用 PEFT 给 dots.tts 的 DiT flow-matching 头(+ 可选 LLM 注意力)注入 LoRA 适配器。
怎么做:    target_modules 默认指向 DiT 的 attention/MLP 投影(modules/backbone/dit.py 的命名)。
为什么这么设计:
    full-param DPO 需同时驻留 policy(含优化器状态) + 冻结 ref，2B 在 48GB×2 偏紧；
    LoRA 只训适配器，且 **ref = `disable_adapter()` 的同一模型** → 省掉第二份模型，显存大降。
对实验意义:
    LoRA vs full-param 是消融维度；也是「2×A6000 也能跑」的可行性保障(见 docs/04-RISKS 决策⑦)。

⚠️ target_modules 名称需对照 dots.tts dit.py 实际层名核对；本机只语法检查。
"""

from __future__ import annotations


def wrap_lora(model, lora_cfg: dict):
    """给模型注入 LoRA。返回带 `disable_adapter()` 上下文的 PEFT 模型。"""
    from peft import LoraConfig, get_peft_model  # type: ignore

    targets = lora_cfg.get("target_modules", [
        # DiT 头(声学渲染)：对齐主要作用处
        "to_q", "to_k", "to_v", "to_out", "fc1", "fc2",
        # 可选：LLM 注意力(语义规划)
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    cfg = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=targets,
        bias="none",
    )
    peft_model = get_peft_model(model, cfg)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    return peft_model
