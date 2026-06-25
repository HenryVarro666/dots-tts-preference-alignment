# alignment/ — 阶段⑤：偏好对齐（项目的方法核心）

**角色**：在 dots.tts-base 上做偏好对齐，产出各 treatment 的 checkpoint。

**为什么存在**：这是整个项目的**方法贡献**——把 DPO 适配到连续-latent flow-matching AR（公开空白），并开源再现 SOAR。

**上游**：`pairs/out/pairs_*.jsonl`（DPO）或 `data/prompts/*_train.jsonl`（open-SOAR）。
**下游**：`inference/` 用对齐后 ckpt 合成评测集。

## 文件
| 文件 | 作用 |
|---|---|
| `losses_dpo.py` | **核心损失**：DPO over flow-matching velocity loss（Diffusion-DPO 适配） |
| `train_dots_tts_dpo.py` | DPO 训练编排（策略 + 冻结 ref + 配对前向 + 共享 RNG） |
| `train_open_soar.py` | reward-free 自校正（on/off-trajectory 对），SOAR 开源再现 |
| `data_pairs.py` | 配对/单样本 DataLoader |
| `dots_preprocess.py` | **唯一**对接 dots.tts 内部的适配器（复用真实 `BasicTtsPipeline`+`PadCollator`，已实现） |
| `check_fm_loss.py` | **开训前自检**：FM 接缝通 + 共享-RNG 决定论 + 随机性存在 |
| `lora_adapter.py` | LoRA 包装（A6000 档；ref = `disable_adapter()`） |

## 方法一句话
赢/输样本共享同一 text，DPO 让策略相对冻结 ref **把赢样本的 FM 损失降得更多、输样本降得更少**。
per-sample L_FM 直接取自 `loss_terms["fm_loss"]`（dots.tts `model(prepared)` 的返回，不碰 DiT 内部）。数学见 [`../docs/01-METHOD.md`](../docs/01-METHOD.md)。

## ⚠️ 命门（务必读 `../docs/04-RISKS.md`）
1. **四前向共享 `(x0,t,VAE-z)`**：policy/ref × win/lose 必须在同一随机状态下前向（`_fm_per_sample` 用 `fork_rng`+`set_rng_state`），否则 FM 损失不可比、DPO 失效。
2. **接缝已实现**（`dots_preprocess.py` 复用 dots.tts 真实 `BasicTtsPipeline.process_sample` + `PadCollator`，数据契约从模型派生）。**开训前用 `check_fm_loss.py` 在 GPU 上验证**：接缝有限值 + 同 RNG 决定论 + 不同 RNG 有差异，三项 PASS 再训。

## 怎么用
```bash
# 0) 开训前自检（命门验证）——三项 PASS 再继续
python check_fm_loss.py --ckpt rednote-hilab/dots.tts-base --manifest ../data/prompts/dots_manifest_train.jsonl

# 1) 训各 treatment
accelerate launch train_dots_tts_dpo.py --config ../configs/dpo_multiobj_fullparam.yaml
accelerate launch train_dots_tts_dpo.py --config ../configs/dpo_weronly.yaml        # reward-hacking 臂
accelerate launch train_open_soar.py    --config ../configs/open_soar_selfcorrect.yaml
```
