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
| `dots_preprocess.py` | **唯一**对接 dots.tts 内部预处理的适配器（耦合都收敛在此） |
| `lora_adapter.py` | LoRA 包装（A6000 档；ref = `disable_adapter()`） |

## 方法一句话
赢/输样本共享同一 text+prompt，DPO 让策略相对冻结 ref **把赢样本的 FM 损失降得更多、输样本降得更少**。数学见 [`../docs/01-METHOD.md`](../docs/01-METHOD.md)。

## ⚠️ 两个命门（务必读 `../docs/04-RISKS.md`）
1. **四前向共享 `(x0,t)`**：policy/ref × win/lose 必须复用同一随机状态（`fork_rng`），否则 FM 损失不可比、DPO 失效。
2. **`dots_preprocess.py` 的 TODO**：`fm_patch_mask` / `target` 需对照 dots.tts `model.py`（`_compute_fm_loss_term` :1080、`prepare_training_batch` :1184）补齐——这是「方法成立」的接缝。

## 怎么用
```bash
accelerate launch train_dots_tts_dpo.py --config ../configs/dpo_multiobj_fullparam.yaml
accelerate launch train_dots_tts_dpo.py --config ../configs/dpo_weronly.yaml        # reward-hacking 臂
accelerate launch train_open_soar.py    --config ../configs/open_soar_selfcorrect.yaml
```
