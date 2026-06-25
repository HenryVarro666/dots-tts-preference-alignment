# 04 · 风险与决策点登记

> 每条：风险 / 为什么要紧 / 缓解 / 触发信号。跑前过一遍，跑中盯着。

## 风险① 候选多样性不足 → 偏好信号弱
- **为什么要紧**：`generate()` 无 seed，若网格制造的差异太小，候选奖励方差低，偏好对全是噪声。
- **缓解**：加大 `num_steps×guidance×speaker` 网格；`aggregate_reward.py` 打印候选池奖励方差，太小就回 ② 扩网格或加 K。
- **触发信号**：聚合时「候选池平均奖励方差」接近 0；DPO 训练 reward_accuracy 卡在 ~0.5。

## 风险② 四前向未共享 (x0,t) → DPO 失效【最高优先级】
- **为什么要紧**：L_FM 依赖随机 (x0,t)；不共享则 policy/ref、win/lose 不可比，似然比失真，训不出偏好。
- **缓解**：`train_dots_tts_dpo._forward_fm` 用 `fork_rng` + 固定 RNG state；写单测验证「同输入同 RNG → 同 L_FM」。
- **触发信号**：reward_margin 不随训练上升、loss 抖动、对齐无效果。

## 风险③ FM 接缝是否正确（已实现，开训前必自检）
- **为什么要紧**：`fm_loss` 若构造错，DPO 损失就是错的。
- **现状**：`dots_preprocess.py` 已复用 dots.tts 真实 `BasicTtsPipeline.process_sample` + `PadCollator`；per-sample L_FM 取自 `model(prepare_training_batch(batch))` 的 `loss_terms["fm_loss"]`；数据契约(sr / `audio_samples_per_llm_token=hop_size×patch_size`)从模型派生 —— 与训练器 L210-215 强校验一致。
- **缓解（必做）**：开训前在 GPU 跑 `python alignment/check_fm_loss.py`，确认三项 PASS：①FM 有限值（接缝通）②同 RNG 决定论（命门）③不同 RNG 有差异。
- **触发信号**：check 报 FAIL；或 L_FM 量级异常（与 dots.tts 训练日志 fm_loss 不在一个量级）。

## 风险④ 流式/离线 SIM 不可比
- **为什么要紧**：流式克隆仅 x-vector（无 prompt_text 续接），与离线条件不同。
- **缓解**：报告时分两列、分开标注；流式只比「流式 vs 流式」的 treatment 差。
- **触发信号**：把流式 SIM 直接和离线 SIM 比并下结论（错误）。

## 风险⑤ seed-tts-eval 脚本耦合集群（`$ARNOLD_WORKER_GPU` / `sudo split`）
- **为什么要紧**：原脚本为 ByteDance 集群写，单机无 sudo 会失败。
- **缓解**：`eval/run_seedtts_eval.sh` 已 `sed` 去 sudo、设 `ARNOLD_WORKER_GPU=1`；跑前 `bash -n` 校验。
- **触发信号**：cal_wer/cal_sim 报 sudo/环境变量错误。

## 风险⑥ 公开集采样率/许可未核实
- **为什么要紧**：VCTK 默认 FLAC、AISHELL-3 44.1k；采样率不符 48k 契约会静默劣化。
- **缓解**：跑前核实（`data/LICENSES.md` 标〔需核实〕）；AISHELL-3 重采样到 48k。
- **触发信号**：合成质量异常低、AudioVAE 编码报形状/采样率错。

## 风险⑦ 显存：full-param DPO 在 48GB 档可能 OOM
- **为什么要紧**：policy+优化器状态+冻结 ref+4 前向激活。
- **缓解**：H100 用 ZeRO-2；A6000 用 `configs/dpo_lora.yaml`(LoRA, ref=disable_adapter) 或 ZeRO-3+grad-checkpoint。
- **触发信号**：CUDA OOM；改 LoRA 或减 micro_batch / 加梯度累积。

## 风险⑧ 自动奖励噪声 → 偏好对有错配
- **为什么要紧**：WER(ASR 误转写)/SIM/UTMOS 都有噪声，可能把「其实更差」的判成 win。
- **缓解**：用 `threshold` 配对策略只留奖励差>τ 的对；多目标降低单指标噪声主导。
- **触发信号**：人工抽听若干 win/lose 对，发现判反。

## 风险⑨（Phase B）对齐增益被蒸馏/量化吃掉
- **为什么要紧**：MeanFlow 区间均值目标可能抹平对齐过的速度场；量化损细节。
- **缓解**：门控后专门测「对齐增益保留比例」；先 INT8 再尝试 INT4。
- **触发信号**：蒸馏/量化后 hard-WER 增益归零。

## 决策点（需用户拍板的，已在 PROTOCOL/configs 给默认）
- full-param(H100) vs LoRA(A6000)：默认按卡选。
- 数据规模/语言优先（默认 LJSpeech 打通 → VCTK(EN) + AISHELL-3(ZH)）。
- 是否做 streaming-aware 对齐(stretch)：A 主体跑通后再定。
