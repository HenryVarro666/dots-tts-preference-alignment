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

## 风险⑩ VoxCPM 探针 8 处〔需核实〕接缝（X10，架构级结论的命门）
- **为什么要紧**：`alignment/voxcpm_preprocess.py` + `probe_fm_reward_voxcpm.py` 把 dots.tts 的反相关探针移植到 **VoxCPM**（tokenizer-free continuous diffusion-AR、手写 MiniCPM-4 backbone + 自有 AudioVAE，16k 入/48k 出，见 `06`）。两者的 diffusion/FM 损失项命名、latent 取法、`(x0,t)` 共享接口、x-vector/说话人条件、AudioVAE encode 与 dots.tts **不同**；**8 处接缝**任一移植错，探针测的就不是「同一个量」，VoxCPM 反相关结论（X10 的架构级证据）即失效——这是把「可能架构特定」升为架构级证据的命门。
- **缓解**：8 处全标〔需核实〕，逐处对照 VoxCPM 源码（`06` 已记 100% first-party、45 文件）核 API；移植后先在 VoxCPM 上做 `check_fm_loss` 等价自检（同 RNG 决定论 + 不同 RNG 有差异 + L_FM 有限值），**通过才跑探针**；VoxCPM 探针须复刻 dots.tts 协议——win/lose 共享同一 `(x0,t)`、K-MC、`no_grad`。
- **触发信号**：VoxCPM `L_FM` 量级与其训练日志不在一个量级；同 RNG 不决定论；VoxCPM 探针相关性与 dots.tts 差异巨大**且无机理解释**（先怀疑接缝错，而非真架构差异）。

## 风险⑪ 多语种 ZH 须用 CER（非 word-WER）+ 采样率（X11）
- **为什么要紧**：AISHELL-3 是中文——中文按词切分无意义，**WER 必须用 CER（字错率）**，沿用英文 word-level WER 会得到系统性错误指标。且 AISHELL-3 **44.1 kHz**、VCTK 默认 FLAC，与 dots.tts **48k 契约**不符，静默重采样会劣化（与风险⑥同源，此处针对多语种臂再强调一次）。
- **缓解**：ZH 评测在 `eval/score_synth.py`/`reward/score_wer.py` 走 **CER 分支**（中文 normalize + char-level）；AISHELL-3 重采样到 **48k**；`data/run_multilingual.md` 逐语种记录 metric（EN=WER / ZH=CER）与采样率约定。
- **触发信号**：ZH WER 异常高（实为 word-split artifact）；中英指标混排不可比；AudioVAE 编码报采样率/形状错。

## 风险⑫ 流式 Table 2 的 DoubleStreaming 版本兼容（X8）
- **为什么要紧**：X8 去掉 `--skip-streaming` 跑流式 Table 2，走 `DotsTtsRuntimeDoubleStreaming.start_double_streaming`（B-12 已按官方 example 对齐 API）。但流式 API 是 **dots.tts 版本敏感**的——服务器上 dots.tts 版本若与 B-12 对齐时不同，`start_double_streaming` 的类名/签名/参数可能漂移；且流式克隆**仅 x-vector**（风险④，无 prompt_text）。
- **缓解**：跑前对照**当前** dots.tts 版本的 `scripts/example_double_streaming.py` 核 API；流式 SIM 只比「流式 vs 流式」的 treatment 差（风险④）；版本/签名不符 **fail-fast**，别让它跑出静默错误的 Table 2。
- **触发信号**：`start_double_streaming` 报 `TypeError`/`AttributeError`；流式输出与离线量级差异异常、或流式 SIM 直接拿去和离线比下结论（错误）。

## 风险⑬ reward_weighted 的权重函数选择（X6b/X7，决定建设性臂与 RQ3 复现成败）
- **为什么要紧**：`alignment/train_reward_weighted.py` + `data_pairs.build_weighted_dataloader` 在 winners 上做 reward-加权回归（非似然目标）；**权重函数**（线性 reward / `exp(reward/τ)` / rank-based / 截断）直接决定它是「温和偏好」还是「激进 reward-hacking」。X7（`reward_weighted_weronly`）要在**非错配框架内干净复现 reward-hacking**（RQ3）——权重选太软 hack 不出来、选太硬训练发散，两端都让 RQ3 复现失败或不可解读。
- **缓解**：权重函数与温度 `τ` 作**显式可配项**，先小规模扫几档；`reward_weighted`（多目标，建设性出路）与 `reward_weighted_weronly`（仅 WER，reward-hacking 复现）**共用同一权重函数**以保证可比；记录所选函数 + provenance。
- **触发信号**：reward_weighted 训练发散 / loss NaN；reward-hacking 复现不出 **graded** 退化（WER 降但 SIM/UTMOS 单调崩）。

## 风险⑭ settings-matched 候选需重生成（X5，非复用第一轮候选）
- **为什么要紧**：`generate_candidates.py --fixed-setting`（默认锁 `num_steps=10`/`guidance=1.2`/`speaker_scale=1.5`）产出的是**一批全新候选**（不是复用第一轮 `candidates_index.jsonl`），须重跑候选生成 + reward 打分 + 配对（~GPU 时间成本）。且 fixed-setting 若选在多样性低的角落（如固定低 guidance），候选多样性可能塌（风险①），探针无信号。
- **缓解**：fixed-setting 取第一轮候选池的**中位 knob**（让差异来自随机性而非 knob 网格）；重生成后先抽查候选池 **reward 方差**（风险①触发信号）再跑探针；settings-matched 探针与 `probe_analysis.py` 偏相关**互为印证、不互相替代**。
- **触发信号**：固定 setting 后候选池平均 reward 方差接近 0（风险①）；settings-matched 下 high-gap pair 配不出（多为 near-tie），探针 N 不足。

## 决策点（需用户拍板的，已在 PROTOCOL/configs 给默认）
- full-param(H100) vs LoRA(A6000)：默认按卡选。
- 数据规模/语言优先（默认 LJSpeech 打通 → VCTK(EN) + AISHELL-3(ZH)）。
- 是否做 streaming-aware 对齐(stretch)：A 主体跑通后再定。
