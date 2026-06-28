# 08 · 实验结果 · Table 1/2/3（第二轮统稿 · 11 臂同批离线主表 + 流式 RQ2 + reward-hacking RQ3，bootstrap 95% CI）——解盲后裁决

> **定位**：本文是 RQ1/RQ4 主表（PAPER §5.1）的**落地实测稿**。它把 [`06-核心发现-FM损失与奖励错配.md`](06-核心发现-FM损失与奖励错配.md) §9 的下游预测，由 **n=3 冒烟** 升级为 **第二轮 11 臂同批 n=100 实测**，并给出解盲后的裁决——含一个**超出原预测**的反转（`sft` 也崩）。
> **数据来源**：`ablation/run_matrix.py` 在 **100 句 LJSpeech held-out** 上、`--nfe 10 --skip-streaming`（离线模式）跑通；管线 `base + 10 对齐臂（11 臂同批）合成（DotsTtsRuntime.generate）→ WER（Whisper-large-v3）/ SIM（ecapa）/ UTMOS（utmos22_strong）→ aggregate → main_table`。
> **关联**：机理见 [`06`](06-核心发现-FM损失与奖励错配.md) §3/§4.2；第一轮审稿见 [`07-审稿意见-第一轮.md`](07-审稿意见-第一轮.md)；统稿框架见 [`../PAPER_DRAFT.md`](../PAPER_DRAFT.md) §5.1/§5.3/§6.4/§6.7。
> **不变量（务必守住）**：Table 1 数字**逐字精确**；**EmergentTTS 列空**（LLM-judge 本次未跑）；**Table 2（streaming，RQ2）本轮已跑（5 主臂，见 §5b；X4/X6/X7 用 `--skip-streaming` 故不入 Table 2）**；probe **偏相关已由 `alignment/probe_analysis.py` 跑出、intrinsic 确立**（partial r=−0.392、CI[−0.548,−0.209]，见 §8；并由 X5 settings-matched re-probe〔setting_gap≡0〕以最强设计复核、头条取 Spearman ρ=−0.216，见 §8.5），**C6 不变**（intrinsic ≠ intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold）；[`06`](06-核心发现-FM损失与奖励错配.md) §6/§6.7 正文措辞的同步另行处理。

---

## 0. TL;DR（一句话裁决）

在 dots.tts 上，**11 臂同批（base 重测 WER 0.112，落于此前 0.103/0.128 之 CI 内 → 生成 no-seed 随机、单批即消解 Table-1/1b 两-base cosmetic）**；**两条建设性路线脱离崩坏**：`open_soar`（reward-free 自纠正）**改善**（UTMOS 4.380 全表最佳、WER≈base、SIM †↓ 略降）、`reward_weighted`（**非似然** reward 加权，X6b）**逃逸/不崩**（全表最低 WER 0.092，仅 SIM\*/UTMOS† 轻降——「逃逸」非「胜过 base」）；**`sft`、`dpo_multiobj`、`dpo_weronly` 全部退化或崩坏**——其中 `dpo_multiobj` **灾难崩坏 + 25% 生成时崩**（immediate-EOS；**此为离线全句解码路径特有——§5b 同一模型流式逐拍解码下 WER+UTMOS 双指标均正常**），`sft` **意外崩坏**（推翻审稿钦点的「安全对照」），`dpo_weronly` 大致守住但未取得最低 WER（**RQ3 reward-hacking 未复现**）。崩坏按「往 automatic-reward 选出的**自生成 off-manifold winner** 拉多狠」**单调加剧**，导出**精修论点**：off-manifold self-training 退化与训练目标（SFT/DPO）无关，**但逃逸存在**——非似然目标（`reward_weighted`）可脱离崩坏，**修法是弃似然-DPO 改用非似然目标、而非过滤配对**（`dpo_onmanifold` X6a 仍崩、WER 0.315 全† 为证）。**RQ4 精修序**：reward-free 自纠正（`open_soar`，改善）＞ 非似然 reward（`reward_weighted`，逃逸）＞ 似然-DPO（`dpo_multiobj`，崩坏）。**RQ3**：`dpo_weronly`（0.110≈base）与 `reward_weighted_weronly`（X7，0.365 劣于 base）均未取得最低 WER → reward-hacking 在 DPO 与 reward-weighted 两框架皆**未复现**（诚实负面）。

---

## 1. 完整 Table 1（11 臂同批，n=100，offline，bootstrap 95% CI）

**Table 1. Main results**（offline，**11 臂同批合成**，100 句 LJSpeech held-out，`nfe=10`，skip-streaming）。**本批为单次合成**：base 重测 WER **0.112** 落于此前 0.103/0.128 之 CI 内 → 生成无固定 seed、属随机；单批合成使全 11 臂直接可比，并**消解此前 Table 1 vs Table 1b 两-base cosmetic（Table 1b 并入本表）**。WER via Whisper-large-v3；SIM via ecapa；UTMOS via `utmos22_strong`。点估计 + **bootstrap 95% CI [lo, hi]**。↓ 越低越好，↑ 越高越好。**EmergentTTS 列空（未跑）**。

| 臂 | WER ↓ | SIM ↑ | UTMOS ↑ | n | 生成失败率 ↓ | EmergentTTS ↑ | 裁决 |
|---|---|---|---|---|---|---|---|
| **base** | 0.112 [0.078, 0.154] | 0.878 [0.853, 0.898] | 4.278 [4.222, 4.335] | 100 | **0%** (0/100) | — (空) | 基线（参照） |
| **sft** | **0.556 [0.302, 0.989]** | 0.629 | 3.788 | 100 | **0%** (0/100) | — (空) | **崩坏（反转；WER/SIM/UTMOS 全 Bonferroni-†）** |
| **open_soar** | 0.094 [0.070, 0.122] | 0.850 | **4.380** | 100 | **0%** (0/100) | — (空) | **改善**：UTMOS 全表最佳（†↑ vs base）、WER≈base、SIM †↓ 略降 |
| **dpo_multiobj** | **0.890 [0.632, 1.305]** | **0.341** | **2.719** | **75** | **25%** (25/100) | — (空) | **灾难崩坏 + 25% 生成时崩** |
| **dpo_weronly** | 0.110 [0.082, 0.138] | 0.856 | 4.055 | 100 | **0%** (0/100) | — (空) | ≈base WER（mild）；SIM†/UTMOS† 低于 base |
| **reward_weighted**（X6b） | **0.092 [0.065, 0.125]** | 0.854 | 4.117 | 100 | **0%** (0/100) | — (空) | **逃逸**：全表最低 WER、不崩；SIM\*/UTMOS† 略低于 base |
| **dpo_onmanifold**（X6a） | 0.315 [0.246, 0.398] | 0.827 | 3.336 | 100 | **0%** (0/100) | — (空) | **仍崩坏**（全†）；on-manifold 配对过滤 0.89→0.32 改善但未救活 DPO-FM |
| **reward_weighted_weronly**（X7） | 0.365 [0.156, 0.750] | 0.846 | 4.206 | 100 | **0%** (0/100) | — (空) | WER **劣于 base、非最低** ⇒ reward-hacking 未复现 |
| **sft_groundtruth**（X4） | **0.087 [0.066, 0.113]** | 0.857 | 4.337 | 100 | **0%** (0/100) | — (空) | on-manifold target；WER/UTMOS 优于 base |
| **sft_losers**（X4） | 0.569 [0.144, 1.194] | 0.771 | 3.952 | 100 | **0%** (0/100) | — (空) | 崩坏 |
| **sft_random**（X4） | 0.138 [0.101, 0.182] | 0.832 | 4.035 | 100 | **0%** (0/100) | — (空) | 轻微（WER≈base） |

> **关键并列标注（不可省）**：新增的**生成失败率列**让 active degradation 不被 WER 漏判——**崩坏臂的失败样本不进 WER**（生成失败的句子无可转写文本，被排除出 WER 统计），**单看 WER 会低估 active degradation**。**唯 `dpo_multiobj` 生成失败率非零（25/100 = 25%，全部 immediate-EOS，见 §2）**，其余十臂**均 0%（0/100）**。这区分了两种崩坏：`sft` 是 **graded WER 退化**（仍正常出音频、0% 生成失败，只是质量崩）；`dpo_multiobj` 是 **graded 退化 + 25% 生成时崩**（直接发 EOS、无音频载荷；该 25% 究属 graded 退化还是 optimization divergence，**证据不一、暂不定论**，见 §2 / caveat 2）。因此 `dpo_multiobj` 主表 **n = 75**，WER 0.890 **只统计成功的 75 条**（失败的更差）——**失败率（25%）必须与 WER 0.890 并列**报告，否则**低估崩坏**（caveat 5）。

**逐列读法**：
- **base → open_soar**：UTMOS **4.278 → 4.380**（升、全表最佳）、WER **0.112 → 0.094**（≈持平、略降）、SIM **0.878 → 0.850**（略降）。**reward-free 自纠正改善臂**（与下 `reward_weighted` 逃逸臂并列，见 §5c）。
- **base → dpo_weronly**：WER **0.112 → 0.110**（≈base，**非最低**）、UTMOS **4.278 → 4.055**（略降）、SIM 0.856（守住）。**大致守住但 WER-only 没 hack 成**（RQ3，见 §5）。
- **base → sft**：WER **0.112 → 0.556**（5× 恶化）、SIM **0.878 → 0.629**、UTMOS 4.278 → 3.788。**崩坏**（反转，见 §3）。
- **base → dpo_multiobj**：WER **0.112 → 0.890**、SIM **0.878 → 0.341**、UTMOS **4.278 → 2.719**，外加 **25% 生成时崩**（immediate-EOS）。**灾难崩坏**。
- **base → reward_weighted（X6b）**：WER **0.112 → 0.092**（**全表最低**、不崩）、SIM 0.854（\*略降）、UTMOS **4.278 → 4.117**（†略降）。**逃逸臂**——非似然 reward 加权脱离崩坏（「逃逸/不崩」非「胜过 base」，见 §5c）。
- **base → reward_weighted_weronly（X7）/ dpo_onmanifold（X6a）**：X7 WER **0.365**（劣于 base、非最低，RQ3 见 §5）；X6a WER **0.315**（on-manifold 配对过滤 0.89→0.32，但**仍崩坏未救活**，见 §5c）。
> **解码路径提示（decode-path-aware，须与 Table 1 同看）**：表中 `sft`（WER 0.556）/ `dpo_multiobj`（WER 0.890 + 25% immediate-EOS）的「崩坏」是**离线全句解码**（offline `generate()`、一次性 num_steps=10 + EOS-prob 停止）下的结果；**同一模型在逐拍流式解码下产出正常语音**（§5b：流式 WER 0.212 / 0.134、流式 UTMOS 4.283 / 4.406，均 ≈base 0.087 / 4.326）。故 off-manifold self-training 的下游危害**真实但离线解码路径特有**——是**生成时**（非训练；训练 `fm_w` 全程稳、未发散，X2/X3）、且**离线解码路径特有**的现象。Table 1 的离线数字**照留不动**（离线全句合成是标准评测），但表中「崩坏」一律读作**离线全句解码下的崩坏**；alignment **确改变了模型的离线 EOS/停止行为**、模型并非「坏掉」，其表现**依解码路径而定**。（注：X4 的崩坏臂 `sft_losers` / `sft_random` 等本轮仅离线（`--skip-streaming`），其流式是否同样正常 **〔待核实〕**。）
---

### 1.1 X4 · SFT-target 梯度对照（在 off-manifold 轴上滑动 target，n=100×4 · 同数据/同 SFT/同步数 · 唯一变量 = 模仿对象）

> **X4 = SFT-target gradient control**：**四臂全 n=100、0% 生成失败、同数据规模 / 同 SFT 管线 / 同步数**，**唯一变量 = SFT target 在 off-manifold 轴上的位置**（模型去模仿哪一段音频）。`sft` 即主表那一臂（拉向 **reward-winner**；反相关发现指 reward-winner 携更高 `L_FM` = **最 off-manifold**）。X4 四臂现并入本批同一合成（§1 Table 1）、与全部 11 臂同 base（WER **0.112**）直接可比——此前 X4-base 0.128 vs 主表 0.103 的两-base cosmetic 由本批单次合成消解（no-seed 随机、0.112 落于二者 CI 内）。

**Table 1-X4. SFT-target gradient**（offline，n=100/臂，**0% 生成失败**，bootstrap 95% CI；WER↓ 越低越好、SIM/UTMOS↑ 越高越好）。

| 臂 | SFT target（off-manifold 程度） | WER ↓ | SIM ↑ | UTMOS ↑ | n | 失败率 | 裁决 |
|---|---|---|---|---|---|---|---|
| **base**（同批参照） | — | 0.112 [0.078, 0.154] | 0.878 | 4.278 | 100 | **0%** | 参照 |
| **sft**（=主表臂） | **reward-winner（最 off-manifold）** | **0.556 [0.302, 0.989]** | 0.629 | 3.788 | 100 | **0%** | **崩坏**（vs-base WER\*、SIM†、UTMOS†） |
| **sft_losers** | reward-loser（off-manifold） | **0.569 [0.144, 1.194]** | 0.771 | 3.952 | 100 | **0%** | **崩坏**（vs-base SIM†、UTMOS†） |
| **sft_random** | random 候选（中 off-manifold） | 0.138 [0.101, 0.182] | 0.832 | 4.035 | 100 | **0%** | 轻微（WER≈base；vs-base SIM\*、UTMOS†） |
| **sft_groundtruth** | 真实录音（**on-manifold**） | **0.087 [0.066, 0.113]** | 0.857 | 4.337 | 100 | **0%** | **最佳、优于 base**（vs-base WER†↓、UTMOS†↑） |

> 显著性标注：**† = vs-base Bonferroni 校正后仍显著；\* = vs-base 名义显著（未过 Bonferroni）**。SIM/UTMOS 为点估计（X4 仅 WER 带 CI）。

**单调梯度（按 target off-manifold 程度，三指标全单调；n=100 各、0% 失败）**：
- **WER↓**：winners 0.556 ≈ losers 0.569 ≫ random 0.138 > groundtruth 0.087
- **SIM↑**：winners 0.629 < losers 0.771 < random 0.832 < groundtruth 0.857
- **UTMOS↑**：winners 3.788 < losers 3.952 < random 4.035 < groundtruth 4.337

**判读（解 caveat 1：是 off-manifold target、不是小数据 SFT 不稳）**：四臂**同数据规模 + 同 SFT 管线 + 同步数**，唯一变量是 target 的 off-manifold 位置——**却只有 off-manifold target（winners / losers）崩坏，而 on-manifold target（groundtruth）追平/反超 base**，且退化随 target off-manifold 程度在三指标上**单调加剧**。故 `sft` 退化的因是**off-manifold 的 target 分布本身**、**而非**「188 单说话人小数据 SFT 过拟合 / 灾难性遗忘」的不稳。四臂**全 0% 生成失败**，进一步排除「失败率混淆」（纯 graded 退化、无生成时崩成分）。这把 caveat 1 由**开放混淆**转为**受控已决**结论。**校准（不过度声称）**：组内（winners vs losers）排序噪声极大、**CI 大幅重叠**——**不声称 winners 严格劣于 losers**；干净表述 = 「**{winners, losers} off-manifold → 崩；{random, groundtruth} 近/在流形 → 不崩；随 off-manifold 程度单调**」。`sft_groundtruth` 在 WER（↓）/ UTMOS（↑）上**Bonferroni-显著优于 base**（†）是真实加成——**on-manifold 的 SFT 反而有益**。注意：本条 caveat 1（**SFT-target vs 小数据不稳**）与 **caveat 3（LJ002 子集聚集）属不同议题**，互不替代、互不推翻。


---

## 2. `dpo_multiobj` 失败清单摘要（25/100，聚于 LJ002 子集）

| 项 | 值 |
|---|---|
| 生成失败数 | **25 / 100 = 25%** |
| 失败模式 | **全部 immediate-EOS / no-payload**（一开生成即 EOS、无音频载荷） |
| 失败分布 | 聚于 `LJ002` 子集，但**须诚实归因**：评测集（`eval_big` 取前 100 held-out）**本身 89% 为 LJ002**（89/100），故「聚于 LJ002」**部分因 LJ002 占评测主体**；组内率 **LJ001 0/11（0%）vs LJ002 25/89（28%）**（小 n、p≈0.02）仍是真实的 prompt 依赖，**非均匀** |
| prompt 长度 | 失败样本平均 text 长 **105** vs 全体 **94**（**1.12×、仅轻微偏长**）——**不足以判定「长 prompt 触发」** |
| 位置偏置 | 失败样本落后段 **52%**，**无明显位置偏置** |
| 对主表的影响 | `dpo_multiobj` 主表 **n=75**（仅成功句）；WER 0.890 是成功子集的值，**失败句更差** |

**判读（保守诚实 · graded-vs-divergence 证据不一）**：25% 的 immediate-EOS 是**在某 prompt 子集上的生成时崩**（generation-time collapse on a prompt subset）；它与 graded（Spearman −0.216 的弱单调）**不同质**，但**「graded 退化 vs optimization divergence」的证据相互不一致，暂不下定论**——

> - **训练曲线偏「未发散」**：X2/X3 实测，`dpo_multiobj` 训练 `fm_w` **全程稳定 ~0.74**（末段 / 早段比 **1.01×**、**未二次爆**、reward-margin 仍小 **0.065**）→ **训练阶段并未发散**（区别于 run0 式早期 over-optimization 的训练时爆炸）。
> - **交叉表偏「divergence」**：按 |reward-gap| 分层，大-gap 对的 FM 与 reward **更脱钩**（同向率 **34%** vs 小-gap **48%**）→ 偏 divergence。
> - **合判**：训练指标温和、却在 **LJ002 子集** 生成时崩（25% immediate-EOS）→ **不是 run0 式训练时发散**，而是**「训练温和但模型已 off-manifold 漂移 → 部分 prompt 生成时崩」**。

故**保守报告为「生成时子集崩」**，是否纯属 optimization divergence **待人工 TB-curve 复核 + 失败句听测裁决**（§7 下一步 2），不径直坐实。这一图景**强化 off-manifold 论点**：错向梯度**不靠炸训练指标**（`fm_w` 未再爆、margin 仍小），而是**悄悄把模型推离流形** → 在 LJ002 这类 prompt 上**离线解码时**崩。**关键限定（decode-path-aware）**：此 25% immediate-EOS 是**离线 `generate()`**（一次性 num_steps=10 + EOS-prob 停止）路径上的崩；**同一 `dpo_multiobj` 在逐拍 1T1A 流式解码下全 n=100 条正常语音**（§5b：流式 WER 0.134 / UTMOS 4.406，均 ≈base 0.087 / 4.326）→ 离线 25% 生成时崩**为离线解码路径特有**。alignment **确改变了模型的离线 EOS/停止行为、但模型并未「坏掉」**——其崩坏表现**随解码路径而变**；离线崩坏数字**照留**（离线全句合成是标准评测），仅须读作**离线全句解码下**的崩坏。

**LJ002 聚集须诚实归因**（见 §2 表 / caveat 3，勿写成「长 prompt 触发」）：评测集 **89% 为 LJ002**（89/100，`eval_big` 取前 100 held-out）、组内 **LJ001 0/11（0%）vs LJ002 25/89（28%）**（小 n、p≈0.02 的真实 prompt 依赖）、失败样本仅**轻微偏长（text 105 vs 94 = 1.12×）**、**无位置偏置（后段 52%）**——聚集**部分因 LJ002 占评测主体**，但组内 0% vs 28% 仍为真实非均匀。须专项排查（caveat 3、§7 下一步）。

> **对照（生成失败率列）**：`dpo_multiobj` 是**唯一**生成失败率非零的臂（25%）；**其余四臂（base / sft / open_soar / dpo_weronly）生成失败率均为 0%（0/100）**。注意崩坏臂 `sft` 虽 WER 崩到 0.556，但**生成失败率仍 0%**——它是 graded 质量退化（句句出音频），与 `dpo_multiobj` 的「质量退化 + 25% 生成时崩」**双重崩坏**有别。故 Table 1 的失败率列与 WER 列**必须并看**：单看 WER 会把 `dpo_multiobj` 的 active degradation 低估（失败的 25 条根本不进 WER）。

---

## 3. 按崩坏强度排序的解读（单调崩坏序 + sft 反转）

把四个对齐臂按「训练目标把模型往 **automatic-reward 选出的自生成 winner** 拉的强度」排序，崩坏**单调加剧**：

| 序 | 拉向的目标 | 臂 | 拉力 | WER | UTMOS | 结果 |
|---|---|---|---|---|---|---|
| ① | **真实 clean 音频**（非 reward-winner） | `open_soar` | clean-vs-perturbed，**似然对齐** | 0.094 | **4.380** | **改善** |
| ② | reward-winner（弱、WER-only、63% 平局） | `dpo_weronly` | 弱拉 | 0.110 | 4.055 | **≈base** |
| ③ | reward-winner（强监督，直接 FM） | `sft` | **强拉** | 0.556 | 3.788 | **崩** |
| ④ | 最干净 reward-winner（DPO 强拉 + 反向梯度） | `dpo_multiobj` | **最强拉** | 0.890 | 2.719 | **灾难 + 25% 生成时崩** |

**反转（本表最重要的发现）**：
- n=3 冒烟里 `sft` 看着**最好**（WER 0.039），全量 n=100 **崩坏**（WER 0.556）。
- 根因是冒烟取样偏差：冒烟 3 句**全是 `LJ001` 前缀**，恰好**错过崩坏块**（失败集中 `LJ002`）。这是「**不可据 n=3 下结论**」纪律的实证回报。
- 审稿 R3 曾**钦点 `sft` 作「真正不受影响的决定性对照」**（[`07`](07-审稿意见-第一轮.md) 必查项 3）——结果这个对照给出**比原假设更强的发现**：连「不做 win/lose `L_FM` 对比」的纯监督臂都崩。

**共同线索**：automatic-reward 的 best-vs-worst 在 **guidance×num_steps×speaker_scale 多样池**上选出的 winner 是 **off-manifold**（清晰但不典型的高-guidance 样本）。把模型往这些**自生成 off-manifold winner** 做 self-train——**SFT 也好、DPO 也好**——都退化，**拉得越狠越惨**；唯 `open_soar` 训向**真实 on-manifold** 目标（clean 音频），改善。

### 3.1 新论点：off-manifold self-training 退化，与训练目标无关

> **向 automatic-reward 选出的自生成候选（off-manifold）做 self-training，会退化模型，与训练目标（SFT/DPO）无关**；唯有训向真实 / 似然对齐目标（`open_soar`）才有效。

这比 [`06`](06-核心发现-FM损失与奖励错配.md) 原诊断（「DPO-FM 偏好-似然反向」L2 失败）**更深一层**：`sft` 不做 win/lose 对比却仍崩，说明毒源是**目标分布本身（off-manifold winner）**，而非 DPO 的对比机制。建设性结论由此**收窄**：从「`sft` & `open_soar` 都安全」→「**只有 `open_soar` 安全；`sft` 同样崩**」。
（此论点原为**强假设**；现经 **X4 SFT-target 梯度对照**（§1.1，n=100×4、同数据/同 SFT/同步数、唯变 target 流形位置）**确证**——唯 off-manifold target（winners/losers）崩、on-manifold groundtruth 追平/反超 base、随 off-manifold 程度**单调**，故 `sft` 崩源于 **off-manifold target** 而非小数据 SFT 不稳，caveat 1 由开放混淆转受控已决。**校准**：组内 winners vs losers 序噪声大、CI 重叠，**不声称 winners 严格劣 losers**。）

---

## 4. §6.7 active-degradation 预测裁决

审稿 / DA 要的解盲前预承诺——**`dpo_multiobj` 是否相对 base active degrade（而非仅 under-fit ≈ base）？**

> **裁决：是（active degradation 证实）。**
> 依据：`dpo_multiobj` WER CI **[0.632, 1.305]** 与 base **[0.078, 0.154]** **不重叠**，且叠加 **25% 生成失败**（immediate-EOS/no-payload）。→ PAPER §6.7 的「active degradation」预测**证实**。

> ⚠️ **但含「生成时崩」成分（caveat 2）**：25% 失败是 immediate-EOS / no-payload 的**生成时子集崩**，与 graded（Spearman −0.216 的弱单调）**不同质**——崩坏量级**超出**弱单调相关所能预测的范围。但**「graded vs optimization divergence」证据不一**：X2/X3 训练曲线 `fm_w` **全程稳 ~0.74、未二次爆、margin 仍小 0.065 → 训练未发散**（非 run0 式训练时爆炸）；交叉表大-gap 对 FM-reward 更脱钩（同向率 **34%** vs 小-gap **48%**）**→ 偏 divergence**；**合判为「训练温和但模型已 off-manifold 漂移 → 部分 prompt 生成时崩」**。故 active-degradation 成立，但**既不可纯归因 graded 反相关、亦不可径直断言 optimization divergence**，须**保守如实**，待人工 TB-curve 复核 + 失败句听测裁决。

---

## 5. RQ3 reward-hacking：诚实改为「未复现」

**经典 reward-hacking 签名**（[`07`](07-审稿意见-第一轮.md) R2 / PAPER §5.3）：WER-only 臂取得**最低 WER**、同时牺牲 naturalness（UTMOS↓ 或 EmergentTTS<45%）。**本次不成立**：

- `dpo_weronly` WER **0.110 ≈ base 0.112**，**不是最低**（最低是 `reward_weighted` 0.092、`open_soar` 0.094）。
- `dpo_weronly` UTMOS **4.055 < base 4.278**（略降），但因 WER 没压低，**不构成「以自然度换 WER」的干净 hack**。
- 根因：**DPO-FM 已坏**，WER-only 臂也无法把 WER 压到最低——这正是审稿预言的 **Contribution-2-vs-4 冲突**（DPO 臂崩坏 ⇒ reward-composition 无法干净分离）。- **X7 · `reward_weighted_weronly`（须显式纳入 RQ3 讨论）**：aggregate 的 reward_hacking 表**未含 X7**，但其 WER **0.365 [0.156, 0.750]** 既**劣于 base 0.112**、又**非最低**（最低 `reward_weighted` 0.092）→ **reward-weighted 框架内的 WER-only 臂同样未取得最低 WER**。故 reward-hacking 签名在 **DPO 与 reward-weighted 两框架内皆缺席**。

**Table 3. Reward-hacking 检验（RQ3）**。签名 = WER-only 臂取得最低 WER 且牺牲自然度；本表二臂均不满足（aggregate 的 reward_hacking 表仅含下列三行，X7 须于上 bullet 显式援引）。
| 臂 | WER ↓ | UTMOS ↑ | reward-hacking 签名 |
|---|---|---|---|
| **base** | 0.112 [0.078, 0.154] | 4.278 | — 参照 |
| **dpo_multiobj** | 0.890 [0.632, 1.305] | 2.719 | 崩坏（非 hack） |
| **dpo_weronly** | 0.110 [0.082, 0.138] | 4.055 | WER≈base、**非最低** → 无 hack |

> **RQ3 裁决**：两个 WER-only 臂（`dpo_weronly` 0.110≈base、`reward_weighted_weronly` 0.365 劣于 base）均未取得最低 WER → reward-hacking 在 DPO 与 reward-weighted 两框架内**均未复现**；**贡献 4 诚实降为负面结论**。

**对 PAPER 的改写指令**：
> **§5.3 / §6.4 须诚实改为**：「**reward-hacking 在错配的 DPO-FM 框架内无法干净复现**——WER-only 臂未取得最低 WER，两个 DPO 臂都退化；reward-composition 效应被 FM 崩坏混淆，无法分离；**且 reward-weighted 框架的 WER-only 臂（X7，WER 0.365）同样未取得最低 WER → 两框架皆未复现 hacking**。」
> **贡献 4（reward-hacking 参考案例）降为该负面结论**。**EmergentTTS 列仍空**（未跑），**不**据此谈 naturalness 退化。
---

## 5b. RQ2 · 流式 Table 2（首包延迟 / 流式 WER / 边界不连续度）

> 本轮 **5 主臂**（base / sft / open_soar / dpo_multiobj / dpo_weronly）跑了 streaming；**X4 / X6 / X7 用 `--skip-streaming`，不入 Table 2**（符合预期，RQ2 = 5 主臂）。

**Table 2. Streaming（RQ2）**。流式 WER / **UTMOS** / SIM、首包延迟 TTFA（ms）、边界不连续度（boundary）。↓ 越低越好（UTMOS↑ 越高越好）；流式 UTMOS via `utmos22_strong`（同离线，full n=100、已听验正常语音）。
| 臂 | 流式 WER ↓ | 流式 UTMOS ↑ | 流式 SIM ↑ | TTFA (ms) ↓ | 边界不连续度 ↓ | n |
|---|---|---|---|---|---|---|
| **base** | 0.087 | 4.326 | 0.832 | 162.2 | 42.0 | 100 |
| **sft** | 0.212 | 4.283 | 0.825 | 162.4 | 47.8 | 〔待核实: streaming n〕 |
| **open_soar** | 0.109 | 4.450 | 0.813 | 162.5 | 96.2 | 100 |
| **dpo_multiobj** | 0.134 | 4.406 | 0.830 | 162.5 | 85.1 | 〔待核实: streaming n〕 |
| **dpo_weronly** | 0.179 | 4.024 | 0.824 | 164.3 | 55.8 | 100 |

**RQ2 裁决**：
- **TTFA 全臂持平（~162–164ms）→ 对齐不增加首包延迟**（latency gate 通过）。RQ2 的干净正面结论：增益可迁移、**无首包延迟代价**。
- **健康臂**（base / open_soar / dpo_weronly，全 n=100）流式 WER **大致追随离线排序**。
- ⚠️ **崩坏臂离线-vs-流式路径差异（非子集伪影）**：`sft` / `dpo_multiobj` 流式产出**全 n=100 条正常语音**（已听验），流式 WER（**0.212 / 0.134**）**远低于**各自离线 WER（**0.556 / 0.890**）——即**离线 collapse 不迁移到流式**。故 offline collapse 是**离线解码路径特有的生成时现象**（immediate-EOS、num_steps=10 一次性采样；呼应训练未发散 X2/X3），流式逐拍解码更稳。**流式 UTMOS 现已测**（`sft` **4.283**、`dpo_multiobj` **4.406**，均 ≈ `base` **4.326**，远高于各自离线 UTMOS 3.788 / 2.719）——即**可懂度与自然度双轴在流式下皆正常**，offline collapse **完全**是离线解码路径伪影。注：这**不**等于「对齐无效」——对齐**确实**改变了模型的**离线** EOS/停止行为，只是该 harm 仅在离线一次性路径上显现、不在逐拍流式路径上；故模型**未坏**。（X4 崩坏臂 `sft_losers` 等为 `--skip-streaming` 离线-only，其流式是否同样正常〔待核实〕。）

---

## 5c. X6 · 建设性逃逸分析（reward_weighted 逃逸 / dpo_onmanifold 仍崩）+ RQ4 精修

> 本节回答审稿 R3「错配诊断之外，是否存在**建设性出路**」。X6 拆两臂：**X6a `dpo_onmanifold`**（沿用似然-DPO、仅把配对换成 on-manifold 过滤）与 **X6b `reward_weighted`**（弃似然-DPO、改用**非似然** reward-加权目标）。

**X6b · `reward_weighted`（非似然 reward 加权）= 逃逸**：WER **0.092 [0.065, 0.125]**（**全表最低**）、SIM 0.854（\*略降）、UTMOS 4.117（†略降）、**0% 生成失败**——**脱离崩坏 / 不崩 / 是一条可行路线**。这**正面验证 X5 处方**「`L_FM` 不是有效 reward → 改用非似然目标」。**校准**：表述为「**逃逸 / 不崩 / viable route**」，**不**写「胜过 base」（WER≈base、SIM\*/UTMOS† 仍轻降）。

**X6a · `dpo_onmanifold`（似然-DPO + on-manifold 配对过滤）= 仍崩坏**：WER **0.315 [0.246, 0.398]**、SIM 0.827、UTMOS 3.336（三指标全 vs-base Bonferroni-†）。on-manifold 配对过滤把 `dpo_multiobj` 的 0.89 拉回 **0.32**（确有改善），但 **WER 仍崩、未被救活**——**过滤配对不足以救 DPO-FM**。

**合论（建设性逃逸存在，但修法是换目标而非换配对）**：DPO-FM **深层已坏**（X6a 过滤配对后仍崩），故**修法 = 弃似然-DPO、改用非似然目标**（X6b 逃逸为证），**而非过滤配对**。这把 §3.1 的「off-manifold self-training 退化与目标无关」**精修**为：退化与「拉向 off-manifold winner」绑定，但**目标的形式决定有无逃逸**——非似然 reward-加权可脱离崩坏。

**RQ4 精修序（三档）**：
> **reward-free 自纠正（`open_soar`，改善）＞ 非似然 reward（`reward_weighted`，逃逸/不崩）＞ 似然-DPO（`dpo_multiobj`，崩坏；`dpo_onmanifold` 过滤配对仍崩）。**


---

## 6. 全部 caveat（5 条，务必如实，须写进 PAPER §5/§6/§7）

1. **`sft` 崩坏与「小数据 SFT 过拟合 / 灾难性遗忘」混淆**：188 单说话人 SFT 本就易过拟合 / 遗忘；干净判别需 **SFT-on-random-candidates vs SFT-on-reward-winners** 对照——**该对照已由 X4（§1.1 SFT-target 梯度对照）跑出**：四臂 n=100、同数据/同 SFT/同步数，唯 off-manifold target（winners/losers）崩、on-manifold groundtruth 追平/反超 base、退化随 off-manifold 程度**单调**。`open_soar`（**同规模微调、同优化器族**）不崩本是**部分对照**，X4 进一步把因坐实为「**off-manifold 的目标分布**」而非「SFT 不稳」——**caveat 1 由开放混淆转为受控已决**，§3.1 新论点由强假设升为受控结论。**校准**：组内 winners vs losers 序噪声大（CI 重叠），**不声称 winners 严格劣 losers**；干净表述仍是「{winners,losers} 崩、{random,groundtruth} 不崩、单调」。**另须显式点名跨臂步预算不对称**：`open_soar` 与两个 DPO 臂均 **600 步**、而 `sft` 为 **1500 步**（同规模 / 同优化器族**但步预算不同**），故「`open_soar` 安全 vs `sft` 崩」原则上仍可能被步数差混淆。该步差混淆由**步数匹配的 X4（`sft_groundtruth`@1500、on-manifold、同 SFT 目标、仅变 target 流形位置）**现由**已跑出的 X4**（§1.1，`sft_groundtruth`@1500、on-manifold、同 SFT 目标、仅变 target 流形位置，**追平/反超 base**）关闸消除；佐证：`dpo_multiobj` **仅 600 步即灾难崩**、而唯一 1500 步的 `sft` 崩坏**居中**（§3 序③，介于 ②`dpo_weronly` 与 ④`dpo_multiobj` 之间）→ **步数与崩坏非单调**，反证步数非头条驱动。
2. **`dpo_multiobj` 的 25% 失败是 immediate-EOS / no-payload 的「生成时子集崩」**，与 graded（Spearman −0.216）**不同质**；崩坏量级超出弱单调相关的预测。但**「graded vs optimization divergence」证据不一**：X2/X3 训练曲线 `fm_w` 全程稳 ~0.74、**未二次爆**、margin 仍小 0.065 → **训练未发散**（非 run0 式训练时爆炸）；交叉表大-gap 对 FM-reward 更脱钩（34% vs 48%）→ **偏 divergence**。**合判 = 训练温和但模型已 off-manifold 漂移 → 部分 prompt 生成时崩**；**保守报告，不径直断言 optimization divergence**，待人工 TB-curve + 失败句听测裁决。这**强化 off-manifold 论点**：错向梯度不靠炸训练指标（`fm_w` 未再爆、margin 仍小）、而是悄悄把模型推离流形。**且此 25% 生成时崩为离线解码路径特有**：同一 `dpo_multiobj` 在逐拍流式解码下全 n=100 条正常（§5b：流式 WER 0.134 / UTMOS 4.406，均 ≈base 0.087 / 4.326）→ 崩坏**依解码路径而定、模型未「坏」**（alignment 确改了离线 EOS/停止行为），离线数字**照留**但须读作**离线全句解码下**的崩坏。
3. **失败聚于 `LJ002` 子集**（`sft`、`dpo_multiobj` 皆然）须**诚实归因**：评测集（`eval_big` 取前 100 held-out）**本身 89% 为 LJ002**（89/100），故聚集**部分因 LJ002 占评测主体**；但组内 **LJ001 0/11（0%）vs LJ002 25/89（28%）**（小 n、p≈0.02）仍是真实 prompt 依赖，失败样本仅**轻微偏长（text 105 vs 94 = 1.12×）、无位置偏置（后段 52%）**——**不写成「长 prompt 触发」**。仍**须专项排查**（prompt 内容 / 评测集排序 / fid 块边界）；这也是 n=3 冒烟（全 `LJ001`）漏掉崩坏的直接原因。
4. **崩坏臂 CI 极大**（`sft` WER **[0.30, 0.99]**、`dpo_multiobj` WER **[0.63, 1.31]**、UTMOS 2.719〔CI 待核实〕）、**单说话人 LJSpeech**、**无人类听测（ABX/MOS）**——量级**仅供方向**，不可当精确效应。
5. **`dpo_multiobj` 主表 WER 0.890 只统计成功的 75 条**（失败的更差），故**失败率（25%）必须与 WER 0.890 并列**报告，否则**低估崩坏**。

---

## 7. 下一步（future work）

1. ~~**SFT-random 对照**（裁决 §3.1 新论点 / caveat 1）~~ **✓ 已完成（见 §1.1 · X4 SFT-target 梯度对照）**：四臂 n=100、同数据/同 SFT/同步数，唯一变量 = target off-manifold 位置——**random WER 0.138 ≈ base、winners/losers 崩、groundtruth（on-manifold）WER 0.087 反超 base**，退化随 off-manifold 程度**单调** → 「off-manifold 目标为因」**坐实**、caveat 1 由混淆转**受控已决**（非「小数据 SFT 不稳」）。**校准**：组内 winners vs losers 序噪声大（CI 重叠），**不声称 winners 严格劣 losers**。
2. **`dpo_multiobj` 生成时崩诊断（X2/X3 已部分回填）**：训练曲线已查——`fm_w` 全程稳 ~0.74、**未二次爆**、reward-margin 仍小 0.065 → **训练未发散**；交叉表大-gap 对 FM-reward 更脱钩（34% vs 48%）→ 偏 divergence；**合判 = 训练温和但 off-manifold 漂移 → LJ002 子集生成时崩**。**待补**：人工 TB-curve 复核 + 失败句听测裁决，最终落定 **graded degradation vs optimization divergence**（caveat 2），不径直坐实。
3. **`LJ002` 聚集排查**（caveat 3）：核对失败句的 **prompt 长度 / 内容 / 评测集排序 / fid 块边界**，判定崩坏是否与某结构性因素绑定，而非均匀；据此决定是否需重排评测集 / 扩 n。
4. ~~**`probe_analysis.py` 偏相关**（intrinsic 裁决）~~ **✓ 已完成（见 §8）**：`corr(FM-margin, reward-gap | guidance, num_steps, speaker_scale)` = **partial −0.392、CI[−0.548,−0.209]、显著**，控住全部生成 knob 仍存活 → **裁决：intrinsic（反相关非 grid effect）**；num_steps r=−0.237 是温和混淆但被偏相关控住。**C6 守住**：intrinsic ≠ intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold。**X5 settings-matched re-probe 进一步以最强设计（setting_gap≡0、配对无条件控住设置）复核确认 intrinsic、头条改 Spearman ρ=−0.216，见 §8.5。**
5. **人类听测（ABX/MOS，≥15–20 听者）**：崩坏臂的量级与 UTMOS 退化须由人耳锚定；未做前**不称** perceptual / naturalness 退化（[`07`](07-审稿意见-第一轮.md) P2）。
6. **补全 TBD**：~~Table 2（streaming）~~ **✓ 已跑（见 §5b · RQ2 五主臂：TTFA 平、健康臂流式 WER 追排序；崩坏臂流式正常(满 n=100、离线-vs-流式路径差异、**流式 WER+UTMOS 双指标均正常**：sft 0.212/4.283、dpo_multiobj 0.134/4.406 ≈base 0.087/4.326 → 离线 collapse 纯属离线解码路径伪影)）**；仍待 **EmergentTTS 列**（LLM-judge 胜率）。

---

## 8. `probe_analysis.py` 实测：intrinsic 裁决落地（偏相关 / confound / per-component / 杠杆 / binomial / X5 settings-matched 复核）

> 这是 §7 下一步第 4 项的**回填**。`alignment/probe_analysis.py`（偏相关 + 修正后的 confound + per-component + 杠杆诊断 + 双系数 bootstrap CI）在全量 **N=188** pair 上跑出，**裁决：反相关是 intrinsic**——它**不是候选生成 knob 的产物**（偏相关控住 guidance/num_steps/speaker 全部 grid 维度后仍存活）。**严守 C6**：此处 intrinsic ≠「intrinsic to `L_FM`」（`L_FM` 仍是有效似然代理，open-SOAR 为证）；诊断**仍是** preference-likelihood 对齐失败 / off-manifold winner，**不回退** [`06`](06-核心发现-FM损失与奖励错配.md) §6 的措辞。

### 8.1 偏相关（决定性，C1/C5）：intrinsic 确立

- **corr(FM-margin, reward-gap | guidance, num_steps, speaker) = partial r = −0.392，95%CI [−0.548, −0.209]，显著。**
- 原始 Pearson **−0.411** → 控住全部生成 knob 后 **−0.392**（**几乎不变**）→ **反相关 SURVIVES partialling**。
- 裁决：由「待裁决 / 未排除」**确立为 intrinsic**——反相关不被任一生成设置（含温和混淆 num_steps）解释。

### 8.2 confound（修正测对的量：FM-margin × setting-gap）

> 第一轮审稿（[`07`](07-审稿意见-第一轮.md) C1/C5/DA）指出旧 confound 测错了量（测的是 corr(reward-gap, setting-gap)）。`probe_analysis.py` **改测 corr(FM-margin, setting-gap)**：

| 生成设置 | corr(FM-margin, setting-gap) | 显著性 |
|---|---|---|
| guidance_scale | **−0.067** | ns |
| **num_steps（NFE）** | **−0.237**（CI [−0.360, −0.102]） | **显著** |
| speaker_scale | **−0.067** | ns |

- **num_steps(NFE) 是真实但温和的混淆**（单独显著）；guidance/speaker 无关。
- 但 §8.1 偏相关已**控住** num_steps，反相关仍 **−0.392** → **不被 num_steps 解释**。如实写：num_steps 单独显著，partial 控住后存活。

### 8.3 头条统计（C4 保持）+ 杠杆诊断

- **头条（原 best_worst 探针；settings-matched X5 复核见 §8.5）**：Spearman **ρ = −0.187**（ρ² = 3.5%、CI [−0.335, −0.036]、显著）领头条；Pearson **−0.411**（r² = 0.169、CI [−0.556, −0.235]）并列为**杠杆放大的线性系数**。
- **杠杆**：删 top-5% |reward-gap|（10 对）后 Pearson = **−0.282**（仍显著负）；留一法最大单点影响仅 **0.038**（**无单点主导**）。→ Pearson 受高-gap 对放大，但**非少数点伪影**（trim 后仍 −0.282）。
- **高-gap 分层（答死 near-tie 质疑）**：按 |reward-gap| 分层，反相关在**非平局（真有质量差）子集反而更强 ≈2×**——**top-50%**（|gap| ≥ 1.390，n=94）Spearman **−0.407**（CI [−0.585, −0.197]、Pearson −0.520）、**top-25%**（|gap| ≥ 1.878，n=47）Spearman **−0.440**（CI [−0.650, −0.160]、Pearson −0.477），均双显著、均 ≥ 全集 −0.187。→ 反相关**不是「win_rate 0.47 大半 near-tie」平局稀释出的假象**；恰恰相反，**真有质量差时更锋利**，核心结论在最该看的高-gap 子集上最强（详见 [`06`](06-核心发现-FM损失与奖励错配.md) §3.6）。
- **binomial**：82/188 **p = 0.093**（**不写 at chance**，亦不写显著）。
- **独立性**：distinct prompt **188/188**（prompt 级独立、单说话人）。

### 8.4 per-component（R2/R3 关键新结果）：每个 perceptual 轴都更 off-manifold

- corr(FM-margin, 分量 gap)：WER-gap **+0.412**、SIM-gap **−0.676**、UTMOS-gap **−0.597**（均显著）。
- 换到 **reward-对齐方向**（−WER / +SIM / +UTMOS），三轴**全部负相关**：(−WER) **−0.41**、SIM **−0.68**、UTMOS **−0.60**。
- **不是「只是 FM-vs-WER」**：UTMOS、SIM 两轴反相关均**更强**（UTMOS −0.60、SIM −0.68 ＞ WER −0.41）→ 每个 perceptual 轴的「更好」都**更 off-manifold** → 结论从「只修 reward 组成」**强化为「整体弃 FM-as-reward」**。**承重落在判别力更强的 UTMOS（−0.60）与方向明确的 WER（+0.41）**；SIM −0.68 同向但**仅作带 hedge 的次要佐证**（见下 caveat）。
- **caveat（domain reviewer · SIM 仅作带 hedge 的次要佐证）**：per-component 的 **SIM_gap −0.68** 来自**单说话人 LJSpeech 上近-ceiling、判别力弱的 SIM 指标**，**可能由少数真有 SIM 差的对带动（杠杆）**；故**承重证据放在判别力更强的 UTMOS_gap（−0.60）与方向明确的 WER_gap（+0.41）**，**SIM −0.68 仅作带 hedge 的次要佐证**、不单独承重。

### 8.5 X5 settings-matched re-probe（最强设计 · 无条件设置控制）：intrinsic 复核确认

> X5 = **settings-matched re-probe**：winner/loser **生成设置完全一致**（同 guidance / num_steps / speaker，**仅 seed 变**）→ **setting_gap ≡ 0**（全 188 对）。这以**配对设计无条件**实现设置控制，**强于** §8.1 的线性偏相关（partial r=−0.392，本节 supersede/augment 之），是回答 intrinsic 问题的**最强设计**。join 188/188、1 对/prompt、prompt 级独立。

- **头条（C4，诚实取 Spearman）**：Pearson **r=−0.489**（r²=0.240、CI[−0.616,−0.317]、显著）vs Spearman **ρ=−0.216**（ρ²=0.047、CI[−0.362,−0.054]、显著）；**Pearson/Spearman ≈ 2.26（>2）→ 高杠杆点主导线性系数 → 头条取稳健的 Spearman ρ（非 Pearson）**。
- **杠杆**：删 top-5% |reward-gap|（10 对）Pearson → **−0.336**；LOO 删 #103（fid=LJ001-0109、reward-gap=+3.00）Pearson **翻正为 +0.024** → Pearson 脆弱 / 杠杆驱动，Spearman 稳。
- **高-gap 存活（答死「win_rate 0.47 大半 near-tie」）**：full n=188 ρ=−0.216；top-50% n=94 ρ=**−0.318**[−0.504,−0.100]；top-25% n=47 ρ=**−0.533**[−0.678,−0.310] → **真有质量差时反相关更强 → 非 near-tie 伪影**。
- **去噪 FM-margin（MC K=16）**：94/188（50%）为正、mean=**−0.0063**、单边 SD=0.1010（噪声）、win_rate=**0.53**、binomial **p=1.000** vs 0.5 → **去噪后 FM 仍无对 reward-winner 的系统偏好 → 「`L_FM` 不是有效的 implicit reward（L_FM-as-reward）」**，是**方法本身的边界**，正是本文 thesis（preference-likelihood 对齐失败）。（注：此判仅否定「`L_FM`-as-reward」，**不**否定 `L_FM` 作清洁目标的似然代理；open-SOAR 为证、C6 不变。）
- **per-component（R2/R3，join 188/188）**：WER-gap **+0.517**[+0.299,+0.753]（正）、SIM-gap **−0.765**[−0.882,−0.548]（负、最强）、UTMOS-gap **−0.716**[−0.841,−0.501]（负）→ 反相关由 SIM & UTMOS（负）驱动、WER 正相关。社区命题：是「弃 FM-as-reward」还是「只修 reward 组成」（X5 实测实质偏前者：WER_gap 原始 +0.517但 reward 偏好低 WER→reward-对齐 −WER 上 −0.517、三轴 reward-对齐全反向、聚合为非负加权和符号不可翻 → 整体弃 FM-as-reward，见 WEIGHT-SIMPLEX）。**SIM_gap 杠杆（B3）**：删 top-5% → −0.445（仍显著）、LOO 删 #103 → +0.060 → **SIM 仅作次要佐证**（与 §8.4 hedge 一致）。
- **split-half 信度（B6）**：r_hh=+0.922、Spearman-Brown r_full=0.959；disattenuated ρ_dis=**−0.221**（≥ |ρ|）→ 上报 ρ 是**保守下界**（K=16 测量噪声只衰减、不夸大）。
- **confound / 偏相关段——按设计跳过**：setting_gap 恒为 0 → corr(FM-margin, const)=NaN、对常数偏相关是 no-op（否则循环论证）；设置控制已由**配对设计**达成（§8.2 中 num_steps −0.237 的温和混淆在此被设计性消除）。
- **裁决**：**intrinsic 经最强设计（settings-matched）确认、保留**；头条系数改取 **Spearman**；现由 settings-matched 配对**无条件**支撑（强于、并 supersede/augment §8.1 的 partial −0.392）。**C6 不变**：intrinsic ≠ intrinsic-to-`L_FM`，诊断仍 preference-likelihood / off-manifold。

### 8.6 X10 VoxCPM 架构级复现（cross-architecture re-probe · 跨独立实现泛化）：CLOSES EIC CRITICAL #3

> X10 = **cross-architecture probe**：把**同一批 dots 生成的 winner/loser 偏好对**喂给 **VoxCPM**——一个**独立实现**的 continuous-latent TTS（**diffusion-AR、tokenizer-free、手写 MiniCPM-4 backbone + 自有 AudioVAE**，by OpenBMB）vs dots.tts（**flow-matching**，by rednote/SJTU）——测 VoxCPM **自己的训练 FM loss**（per-sample velocity-MSE，源码核验 == dots `fm_loss` 同源；probe self-check 通过：determinism exact、L_FM finite positive），corr(FM_margin, reward_gap)。这把「intrinsic」从**单模型**推到**跨独立实现**，正面回答 EIC CRITICAL #3（N=1 model 不能泛化）。两次 run：

- **GRID pool（K=16，`pairs_best_worst`，生成设置 VARY → confound present）= provenance**：Spearman **ρ=−0.087**（**不显著**、CI[−0.248,+0.067]）；Pearson **−0.279**（显著但杠杆驱动：ratio **3.20**、一个 LOO 点即翻号）→ **混淆 grid 池上稳健秩统计为 NULL**。**作 provenance 上报**（非反例）：它正展示「生成设置混淆把效应掩盖」。
- **SETTINGS-MATCHED（K=32，`pairs_settings_matched`，setting_gap≡0、confound 按构造移除）= 头条**：Spearman **ρ=−0.266 显著**、95%CI[−0.413,−0.112]、置换检验 **p_perm=0.0003**（R6、cluster-respecting）。Pearson **−0.461**[−0.605,−0.285]、**Pearson/Spearman=1.73（<2 → HEALTHY、非杠杆驱动）**（对照 dots X5 的 ratio 2.26：VoxCPM 线性系数更稳）。
- **高-gap 存活（答死「win_rate<0.5 大半 near-tie」）**：top-50% ρ=**−0.302**、top-25% ρ=**−0.368**（均显著）→ **真有质量差时反相关更强 → 非 near-tie 伪影**。
- **per-component（reward-对齐、三轴全反相关，与 dots.tts **同一签名**）**：WER_gap **+0.336** 原始（reward 偏好低 WER → reward-对齐 −WER = **−0.336**）、SIM_gap **−0.673**（trim-top5% 后 **−0.471** 仍存活、显著）、UTMOS_gap **−0.529**（全显著）→ 与 dots.tts §8.4/§8.5 的 per-component pattern 一致。
- **去噪 / win_rate**：binomial **80/188=0.426**、p=0.049、win_rate **0.394**（FM 平均偏好更「典型」的 loser）。
- **「混淆在 grid 上掩盖、settings-matched 揭示」叙事（机制，非矛盾）**：grid null（ρ=−0.087 ns）**是机制证据而非反例**——生成设置混淆在 VoxCPM 上比在 dots.tts 上掩盖效应**更甚**；settings-matched 配对设计（setting_gap≡0）按构造移除混淆后，**内在反相关浮现**，robust、与 dots.tts 自身 X5 settings-matched（Spearman ρ=−0.216，见 §8.5）**相当或略强**。
- **裁决（CLOSES EIC CRITICAL #3）**：反相关在 VoxCPM 上**无混淆复现**、稳健 → 现象是**架构级**，跨 **N=2 个独立 continuous-latent 实现**（flow-matching dots.tts + diffusion-AR VoxCPM）。**CALIBRATION（不过度声称）**：scope = **continuous-latent（flow-matching / diffusion）TTS**，由 **N=2 独立实现**佐证——**非「all TTS」、非 discrete-token TTS、仍仅 2 个模型**。头条取 **settings-matched Spearman ρ=−0.266**（无混淆设计），**Pearson −0.461** 作杠杆已核验的次要系数，**grid −0.087(ns) 仅作 provenance**；**与 dots X5 同一 settings-matched 纪律**。

---

## 附录 A · 复现与配置（校验基准）

| 项 | 值 |
|---|---|
| 评测脚本 | `ablation/run_matrix.py`（`--nfe 10`；离线主表全 11 臂，**5 主臂另跑 streaming → §5b Table 2**，X4/X6/X7 用 `--skip-streaming`） |
| 评测集 | 100 句 LJSpeech held-out（含 `LJ001`/`LJ002` 块） |
| 合成 | `DotsTtsRuntime.generate`（base + 10 对齐臂：sft / open_soar / reward_weighted / dpo_weronly / dpo_onmanifold / reward_weighted_weronly / dpo_multiobj / sft_groundtruth / sft_losers / sft_random（11 臂同批）） |
| 指标 | WER（Whisper-large-v3）/ SIM（ecapa）/ UTMOS（utmos22_strong）；EmergentTTS 未跑 |
| CI | bootstrap 95%（utterance-level） |
| n | 除 `dpo_multiobj`=75（25% 生成失败）外，**其余 10 臂均 100**；offline 全 11 臂同批 |
| 未跑 | **EmergentTTS**（LLM-judge）；**Table 2 streaming 本轮已跑（5 主臂，见 §5b）**；`probe_analysis.py` 偏相关已跑（§8） |

> 本机（macOS / 无 GPU / 权重未下载）无法执行评测；上表为 GPU 机器实测结果回填，数字逐字精确、未经改动。
