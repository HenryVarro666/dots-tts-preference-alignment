# 08 · 实验结果 · Table 1（全量 n=100 离线主表，bootstrap 95% CI）——解盲后裁决

> **定位**：本文是 RQ1/RQ4 主表（PAPER §5.1）的**落地实测稿**。它把 [`06-核心发现-FM损失与奖励错配.md`](06-核心发现-FM损失与奖励错配.md) §9 的下游预测，由 **n=3 冒烟** 升级为 **全量 n=100 实测**，并给出解盲后的裁决——含一个**超出原预测**的反转（`sft` 也崩）。
> **数据来源**：`ablation/run_matrix.py` 在 **100 句 LJSpeech held-out** 上、`--nfe 10 --skip-streaming`（离线模式）跑通；管线 `base + 4臂 合成（DotsTtsRuntime.generate）→ WER（Whisper-large-v3）/ SIM（ecapa）/ UTMOS（utmos22_strong）→ aggregate → main_table`。
> **关联**：机理见 [`06`](06-核心发现-FM损失与奖励错配.md) §3/§4.2；第一轮审稿见 [`07-审稿意见-第一轮.md`](07-审稿意见-第一轮.md)；统稿框架见 [`../PAPER_DRAFT.md`](../PAPER_DRAFT.md) §5.1/§5.3/§6.4/§6.7。
> **不变量（务必守住）**：Table 1 数字**逐字精确**；**EmergentTTS 列空**（LLM-judge 本次未跑）；**Table 2（streaming）仍 TBD/planned**（本次 `--skip-streaming`）；probe **偏相关已由 `alignment/probe_analysis.py` 跑出、intrinsic 确立**（partial r=−0.392、CI[−0.548,−0.209]，见 §8），**C6 不变**（intrinsic ≠ intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold）；[`06`](06-核心发现-FM损失与奖励错配.md) §6/§6.7 正文措辞的同步另行处理。

---

## 0. TL;DR（一句话裁决）

在 dots.tts 上，**5 臂里只有 `open_soar` 改善（UTMOS 4.408 五臂最佳、WER≈base、SIM 略降）**；**`sft`、`dpo_multiobj`、`dpo_weronly` 全部退化或崩坏**——其中 `dpo_multiobj` **灾难崩坏 + 25% 生成发散**，`sft` **意外崩坏**（推翻审稿钦点的「安全对照」），`dpo_weronly` 大致守住但未取得最低 WER（**RQ3 reward-hacking 未复现**）。崩坏按「往 automatic-reward 选出的**自生成 off-manifold winner** 拉多狠」**单调加剧**，导出新论点：**off-manifold self-training 退化，与训练目标（SFT/DPO）无关**；唯训向真实/似然对齐目标（`open_soar`）有效。

---

## 1. 完整 Table 1（n=100，offline，bootstrap 95% CI）

**Table 1. Main results**（offline，100 句 LJSpeech held-out，`nfe=10`，skip-streaming）。WER via Whisper-large-v3；SIM via ecapa；UTMOS via `utmos22_strong`。点估计 + **bootstrap 95% CI [lo, hi]**。↓ 越低越好，↑ 越高越好。**EmergentTTS 列空（未跑）**。

| 臂 | WER ↓ | SIM ↑ | UTMOS ↑ | n | 生成失败率 ↓ | EmergentTTS ↑ | 裁决 |
|---|---|---|---|---|---|---|---|
| **base** | 0.103 [0.071, 0.142] | 0.891 [0.877, 0.904] | 4.309 [4.273, 4.347] | 100 | **0%** (0/100) | — (空) | 基线 |
| **sft** | **0.545 [0.251, 1.059]** | **0.673 [0.605, 0.741]** | 3.917 [3.758, 4.064] | 100 | **0%** (0/100) | — (空) | **崩坏（反转）** |
| **open_soar** | **0.093 [0.070, 0.121]** | 0.844 [0.831, 0.860] | **4.408 [4.386, 4.432]** | 100 | **0%** (0/100) | — (空) | **唯一改善** |
| **dpo_multiobj** | **0.752 [0.566, 1.035]** | **0.365 [0.280, 0.460]** | **2.841 [2.583, 3.105]** | **75** | **25%** (25/100) | — (空) | **灾难崩坏 + 25% 失败** |
| **dpo_weronly** | 0.115 [0.087, 0.145] | 0.852 [0.836, 0.868] | 4.112 [4.050, 4.172] | 100 | **0%** (0/100) | — (空) | 大致守住 |

> **关键并列标注（不可省）**：新增的**生成失败率列**让 active degradation 不被 WER 漏判——**崩坏臂的失败样本不进 WER**（生成失败的句子无可转写文本，被排除出 WER 统计），**单看 WER 会低估 active degradation**。**唯 `dpo_multiobj` 生成失败率非零（25/100 = 25%，全部 immediate-EOS，见 §2）**，其余四臂**均 0%（0/100）**。这区分了两种崩坏：`sft` 是 **graded WER 退化**（仍正常出音频、0% 生成失败，只是质量崩）；`dpo_multiobj` 是 **graded 退化 + 25% 生成发散**（直接发 EOS、无音频载荷）。因此 `dpo_multiobj` 主表 **n = 75**，WER 0.752 **只统计成功的 75 条**（失败的更差）——**失败率（25%）必须与 WER 0.752 并列**报告，否则**低估崩坏**（caveat 5）。

**逐列读法**：
- **base → open_soar**：UTMOS **4.309 → 4.408**（升、五臂最佳）、WER **0.103 → 0.093**（≈持平、略降）、SIM **0.891 → 0.844**（略降）。**唯一净改善臂**。
- **base → dpo_weronly**：WER **0.103 → 0.115**（≈base，**非最低**）、UTMOS **4.309 → 4.112**（略降）、SIM 0.852（守住）。**大致守住但 WER-only 没 hack 成**（RQ3，见 §5）。
- **base → sft**：WER **0.103 → 0.545**（5× 恶化）、SIM **0.891 → 0.673**、UTMOS 4.309 → 3.917。**崩坏**（反转，见 §3）。
- **base → dpo_multiobj**：WER **0.103 → 0.752**、SIM **0.891 → 0.365**、UTMOS **4.309 → 2.841**，外加 **25% 发散**。**灾难崩坏**。

---

## 2. `dpo_multiobj` 失败清单摘要（25/100，全 LJ002）

| 项 | 值 |
|---|---|
| 生成失败数 | **25 / 100 = 25%** |
| 失败模式 | **全部 immediate-EOS / no-payload**（一开生成即 EOS、无音频载荷） |
| 失败分布 | **全集中在 `LJ002-xxxx` 块**（非均匀散布） |
| 对主表的影响 | `dpo_multiobj` 主表 **n=75**（仅成功句）；WER 0.752 是成功子集的值，**失败句更差** |

**判读**：25% 的 immediate-EOS 不是「graded 退化」，而是 **optimization divergence**——策略被反向梯度推到「直接发 EOS」的退化吸引子。这与 graded（Spearman −0.187 的弱单调）**不同质**（caveat 2）。**失败全落在 `LJ002` 块**这一聚集（与 `sft` 崩坏块一致，见 §3）强烈提示：崩坏与 **prompt 长度 / 内容 / 评测集排序**绑定，**非均匀崩坏**，须专项排查（caveat 3、§7 下一步）。

> **对照（生成失败率列）**：`dpo_multiobj` 是**唯一**生成失败率非零的臂（25%）；**其余四臂（base / sft / open_soar / dpo_weronly）生成失败率均为 0%（0/100）**。注意崩坏臂 `sft` 虽 WER 崩到 0.545，但**生成失败率仍 0%**——它是 graded 质量退化（句句出音频），与 `dpo_multiobj` 的「质量退化 + 生成发散」**双重崩坏**有别。故 Table 1 的失败率列与 WER 列**必须并看**：单看 WER 会把 `dpo_multiobj` 的 active degradation 低估（失败的 25 条根本不进 WER）。

---

## 3. 按崩坏强度排序的解读（单调崩坏序 + sft 反转）

把四个对齐臂按「训练目标把模型往 **automatic-reward 选出的自生成 winner** 拉的强度」排序，崩坏**单调加剧**：

| 序 | 拉向的目标 | 臂 | 拉力 | WER | UTMOS | 结果 |
|---|---|---|---|---|---|---|
| ① | **真实 clean 音频**（非 reward-winner） | `open_soar` | clean-vs-perturbed，**似然对齐** | 0.093 | **4.408** | **改善** |
| ② | reward-winner（弱、WER-only、63% 平局） | `dpo_weronly` | 弱拉 | 0.115 | 4.112 | **≈base** |
| ③ | reward-winner（强监督，直接 FM） | `sft` | **强拉** | 0.545 | 3.917 | **崩** |
| ④ | 最干净 reward-winner（DPO 强拉 + 反向梯度） | `dpo_multiobj` | **最强拉** | 0.752 | 2.841 | **灾难 + 25% 发散** |

**反转（本表最重要的发现）**：
- n=3 冒烟里 `sft` 看着**最好**（WER 0.039），全量 n=100 **崩坏**（WER 0.545）。
- 根因是冒烟取样偏差：冒烟 3 句**全是 `LJ001` 前缀**，恰好**错过崩坏块**（失败集中 `LJ002`）。这是「**不可据 n=3 下结论**」纪律的实证回报。
- 审稿 R3 曾**钦点 `sft` 作「真正不受影响的决定性对照」**（[`07`](07-审稿意见-第一轮.md) 必查项 3）——结果这个对照给出**比原假设更强的发现**：连「不做 win/lose `L_FM` 对比」的纯监督臂都崩。

**共同线索**：automatic-reward 的 best-vs-worst 在 **guidance×num_steps×speaker_scale 多样池**上选出的 winner 是 **off-manifold**（清晰但不典型的高-guidance 样本）。把模型往这些**自生成 off-manifold winner** 做 self-train——**SFT 也好、DPO 也好**——都退化，**拉得越狠越惨**；唯 `open_soar` 训向**真实 on-manifold** 目标（clean 音频），改善。

### 3.1 新论点：off-manifold self-training 退化，与训练目标无关

> **向 automatic-reward 选出的自生成候选（off-manifold）做 self-training，会退化模型，与训练目标（SFT/DPO）无关**；唯有训向真实 / 似然对齐目标（`open_soar`）才有效。

这比 [`06`](06-核心发现-FM损失与奖励错配.md) 原诊断（「DPO-FM 偏好-似然反向」L2 失败）**更深一层**：`sft` 不做 win/lose 对比却仍崩，说明毒源是**目标分布本身（off-manifold winner）**，而非 DPO 的对比机制。建设性结论由此**收窄**：从「`sft` & `open_soar` 都安全」→「**只有 `open_soar` 安全；`sft` 同样崩**」。
（此论点为**强假设**，须严守 §6 caveat 1——`sft` 崩与小数据 SFT 过拟合尚未干净分离。）

---

## 4. §6.7 active-degradation 预测裁决

审稿 / DA 要的解盲前预承诺——**`dpo_multiobj` 是否相对 base active degrade（而非仅 under-fit ≈ base）？**

> **裁决：是（active degradation 证实）。**
> 依据：`dpo_multiobj` WER CI **[0.566, 1.035]** 与 base **[0.071, 0.142]** **不重叠**，且叠加 **25% 生成失败**（immediate-EOS/no-payload）。→ PAPER §6.7 的「active degradation」预测**证实**。

> ⚠️ **但含发散成分（caveat 2）**：25% 失败是 immediate-EOS / optimization **divergence**，与 graded（Spearman −0.187 的弱单调）**不同质**——崩坏量级**超出**弱单调相关所能预测的范围，更像**反相关梯度驱动的 over-optimization 发散**（DA 早疑）。故 active-degradation 成立，但**不可纯归因 graded 反相关**，须如实区分 **graded degradation vs optimization divergence**。

---

## 5. RQ3 reward-hacking：诚实改为「未复现」

**经典 reward-hacking 签名**（[`07`](07-审稿意见-第一轮.md) R2 / PAPER §5.3）：WER-only 臂取得**最低 WER**、同时牺牲 naturalness（UTMOS↓ 或 EmergentTTS<45%）。**本次不成立**：

- `dpo_weronly` WER **0.115 ≈ base 0.103**，**不是最低**（最低是 `open_soar` 0.093）。
- `dpo_weronly` UTMOS **4.112 < base 4.309**（略降），但因 WER 没压低，**不构成「以自然度换 WER」的干净 hack**。
- 根因：**DPO-FM 已坏**，WER-only 臂也无法把 WER 压到最低——这正是审稿预言的 **Contribution-2-vs-4 冲突**（DPO 臂崩坏 ⇒ reward-composition 无法干净分离）。

**对 PAPER 的改写指令**：
> **§5.3 / §6.4 须诚实改为**：「**reward-hacking 在错配的 DPO-FM 框架内无法干净复现**——WER-only 臂未取得最低 WER，两个 DPO 臂都退化；reward-composition 效应被 FM 崩坏混淆，无法分离。」
> **贡献 4（reward-hacking 参考案例）降为该负面结论**。**EmergentTTS 列仍空**（未跑），**不**据此谈 naturalness 退化。

---

## 6. 全部 caveat（5 条，务必如实，须写进 PAPER §5/§6/§7）

1. **`sft` 崩坏与「小数据 SFT 过拟合 / 灾难性遗忘」混淆**：188 单说话人 SFT 本就易过拟合 / 遗忘；干净判别需 **SFT-on-random-candidates vs SFT-on-reward-winners** 对照（future work）。`open_soar`（**同规模微调、同优化器族**）不崩是**部分对照**，指向「**目标分布（off-manifold winner）**」而非单纯「SFT 不稳」——但**未完全排除**，§3.1 新论点仍是强假设。
2. **`dpo_multiobj` 的 25% 失败是 immediate-EOS / 发散**，与 graded（Spearman −0.187）**不同质**——更像反相关梯度驱动的 **over-optimization 发散**；崩坏量级超出弱单调相关的预测，须如实区分 **graded degradation vs optimization divergence**。
3. **失败全集中在 `LJ002` 块**（`sft`、`dpo_multiobj` 皆然）→ 疑与 **prompt 长度 / 内容 / 评测集排序**有关，**须查**，**非均匀崩坏**；这也是 n=3 冒烟（全 `LJ001`）漏掉崩坏的直接原因。
4. **崩坏臂 CI 极大**（`sft` WER **[0.25, 1.06]**、`dpo_multiobj` WER **[0.57, 1.04]**、UTMOS **[2.58, 3.11]**）、**单说话人 LJSpeech**、**无人类听测（ABX/MOS）**——量级**仅供方向**，不可当精确效应。
5. **`dpo_multiobj` 主表 WER 0.752 只统计成功的 75 条**（失败的更差），故**失败率（25%）必须与 WER 0.752 并列**报告，否则**低估崩坏**。

---

## 7. 下一步（future work）

1. **SFT-random 对照**（裁决 §3.1 新论点 / caveat 1）：跑 **SFT-on-random-candidates vs SFT-on-reward-winners** —— 若 random 不崩、reward-winner 崩，则「off-manifold 目标」为因，新论点坐实；若两者都崩，则混入「小数据 SFT 不稳」，须降级表述。
2. **`dpo_multiobj` 发散诊断**：定位 25% immediate-EOS 的触发——训练曲线 EOS-loss / reward-margin 轨迹、是否 over-optimization 吸引子；区分 graded degradation vs optimization divergence（caveat 2）。
3. **`LJ002` 聚集排查**（caveat 3）：核对失败句的 **prompt 长度 / 内容 / 评测集排序 / fid 块边界**，判定崩坏是否与某结构性因素绑定，而非均匀；据此决定是否需重排评测集 / 扩 n。
4. ~~**`probe_analysis.py` 偏相关**（intrinsic 裁决）~~ **✓ 已完成（见 §8）**：`corr(FM-margin, reward-gap | guidance, num_steps, speaker_scale)` = **partial −0.392、CI[−0.548,−0.209]、显著**，控住全部生成 knob 仍存活 → **裁决：intrinsic（反相关非 grid effect）**；num_steps r=−0.237 是温和混淆但被偏相关控住。**C6 守住**：intrinsic ≠ intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold。
5. **人类听测（ABX/MOS，≥15–20 听者）**：崩坏臂的量级与 UTMOS 退化须由人耳锚定；未做前**不称** perceptual / naturalness 退化（[`07`](07-审稿意见-第一轮.md) P2）。
6. **补全 TBD**：Table 2（streaming：首包延迟 / RTF / 边界不连续度 / 流式 WER，本次 `--skip-streaming` 未跑）；EmergentTTS 列（LLM-judge 胜率）。

---

## 8. `probe_analysis.py` 实测：intrinsic 裁决落地（偏相关 / confound / per-component / 杠杆 / binomial）

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

- **头条**：Spearman **ρ = −0.187**（ρ² = 3.5%、CI [−0.335, −0.036]、显著）领头条；Pearson **−0.411**（r² = 0.169、CI [−0.556, −0.235]）并列为**杠杆放大的线性系数**。
- **杠杆**：删 top-5% |reward-gap|（10 对）后 Pearson = **−0.282**（仍显著负）；留一法最大单点影响仅 **0.038**（**无单点主导**）。→ Pearson 受高-gap 对放大，但**非少数点伪影**（trim 后仍 −0.282）。
- **高-gap 分层（答死 near-tie 质疑）**：按 |reward-gap| 分层，反相关在**非平局（真有质量差）子集反而更强 ≈2×**——**top-50%**（|gap| ≥ 1.390，n=94）Spearman **−0.407**（CI [−0.585, −0.197]、Pearson −0.520）、**top-25%**（|gap| ≥ 1.878，n=47）Spearman **−0.440**（CI [−0.650, −0.160]、Pearson −0.477），均双显著、均 ≥ 全集 −0.187。→ 反相关**不是「win_rate 0.47 大半 near-tie」平局稀释出的假象**；恰恰相反，**真有质量差时更锋利**，核心结论在最该看的高-gap 子集上最强（详见 [`06`](06-核心发现-FM损失与奖励错配.md) §3.6）。
- **binomial**：82/188 **p = 0.093**（**不写 at chance**，亦不写显著）。
- **独立性**：distinct prompt **188/188**（prompt 级独立、单说话人）。

### 8.4 per-component（R2/R3 关键新结果）：每个 perceptual 轴都更 off-manifold

- corr(FM-margin, 分量 gap)：WER-gap **+0.412**、SIM-gap **−0.676**、UTMOS-gap **−0.597**（均显著）。
- 换到 **reward-对齐方向**（−WER / +SIM / +UTMOS），三轴**全部负相关**：(−WER) **−0.41**、SIM **−0.68**、UTMOS **−0.60**。
- **不是「只是 FM-vs-WER」**：SIM/UTMOS 反相关**更强** → 每个 perceptual 轴的「更好」都**更 off-manifold** → 结论从「只修 reward 组成」**强化为「整体弃 FM-as-reward」**。
- **caveat（domain reviewer）**：SIM 在单说话人 LJSpeech 近 ceiling、判别弱，其 **−0.68 须标注可能被少数有 SIM 差的对带动**；**UTMOS −0.60 较实**。

---

## 附录 A · 复现与配置（校验基准）

| 项 | 值 |
|---|---|
| 评测脚本 | `ablation/run_matrix.py`（`--nfe 10 --skip-streaming`，离线模式） |
| 评测集 | 100 句 LJSpeech held-out（含 `LJ001`/`LJ002` 块） |
| 合成 | `DotsTtsRuntime.generate`（base + 4 臂：sft / open_soar / dpo_multiobj / dpo_weronly） |
| 指标 | WER（Whisper-large-v3）/ SIM（ecapa）/ UTMOS（utmos22_strong）；EmergentTTS 未跑 |
| CI | bootstrap 95%（utterance-level） |
| n | base/sft/open_soar/dpo_weronly = 100；**dpo_multiobj = 75（25% 生成失败）** |
| 未跑 | Table 2 streaming（`--skip-streaming`）、EmergentTTS（`probe_analysis.py` 偏相关**已跑**，见 §8） |

> 本机（macOS / 无 GPU / 权重未下载）无法执行评测；上表为 GPU 机器实测结果回填，数字逐字精确、未经改动。
