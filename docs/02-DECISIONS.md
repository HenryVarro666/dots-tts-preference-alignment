# 02 · 关键决策记录（ADR：背景 / 选择 / 理由 / 被否方案 / 影响）

> 把每个重要设计抉择固化下来，方便日后回看「当初为什么这么定」。

## ADR-1 用 flow-matching 损失当 DPO 的隐式奖励（而非自训 reward model）
- **背景**：连续-AR 无 token log-prob，标准 DPO 不可用。
- **选择**：Diffusion-DPO 思路——用 velocity-MSE 当 −log p 代理。
- **理由**：直接复用 dots.tts 既有 `_compute_fm_loss_term`，零额外网络；理论有据。
- **被否**：①自训显式 reward model 再 PPO（实现重、易不稳、reward hacking 更难控）；②只做 SFT（学不到「相对偏好」）。
- **影响**：方法轻、可复现；命门转移到「共享 (x0,t)」。

## ADR-2 四前向共享 (x0, t)（最高优先级正确性约束）
- **背景**：L_FM 依赖随机 (x0,t)，不共享则 policy/ref、win/lose 不可比。
- **选择**：`fork_rng` + 固定 RNG state 包住 FM 采样，四前向复用同一份。
- **理由**：Diffusion-DPO 同样共享时间步；不共享 → 似然比失效 → 训不出偏好。
- **被否**：各前向独立采样（错误，会引入方差淹没信号）。
- **影响**：实现需小心；列为 04-RISKS 风险②、代码注释高亮。

## ADR-3 多目标 z-norm 奖励（WER+SIM+UTMOS），并**故意保留**仅-WER 臂
- **背景**：只压 WER 易牺牲自然度/音色（Seed-TTS 口述、05 §3.4）。
- **选择**：默认多目标(0.4/0.3/0.3)，按候选池 z-norm；另设 `dpo_weronly` 对照臂。
- **理由**：多目标回答 RQ1；仅-WER 臂**主动复现 reward hacking**（RQ3），是项目最有冲击力的负面结果。
- **被否**：只报多目标好结果（缺少反例就缺说服力）。
- **影响**：奖励权重成为核心操纵变量；结果三表之一专门给 reward hacking。

## ADR-4 候选多样性靠超参网格 + 逐候选 seed（因 generate 无 seed 参数）
- **背景**：`DotsTtsRuntime.generate()` 无 seed（已读源码确认）。
- **选择**：`num_steps×guidance×speaker` 网格 + 每候选 `seed_everything`。
- **理由**：在 API 约束下务实制造可控差异。
- **被否**：改 dots.tts 源码加 seed（增加耦合与维护成本）。
- **影响**：需在 reward 阶段抽查奖励方差确认多样性足够（风险①）。

## ADR-5 流式设为一等评测轴；流式克隆仅 x-vector
- **背景**：用户重视流式；dots.tts 1T1A 流式**拒 prompt_text**（源码 :117）。
- **选择**：离线+流式双模评测；流式用 prompt-audio-only 条件。
- **理由**：要 claim「增益迁移到流式」须走真实流式路径；x-vector-only 是 dots.tts 的硬约束。
- **被否**：拿离线结果替代流式（解码 schedule 不同，不成立）。
- **影响**：流式/离线 SIM 非严格可比，报告时分开标注。

## ADR-6 full-param(H100) vs LoRA(A6000) 双轨；LoRA 时 ref=disable_adapter
- **背景**：2×A6000(48GB) 全参 DPO + 冻结 ref 偏紧。
- **选择**：默认全参(H100, ZeRO-2)；A6000 走 LoRA，ref=关掉适配器的同一模型。
- **理由**：LoRA 省掉第二份模型，显存大降；同时作 LoRA-vs-full 消融。
- **被否**：一律全参（48GB 档可能 OOM）。
- **影响**：成为可行性保障 + 一个消融维度。

## ADR-7 把对 dots.tts 内部的耦合收敛到单一 `dots_preprocess.py`
- **背景**：AudioVAE 编码、x-vector、token schedule 都依赖 dots.tts 内部。
- **选择**：所有耦合放进一个适配器；上游只认统一接口。
- **理由**：dots.tts 版本变动时只改一处；上游可独立测试。
- **影响**：`dots_preprocess.py` 的几处 TODO 是「方法落地」的接缝，需对照源码补全。

## ADR-8 A→B 分阶段、B 门控
- **背景**：用户明确「先做出 A，效果好再推进 B」。
- **选择**：A 完整脚手架先行；B 仅脚手架 + 门控判据，达标才执行。
- **理由**：避免在未验证对齐增益前就投入蒸馏/量化。
- **影响**：B 的核心问题（对齐能否在蒸馏后存活）作为 future-work 级开放问题。

## ADR-9 DPO anchor 锚 winner 的 FM 声学损失（非 ce/eos）；并要求 pair 具备足够质量差
- **背景**：连续-latent DPO 只优化 winner/loser 的**相对量**（reward_margin），不约束绝对质量；模型最廉价的拉大 margin 方式是「搞烂 loser」，会连带把 winner 的 FM 声学损失一起放飞 → 全局 over-optimization（run0：reward_margin 93–170、winner `fm_w` 从 ~1 爆到 11–86、`reward_acc=1.0` 是欺骗性的）。雪上加霜的是，TTS 设定下文本是条件，名义 SFT anchor 里的 `ce≈0`、`eos≈0.05` → 只锚 ce+eos 的 anchor **形同虚设**，压根没在约束生成质量。
- **选择**：① anchor 改锚 winner 的 **FM 声学损失** `fm_pol_w.mean()`（再叠加 `ce_eos`），让 SFT 项真正锚在 TTS 里非零、决定音质的那一项；② 超参定为 **β=0.5、λ_sft=0.2、max_train_steps=600**（三份 config `dpo_multiobj_fullparam.yaml` / `dpo_weronly.yaml` / `open_soar_selfcorrect.yaml` 已统一为此值）；③ 把「pair 必须有足够奖励差」确立为**数据侧硬约束**，不再指望 loss/β 单独解决。
- **理由**：调参时间线暴露了一个**窄稳定窗口**——anchor 太弱（λ_sft=0.1）→ 退化式全局过拟合（靠搞烂 loser 廉价拉 margin，FM 爆、生成崩）；anchor 太强（λ_sft=0.5）→ 偏好信号被压死（按住 `fm_w` 会连带按住 `fm_l`，因同 prompt 候选声学相似、近似平局，margin→0，DPO 退化成「在 winner 上做 SFT」，dpo loss 卡在 ln2=0.693）。即便取平衡中点（λ_sft=0.2、β=0.3–0.5），可达 margin 仍受**pair 质量差上限**制约：run3（β=0.5）出现**间歇性分离**——多数 step 仍平（margin~0.01），但真有质量差的 pair 会炸开真 margin（step593：margin=0.778、`fm_l`=1.45 vs `fm_w`=0.75；step598：margin=0.478、`fm_l`=1.12 vs `fm_w`=0.62），且 `fm_w` 守在 ~1、anchor 没破。这证明平均 margin 低不是超参问题，而是 188 个 pair 里大多数是近似平局。配套经验：盯**绝对**质量量 `fm_w`，别只看相对 `reward_acc`/`margin`（`reward_acc=1.0` 曾掩盖一次彻底崩坏）。
- **被否**：① 只锚 ce/eos（B-15 已证无效：TTS 下 ce/eos≈0，anchor 等于不存在）；② 纯靠调 λ_sft/β 救场（已证只能在「FM 爆 over-optimization」↔「margin→0 欠拟合」之间摆动，治标不治本——可达 margin 的天花板在数据，不在超参）。
- **影响**：indexing 到 `docs/05` 的 **B-15**（anchor 修复、过拟合诊断）与 **B-16**（窄窗口/数据质量差瓶颈）。FM-loss 作隐式奖励只能分开「真不一样」的 pair → **pair 构造（margin 阈值、更大候选池）与 loss/β 同等重要**。**当前状态（如实标注）**：β=0.5 保留；下一步是统计 pair 的奖励差分布，若多为近似平局则用 margin 阈值重造 pair——此步**尚未执行、仍在诊断中，不得记为「已修复」**。

## ADR-10 DPO-FM 诊断重定位为 preference-likelihood 对齐失败（似然代理本身有效）；intrinsic 确立（偏相关存活，partial −0.392）、头条 Spearman；主线转警示性研究，保留 sft/open_soar 臂
- **背景**：ADR-1 用 `−L_FM` 当 DPO 的隐式奖励（Diffusion-DPO 思路），其前提是「`L_FM` 跟 automatic-metric reward 同向」。决定性探针 `alignment/probe_fm_reward.py`（全量 **N=188** pair × K=16 MC，win/lose 共享同一 (x0,t)、no_grad）实测：去噪 FM margin（=`L_FM^lose−L_FM^win`）与 reward 差**反相关**——稳健量 **Spearman ρ=−0.187（p≈0.01、ρ²≈3.5%、按 Cohen 属弱单调）**为头条，**Pearson −0.411（p≈1e-8）并列为「杠杆放大的线性系数」**（N=188 临界 |r|≈0.143、两者均过线）；同时 FM 对随机 winner/loser 平均无系统偏好（**82/188 (44%)** 为正、均值 **−0.0071**、win_rate **0.47**，binomial p=0.093、略低于 0.5 但不显著、大半 near-tie 产物）。这把 ADR-9（B-15/B-16）那条「调不动」的调参史从「超参/数据」上推到**方法层**：FM 既对 winner 无平均偏好（→ margin→0 欠拟合）、又与 reward 反相关（→ 指错方向）。
- **诊断重定位（C6，最重要）**：**不再说「`L_FM` 错配 / intrinsic to `L_FM`」**。`L_FM` 是**有效的似然代理**——`open_soar` 用同一 `L_FM`、似然对齐的偏好训得满分即证据。真正失败的是 **automatic-metric reward 导出的偏好（在 guidance 多样候选池上 best-vs-worst）与模型自身 flow 似然反向**——教科书级 **likelihood-vs-quality / classifier-free-guidance off-manifold** gap（cf. Theis et al. 2016；Ho & Salimans 2022；Gao et al. 2023 reward over-optimization）。两层须分清：**likelihood-surrogate validity（完好）** vs **preference-likelihood alignment（真失败）**。一句话：likelihood-based DPO 在 reward 偏好呈 off-manifold 属性时是**错的工具**；正路是**似然对齐的偏好**（open_soar）或**非似然目标**（winners 上 reward-weighted regression / 经可微 vocoder 直接 reward 反传 / 学习式 value 头）。novelty 重定位为「已知 likelihood-quality gap 在连续-latent flow-matching TTS 上的**领域量化** + 该架构**首个（among the first, to our knowledge）** DPO 实例 + open-SOAR」，**删「首次量化 / 中心发现」式措辞**。
- **guidance-confound 已由 `probe_analysis.py` 裁决：intrinsic 确立（偏相关存活，partial −0.392）**：旧元数据探针 `candidates/inspect_guidance_confound.py`（纯元数据 join pairs↔candidates_index，188 对、0 join 失败）测得 winner/loser 三组生成设置统计上无系统差异、与 reward 差仅弱相关——guidance_scale：winner **1.114** vs loser **1.093**（差 +0.021；winner 更高仅 39%、更低 29%、相等 32%；Pearson(reward差,guidance差)=**+0.108**）；num_steps：**11.85** vs **10.44**（差 +1.42；winner 更高 51%；Pearson **+0.118**）；speaker_scale：**1.371** vs **1.339**（差 +0.032；Pearson **+0.108**），三组 **|Pearson|≤0.12**。**但第一轮审稿（`docs/07` C1/C5/DA）指出此处测错了量**（confound 测的是 corr(reward-gap, setting-gap)，**该测的是 corr(FM-margin, setting-gap)**），故旧测属接受零假设的 under-powered（三相关全为预测方向、CI 上界达 +0.26）。`alignment/probe_analysis.py` 已换对量并跑出：**(1) confound（FM-margin×setting-gap）**——guidance r=−0.067(ns)、**num_steps r=−0.237（显著、CI[−0.360,−0.102]）**、speaker r=−0.067(ns)，即 **num_steps 是真实但温和的混淆**、guidance/speaker 无关；**(2) 决定性偏相关** corr(FM-margin, reward-gap | guidance,num_steps,speaker) = **partial r=−0.392，95%CI[−0.548,−0.209]，显著**——原始 Pearson −0.411 控住全部生成 knob（含 num_steps）后 → −0.392（几乎不变），**反相关存活 partialling、不被 num_steps 解释**。**裁决：intrinsic 确立**（反相关不是生成 knob 的产物）。**严守 C6**：此 intrinsic ≠「intrinsic to `L_FM`」（`L_FM` 仍有效、open-SOAR 为证），诊断仍是 preference-likelihood 对齐失败 / off-manifold winner。
- **统计严谨性（全量 N=188，已收紧 + 诚实修正）**：N=188 临界 |r|≈0.143，**Pearson −0.411（p≈1e-8）与 Spearman −0.187（p≈0.01）双双显著**（n=24 时 Spearman 不显著的弱环已消除）。单次 margin **std=0.0892 ≫ |mean| 0.0071**，印证单步信号高噪、须 MC 去噪后才看清「平均无偏好」（binomial 82/188 vs 0.5 **p=0.093**，略低于 0.5 但**不显著**，故此项大半是 near-tie 数据产物、不作独立 intrinsic 证据）。**修正**：早期 n=24 pilot 表观更强（4/24 为正、win_rate 0.33、Pearson −0.496、Spearman −0.302），其「winner **系统性** FM 更差」是小样本 / 高杠杆点的**高估**——全量回归到 win_rate **0.47**、44% 为正。故头条改用稳健的 **Spearman ρ=−0.187（ρ²≈3.5%、弱单调）**、**Pearson −0.411 并列标注为「杠杆放大的线性系数」**；平均效应（winner 系统性更差）不再作主结论，n=24 仅留作 pilot。
- **选择**：① **暂缓盲目放大**（不再加训练步数 / 不再造更多同构 pair / 不强推「DPO-FM 是可用配方」的叙事）；② **诊断重定位**——从「找一个能用的 DPO-FM 配方」转向**机理性/警示性研究**，标签从「`L_FM` 错配」改为「**preference-likelihood 对齐失败**」（似然代理本身有效）；③ **intrinsic 确立**——`probe_analysis.py` 换对量后，偏相关 corr(FM-margin, reward-gap | guidance,num_steps,speaker)=**partial −0.392（CI[−0.548,−0.209]、显著）**存活，反相关不被任一生成设置（含温和混淆 num_steps r=−0.237）解释（**C6：非 intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold**）；④ **头条 Spearman ρ=−0.187**（Pearson −0.411 并列为杠杆放大的线性系数）；⑤ **保留 `sft` 与 `open_soar` 两臂**作为不依赖「`L_FM` 当 logprob 且偏好须顺似然」的对照/建设性臂；⑥ **下一步** = 跑 `probe_analysis.py`（偏相关裁决 intrinsic）+ 评测两条建设性臂的 held-out 增益。
- **理由**：`sft`（仅在 winners 上监督 FM，不做 win/lose 的 `L_FM` 对比，只把模型拉向高质量音频）与 `open_soar`（reward-free、偏好建立在「轨迹洁净度」而非外部 reward、且与似然同向）**结构上不踩这个对齐坑**，即使 DPO-FM 被证伪它们仍是有效臂——是当前最稳妥的保底路线。诊断标签定为「preference-likelihood 失败 + intrinsic 确立（偏相关存活）」，是因为审稿正确指出：似然代理（`L_FM`）有效性与偏好-似然对齐是**两层**，open_soar 训得满分恰证前者完好——故 intrinsic 仅指「反相关不是生成 grid 的产物」（由换对量后的偏相关裁决），**不是**把反相关归因到 `L_FM` 本身；后者证据不足，仍按 preference-likelihood 对齐失败处理。
- **被否方案**：① 直接判 DPO-FM 失败并删臂——仍**过早**，DPO-FM 的负面结果本身是论文 §6/§7 的核心机理证据，应留作对照；② 继续加大 β/步数/数据规模硬刷 margin——已被 B-15/B-16 证明只在「过拟合 ↔ 欠拟合」间摆动，偏好方向不顺似然时放大只会更糟；③ 把 n=24 pilot 的单点量级（−0.496/−0.302、win_rate 0.33、4/24）当全量定论——全量 N=188 已证 win_rate 回归 **0.47**、44% 为正，头条以稳健 **Spearman −0.187** 为准；④ **据旧（测错量的）confound 断言 intrinsic /「guidance-confound 已排除」**——被审稿否决：旧 confound 测错了量、属接受零假设；intrinsic 须由**换对量后的偏相关**裁决（现已存活：partial −0.392），不得据 under-powered 的旧测下结论。
- **影响**：indexing 到 `docs/05` **B-17** 与 **`docs/06-核心发现-FM损失与奖励错配.md`**（完整数据/机理/逐对佐证）。贡献定位：从「一个能用的 DPO 配方」转为「**机理性警示**——在 **dots.tts** 上把 likelihood-based DPO/Diffusion-DPO 直接搬到连续-latent flow-matching TTS，当 automatic-metric reward 偏好呈 off-manifold 属性时是错配的；**可能架构特定（待 VoxCPM 等二次复现）**，故收窄到 dots.tts、不外推整类」。建设性候选方向（非 FM-loss-DPO）：winners 上的 reward-weighted regression、经可微 vocoder 的直接 reward 反传、学习式 value/preference 头取代「`L_FM` 当 logprob」。**当前状态（如实标注）**：诊断重定位为 **preference-likelihood 对齐失败**（似然代理 `L_FM` 本身有效，open_soar 为证）；**intrinsic 确立**（`probe_analysis.py` 偏相关 corr(FM-margin, reward-gap | guidance,num_steps,speaker)=**partial −0.392、CI[−0.548,−0.209]、显著**存活；num_steps r=−0.237 温和混淆但被偏相关控住；**C6：非 intrinsic-to-`L_FM`，诊断仍 preference-likelihood/off-manifold**）；头条稳健 **Spearman −0.187（ρ²≈3.5%）**、Pearson −0.411 为杠杆放大的线性系数。早期 n=24 pilot 的平均效应（win_rate 0.33、4/24 为正、Pearson −0.496）系小样本**高估**、仅留作 pilot。结论为**过程性重定位**：intrinsic 指反相关非生成-grid 产物（已裁决），但**不等于**断言 `L_FM` 本身内在错配——诊断仍是 preference-likelihood 对齐失败。
