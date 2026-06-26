# 01 · 方法：连续-latent flow-matching AR 上的 DPO（数学与依据）

## 1. 背景：为什么不能直接套标准 DPO

标准 DPO 目标：
```
L_DPO = − E_(x,y_w,y_l) [ log σ( β·( log (π_θ(y_w|x)/π_ref(y_w|x)) − log (π_θ(y_l|x)/π_ref(y_l|x)) ) ) ]
```
它要求能算序列 `y` 的 **log-likelihood** `log π(y|x)`。

dots.tts 是**连续-latent AR**：LLM 出隐状态，DiT flow-matching head 在**连续** AudioVAE latent 上生成——**没有离散 audio token，没有现成的 token log-prob**。所以标准 DPO 不能直接用。这正是 `../../学习资料/05` §3.13 指出的「连续 AR 上的 DPO 偏好对构造空白」。

## 2. 代理：用 flow-matching 损失当 −log p

借鉴 **Diffusion-DPO（Wallace et al., 2023, *Diffusion Model Alignment Using DPO*）**：扩散/流匹配模型的**训练损失是 −log p 的代理**（注：对 CFM 这是**启发式**，非已证变分界——CFM 目标是向量场 MSE 的无偏估计，cf. Lipman 2023；见下文实证警示）。对 flow-matching，dots.tts 训练用的 velocity-MSE 即：
```
L_FM(model, x) = E_{t, x0} || v_θ(x_t, t | g_cond) − u_t ||²
其中 x_t = (1−t)·x0 + t·x1,   u_t = x1 − (1−σ)·x0   (dots.tts core.py compute_u_t, σ=0)
        x1 = AudioVAE.encode(目标音频),   x0 ~ N(0,I) 噪声
```
`L_FM` 越小 → 该样本在模型下「越可能」。于是用 `−L_FM` 当 `log p` 的代理。

> **⚠️ 实证警示（必读）**：本方法的隐含前提是「automatic-metric reward 偏好跟模型 flow 似然同向——被偏好候选 `L_FM` 更低」。决定性探针 `alignment/probe_fm_reward.py`（全量 **N=188** pair × K=16 MC）**实测推翻了这个前提**，两条依据：(1) FM 对随机 winner/loser **平均无系统偏好**（82/188=44% 的 winner FM 更优、win_rate 0.47、mean margin −0.0071、std 0.0892；binomial p=0.093，略低于 0.5 但**不显著**，大半 near-tie 产物、不作独立证据）→ 对应训练里 `reward_margin→0` 的欠拟合；(2) 去噪后 FM margin 与 reward 差**反相关**——稳健量 **Spearman −0.187（p≈0.01、ρ²≈3.5%、弱单调）**为头条，**Pearson −0.411（p≈1e-8）并列为「杠杆放大的线性系数」**（N=188 临界 |r|≈0.143、双显著），即在质量差最大的 pair 上 `L_FM` 反而最不偏好 winner（reward 越高、似然越低）。**关键区分（C6）**：失败的不是 `L_FM` 作为似然代理本身（它**有效**——open-SOAR 用同一 `L_FM`、似然对齐的偏好训得满分），而是 **preference-likelihood 对齐**（automatic-metric reward 偏好与 flow 似然反向）——教科书级 **likelihood-vs-quality / classifier-free-guidance off-manifold** gap（cf. Theis et al. 2016；Ho & Salimans 2022；Gao et al. 2023）。故「`−L_FM` 当 `log p`」用 likelihood-based DPO 在 off-manifold reward 偏好下是**错的工具**（见 `05` B-17、`06`）。
> **统计显著性（全量 N=188，已收紧 + 诚实修正）**：N=188 临界 |r|≈0.143，**Pearson −0.411（p≈1e-8）与 Spearman −0.187（p≈0.01）双双显著**（n=24 时 Spearman 不显著的弱环已消除）。**修正**：早期 n=24 pilot 表观更强（win_rate 0.33、4/24 为正、Pearson −0.496、Spearman −0.302），其「winner **平均**更差」是小样本 / 高杠杆点的**高估**——全量回归到 win_rate 0.47、44% 为正；故平均效应不再作主张。主结论改以**稳健的 Spearman ρ=−0.187（ρ²≈3.5%、弱单调）**为头条，**Pearson −0.411 并列标注为「杠杆放大的线性系数」**，n=24 仅留作 pilot。**完整数据/机理/影响见 `docs/06-核心发现-FM损失与奖励错配.md`**。
> **guidance-confound 已由 `probe_analysis.py` 裁决：intrinsic 确立（偏相关存活，partial −0.392）**：曾担心反相关是**候选生成 guidance 混淆**（winner 系统性高 guidance）——元数据探针 `candidates/inspect_guidance_confound.py`（纯元数据 join pairs↔candidates_index，188 对、0 join 失败）测得 winner/loser 生成设置统计上无系统差异——guidance_scale 1.114 vs 1.093（差 +0.021，winner 更高仅 39%、更低 29%、相等 32%）、num_steps 11.85 vs 10.44（差 +1.42，winner 更高 51%）、speaker_scale 1.371 vs 1.339（差 +0.032），各设置与 reward 差仅**弱相关 |Pearson|≤0.12**（分别 +0.108 / +0.118 / +0.108）。**但第一轮审稿（`docs/07` C1/C5）指出此处测错了量**（confound 测的是 corr(reward-gap, setting-gap)，**该测的是 corr(FM-margin, setting-gap)**），旧测属接受零假设的 under-powered。`alignment/probe_analysis.py` 换对量后实测：confound（FM-margin×setting-gap）guidance r=−0.067(ns)、**num_steps r=−0.237（显著、CI[−0.360,−0.102]）**、speaker r=−0.067(ns)（**num_steps 温和混淆**、guidance/speaker 无关）；决定性**偏相关** corr(FM-margin, reward-gap | guidance,num_steps,speaker)=**partial −0.392（95%CI[−0.548,−0.209]、显著）**——原始 Pearson −0.411 控住全部生成 knob 后 → −0.392（几乎不变），**反相关存活 partialling、不被 num_steps 解释**。**裁决：intrinsic 确立**（反相关非生成-grid 产物）；**C6**：此 intrinsic ≠「intrinsic to `L_FM`」（`L_FM` 仍有效、open-SOAR 为证），诊断仍 preference-likelihood 对齐失败 / off-manifold winner。注：§5 的 **open-SOAR 与 anchor SFT 均不把 `L_FM` 当 logprob**（SFT 是 winners 上的 FM/CE+EOS、open-SOAR 是 reward-free 自校正、与似然同向），故**不踩此 preference-likelihood 对齐坑**（open-SOAR 的「学满分」是 by construction，真证据是 held-out 音质 Table 1）。

## 3. DPO over flow-matching（本项目的目标）

定义策略相对参考对某样本的「似然提升」：
```
Δ(x) = L_FM(ref, x) − L_FM(policy, x)      # 策略把损失降得越多 ⇒ 似然提升越大
```
偏好对 (x^w, x^l) 共享同一 (text, prompt) 条件，目标：
```
L = − log σ( β·( Δ(x^w) − Δ(x^l) ) )       # 推高赢样本、压低输样本的相对似然
```
展开成可实现的 velocity-MSE 形式（`alignment/losses_dpo.py` 即此式）：
```
inner = −β · [ (||v_θ(x^w)−u^w||² − ||v_ref(x^w)−u^w||²)
             − (||v_θ(x^l)−u^l||² − ||v_ref(x^l)−u^l||²) ]
L = − log σ(inner)
```
（每项为 per-sample、按音频 span patch mask 平均，复刻 dots.tts `_compute_fm_loss_term` 的数学。）

总损失加一个 **anchor SFT** 正则（winners 上的 CE+EOS），防文本条件/停止预测漂移：
```
L_total = L_DPO + λ_sft · (CE + EOS)_on_winners
```

## 4. 正确性命门：共享 (x0, t)

`L_FM` 依赖随机抽样的噪声 `x0` 与时间步 `t`。要让 `L_FM(ref,·)` 与 `L_FM(policy,·)`、以及 `x^w` 与 `x^l` 之间**可比**，四次前向必须**复用同一 (x0, t)**（Diffusion-DPO 也共享时间步）。实现：在 `train_dots_tts_dpo.py` 用 `torch.random.fork_rng()` + 固定 RNG state 包住 FM 采样。**这是方法成立的前提**（见 04-RISKS 风险②）。

## 5. open-SOAR：reward-free 自校正（我们的再现，非原 SOAR）

dots.tts 的 SOAR 训练程序未开源（仓库只有 soar 权重；`../../参考代码/核心代码分析.md` 证实 reward-free、只作用于 flow-matching 头）。我们的再现：
- winner = 干净目标轨迹（标准 `u_t`）；
- loser = 从**扰动/高噪起点**的 rollout 速度场（`soar_noise_scale`），但 **target 仍是干净 `u_t`**；
- 用同一 DPO 形式让策略偏向「无论起点多偏都指回干净目标」= 自校正。

它**不用任何外部奖励**，因此与多目标 DPO 对照即可分离「自校正」与「外部偏好」的贡献（RQ4）。
> 明确标注：这是**开源再现/近似**，非 Boson 原始 SOAR；命名 open-SOAR 以示区别。

## 6. 与离散派 DPO 的关系
离散 codec-LM（VALL-E 等）可直接用 token log-prob 做 DPO；连续派一直缺这一步。本方法把离散派成熟的偏好对齐工具链**迁移到连续-latent AR**，正是 `05` §6.2 点名的方向。

## 参考
- Wallace et al., *Diffusion Model Alignment Using Direct Preference Optimization*, 2023（Diffusion-DPO）。
- Rafailov et al., *DPO: Your Language Model is Secretly a Reward Model*, 2023（DPO 原型）。
- dots.tts：`../../05-dots.tts-技术详解.md`、`../../参考代码/dots.tts/`（FM 损失/采样的确切实现）。
- 立项依据：`../../学习资料/05-前沿方向与研究机会.md` §3.4 / §3.13 / §6.2。
> 引文如需进论文，按仓库规范逐条核验 arXiv id 再写。
