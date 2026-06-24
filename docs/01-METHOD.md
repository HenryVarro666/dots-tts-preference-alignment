# 01 · 方法：连续-latent flow-matching AR 上的 DPO（数学与依据）

## 1. 背景：为什么不能直接套标准 DPO

标准 DPO 目标：
```
L_DPO = − E_(x,y_w,y_l) [ log σ( β·( log (π_θ(y_w|x)/π_ref(y_w|x)) − log (π_θ(y_l|x)/π_ref(y_l|x)) ) ) ]
```
它要求能算序列 `y` 的 **log-likelihood** `log π(y|x)`。

dots.tts 是**连续-latent AR**：LLM 出隐状态，DiT flow-matching head 在**连续** AudioVAE latent 上生成——**没有离散 audio token，没有现成的 token log-prob**。所以标准 DPO 不能直接用。这正是 `../../学习资料/05` §3.13 指出的「连续 AR 上的 DPO 偏好对构造空白」。

## 2. 代理：用 flow-matching 损失当 −log p

借鉴 **Diffusion-DPO（Wallace et al., 2023, *Diffusion Model Alignment Using DPO*）**：扩散/流匹配模型的**训练损失本身就是 −log p 的(变分)代理**。对 flow-matching，dots.tts 训练用的 velocity-MSE 即：
```
L_FM(model, x) = E_{t, x0} || v_θ(x_t, t | g_cond) − u_t ||²
其中 x_t = (1−t)·x0 + t·x1,   u_t = x1 − (1−σ)·x0   (dots.tts core.py compute_u_t, σ=0)
        x1 = AudioVAE.encode(目标音频),   x0 ~ N(0,I) 噪声
```
`L_FM` 越小 → 该样本在模型下「越可能」。于是用 `−L_FM` 当 `log p` 的代理。

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
