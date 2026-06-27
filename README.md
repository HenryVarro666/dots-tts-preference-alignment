# dots.tts 连续-latent 偏好对齐 · 机理性警示研究（open-SOAR / DPO 诊断）

> **论文标题**：*When Likelihood Disagrees with Quality: A Cautionary Diagnosis of DPO for Continuous-Latent Flow-Matching TTS (dots.tts)*
>
> 一个**机理性警示研究（cautionary diagnosis）**：在 [dots.tts](../../参考代码/dots.tts/)（2B 全连续潜在 TTS，Apache-2.0）上系统证明——**向 automatic-reward 选出的自生成 off-manifold winner 做 self-training 会退化模型，且与训练目标（SFT 还是 DPO）无关**，只随「往 off-manifold 拉的强度」单调加剧；**唯独 open_soar（训向真实 on-manifold 目标）改善**。机理是 **flow-matching 似然与三个 perceptual reward 轴内在反相关**。
>
> 立项动机见上层研究议程 [`../../学习资料/05-前沿方向与研究机会.md`](../../学习资料/05-前沿方向与研究机会.md) §6.2 / §3.13 / §3.4。**注意：项目已从最初设想的「开源可用 DPO 配方 + A→B 蒸馏」重定位为下文的警示性诊断——下面的核心发现就是这次重定位的实证依据。**

## 一句话定位（这个项目最终在说什么）

> 在连续-latent flow-matching TTS 上，**用自动奖励驱动的自生成偏好数据做对齐/自训练，会把模型拉向 off-manifold 而退化**——这是一份**可证伪的负面/警示结果**，连带给出反相关的机理诊断。它**不是**一份「拿去就能用的 DPO 配方」，而是「**为什么这条直觉路线会崩、崩在哪、什么才有效**」。

## 核心发现（已实测，n=100 离线主表，bootstrap 95% CI）

**Table 1（offline，100 句 LJSpeech held-out，nfe=10，skip-streaming；WER↓ via Whisper-large-v3 / SIM↑ via ecapa / UTMOS↑ via utmos22_strong）**

| 臂 | WER ↓ | SIM ↑ | UTMOS ↑ | n | 裁决 |
|---|---|---|---|---|---|
| **base** | 0.103 | 0.891 | 4.309 | 100 | 基线 |
| **open_soar** | **0.093** | 0.844 | **4.408** | 100 | **唯一改善**（UTMOS 五臂最佳、WER≈base、SIM 略降） |
| **dpo_weronly** | 0.115 | 0.852 | 4.112 | 100 | ≈base（WER-only 未取得最低 WER） |
| **sft** | 0.545 | 0.673 | 3.917 | 100 | **崩坏**（推翻审稿钦点的「安全对照」） |
| **dpo_multiobj** | 0.752 | 0.365 | 2.841 | **75** | **灾难崩坏 + 25% 生成失败** |

> `dpo_multiobj` 的 **n=75**：100 句中 **25/100=25% 生成失败**（全部 immediate-EOS/no-payload、聚集在 `LJ002` 块）；主表 WER 0.752 **只统计成功的 75 条**，**失败率（25%）必须与 WER 并列**，否则低估崩坏。

**两条主结论**：

1. **off-manifold self-training 退化，与训练目标无关。** 把四个对齐臂按「往 automatic-reward 选出的自生成 winner（off-manifold）拉的强度」排序，崩坏**单调加剧**：`open_soar`（训向真实 clean 音频=on-manifold，**改善**）→ `dpo_weronly`（弱拉，≈base）→ `sft`（强拉、纯监督，**崩**）→ `dpo_multiobj`（最强拉 + 反向梯度，**灾难 + 25% 生成失败**）。连「不做 win/lose `L_FM` 对比」的纯 SFT 都崩 → 毒源是**目标分布本身（off-manifold winner）**，而非 DPO 的对比机制。
2. **FM 似然与 perceptual reward 内在反相关。** 头条 **Spearman ρ=−0.187**；**偏相关 −0.392**（控住 guidance/num_steps/speaker_scale 全部生成 knob 后存活，CI[−0.548,−0.209]）→ **intrinsic 确立**（非 grid 副产品）。换到 reward-对齐方向，三个 perceptual 轴**全负**：(−WER) **−0.41** / SIM **−0.68** / UTMOS **−0.60**（**SIM/UTMOS 比 WER 更强** → 不是「只修 reward 组成」能救，须整体弃 FM-as-reward）。

**机理定性（C6 严守）**：失败**不在** `L_FM` 本身——**`L_FM` 是有效似然代理**（`open_soar` 用同一损失训出五臂最佳 UTMOS 为证）；失败在 **preference-likelihood 对齐**——偏好对的 winner 是 off-manifold，把似然往那里拉就与质量背离。这正是已知的 **likelihood-vs-quality / CFG gap** 在 TTS 偏好对齐上的具体体现。

**RQ3 reward-hacking（诚实改写）**：经典 reward-hacking（WER-only 臂取最低 WER 却牺牲自然度）**在错配的 DPO-FM 框架内未复现**——`dpo_weronly` WER 0.115≈base、并非最低（最低是 open_soar 0.093），两个 DPO 臂都退化、reward-composition 效应被 FM 崩坏混淆。**改在非错配框架内（reward-weighted 回归，X7）干净复现**。

完整裁决与统计见 [`docs/08-实验结果-Table1.md`](docs/08-实验结果-Table1.md)、机理推导见 [`docs/06-核心发现-FM损失与奖励错配.md`](docs/06-核心发现-FM损失与奖励错配.md)、第一轮审稿见 [`docs/07-审稿意见-第一轮.md`](docs/07-审稿意见-第一轮.md)。

## 已知 caveat（写进结论，不藏）

- **`sft` 崩坏与「小数据 SFT 过拟合/灾难性遗忘」尚未干净分离**（188 单说话人）——待 **X4（SFT-target 梯度对照：winners/losers/random/groundtruth）** 隔离；`open_soar` 同规模微调不崩是**部分对照**，故「off-manifold 退化」目前是**强假设**。
- **`dpo_multiobj` 的 25% 是 immediate-EOS 的「生成时子集崩」**——崩坏量级超出弱单调相关（ρ=−0.187）所能预测，但**「graded 退化 vs optimization divergence」证据不一、暂不定论**（X2/X3 实测：训练 `fm_w` 稳 ~0.74、未二次爆、margin 仍小 0.065 → **训练未发散**；交叉表 34% vs 48% 偏 divergence；**合判 = 生成时子集崩**），保守报告、不径直坐实 divergence。
- **失败聚于 `LJ002` 块须诚实归因**（X2/X3 已诊断）：评测集本身 **89% 为 LJ002**（89/100），故聚集**部分因 LJ002 占评测主体**；组内 **LJ001 0/11（0%）vs LJ002 25/89（28%）**（小 n、p≈0.02）仍是真实 prompt 依赖、**非均匀**，失败样本仅**轻微偏长（1.12×）、无位置偏置**——**不写成「长 prompt 触发」**。
- **无人类听测（ABX/MOS）**——崩坏量级与 UTMOS 退化未由人耳锚定，未做前**不称** perceptual/naturalness 退化。

## 第二轮实验程序（脚手架已建，本机仅 py_compile，GPU 上跑，结果待出）

针对上述 caveat 与「负面结果配上『该怎么做』」，第二轮 X1–X13（P0 再分析 → P4 camera-ready）的**一条龙跑序与意义见 [`docs/09-第二轮实验程序.md`](docs/09-第二轮实验程序.md)**。关键臂已全部进 `ablation/run_matrix.py` 的 `TREATMENTS`（`sft_losers/sft_random/sft_groundtruth`、`dpo_onmanifold`、`reward_weighted{,_weronly}`），所有 JSONL 契约 schema 不变。一句话索引：

- **P0 再分析（CPU）**：X1 高-gap 子集探针扩展、X2 失败率入表、X2/X3 聚集/发散诊断（`eval/diagnose_eval.py`）。
- **P1 核心（~1–2 GPU日）**：**X4** SFT-target 梯度（隔离 off-manifold vs 小数据 SFT）、**X5** settings-matched 复探针（`--fixed-setting`）。
- **P2 建设性 + RQ2/3（~3–5 GPU日）**：**X6a** `dpo_onmanifold`（只用 on-manifold winner 救 DPO）、**X6b/X7** `reward_weighted{,_weronly}`（非错配框架内干净复现 reward-hacking）、**X8** 流式 Table 2、**X9** EmergentTTS。
- **P3 泛化（~1–2 GPU周）**：**X10** VoxCPM 二模型探针（升架构级）、**X11** VCTK/AISHELL-3 多语种。
- **P4**：X12 人类 ABX/MOS（暂不可行→备协议 + 记 limitation）、X13 camera-ready。

## A→B 蒸馏的现状（降为远期 future work）

最初设想的 **Phase B（少步蒸馏 NFE 1~2 + INT8/INT4 量化塞进 8GB Turing，对接 `../../04` 部署指南）已不是主线、未执行**——它以「Phase A 出一个值得部署的对齐配方」为前提，而 Phase A 的裁决是**警示性负面结果**（除 open_soar 外均退化）。因此 Phase B 降为**远期 future work**（开放问题：「偏好对齐/自校正能否在少步蒸馏+量化后存活」），脚手架仍在 `phaseB/` 仅作占位，不在当前论文范围内。

## 目录速览

| 目录 | 作用 |
|---|---|
| `env/` | 一键装环境（dots.tts + seed-tts-eval + speechmos + WavLM-SV ckpt） |
| `data/` | 从公开集（VCTK/AISHELL-3/LJSpeech）构造 prompt 清单；多语种见 `data/run_multilingual.md` |
| `candidates/` | 用 dots.tts 对每个 prompt 采 K 个候选（X5：`--fixed-setting` settings-matched 复探针） |
| `reward/` | 多目标打分（WER + SIM + UTMOS）→ z-norm 聚合奖励 |
| `pairs/` | 构造偏好对（win/lose）；`extract_{winners,losers,random}.py`（X4 SFT-target 对照）、`filter_pairs_by_likelihood.py`（X6a on-manifold 过滤） |
| `alignment/` | **核心**：DPO-on-flow-matching 损失 + trainer + open-SOAR + `probe_analysis.py`（intrinsic 偏相关裁决）+ `train_reward_weighted.py`（X6b/X7）+ `probe_fm_reward_voxcpm.py`（X10） |
| `inference/` | 离线 & 1T1A 流式批量合成 |
| `eval/` | Seed-TTS-eval(WER/SIM) + UTMOS + EmergentTTS + 流式指标 + `diagnose_eval.py`（X2/X3 崩坏归因） |
| `ablation/` `results/` | 跑实验矩阵（`run_matrix.py`，TREATMENTS 含全部新臂）、汇总出表/图（失败率入主表） |
| `phaseB/` | **远期 future work** 占位（蒸馏+量化，未执行） |
| `docs/` | **全程文档化**：00 总览 / 01 方法 / 02 决策 / 03 复现 / 04 风险 / 05 踩坑 / 06 核心发现 / 07 审稿 / 08 Table 1 / **09 第二轮程序** |

## 怎么读这个项目

1. 先读 [`docs/00-总览与步骤详解.md`](docs/00-总览与步骤详解.md)——理解「为什么管线长这样」。
2. **看结论**：核心机理发现 [`docs/06-核心发现-FM损失与奖励错配.md`](docs/06-核心发现-FM损失与奖励错配.md) → 实测裁决 [`docs/08-实验结果-Table1.md`](docs/08-实验结果-Table1.md) → 第一轮审稿 [`docs/07-审稿意见-第一轮.md`](docs/07-审稿意见-第一轮.md)。
3. 方法细节看 [`docs/01-METHOD.md`](docs/01-METHOD.md)；关键设计抉择看 [`docs/02-DECISIONS.md`](docs/02-DECISIONS.md)。
4. 真要跑：第一轮按 [`docs/03-REPRODUCE.md`](docs/03-REPRODUCE.md)、第二轮按 [`docs/09-第二轮实验程序.md`](docs/09-第二轮实验程序.md)，边跑边填 [`docs/RUN_JOURNAL_TEMPLATE.md`](docs/RUN_JOURNAL_TEMPLATE.md)。
5. **遇到报错/warning**：先查 [`docs/05-踩坑与修复记录.md`](docs/05-踩坑与修复记录.md)——已记录所有遇到过的错误、warning、改进（现象/根因/修复/为什么/commit）。

## 🚀 远程服务器快速开始（独立部署）

本仓库可**独立 clone 运行**（不依赖上层 TTS 知识库）。`env/setup_env.sh` 会自动把
dots.tts 与 seed-tts-eval 克隆到本项目 `.deps/`（代码 only，无 LFS 权重）；`common.py`
会自动定位它们（也可用 `DOTS_TTS_REPO` / `SEED_TTS_EVAL` 环境变量覆盖）。

```bash
git clone <this-repo-url> dots-tts-preference-alignment
cd dots-tts-preference-alignment
bash env/setup_env.sh          # 克隆依赖 + 建 venv + 装包 + 提示放置 WavLM ckpt
source .venv/bin/activate
# 还需: 下载 dots.tts-base 权重(HF) + WavLM-SV ckpt(见 setup 提示) + 可选 clone EmergentTTS-Eval
# 然后按 docs/03-REPRODUCE.md（第一轮）→ docs/09-第二轮实验程序.md（第二轮）端到端跑（2×H100 / 2×A6000）
```

> 文档里形如 `../../学习资料/05` 的链接指向作者的 TTS 知识库（立项背景），独立部署时这些链接不可达，不影响运行。

## ⚠️ 重要说明

- **本机（macOS / 无 GPU / dots.tts 权重未下载）只能语法检查**：`python3 -m py_compile`、`bash -n`。**绝不声称脚本已端到端跑通。** Table 1 等实测数字为 GPU 机器回填、逐字精确；第二轮（X1–X13）结果**待出**。
- 真正运行需要 **NVIDIA GPU**（设计目标 **2×H100 80GB** 或 **2×A6000 ≥48GB**）。
- 仅用**公开数据集**；新文件保持 **Apache-2.0**（与 dots.tts 一致，可商用）。
- 一切「谁超过谁」的数字都标注证据等级；厂商自评/自建集不与第三方混谈（沿用仓库规范）。社区估算/未官方确认项标 `〔需核实〕`。

## License

Apache-2.0（与 dots.tts 一致）。各公开数据集许可见 [`data/LICENSES.md`](data/LICENSES.md)。
