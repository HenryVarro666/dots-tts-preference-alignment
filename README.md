# dots.tts 连续-latent 偏好对齐（open-SOAR / DPO）

> 一个**前沿探索性实验项目**：在 [dots.tts](../../参考代码/dots.tts/)（2B 全连续潜在 TTS，Apache-2.0）代码基础上，做一套**开源、可复现的连续-AR 偏好优化配方**，专攻 AR 鲁棒性且**不 reward-hack**，并把**流式（1T1A）质量**设为一等评测轴。
>
> 立项动机见上层研究议程 [`../../学习资料/05-前沿方向与研究机会.md`](../../学习资料/05-前沿方向与研究机会.md) §6.2 / §3.13 / §3.4。

## 为什么这个实验值得做（一句话）

- **连续-latent AR 上的 DPO/偏好对构造是公开空白**（05 文档 §3.13 钦点）。
- dots.tts 的 **SOAR 自校正训练程序未开源**（仓库只有 `dots.tts-soar` 权重）——我们做一份**开源再现**本身就是贡献。
- 直击 05 文档排第二的痛点：**AR 鲁棒性 + reward hacking**，并产出一个**可证伪的负面结果**。

## A→B 主线（分阶段）

```
Phase A  偏好对齐：base → SFT → open-SOAR → DPO(多目标) → DPO(仅WER, reward-hacking 对照)
            │  评测：离线 + 1T1A 流式 双模
            ▼  （A 出效果，门控通过后才进 B）
Phase B  少步蒸馏(NFE 1~2) + INT8/INT4 量化 → 塞进 8GB Turing（对接 ../../04 部署指南）
```

**当前阶段：先完整做 A。** Phase B 是门控的脚手架，达标后才执行（见 [`phaseB/PHASE_B_PROTOCOL.md`](phaseB/PHASE_B_PROTOCOL.md)）。

## 目录速览

| 目录 | 作用 |
|---|---|
| `env/` | 一键装环境（dots.tts + seed-tts-eval + speechmos + WavLM-SV ckpt） |
| `data/` | 从公开集（VCTK/AISHELL-3/LJSpeech）构造 prompt 清单 |
| `candidates/` | 用 dots.tts 对每个 prompt 采样 K 个候选 |
| `reward/` | 多目标打分（WER + SIM + UTMOS）→ z-norm 聚合奖励 |
| `pairs/` | 构造偏好对（win/lose） |
| `alignment/` | **核心**：DPO-on-flow-matching 损失 + trainer 子类 + open-SOAR |
| `inference/` | 离线 & 1T1A 流式批量合成 |
| `eval/` | Seed-TTS-eval(WER/SIM) + UTMOS + EmergentTTS + 流式指标 |
| `ablation/` `results/` | 跑实验矩阵、汇总出表/图 |
| `phaseB/` | 门控的蒸馏+量化脚手架 |
| `docs/` | **全程文档化**：每步细节与意义、决策记录(ADR)、复现指南、运行日志模板 |

## 怎么读这个项目

1. 先读 [`docs/00-总览与步骤详解.md`](docs/00-总览与步骤详解.md)——理解「为什么管线长这样」。
2. 方法细节看 [`docs/01-METHOD.md`](docs/01-METHOD.md)；关键设计抉择看 [`docs/02-DECISIONS.md`](docs/02-DECISIONS.md)。
3. 真要跑：按 [`docs/03-REPRODUCE.md`](docs/03-REPRODUCE.md) 在 GPU 机器上逐步执行，边跑边填 [`docs/RUN_JOURNAL_TEMPLATE.md`](docs/RUN_JOURNAL_TEMPLATE.md)。

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
# 然后按 docs/03-REPRODUCE.md 端到端跑（2×H100 / 2×A6000）
```

> 文档里形如 `../../学习资料/05` 的链接指向作者的 TTS 知识库（立项背景），独立部署时这些链接不可达，不影响运行。

## ⚠️ 重要说明

- **本机（macOS / 无 GPU / dots.tts 权重未下载）只能语法检查**：`python3 -m py_compile`、`bash -n`。**绝不声称脚本已端到端跑通。**
- 真正运行需要 **NVIDIA GPU**（设计目标 **2×H100 80GB** 或 **2×A6000 ≥48GB**）。
- 仅用**公开数据集**；新文件保持 **Apache-2.0**（与 dots.tts 一致，可商用）。
- 一切「谁超过谁」的数字都标注证据等级；厂商自评/自建集不与第三方混谈（沿用仓库规范）。

## License

Apache-2.0（与 dots.tts 一致）。各公开数据集许可见 [`data/LICENSES.md`](data/LICENSES.md)。
