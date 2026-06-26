# 03 · 复现指南（第一轮：用户在 GPU 机器上的端到端跑序）

> ⚠️ 本机(macOS/无 GPU/权重未下载)**不能跑**，只能 `py_compile`/`bash -n`。下列在 **2×H100 或 2×A6000** 上执行。
> 每条命令都标了「为什么这一步」。边跑边把实际命令/数字填进 [`RUN_JOURNAL_TEMPLATE.md`](RUN_JOURNAL_TEMPLATE.md) 的副本。
>
> **范围**：本篇是**第一轮**跑序（产出 Table 1 的 base + 4 对齐臂裁决，见 [`08-实验结果-Table1.md`](08-实验结果-Table1.md)）。**第二轮（X1–X13：隔离 off-manifold、建设性 on-manifold/reward-weighted、流式、跨架构/多语种、camera-ready）的一条龙 GPU 跑序见 [`09-第二轮实验程序.md`](09-第二轮实验程序.md)**（§7 给出指针）。

## 0. 环境
```bash
bash env/setup_env.sh                  # 为什么: 固化 dots.tts + seed-tts-eval + speechmos + WavLM ckpt
source .venv/bin/activate
python -c "import dots_tts, jiwer, peft; print('ok')"
# 配 accelerate（2×H100 建议 deepspeed zero_stage=2, bf16）
accelerate config
```

## 1. 数据（公开集）
```bash
# LJSpeech-48kHz：dots.tts 下载器一步下载+解包+产出 {fid,audio,text} manifest（audio 为绝对路径）
python .deps/dots.tts/scripts/prepare_train_jsonl_manifest.py --output-dir data/corpora/ljspeech48k --valid-size 0
#   ↑ 独立部署时 dots.tts 在 .deps/；在知识库内则是 参考代码/dots.tts/。--help 看全部参数。

# 直接读该 manifest（最稳），先取 200 条打通
python data/prepare_prompts.py --corpus dots_manifest \
       --source data/corpora/ljspeech48k/ljspeech_48khz_manifest_train.jsonl --lang en --limit 200
# 跑通后再上 VCTK / AISHELL-3 全量（--corpus vctk/aishell3 --source <语料目录>），见 data/fetch_public_corpora.sh
```

## 2. 候选 → 打分 → 偏好对
```bash
python candidates/generate_candidates.py --prompts data/prompts/ljspeech_train.jsonl --k 6   # 为什么: 制造有高低差的候选池
python reward/score_wer.py && python reward/score_sim.py && python reward/score_utmos.py       # 为什么: 多目标客观信号
python reward/aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3                        # 多目标
python reward/aggregate_reward.py --w-wer 1 --w-sim 0 --w-utmos 0 --out reward/out/rewards_weronly.jsonl  # reward-hacking 臂
python pairs/build_preference_pairs.py --strategy best_worst
python pairs/build_preference_pairs.py --rewards reward/out/rewards_weronly.jsonl --strategy best_worst   # 注意改输出名为 *_weronly
# ↑ 检查 aggregate 打印的「候选池奖励方差」，太小回 ②加大网格
```

## 2.5 全量数据准备（挂 tmux 后台，~3h）
> ⚠️ **新 tmux/终端不继承 venv**！进了 tmux 先激活（或直接用脚本——它会自动激活 venv）。
```bash
tmux new -s dataprep
source .venv/bin/activate            # 关键：新会话必须激活 venv（which python 应是 .venv/...）
# 候选→打分(WER/SIM/UTMOS)→聚合(多目标+仅WER)→两套偏好对，一条龙（脚本顶部也会自动 source venv）：
bash run_full_dataprep.sh data/prompts/dots_manifest_train.jsonl 6 16 2>&1 | tee logs_dataprep.txt
# Ctrl-b d 脱离；tmux attach -t dataprep 回来看
```

## 3. 对齐训练（产出各 treatment）
```bash
# ★ 开训前自检（命门）：FM 接缝通 + 同 RNG 决定论 + 不同 RNG 有差异，三项 PASS 再训
python alignment/check_fm_loss.py --ckpt rednote-hilab/dots.tts-base \
       --manifest data/prompts/dots_manifest_train.jsonl

accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_fullparam.yaml   # 头条(RQ1)
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_weronly.yaml               # reward-hacking 臂(RQ3)
accelerate launch alignment/train_open_soar.py    --config configs/open_soar_selfcorrect.yaml     # 自校正(RQ4)
accelerate launch alignment/train_sft.py          --config configs/sft_baseline.yaml              # 控制变量(仅winners)
# base 不训（评测时直接用 rednote-hilab/dots.tts-base）。A6000 档: dpo 改 configs/dpo_lora.yaml

# 看训练曲线（任选）：
#   实时 TensorBoard（需 SSH 端口转发 ssh -L 6006:localhost:6006 user@server）
tensorboard --logdir alignment/out/dpo_multiobj/tb --port 6006
#   或 离线出 PNG（免端口转发，scp 回本地看）
python alignment/plot_metrics.py --run alignment/out/dpo_multiobj
```

## 4. 合成 + 评测（离线 + 流式）
```bash
# 一键编排：5 个 treatment(含 base) × {离线,流式} 合成 → score_synth(WER/SIM/UTMOS) → 流式指标
python ablation/run_matrix.py --manifest data/prompts/eval_big.jsonl --nfe 10 --dry-run  # 先看命令
python ablation/run_matrix.py --manifest data/prompts/eval_big.jsonl --nfe 10            # 真跑(默认 ecapa SIM)
#   评测用 score_synth（复用 reward 的 transformers-Whisper/ECAPA/torch.hub-UTMOS 后端，
#   不用 seed-tts-eval shell，避免 s3prl/select.py 依赖坑）。
#   NFE 迁移消融: --nfe 4 8 10 16    指定 treatment: --treatments base dpo_multiobj dpo_weronly
# EmergentTTS 单独跑（需先 clone）
python eval/run_emergent_tts.py --ckpt alignment/out/dpo_multiobj --treatment dpo_multiobj_nfe10
```

## 5. 汇总
```bash
python results/aggregate_results.py    # 出主表/流式表/reward-hacking 表（每数可溯源）
python results/plots.py                # reward-hacking 图 + Pareto 图
```

## 6. 第一轮裁决 → 接第二轮（不再门控 Phase B）

第一轮 Table 1 的裁决是**警示性负面结果**：5 臂里**只有 `open_soar` 改善**，`sft`/`dpo_multiobj`/`dpo_weronly` 退化或崩坏（`dpo_multiobj` 灾难崩坏 + 25% 生成失败）。**故不再进入「Phase B 蒸馏量化」**——其前提「Phase A 出可部署配方」不成立。下一步改为**第二轮诊断（X1–X13）**：隔离 `sft` 崩坏归因、建设性救回（on-manifold / reward-weighted）、流式、跨架构/多语种、camera-ready。

## 7. 第二轮（X1–X13）GPU 跑序 → 一条龙见 docs/09

完整命令与意义见 [`09-第二轮实验程序.md`](09-第二轮实验程序.md)。摘要（按 GPU 预算分级）：

```bash
# —— P0 再分析（CPU、秒级，先做）——
python alignment/probe_analysis.py            # X1：高-gap 子集 + 偏相关(intrinsic) + per-component + 杠杆
python results/aggregate_results.py           # X2：失败率(25/100)入主表（与 WER 并列）
python eval/diagnose_eval.py \
    --eval-manifest data/prompts/eval_big.jsonl \
    --report inference/out/dpo_multiobj_nfe10/_synthesis_report.json \
    --metrics alignment/out/dpo_multiobj/metrics.jsonl \
    --csv alignment/out/probe_per_pair.csv     # X2/X3：LJ002 聚集 + 发散 vs graded

# —— P1 核心（~1–2 GPU日）——
python pairs/extract_losers.py && python pairs/extract_random.py   # X4：SFT-target 抽取（winners 已有）
accelerate launch alignment/train_sft.py --config configs/sft_losers.yaml        # X4
accelerate launch alignment/train_sft.py --config configs/sft_random.yaml        # X4：隔离 off-manifold vs 小数据SFT
accelerate launch alignment/train_sft.py --config configs/sft_groundtruth.yaml   # X4：on-manifold 真实录音
python candidates/generate_candidates.py --prompts data/prompts/dots_manifest_train.jsonl \
       --fixed-setting --fix-num-steps 10 --fix-guidance 1.2 --fix-speaker 1.5    # X5：settings-matched 复探针
python alignment/probe_analysis.py             # X5：复证反相关在固定生成设置下仍存活

# —— P2 建设性 + RQ2/3（~3–5 GPU日）——
python pairs/filter_pairs_by_likelihood.py     # X6a：只留 on-manifold winner
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_onmanifold.yaml        # X6a：救回 DPO-FM
accelerate launch alignment/train_reward_weighted.py --config configs/reward_weighted.yaml          # X6b
accelerate launch alignment/train_reward_weighted.py --config configs/reward_weighted_weronly.yaml  # X7：非错配框架内干净复现 reward-hacking
python ablation/run_matrix.py --manifest data/prompts/eval_big.jsonl --nfe 10 \
       --treatments dpo_onmanifold reward_weighted reward_weighted_weronly        # 评测新臂
python ablation/run_matrix.py --manifest data/prompts/eval_big.jsonl --nfe 10           # X8：去掉 --skip-streaming 出 Table 2
python eval/run_emergent_tts.py --ckpt alignment/out/open_soar --treatment open_soar_nfe10        # X9〔API 需核实〕

# —— P3 泛化（~1–2 GPU周）——
python alignment/voxcpm_preprocess.py && python alignment/probe_fm_reward_voxcpm.py   # X10：VoxCPM 跨架构复证〔8 处需核实〕
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_vctk.yaml          # X11：多语种（vctk/aishell3 同理）
# 多语种数据准备见 data/run_multilingual.md

python results/aggregate_results.py && python results/plots.py   # 汇总（失败率列 + Pareto/机理图）
```

> P4（X12 人类 ABX/MOS 暂不可行→备协议 + 记 limitation；X13 camera-ready：核 arXiv id / §3.1 架构代号 / EmergentTTS provenance）见 docs/09 §P4。

## 本机能做的（设计期校验）
```bash
find . -name '*.py' -exec python3 -m py_compile {} +
find . -name '*.sh' -exec bash -n {} +
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
```
