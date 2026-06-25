# 03 · 复现指南（用户在 GPU 机器上的端到端跑序）

> ⚠️ 本机(macOS/无 GPU/权重未下载)**不能跑**，只能 `py_compile`/`bash -n`。下列在 **2×H100 或 2×A6000** 上执行。
> 每条命令都标了「为什么这一步」。边跑边把实际命令/数字填进 [`RUN_JOURNAL_TEMPLATE.md`](RUN_JOURNAL_TEMPLATE.md) 的副本。

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

accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_fullparam.yaml   # 头条
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_weronly.yaml               # reward-hacking 臂
accelerate launch alignment/train_open_soar.py    --config configs/open_soar_selfcorrect.yaml     # 中间臂
accelerate launch alignment/train_dots_tts_dpo.py --config configs/sft_baseline.yaml              # 控制变量
# A6000 档: 改用 configs/dpo_lora.yaml

# 看训练曲线（任选）：
#   实时 TensorBoard（需 SSH 端口转发 ssh -L 6006:localhost:6006 user@server）
tensorboard --logdir alignment/out/dpo_multiobj/tb --port 6006
#   或 离线出 PNG（免端口转发，scp 回本地看）
python alignment/plot_metrics.py --run alignment/out/dpo_multiobj
```

## 4. 合成 + 评测（离线 + 流式）
```bash
# 由 ablation 一键编排（推荐）；也可手动逐 treatment 跑
python ablation/run_matrix.py --manifest data/prompts/ljspeech_eval.jsonl --lang en --nfe 10 --dry-run  # 先看命令
python ablation/run_matrix.py --manifest data/prompts/ljspeech_eval.jsonl --lang en --nfe 4 8 10 16     # NFE 迁移消融
# EmergentTTS 单独跑（需先 clone）
python eval/run_emergent_tts.py --ckpt alignment/out/dpo_multiobj --treatment dpo_multiobj
```

## 5. 汇总
```bash
python results/aggregate_results.py    # 出主表/流式表/reward-hacking 表（每数可溯源）
python results/plots.py                # reward-hacking 图 + Pareto 图
```

## 6. 判定 → Phase B（仅当 A 达标）
```bash
# 判据见 phaseB/PHASE_B_PROTOCOL.md
accelerate launch phaseB/distill_aligned.py --teacher alignment/out/dpo_multiobj --target-steps 2
python phaseB/quantize.py --ckpt phaseB/out/distill_aligned --mode int8 --out phaseB/out/quantized
python phaseB/eval_8gb_matrix.py --ckpt-root phaseB/out --manifest data/prompts/ljspeech_eval.jsonl
```

## 本机能做的（设计期校验）
```bash
find . -name '*.py' -exec python3 -m py_compile {} +
find . -name '*.sh' -exec bash -n {} +
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
```
