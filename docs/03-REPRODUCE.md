# 03 · 复现指南（用户在 GPU 机器上的端到端跑序）

> ⚠️ 本机(macOS/无 GPU/权重未下载)**不能跑**，只能 `py_compile`/`bash -n`。下列在 **2×H100 或 2×A6000** 上执行。
> 每条命令都标了「为什么这一步」。边跑边把实际命令/数字填进 [`RUN_JOURNAL_TEMPLATE.md`](RUN_JOURNAL_TEMPLATE.md) 的副本。

## 0. 环境
```bash
bash env/setup_env.sh                  # 为什么: 固化 dots.tts + seed-tts-eval + speechmos + WavLM ckpt
source .venv/bin/activate
python -c "import dots_tts, speechmos, jiwer, peft; print('ok')"
# 配 accelerate（2×H100 建议 deepspeed zero_stage=2, bf16）
accelerate config
```

## 1. 数据（公开集）
```bash
bash data/fetch_public_corpora.sh      # 为什么: 仅用公开集（无私有数据）
python data/prepare_prompts.py --corpus ljspeech --root data/corpora/ljspeech48k --limit 200  # 先小跑打通
# 跑通后再上 VCTK / AISHELL-3 全量
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

## 3. 对齐训练（产出各 treatment）
```bash
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_fullparam.yaml   # 头条
accelerate launch alignment/train_dots_tts_dpo.py --config configs/dpo_weronly.yaml               # reward-hacking 臂
accelerate launch alignment/train_open_soar.py    --config configs/open_soar_selfcorrect.yaml     # 中间臂
accelerate launch alignment/train_dots_tts_dpo.py --config configs/sft_baseline.yaml              # 控制变量
# A6000 档: 改用 configs/dpo_lora.yaml
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
