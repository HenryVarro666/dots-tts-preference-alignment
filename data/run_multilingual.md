# run_multilingual.md — 多语种 / 多说话人 scale-up 端到端跑序（VCTK + AISHELL-3）

**这是什么**：把现有 LJSpeech 单语单说话人管线，扩到 **VCTK（英、多说话人）** 与 **AISHELL-3（中、多说话人）** 的完整跑序。
**为什么做（对 claim 的意义）**：
- **多说话人让 SIM 变可判别** —— LJSpeech 单说话人时「synth vs 参考音」的 SIM 近-ceiling，对各 treatment 不可区分（domain reviewer 的核心质疑）；VCTK/AISHELL-3 每条 prompt 是不同说话人，SIM 有真实方差 → SIM 成为可判别指标。
- **规模超 188 对** —— LJSpeech 只产出 ~188 对；VCTK（~110 说话人、~44k 句）、AISHELL-3（~218 说话人、~88k 句）可产出远多于 188 的偏好对（用 `--limit` 控规模）。
- **加中文泛化** —— AISHELL-3 检验头条 claim 跨语言是否成立。**ZH 的 WER 实为 CER**（`reward/score_wer.py` 与 `eval/score_synth.py` 已按每行 `lang` 字段自动走 funasr/Paraformer + `jiwer.cer`）。

**管线代码不动**：数据适配器（`prepare_prompts.py` 的 `adapt_vctk` / `adapt_aishell3`）已就绪；本文件只是「用既有脚本 + 各语料专属产物文件名」的编排说明，外加 6 个 config 变体（`configs/{dpo_multiobj,open_soar,sft}_{vctk,aishell3}.yaml`）。

> ⚠️ 本机（macOS、无 GPU）只能 `bash -n` / `py_compile` 语法检查；真正下载/生成/训练/评测都在 GPU 机器上跑。

---

## 0. 三条必须先记住的「固定路径」陷阱（决定跑序形态）

既有管线的中间产物是**固定路径、会被覆盖**，这直接决定了「一个语料整跑完再跑下一个」的串行形态：

1. **候选索引固定**：`candidates/generate_candidates.py` 把索引写死到 `candidates/out/candidates_index.jsonl`（单文件）。
   - 各语料的 wav 目录按 `fid` 隔离不冲突（`vctk-*` vs `aishell3-*`），但**索引文件会被后一个语料覆盖**。
2. **打分文件固定**：`reward/out/{wer,sim,utmos,rewards}.jsonl` 路径写死，后一个语料覆盖前一个。
3. **build_preference_pairs 读死索引**：它只读 `candidates/out/candidates_index.jsonl`。

**结论**：必须**一个语料从「候选生成 → 打分 → 聚合 → 配对」整段跑完**，并在「配对/抽 winners」这步用 `--out` 把**最终产物落到语料专属文件名**（这些才是 config 引用、需要长期留存的），再开始下一个语料。中间的候选/打分文件被覆盖无所谓——配对产物已固化。

各 config 引用的语料专属产物（由下方 `--out` 落地）：

| config | 引用字段 | 语料专属产物 |
|---|---|---|
| `dpo_multiobj_vctk.yaml` | `dpo.pairs_file` | `pairs/out/pairs_best_worst_vctk.jsonl` |
| `sft_vctk.yaml` | `train_manifest` | `pairs/out/winners_only_vctk.jsonl` |
| `open_soar_vctk.yaml` | `train_manifest` | `data/prompts/vctk_train.jsonl`（prepare_prompts 直接产出，稳定不覆盖） |
| `dpo_multiobj_aishell3.yaml` | `dpo.pairs_file` | `pairs/out/pairs_best_worst_aishell3.jsonl` |
| `sft_aishell3.yaml` | `train_manifest` | `pairs/out/winners_only_aishell3.jsonl` |
| `open_soar_aishell3.yaml` | `train_manifest` | `data/prompts/aishell3_train.jsonl` |

---

## 1. VCTK（EN，多说话人）端到端

```bash
cd /Users/caochao/Documents/TTS/实验项目/dots-tts-preference-alignment
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# (1) 取数：见 fetch_public_corpora.sh 的 [VCTK-0.92] 段（CC-BY-4.0；需手动同意许可下载解压）
bash data/fetch_public_corpora.sh                     # 按提示把 VCTK 解到 data/corpora/vctk
#     为什么手动：VCTK 官方下载需同意许可，无法纯脚本化。
#     〔需核实〕wav48_silence_trimmed 为 FLAC、采样率/格式跑前确认（见 data/LICENSES.md）。

# (2) 统一 prompt 清单（adapt_vctk 已存在；lang 自动=en）
python data/prepare_prompts.py --corpus vctk --source data/corpora/vctk --limit 400
#     产物: data/prompts/vctk_{train,eval}.jsonl（按 fid 哈希切，无泄漏）
#     --limit 控规模：留空=全量(数万句)；400 已远超 LJSpeech 的 188，足够把 pairs 拉到 >188。

# (3) 候选生成（最耗时；fid 以 vctk- 开头，wav 目录不与 LJSpeech/aishell3 冲突）
python candidates/generate_candidates.py --prompts data/prompts/vctk_train.jsonl --k 6

# (4) 三路打分。lang=en 由候选行携带 → score_wer 自动走 Whisper-large-v3（无需 --lang）
python reward/score_wer.py   --batch-size 16
python reward/score_utmos.py
python reward/score_sim.py   --backend ecapa
#     为什么这里 SIM 才有意义：多说话人 → SIM 有方差、可判别（单说话人时近-ceiling 无用）。

# (5) 聚合多目标奖励（按候选池 z-norm 后加权）
python reward/aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3 --out reward/out/rewards.jsonl

# (6) 配对 + 抽 winners —— 关键：用 --out 落到 VCTK 专属文件名（config 引用的就是它们）
python pairs/build_preference_pairs.py --rewards reward/out/rewards.jsonl \
       --strategy best_worst --out pairs/out/pairs_best_worst_vctk.jsonl
python pairs/extract_winners.py --pairs pairs/out/pairs_best_worst_vctk.jsonl \
       --out pairs/out/winners_only_vctk.jsonl

# (7) FM-reward 探针（在 VCTK 专属 pairs 上复测「FM 损失是否有效隐式奖励」）
python alignment/probe_fm_reward.py --pairs pairs/out/pairs_best_worst_vctk.jsonl --n 24 --k 24

# (8) 训练四臂（base 不训；用各 VCTK config）
python alignment/train_sft.py          --config configs/sft_vctk.yaml
python alignment/train_open_soar.py    --config configs/open_soar_vctk.yaml
python alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_vctk.yaml
#     产物: alignment/out/{sft,open_soar,dpo_multiobj}_vctk/

# (9) 评测矩阵 + 汇总（见下方 §3 的 run_matrix 硬编码路径桥接）
#     评测集直接用 prepare_prompts 切出的 held-out: data/prompts/vctk_eval.jsonl
```

---

## 2. AISHELL-3（ZH，多说话人）端到端

```bash
cd /Users/caochao/Documents/TTS/实验项目/dots-tts-preference-alignment
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# (1) 取数：见 fetch_public_corpora.sh 的 [AISHELL-3] 段（Apache-2.0；openslr.org/93）
bash data/fetch_public_corpora.sh                     # 解到 data/corpora/aishell3
#     〔需核实〕原生 44.1kHz；候选生成时由 dots.tts 内部重采样到 48kHz（不在此预重采样）。

# (2) 统一 prompt 清单（adapt_aishell3 已存在；lang 自动=zh → 下游全程走中文路由）
python data/prepare_prompts.py --corpus aishell3 --source data/corpora/aishell3 --limit 400
#     产物: data/prompts/aishell3_{train,eval}.jsonl（每行 lang=zh）

# (3) 候选生成（fid 以 aishell3- 开头）
python candidates/generate_candidates.py --prompts data/prompts/aishell3_train.jsonl --k 6

# (4) 三路打分。**ZH 路由不靠 --lang**：候选行 lang=zh → score_wer 自动走 funasr/Paraformer 且报 CER。
python reward/score_wer.py   --batch-size 16          # ZH=CER（score_wer 已支持，jiwer.cer）
python reward/score_utmos.py                          # UTMOS 语种无关
python reward/score_sim.py   --backend ecapa          # 说话人嵌入，语种无关；多说话人 → 可判别

# (5) 聚合多目标奖励
python reward/aggregate_reward.py --w-wer 0.4 --w-sim 0.3 --w-utmos 0.3 --out reward/out/rewards.jsonl

# (6) 配对 + 抽 winners —— 用 --out 落到 AISHELL-3 专属文件名
python pairs/build_preference_pairs.py --rewards reward/out/rewards.jsonl \
       --strategy best_worst --out pairs/out/pairs_best_worst_aishell3.jsonl
python pairs/extract_winners.py --pairs pairs/out/pairs_best_worst_aishell3.jsonl \
       --out pairs/out/winners_only_aishell3.jsonl

# (7) FM-reward 探针（中文 pairs 上复测）
python alignment/probe_fm_reward.py --pairs pairs/out/pairs_best_worst_aishell3.jsonl --n 24 --k 24

# (8) 训练四臂
python alignment/train_sft.py          --config configs/sft_aishell3.yaml
python alignment/train_open_soar.py    --config configs/open_soar_aishell3.yaml
python alignment/train_dots_tts_dpo.py --config configs/dpo_multiobj_aishell3.yaml

# (9) 评测矩阵 + 汇总（见 §3；评测集 = data/prompts/aishell3_eval.jsonl）
```

---

## 3. 评测矩阵 + 汇总（两语料通用；含 run_matrix 路径桥接）

**已核实的两个 run_matrix 行为，必须知道**：
- `ablation/run_matrix.py` 的 `TREATMENTS` 把各臂 ckpt **硬编码**成 `alignment/out/<treat>`（**无 `_vctk`/`_aishell3` 后缀**）。本任务**不动管线代码**，故用临时 symlink 把语料 ckpt 桥接到这些默认路径。
- `run_matrix.py` 声明了 `--lang` 但**从不向子脚本传递**（解析后未使用）；真正的 ZH→CER 路由完全由**评测清单每行的 `lang` 字段**驱动（`prepare_prompts` 已置 `aishell3_eval.jsonl` 为 `lang=zh`，`batch_synthesize.py` 与 `score_synth.py` 都按 `r.get("lang")` 路由）。所以下面给 `--lang zh` 只是**登记/可读性**，对路由是 no-op；不给也照样走 CER。

```bash
# ---- 以 VCTK 为例（AISHELL-3 把 vctk→aishell3、--lang en→zh、eval 清单换名即可）----

# (a) 桥接：把语料 ckpt symlink 到 run_matrix 硬编码的默认路径（不改代码）
for t in sft open_soar dpo_multiobj; do
  ln -sfn "$(pwd)/alignment/out/${t}_vctk" "alignment/out/${t}"
done
#     base 臂指向 HF id rednote-hilab/dots.tts-base，语种无关、无需桥接。

# (b) 跑矩阵：离线主证据先跑通（流式可后补，--skip-streaming）
python ablation/run_matrix.py --manifest data/prompts/vctk_eval.jsonl --lang en \
       --treatments base sft open_soar dpo_multiobj --sim-backend ecapa --skip-streaming
#     ZH 时：--manifest data/prompts/aishell3_eval.jsonl --lang zh（--lang 仅登记；路由靠清单 lang 字段）

# (c) 汇总成主表/流式表/reward-hacking 表
python results/aggregate_results.py

# (d) 归档本语料结果，再跑下一个语料（ledger.jsonl 与 results/out/* 是固定路径，会被下一个语料覆盖）
mkdir -p results/out_vctk && cp -a results/out/* results/out_vctk/ 2>/dev/null || true
cp -a ablation/ledger.jsonl ablation/ledger_vctk.jsonl
#     清桥接 symlink，避免污染下一个语料：
for t in sft open_soar dpo_multiobj; do rm -f "alignment/out/${t}"; done
```

> 评测的 tag 形如 `<treat>_nfe10`，**两语料会撞名**（都产 `dpo_multiobj_nfe10`）。故必须如 (d) 在每个语料评完后归档 `results/out/` 与 `ablation/ledger.jsonl`，再跑下一个语料。
> EmergentTTS（表现力）不在 run_matrix 内，如需该列另跑 `eval/run_emergent_tts.py --treatment <tag>`（缺则主表该列留空，不报错）。

---

## 4. 一图流跑序（每语料）

```
fetch_public_corpora(vctk/aishell3)
  → prepare_prompts --corpus vctk/aishell3            # data/prompts/<corpus>_{train,eval}.jsonl
  → generate_candidates --prompts <corpus>_train      # candidates/out/（固定路径，串行跑）
  → score_wer / score_utmos / score_sim               # ZH 走 CER 由 lang 字段自动路由，无 --lang
  → aggregate_reward --out reward/out/rewards.jsonl
  → build_preference_pairs --out pairs_best_worst_<corpus>.jsonl
  → extract_winners        --out winners_only_<corpus>.jsonl
  → probe_fm_reward --pairs pairs_best_worst_<corpus>.jsonl
  → train_sft / train_open_soar / train_dots_tts_dpo  --config <arm>_<corpus>.yaml
  → (symlink 桥接) run_matrix --manifest <corpus>_eval.jsonl [--lang zh]
  → aggregate_results → 归档 results/out_<corpus>/ + ledger_<corpus>.jsonl
```

**意义复述（写进论文 §泛化/§讨论）**：多说话人解 SIM 近-ceiling（可判别）、规模 >188 对、加中文检验跨语言泛化；ZH 全程 WER=CER 且由 `lang` 字段自动路由——无需也不应手动塞 `--lang` 给 score_* / run_matrix 来改路由。
