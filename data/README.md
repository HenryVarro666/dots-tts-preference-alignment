# data/ — 阶段①：构造 prompt 清单

**角色**：把异质的公开语料统一成下游唯一认得的 schema（`PromptRow`），并切出无泄漏的 train/eval 子集。

**为什么存在**：下游所有阶段（candidates/reward/pairs/alignment）都不该关心数据来自 VCTK 还是 AISHELL-3——这里做「数据源 ↔ 算法」的解耦层。

**上游**：`fetch_public_corpora.sh` 下载的原始语料。
**下游**：`candidates/generate_candidates.py` 读 `prompts/*_train.jsonl`。

**产物**：`prompts/{corpus}_{train,eval}.jsonl`，每行 = `{fid,text,prompt_audio,prompt_text,lang,source}`。

**关键设计**：
- 切分按 `fid` 稳定哈希 → 确定性、可复现、**训练/评测零泄漏**（结论可信底线）。
- 非 48kHz 语料（AISHELL-3 44.1k）不在此重采样，交给 dots.tts 推理内部处理，避免重复音频依赖。

**怎么用**：
```bash
# 推荐：LJSpeech 用 dots.tts 下载器一步产出 manifest，再直接读它（最稳，audio 为绝对路径）
python ../.deps/dots.tts/scripts/prepare_train_jsonl_manifest.py --output-dir corpora/ljspeech48k --valid-size 0
python prepare_prompts.py --corpus dots_manifest --source corpora/ljspeech48k/ljspeech_48khz_manifest_train.jsonl --lang en --limit 200

# VCTK / AISHELL-3 用各自原生目录
bash fetch_public_corpora.sh
python prepare_prompts.py --corpus vctk     --source corpora/vctk
python prepare_prompts.py --corpus aishell3 --source corpora/aishell3
```

> `--corpus dots_manifest` 直接读 `{fid,audio,text}` 清单；也可喂任何 dots.tts 风格 manifest。
> `--root`/`--manifest` 是 `--source` 的别名（向后兼容）。

许可/采样率见 [`LICENSES.md`](LICENSES.md)（标〔需核实〕项跑前确认）。
