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
bash fetch_public_corpora.sh
python prepare_prompts.py --corpus ljspeech --root corpora/ljspeech48k --limit 200   # 先小跑打通
python prepare_prompts.py --corpus vctk     --root corpora/vctk
python prepare_prompts.py --corpus aishell3 --root corpora/aishell3
```

许可/采样率见 [`LICENSES.md`](LICENSES.md)（标〔需核实〕项跑前确认）。
