# 数据集许可与采样率（跑前务必核实标〔需核实〕项）

> 沿用仓库规范：**外部信息一律先核实再用**。下面采样率/许可如与官方页不符，以官方为准。

| 语料 | 语言 | 许可 | 采样率 | 商用 | 备注 |
|---|---|---|---|---|---|
| **LJSpeech-48kHz** | EN(单说话人) | Public Domain (LJSpeech 原始)；48kHz 重制版见来源 | 48 kHz | ✅ | 复用 dots.tts `prepare_train_jsonl_manifest.py`（`alibabasglab/LJSpeech-1.1-48kHz`），与 48kHz 训练契约一致，**最低风险首跑** |
| **VCTK-0.92** | EN(多说话人) | CC-BY-4.0 | 〔需核实〕标准发布 48 kHz，但 `wav48_silence_trimmed` 为 **FLAC** | ✅(署名) | 多说话人、克隆评测友好；格式/采样率跑前确认 |
| **AISHELL-3** | ZH(多说话人) | Apache-2.0 | **44.1 kHz** → 需重采样到 48 kHz | ✅ | 中文多说话人；重采样由 dots.tts 内部 `high_quality_resample` 处理 |

## 评测集（与训练集严格不重叠）
| 集 | 来源 | 用途 |
|---|---|---|
| seed-tts-eval `test-en` / `test-zh` / `test-zh-hard` | `../../参考代码/seed-tts-eval/`（Common Voice EN + DiDiSpeech-2 ZH + hard-case） | WER/SIM 客观评测的事实标准 |
| EmergentTTS-Eval | 需另 clone（arXiv 2505.23009 / `boson-ai/EmergentTTS-Eval-public`） | 表现力 LLM-judge 胜率 |

## 合规原则
- **仅公开数据**；不引入任何私有/未授权语料。
- 新产出（候选音频、对齐权重）随 **Apache-2.0**；但**受各源语料许可约束**（如 VCTK 需署名）。
- 训练/偏好用 `*_train.jsonl`，评测用独立 seed-tts-eval / EmergentTTS，**不混用**。
