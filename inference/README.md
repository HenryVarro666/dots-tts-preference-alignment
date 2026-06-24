# inference/ — 阶段⑥：批量合成（离线 + 1T1A 流式）

**角色**：用对齐后 ckpt 把评测集合成成 wav，供 `eval/` 打分。**离线与流式双模**。

**为什么存在**：评测必须「同集、同设置、只换权重」才能归因到对齐；流式必须走**真实流式路径**重测（不能拿离线结果替代）。

**上游**：对齐 ckpt（`alignment/`）+ 评测清单。
**下游**：`eval/`。

| 文件 | 作用 | 关键点 |
|---|---|---|
| `batch_synthesize.py` | 离线合成 | `--num-steps` 暴露部署 NFE（NFE 迁移消融） |
| `batch_synthesize_streaming.py` | 1T1A 流式合成 | 采集**首包延迟**；**流式仅 x-vector 克隆**（dots.tts 流式拒 `prompt_text`，源码 :117） |

**怎么用**：
```bash
python batch_synthesize.py           --ckpt <ckpt> --manifest <eval.jsonl> --treatment dpo_multiobj --num-steps 10
python batch_synthesize_streaming.py --ckpt <ckpt> --manifest <eval.jsonl> --treatment dpo_multiobj
```

> ⚠️ 流式/离线的 SIM 非严格可比（流式无 prompt_text 续接），报告时分开标注（见 `../docs/04-RISKS.md`）。
