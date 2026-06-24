# results/ — 阶段⑨：汇总与出图

**角色**：把散落的评测产物收敛成「能直接进论文的表/图」，且每格可溯源到 run。

**为什么存在**：主表→RQ1、流式表→RQ2、reward-hacking 表/图→RQ3。这是论文 Results 的来源。

**上游**：`ablation/ledger.jsonl` + 各评测产物。
**下游**：`PAPER_SKELETON.md` 引用这些表/图。

| 文件 | 产物 |
|---|---|
| `aggregate_results.py` | `out/master.csv` + `main_table.md` / `streaming_table.md` / `reward_hacking.md` |
| `plots.py` | `out/reward_hacking.png` / `out/pareto.png` |

**关键设计**：master.csv 每行带 `ckpt` 列 → **数字可溯源**（哪个权重/设置跑出来的）。

**怎么用**：
```bash
python aggregate_results.py
python plots.py
```

> WER/SIM 两列来自 seed-tts-eval 终端输出，需回填（或扩展 `aggregate_results.py` 解析其输出文件）。
