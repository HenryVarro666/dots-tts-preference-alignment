# ablation/ — 阶段⑧：实验矩阵编排

**角色**：把「梯队 × 消融 × 评测模式」串成可复现的批量 run，并写**可溯源账本**。

**为什么存在**：实验最容易乱在「这个数字哪个设置跑的」。`run_matrix.py` 用 subprocess 串既有脚本（不重复造轮子），把每个数字绑定到 (treatment, config, ckpt, NFE, 模式, 产物路径)。

**上游**：各 treatment 的对齐 ckpt + 评测集。
**下游**：`results/aggregate_results.py` 读账本汇总。

**产物**：`ledger.jsonl`（每行 = 一个 run 的完整设置与产物路径）。

**怎么用**：
```bash
python run_matrix.py --manifest <eval.jsonl> --lang zh --nfe 10 --dry-run   # 先 dry-run 看命令
python run_matrix.py --manifest <eval.jsonl> --lang zh --nfe 4 8 10 16      # NFE 迁移消融
```

> 消融维度（β / 奖励组成 / 配对策略）通过传入不同 `configs/*.yaml` 训出的 ckpt 进入矩阵；
> 见 `PROTOCOL.md` 第 3 节。
