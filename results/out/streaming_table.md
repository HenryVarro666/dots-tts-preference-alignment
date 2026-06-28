# Table 2 流式表

| treatment | NFE | 流式WER↓ | 流式SIM↑ | 流式UTMOS↑ | 首包 p50(ms)↓ | 边界不连续度↓ |
|---|---|---|---|---|---|---|
| base | 10 | 0.0874 | 0.8317 | 4.326 | 162.2 | 42.0009 |
| sft | 10 | 0.2121 | 0.8245 | 4.283 | 162.4 | 47.7513 |
| open_soar | 10 | 0.1091 | 0.8128 | 4.450 | 162.5 | 96.2096 |
| dpo_multiobj | 10 | 0.1338 | 0.8299 | 4.406 | 162.5 | 85.0564 |
| dpo_weronly | 10 | 0.1787 | 0.8243 | 4.024 | 164.3 | 55.8316 |
| dpo_onmanifold | 10 |  |  |  |  |  |
| reward_weighted | 10 |  |  |  |  |  |
| reward_weighted_weronly | 10 |  |  |  |  |  |
| sft_groundtruth | 10 |  |  |  |  |  |
| sft_losers | 10 |  |  |  |  |  |
| sft_random | 10 |  |  |  |  |  |
