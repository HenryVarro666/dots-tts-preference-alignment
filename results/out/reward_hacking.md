# Table 3 reward hacking 对照

| treatment | WER↓ | UTMOS↑ | EmergentTTS↑ | vs base 校正 |
|---|---|---|---|---|
| base | 0.1122 [0.0783,0.1538] | 4.278 [4.222,4.335] |  | —(参照) |
| dpo_multiobj | 0.8897 [0.6317,1.3045] | 2.719 [2.446,2.994] |  | WER†, SIM†, UTMOS† |
| dpo_weronly | 0.1095 [0.0821,0.1379] | 4.055 [4.002,4.110] |  | SIM†, UTMOS† |

> 判读(PROTOCOL §6)：dpo_weronly 若 **WER 最低但 UTMOS/EmergentTTS 退化** = 复现 reward hacking；若仅WER未伤自然度→多目标前提不成立，如实报告。
> vs base 校正：标记取自**全消融网格**的多重比较族(同 main_table，†=Bonferroni/*=FDR/无=校正后不显著)，非本子表内重算——FDR 族是整张消融网格、非这 3 行。
