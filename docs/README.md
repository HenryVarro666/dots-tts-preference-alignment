# `docs/` — research record & navigation index

Supporting documentation for **[*When Likelihood Disagrees with Quality*](../PAPER_DRAFT.md)** (see the [project README](../README.md) for the overview). The documents are written in Chinese (technical terms in English); descriptions below are in English so a reviewer can navigate.

> Reading order for the impatient: **[06 Core finding](#core-finding) → [08 Results](#results-tables) → [02 Decision log](#rigor--process) → [01 Method](#method--reproduction)**.

## Core finding
| Doc | What it is |
|---|---|
| [`06-核心发现-FM损失与奖励错配.md`](06-核心发现-FM损失与奖励错配.md) | **The central diagnosis** — the flow-matching likelihood surrogate is *valid as a likelihood* but **fails as an implicit reward**: the automatic-metric preference anti-correlates with the model's own likelihood. Probe design, the settings-matched control, leverage/disattenuation analysis, cross-architecture (VoxCPM2) replication. |

## Results tables
| Doc | What it is |
|---|---|
| [`08-实验结果-Table1.md`](08-实验结果-Table1.md) | **All result tables** — the 11-arm offline main table (single consolidated batch, bootstrap CIs), the streaming RQ2 table (TTFA / streaming WER+UTMOS / decode-path discrepancy), the reward-hacking RQ3 table, and the X4 SFT-target off-manifold gradient. |

## Roadmap
| Doc | What it is |
|---|---|
| [`11-未来工作与展望.md`](11-未来工作与展望.md) | **Future work & roadmap** — the consolidated single source-of-truth: completed experiments (X4–X14, with results), open resubmit experiments, limitations, future prospects, deferred (budget/scope) items, and a 〔待核实〕 interface tracker. Pulls together what was scattered across §7/§8, the plan, and the ADRs. |
| [`12-EMNLP审稿补充实验.md`](12-EMNLP审稿补充实验.md) | **EMNLP review-driven supplementary experiments** — runnable protocols for the experiments the peer-review panel flagged as gating a credible Findings resubmit (E1 selection-artifact null control, E2 per-block/worst-case EOS, E3 re-synthesis variance) + E4/E5 for Main. Each with reviewer-concern mapping, feasibility, steps, success criterion, and where it lands in the paper. |

## Method & reproduction
| Doc | What it is |
|---|---|
| [`01-METHOD.md`](01-METHOD.md) | The DPO-FM derivation on continuous-latent flow-matching AR TTS — the math and its justification (four shared-noise forward passes, the `L_FM` surrogate, open-SOAR). |
| [`00-总览与步骤详解.md`](00-总览与步骤详解.md) | End-to-end walkthrough — every pipeline step: *what / how / why / what it means for the experiment*. |
| [`03-REPRODUCE.md`](03-REPRODUCE.md) | First-round reproduction guide (the GPU-host run sequence). |
| [`09-第二轮实验程序.md`](09-第二轮实验程序.md) | Second-round experiment program (X1–X13 consolidated run plan). |

## Rigor & process
| Doc | What it is |
|---|---|
| [`02-DECISIONS.md`](02-DECISIONS.md) | **Architecture Decision Records** — every load-bearing choice (background / choice / rationale / rejected alternatives / impact), including the 12-perspective adversarial self-review. |
| [`07-审稿意见-第一轮.md`](07-审稿意见-第一轮.md) | Simulated first-round peer review (5 blind reviewers + a meta-review) used to harden the paper. |
| [`04-RISKS.md`](04-RISKS.md) | Risk & decision-point register. |
| [`05-踩坑与修复记录.md`](05-踩坑与修复记录.md) | Pitfalls-and-fixes ledger — every error / warning / improvement, recorded. |
| [`09-引用与arXiv核验清单.md`](09-引用与arXiv核验清单.md) | Citation & arXiv-id verification checklist (every id curl-title-matched; feeds [`../refs.bib`](../refs.bib)). |

## Archive
| Doc | What it is |
|---|---|
| [`10-项目说明-原始详细中文版-存档.md`](10-项目说明-原始详细中文版-存档.md) | Verbatim archive of the original detailed Chinese project README (experiment narrative & history). Final numbers: see [`PAPER_DRAFT.md`](../PAPER_DRAFT.md) / `08`. |
| [`RUN_JOURNAL_TEMPLATE.md`](RUN_JOURNAL_TEMPLATE.md) | Run-journal template. |
