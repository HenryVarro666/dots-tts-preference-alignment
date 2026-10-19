# When Likelihood Disagrees with Quality

### A cautionary diagnosis of Direct Preference Optimization for continuous-latent flow-matching TTS

> **Research question.** Can you align a modern continuous-latent text-to-speech model the obvious way — generate candidates, score them with automatic metrics, and run DPO toward the winners?
>
> **Answer (this project).** No — and *why* is the interesting part. The automatic-metric reward systematically prefers the candidate the model finds **less** likely, so the natural DPO adaptation optimizes in the wrong direction and the model degrades. I prove this with a held-out benchmark, a training-free mechanistic probe, and a cross-architecture replication, then show which alignment routes *do* work.

A **self-directed end-to-end research project**: data → candidates → reward → preference pairs → four training objectives → offline & streaming evaluation → statistical analysis → paper. The headline is a **falsifiable negative/diagnostic result**, not a result the project was hoping to find.

> 📄 Paper: [`PAPER_DRAFT.md`](PAPER_DRAFT.md) · 🧪 Findings & results: [`docs/`](docs/) · 🧭 Decision log / 12-perspective red-team: [`docs/02-DECISIONS.md`](docs/02-DECISIONS.md) · 📚 Verified bibliography: [`refs.bib`](refs.bib) · 🗄️ Detailed project record (中文): [`docs/10-项目说明-原始详细中文版-存档.md`](docs/10-项目说明-原始详细中文版-存档.md)
>
> License **Apache-2.0** (code) · Models studied: **dots.tts** (2B, flow-matching, Apache-2.0) + **VoxCPM2** (2B, diffusion-AR, OpenBMB)

---



<!-- problem-framing:start -->

## 问题与验证

> **在相同数据与训练预算下，基于 FM surrogate 的偏好训练，能否在独立质量评价中超过 winner-only SFT，并在生成条件变化后保持收益？**

**为什么值得解决：** 连续语音模型的后训练需要把偏好信号转化为可听见的生成改善。只有训练损失下降或自动评分提高，还不足以判断方法是否适合实际合成。

**核心矛盾：** 自动指标、训练 surrogate 与真实生成质量可能不一致；不同提示和解码条件也可能改变观察到的退化。FM loss 不自动等于真实 likelihood，与奖励的相关性不能单独证明训练失败的因果机制。

[完整问题与验证方案](PROBLEM.md) · [可复用 Prompt](PROBLEM_PROMPT.md)

<!-- problem-framing:end -->

## TL;DR

Continuous-latent TTS (flow-matching over continuous audio latents — **no discrete audio tokens**) has no native preference-alignment recipe, because DPO needs token log-probabilities the model never produces. The *natural* fix, **DPO-FM**, substitutes the flow-matching loss `L_FM` as a likelihood surrogate (following Diffusion-DPO). I tested its single load-bearing assumption — *does an automatic-metric reward prefer the candidate the model finds more likely?* — and found the **opposite**:

- **The reward preference is monotonically *anti-correlated* with the model's own flow likelihood** (settings-matched Spearman **ρ = −0.216**, N = 188, strengthening on the highest-quality-gap pairs). The reward systematically picks the **off-manifold** member of each pair.
- This is **not** a failure of `L_FM` as a likelihood proxy (a likelihood-aligned preference optimizes it to saturation) — it's a failure of the **reward↔likelihood coupling**.
- Held-out (n = 100): **only the likelihood-aligned, on-manifold route improves**; every arm trained toward reward-selected self-generated targets — DPO-FM *and* plain SFT-on-winners — **collapses**.
- The collapse is **decode-path-specific**: the same checkpoints **stream perfectly normal speech** under production (1T1A) decoding — a generation-time artifact, not a broken model.
- The diagnosis **replicates confound-free on a second, independently-built architecture** (VoxCPM2, diffusion-AR; settings-matched ρ = −0.266) → architecture-level evidence across **N = 2** continuous-latent implementations.

---

## Why this matters

Preference alignment (RLHF/DPO) is the default post-training tool for LLMs, and the obvious move is to port it to speech. This project shows the obvious port has a **silent failure mode that standard metrics hide**: training "succeeds" (loss never diverges), yet the model degrades at generation time on a subset of inputs. The deliverable is a **mechanism + a cheap diagnostic you can run *before* spending a GPU-week** — measure the sign of the likelihood↔reward coupling on held-out pairs first.

Concrete takeaways for production speech systems:
- **Reward design ≠ reward optimization.** When the reward prefers off-manifold (e.g. over-guided) samples, *no reweighting of the WER/speaker/UTMOS terms fixes it* — all three axes anti-correlate with likelihood.
- **Offline eval can mislead about streaming.** A catastrophic offline collapse here did *not* transfer to per-token streaming — the deployment-relevant path was fine. Testing one decode path would have given the wrong answer in both directions.
- **What works:** reward-free self-correction toward real targets (open-SOAR), or a *non-likelihood* reward-weighted objective anchored to real recordings.

---

## Key results

### 1. Held-out benchmark (offline, n = 100 LJSpeech, bootstrap 95% CI)

WER ↓ (Whisper-large-v3) · SIM ↑ (ECAPA-TDNN) · UTMOS ↑ (naturalness MOS predictor)

| Arm | WER ↓ | SIM ↑ | UTMOS ↑ | n | Verdict |
|---|---|---|---|---|---|
| `base` (dots.tts) | 0.112 | 0.878 | 4.278 | 100 | reference |
| **`open_soar`** | 0.094 | 0.850 | **4.380** | 100 | **only arm that improves** — best UTMOS, WER ≈ base |
| `dpo_weronly` | 0.110 | 0.856 | 4.055 | 100 | ≈ base (WER-only, mild) |
| `sft` (on winners) | **0.556** | 0.629 | 3.788 | 100 | **collapse** — overturns the "safe control" |
| `dpo_multiobj` | **0.890** | 0.341 | 2.719 | **75** | **catastrophic collapse + 25% generation failure** |

> `dpo_multiobj` failed to generate on **25/100** prompts (immediate-EOS); its WER is over the 75 that synthesized — the **failure rate is reported alongside the WER**, never hidden by it.

### 2. The collapse is monotone in the SFT target's "off-manifold-ness" (controlled gradient)

Same data size, pipeline, and step budget — only the target changes:

| SFT target | WER ↓ | UTMOS ↑ | |
|---|---|---|---|
| `sft_groundtruth` — **real recording (on-manifold)** | **0.087** | 4.337 | **best, beats base** |
| `sft_random` — random candidate (mid) | 0.138 | 4.035 | mild |
| `sft` — reward-*winner* (most off-manifold) | 0.556 | 3.788 | collapse |
| `sft_losers` — reward-*loser* (off-manifold) | 0.569 | 3.952 | collapse |

→ The poison is the **target distribution (off-manifold), not the DPO contrast** — even plain SFT collapses when pointed off-manifold. This resolves the most-cited reviewer confound (small-data SFT instability): at fixed data size, the *on-manifold* target does **not** collapse.

### 3. The mechanism — a training-free likelihood↔reward probe (the core finding)

Over all **N = 188** preference pairs, I measure the **denoised flow-matching margin** (`L_FM(loser) − L_FM(winner)`, positive iff the winner is the better fit as DPO-FM assumes) against the reward gap:

- **Headline (settings-matched): Spearman ρ = −0.216** (ρ² ≈ 4.7%), 95% CI excludes 0; **strengthens** on high-quality-gap strata (top-50% −0.318, top-25% −0.533).
- Reported alongside (not headlined): **Pearson −0.489**, explicitly flagged as a *leverage-amplified* linear coefficient — deleting one high-leverage pair flips its sign, while the rank statistic is stable. Leading with the robust statistic is a deliberate choice.
- **Every reward axis anti-correlates** in the reward-aligned direction: −WER −0.52, SIM −0.77, UTMOS −0.72. SIM/UTMOS are *stronger* than WER → you can't "fix the reward weights," you must abandon `L_FM`-as-reward.
- **Conclusion:** `L_FM` is a valid *likelihood* surrogate but **not a valid implicit reward** — a boundary of the method itself.

### 4. Cross-architecture replication (closes the "N = 1" generalization concern)

The *same* dots-generated preference pairs, scored by **VoxCPM2's own** training FM loss (independent group; diffusion-AR vs. dots' flow-matching):

| Design | Spearman ρ | |
|---|---|---|
| Confounded grid pool (settings vary) | −0.087 (n.s.) | confound *masks* the effect — reported as provenance |
| **Settings-matched** (setting-gap ≡ 0) | **−0.266** (p_perm = 3×10⁻⁴) | effect **returns and is significant** |

→ Architecture-level across **two independent continuous-latent implementations**, not a dots.tts idiosyncrasy. Scope held honestly to N = 2 — *not* "all TTS."

### 5. Streaming reality check

Under production 1T1A streaming: **time-to-first-audio is flat (~162 ms across all arms)** — alignment adds no latency. Decisively, the collapsed arms' offline failure **does not transfer** — they stream full n = 100 normal-speech utterances at streaming WER 0.212 / 0.134 and **UTMOS 4.283 / 4.406 ≈ base 4.326** — confirming the collapse is an **offline-decode-path artifact**, not a damaged model.

### 6. Honest negative: reward hacking *not* reproduced

The pre-registered WER-only-vs-multi-objective reward-hacking comparison **did not reproduce hacking** — neither WER-only arm attains the lowest WER. Reported as a negative result rather than buried, because the mis-specified objective never reaches the regime where hacking would appear.

---

## What makes this *research*, not just a run

- **Falsifiable framing.** A single load-bearing premise, stated up front, tested directly, and **refuted** — the paper is built around the refutation.
- **Statistical rigor.** Bootstrap 95% CIs on every delta; **Spearman led over leverage-inflated Pearson**; permutation tests respecting speaker clustering; disattenuation showing the reported coefficient is a *conservative lower bound*; pre-specified gap strata as a robustness check.
- **Confound elimination by design.** The headline isn't a partial correlation — it's a **settings-matched probe** that pairs winner/loser at identical generation settings, removing the entire sampling grid *by construction*.
- **Scientific honesty.** A retracted over-claim (an n = 24 pilot that regressed to chance at scale, reported openly); scope capped at "two architectures, not all TTS"; the model explicitly **not** called "broken" once streaming evidence arrived.
- **Adversarial self-review.** A 12-perspective internal red-team ([`docs/02-DECISIONS.md`](docs/02-DECISIONS.md)) and a verified, curl-checked bibliography ([`refs.bib`](refs.bib) — 34 entries, every arXiv id title-matched).
- **Full ownership of the stack**, from corpus prep to the final write-up.

---

## Pipeline & repository

```
data → candidates → reward scoring → preference pairs → alignment → evaluation → analysis
```

| Path | What's there |
|---|---|
| [`alignment/`](alignment/) | DPO-FM objective, open-SOAR, reward-weighted training; the `L_FM`↔reward probes (dots.tts + VoxCPM2), permutation/leverage analysis |
| [`reward/`](reward/) | WER / speaker-SIM / UTMOS scorers (batched, cached) |
| [`pairs/`](pairs/) | best-worst preference-pair construction, on-manifold filtering, settings-matched pairing |
| [`candidates/`](candidates/) | candidate generation over the sampling grid |
| [`inference/`](inference/) | offline + 1T1A streaming synthesis |
| [`ablation/`](ablation/) · [`results/`](results/) | experiment matrix, aggregation, plots |
| `run_x{4,5,6,7,8,10,14}.sh` | one-command experiment wrappers (self-check → run → analyze → archive) |
| [`docs/`](docs/) | findings, decision log (ADRs), result tables, pitfalls ledger, citation/arXiv checklist, archived project record |
| [`PAPER_DRAFT.md`](PAPER_DRAFT.md) | the full write-up |

### Reproduce

```bash
bash env/setup_env.sh                 # clone dots.tts/VoxCPM2, install deps, fetch SV checkpoint
bash run_full_dataprep.sh             # corpus → candidates → reward → preference pairs
bash run_train_all.sh 0 open_soar dpo_multiobj dpo_weronly sft   # train arms (GPU)
python alignment/probe_fm_reward.py --pairs pairs/out/pairs_best_worst.jsonl --n 188 --k 16
python results/aggregate_results.py   # Tables 1–3
```

Requires an NVIDIA GPU for training/eval; the probes are forward-only.

### Data availability

Code, analysis scripts, and all result **tables/numbers** (in [`docs/`](docs/) and [`PAPER_DRAFT.md`](PAPER_DRAFT.md)) are version-controlled. Large or trivially regenerable artifacts — model checkpoints, generated audio, per-pair CSVs, logs — are **git-ignored by design** and reproduced by the scripts above; the raw run outputs live on the GPU host.

---

## About

Independent research on preference alignment for continuous-latent speech generation — spanning modern TTS architectures (flow-matching & diffusion-AR), DPO/RLHF theory, and rigorous speech evaluation. Designed, built, and analyzed end-to-end.

*Prepared as a portfolio piece for an AI Research Scientist application. The full Chinese project record (experiment narrative and decision history) is archived at [`docs/10-项目说明-原始详细中文版-存档.md`](docs/10-项目说明-原始详细中文版-存档.md). Questions welcome via Issues.*
