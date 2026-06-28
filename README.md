# When Likelihood Disagrees with Quality

### A cautionary diagnosis of Direct Preference Optimization for continuous-latent flow-matching TTS

**A self-directed research project** that adapts DPO to a modern continuous-latent text-to-speech model (dots.tts), then shows — with a direct, training-free probe — *why the natural adaptation is the wrong tool*, and which alignment routes actually work. The result is a **falsifiable negative/diagnostic finding** backed by a held-out benchmark, a mechanistic probe, and a cross-architecture replication.

> 📄 Full paper: [`PAPER_DRAFT.md`](PAPER_DRAFT.md) · 📊 Results & decisions: [`docs/`](docs/) · 📚 Verified bibliography: [`refs.bib`](refs.bib) · 🇨🇳 中文版: [`README.zh-CN.md`](README.zh-CN.md)
>
> License: **Apache-2.0** (code) · Models studied: dots.tts (2B, Apache-2.0) + VoxCPM2 (2B, OpenBMB)

---

## TL;DR

Continuous-latent TTS (flow-matching over continuous audio latents — **no discrete audio tokens**) has no native way to do preference alignment, because DPO needs token log-probabilities the model never produces. The *natural* fix, **DPO-FM**, substitutes the flow-matching loss `L_FM` as a likelihood surrogate (following Diffusion-DPO). I tested its one load-bearing assumption — *does an automatic-metric reward prefer the candidate the model finds more likely?* — and found the **opposite**:

- **An automatic-metric reward preference is monotonically *anti-correlated* with the model's own flow likelihood** (settings-matched Spearman **ρ = −0.216**, N = 188, strengthening on the highest-quality-gap pairs). The reward systematically prefers the **off-manifold** member of each pair.
- This is **not** a failure of `L_FM` as a likelihood proxy (a likelihood-aligned preference optimizes it to saturation) — it's a failure of the **reward↔likelihood coupling**.
- On a 100-utterance held-out set, **only the likelihood-aligned, on-manifold route improves**; every arm trained toward reward-selected self-generated targets — DPO-FM *and* plain SFT-on-winners — **collapses**.
- The collapse is **decode-path-specific**: the same checkpoints **stream perfectly normal speech** under production (1T1A) decoding — so it's a generation-time artifact, not a broken model.
- The diagnosis **replicates confound-free on a second, independently-built architecture** (VoxCPM2, diffusion-AR; settings-matched ρ = −0.266) → architecture-level evidence across **N = 2** continuous-latent implementations.

**One sentence:** *the intuitive "generate candidates → score with automatic metrics → DPO toward the winners" recipe degrades continuous-latent TTS, and here is exactly why, where, and what works instead.*

---

## Why this matters

Preference alignment (RLHF/DPO) is the default post-training tool for LLMs, and the obvious move is to port it to speech. This project shows that the obvious port has a **silent failure mode** that standard metrics hide: training "succeeds" (loss doesn't diverge), yet the model degrades at generation time on a subset of inputs. The contribution is a **mechanism + a diagnostic you can run before spending a GPU-week on alignment** — measure the sign of the likelihood↔reward coupling on held-out pairs first.

For production speech systems the practical takeaways are concrete:
- **Reward design ≠ reward optimization.** When the reward prefers off-manifold (e.g. over-guided) samples, *no reweighting of WER/speaker/UTMOS terms fixes it* — all three axes anti-correlate with likelihood.
- **Offline eval can lie about streaming behavior.** A catastrophic offline collapse here did *not* transfer to per-token streaming decoding — the deployment-relevant path was fine. Evaluating only one decode path would have produced the wrong conclusion in both directions.
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

### 2. The collapse is monotone in "off-manifold-ness" of the SFT target (controlled gradient, X4)

Same data size, pipeline, and step budget — only the target changes:

| SFT target | WER ↓ | UTMOS ↑ | |
|---|---|---|---|
| `sft_groundtruth` — **real recording (on-manifold)** | **0.087** | 4.337 | **best, beats base** |
| `sft_random` — random candidate (mid) | 0.138 | 4.035 | mild |
| `sft` — reward-*winner* (most off-manifold) | 0.556 | 3.788 | collapse |
| `sft_losers` — reward-*loser* (off-manifold) | 0.569 | 3.952 | collapse |

→ The poison is the **target distribution (off-manifold), not the DPO contrast** — even plain supervised fine-tuning collapses when pointed off-manifold. This resolves the most-cited reviewer confound (small-data SFT instability): at fixed data size, the *on-manifold* SFT target does **not** collapse.

### 3. The mechanism — a training-free likelihood↔reward probe (the core finding)

Over all **N = 188** preference pairs, I measure the **denoised flow-matching margin** (`L_FM(loser) − L_FM(winner)`, which is positive iff the winner is the better fit, as DPO-FM assumes) against the reward gap:

- **Headline (settings-matched): Spearman ρ = −0.216** (ρ² ≈ 4.7%), 95% CI excludes 0; **strengthens** on high-quality-gap strata (top-50% −0.318, top-25% −0.533).
- Reported alongside (not as the headline): **Pearson −0.489**, explicitly flagged as a *leverage-amplified* linear coefficient — deleting one high-leverage pair flips its sign, while the rank statistic is stable. (Leading with the robust statistic is a deliberate methodological choice.)
- **Every reward axis anti-correlates** in the reward-aligned direction: −WER −0.52, SIM −0.77, UTMOS −0.72. SIM/UTMOS are *stronger* than WER → you cannot "fix the reward weights," you must abandon `L_FM`-as-reward.
- **Conclusion:** `L_FM` is a perfectly valid *likelihood* surrogate, but **not a valid implicit reward** — a boundary of the method itself.

### 4. Cross-architecture replication (closes the "N = 1" generalization concern, X10)

The *same* dots-generated preference pairs, scored by **VoxCPM2's own** training FM loss (an independent group, diffusion-AR vs. dots' flow-matching):

| Design | Spearman ρ | |
|---|---|---|
| Confounded grid pool (settings vary) | −0.087 (n.s.) | confound *masks* the effect — reported as provenance |
| **Settings-matched** (setting-gap ≡ 0) | **−0.266** (p_perm = 3×10⁻⁴) | effect **returns and is significant** |

→ The phenomenon is **architecture-level across two independent continuous-latent implementations**, not a dots.tts idiosyncrasy. Scope is held honestly to N = 2 — *not* "all TTS."

### 5. Streaming reality check (X8)

Under production 1T1A streaming decoding: **time-to-first-audio is flat (~162 ms across all arms)** — alignment adds no latency. Decisively, the collapsed arms' offline failure **does not transfer**: they stream full n = 100 normal-speech utterances at streaming WER 0.212 / 0.134 and **UTMOS 4.283 / 4.406 ≈ base 4.326** — confirming the collapse is an **offline-decode-path artifact**, not a damaged model.

### 6. Honest negative: reward hacking *not* reproduced (X7)

The pre-registered WER-only-vs-multi-objective reward-hacking comparison **did not reproduce hacking** — neither WER-only arm attains the lowest WER. Reported as a negative result rather than buried, because the mis-specified objective never reaches the regime where hacking would appear.

---

## What makes this *research*, not just a run

The project was executed end-to-end with the discipline I'd bring to a research role:

- **Falsifiable framing.** A single load-bearing premise, stated up front, tested directly, and **refuted** — the paper is organized around that refutation rather than a result it was hoping for.
- **Statistical rigor.** Bootstrap 95% CIs on every delta; **Spearman led over leverage-inflated Pearson**; permutation tests respecting speaker clustering; disattenuation showing the reported coefficient is a *conservative lower bound*; pre-specified gap strata as a reviewer-style robustness check.
- **Confound elimination by design.** The headline isn't a partial correlation — it's a **settings-matched probe** that pairs winner/loser at identical generation settings, removing the entire sampling grid *by construction*.
- **Scientific honesty.** A retracted over-claim (an n = 24 pilot that regressed to chance at scale, reported openly); scope capped at "two architectures, not all TTS"; the model is explicitly **not** called "broken" once streaming evidence came in.
- **Adversarial self-review.** A 12-perspective internal red-team ([`docs/02-DECISIONS.md`](docs/02-DECISIONS.md)) plus a verified, curl-checked bibliography ([`refs.bib`](refs.bib), 34 entries, every arXiv id title-matched).
- **Full ownership of the stack** — data prep → candidate generation → reward scoring → preference-pair construction → four training objectives → offline + streaming evaluation → statistical analysis → paper.

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
| [`docs/`](docs/) | findings, decision log (ADRs), result tables, pitfalls ledger, citation/arXiv checklist |
| [`PAPER_DRAFT.md`](PAPER_DRAFT.md) | the full write-up |

### Reproduce

```bash
bash env/setup_env.sh                 # clone dots.tts/VoxCPM2, install deps, fetch SV checkpoint
bash run_full_dataprep.sh             # corpus → candidates → reward → preference pairs
bash run_train_all.sh 0 open_soar dpo_multiobj dpo_weronly sft   # train arms (GPU)
python alignment/probe_fm_reward.py --pairs pairs/out/pairs_best_worst.jsonl --n 188 --k 16
python results/aggregate_results.py   # Tables 1–3
```

Requires an NVIDIA GPU (training/eval); the probes are forward-only.

### Data availability

Code, analysis scripts, and all result **tables/numbers** (in [`docs/`](docs/) and [`PAPER_DRAFT.md`](PAPER_DRAFT.md)) are version-controlled. Large or trivially regenerable artifacts — model checkpoints, generated audio, per-pair CSVs, logs — are **git-ignored by design** and reproduced by the scripts above; the raw run outputs live on the GPU host.

---

## About

Independent research project on preference alignment for continuous-latent speech generation — spanning modern TTS architectures (flow-matching & diffusion-AR), DPO/RLHF theory, and rigorous speech evaluation. Built and analyzed end-to-end by one researcher.

*Prepared as a portfolio piece for an AI Research Scientist application. Questions and discussion welcome via Issues.*
