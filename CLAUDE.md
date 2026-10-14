# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **self-directed research project**, not a product or library: preference alignment for **dots.tts** (a 2B continuous-latent flow-matching TTS — *no discrete audio tokens*). The headline deliverable is a **falsifiable negative/diagnostic result**, currently being written up as a paper. The empirical claim: the natural way to port DPO here (substitute the flow-matching loss `L_FM` as a likelihood surrogate, à la Diffusion-DPO → "**DPO-FM**") rests on one premise — that an automatic-metric reward agrees with the model's own flow likelihood — and that premise is **false** (the reward anti-correlates with `L_FM`), so the alignment optimizes in the wrong direction and the model collapses offline.

It is **not a pip-installable package**. Scripts run as files and share state via `sys.path.insert(...)` + `from common import ...`. `pyproject.toml` exists only to install *incremental* dependencies on top of an already-installed dots.tts (it declares `packages=[]`, `py-modules=[]` on purpose).

## Critical operational constraints

- **This host (macOS, no GPU, dots.tts weights not downloaded) can only syntax-check.** Real model/training/eval stages run on a **remote GPU machine**. Never claim a script was run or verified end-to-end here — validate with `py_compile` / `bash -n` only, and say so.
- `common.py` deliberately **does not import torch at module top level** so every file compiles on this CPU-only host. Keep new shared utilities torch-free; put torch imports inside functions or in the stage scripts.
- **Experiment artifacts are gitignored by design** (`candidates/out/`, `reward/out/`, `pairs/out/`, `alignment/out/`, `results/out/`, `inference/out*/`, `ablation/ledger.jsonl`, `*.wav`, `*.log`, downloaded corpora/weights). Results on the remote machine are **not** synced to GitHub — do not assume a result is committed because a run "finished."
- **The paper `.tex` lives OUTSIDE this repo** (local, non-git, compiled on Overleaf — no LaTeX on this host). In-repo paper sources are `PAPER_DRAFT.md` (markdown) and `refs.bib` (verified bibliography). `docs/12` is the single source of truth for the review-driven supplementary experiments.

## Commands

```bash
# The only "test" available on this host — syntax-check everything, then clean bytecode
python3 -m py_compile common.py utmos_backend.py */*.py
bash -n run_*.sh
rm -rf __pycache__ */__pycache__

# GPU machine: one-time environment (idempotent; clones dots.tts -e, installs incremental
# deps, mounts VoxCPM via .pth, downloads WavLM-SV). dots.tts/seed-tts-eval must be clonable.
bash env/setup_env.sh

# Full pipeline on the GPU machine (each is self-contained; run under tmux)
bash run_full_dataprep.sh                 # data → candidates → reward → pairs
bash run_train_all.sh <gpu> [arms...]     # train treatments (dispatches arms; splittable across 2 cards)
```

`run_x{N}.sh` and `run_e*/run_dose.sh` are **one-command, self-contained experiments** (self-check → run → aggregate → timestamped archive under `results/out/<exp>_<stamp>/`, with `--out`/`tee`). Pre-flight gate before any DPO training: `alignment/check_fm_loss.py` (verifies the FM seam + shared-RNG determinism). The probe/analysis re-runs (`alignment/probe_analysis.py`, `eval/dose_response.py`, `eval/analyze_collapse_blocks.py`) are **CPU-only and cheap** — they re-analyze existing artifacts and can be iterated freely.

## Architecture

**One linear, fid-traceable pipeline** (the contract is `PROTOCOL.md` + `common.py`):

```
data/prompts → candidates/ → reward/ → pairs/ → alignment/ → inference/ → eval/ → results/ → docs+paper
 (PromptRow)   (CandidateRow) (RewardRow) (PairRow)  (checkpoints)  (wavs)   (metrics) (ledger)
```

Every record carries a global **`fid`** so any final metric traces back to its prompt. The JSONL field schema for every stage is centralized as frozen dataclasses in `common.py` (`PromptRow`/`CandidateRow`/`RewardRow`/`PairRow`) — the single source of truth that prevents upstream-writes-X / downstream-reads-Y drift. Downstream stages reference those constants, never ad-hoc strings.

Load-bearing structural facts (require reading several files to see):

- **`common.py` is the shared contract.** `PATHS` resolves the dots.tts / seed-tts-eval / VoxCPM locations relative to the repo root, **env-overridable** for the GPU machine. It also provides `read_jsonl`/`write_jsonl`, `zscore`, `ensure_dots_tts_importable()` (the import shim), benign-warning suppression, and `file_lock` — a cross-process `flock` that serializes the *fast* critical sections (ledger write / main-table aggregation) so `x4/x6/x7/x8` can run in parallel on two cards without corrupting `ablation/ledger.jsonl`.
- **Seams isolate the external models.** `alignment/dots_preprocess.py` is the **only** file that touches dots.tts internal preprocessing; `alignment/voxcpm_preprocess.py` is its deliberate architecture-level mirror for VoxCPM2. This single-adapter design is what makes the cross-architecture replication (X10) clean — keep model-specific coupling inside these two files.
- **DPO-FM core** is `alignment/losses_dpo.py`: Diffusion-DPO adapted to flow-matching as four velocity-MSE passes (policy/ref × winner/loser) that **share one `(x0, t)` draw per step** (the "shared-noise invariant" — without it the loss conflates preference signal with sampling variance). Training arms: `train_dots_tts_dpo.py` (DPO), `train_open_soar.py` (reward-free self-correction toward real targets — the one arm that improves), `train_sft.py` (SFT-on-winners control), `train_reward_weighted.py` (non-DPO reward-weighted regression). Arms are dispatched by `run_train_all.sh` / `ablation/run_matrix.py`'s `TREATMENTS` map.
- **The probe is the central diagnosis, separate from training** — `probe_fm_reward.py` (training-free `no_grad`: correlate the denoised FM margin against the reward gap over the 188 pairs), `probe_fm_reward_voxcpm.py` (cross-arch), `probe_analysis.py` (statistical hardening: settings-matched/partial correlations, selection/order-statistic null controls), `probe_lfm_validity.py` (typicality). These are cheap and rerunnable; most reviewer responses are probe re-analyses, not retraining.
- **Metric backends are single-sourced and shared by reward *and* eval.** `utmos_backend.py` loads UTMOS via `torch.hub` (tarepan/SpeechMOS `utmos22_strong`) — **not** the PyPI `speechmos` package (that is Microsoft DNSMOS/PLCMOS, used only by the `★2` cross-check). `reward/score_{wer,sim,utmos}.py` produce the reward; `eval/score_synth.py` reuses the same backends so reward and eval scores are commensurable. `eval/streaming_metrics.py` covers the 1T1A streaming path (TTFA, chunk-boundary discontinuity).

## Conventions

- **Experiment codes are load-bearing names.** `run_x{N}.sh` = the experiment plan's X-codes (X4 SFT-target gradient, X5 settings-matched re-probe, X6 constructive routes, X7 reward-hacking, X8 streaming, X10 VoxCPM cross-arch, X14 typicality anchor). `run_e1/e2/e3.sh`, `run_e2b.sh` (★2), `run_dose.sh` (★1) = the review-driven supplementary experiments. Keep these codes consistent across `PROTOCOL.md`, `docs/`, and the wrapper names.
- **`docs/` are numbered single-source-of-truth files**, not scratch: `PROTOCOL.md` ("what to run, vs what, report what, what wins"), `docs/02-DECISIONS.md` (decision log + 12-perspective red-team — consult before reversing a design choice), `docs/05` (pitfalls & fixes), `docs/06` (the core finding), `docs/08` (Table 1), `docs/12` (review-driven supplementary). Editing one stage's structure means updating the docs and protocol that reference it.
- **Every script header documents 做什么 / 怎么做 / 为什么 / 对实验意义 / 前置条件 / 预期结果 / 注意** (what / how / why / meaning / prerequisites / expected result / caveats). Match this when adding scripts. Prose is **Chinese-primary with English technical terms**.
- **Always persist results durably** (the `run_x*.sh` pattern): pass `--out`/`--out-csv`, `tee` stdout to a log, and copy into a dated `results/out/<exp>_<stamp>/` archive. A run with no saved artifact did not happen.
- **External facts are verified, not assumed** (arXiv ids / hub URLs in `refs.bib` and docs were each confirmed). Tag any community/estimated number as such; never present it as official.

## Load-bearing findings (so you don't re-derive them)

- Headline: settings-matched **Spearman ρ = −0.216** (N = 188), reward gap anti-correlated with the flow-likelihood margin, strengthening on high-quality-gap strata. Lead the **rank** statistic; Pearson (−0.489) is reported only as a leverage-amplified coefficient (sign flips on one deleted pair).
- This is **not** a broken `L_FM` (open-SOAR optimizes it to saturation) — it is a failure of the **preference↔likelihood coupling**. The anti-correlation is an **extreme-contrast, reward-specific** effect (E1 Null A `p=0.004`), not a smooth content-level coupling.
- Offline **collapse is decode-path-specific**: the collapsed checkpoints **stream normal speech** (1T1A) — a generation-time EOS/stopping artifact, not a broken model.
- **Two independent levers rescue the collapse**: a *non-likelihood* reward-weighted objective, **or** a real on-manifold target (SFT-on-recordings). Collapse needs *both* a likelihood/imitation objective *and* an off-manifold target.
- Cross-architecture (VoxCPM2, diffusion-AR, ρ = −0.266) scores the **same** pairs → **supporting, statistically-dependent** evidence, *not* an independent N=2 replication. Scope is **continuous-latent** TTS, not all TTS and not discrete-token.
