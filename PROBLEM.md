# Research question and verification

> **With the same data and training budget, can preference training built on a flow-matching (FM) surrogate beat winner-only SFT in an independent quality evaluation, and keep its gains when generation conditions change?**

**Why it matters:** Post-training a continuous-latent speech model has to turn preference signals into improvements you can hear. A falling training loss or a rising automatic score does not by itself show that the method suits real synthesis.

**Central tension:** Automatic metrics, the training surrogate and true generation quality can disagree, and different prompts and decoding settings can change the degradation observed. FM loss is not automatically the true likelihood, and its correlation with the reward cannot on its own prove the causal mechanism behind a training failure.

**Current evidence and gap:** The [README](README.md) and the [experiment docs](docs/) describe the multi-arm training, the probes and the offline and streaming observations. Existing results should be read under their original conditions; the separate roles of the generation source, the prompt conditions and the surrogate still need controlled experiments.

**Decisive test:** Fix data, compute and evaluation rules, and compare SFT on real recordings, winner-only SFT and preference training. Match inference conditions wherever they can be matched, and use independent evaluation, blind listening and several seeds. Check the scope with pairs a second model generates itself. If the gains disappear once conditions are matched, or plain SFT works just as well, narrow the claims about the mechanism and the method.

This section defines questions to be tested; it adds no experimental results and does not change the existing paper or history.

中文版：[PROBLEM.zh.md](PROBLEM.zh.md)
