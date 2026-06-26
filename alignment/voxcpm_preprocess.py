"""
voxcpm_preprocess.py — 唯一对接 VoxCPM(VoxCPM2) 训练侧的「适配器」(seam)，dots_preprocess 的架构级镜像

做什么:
    把 (fid, text, audio) 处理成 VoxCPM2 **可前向的训练输入**，并从 `VoxCPM2Model.forward`
    取每条样本的 **diffusion velocity-MSE**(`outputs["loss/diff"]`)。该量与 dots.tts 的
    flow-matching `fm_loss` **同构**(都是「速度场 MSE 的 mask 平均」)，是把 FM-reward 反相关探针
    从 dots.tts(N=1) 泛化到第二个连续-latent TTS 的接缝。所有与 VoxCPM 内部的耦合都收敛到这一个文件。

怎么做(已对照源码核验，复用 VoxCPM 真实训练组件；含命令+输入输出):
    复用链路(全部来自 `参考代码/VoxCPM/src/voxcpm/`):
      - text → ids:  `model.text_tokenizer(text)` → list[int]
            (核验: scripts/train_voxcpm_finetune.py:119 `text_ids=[tokenizer(t) ...]`；
             model/voxcpm2.py 推理侧亦 `torch.LongTensor(self.text_tokenizer(text))`)
      - audio → 16k 波形 → VAE latent → text/audio 交错的训练张量:
            **真实组件** `AudioFeatureProcessingPacker`(training/packers.py) —— 它做
            VAE 编码(`encode_audio`)、patch 化、`process_tts_data` 拼 `[text,101,audio,102]`、
            产出 text_tokens/text_mask/audio_feats/audio_mask/loss_mask/position_ids/labels。
      - collate: 复刻 `HFVoxCPMDataset.collate_fn`(training/data.py:166) 的极小逻辑(pad+stack)，
            避免引入 `datasets`/`argbind` 重依赖(它们只在 data.py 顶层 import)。
      - 前向: `model(text_tokens,text_mask,audio_feats,audio_mask,loss_mask,position_ids,labels,
              progress=0.0)` —— 入参顺序与 train_voxcpm_finetune.py:294-303 完全一致，
              返回 dict，`outputs["loss/diff"]` = `UnifiedCFM.compute_loss` 的 mask-平均速度 MSE。
    输入: pairs jsonl 的一行里的 (fid,text,audio_path)。
    输出: process()→单样本 dict；collate()→batch；prepare_and_forward()→outputs(含 loss/diff)；
          diff_loss_per_sample(outputs)→(1,) 标量(仅 batch=1 成立，见下)。
    命令: 不单独运行；被 `alignment/probe_fm_reward_voxcpm.py` import。

为什么这么设计:
    复用 VoxCPM 自己的 packer，保证喂给 `forward` 的张量与它训练时**完全同构**——这样测到的
    `loss/diff` 才真的是「VoxCPM 在该 (text,audio) 上的训练 FM 损失」，与 dots.tts 的 `fm_loss`
    可在「同一种隐式奖励代理」的意义下横向对比。把耦合收敛到单一 seam，便于 win/lose 对齐。

对 claim 的意义(解审稿 EIC CRITICAL #3：N=1 模型不能泛到整类):
    若 VoxCPM 上 FM-reward 反相关**复现** → 结论升到 **continuous-latent TTS 架构级**
    (dots.tts=flow-matching、VoxCPM=diffusion，两个独立实现都成立)；
    若**不复现** → 把结论**诚实收窄**为 dots.tts 的特性。

前置条件:
    可 import `voxcpm`(把 `参考代码/VoxCPM/src` 放进 sys.path，或设环境变量 VOXCPM_REPO 指向 repo 根)；
    有 `openbmb/VoxCPM2` 权重(运行时 snapshot_download 或本地目录)；CUDA GPU(bf16)。
预期结果:
    process()→ {fid,text_ids,audio_array,...}；collate()→ {text_tokens,audio_tokens,...}；
    prepare_and_forward()→ {"loss/diff":..., "loss/stop":..., "feat_gt":..., "feat_pred":None}。
⚠️ 本机(macOS,无 GPU,未下载权重)只语法检查，不执行 import torch/voxcpm、不跑模型。

────────────────────────────────────────────────────────────────────────────
本文件「不能仅凭代码核验」的点，全部显式标注(仓库纪律)：
  〔需核实 A — VoxCPM 训练侧接口可跑性〕`training/packers.py`/`training/data.py` 是**训练专用代码**，
     本机无 GPU/权重、且 `datasets`/`argbind` 可能未装；packer 在真实 `AudioVAEV2` 权重上的端到端
     行为(VAE 编码维度、patch 对齐)未实跑验证。已用「直接构造 packer + 复刻 collate」最小化依赖。
  〔需核实 B〕`model.text_tokenizer(text)` 返回 list[int] —— 据 train_voxcpm_finetune.py:119 与
     voxcpm2 推理路径推断，未实跑。
  〔需核实 C〕音频按 `model.audio_vae.sample_rate`(=16000) 加载 —— 据 train_voxcpm_finetune.py:105
     `expected_sr=base_model.audio_vae.sample_rate`；packer.encode_audio 假设 wav 已是该 SR。
  〔需核实 D — dit_mean_mode / config.json〕若真实 config 的 `dit_config.dit_mean_mode=True`(MeanFlow)，
     `compute_loss` 走 `torch.func.jvp` 分支，在 `torch.no_grad()` 下能否求值未验证；默认假设 False
     (源码默认 `dit_mean_mode=False`，走普通 `u_pred=model_fn(y,r,t)`)。探针对该量**取值无关**，
     仅读 `loss/diff`，但若 jvp 在 no_grad 报错需改成 enable_grad 测量。
  〔需核实 E〕`loss/diff` == 「per-frame velocity-MSE 的 mask 平均」：成立的前提是
     `UnifiedCFM.adaptive_loss_weighting` 的 `p=0.0`(→ weights==mask，无加权)，
     这是源码里 compute_loss 的调用默认，但真实权重附带的 CfmConfig 未下载核对。
     且**仅 batch=1 时** `loss/diff` 才等于该单样本的标量(B>1 会被聚合成一个数)。
  〔需核实 F〕仅覆盖 `architecture==voxcpm2`；v1 `VoxCPMModel.forward` 接口不同，未适配。
  〔需核实 G〕`training_cfg_rate`(默认 0.1) 会在 ~10% 的 draw 把 `mu` 置零(CFG dropout)——这是
     VoxCPM 自带的训练随机量，与 dots.tts 的随机量不同源；靠「win/lose 同 draw 共享 seed + K 次 MC
     平均」中和(同 draw 内 win/lose 得到同一 cfg_mask)，但它是两边不完全可比的已知差异。
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch  # py_compile 不执行 import，本机无 torch 也能语法检查通过

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402  仅用其 repo_root 定位 VoxCPM；不改 JSONL schema


# --------------------------------------------------------------------------- #
# 让 `import voxcpm` 免 pip 安装即可用(与 common.ensure_dots_tts_importable 同款做法)。
# 〔需核实 A/H〕路径/依赖在本机不可验证。
# --------------------------------------------------------------------------- #
def ensure_voxcpm_importable() -> None:
    """把 VoxCPM 的 `src` 加进 sys.path。

    优先级: 环境变量 VOXCPM_REPO(指向 repo 根，或直接指向 src) → 知识库内 `参考代码/VoxCPM/src`。
    设了 VOXCPM_REPO 但路径不存在 → fail-fast(别静默回退导致后面 import 报莫名错)。
    """
    env = os.environ.get("VOXCPM_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        # 允许指向 repo 根(含 src/) 或直接指向 src/
        src = p / "src" if (p / "src" / "voxcpm").exists() else p
        if not (src / "voxcpm").exists():
            raise FileNotFoundError(f"VOXCPM_REPO 指向的路径下找不到 voxcpm 包: {src}")
    else:
        src = PATHS.repo_root / "参考代码" / "VoxCPM" / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


# --------------------------------------------------------------------------- #
# collate：复刻 HFVoxCPMDataset.collate_fn(training/data.py:166-189) 的最小逻辑。
# 为什么复刻而非 import：data.py 顶层 `import argbind, datasets`(重依赖，且非推理必备)；
# packers.py 则零外部重依赖(只 torch/einops)，故 packer 直接复用、collate 在此复刻。
# 〔需核实 A〕复刻逻辑与上游一致性靠读码保证，未实跑比对。
# --------------------------------------------------------------------------- #
def _pad_sequences(seqs: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    """把变长 1-D 张量 pad 到等长再 stack(等价 HFVoxCPMDataset.pad_sequences)。"""
    if not seqs:
        return torch.empty(0)
    max_len = max(s.shape[0] for s in seqs)
    out = []
    for s in seqs:
        if s.shape[0] < max_len:
            s = torch.nn.functional.pad(s, (0, max_len - s.shape[0]), value=pad_value)
        out.append(s)
    return torch.stack(out)


def voxcpm_collate(batch: list[dict]) -> dict:
    """list[单样本 dict] → batch dict(键同 HFVoxCPMDataset.collate_fn 的 tts 路径)。

    单样本 dict 需含: text_ids(list[int]), audio_array(1-D 波形), dataset_id(int), is_prompt(bool)。
    本探针不用 ref_audio(纯 tts 任务)，故不产 ref_audio_tokens。
    """
    text_tensors = [torch.tensor(s["text_ids"], dtype=torch.int32) for s in batch]
    audio_tensors = [torch.tensor(s["audio_array"], dtype=torch.float32) for s in batch]
    dataset_ids = torch.tensor([int(s.get("dataset_id", 0)) for s in batch], dtype=torch.int32)
    is_prompts = [bool(s.get("is_prompt", False)) for s in batch]

    # pad 哨兵必须是 -100 / -100.0：packer.unpad_* 靠 `==-100` 找首个 pad 位再裁掉。
    text_padded = _pad_sequences(text_tensors, pad_value=-100)
    audio_padded = _pad_sequences(audio_tensors, pad_value=-100.0)
    task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)  # tts 任务 id=1

    return {
        "text_tokens": text_padded,
        "audio_tokens": audio_padded,
        "task_ids": task_ids,
        "dataset_ids": dataset_ids,
        "is_prompts": is_prompts,
    }


def diff_loss_per_sample(outputs: dict) -> torch.Tensor:
    """从 VoxCPM2Model.forward 的输出取每条样本的 diffusion velocity-MSE 标量。

    与 dots.tts 的 `losses_dpo.fm_loss_per_sample` 对位(都返回 (B,))。
    ⚠️〔需核实 E〕VoxCPM 的 `UnifiedCFM.compute_loss` **在内部就把整批帧聚合成一个标量**
    ((weights*losses).sum()/Σmask，weights==mask)，所以 `loss/diff` 只有在 **batch=1** 时
    才等于「该单条样本」的 per-sample 值。本探针严格 batch=1 前向(每次只过一条音频)。
    """
    val = outputs["loss/diff"]
    return val.reshape(1)  # 0-dim 标量 → (1,)，统一成 (B=1,) 形状


class VoxCPMPreprocessor:
    """持有 model + packer，对外暴露 process/collate/prepare_and_forward(镜像 DotsPreprocessor)。"""

    def __init__(self, model, device):
        ensure_voxcpm_importable()
        from voxcpm.training.packers import AudioFeatureProcessingPacker  # type: ignore

        self.model = model
        self.device = device
        # 〔需核实 B〕text_tokenizer 直接来自 VoxCPM2Model；调用约定 list[int]。
        self.tokenizer = model.text_tokenizer
        # 〔需核实 C〕编码 SR = AudioVAE 的 sample_rate(16k)；packer.encode_audio 假定 wav 已是该 SR。
        self.sample_rate = int(model.audio_vae.sample_rate)

        # 防御性把 VAE 放到目标 device(BatchProcessor 也是这么做的)。
        model.audio_vae.to(device)
        # **真实组件**：VoxCPM 训练用的 packer，做 audio→latent / text→cond 的全部拼装。
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=1,
            max_len=int(model.config.max_length),
            patch_size=int(model.config.patch_size),
            feat_dim=int(model.config.feat_dim),
            audio_vae=model.audio_vae,
        )

    @classmethod
    def from_pretrained(cls, ckpt: str = "openbmb/VoxCPM2", device: str | None = None) -> "VoxCPMPreprocessor":
        """加载 VoxCPM2(façade 负责 snapshot_download / 本地目录解析)，取底层 VoxCPM2Model。

        关键开关:
          - load_denoiser=False : 不要 zipenhancer(免 modelscope 依赖；探针不需要降噪)。
          - optimize=False      : 不 torch.compile(免 triton；保证逐次前向在固定 seed 下可复现)。
        〔需核实 F〕只接受 VoxCPM2(architecture=voxcpm2)；façade 会按 config.json 分派，若拿到 v1 报错。
        """
        ensure_voxcpm_importable()
        from voxcpm.core import VoxCPM  # type: ignore
        from voxcpm.model.voxcpm2 import VoxCPM2Model  # type: ignore

        facade = VoxCPM.from_pretrained(
            hf_model_id=ckpt,
            load_denoiser=False,
            optimize=False,
            device=device,
        )
        model = facade.tts_model
        if not isinstance(model, VoxCPM2Model):
            raise TypeError(
                f"voxcpm_preprocess 只适配 VoxCPM2(architecture=voxcpm2)，但加载到 {type(model).__name__}。"
                f"请用 openbmb/VoxCPM2 之类的 v2 权重。〔需核实 F〕"
            )
        model.eval()
        # model.device 是字符串("cuda"/"cpu"/"mps")，from_local 已把权重/缓存放到该 device。
        return cls(model, model.device)

    # --- 单样本 / 成批(只依赖 tokenizer + librosa，可独立)---
    def process(self, fid: str, text: str, audio: str, dataset_id: int = 0) -> dict:
        """(fid,text,audio_path) → 单样本 dict(供 voxcpm_collate)。

        与 dots_preprocess.DotsPreprocessor.process 对位。is_prompt=False ⇒ loss 落在整段目标音频上
        (process_tts_data 的 tts 任务)，与 dots.tts BasicTtsPipeline 的 (text,audio) 无参考续写同构。
        """
        import librosa  # 延迟 import：本机语法检查不触发

        # 〔需核实 C〕按 VAE 的 sample_rate 重采样为单声道波形。
        wav, sr = librosa.load(audio, sr=self.sample_rate, mono=True)
        return {
            "fid": str(fid),                     # 仅溯源用，collate 不读
            "text_ids": list(self.tokenizer(text)),  # 〔需核实 B〕list[int]
            "audio_array": wav,
            "audio_sampling_rate": int(sr),
            "dataset_id": int(dataset_id),
            "is_prompt": False,
        }

    def collate(self, samples: list[dict]) -> dict:
        return voxcpm_collate(samples)

    # --- 前向(在主进程、GPU 上)---
    def _pack(self, batch: dict) -> dict:
        """batch(collate 产物) → packer → 训练张量(镜像 training/data.py:BatchProcessor.__call__)。"""
        dev = self.device
        audio_tokens = batch["audio_tokens"].to(dev)
        text_tokens = batch["text_tokens"].to(dev)
        task_ids = batch["task_ids"].to(dev)
        dataset_ids = batch["dataset_ids"].to(dev)
        ref_audio_tokens = batch["ref_audio_tokens"].to(dev) if "ref_audio_tokens" in batch else None
        return self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
            ref_audio_tokens=ref_audio_tokens,
        )

    def prepare_and_forward(self, batch: dict) -> dict:
        """batch → packer 补 audio_feats/loss_mask/labels → model(...) → outputs(含 loss/diff)。

        入参顺序严格对齐 train_voxcpm_finetune.py:294-303。progress=0.0：对默认(非 mean_mode)配置无影响
        (只改 ratio_r_neq_t/noise_cond，二者在默认 CfmConfig 下为 0)。〔需核实 D/E〕
        (z,t) 的采样发生在 `compute_loss` 内部 → 由调用方(探针)用 fork_rng+manual_seed 固定，
        实现 win/lose 同 draw 共享 (z,t) 的命门。
        """
        packed = self._pack(batch)
        outputs = self.model(
            packed["text_tokens"],
            packed["text_mask"],
            packed["audio_feats"],
            packed["audio_mask"],
            packed["loss_mask"],
            packed["position_ids"],
            packed["labels"],
            progress=0.0,
        )
        return outputs
