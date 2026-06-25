"""
common.py — 全项目共享契约与工具（路径 / JSONL schema / 小工具）

做什么:
    集中定义本实验所有阶段共用的「数据契约」(JSONL 字段)、路径解析、以及
    免安装导入 dots_tts 的 sys.path 垫片。所有脚本都 `from common import ...`，
    保证 data→candidates→reward→pairs→alignment→eval 各阶段的字段不漂移。
怎么做:
    纯标准库实现 (json/os/pathlib/dataclasses)，不在模块顶层 import torch，
    这样在任何机器上都能 `python3 -m py_compile` 通过、也能被轻量脚本导入。
为什么这么设计:
    多阶段管线最易出的 bug 是「上游写的字段名 ≠ 下游读的字段名」。把 schema 收敛到
    一处单一事实源 (single source of truth)，下游全部引用常量，从结构上杜绝漂移。
对实验意义:
    保证可复现与可溯源 —— 每条记录都能从 fid 一路追到最终指标表 (results ledger)。

前置条件: 无 (纯标准库)。
预期结果: import 成功；`PATHS` 指向 dots.tts / seed-tts-eval 的真实位置。
注意: 本机 (macOS, 无 GPU, 未下载权重) 只能语法检查，不能真正跑模型阶段。
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Iterator


# --------------------------------------------------------------------------- #
# 静音「已知良性」warning（详见 docs/05 §C）。每个脚本都 import common，一处生效、全项目干净。
# 原则：只压不影响结果的提示，且 **保留真实下载进度条**（大文件首次下载时需要看到）。
#   - TRANSFORMERS_VERBOSITY=error：压 transformers 的「model type 'dots_tts'」与
#     「tokenizer incorrect regex / fix_mistral_regex」提示（后者**绝不能真去改**，会损坏分词）。
#   - weight_norm FutureWarning：来自 dots.tts 的 BigVGAN vocoder 用了旧 torch API，dots.tts 内部、与我们无关。
# 用 setdefault：用户可用环境变量覆盖（如想看全部 warning，导出 TRANSFORMERS_VERBOSITY=warning）。
# --------------------------------------------------------------------------- #
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=r".*weight_norm.*is deprecated.*")


# --------------------------------------------------------------------------- #
# 路径解析：以本文件位置反推仓库根，允许用环境变量覆盖（方便迁移到 GPU 机器）。
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
# .../TTS/实验项目/dots-tts-preference-alignment/common.py -> parents[2] == .../TTS
_DEFAULT_REPO_ROOT = _THIS.parents[2]


@dataclass(frozen=True)
class _Paths:
    repo_root: Path
    project_root: Path
    dots_tts_repo: Path
    dots_tts_src: Path
    seed_tts_eval: Path

    @property
    def prompts_dir(self) -> Path:
        return self.project_root / "data" / "prompts"

    @property
    def candidates_dir(self) -> Path:
        return self.project_root / "candidates" / "out"

    @property
    def rewards_dir(self) -> Path:
        return self.project_root / "reward" / "out"

    @property
    def pairs_dir(self) -> Path:
        return self.project_root / "pairs" / "out"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / "out"


def _first_existing(candidates: list[Path], default: Path) -> Path:
    """返回第一个真实存在的候选；都不存在则返回 default（供 setup_env 克隆到此）。"""
    for c in candidates:
        if c.exists():
            return c
    return default


def _resolve_paths() -> _Paths:
    """同时支持两种部署:
    (1) 在 TTS 知识库内: dots.tts/seed-tts-eval 在 `../../参考代码/` 下;
    (2) 独立仓库(远程服务器 git pull): 由 env/setup_env.sh 克隆到本项目 `.deps/` 下。
    可用环境变量 DOTS_TTS_REPO / SEED_TTS_EVAL 显式覆盖。
    """
    repo_root = Path(os.environ.get("TTS_REPO_ROOT", _DEFAULT_REPO_ROOT)).resolve()
    project_root = _THIS.parent
    deps = project_root / ".deps"

    dots_repo = _first_existing(
        [Path(os.environ["DOTS_TTS_REPO"]).resolve()] if os.environ.get("DOTS_TTS_REPO") else [],
        Path(),
    ) if os.environ.get("DOTS_TTS_REPO") else _first_existing(
        [repo_root / "参考代码" / "dots.tts", deps / "dots.tts"],
        deps / "dots.tts",
    )
    seed_eval = _first_existing(
        [Path(os.environ["SEED_TTS_EVAL"]).resolve()] if os.environ.get("SEED_TTS_EVAL") else [],
        Path(),
    ) if os.environ.get("SEED_TTS_EVAL") else _first_existing(
        [repo_root / "参考代码" / "seed-tts-eval", deps / "seed-tts-eval"],
        deps / "seed-tts-eval",
    )
    return _Paths(
        repo_root=repo_root,
        project_root=project_root,
        dots_tts_repo=dots_repo,
        dots_tts_src=dots_repo / "src",
        seed_tts_eval=seed_eval,
    )


PATHS = _resolve_paths()


def ensure_dots_tts_importable() -> None:
    """把 dots.tts 的 src 加进 sys.path，免 `pip install` 即可 `import dots_tts`。

    为什么: 与 dots.tts 仓库自带的 `scripts/example_double_streaming.py` 同款做法，
    避免污染全局环境，也让实验文件夹自包含、可整体迁移。
    """
    src = str(PATHS.dots_tts_src)
    if src not in sys.path:
        sys.path.insert(0, src)


# --------------------------------------------------------------------------- #
# 数据契约 (JSONL schema)。每阶段读写都用这些 dataclass，字段名即契约。
# --------------------------------------------------------------------------- #
@dataclass
class PromptRow:
    """一条合成请求（zero-shot 克隆）。data/ 阶段产出。"""

    fid: str                 # 全局唯一 id，贯穿所有阶段，用于溯源
    text: str                # 待合成文本（= 评测时的参考文本）
    prompt_audio: str        # 参考音频路径（音色来源）
    prompt_text: str         # 参考音频转写（dots.tts 离线克隆需要；流式不需要）
    lang: str                # "en" / "zh" / ...
    source: str = ""         # 数据来源语料（vctk / aishell3 / ljspeech）


@dataclass
class CandidateRow:
    """某个 prompt 的第 k 个候选音频 + 其生成参数。candidates/ 阶段产出。"""

    fid: str
    k: int
    audio: str               # 候选 wav 路径 (48kHz)
    text: str
    prompt_audio: str
    prompt_text: str
    lang: str
    num_steps: int           # NFE
    guidance_scale: float    # CFG
    speaker_scale: float
    seed: int


@dataclass
class RewardRow:
    """候选的多目标打分。reward/ 阶段产出。"""

    fid: str
    k: int
    audio: str
    lang: str
    wer: float               # 越低越好
    sim: float               # 越高越好
    utmos: float             # 越高越好
    reward: float            # 聚合奖励 R（z-norm 后加权），越高越好
    # 归一化前的 z 分量，便于事后审计权重影响
    wer_z: float = 0.0
    sim_z: float = 0.0
    utmos_z: float = 0.0


@dataclass
class PairRow:
    """一对偏好样本（共享同一 text+prompt 条件）。pairs/ 阶段产出。"""

    fid: str
    text: str
    prompt_audio: str
    prompt_text: str
    lang: str
    win_audio: str
    lose_audio: str
    reward_win: float
    reward_lose: float
    strategy: str            # best_worst / top2_bottom2 / threshold


# --------------------------------------------------------------------------- #
# JSONL IO（统一编码、统一 dataclass<->dict）。
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = asdict(r) if hasattr(r, "__dataclass_fields__") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def zscore(values: list[float]) -> list[float]:
    """对一组数做 z-normalize；标准差为 0 时返回全 0（避免除零）。

    为什么: WER(毫秒级百分比)、SIM(0~1 余弦)、UTMOS(1~5) 量纲不同，
    直接加权无意义；按「每个 prompt 的候选池」做 z-norm 才可比。
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


__all__ = [
    "PATHS",
    "ensure_dots_tts_importable",
    "PromptRow",
    "CandidateRow",
    "RewardRow",
    "PairRow",
    "write_jsonl",
    "read_jsonl",
    "zscore",
]
