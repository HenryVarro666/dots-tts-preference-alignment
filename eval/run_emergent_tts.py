"""
run_emergent_tts.py — 评测：EmergentTTS-Eval 表现力胜率（成对 A/B LLM-judge）

做什么:
    在 EmergentTTS-Eval(1645 条，6 类难例)上跑某 treatment（被测系统 DUT），与固定参照
    (gpt-4o-mini-tts)做 head-to-head：同一段文本各合成一条音频，交 LLM-judge 二选一，
    统计**每类目胜率 + overall 胜率 + 左右序 swap 一致率**，写一份可被 aggregate 读的 json。
怎么做（命令 + 输入输出）:
    命令:  python run_emergent_tts.py --ckpt <dots.tts-ckpt> --treatment dpo_multiobj \
                                      [--judge gemini-2.5-pro] [--reference gpt-4o-mini-tts] \
                                      [--limit N] [--seed 12345] [--dry-run]
    输入:  ① dots.tts 权重(--ckpt) 作为 DUT 后端；② 已 clone 的 EmergentTTS-Eval-public
           (--emergent-root，arXiv 2505.23009) 提供 6 类评测文本；③ judge / 参照 TTS 的 API key。
    输出:  results/out/{treatment}.emergent.json
           = {treatment, judge, reference, n, categories:{<cat>:win_rate}, overall,
              swap_consistency, note}。aggregate_results.py 读其 'overall'（并预留 categories hook）。
    --dry-run: 不连任何外部依赖，只落「全 6 类 + 字段齐全、值为 null」的模板，
               用于在本机(无 GPU/无权重/无 key)核验输出 schema 与 aggregate 对接。
为什么:
    WER/SIM/UTMOS 测不出情感/副语言/句法复杂度等**表现力**；EmergentTTS-Eval 是少数第三方
    表现力基准，且其官方协议本就是「对固定参照的成对 LLM-judge 偏好」——本脚本复刻该协议。
对实验意义（claim）:
    ⚠️ 关键预期(docs/05 line 325)：SOAR/对齐可能在表现力上有 tradeoff（dots.tts 自身 SOAR
    把 Emotions 72.7→63.9）。本列用于检验对齐是否以牺牲表现力换鲁棒——RQ3 的另一面、Table 3 证据。
    胜率是「对固定参照的 LLM-judge 偏好」，**非绝对分**；swap 一致率低则该胜率不可信(judge 有位置偏好)。

前置条件:
    - `git clone https://github.com/boson-ai/EmergentTTS-Eval-public <--emergent-root>`（arXiv 2505.23009）。
    - judge API key（如 GEMINI_API_KEY）+ 参照 TTS API key（如 OPENAI_API_KEY，gpt-4o-mini-tts）。
    - dots.tts 权重与 runtime（离线合成 DUT）。
预期结果: results/out/{treatment}.emergent.json（6 类胜率 + overall + swap 一致率）。

⚠️ 本机只语法检查（macOS / 无 GPU / 未下载权重 / 无 API key）：真实胜率必须在 GPU + 有 key 的机器上跑。
⚠️〔需核实〕标注的三处外部接线（EmergentTTS 数据格式 / judge API / 参照 TTS API）无法仅凭本仓代码核验，
   均以 NotImplementedError 显式抛出，并写明「该填什么」——补全后即为真实可跑路径（仓库纪律）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PATHS  # noqa: E402

# EmergentTTS-Eval 的 6 个难例类目（键名即输出 schema 的 categories 键，勿改顺序/拼写）。
CATEGORIES: list[str] = [
    "Emotions",
    "Paralinguistics",
    "SyntacticComplexity",
    "ForeignWords",
    "ComplexPronunciation",
    "QuestionsErrors",
]

# 每类目「该听什么」——注入 judge prompt，让偏好判断聚焦该类目的表现力维度，而非泛泛比好听。
CATEGORY_GUIDANCE: dict[str, str] = {
    "Emotions": "目标情绪（喜怒哀惧等）是否被准确、自然地传达；语调/能量是否贴合语义。",
    "Paralinguistics": "副语言事件（笑声、叹气、哭腔、耳语、强调、停顿）是否被恰当地实现而非读字面。",
    "SyntacticComplexity": "长难句/嵌套从句下的韵律分组、断句与重音是否正确，不破句、不机械。",
    "ForeignWords": "句中外来词/外语片段的发音是否地道、与本语流自然衔接。",
    "ComplexPronunciation": "易错词、缩写、数字、专有名词、同形异音词的发音是否正确。",
    "QuestionsErrors": "疑问句的升调/语气是否到位；对文本中的口误/自我纠正是否处理得当。",
}

DEFAULT_REFERENCE = "gpt-4o-mini-tts"  # 固定参照系统（占位；真实合成见 synthesize_reference）

# 判分要求 judge 只回严格 JSON，便于解析；位置标签固定为 A / B，由本脚本控制左右序与 swap。
JUDGE_SYSTEM_PROMPT = (
    "你是严格的语音表现力评审。你将听到对同一段文本合成的两条音频：System A 与 System B。"
    "只依据指定的表现力维度判断哪条更好，忽略与该维度无关的音质差异。"
    '只输出 JSON，形如 {\"winner\": \"A\"} 或 {\"winner\": \"B\"}，不要任何解释。'
)


def build_judge_prompt(category: str, text: str) -> str:
    """构造单条成对判分的文本指令（音频由各 provider 的多模态通道附带，见 call_judge 的 TODO）。"""
    guidance = CATEGORY_GUIDANCE.get(category, "整体表现力。")
    return (
        f"评测类目: {category}\n"
        f"判断维度: {guidance}\n"
        f"合成文本: 「{text}」\n"
        "System A 与 System B 为同一文本的两条合成音频。哪条在上述维度上更好？"
        '只回 {\"winner\": \"A\"} 或 {\"winner\": \"B\"}。'
    )


def parse_judge_choice(response_text: str) -> str:
    """从 judge 回复里解析出 'A' / 'B'；先按 JSON，再按 winner=X，最后退化为「独立字母」匹配。

    用 \\b 词边界，避免把 "better"/"abandon" 里的 b/a 误当成选项（朴素子串嗅探会踩此坑）。
    """
    try:
        obj = json.loads(response_text.strip())
        w = str(obj.get("winner", "")).strip().upper()
        if w in ("A", "B"):
            return w
    except (json.JSONDecodeError, AttributeError):
        pass
    t = response_text.upper()
    mkv = re.search(r"WINNER\W{0,4}\b([AB])\b", t)  # winner: A / "winner":"B"
    if mkv:
        return mkv.group(1)
    has_a = re.search(r"\bA\b", t) is not None      # 独立 token A / B（"BETTER" 不算 B）
    has_b = re.search(r"\bB\b", t) is not None
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    raise ValueError(f"无法从 judge 回复解析 winner: {response_text!r}")


# --------------------------------------------------------------------------- #
# 三处外部接线点：均**无法仅凭本仓代码核验**，故显式抛错并写明该填什么（仓库纪律）。
# 补全这三个函数后，main() 的成对评测主循环即为真实可跑路径。
# --------------------------------------------------------------------------- #
def load_emergent_items(root: Path, limit: int | None) -> list[dict]:
    """读 EmergentTTS-Eval-public 的评测条目 → [{category, text, ...}, ...]。

    ⚠️〔需核实〕EmergentTTS-Eval-public 的确切数据文件与字段名（categories 标签、文本字段、
       是否带 reference prompt / speaker 条件）需对照其 repo 核实；其官方 run_eval 入口亦在此接。
       期望每条至少含: {"category": <CATEGORIES 之一>, "text": <要合成的文本>}。
    """
    raise NotImplementedError(
        f"〔需核实〕请对照 {root} 的数据格式实现 load_emergent_items：返回 "
        '[{"category","text"}...]；并核对其官方 run_eval API。'
    )


def call_judge(judge_model: str, category: str, text: str,
               audio_a: str, audio_b: str) -> str:
    """把成对音频 + build_judge_prompt 交给多模态 LLM-judge，返回 'A' / 'B'。

    ⚠️〔需核实 / 需 API key〕judge（默认 gemini-2.5-pro）的音频入参方式与鉴权需核实：
       - Gemini: 上传/内联两段音频 + JUDGE_SYSTEM_PROMPT + build_judge_prompt，再 parse_judge_choice。
       - 或直接复用 EmergentTTS-Eval 自带 judge（其 API 同样〔需核实〕）。
    """
    _ = (JUDGE_SYSTEM_PROMPT, build_judge_prompt(category, text), audio_a, audio_b)
    raise NotImplementedError(
        f"〔需核实/需 API key〕请实现 judge='{judge_model}' 的成对音频判分（附两段音频，"
        "用 parse_judge_choice 解析 winner）。"
    )


def synthesize_reference(text: str, reference: str, work_dir: Path) -> str:
    """用固定参照系统(默认 gpt-4o-mini-tts)合成一条 wav，返回路径。

    ⚠️〔需 API key〕参照合成（gpt-4o-mini-tts，OpenAI /v1/audio/speech）需 OPENAI_API_KEY；
       入参/响应格式以 OpenAI 官方为准〔需核实〕。换参照系统时只改本函数。
    """
    _ = (text, work_dir)
    raise NotImplementedError(
        f"〔需 API key〕请实现参照系统 '{reference}' 的合成（如 OpenAI gpt-4o-mini-tts），返回 wav 路径。"
    )


def make_dut_tts_fn(runtime, work_dir: Path) -> Callable[..., str]:
    """被测系统(DUT) = 本项目 dots.tts runtime。返回 tts_fn(text, prompt_audio=None)->wav 路径。

    （沿用原 stub 的 runtime.generate 接法；soundfile/torch 等重依赖只在调用时 import，保证本文件 import 干净。）
    """
    def tts_fn(text: str, prompt_audio: str | None = None) -> str:
        res = runtime.generate(text=text, prompt_audio_path=prompt_audio,
                               prompt_text="", normalize_text=True)
        import soundfile as sf  # type: ignore
        tmp = work_dir / f"dut_{abs(hash(text))}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        audio = res["audio"]
        try:
            audio = audio.float().cpu().squeeze().numpy()
        except AttributeError:
            pass
        sf.write(str(tmp), audio, int(res["sample_rate"]))
        return str(tmp)

    return tts_fn


def judge_pair_with_swap(judge_model: str, category: str, text: str,
                         dut_wav: str, ref_wav: str, rng: random.Random) -> tuple[float, bool]:
    """对一条样本做**左右序随机化 + swap 一致性检查**：正反各问一次，记一致性。

    返回 (dut_score, consistent):
      - 第 1 问随机决定 DUT 落 A 还是 B；第 2 问把左右位置对调（消除 judge 的位置偏好）。
      - dut_score ∈ {0, 0.5, 1}：两问中 DUT 获胜的次数 / 2（分歧自然记 0.5，等价平局）。
      - consistent: 两问是否都判同一系统赢（与位置无关）→ 汇总成 swap_consistency。
    """
    # 第 1 问：随机左右序。
    dut_is_a = rng.random() < 0.5
    a1, b1 = (dut_wav, ref_wav) if dut_is_a else (ref_wav, dut_wav)
    win1 = call_judge(judge_model, category, text, a1, b1)
    dut_won_1 = (win1 == "A") == dut_is_a

    # 第 2 问：位置对调（DUT 落到另一侧）。
    a2, b2 = b1, a1
    win2 = call_judge(judge_model, category, text, a2, b2)
    dut_won_2 = (win2 == "A") == (not dut_is_a)

    dut_score = (int(dut_won_1) + int(dut_won_2)) / 2.0
    consistent = dut_won_1 == dut_won_2
    return dut_score, consistent


def _template_result(treatment: str, judge: str, reference: str) -> dict:
    """字段齐全、值为 null 的输出模板（--dry-run / 未接线时落盘，供 aggregate 安全读取）。"""
    return {
        "treatment": treatment,
        "judge": judge,
        "reference": reference,
        "n": 0,
        "categories": {c: None for c in CATEGORIES},
        "overall": None,
        "swap_consistency": None,
        "note": f"win-rate vs {reference}（非绝对分；swap 一致率低则胜率不可信）",
    }


def run_head_to_head(items: list[dict], dut_tts_fn: Callable[..., str], judge_model: str,
                     reference: str, work_dir: Path, rng: random.Random) -> dict:
    """成对评测主循环：逐条合成 DUT/参照 → swap 一致的 judge → per-category 胜率聚合。"""
    per_cat_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    consist_flags: list[bool] = []

    for it in items:
        cat = it["category"]
        text = it["text"]
        if cat not in per_cat_scores:  # 容错：未知类目并入聚合但不建新列
            per_cat_scores[cat] = []
        dut_wav = dut_tts_fn(text, it.get("prompt_audio"))
        ref_wav = synthesize_reference(text, reference, work_dir)
        score, consistent = judge_pair_with_swap(judge_model, cat, text, dut_wav, ref_wav, rng)
        per_cat_scores[cat].append(score)
        consist_flags.append(consistent)

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    categories = {c: _mean(per_cat_scores.get(c, [])) for c in CATEGORIES}
    all_scores = [s for xs in per_cat_scores.values() for s in xs]
    result = _template_result("", judge_model, reference)
    result.update({
        "n": len(all_scores),
        "categories": categories,
        "overall": _mean(all_scores),  # micro-average（按样本，非按类目均值）
        "swap_consistency": (round(sum(consist_flags) / len(consist_flags), 4)
                             if consist_flags else None),
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="EmergentTTS-Eval 表现力评测（成对 A/B LLM-judge）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--emergent-root",
                    default=str(PATHS.project_root / ".third_party" / "EmergentTTS-Eval"))
    ap.add_argument("--judge", default="gemini-2.5-pro")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条（调试用）")
    ap.add_argument("--seed", type=int, default=12345, help="左右序随机化种子（可复现）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不连外部依赖，仅落输出 schema 模板（本机核验 aggregate 对接用）")
    args = ap.parse_args()

    out = PATHS.results_dir / f"{args.treatment}.emergent.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        result = _template_result(args.treatment, args.judge, args.reference)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ [dry-run] 输出 schema 模板已写 -> {out}（补全 3 处外部接线后产出真实胜率）")
        return

    root = Path(args.emergent_root)
    if not root.exists():
        raise SystemExit(
            f"未找到 EmergentTTS-Eval: {root}\n"
            f"请先: git clone https://github.com/boson-ai/EmergentTTS-Eval-public {root}\n"
            f"（或先用 --dry-run 仅核验输出 schema）")

    sys.path.insert(0, str(root))
    from common import ensure_dots_tts_importable  # noqa: E402
    ensure_dots_tts_importable()
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore  # noqa: E402
    runtime = DotsTtsRuntime.from_pretrained(args.ckpt)

    work_dir = root / "_synth" / args.treatment
    dut_tts_fn = make_dut_tts_fn(runtime, work_dir)
    rng = random.Random(args.seed)

    # ① 取评测条目 → ② 成对评测主循环（含 swap 一致性）→ ③ 聚合 → ④ 落盘。
    items = load_emergent_items(root, args.limit)  # 〔需核实〕外部数据格式
    if args.limit is not None:
        items = items[: args.limit]
    result = run_head_to_head(items, dut_tts_fn, args.judge, args.reference, work_dir, rng)
    result["treatment"] = args.treatment

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ [{args.treatment}] overall={result['overall']}  "
          f"swap一致率={result['swap_consistency']}  -> {out}")


if __name__ == "__main__":
    main()
