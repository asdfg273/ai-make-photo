# utils/prompt_budget.py
"""
提示词预算与分块诊断 —— 提示词长度可见化（**默认绝不删改用户内容**）

⚠️ 重要更正（已实测，2026-09）：
  本项目 `encode_prompt` 用 Compel 且 `truncate_long_prompts=False`，
  超长提示词会被**分块编码并拼接**，并没有 77 token 硬上限：

      15 token  -> embeds 序列  77   -> 正常出图
      85 token  -> embeds 序列 154   -> 正常出图
      355 token -> embeds 序列 385   -> 正常出图

  所以本模块**不再做任何自动截断**。曾经那版按 77 硬截断是错的，
  它会直接删掉用户写的内容（用户明确反馈过"我加了超长提示词组件"）。

  现在只做三件事：
    1. count_tokens()：用真实 CLIP tokenizer 数 token（懒加载，失败降级估算）
    2. size_report()  ：报告 token 数 / 分块数 / 是否含未翻译中文
    3. enforce()      ：**只告警、不改文本**；若显式传入 hard_cap 且确实超过
                        该上限（默认 None = 不截断），才按语义段截断

为什么仍然值得计数：
  长提示词虽然能生效，但注意力会**摊薄**到所有块上——把最重要的标签放在
  前 77 token（第一块）里，效果明显好于堆在末尾。所以计数是给用户做
  "重点前置"的依据，而不是用来删东西。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CLIP_CHUNK = 77                # 单个 CLIP 编码块的长度（不是上限，只是分块粒度）
# 显式截断的安全阈值：默认为 None（永不截断）。
# 实测 385 长度的序列都能正常出图；这里给一个很宽的兜底，
# 只有用户显式要求时才生效。
DEFAULT_HARD_CAP = None

# CJK 字符大约 1 字符 ≈ 1 token
_CJK_RE = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf"
                     r"\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
_SEP_RE = re.compile(r"[,，;；\n]+")


def has_cjk(text: str) -> bool:
    """是否含中日韩字符（用于提示"没翻译干净"）。"""
    return bool(text and _CJK_RE.search(text))


def split_segments(text: str) -> list[str]:
    """按中英文逗号/分号/换行切段，去掉空段。"""
    if not text:
        return []
    return [s.strip() for s in _SEP_RE.split(text) if s and s.strip()]


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的粗估：CJK 1 字符≈1 token，其余按 4 字符≈1 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = max(0, len(text) - cjk)
    return cjk + max(1, rest // 4)


# ============================================================
#  tokenizer 懒加载
# ============================================================
_TOKENIZER = None
_TOKENIZER_TRIED = False


def _try_load_tokenizer():
    """从项目缓存里找一个可用的 CLIP tokenizer（不联网）。"""
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER is not None or _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        import os
        from transformers import CLIPTokenizer
        from utils.paths import CACHE_DIR
        hub = os.path.join(CACHE_DIR, "huggingface", "hub")
        for sub in ("models--stable-diffusion-v1-5--stable-diffusion-v1-5",
                    "models--stabilityai--stable-diffusion-xl-base-1.0",
                    "models--openai--clip-vit-large-patch14"):
            snaps = os.path.join(hub, sub, "snapshots")
            if not os.path.isdir(snaps):
                continue
            for snap in sorted(os.listdir(snaps)):
                for cand in (os.path.join(snaps, snap, "tokenizer"),
                             os.path.join(snaps, snap)):
                    if (os.path.isfile(os.path.join(cand, "tokenizer_config.json"))
                            or os.path.isfile(os.path.join(cand, "vocab.json"))):
                        _TOKENIZER = CLIPTokenizer.from_pretrained(cand)
                        logger.debug(f"[prompt_budget] tokenizer 就绪: {cand}")
                        return _TOKENIZER
    except Exception as e:
        logger.debug(f"[prompt_budget] tokenizer 懒加载失败，改用估算: {e}")
    return _TOKENIZER


def set_tokenizer(tok) -> None:
    """外部注入（例如复用模型管线里的 tokenizer）。"""
    global _TOKENIZER, _TOKENIZER_TRIED
    _TOKENIZER = tok
    _TOKENIZER_TRIED = True


def count_tokens(text: str) -> int:
    """数 token；拿不到 tokenizer 时返回估算值。"""
    if not text:
        return 0
    tok = _try_load_tokenizer()
    if tok is None:
        return estimate_tokens(text)
    try:
        # truncation=False 才能看到真实长度。
        # 注意：权重语法 (词:1.15) 会被 Compel 剥离，这里不做特殊处理。
        return len(tok(text, truncation=False)["input_ids"])
    except Exception:
        return estimate_tokens(text)


# ============================================================
#  预算执行
# ============================================================
@dataclass
class BudgetReport:
    """一次提示词的长度报告。"""
    positive_tokens: int = 0
    negative_tokens: int = 0
    chunk_size: int = CLIP_CHUNK
    untranslated_cjk: bool = False
    truncated: bool = False
    dropped: list[str] = field(default_factory=list)
    prompt: str = ""
    negative: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def positive_chunks(self) -> int:
        """正向提示词会占用多少个 CLIP 编码块。"""
        return max(1, -(-self.positive_tokens // self.chunk_size))

    @property
    def negative_chunks(self) -> int:
        return max(1, -(-self.negative_tokens // self.chunk_size))

    @property
    def ok(self) -> bool:
        # 现在"超长"不再是错误，只是信息；只有显式截断才算不 ok
        return not self.truncated

    def summary(self) -> str:
        bits = [f"正向 {self.positive_tokens} token / {self.positive_chunks} 块"]
        if self.negative_tokens:
            bits.append(f"负向 {self.negative_tokens} token / {self.negative_chunks} 块")
        if self.positive_chunks > 1:
            bits.append("（超长提示词会分块拼接，可正常生效）")
        if self.truncated:
            bits.append(f"⚠ 已按显式上限截断，丢弃 {len(self.dropped)} 段")
        if self.untranslated_cjk:
            bits.append("⚠ 仍含未翻译中文")
        return " | ".join(bits)


def _truncate_to_budget(text: str, budget: int) -> tuple[str, list[str]]:
    """按段边界截断，返回 (新文本, 被丢弃的段列表)。

    按段截断而不是按字符截断：宁可少一个风格词，也不要留下半个词
    （半个 token 序列同样会破坏语义）。
    单段本身就超预算时，退化为该段内部的逐词截断。
    """
    segs = split_segments(text)
    if not segs:
        return text, []

    kept, used, idx = [], 0, 0
    for i, s in enumerate(segs):
        cost = count_tokens(s) + 1        # +1 为分隔逗号
        if used + cost > budget:
            idx = i
            break
        kept.append(s)
        used += cost
    else:
        return text, []                   # 全部装得下

    dropped = segs[idx:]

    # 第一段就放不下：只能在该段内部按空白切词，尽量保住
    if not kept:
        words = segs[0].split()
        acc, used = [], 0
        for w in words:
            cost = count_tokens(w) + 1
            if used + cost > budget:
                break
            acc.append(w)
            used += cost
        kept = [" ".join(acc)] if acc else []
        dropped = ([segs[0]] if len(acc) < len(words) else []) + segs[1:]

    return ", ".join(kept), dropped


def enforce(prompt: str,
            negative: str = "",
            hard_cap: int | None = DEFAULT_HARD_CAP,
            tag: str = "") -> BudgetReport:
    """报告提示词长度；**默认不改动任何文本**。

    hard_cap:
      None（默认）—— 永不截断。超长提示词由 Compel 分块编码后照常生效。
      int         —— 仅当显式传入且确实超过时，才按语义段截断（安全兜底）。

    返回的 report.prompt / report.negative 就是应当送进模型的文本
    （默认与原输入完全一致）。
    """
    report = BudgetReport()
    report.prompt = prompt
    report.negative = negative

    report.positive_tokens = count_tokens(prompt)
    report.negative_tokens = count_tokens(negative)

    if has_cjk(prompt):
        report.untranslated_cjk = True
        report.notes.append(
            "提示词里仍有中文（词典未命中）。中文 1 字≈1 token，"
            "等义英文只要约 1/3 —— 中文残留会让提示词很快就占满多个编码块，"
            "并摊薄注意力。建议把这些词加进词典，或改用「AI 智能改写」。")

    # 只有显式给了 hard_cap 才可能截断
    for label, text in (("正向", prompt), ("负向", negative)):
        if not text:
            continue
        if hard_cap is None:
            continue
        if count_tokens(text) <= int(hard_cap):
            continue
        new_text, dropped = _truncate_to_budget(text, int(hard_cap))
        if not dropped:
            continue
        if label == "正向":
            report.truncated = True
            report.dropped = dropped
            prompt = new_text
        else:
            negative = new_text
        report.notes.append(
            f"{label}提示词超过显式上限 {hard_cap} token，已按语义段截断；"
            f"被丢弃：{'、'.join(dropped[:8])}")

    report.prompt = prompt
    report.negative = negative

    # 多块说明：这是"信息"而不是"警告"
    if report.positive_chunks > 1:
        report.notes.append(
            f"正向提示词会占用 {report.positive_chunks} 个 CLIP 编码块。"
            f"超长提示词可以正常生效，但注意力会摊薄到所有块上 —— "
            f"把最重要的标签放在**最前面**（第一块内）效果最好。")

    if report.truncated:
        logger.warning("📏 [提示词] %s%s", f"{tag} " if tag else "",
                       report.summary())
        for n in report.notes:
            logger.warning("   ↳ %s", n)
    else:
        logger.info("📏 [提示词] %s%s", f"{tag} " if tag else "",
                    report.summary())
        for n in report.notes:
            logger.debug("   ↳ %s", n)

    return report
