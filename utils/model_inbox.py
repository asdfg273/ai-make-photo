# utils/model_inbox.py
"""models/puton 「投递箱」：把用户丢进来的模型自动归位到对应架构目录。

为什么需要它
------------
用户下载模型时几乎不可能知道它是 SD1.5 还是 SDXL（甚至不知道"架构"
是什么）。所以提供一个"丢进来就行"的目录：程序启动时读文件头判定
架构，移动到 models/<架构目录>/，并弹窗告诉用户去了哪里。

安全约定（重要）
----------------
· 只读 safetensors 文件头判架构，**不加载权重** —— 毫秒级、不占显存
· **不下载任何东西**（模型一律由用户自行下载，本项目不提供下载入口）
· 只移动「识别成功的底模 / LoRA」；未知或非底模（ControlNet/VAE）
  一律原地保留并说明原因，绝不猜、绝不误删
· 同名冲突**绝不覆盖**：大小相同视为重复（跳过），否则自动加后缀
· 正在下载的中间文件（.part/.tmp/.crdownload/0 字节）一律忽略
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field

from utils import paths

logger = logging.getLogger(__name__)

INBOX_DIRNAME = "puton"

# 判不出底模的 DiT / 通用 LoRA 的落点。
# 与项目既有约定一致（loras/_unknown_dit/ 已存在），
# 不能用 LORA_DIR 根目录，否则会和子目录混在一起。
UNKNOWN_LORA_SUBDIR = "_unknown_dit"

# 明显不是模型的文件
_SKIP_NAMES = {"desktop.ini", "thumbs.db", ".ds_store", ".gitkeep", "readme.txt"}
# 下载器/压缩工具的中间产物
_SKIP_SUFFIX = (".part", ".tmp", ".download", ".crdownload", ".aria2", ".!qb",
                ".zip", ".rar", ".7z")
_MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".bin")

_README = """\
把下载好的模型直接丢进这个文件夹（puton）就行。

程序每次启动会自动：
  1. 读取文件头判断它是 SD1.5 / SDXL / Flux / SD3 / Anima / MOSS-Image / LoRA；
  2. 移动到 models/ 或 loras/ 下正确的子目录；
  3. 弹窗告诉你每个文件去了哪里。

不用你自己判断架构，也不用改名。
识别不了的文件会原样留在这里，并在弹窗里说明原因。

注意：
  · 请不要放压缩包（.zip/.7z），先解压出模型文件再放进来；
  · 正在下载中的文件（.part/.tmp）会被忽略；
  · 同名文件不会覆盖，如有冲突会自动加 _1、_2 后缀；
  · LoRA 会按架构分到 loras/sd1.5、loras/sdxl 等子目录，
    判不出底模的进 loras/_unknown_dit；
  · Flux / SD3 除了底模还需要「引擎组件」（约 10GB），
    请在扩展市场 → 新架构引擎 里安装，否则放进来也用不了。
  · Anima 除了底模还需要「基座组件」（Qwen3 文本编码器 / VAE，约 1.7GB）：
    把整个基座目录（里面有 modular_model_index.json）也丢进来，会归位到
    models/anima/。没有基座 Anima 出不了图，程序不会自动下载。
  · MOSS-Image 本身就是**一个目录**（含 model_index.json 与
    pipeline_moss_image.py，bf16 约 10GB），整个目录丢进来即可，
    会归位到 models/mossimage/。ComfyUI 的单文件版本本程序不支持。
"""


@dataclass
class Entry:
    """一个待处理条目的判定与处理结果。"""
    src: str                      # 源绝对路径
    name: str                     # 文件名（或目录名）
    kind: str = "unknown"         # model | lora | other | unknown
    arch_id: str = ""
    display: str = ""
    target: str = ""              # 目标绝对路径
    status: str = "kept"          # moved | duplicate | kept | failed
    note: str = ""
    is_dir: bool = False

    @property
    def rel_target(self) -> str:
        """相对项目根的目标路径，用于展示。"""
        if not self.target:
            return ""
        try:
            return os.path.relpath(self.target, paths.PROJECT_ROOT).replace("\\", "/")
        except Exception:
            return self.target


@dataclass
class Report:
    entries: list[Entry] = field(default_factory=list)

    @property
    def moved(self) -> list[Entry]:
        return [e for e in self.entries if e.status == "moved"]

    @property
    def duplicates(self) -> list[Entry]:
        return [e for e in self.entries if e.status == "duplicate"]

    @property
    def failed(self) -> list[Entry]:
        return [e for e in self.entries if e.status == "failed"]

    @property
    def kept(self) -> list[Entry]:
        """未移动且不是重复的（识别失败 / 非底模）。"""
        return [e for e in self.entries
                if e.status == "kept" and e.kind in ("unknown", "other")]

    def summary_text(self, max_items: int = 12) -> str:
        """给弹窗用的中文说明。"""
        lines: list[str] = []

        if self.moved:
            lines.append(f"✅ 已自动归位 {len(self.moved)} 个文件")
            for e in self.moved[:max_items]:
                lines.append(f"   · {e.name}\n       → {e.rel_target}"
                             f"（{e.display}）")
                if e.note:
                    lines.append(f"       {e.note}")
            if len(self.moved) > max_items:
                lines.append(f"   …还有 {len(self.moved) - max_items} 个")

        if self.duplicates:
            if lines:
                lines.append("")
            lines.append(f"⏭ 跳过 {len(self.duplicates)} 个（目标位置已有相同文件）")
            for e in self.duplicates[:max_items]:
                lines.append(f"   · {e.name}")
            lines.append("   （确认无误后可删除 puton 里的这些副本）")

        if self.failed:
            if lines:
                lines.append("")
            lines.append(f"❌ 失败 {len(self.failed)} 个")
            for e in self.failed[:max_items]:
                lines.append(f"   · {e.name} —— {e.note}")

        if self.kept:
            if lines:
                lines.append("")
            lines.append(f"⚠️ 无法自动处理 {len(self.kept)} 个（已原样保留）")
            for e in self.kept[:max_items]:
                reason = e.note or "无法识别架构"
                lines.append(f"   · {e.name} —— {reason}")

        if not lines:
            return ""

        lines.append("")
        lines.append("说明：程序只读取文件头判断架构，不会加载或上传你的模型，"
                     "也不会联网下载任何东西。")
        return "\n".join(lines)


def inbox_dir() -> str:
    return os.path.join(paths.MODEL_DIR, INBOX_DIRNAME)


def ensure_inbox() -> str:
    """确保投递箱存在，并放一份说明（首次运行时生成）。"""
    d = inbox_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        logger.warning(f"⚠️ 无法创建投递箱目录 {d}: {e}")
        return d
    readme = os.path.join(d, "README.txt")
    # 内容是程序生成的，所以**版本变了就覆盖**：老用户升级后仍能看到
    # 新架构（如 MOSS-Image / Anima 基座）的放置说明，而不是半年前的旧文件。
    old = ""
    if os.path.exists(readme):
        try:
            with open(readme, "r", encoding="utf-8") as f:
                old = f.read()
        except OSError as e:
            logger.debug(f"读取投递箱说明失败（将覆盖）: {e}")
    if old != _README:
        try:
            with open(readme, "w", encoding="utf-8") as f:
                f.write(_README)
        except OSError as e:
            logger.debug(f"投递箱说明写入失败（忽略）: {e}")
    return d


def _skip(name: str) -> bool:
    low = name.lower()
    if name.startswith("."):
        return True
    if low in _SKIP_NAMES:
        return True
    return low.endswith(_SKIP_SUFFIX)


def _target_for(arch_id: str, info, name: str) -> tuple[str, str]:
    """(kind, 目标绝对路径)。无法归位时返回 ("unknown"/"other", "")。"""
    if arch_id.startswith("lora"):
        # lora_subdir 为空 = 判不出底模（lora / lora_dit）→ 进 _unknown_dit
        sub = info.lora_subdir or UNKNOWN_LORA_SUBDIR
        return "lora", os.path.join(paths.LORA_DIR, sub, name)
    if not getattr(info.caps, "is_base_model", False):
        # ControlNet / VAE：目录结构特殊，交给用户手动放
        return "other", ""
    sub = getattr(info, "model_subdir", None) or arch_id
    return "model", os.path.join(paths.MODEL_DIR, sub, name)


def _classify(path: str, name: str, is_dir: bool) -> Entry:
    from core import arch as arch_mod

    e = Entry(src=path, name=name, is_dir=is_dir)
    try:
        det = arch_mod.detect(path)
    except Exception as ex:                       # 检测本身出错也不能中断启动
        e.kind, e.status = "unknown", "kept"
        e.note = f"架构检测异常: {ex}"
        return e

    e.arch_id = det.arch_id
    e.display = det.info.display_name

    if det.arch_id == "unknown":
        e.kind, e.status = "unknown", "kept"
        hint = det.evidence or "无匹配特征"
        if det.top_prefixes:
            hint += f"；键名前缀: {', '.join(det.top_prefixes[:3])}"
        e.note = f"无法识别架构（{hint}）"
        return e

    kind, target = _target_for(det.arch_id, det.info, name)
    e.kind = kind
    if not target:
        e.status = "kept"
        e.note = det.info.unsupported_reason or "非底模文件，请手动放入对应目录"
        return e

    e.target = target
    # 「（非底模）」这种后缀在弹窗里读着别扭（如 LoRA），去掉
    e.display = (det.info.display_name or "").replace("（非底模）", "").strip()

    # ⚠️ 只有**底模**才谈得上"引擎是否接通"。
    #    LoRA 的 ArchInfo.supported 也是 False（因为它不能单独加载），
    #    以前一并套用这句提示，会给用户"我的 LoRA 不能用"的错误印象。
    if kind == "model" and not det.info.supported:
        # 直接用注册表里的 unsupported_reason：它是唯一出处，
        # 里面有"去哪儿装、装多大、许可证如何"这些可操作信息。
        reason = (det.info.unsupported_reason or "该架构引擎尚未接通").strip()
        e.note = f"⚠️ {e.display} 已归位，但暂时无法用于生成：{reason}"
    return e


def scan() -> list[Entry]:
    """扫描投递箱并判定每个条目（**不移动任何文件**）。"""
    d = inbox_dir()
    if not os.path.isdir(d):
        return []

    out: list[Entry] = []
    try:
        names = sorted(os.listdir(d))
    except OSError as e:
        logger.warning(f"⚠️ 无法读取投递箱 {d}: {e}")
        return []

    for name in names:
        if _skip(name):
            continue
        p = os.path.join(d, name)
        try:
            if os.path.isdir(p):
                # diffusers 目录格式（含 model_index.json）也支持
                out.append(_classify(p, name, True))
                continue
            if not os.path.isfile(p):
                continue
            if not name.lower().endswith(_MODEL_EXT):
                continue
            if os.path.getsize(p) == 0:       # 还在下载
                continue
        except OSError:
            continue
        out.append(_classify(p, name, False))
    return out


def _unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    for i in range(1, 1000):
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
    return f"{base}_{uuid.uuid4().hex[:8]}{ext}"


def _move(e: Entry) -> None:
    tgt_dir = os.path.dirname(e.target)
    try:
        os.makedirs(tgt_dir, exist_ok=True)
    except OSError as ex:
        e.status, e.note = "failed", f"无法创建目录 {tgt_dir}: {ex}"
        return

    final = e.target
    if os.path.exists(final):
        both_files = (not e.is_dir) and os.path.isfile(final)
        if both_files:
            try:
                if os.path.getsize(final) == os.path.getsize(e.src):
                    e.status = "duplicate"
                    e.note = "目标位置已有同名且大小相同的文件"
                    return
            except OSError:
                pass
        # 目录、或同名但大小不同 → 绝不覆盖，改名
        final = _unique_path(final)

    try:
        shutil.move(e.src, final)
    except Exception as ex:
        e.status, e.note = "failed", f"移动失败: {ex}"
        return

    renamed = os.path.basename(final) != e.name
    e.target = final
    e.status = "moved"
    if renamed:
        suffix = f"（同名冲突，已重命名为 {os.path.basename(final)}）"
        e.note = (e.note + " " if e.note else "") + suffix


def run(apply: bool = True) -> Report:
    """扫描并（默认）归位。任何异常都不应影响程序启动。"""
    ensure_inbox()
    entries = scan()
    if apply:
        for e in entries:
            if e.status == "kept" and e.kind in ("model", "lora") and e.target:
                _move(e)
    rep = Report(entries)
    if rep.moved:
        logger.info("📥 投递箱已归位 %d 个文件: %s", len(rep.moved),
                    ", ".join(f"{e.name}→{e.rel_target}" for e in rep.moved))
    if rep.kept:
        logger.info("📥 投递箱 %d 个文件无法自动处理（已保留）",
                    len(rep.kept))
    return rep
