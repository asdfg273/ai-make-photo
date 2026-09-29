# workflow/resource_store.py
"""统一模型资源清单：工作流按 `resource_id` 引用模型，**不**硬编码绝对路径。

修的是什么问题
--------------
以前 `LoadCheckpoint.path` / `LoadVAE.path` / `LoadLoRA.lora_N` 存的都是
「用户这台机器上的一个字符串」，多数情况是绝对路径（如
`E:\\pythonprojects\\ai  make photo\\models\\sd15\\meinamix_v12Final.safetensors`）。
于是共享工作流必然出下面几类事：

  · **换一台机器就崩**：对方根本没有 `E:\\...` 这个路径；
  · **模型挪个目录就找不到**：同机换个盘符 / 换成 `models/sdxl/` 就失联；
  · **架构对不上**：拿 SD1.5 的 LoRA 去挂 SDXL 底模，跑到一半才炸；
  · **下错版本**：同名不同版（vPred / 非 vPred、dev / schnell）分不清；
  · **依赖不透明**：打开工作流看不到"我需要哪些模型、多大、去哪下"。

本模块把「模型」抽象成一条 **`ModelResource` 记录**，工作流里只写
`resource_id` + `filename` 等**与机器无关**的字段（见 `to_ref()` /
`resolve()`），本机路径由这里在运行时解析出来。

`resource_id` 为什么**绝不能**含路径
----------------------------------
`resource_id` 是"这个模型是谁"的身份，必须**跨机器、跨目录稳定**：

  · 若把绝对路径（或路径的哈希）算进去，同一个文件在
    `D:\\models\\x.safetensors` 与 `E:\\ai\\models\\x.safetensors`
    会得到两个不同 ID —— 工作流在别人机器上永远匹配不上；
  · 若用文件名 + 架构 + 类型（`type/architecture/stem`），
    同一个人把同一个模型放到哪都还是同一个 ID，别人下载同一个模型
    也能落到同一个 ID 上，这就是"可迁移"的全部原理。

路径只作为**本地解析的结果**存在（`ModelResource.path`），需要时
（如"只在本机保存/继续"）才写进引用，默认写 `null`。

为什么 sha256 默认**不算**
-------------------------
底模 2~7GB，算一次 sha256 要把整个文件读一遍。正常扫描只 `stat` 取
大小（毫秒级），只有显式 `force_sha=True` 或 `to_ref(with_sha=True)`
才走 `utils.image_metadata.file_sha256()` —— 它自带磁盘缓存
（`models_cache/file_hashes.json`，按 路径+大小+mtime 命中），所以同一
文件第二次几乎免费，但**首次仍是一次全量读**，绝不能挂在普通扫描上。

与其他模块的关系
----------------
· `workflow/resource_manager.py` —— 按**节点声明**汇总"这条流程需要什么"
  并映射到扩展市场条目；本模块管的是**具体模型文件**的清单与解析，
  两者互补，不重复。
· `ui/workflow_editor/param_sources.py` —— 只给下拉框列文件名；
  本模块是它的"带身份/带架构/带哈希"的上位版本，且**不依赖 Qt**，
  可以被执行器、校验器、CLI 共用。
· `utils/image_metadata.file_sha256` / `utils/model_scanner` /
  `utils/paths` —— 本模块复用它们，不重复实现扫描与哈希。

设计约束
--------
· **headless**：不 import Qt、不 import `ui/`，`import workflow.resource_store`
  不触发 torch/diffusers（重依赖都在函数内部 import）；
· **不抛异常**：`resolve` / `resolve_many` / `missing_report` / `collect_refs`
  对畸形引用一律降级成 `missing` 并记 debug 日志 —— 一个坏引用不该让
  打开工作流、跑校验、出报告这些事全挂掉。
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "ModelResource", "make_resource_id",
    "build_catalog", "catalog", "invalidate",
    "by_id", "by_filename", "by_sha256",
    "to_ref", "resolve", "resolve_many", "missing_report",
    "refs_for_node", "collect_refs",
    "catalog_path", "save_catalog", "load_catalog",
]

# ── 公开常量 ─────────────────────────────────────────────────────────
#: 资源类型。`engine` 指"新架构引擎组件"（如 Flux 的 T5/CLIP/VAE），
#: 由扩展市场写入（见 `utils.extension_manager.EXTENSIONS` 的
#: `category="新架构引擎"` 条目），本模块**只扫描本机已有文件**，
#: 不去联网或调用扩展市场，所以正常扫描不会产出 engine 记录。
RESOURCE_TYPES = ("checkpoint", "lora", "vae", "controlnet", "upscaler", "engine")

#: 已知架构名（与 `core.arch.REGISTRY` 对齐；`video` 给视频模型留位）。
KNOWN_ARCHITECTURES = ("sd15", "sd21", "sdxl", "pony", "anima",
                       "flux", "sd3", "video", "unknown")

#: 许可证未知时的默认值（绝大多数本地文件确实无法得知许可证）。
DEFAULT_LICENSE = "未知"

_CATALOG_FILE = "resources.json"
_CATALOG_VERSION = 1

# 扩展名 → format 字段
_EXT_FORMAT = {
    ".safetensors": "safetensors",
    ".ckpt": "ckpt",
    ".pt": "pt",
    ".pth": "pth",
    ".gguf": "gguf",
    ".bin": "bin",
    ".onnx": "onnx",
}
# 底模：沿用 utils.model_scanner 的默认后缀（不含 .gguf 由 MODEL_TYPES 决定）
_WEIGHT_EXTS = (".safetensors", ".ckpt")
# LoRA / VAE：与 ui/workflow_editor/param_sources.py 的 _LORA_EXTS 一致
_LORA_EXTS = (".safetensors", ".ckpt", ".pt")
# ControlNet / 放大模型：多了 .pth
_ADAPTER_EXTS = (".safetensors", ".ckpt", ".pt", ".pth")

# 扫描时跳过的目录（HF 缓存里有成千上万个 .metadata 小文件，白跑）
_SKIP_DIRS = {"__pycache__", ".git", ".cache"}

# resource_id 里需要折叠成下划线的字符：空白、正斜杠、反斜杠
_ID_SEP_RE = re.compile(r"[\s/\\]+")

# 架构归族：**判"能不能通用"要按族，不按字面**
#   · sd21 / sd2 与 sd15 在本项目共用 `models/sd15` 与 `loras/sd1.5`；
#   · pony 是 SDXL 微调，权重与 LoRA 同族（`loras/sdxl/`）；
#   · lora_* 前缀只是"这是个 LoRA"的标注，去掉后再比。
_ARCH_FAMILY = {
    "sd15": "sd15", "sd1.5": "sd15", "sd1_5": "sd15",
    "sd21": "sd15", "sd2": "sd15", "sd2.1": "sd15",
    "sdxl": "sdxl", "xl": "sdxl", "pony": "sdxl",
    "sd3": "sd3", "sd35": "sd3", "sd3.5": "sd3",
    "flux": "flux", "anima": "anima", "video": "video",
}

#: LoRA 子目录名 → 架构（缺注册表时的兜底；正常走 `core.arch.REGISTRY`
#: 的 `lora_subdir` 派生，新增架构不用改这里）
_LORA_DIR_FALLBACK = {
    "sd1.5": "lora_sd15", "sd15": "lora_sd15", "sd1_5": "lora_sd15",
    "sdxl": "lora_sdxl", "pony": "lora_sdxl",
    "_unknown_dit": "lora_dit", "unknown_dit": "lora_dit",
    "dit": "lora_dit",
}


# ======================================================================
#  小工具
# ======================================================================
def _s(v) -> str:
    """任何值 → 去空白字符串；None → ""。"""
    if v is None:
        return ""
    try:
        return str(v).strip()
    except Exception:
        return ""


def _norm_sha(v) -> str:
    """规范化 sha256：未计算/失败时统一成 ""（而不是 "unknown"）。

    `file_sha256()` 失败会返回字面量 "unknown"，直接塞进资源记录会
    让"有没有哈希"变得难判，所以在边界处归一化。
    """
    s = _s(v).lower()
    if s in ("", "unknown", "none", "null", "n/a"):
        return ""
    return s


def _format_of(path: str) -> str:
    """扩展名 → format 字段（未知扩展名去掉点原样返回）。"""
    ext = os.path.splitext(_s(path))[1].lower()
    return _EXT_FORMAT.get(ext, ext.lstrip(".").lower())


def _size_of(path: str) -> int:
    """文件字节数；取不到（不存在/无权限）返回 0，绝不抛。"""
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0
    except Exception:
        return 0


def _sha_of(path: str) -> str:
    """按需计算 sha256 —— **唯一**的哈希入口。

    复用 `utils.image_metadata.file_sha256()`：它自带
    `models_cache/file_hashes.json` 磁盘缓存（键 = 绝对路径 + 大小 + mtime），
    本模块不再写第二套缓存，避免两份缓存对不上。
    """
    if not path:
        return ""
    try:
        from utils.image_metadata import file_sha256
        return _norm_sha(file_sha256(path))
    except Exception as e:
        logger.debug(f"计算 sha256 失败（按未计算处理）: {path}: {e}")
        return ""


def _id_part(value: str, default: str) -> str:
    """把一段文本压成 id 片段：小写、空白/斜杠折叠成下划线。"""
    s = _ID_SEP_RE.sub("_", _s(value)).strip("_").lower()
    return s or default


def make_resource_id(filename: str, architecture: str = "unknown",
                     type: str = "checkpoint") -> str:
    """生成稳定 `resource_id`：`{type}/{architecture}/{stem}`。

    例：`meinamix_v12Final.safetensors` + `sd15` + `checkpoint`
    → `checkpoint/sd15/meinamix_v12final`。

    实现细节：**先 `os.path.basename()` 丢掉全部路径成分**，再把剩下的
    空白 / 斜杠折叠成下划线、转小写。所以
    `make_resource_id("a b/c\\d.SAFETENSORS", "sdxl", "lora")`
    → `lora/sdxl/d` —— 路径片段一定不会渗进 ID。

    为什么**绝不**用绝对路径（或其哈希）参与生成
    -------------------------------------------
    1. 工作流要能跨机器打开：路径含用户名/盘符，换台机器 ID 就变了，
       工作流里记录的 ID 永远匹配不到任何资源；
    2. 同一台机器上换个目录（`models/sd15/` → `models/puton/`）也不该
       让身份改变 —— 否则"移动模型"等于"资源凭空消失"；
    3. ID 要**可读、可手改**：用户看 `checkpoint/sdxl/noobaixl_vpred10version`
       就知道缺什么，看路径看不出"这是哪个架构"；
    4. 别人下载同一个模型、放进自己的目录，也能落到**同一个 ID**，
       这才是"共享工作流"能成立的前提。

    代价：同目录下不同架构的同名文件会撞 ID —— 由
    `architecture` 段隔开（`sd15`/`sdxl` 不同），同类型同架构同名
    本就该是同一个模型，撞了说明文件确实重名，属于用户要处理的事。
    """
    stem = os.path.splitext(os.path.basename(_s(filename)))[0]
    stem = _id_part(stem, "unnamed")
    return f"{_id_part(type, 'checkpoint')}/{_id_part(architecture, 'unknown')}/{stem}"


def _arch_family(value: str) -> str:
    """架构 → 族名（去掉 `lora_` 前缀并按 `_ARCH_FAMILY` 归并）。"""
    key = _s(value).lower()
    for prefix in ("lora_", "lora-", "lora "):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return _ARCH_FAMILY.get(key, key)


def _arch_compatible(ref_arch: str, res_arch: str) -> bool:
    """引用里写的架构与本地资源的架构是否**同族**。

    任一侧为空 / `unknown` 时视为"无法判定" → 兼容（不误报）。
    这样 sd21↔sd15（本项目共用目录）、pony↔sdxl（同族权重）不会误报，
    而 sd15 ↔ flux/sdxl 会如实报 `arch_mismatch`。
    """
    a, b = _arch_family(ref_arch), _arch_family(res_arch)
    if not a or not b or "unknown" in (a, b):
        return True
    return a == b


def _guess_type_from_name(path: str) -> str:
    """按文件名猜资源类型（只在 `local_path` 兜底建记录时用）。"""
    low = _s(path).lower()
    if "lora" in low:
        return "lora"
    if "vae" in low:
        return "vae"
    if "controlnet" in low or "control_" in low:
        return "controlnet"
    if "upscal" in low:
        return "upscaler"
    return "checkpoint"


def _guess_arch_from_name(text: str) -> str:
    """按目录/文件名猜架构 —— **纯启发式**，猜不到就 `unknown`。

    为什么是启发式：ControlNet/放大模型不是底模，`core.arch.detect()`
    对它们返回的是 `controlnet` 而不是"能挂哪个底模"的架构；要真判定
    得读权重头再推断，代价与收益不成比例。所以这里只看名字里的
    `sdxl`/`XL`/`sd15` 线索，判不出就诚实地写 `unknown`
    （`resolve()` 对 `unknown` 不做架构校验，不会误报）。
    """
    low = _s(text).lower()
    compact = low.replace("_", "").replace("-", "").replace(" ", "")
    if "sdxl" in compact or re.search(r"[a-z0-9]xl", compact):
        return "sdxl"
    if "sd15" in compact or "sd1.5" in low:
        return "sd15"
    if "flux" in compact:
        return "flux"
    return "unknown"


def _walk_files(root: str, exts: tuple, *, name_filter=None) -> list:
    """递归收集 root 下指定后缀的文件绝对路径（排序、去重、跳过缓存目录）。

    目录不存在 / 没权限一律当"没有文件"，不向外抛 —— 本模块的每个
    调用点都只想要一个列表。
    """
    found = []
    if not root or not os.path.isdir(root):
        return found
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if not fn.lower().endswith(exts):
                    continue
                if name_filter is not None and not name_filter(fn):
                    continue
                found.append(os.path.join(dirpath, fn))
    except Exception as e:
        logger.debug(f"遍历目录失败（按已收集内容返回）: {root}: {e}")
    return sorted(found)


def _make_local(path: str, name: str, rtype: str, arch: str,
                force_sha: bool = False, *, source: str = "local",
                source_url: str = "", license_text: str = DEFAULT_LICENSE,
                ) -> ModelResource:
    """由本地文件造一条记录。`force_sha` 控制是否**现在**算哈希。"""
    full = os.path.abspath(path) if path else ""
    disp = name or os.path.basename(full)
    return ModelResource(
        resource_id=make_resource_id(disp or full, arch, rtype),
        name=disp,
        path=full,
        type=rtype,
        architecture=arch or "unknown",
        format=_format_of(full or disp),
        size_bytes=_size_of(full),
        sha256=_sha_of(full) if (force_sha and full) else "",
        source=source,
        source_url=source_url,
        license=license_text,
    )


# ======================================================================
#  资源记录
# ======================================================================
@dataclass
class ModelResource:
    """一条模型资源记录（dataclass，可 JSON 往返）。

    字段全部带默认值：这样 `from_dict()` 面对"老版本写的一半字段"
    或"别人手写的残缺 JSON"都能读进来，而不是抛 KeyError。

    `path` 允许为空 = **未下载**（引用存在、文件不在本机）。
    `sha256` 为空 = **尚未计算**（默认不算是刻意的，见模块文档）。
    """

    resource_id: str = ""
    name: str = ""
    path: str = ""
    type: str = "checkpoint"
    architecture: str = "unknown"
    format: str = ""
    size_bytes: int = 0
    sha256: str = ""
    source: str = "local"
    source_url: str = ""
    license: str = DEFAULT_LICENSE
    verified: bool = False

    def __post_init__(self):
        """构造后规范化：脏值（None / 数字 / "unknown" 哈希）一律修正。

        也在 `resource_id` 为空时补一个（按 `make_resource_id()`），
        这样手写或老格式的记录不会带着空 ID 在清单里"隐形"。
        """
        self.resource_id = _s(self.resource_id)
        self.name = _s(self.name)
        self.path = _s(self.path)
        self.type = _s(self.type) or "checkpoint"
        self.architecture = _s(self.architecture) or "unknown"
        self.format = _s(self.format) or _format_of(self.path or self.name)
        try:
            self.size_bytes = int(self.size_bytes or 0)
        except (TypeError, ValueError):
            self.size_bytes = 0
        self.sha256 = _norm_sha(self.sha256)
        self.source = _s(self.source) or "local"
        self.source_url = _s(self.source_url)
        self.license = _s(self.license) or DEFAULT_LICENSE
        self.verified = bool(self.verified)
        if not self.resource_id:
            self.resource_id = make_resource_id(
                self.name or self.path or "unnamed",
                self.architecture, self.type)

    # ---- 展示 / 序列化 ----
    def to_dict(self) -> dict:
        """→ 普通 dict（JSON 友好，键名与字段同名）。"""
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "architecture": self.architecture,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "source": self.source,
            "source_url": self.source_url,
            "license": self.license,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d) -> "ModelResource":
        """dict → 记录。**容错**：缺键、类型不对、甚至整个不是 dict 都不抛。

        `resource_id` 缺失时按 `make_resource_id()` 现场补一个 ——
        这样"手写的清单"或"老文件"也能参与解析，不会因为少一个 ID
        就让整份清单不可用。
        """
        if not isinstance(d, dict):
            return cls()
        name = _s(d.get("name")) or os.path.basename(_s(d.get("path")))
        path = _s(d.get("path"))
        rtype = _s(d.get("type")) or "checkpoint"
        arch = _s(d.get("architecture")) or "unknown"
        rid = _s(d.get("resource_id"))
        if not rid:
            rid = make_resource_id(name or path or "unnamed", arch, rtype)
        return cls(
            resource_id=rid,
            name=name,
            path=path,
            type=rtype,
            architecture=arch,
            format=_s(d.get("format")) or _format_of(path),
            size_bytes=d.get("size_bytes") or 0,
            sha256=d.get("sha256") or "",
            source=_s(d.get("source")) or "local",
            source_url=_s(d.get("source_url")),
            license=_s(d.get("license")) or DEFAULT_LICENSE,
            verified=bool(d.get("verified", False)),
        )

    def __repr__(self) -> str:  # 排错用，不进 JSON
        return (f"ModelResource({self.resource_id!r}, path={self.path!r}, "
                f"size={self.size_bytes}, sha={self.sha256[:10]!r})")


# ======================================================================
#  本机清单扫描
# ======================================================================
def _scan_checkpoints(force_sha: bool) -> list:
    """底模：复用 `utils.model_scanner`（按架构注册表派生的子目录逐个数）。

    架构直接取子目录名（`MODEL_TYPES` / `scan_models` 的 `type` 字段）——
    这正是"用户把它放在哪个架构目录"的声明，也避免为了认架构去读
    每个底模的 safetensors header（10 个文件虽快，但没必要）。
    """
    out = []
    try:
        from utils.model_scanner import MODEL_TYPES, scan_models
    except Exception as e:
        logger.debug(f"模型扫描器不可用（跳过底模扫描）: {e}")
        return out

    for subdir in MODEL_TYPES:
        try:
            items = scan_models(subdir)
        except Exception as e:
            # 单个子目录坏了不该拖掉其它架构
            logger.debug(f"扫描模型子目录失败: {subdir}: {e}")
            continue
        for item in items or []:
            try:
                p = _s(item.get("path"))
                name = _s(item.get("name")) or os.path.basename(p)
                arch = _s(item.get("type")) or _s(subdir) or "unknown"
                if not p:
                    continue
                out.append(_make_local(p, name, "checkpoint", arch, force_sha))
            except Exception as e:
                logger.debug(f"构造底模记录失败（跳过）: {e}")
    return out


def _lora_subdir_map() -> dict:
    """LoRA 子目录名 → 架构。优先从 `core.arch.REGISTRY` 派生。

    这样新增架构只要改注册表（`lora_subdir`），本模块不用动；
    注册表拿不到时回落到 `_LORA_DIR_FALLBACK`。
    """
    mapping = dict(_LORA_DIR_FALLBACK)
    try:
        from core.arch import REGISTRY
        for arch_id, info in REGISTRY.items():
            if not arch_id.startswith("lora_"):
                continue
            subdir = _s(getattr(info, "lora_subdir", ""))
            if subdir and subdir not in mapping:
                mapping[subdir] = arch_id
    except Exception as e:
        logger.debug(f"从架构注册表派生 LoRA 目录映射失败（用兜底表）: {e}")
    return mapping


def _scan_loras(force_sha: bool) -> list:
    """LoRA：递归 `loras/`，架构由**父目录名**推断。

    为什么按目录而不是按权重判：LoRA 的架构判定要读 header 里的
    cross-attn 维度（见 `core.arch.signatures`），几十个文件逐个读头
    不划算；而本项目本来就要求用户把 LoRA 放进 `loras/sd1.5/`、
    `loras/sdxl/`、`loras/_unknown_dit/` —— 目录名就是用户给出的声明，
    判不出时写 `unknown`（不会误报架构不符）。
    """
    out = []
    try:
        from utils import paths
    except Exception as e:
        logger.debug(f"路径模块不可用（跳过 LoRA 扫描）: {e}")
        return out

    mapping = _lora_subdir_map()
    for p in _walk_files(paths.LORA_DIR, _LORA_EXTS):
        try:
            parent = os.path.basename(os.path.dirname(p)).strip().lower()
            arch = mapping.get(parent, "unknown")
            out.append(_make_local(p, os.path.basename(p), "lora", arch, force_sha))
        except Exception as e:
            logger.debug(f"构造 LoRA 记录失败（跳过）: {e}")
    return out


def _scan_vae(force_sha: bool) -> list:
    """独立 VAE：`models/` + `weights/` 下**名字里带 vae** 的权重文件。

    沿用 `run_load_vae` / `param_sources._scan_vae` 的搜索范围与判据
    （项目没有独立的 VAE 目录清单），保证"界面列得出的 VAE"与
    "清单里的 VAE"是同一批。
    """
    out = []
    try:
        from utils import paths
    except Exception as e:
        logger.debug(f"路径模块不可用（跳过 VAE 扫描）: {e}")
        return out

    for root in (paths.MODEL_DIR, paths.WEIGHTS_DIR):
        for p in _walk_files(root, _LORA_EXTS,
                             name_filter=lambda fn: "vae" in fn.lower()):
            try:
                out.append(_make_local(p, os.path.basename(p), "vae",
                                       "unknown", force_sha))
            except Exception as e:
                logger.debug(f"构造 VAE 记录失败（跳过）: {e}")
    return out


def _dir_kind(name: str) -> str:
    """目录名 → 资源类型（controlnet / upscaler），判不出返回 ""。"""
    low = _s(name).lower()
    if "controlnet" in low:
        return "controlnet"
    if "upscal" in low:
        return "upscaler"
    return ""


def _adapter_dirs() -> list:
    """[(目录, 资源类型)] —— ControlNet / 放大模型的候选目录。

    `paths.CONTROLNET_DIR` 一定按 ControlNet 处理（即使名字不匹配），
    再扫项目根 / `models/` / `weights/` 下名字含 controlnet / upscal 的
    目录。**目录不存在就返回空**，绝不报错 —— 这两类资源在多数机器上
    本来就没有。
    """
    candidates = []
    try:
        from utils import paths
    except Exception as e:
        logger.debug(f"路径模块不可用（跳过适配器目录扫描）: {e}")
        return candidates

    cdir = _s(getattr(paths, "CONTROLNET_DIR", ""))
    if cdir and os.path.isdir(cdir):
        candidates.append((cdir, "controlnet"))

    roots = (paths.PROJECT_ROOT, paths.MODEL_DIR, paths.WEIGHTS_DIR)
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            kind = _dir_kind(name)
            if not kind:
                continue
            full = os.path.join(root, name)
            if os.path.isdir(full):
                candidates.append((full, kind))

    # 去重（CONTROLNET_DIR 也可能是根目录下的 controlnets 命中项）
    out, seen = [], set()
    for full, kind in candidates:
        key = os.path.normcase(os.path.abspath(full))
        if key in seen:
            continue
        seen.add(key)
        out.append((full, kind))
    return out


def _scan_adapters(force_sha: bool) -> list:
    """ControlNet / 放大模型：扫候选目录里的权重文件。

    类型由目录名决定；架构只能是启发式（见 `_guess_arch_from_name`）。
    目录/文件一个都没有时返回 `[]` —— 这是正常情况，不是错误。
    """
    out = []
    for root, kind in _adapter_dirs():
        for p in _walk_files(root, _ADAPTER_EXTS):
            try:
                rel = os.path.relpath(p, root)
                arch = _guess_arch_from_name(rel)
                out.append(_make_local(p, os.path.basename(p), kind,
                                       arch, force_sha))
            except Exception as e:
                logger.debug(f"构造 {kind} 记录失败（跳过）: {e}")
    return out


def build_catalog(force_sha: bool = False) -> list:
    """扫描本机模型 → `list[ModelResource]`（类型/架构/名字排序）。

    `force_sha=False`（默认）时 **sha256 一律为空字符串**：
    这一步只 `stat` 取大小，不读文件内容。底模动辄数 GB，正常打开
    界面/跑校验绝不能顺带把十几个 GB 读一遍 —— 需要校验哈希时，
    由调用方显式 `force_sha=True`（或下载完成后调 `_sha_of`），
    而且 `file_sha256()` 会命中磁盘缓存，不会重复算。

    扫描范围：底模（`utils.model_scanner`）、`loras/`、VAE
    （`models/` + `weights/`）、ControlNet / 放大模型目录。
    `engine` 类资源不在这里产出（它由扩展市场写入，见 `RESOURCE_TYPES`
    的注释）。任何一个子扫描失败都只记 debug 日志，返回已收集的部分。
    """
    resources = []
    seen = set()

    def _push(items):
        for r in items:
            try:
                key = os.path.normcase(os.path.abspath(r.path)) if r.path \
                    else f"id:{r.resource_id}"
                if key in seen:
                    logger.debug(f"同名/同路径资源已收录，跳过重复项: {r.path or r.resource_id}")
                    continue
                seen.add(key)
                resources.append(r)
            except Exception as e:
                logger.debug(f"收录资源失败（跳过）: {e}")

    for scanner, label in (
        (_scan_checkpoints, "底模"),
        (_scan_loras, "LoRA"),
        (_scan_vae, "VAE"),
        (_scan_adapters, "ControlNet/放大"),
    ):
        try:
            _push(scanner(force_sha))
        except Exception as e:
            logger.warning(f"⚠️ {label} 扫描失败（忽略该类）: {e}")

    resources.sort(key=lambda r: (r.type, r.architecture, r.name.lower()))
    return resources


# ── 进程内缓存 ───────────────────────────────────────────────────────
_CATALOG_CACHE = None
_CATALOG_LOCK = threading.Lock()


def catalog() -> list:
    """本机清单（**带缓存**）：一个进程最多重建一次。

    为什么要缓存：解析一个引用的顺序是 `resource_id → sha256 →
    filename → local_path`，每步都要在清单里查；一张工作流几十个引用、
    校验/预览/执行各来一遍，不缓存就是反复扫盘。用户新放了模型时调
    `invalidate()`（或重启）即可看到。

    返回的是**新列表**（元素是缓存里的同一批记录对象，`to_ref(with_sha=True)`
    算出的哈希会写回记录并在进程内复用）——
    调用方增删/替换这个列表不会污染缓存。
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        with _CATALOG_LOCK:
            if _CATALOG_CACHE is None:
                try:
                    _CATALOG_CACHE = build_catalog()
                except Exception as e:
                    logger.warning(f"⚠️ 构建模型清单失败（按空清单处理）: {e}")
                    _CATALOG_CACHE = []
    return list(_CATALOG_CACHE)


def invalidate() -> None:
    """清空清单缓存：用户刚放/删了模型，或刚写完 `models_cache/resources.json`。"""
    global _CATALOG_CACHE
    with _CATALOG_LOCK:
        _CATALOG_CACHE = None


def by_id(resource_id: str) -> "ModelResource | None":
    """按 `resource_id` 查（大小写不敏感兜底）。找不到返回 None。"""
    key = _s(resource_id)
    if not key:
        return None
    low = key.lower()
    for r in catalog():
        if r.resource_id == key:
            return r
    for r in catalog():
        if r.resource_id.lower() == low:
            return r
    return None


def by_filename(filename: str) -> "ModelResource | None":
    """按**文件名**（basename，大小写不敏感）查。找不到返回 None。

    这是"老工作流兜底"的关键：老文件里只存了一个名字/路径，
    仍能在这里找到对应资源，从而升级成带 `resource_id` 的引用。
    """
    key = os.path.basename(_s(filename).replace("\\", "/"))
    if not key:
        return None
    low = key.lower()
    for r in catalog():
        if os.path.basename(r.path).lower() == low \
                or os.path.basename(r.name).lower() == low:
            return r
    return None


def by_sha256(sha: str) -> "ModelResource | None":
    """按 sha256 查。清单里没算过哈希则一定找不到（空哈希不算匹配）。"""
    key = _norm_sha(sha)
    if not key:
        return None
    for r in catalog():
        if r.sha256 and r.sha256 == key:
            return r
    return None


# ======================================================================
#  引用（写进工作流的那一层）
# ======================================================================
#: 引用的标准键集 —— 都是与机器无关（或可选）的字段，**不含期望路径**。
_REF_KEYS = ("resource_id", "filename", "model_type", "architecture",
             "sha256", "source", "source_url", "local_path")


def _blank_ref() -> dict:
    """空引用：键齐全，值都是空/None，便于上层直接 `.get()`。"""
    return {"resource_id": "", "filename": "", "model_type": "",
            "architecture": "", "sha256": "", "source": "",
            "source_url": "", "local_path": None}


def to_ref(resource: ModelResource, *, with_sha: bool = False,
           with_local_path: bool = False) -> dict:
    """资源记录 → 写进工作流的引用字典。

    返回：`{"resource_id", "filename", "model_type", "architecture",
    "sha256", "source", "source_url", "local_path"}`

    `local_path` **默认 None** —— 这是"跨机器可迁移"的关键：工作流
    描述"我要哪个模型"，而不是"我上次把它放在哪个绝对路径"。
    只有"就在本机保存/继续"这类场景才 `with_local_path=True`
    （写实际路径），导入到别的机器时该字段会被忽略（只作兜底）。

    `sha256` 默认写空串（见模块文档的"为什么 sha 默认不算"）；
    需要强校验时 `with_sha=True`，此时若记录里还没算过，会调
    `file_sha256()` 现算（有磁盘缓存，同一文件第二次几乎免费），
    并把结果**写回记录** —— 这样 `by_sha256()` 立刻能按哈希反查，
    后续引用/`save_catalog()` 也不必重算。
    """
    try:
        if resource is None:
            return _blank_ref()
        sha = _norm_sha(getattr(resource, "sha256", ""))
        if with_sha and not sha:
            sha = _sha_of(_s(getattr(resource, "path", "")))
            if sha:
                try:
                    resource.sha256 = sha
                except Exception as e:
                    logger.debug(f"哈希写回记录失败（不影响本次使用）: {e}")
        path = _s(getattr(resource, "path", ""))
        ref = {
            "resource_id": _s(getattr(resource, "resource_id", "")),
            "filename": os.path.basename(path) or _s(getattr(resource, "name", "")),
            "model_type": _s(getattr(resource, "type", "")),
            "architecture": _s(getattr(resource, "architecture", "")),
            "sha256": sha,
            "source": _s(getattr(resource, "source", "")),
            "source_url": _s(getattr(resource, "source_url", "")),
            "local_path": path if (with_local_path and path) else None,
        }
        return ref
    except Exception as e:
        logger.debug(f"生成资源引用失败（返回空引用）: {e}")
        return _blank_ref()


def _ref_field(ref: dict, *names) -> str:
    """从引用里按顺序取第一个非空字段（兼容 `model_type` / `type` 等别名）。"""
    for n in names:
        v = _s(ref.get(n))
        if v:
            return v
    return ""


def resolve(ref: dict) -> tuple:
    """引用 → `(ModelResource | None, status, detail)`。

    `status` 取值与含义：

    ==================  ====================================================
    `ok`                命中，且（给了 sha 且本地算过时）哈希一致、架构同族
    `found_elsewhere`   找到了文件，但**不是**按它自己声明的身份命中的
                        （比如只靠文件名在别的目录里捞到）
    `missing`           本地完全找不到
    `arch_mismatch`     找到了，但架构与引用声明的不符（下载错了/放错目录）
    `sha_mismatch`      找到了，但哈希与引用声明的不一致（同名不同版本）
    ==================  ====================================================

    解析顺序：`resource_id` → `sha256` → `filename` → `local_path` 兜底。
    先按 ID（最可靠、可迁移），ID 不在清单里再看哈希（内容同一即同一
    模型），再退化到文件名（老工作流），最后才看引用里记的路径。

    **绝不抛异常**：畸形/残缺的引用（`{}`、`None`、字段类型不对）
    一律按 `missing` 返回并记 debug 日志 —— 一个坏引用不该让整张
    工作流的校验或报告挂掉。
    """
    try:
        return _resolve_inner(ref)
    except Exception as e:
        logger.debug("解析资源引用失败（按缺失处理）", exc_info=True)
        return None, "missing", f"解析引用时出错（已按缺失处理）: {e}"


def _resolve_inner(ref) -> tuple:
    if not isinstance(ref, dict):
        return None, "missing", "引用不是字典（已按缺失处理）"

    rid = _s(ref.get("resource_id"))
    sha = _norm_sha(ref.get("sha256"))
    fname = _s(ref.get("filename"))
    arch = _s(ref.get("architecture"))
    mtype = _ref_field(ref, "model_type", "type")
    lpath = _s(ref.get("local_path"))

    if not (rid or sha or fname or lpath):
        return None, "missing", "引用里没有任何可用字段（resource_id/sha256/filename/local_path 全空）"

    resource = None
    how = ""

    if rid:
        resource = by_id(rid)
        how = "resource_id" if resource is not None else ""
    if resource is None and sha:
        resource = by_sha256(sha)
        how = "sha256" if resource is not None else ""
    if resource is None and fname:
        resource = by_filename(fname)
        how = "filename" if resource is not None else ""
    if resource is None and lpath and os.path.isfile(lpath):
        # 兜底：引用记的路径上确实有文件 —— 现场建一条记录
        # （不进清单缓存，只是"本次能用"）。
        resource = _make_local(lpath, os.path.basename(lpath),
                               mtype or _guess_type_from_name(lpath),
                               arch or "unknown", False)
        how = "local_path"

    if resource is None:
        name = rid or fname or lpath
        detail = f"本地找不到该资源：{name}"
        if arch:
            detail += f"（期望架构 {arch}）"
        return None, "missing", detail

    # ── 校验：哈希 > 架构 > 位置 ──────────────────────────────
    # 哈希最强（同名不同版本只能靠它区分）；架构次之（下错架构是硬伤）；
    # 位置最弱（只是"没按预期身份命中"）。
    if sha and resource.sha256 and sha != resource.sha256:
        return resource, "sha_mismatch", (
            f"哈希不一致：引用 {sha[:12]}… 本地 {resource.sha256[:12]}…"
            f"（文件：{resource.path}）—— 很可能是同名不同版本")

    if arch and not _arch_compatible(arch, resource.architecture):
        return resource, "arch_mismatch", (
            f"架构不符：引用声明 {arch}，本地该资源是 {resource.architecture}"
            f"（文件：{resource.path}）—— 检查是否下载/放错了架构目录")

    if how == "local_path":
        return resource, "found_elsewhere", (
            f"清单里没有这条资源，但引用给出的路径上确实有文件：{resource.path}"
            f"（按外部/临时路径使用）")
    if how == "filename" and rid:
        return resource, "found_elsewhere", (
            f"按 resource_id（{rid}）没找到，但同名文件在别处存在："
            f"{resource.path} —— 可能是换机后目录不同或版本不同")

    if how == "filename":
        detail = f"按文件名找到：{resource.path}"
    elif how == "sha256":
        detail = f"按 sha256 命中：{resource.path}"
    else:
        detail = f"匹配 {resource.resource_id}：{resource.path}"
    if sha and not resource.sha256:
        detail += "（本地未计算 sha256，未能校验）"
    return resource, "ok", detail


def resolve_many(refs: list) -> list:
    """批量解析 → `[{"ref", "resource", "status", "detail", "present"}]`。

    `present` = 本地确实可用（`ok` / `found_elsewhere`）。
    `arch_mismatch` / `sha_mismatch` 的文件虽然在本地，但**不算可用**
    （用错了模型比没有更糟），它们既不计 present 也不计 missing。
    输入不是列表、元素不是 dict 都只按 missing 处理，绝不抛。
    """
    out = []
    try:
        items = list(refs) if refs else []
    except TypeError:
        logger.debug("resolve_many 收到不可迭代对象（按空列表处理）")
        items = []
    for ref in items:
        try:
            resource, status, detail = resolve(ref)
        except Exception as e:      # resolve 已兜底，这里再保一层
            logger.debug(f"解析引用异常（按缺失处理）: {e}")
            resource, status, detail = None, "missing", f"解析异常（已按缺失处理）: {e}"
        present = bool(resource is not None and status in ("ok", "found_elsewhere"))
        out.append({"ref": ref, "resource": resource, "status": status,
                    "detail": detail, "present": present})
    return out


def missing_report(refs: list) -> dict:
    """资源体检报告（给"打开工作流先看缺什么"的界面用）。

    返回：
        {
          "ok": bool,              # 无缺失、无不匹配
          "present": [...],        # 可用的条目（resolve_many 的元素）
          "missing": [...],        # 本地没有的
          "mismatched": [...],     # 架构/哈希不符的（有文件但用错）
          "downloadable": [...],   # status=="missing" 且 source_url 非空
          "total": int,
          "summary": str,          # 中文一句话
        }

    `downloadable` 就是界面拿去渲染"资源卡片 + 去下载"按钮的那批条目：
    能下载的前提是引用里带了 `source_url`（来源清单/扩展市场写入），
    本模块**只读不下载**（与 `workflow/resource_manager.py` 的策略一致：
    执行工作流绝不联网）。
    """
    report = {"ok": True, "present": [], "missing": [], "mismatched": [],
              "downloadable": [], "total": 0, "summary": ""}
    try:
        entries = resolve_many(refs)
    except Exception as e:
        logger.debug(f"生成资源报告失败（返回空报告）: {e}", exc_info=True)
        report["summary"] = f"资源检查失败（已忽略）: {e}"
        return report

    report["total"] = len(entries)
    for e in entries:
        st = e.get("status")
        if st in ("ok", "found_elsewhere"):
            report["present"].append(e)
        elif st == "missing":
            report["missing"].append(e)
        else:
            report["mismatched"].append(e)
        if st == "missing":
            ref = e.get("ref")
            if isinstance(ref, dict) and _s(ref.get("source_url")):
                report["downloadable"].append(e)

    report["ok"] = not report["missing"] and not report["mismatched"]

    summary = (f"共检查 {report['total']} 项模型资源："
               f"{len(report['present'])} 项可用，"
               f"{len(report['missing'])} 项缺失，"
               f"{len(report['mismatched'])} 项不匹配")
    if report["downloadable"]:
        summary += f"；其中 {len(report['downloadable'])} 项提供了下载地址，可直接下载"
    elif report["missing"]:
        summary += "；缺失项没有下载地址，需要自行获取"
    report["summary"] = summary + "。"
    return report


# ======================================================================
#  工作流集成（纯函数，不碰界面）
# ======================================================================
#: 节点类型 → 承载模型引用的参数名（与 nodes/model_nodes.py 的 ParamSpec
#: 一一对应：LoadCheckpoint.path / LoadVAE.path / LoadLoRA.lora_1..3）
_NODE_MODEL_PARAMS = {
    "LoadCheckpoint": ("path",),
    "LoadVAE": ("path",),
    "LoadLoRA": ("lora_1", "lora_2", "lora_3"),
}
#: 节点类型 → 缺省资源类型（老工作流退化成文件名引用时用）
_NODE_MODEL_TYPE = {
    "LoadCheckpoint": "checkpoint",
    "LoadVAE": "vae",
    "LoadLoRA": "lora",
}


def _ref_from_path(path: str, model_type: str = "") -> dict:
    """老工作流兜底：由一个路径/文件名字符串造一份引用。

    能查到本机清单时顺手补上 `resource_id` / `architecture` / `sha256`，
    这样"老工作流"也能升级成"按 ID 解析"，并如实报告缺失；
    查不到就只留文件名（检查时按文件名找，这也是它唯一能做的）。
    `local_path` 保留原样 —— 它是这份老引用仅有的定位线索。
    """
    raw = _s(path)
    ref = _blank_ref()
    name = os.path.basename(raw.replace("\\", "/").rstrip("/")) or raw
    ref["filename"] = name
    ref["model_type"] = model_type
    ref["source"] = "local"
    ref["local_path"] = raw or None
    try:
        r = by_filename(name)
    except Exception as e:
        logger.debug(f"按文件名查清单失败（继续用退化引用）: {e}")
        r = None
    if r is not None:
        ref["resource_id"] = r.resource_id
        ref["architecture"] = r.architecture
        ref["model_type"] = model_type or r.type
        ref["sha256"] = r.sha256
        ref["source"] = r.source or "local"
        ref["source_url"] = r.source_url
    return ref


def refs_for_node(node) -> list:
    """一个节点上的全部模型引用（`LoadCheckpoint` / `LoadVAE` / `LoadLoRA`）。

    两种参数形态都支持：
      · **资源引用模式**：参数里同时有 `resource_id` + `filename`
        （LoRA 多槽位用 `resource_id_1` / `filename_1` …；单路径节点
        用扁平键）。这是新工作流的写法，直接用。
      · **老写法**：只有 `path` / `lora_N` 字符串 → 退化成按文件名构造
        一份引用（见 `_ref_from_path`），保证老工作流仍能被检查。

    不是模型节点、params 缺失/类型不对、未知节点类型 → 返回 `[]`，不抛。
    """
    try:
        return _refs_for_node_inner(node)
    except Exception as e:
        logger.debug(f"收集节点资源引用失败（按空处理）: {e}", exc_info=True)
        return []


def _refs_for_node_inner(node) -> list:
    ntype = _s(getattr(node, "type", ""))
    names = _NODE_MODEL_PARAMS.get(ntype)
    if not names:
        return []
    params = getattr(node, "params", None)
    if not isinstance(params, dict):
        params = {}

    out = []
    for i, pname in enumerate(names, start=1):
        value = _s(params.get(pname))
        # 资源引用模式：先看本槽位的键，再看扁平键（第 1 槽位）
        rid = _s(params.get(f"resource_id_{i}"))
        fname = _s(params.get(f"filename_{i}"))
        if not (rid and fname) and i == 1:
            rid = rid or _s(params.get("resource_id"))
            fname = fname or _s(params.get("filename"))

        if rid and fname:
            ref = _blank_ref()
            ref["resource_id"] = rid
            ref["filename"] = fname
            ref["model_type"] = _s(params.get(f"model_type_{i}")) \
                or _s(params.get("model_type")) or _NODE_MODEL_TYPE.get(ntype, "")
            ref["architecture"] = _s(params.get(f"architecture_{i}")) \
                or _s(params.get("architecture"))
            ref["sha256"] = _s(params.get(f"sha256_{i}")) \
                or _s(params.get("sha256"))
            ref["source"] = _s(params.get("source"))
            ref["source_url"] = _s(params.get(f"source_url_{i}")) \
                or _s(params.get("source_url"))
            ref["local_path"] = value or None
            out.append(ref)
        elif value:
            out.append(_ref_from_path(value, _NODE_MODEL_TYPE.get(ntype, "")))
    return out


def collect_refs(wf) -> list:
    """从工作流收集**去重后**的模型引用（任何带 `.nodes` 的对象都行）。

    去重键 = `resource_id`（有则用）否则 `filename`（小写）—— 同一底模
    在同一张图里被 LoadCheckpoint 与 LoadVAE 各引一次只算一项，
    报告里也不会重复刷屏。

    `wf` 为 None / 没有 `.nodes` / 某个节点畸形 → 跳过，绝不抛。
    """
    try:
        nodes = list(getattr(wf, "nodes", None) or [])
    except Exception as e:
        logger.debug(f"读取工作流节点失败（按空处理）: {e}")
        return []

    out, seen = [], set()
    for n in nodes:
        for ref in refs_for_node(n):
            try:
                key = _s(ref.get("resource_id")) or \
                    os.path.basename(_s(ref.get("filename"))).lower()
            except Exception:
                key = ""
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ref)
    return out


# ======================================================================
#  持久化（让一台机器可以把清单"发出去"）
# ======================================================================
def catalog_path(path=None) -> str:
    """清单文件路径：默认 `models_cache/resources.json`。

    `models_cache/` 与哈希缓存同目录：两者都是"可重建的本地缓存"，
    一起被清理/一起被忽略都合理。
    """
    target = _s(path)
    if target:
        return os.path.abspath(target)
    try:
        from utils import paths
        return os.path.join(paths.CACHE_DIR, _CATALOG_FILE)
    except Exception as e:
        logger.debug(f"取缓存目录失败（回落到当前目录）: {e}")
        return os.path.abspath(_CATALOG_FILE)


def save_catalog(path=None, resources=None) -> bool:
    """把清单写成 JSON（默认 `models_cache/resources.json`）。

    `resources=None` 时保存当前 `catalog()`。格式带 `version` 外壳，
    便于以后加字段；`load_catalog()` 同时兼容"裸列表"的老格式。
    写失败（无权限/磁盘满）返回 False，只记日志 —— 存清单失败不该
    影响用户正在做的事。
    """
    target = catalog_path(path)
    try:
        items = list(resources) if resources is not None else catalog()
    except Exception as e:
        logger.warning(f"⚠️ 取清单失败，未写入 {target}: {e}")
        return False

    payload = {
        "version": _CATALOG_VERSION,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(items),
        "resources": [r.to_dict() if hasattr(r, "to_dict") else dict(r)
                      for r in items],
    }
    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning(f"⚠️ 写入资源清单失败 {target}: {e}")
        return False


def load_catalog(path=None) -> list:
    """读 JSON 清单 → `list[ModelResource]`。

    文件不存在 / JSON 损坏 / 单个条目畸形都只跳过（记日志），
    返回能读出来的部分 —— 清单是**辅助**信息，坏掉不该让调用方崩。
    支持两种格式：`{"resources": [...]}`（本模块写的）与裸列表（手写）。
    """
    target = catalog_path(path)
    if not os.path.isfile(target):
        return []
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ 读取资源清单失败（按空清单处理）{target}: {e}")
        return []

    items = data.get("resources") if isinstance(data, dict) else data
    if not isinstance(items, list):
        logger.warning(f"⚠️ 资源清单格式不认识（按空清单处理）: {target}")
        return []

    out = []
    for item in items:
        try:
            out.append(ModelResource.from_dict(item))
        except Exception as e:
            logger.debug(f"清单条目解析失败（跳过）: {e}")
    return out
