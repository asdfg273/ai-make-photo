# core/ipa_multi.py
"""IP-Adapter **多参考图**：角色（身份/风格/构图）、每图权重、区域掩码。

为什么单独一个模块
------------------
"多参考图"这套语义（角色 → 权重文件、权重 → 每个适配器的 scale、参考图 →
`ip_adapter_image` 列表、掩码 → `ip_adapter_masks`）同时被三个地方需要：

· `nodes/ipa_nodes.py`（Identity / Style / Composition 三个累加节点）
· `nodes/adapter_nodes.py`（`IPAdapterApply` 的 `refs_json` 多参考入口）
· `core/model_manager.py`（一次安装 N 个适配器）

放在 core 里、且**不 import torch/diffusers/PIL 之外的重依赖**，好处是这些
规则可以用纯脚本（无 GPU、无界面）断言，而不是只能靠"跑一张图看看"。

角色 → 权重文件的映射是**真的**（h94/IP-Adapter 仓库真实存在的三个文件）
--------------------------------------------------------------------
已核对 `ModelManager._IPA_WEIGHTS` 与 h94/IP-Adapter 的目录布局：

    身份 identity    → plus-face  `ip-adapter-plus-face_sd15.safetensors`
                       （人脸专用，官方文档明确用于 face transfer）
    风格 style       → plus       `ip-adapter-plus_sd15.safetensors`
                       （plus = 更强的图像条件，官方推荐做风格迁移）
    构图 composition → base       `ip-adapter_sd15.safetensors`
                       （原始版，全局构图/色彩引导，逐层影响最均衡）

也就是说"角色"不是一个随便起的名字：三个角色对应三个**训练目标不同**的
权重文件。本机只装了 `base`（扩展市场 `ip_adapter_sd15` 装的就是它），
所以 `resolve_role` 会在文件缺失时**降级到已安装的那一个**并留下中文警告，
而不是整条工作流失败。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
#  角色定义
# ============================================================

#: 角色 → ModelManager 认的权重档位（variant）。
#: 取值必须是 `ModelManager._IPA_WEIGHTS` 里真实存在的名字。
ROLE_VARIANTS = {
    "identity": "plus-face",     # 身份：人脸专用
    "style": "plus",             # 风格：plus
    "composition": "base",       # 构图：基础版
}

#: 中文显示名（报错与元数据里用中文，用户不用猜英文缩写）
ROLE_LABELS = {
    "identity": "身份",
    "style": "风格",
    "composition": "构图",
}

#: 反查：档位 → 角色（同一个档位可能被多个角色请求）
VARIANT_ROLES = {v: k for k, v in ROLE_VARIANTS.items()}

#: 各种写法归一化到内部角色名（界面/JSON 里用户可能写中文或英文）
_ROLE_ALIASES = {
    "identity": "identity", "id": "identity", "face": "identity",
    "person": "identity", "character": "identity",
    "身份": "identity", "人物": "identity", "人物身份": "identity",
    "脸": "identity", "人脸": "identity",
    "style": "style", "st": "style", "风格": "style", "画风": "style",
    "composition": "composition", "comp": "composition", "layout": "composition",
    "构图": "composition", "结构": "composition", "布局": "composition",
}

#: 权重档位别名（界面下拉写 "standard"，h94 官方叫 "base"）
VARIANT_ALIASES = {
    "standard": "base", "base": "base", "light": "base",
    "plus": "plus", "plus_face": "plus-face", "plus-face": "plus-face",
    "plusface": "plus-face", "face": "plus-face",
}

DEFAULT_ROLE = "style"
DEFAULT_WEIGHT = 0.6


def normalize_role(value: Any) -> str:
    """把任意写法归一化成 identity / style / composition（不认识的按 style）。"""
    key = str(value or "").strip().lower().replace(" ", "").replace("-", "")
    if not key:
        return DEFAULT_ROLE
    # 允许 "plus-face" / "plus_face" 这种直接给档位的写法
    key2 = key.replace("_", "-")
    if key2 == "plus-face":
        return "identity"
    return _ROLE_ALIASES.get(key, _ROLE_ALIASES.get(key2, DEFAULT_ROLE))


def normalize_variant(value: Any) -> str:
    """把界面/JSON 里的档位写法归一化成 base / plus / plus-face。"""
    key = str(value or "").strip().lower().replace(" ", "")
    return VARIANT_ALIASES.get(key, key or "plus")


def role_label(role: str) -> str:
    return ROLE_LABELS.get(normalize_role(role), "风格")


def variant_for_role(role: str) -> str:
    """该角色**理想**应该用的权重档位（可能本机没装）。"""
    return ROLE_VARIANTS.get(normalize_role(role), ROLE_VARIANTS[DEFAULT_ROLE])


def role_for_variant(variant: str) -> str:
    """档位反查角色（`standard` 会先归一化成 `base`）。"""
    return VARIANT_ROLES.get(normalize_variant(variant), DEFAULT_ROLE)


# ============================================================
#  参考图条目
# ============================================================

@dataclass
class IpaReference:
    """一条参考图定义：图 + 角色 + 权重 +（可选）只作用于某区域的掩码。

    字段
    ----
    role        : identity / style / composition
    weight      : 该参考图的 IP-Adapter 强度（0~2），**直接就是** diffusers
                  `set_ip_adapter_scale()` 的 per-adapter scale
    image       : PIL.Image（内存里的图，优先级高于 path）
    path        : 图片路径（没有 image 时从这里读；也可能为空）
    mask        : PIL.Image "L"（255=该参考图只影响这块）
    mask_path   : 掩码路径（没有 mask 时从这里读）
    variant     : 实际使用的档位（由 ModelManager 解析后回填）
    variant_requested : 角色"理想"档位（缺文件降级时用来说明差距）
    enabled     : False 表示这条只是占位（保持适配器编号对齐，不参与影响）
    note        : 降级/跳过的中文说明
    """

    role: str = DEFAULT_ROLE
    weight: float = DEFAULT_WEIGHT
    image: Any = None
    path: str = ""
    mask: Any = None
    mask_path: str = ""
    variant: str = ""
    variant_requested: str = ""
    enabled: bool = True
    note: str = ""

    def __post_init__(self):
        self.role = normalize_role(self.role)
        self.weight = _f(self.weight, DEFAULT_WEIGHT)
        self.variant_requested = (normalize_variant(self.variant_requested)
                                  or variant_for_role(self.role))
        if self.variant:
            self.variant = normalize_variant(self.variant)
        self.path = _s(self.path)
        self.mask_path = _s(self.mask_path)
        self.note = _s(self.note)

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        d = {
            "role": self.role,
            "weight": self.weight,
            "path": self.path,
            "mask_path": self.mask_path,
            "enabled": bool(self.enabled),
        }
        if self.variant:
            d["variant"] = self.variant
        if self.variant_requested:
            d["variant_requested"] = self.variant_requested
        if self.note:
            d["note"] = self.note
        return d

    def describe(self) -> str:
        """人能一眼看懂的一行摘要（进元数据/状态栏）。"""
        tag = f"{role_label(self.role)}"
        if self.variant:
            tag += f"({self.variant})"
        src = (os.path.basename(self.path) if self.path
               else ("内存图" if self.image is not None else "无图"))
        extra = "·区域" if (self.mask is not None or self.mask_path) else ""
        off = "" if self.enabled else "·停用"
        return f"{tag} {self.weight:.2f} ← {src}{extra}{off}"


def _f(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _s(value) -> str:
    return "" if value is None else str(value).strip()


def _b(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "是", "开")
    return bool(value)


# ============================================================
#  JSON 解析（**绝不抛异常**）
# ============================================================

def references_from_json(text: Any) -> List[IpaReference]:
    """解析参考图 JSON → `list[IpaReference]`。

    形状（与区域功能的 `regions_json` 同一风格）::

        [
          {"role":"identity",    "path":"E:/refs/face.png", "weight":0.8},
          {"role":"style",       "path":"E:/refs/style.jpg","weight":0.6},
          {"role":"composition", "path":"E:/refs/layout.png","weight":0.45,
           "mask_path":"E:/masks/sky.png"},
          {"role":"style", "weight":0.5, "enabled":false}      ← 占位停用
        ]

    容错规则（**任何垃圾输入都返回 list，不抛异常**）：
      · 非字符串 / 空串 / 非法 JSON → `[]`
      · 外层带 `{"refs": [...]}` / `{"references": [...]}` 包装 → 取内层
      · 单个对象（不是数组）→ 当成一条
      · 条目不是 dict → 丢弃；缺 role → style；缺 weight → 0.6
      · 权重自动夹到 0~2（越界不报错，夹取 + 记录）

    为什么这么强调容错：这段 JSON 会从工作流文件、界面文本框、旧版本文件
    三个地方进来。任何一种格式漂移都不该让整张图跑不出来 —— 用户应该看到
    "这条参考图被跳过了/权重被夹取了"，而不是一个 JSONDecodeError。
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return coerce_references(text)
    if isinstance(text, dict):
        return coerce_references(text)
    raw = str(text).strip()
    if not raw or raw in ("[]", "{}"):
        return []
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"⚠️ 参考图 JSON 解析失败，按空列表处理: {e}")
        return []

    if isinstance(data, dict):
        inner = None
        for key in ("refs", "references", "ref_images", "参考图"):
            if key in data:
                inner = data[key]
                break
        if inner is not None:
            data = inner
        elif _dict_looks_like_ref(data):
            data = [data]          # 单个参考图对象
        else:
            # 一个"看着不像参考图"的字典（例如 {"a":1}）：按空列表处理。
            # 不能硬当成一条参考图 —— 那样会静默产生一条"没有图"的参考图，
            # 到了采样才报"参考图读不到"，用户完全不知道是自己 JSON 写错了。
            logger.warning(f"⚠️ 参考图 JSON 里没有可识别的字段，按空列表处理: "
                           f"{list(data)[:6]}")
            return []
    if not isinstance(data, (list, tuple)):
        logger.warning(f"⚠️ 参考图 JSON 不是数组（{type(data).__name__}），按空列表处理")
        return []
    return coerce_references(data)


#: 一个 dict 里出现这些键，才算"这是一条参考图"
_REF_KEYS = ("path", "image_path", "file", "src", "image", "mask_path",
             "mask", "role", "kind", "type", "variant", "adapter")


def _dict_looks_like_ref(d: dict) -> bool:
    return any(k in d for k in _REF_KEYS)


def coerce_references(items: Iterable[Any]) -> List[IpaReference]:
    """把 dict / IpaReference 混合列表规范化成 `list[IpaReference]`（不抛）。"""
    out: List[IpaReference] = []
    if items is None:
        return out
    if isinstance(items, (IpaReference, dict)):
        items = [items]
    for item in items:
        if isinstance(item, IpaReference):
            out.append(item)
            continue
        if not isinstance(item, dict):
            logger.warning(f"⚠️ 忽略无法识别的参考图条目: {item!r}")
            continue
        out.append(_ref_from_dict(item))
    return out


def _ref_from_dict(d: dict) -> IpaReference:
    """单个条目 → IpaReference（字段缺省/类型怪异都不抛）。"""
    # 图片：兼容 path / image_path / file / 名字几种写法
    path = ""
    for key in ("path", "image_path", "file", "src", "image"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            path = v.strip()
            break
    mask_path = ""
    for key in ("mask_path", "mask", "region_mask"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            mask_path = v.strip()
            break

    role_raw = d.get("role", d.get("kind", d.get("type")))
    variant_raw = d.get("variant", d.get("adapter"))
    if variant_raw and not role_raw:
        role_raw = role_for_variant(str(variant_raw))

    weight = _f(d.get("weight", d.get("scale", d.get("strength"))),
                DEFAULT_WEIGHT)
    if weight < 0.0:
        logger.warning(f"⚠️ 参考图权重 {weight} 越界，已夹到 0.0")
        weight = 0.0
    elif weight > 2.0:
        logger.warning(f"⚠️ 参考图权重 {weight} 越界，已夹到 2.0")
        weight = 2.0

    ref = IpaReference(
        role=normalize_role(role_raw),
        weight=weight,
        path=path,
        mask_path=mask_path,
        variant=(normalize_variant(variant_raw) if variant_raw else ""),
        enabled=_b(d.get("enabled"), True),
    )
    return ref


def references_to_json(refs: Iterable[Any]) -> str:
    """`list[IpaReference|dict]` → JSON 文本（写回界面/工作流用）。"""
    items = []
    for r in (refs or []):
        items.append(r.to_dict() if isinstance(r, IpaReference) else dict(r))
    return json.dumps(items, ensure_ascii=False, indent=2)


def describe_references(refs: Iterable[Any]) -> str:
    """一串参考图的中文摘要（元数据/状态栏）。"""
    items = coerce_references(refs)
    if not items:
        return "无参考图"
    return " + ".join(r.describe() for r in items)


# ============================================================
#  角色 → 实际档位（优雅降级）
# ============================================================

def resolve_role_plan(refs: Iterable[Any],
                      installed: Optional[Iterable[str]] = None,
                      *, on_degrade=None):
    """为每条参考图选定**实际可用**的档位，返回 `(计划列表, 警告列表)`。

    规则（按优先级）：
      1. 显式指定了 `variant` 且本机装了 → 用它（用户说了算）；
      2. 角色对应的档位装了 → 用它（**这是真正实现角色语义的情况**）；
      3. 本机装了别的档位 → 用**已安装里优先级最高的**那个替代，
         角色语义退化为"只是权重不同"，并记一条中文警告；
      4. 一个都没装 → 抛 `RuntimeError`（含缺哪个文件、怎么装）。
         这种情况绝不能静默继续：装 N 个不存在的适配器会变成"没效果"，
         用户只会以为"开了没用"。

    ⚠️ 第 3 步的降级是**有意为之**的：本机只有 base 时，"身份/风格/构图"
    三者拿到的是同一个权重文件，效果差异只剩权重。警告必须如实说出来，
    而不能假装角色生效了。

    `on_degrade(role, want, got, index)` 可选：每发生一次降级回调一次。
    节点（`nodes/ipa_nodes.py`）用它在**当时**就写一条警告，
    采样层再据此去重，避免同一个问题在元数据里出现两遍。

    `installed` 传 None 时由调用方（通常是 ModelManager）负责填；
    这里只做纯逻辑判断，方便无 GPU 测试。
    """
    items = coerce_references(refs)
    if not items:
        return [], []

    inst = [normalize_variant(v) for v in (installed or [])]
    # 去重但保持顺序（顺序 = 优先级，由 installed_ipa_variants 排好）
    seen = set()
    inst = [v for v in inst if not (v in seen or seen.add(v))]

    if not inst:
        need = sorted({variant_for_role(r.role) for r in items})
        raise RuntimeError(
            "❌ 本机没有任何可用的 IP-Adapter 权重，无法建立多参考图。\n"
            f"   需要的文件（按角色）: {', '.join(need)}\n"
            "   请到「工具 → 🛒 扩展市场 → IP-Adapter SD1.5」安装，"
            "或把 ip-adapter*.safetensors 放进 models/ip_adapter/。")

    plan: List[IpaReference] = []
    warnings: List[str] = []
    for i, r in enumerate(items, start=1):
        want = r.variant or variant_for_role(r.role)
        if want in inst:
            chosen, fell_back = want, False
        else:
            chosen, fell_back = inst[0], True

        # 复制一份，避免修改调用方传进来的对象（界面可能正持有它）
        nr = IpaReference(
            role=r.role, weight=r.weight, image=r.image, path=r.path,
            mask=r.mask, mask_path=r.mask_path, variant=chosen,
            variant_requested=want, enabled=r.enabled, note=r.note)
        if fell_back:
            msg = (f"参考图 {i}（{role_label(nr.role)}）需要档位「{want}」，"
                   f"本机未安装 → 降级使用「{chosen}」。"
                   f"角色语义（{role_label(nr.role)}专用权重）未生效，"
                   f"仅权重 {nr.weight:.2f} 生效。"
                   f"想要完整效果请安装 {want} 对应的权重文件。")
            nr.note = msg
            warnings.append(msg)
            if callable(on_degrade):
                try:
                    on_degrade(nr.role, want, chosen, i)
                except Exception as e:                # pragma: no cover - 防御
                    logger.warning(f"⚠️ on_degrade 回调失败: {e}")
        plan.append(nr)
    return plan, warnings


# ============================================================
#  参考图 → 真正能喂给管线的值
# ============================================================

def load_reference_image(ref: "IpaReference"):
    """把一条参考图变成 PIL.Image（内存图优先）；取不到返回 None。

    ⚠️ 调用方拿到的图**不要就地改**：这里始终返回新对象（PIL 的
    `convert()` 会建新图；如果本来就是 RGB 且没有 mask 要合成，也会 copy）。
    """
    img = getattr(ref, "image", None)
    if img is not None and hasattr(img, "convert"):
        try:
            return img.convert("RGB")
        except Exception:
            return None
    path = _s(getattr(ref, "path", ""))
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            out = im.convert("RGB")
            out.load()
        return out
    except Exception as e:
        logger.warning(f"⚠️ 参考图读取失败（{path}）: {e}")
        return None


def apply_reference_mask(ref: "IpaReference", image, *, background=(127, 127, 127)):
    """把参考图裁剪到它自己的掩码范围内，返回新图（原图不动）。

    为什么在**参考图**上动手而不只是限制注意力：IP-Adapter 的图像条件来自
    CLIP 对**整张参考图**的编码。如果只把注意力限制在区域上，编码出来的
    特征仍然混着区域外的内容（比如"只想把衣服换成参考图里的衣服"，
    结果把参考图的脸也一起编码进去了）。先把参考图按掩码抠出来再整张送去
    编码，区域外的内容就不会进 CLIP —— 这是"该参考图只影响该区域"里
    **必要**的一半（另一半是注意力掩码 `ip_adapter_masks`）。

    掩码来源：`ref.mask`（内存图）优先，其次 `mask_path`。都没有 → 原样返回。
    """
    if image is None:
        return None
    mask = getattr(ref, "mask", None)
    if mask is None:
        p = _s(getattr(ref, "mask_path", ""))
        if p and os.path.isfile(p):
            try:
                from PIL import Image
                with Image.open(p) as im:
                    mask = im.convert("L")
                    mask.load()
            except Exception as e:
                logger.warning(f"⚠️ 参考图掩码读取失败（{p}）: {e}")
                mask = None
    if mask is None:
        return image
    try:
        from PIL import Image
        m = mask.convert("L")
        if m.size != image.size:
            m = m.resize(image.size, Image.Resampling.BILINEAR)
        bg = Image.new("RGB", image.size, background)
        out = bg.copy()
        out.paste(image.convert("RGB"), (0, 0), m)
        return out
    except Exception as e:
        logger.warning(f"⚠️ 参考图掩码应用失败（改为整图参与）: {e}")
        return image


def build_ip_adapter_masks(refs: Iterable[Any], width: int, height: int,
                           *, scale: int = 8, device: str = "cpu"):
    """构造 `cross_attention_kwargs["ip_adapter_masks"]` 需要的张量列表。

    依据（已对 diffusers 0.40 源码核实）
    ------------------------------------
    `IPAdapterAttnProcessor2_0.forward(..., ip_adapter_masks=None)` 要求：

        ip_adapter_masks : list，长度 == self.scale 的长度 == ip_hidden_states 长度
        每个元素：None 或 shape [1, num_images_for_ip_adapter, height, width]
                  的 4 维张量（注释里写 "Please use IPAdapterMaskProcessor
                  to preprocess your mask"）

    而 `cross_attention_kwargs` 会被逐层透传到 attn2（见
    `models/attention.py` BasicTransformerBlock.forward → `**cross_attention_kwargs`），
    所以**不需要改 diffusers 源码**就能做到"每条参考图只管自己那块区域"。

    注意：这是**注意力层**的掩码，作用在潜空间分辨率上
    （`width // 8, height // 8`），不是像素级掩码。

    返回 `list[None | torch.Tensor]`，长度 == 参考图条数；
    没有任何掩码时返回 None（让调用方完全不传这个 kwargs）。
    """
    items = coerce_references(refs)
    if not items:
        return None
    if not any((r.mask is not None or r.mask_path) for r in items):
        return None

    import torch  # 局部 import：core/ipa_multi 在无 torch 的纯逻辑测试里也能用

    h = max(1, int(height) // int(scale or 8))
    w = max(1, int(width) // int(scale or 8))
    out = []
    for r in items:
        mask = getattr(r, "mask", None)
        if mask is None and _s(getattr(r, "mask_path", "")):
            p = _s(r.mask_path)
            if os.path.isfile(p):
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        mask = im.convert("L")
                        mask.load()
                except Exception as e:
                    logger.warning(f"⚠️ 掩码读取失败（{p}）: {e}")
                    mask = None
        if mask is None:
            out.append(None)
            continue
        try:
            from PIL import Image
            import numpy as _np
            m = mask.convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.Resampling.BILINEAR)
            arr = _np.asarray(m, dtype="float32") / 255.0
            t = torch.from_numpy(arr).view(1, 1, h, w).to(device=device,
                                                         dtype=torch.float32)
            out.append(t)
        except Exception as e:
            logger.warning(f"⚠️ 掩码转张量失败（该参考图按全图处理）: {e}")
            out.append(None)
    return out




def vae_scale_factor(pipe) -> int:
    """潜空间下采样倍率（IP-Adapter 的注意力掩码要按潜空间分辨率给）。

    这里才是唯一实现：`nodes/sampling_nodes._vae_scale_factor` 只是转发过来。
    以前两边各写一份，"按 8 算"和"按管线的真实倍率算"一旦分叉，掩码就会
    缩放两次或一次都没缩 —— 症状是区域参考图悄悄变成整图参考图。
    """
    for obj in (getattr(pipe, "vae", None),
                getattr(getattr(pipe, "unet", None), "config", None)):
        v = getattr(getattr(obj, "config", None), "vae_scale_factor", None)
        if v is None:
            v = getattr(obj, "vae_scale_factor", None)
        try:
            if v:
                return int(v)
        except Exception:
            continue
    return 8


def enabled_refs(refs: Iterable[Any]) -> List[Any]:
    """只留下 `enabled` 的条目（dict 与 IpaReference 通吃）。

    收敛动机：这条过滤原本散在 4 处（主界面 `_enabled_ipa_refs()`、
    `_apply_ipa_refs_if_enabled`、工作流 `apply_ip_adapter_kwargs`、
    `ipa_nodes.pending_references` 之后），各处写法不同（`r.get("enabled")`
    vs `getattr(r, "enabled")`）。漏一处就会出现"界面上停用了，生成时仍然生效"。
    """
    out = []
    for r in refs or []:
        if isinstance(r, dict):
            on = bool(r.get("enabled", True))
        else:
            on = bool(getattr(r, "enabled", True))
        if on:
            out.append(r)
    return out


def collect_reference_images(refs: Iterable[Any], *,
                            on_missing: str = "degrade",
                            log=None, _placeholder=None):
    """把参考图收成 `ip_adapter_image` 需要的 PIL 列表。

    返回 `(images, placeholders)`。**长度必须等于适配器数量**，所以权重 ≤0
    或读不到的条目要留 1x1 占位图，不能让列表塌缩（否则 diffusers 会报
    "长度不匹配"，而且适配器编号全错位）。

    `on_missing`：
      · `"degrade"` —— 读不到就把该条 weight 置 0 并占位（主界面用这个，
        一张图读不到不该让整次生成失败）。
      · `"raise"`   —— 直接抛 RuntimeError（工作流用这个，节点的语义是
        "你要的东西我给不了就说清楚"）。

    两条路径共用本函数，正是为了避免七步流程在两边各写一遍然后慢慢漂移。
    """
    from PIL import Image

    items = coerce_references(refs)
    if _placeholder is None:
        def _placeholder():
            return Image.new("RGB", (1, 1), (0, 0, 0))

    images = []
    placeholders = 0
    for r in items:
        if float(getattr(r, "weight", 1.0)) <= 0.0:
            images.append(_placeholder())
            placeholders += 1
            continue
        img = load_reference_image(r)
        if img is None:
            msg = (f"参考图读不到（{role_label(r.role)}）："
                   f"{r.path or '（既没有路径也没有内存图）'}")
            if on_missing == "raise":
                raise RuntimeError(
                    f"❌ {msg}\n"
                    f"   请检查文件是否存在，或用 IMAGE 端口连线给它。")
            if log is not None:
                log(f"{msg} → 该张强度按 0 处理")
            r.weight = 0.0
            images.append(_placeholder())
            placeholders += 1
            continue
        if r.mask is not None or r.mask_path:
            img = apply_reference_mask(r, img)
        images.append(img)
    return images, placeholders


def plan_masks(refs: Iterable[Any], width: int, height: int, *,
               pipe=None, scale: Optional[int] = None, device: str = "cpu"):
    """按管线真实倍率算注意力掩码；没有掩码时返回 None。"""
    if scale is None:
        scale = vae_scale_factor(pipe)
    return build_ip_adapter_masks(
        refs, width or 512, height or 768, scale=scale, device=device)


# ============================================================
#  七步流程的**唯一实现**：build_runtime()
# ============================================================

@dataclass
class IpaRuntime:
    """一次"多参考图就绪"的完整结果（图 + 强度 + 掩码 + 元数据）。

    为什么要这个对象：主界面 `_apply_ipa_refs_if_enabled` 和工作流
    `apply_ip_adapter_kwargs` 之前各自把"角色→档位 → 收图 → 装 N 个 →
    算掩码 → 组装元数据"这七步写了一遍。两套实现一开始一样，改一处忘一处
    就会漂移 —— 这个项目在 output_dir、LoRA/CN/IPA 读数、超分签名上都吃过
    这个亏。现在两边都只是**薄封装**，差异只在失败策略与提示文案。

    `info` 里的键就是元数据字段名（`ip_adapter_*`），两边共用一套。
    """

    ok: bool = False
    refs: List[IpaReference] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    masks: Optional[List[Any]] = None
    variants: List[str] = field(default_factory=list)
    encoder_dir: str = ""
    placeholders: int = 0
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    plan: Optional["MultiAdapterPlan"] = None
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        """一行中文摘要（状态栏/日志/元数据共用）。"""
        if self.refs:
            return describe_references(self.refs)
        return "无参考图"

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "refs": list(self.refs),
            "images": list(self.images),
            "scales": [float(s) for s in self.scales],
            "masks": self.masks,
            "variants": list(self.variants),
            "encoder_dir": self.encoder_dir,
            "placeholders": int(self.placeholders),
            "warnings": list(self.warnings),
            "error": self.error,
            "summary": self.summary(),
            "info": dict(self.info),
        }


def runtime_info(rt: IpaRuntime) -> dict:
    """把 runtime 拍平成元数据字段（键名就是 `ip_adapter_*`）。

    单独一个函数是为了让两条路写进元数据的**字段完全一致** —— 以前主界面
    那条根本不写元数据，工作流那条写九个键，用户回头查"这张图用了哪几张
    参考图"时，两条路得到的信息量不同。
    """
    refs = list(rt.refs or [])
    return {
        "ip_adapter_enabled": "true",
        "ip_adapter_refs": len(refs),
        "ip_adapter_roles": [getattr(r, "role", "") for r in refs],
        "ip_adapter_variants": list(rt.variants),
        "ip_adapter_weights": [float(getattr(r, "weight", 0.0)) for r in refs],
        "ip_adapter_masked_refs": sum(
            1 for r in refs
            if (getattr(r, "mask", None) is not None
                or getattr(r, "mask_path", ""))),
        "ip_adapter_placeholder_refs": int(rt.placeholders),
        "ip_adapter_encoder": rt.encoder_dir or "",
        "ip_adapter_summary": rt.summary(),
    }


def build_runtime(refs: Iterable[Any], ai, *,
                  width: int = 0, height: int = 0,
                  pipe=None, on_missing: str = "degrade",
                  log=None, device: str = "cpu") -> IpaRuntime:
    """把参考图变成"可以进采样器"的运行时载荷（七步合一）。

    步骤：过滤启用项 → 角色→档位（缺文件降级+警告）→ 收图（权重 ≤0 或读不到
    留占位）→ 一次安装 N 个适配器 → 区域掩码 → 元数据。

    参数
    ----
    refs : 参考图列表（dict 或 `IpaReference`，停用的条目会被过滤）
    ai   : `ModelManager`（需要 `installed_ipa_variants` 与
           `prepare_ip_adapter_multi` 两个方法）
    on_missing : `"degrade"` 读不到就按 0 强度占位（主界面）；
                 `"raise"` 直接抛（工作流 —— 节点的语义是"给不了要说清楚"）

    返回 `IpaRuntime`；**不抛异常**，失败写进 `.ok/.error`，由调用方决定
    是整体失败还是降级继续。（`on_missing="raise"` 时读图失败仍会抛，
    这是刻意的：那是用户在界面上看得懂的错。）
    """
    rt = IpaRuntime()
    if ai is None:
        rt.error = "缺少模型管理器（ctx.ai 为空），无法安装 IP-Adapter。"
        return rt
    try:
        items = coerce_references(enabled_refs(refs))
    except Exception as exc:                    # pragma: no cover - 防御
        rt.error = f"参考图列表无法解析: {exc}"
        return rt
    if not items:
        rt.error = "没有启用中的参考图"
        return rt

    try:
        installed = ai.installed_ipa_variants()
    except Exception as exc:
        logger.warning(f"查询本机 IP-Adapter 档位失败（按未安装处理）: {exc}")
        installed = []

    # 1) 角色 → 实际档位（缺文件降级并留中文警告，一个都没有才报错）
    try:
        resolved, warns = resolve_role_plan(items, installed)
    except Exception as exc:
        rt.error = str(exc)
        return rt
    rt.warnings.extend(warns)

    # 2) 参考图 → PIL（长度必须等于适配器数量）
    rt.images, rt.placeholders = collect_reference_images(
        resolved, on_missing=on_missing, log=log)

    # 3) 一次安装 N 个适配器（共享 UNet 只装一次，重复调是覆盖语义）
    plan = ai.prepare_ip_adapter_multi(resolved)
    rt.plan = plan
    for w in (getattr(plan, "warnings", None) or []):
        if w not in rt.warnings:
            rt.warnings.append(w)
    if not getattr(plan, "ok", False):
        rt.error = getattr(plan, "error", "") or "未知原因"
        return rt

    rt.refs = list(getattr(plan, "refs", None) or resolved)
    rt.variants = list(getattr(plan, "variants", None) or [])
    rt.encoder_dir = getattr(plan, "encoder_dir", "") or ""
    rt.scales = [float(s) for s in (getattr(plan, "scales", None) or [])]
    if not rt.scales:
        rt.scales = [float(getattr(r, "weight", 0.0)) for r in rt.refs]

    # 4) 区域蒙版 → 注意力掩码（按管线真实倍率缩到潜空间）
    if width and height:
        try:
            rt.masks = plan_masks(rt.refs, int(width), int(height),
                                  pipe=pipe, device=device)
        except Exception as exc:
            logger.warning(f"参考图区域掩码构造失败（改为整图参与）: {exc}")
            rt.masks = None

    rt.info = runtime_info(rt)
    rt.ok = True
    return rt


def role_table() -> dict:
    """角色 → 档位/文件/中文说明。文档、界面提示、测试共用一份。"""
    from core.model_manager import ModelManager  # 延迟 import（避免环）
    out = {}
    for role, variant in ROLE_VARIANTS.items():
        entry = ModelManager._IPA_WEIGHTS.get((False, variant)) or ("", "")
        out[role] = {
            "label": ROLE_LABELS[role],
            "variant": variant,
            "file": entry[1],
            "subfolder": entry[0],
        }
    return out


# ============================================================
#  安装结果（ModelManager.prepare_ip_adapter_multi 的返回值）
# ============================================================

@dataclass
class MultiAdapterPlan:
    """一次"装 N 个适配器"的结果汇总。

    为什么要有这个对象：节点需要知道自己**实际**装上了什么（几条参考图、
    哪几个档位、哪些降级了、缺什么文件），才能把中文警告写进
    `ctx.metadata["warnings"]`。只返回 True/False 的话，"降级"就会被静默吞掉。
    """

    ok: bool = False
    count: int = 0
    variants: List[str] = field(default_factory=list)
    scales: List[float] = field(default_factory=list)
    refs: List[IpaReference] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    weight_names: List[str] = field(default_factory=list)
    subfolders: List[str] = field(default_factory=list)
    encoder_dir: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    def summary(self) -> str:
        """一行中文摘要（进元数据/状态栏）。"""
        if not self.ok:
            return f"失败: {self.error or '未知原因'}"
        parts = []
        for r, v in zip(self.refs, self.variants):
            parts.append(f"{role_label(r.role)}={v}@{r.weight:.2f}")
        return " + ".join(parts) or "无"

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "count": int(self.count),
            "variants": list(self.variants),
            "scales": [float(s) for s in self.scales],
            "sources": list(self.sources),
            "weight_names": list(self.weight_names),
            "encoder_dir": self.encoder_dir,
            "warnings": list(self.warnings),
            "error": self.error,
            "summary": self.summary(),
        }


__all__ = [
    "ROLE_VARIANTS", "ROLE_LABELS", "VARIANT_ALIASES", "DEFAULT_ROLE",
    "DEFAULT_WEIGHT", "IpaReference",
    "normalize_role", "normalize_variant", "role_label", "variant_for_role",
    "role_for_variant", "references_from_json", "coerce_references",
    "references_to_json", "describe_references", "resolve_role_plan",
    "role_table", "load_reference_image", "apply_reference_mask",
    "build_ip_adapter_masks", "MultiAdapterPlan",
    "vae_scale_factor", "enabled_refs", "collect_reference_images",
    "plan_masks", "IpaRuntime", "runtime_info", "build_runtime",
]
