# workflow/comfy_compat.py
"""ComfyUI 工作流兼容层：导入 / 导出。

支持两种格式：

* **API 格式**（ComfyUI 后端 `/prompt` 提交体，很多第三方站点导出的就是它）::

      {"3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, ...}},
       "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}}}

* **UI 格式**（ComfyUI 界面里 "Save" 存下的 workflow.json）::

      {"nodes": [{"id": 3, "type": "KSampler", "widgets_values": [...],
                  "inputs": [{"name": "model", "link": 7}, ...]}],
       "links": [[7, 4, 0, 3, 0, "MODEL"], ...]}

设计取舍
--------
1. **不做"尽力而为的静默降级"**。ComfyUI 的节点比本项目多得多，遇到不认识的
   节点必须写进 `ComfyReport.unsupported` 并在界面上显示，否则用户会得到一张
   看起来正常、实际少了三个节点的图 —— 这正是本项目在 LoRA / IP-Adapter 上
   反复吃过亏的地方。
2. **一次只映射能如实表达的语义**。例如 `EmptyLatentImage` 没有对应节点，它的
   宽高被"折"进 `Sampler` 的参数里；`CLIPSetLastLayer` 的层数折进
   `LoadCheckpoint.clip_skip`。折起来的节点记在 `ComfyReport.folded`。
3. 导出只输出**能完整表达**的节点，其余记进报告的 unsupported。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from workflow.registry import exists as node_exists
from workflow.schema import Node, Workflow

logger = logging.getLogger(__name__)

FORMAT_API = "api"
FORMAT_UI = "ui"

# ComfyUI 里"没有对应节点、但要吃掉它的值"的类型
IGNORED_TYPES = ("Note", "MarkdownNote")


@dataclass
class ComfyReport:
    """一次导入/导出的结果报告。"""

    kind: str = FORMAT_API
    mapped: list = field(default_factory=list)        # ["KSampler#3 → Sampler"]
    folded: list = field(default_factory=list)        # ["EmptyLatentImage#5 → Sampler.width/height"]
    unsupported: list = field(default_factory=list)   # [(comfy_id, class_type)]
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.warnings

    def summary_text(self) -> str:
        lines = [f"格式: {'API' if self.kind == FORMAT_API else 'UI'}",
                 f"已映射: {len(self.mapped)} 个节点",
                 f"已折叠: {len(self.folded)} 个节点"]
        if self.unsupported:
            lines.append(f"❌ 不支持: {len(self.unsupported)} 个节点")
            for cid, cls in self.unsupported[:20]:
                lines.append(f"   · #{cid} {cls}")
            if len(self.unsupported) > 20:
                lines.append(f"   · …还有 {len(self.unsupported) - 20} 个")
        if self.warnings:
            lines.append(f"⚠️ 提示: {len(self.warnings)} 条")
            for w in self.warnings[:20]:
                lines.append(f"   · {w}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _ComfyMap:
    """一个 ComfyUI 类型 → 本项目节点的映射。"""

    ours: str
    params: dict = field(default_factory=dict)   # comfy 输入名 → 本项目参数名
    links: dict = field(default_factory=dict)    # comfy 输入名 → 本项目输入端口
    outs: tuple = ()                             # 输出槽位顺序 → 本项目端口名
    note: str = ""


@dataclass(frozen=True)
class _ComfyFold:
    """一个"值被折进别的节点"的 ComfyUI 类型。"""

    keys: dict = field(default_factory=dict)     # comfy 输入名 → 上下文键
    note: str = ""


# ---------------------------------------------------------------
# 映射表
# ---------------------------------------------------------------
_MAP: dict[str, _ComfyMap] = {
    "CheckpointLoaderSimple": _ComfyMap(
        "LoadCheckpoint",
        params={"ckpt_name": "path"},
        outs=("model", "vae", "metadata")),
    "CheckpointLoader": _ComfyMap(
        "LoadCheckpoint",
        params={"ckpt_name": "path"},
        outs=("model", "vae", "metadata")),
    "VAELoader": _ComfyMap(
        "LoadVAE",
        params={"vae_name": "path"},
        links={"model": "model"},
        outs=("model", "vae")),
    "LoraLoader": _ComfyMap(
        "LoadLoRA",
        params={"lora_name": "lora_1", "strength_model": "weight_1"},
        links={"model": "model"},
        outs=("model", "lora", "metadata")),
    "CLIPTextEncode": _ComfyMap(
        "TextEncode",
        params={"text": "text"},
        links={"clip": "model"},
        outs=("conditioning", "metadata")),
    "KSampler": _ComfyMap(
        "Sampler",
        params={"seed": "seed", "steps": "steps", "cfg": "cfg",
                "sampler_name": "sampler", "scheduler": "scheduler",
                "denoise": "strength"},
        links={"model": "model", "positive": "positive", "negative": "negative"},
        outs=("latent", "image", "metadata")),
    "KSamplerAdvanced": _ComfyMap(
        "Sampler",
        params={"noise_seed": "seed", "steps": "steps", "cfg": "cfg",
                "sampler_name": "sampler", "scheduler": "scheduler"},
        links={"model": "model", "positive": "positive", "negative": "negative"},
        outs=("latent", "image", "metadata"),
        note="KSamplerAdvanced 的 start_at_step/end_at_step 无法表达，已忽略"),
    "VAEDecode": _ComfyMap(
        "VAEDecode",
        links={"samples": "latent", "vae": "model"},
        outs=("image",)),
    "VAEEncode": _ComfyMap(
        "VAEEncode",
        links={"pixels": "image", "vae": "model"},
        outs=("latent",)),
    "SaveImage": _ComfyMap(
        "SaveImage",
        params={"filename_prefix": "filename_prefix"},
        links={"images": "image"},
        outs=("path", "metadata")),
    "PreviewImage": _ComfyMap(
        "SaveImage",
        links={"images": "image"},
        outs=("path", "metadata"),
        note="PreviewImage 在 ComfyUI 里只是预览；这里落到 SaveImage"),
    "LoadImage": _ComfyMap(
        "LoadImage",
        params={"image": "path"},
        outs=("image", "mask", "metadata")),
    "ImageScale": _ComfyMap(
        "ImageResize",
        params={"width": "width", "height": "height"},
        links={"image": "image"},
        outs=("image",),
        note="ImageScale 的 crop/upscale_method 未映射"),
    "ImageScaleBy": _ComfyMap(
        "Upscale",
        params={"scale_by": "scale"},
        links={"image": "image"},
        outs=("image", "metadata")),
    "ControlNetApplyAdvanced": _ComfyMap(
        "ControlNetApply",
        params={"strength": "weight"},
        links={"image": "control", "model": "model"},
        outs=("model", "metadata"),
        note="positive/negative 未映射；control_net 权重文件需在节点里另选"),
    "ControlNetApply": _ComfyMap(
        "ControlNetApply",
        params={"strength": "weight"},
        links={"image": "control", "model": "model"},
        outs=("model", "metadata")),
    "IPAdapterApply": _ComfyMap(
        "IPAdapterApply",
        params={"weight": "weight"},
        links={"model": "model", "image": "image"},
        outs=("model", "metadata"),
        note="ComfyUI 的 IPAdapter 家族参数很多，这里只取 weight"),
}

_FOLD: dict[str, _ComfyFold] = {
    "EmptyLatentImage": _ComfyFold({"width": "width", "height": "height"}),
    "EmptySD3LatentImage": _ComfyFold({"width": "width", "height": "height"}),
    "CLIPSetLastLayer": _ComfyFold({"stop_at_clip_layer": "clip_skip"}),
}

# UI 格式里 widgets_values 的顺序（只登记用得到的；不在表里就当不支持）
_UI_WIDGETS: dict[str, tuple] = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "VAELoader": ("vae_name",),
    "LoraLoader": ("lora_name", "strength_model", "strength_clip"),
    "CLIPTextEncode": ("text",),
    "CLIPSetLastLayer": ("stop_at_clip_layer",),
    "KSampler": ("seed", "control_after_generate", "steps", "cfg",
                 "sampler_name", "scheduler", "denoise"),
    "EmptyLatentImage": ("width", "height", "batch_size"),
    "SaveImage": ("filename_prefix",),
    "PreviewImage": (),
    "LoadImage": ("image", "upload"),
    "VAEDecode": (),
    "VAEEncode": (),
}


# ---------------------------------------------------------------
# 格式识别
# ---------------------------------------------------------------
def is_api_format(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "nodes" in data and isinstance(data.get("nodes"), list):
        return False
    vals = list(data.values())
    if not vals:
        return False
    return all(isinstance(v, dict) and "class_type" in v for v in vals)


def is_ui_format(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


def detect_format(data: Any) -> str:
    if is_api_format(data):
        return FORMAT_API
    if is_ui_format(data):
        return FORMAT_UI
    raise ValueError("无法识别的工作流格式：既不是 ComfyUI API 格式，"
                     "也不是 UI 格式（缺 nodes 数组）")


# ---------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------
def _is_link(val: Any) -> bool:
    return (isinstance(val, (list, tuple)) and len(val) == 2
            and isinstance(val[0], (str, int)) and not isinstance(val[0], bool)
            and isinstance(val[1], int))


# 这些参数的值是"本机文件路径/文件名"。ComfyUI 的 ckpt_name / lora_name 常带
# 子目录（"SDXL\\mix2.safetensors"），而本项目是按文件名在模型目录里查，
# 直接用会找不到；导入时归一成 basename 并留一条 warning。
_FILE_PARAMS = {"path", "lora_1", "lora_2", "lora_3"}


def _basename_any(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("\\", "/").rstrip("/").split("/")[-1]


def _coerce_param(ours: str, key: str, val: Any) -> Any:
    """把 ComfyUI 的取值翻译成本项目节点的参数值。"""
    if ours == "Sampler" and key == "sampler":
        try:
            from core import samplers as _s
            d = _s.find_sampler(str(val))
            if d is not None:
                return d.name
        except Exception as e:  # pragma: no cover - 只在 samplers 损坏时
            logger.debug("采样器名映射失败 %r: %s", val, e)
        return str(val)
    if ours == "Sampler" and key == "scheduler":
        try:
            from core import samplers as _s
            return _s.normalize_sigma(str(val))
        except Exception as e:  # pragma: no cover
            logger.debug("sigma 风格映射失败 %r: %s", val, e)
        return str(val)
    if ours == "LoadCheckpoint" and key == "clip_skip":
        try:
            v = int(val)
        except (TypeError, ValueError):
            return 0
        # ComfyUI 用负数表示"停在倒数第几层"
        return abs(v) if v < 0 else v
    return val


def _comfy_sampler_name(display: str) -> str:
    try:
        from core import samplers as _s
        return _s.to_comfy(display)
    except Exception:
        return str(display).lower().replace(" ", "_")


def _comfy_sigma_name(style: str) -> str:
    try:
        from core import samplers as _s
        return _s.sigma_to_comfy(style)
    except Exception:
        return "normal"


# ---------------------------------------------------------------
# UI 格式 → API 格式
# ---------------------------------------------------------------
def ui_to_api(data: dict) -> tuple[dict, ComfyReport]:
    """把 ComfyUI 的 UI 格式（nodes/links）转成 API 格式的等价值。"""
    rep = ComfyReport(kind=FORMAT_UI)
    nodes = data.get("nodes") or []
    link_src: dict[int, tuple] = {}
    for lk in (data.get("links") or []):
        if not isinstance(lk, (list, tuple)) or len(lk) < 5:
            continue
        try:
            lid, src, sslot = int(lk[0]), int(lk[1]), int(lk[2])
        except (TypeError, ValueError):
            continue
        link_src[lid] = (str(src), sslot)

    api: dict[str, dict] = {}
    for nd in nodes:
        if not isinstance(nd, dict):
            continue
        nid = nd.get("id")
        cls = nd.get("type") or nd.get("class_type") or ""
        if nid is None or not cls:
            continue
        if cls in IGNORED_TYPES:
            continue
        inputs: dict[str, Any] = {}
        # 1) 连线
        for slot in (nd.get("inputs") or []):
            if not isinstance(slot, dict):
                continue
            name = slot.get("name")
            lid = slot.get("link")
            if name and lid is not None and int(lid) in link_src:
                src, sslot = link_src[int(lid)]
                inputs[name] = [src, sslot]
        # 2) 控件值（按已知顺序）
        wv = nd.get("widgets_values")
        names = _UI_WIDGETS.get(cls)
        if isinstance(wv, dict):
            # 有些导出把 widgets_values 存成 dict
            for k, v in wv.items():
                inputs.setdefault(k, v)
        elif isinstance(wv, (list, tuple)) and names:
            for k, v in zip(names, wv):
                if k == "control_after_generate":
                    continue
                inputs.setdefault(k, v)
        elif isinstance(wv, (list, tuple)) and wv:
            rep.warnings.append(f"#{nid} {cls}: 不认识 widgets_values 的顺序，已跳过控件值")
        api[str(nid)] = {"class_type": cls, "inputs": inputs}
    if not api:
        rep.warnings.append("UI 格式里没有可识别的节点")
    rep.mapped.append(f"UI→API: {len(api)} 个节点")
    return api, rep


# ---------------------------------------------------------------
# 导入
# ---------------------------------------------------------------
def _apply_fold(fold_id: str, api: dict, id_map: dict, wf: Workflow,
                target: Optional[Node], rep: ComfyReport) -> bool:
    """把一个"折叠节点"的值灌进下游节点，返回是否有内容被采纳。"""
    nd = api[fold_id]
    cls = nd.get("class_type") or ""
    spec = _FOLD.get(cls)
    if spec is None or target is None:
        return False
    src_inputs = nd.get("inputs") or {}
    applied = False
    if cls in ("EmptyLatentImage", "EmptySD3LatentImage"):
        allowed = _known_params(target.type)
        for cin, key in spec.keys.items():
            val = src_inputs.get(cin)
            if _is_link(val) or val is None:
                continue
            if key not in allowed:
                rep.warnings.append(
                    f"{cls}#{fold_id}: {target.type} 没有 {key} 参数，宽高丢失")
                continue
            target.params[key] = int(val)
            applied = True
        if applied:
            rep.folded.append(f"{cls}#{fold_id} → {target.id}.width/height")
        return applied
    if cls == "CLIPSetLastLayer":
        raw = src_inputs.get("stop_at_clip_layer")
        if _is_link(raw) or raw is None:
            return False
        # 顺着它的 clip 输入找到上游 LoadCheckpoint
        up = src_inputs.get("clip")
        if _is_link(up):
            src_id, _ = str(up[0]), int(up[1])
            ours = id_map.get(src_id)
            upstream = wf.node(ours) if ours else None
            if upstream is not None and upstream.type == "LoadCheckpoint":
                upstream.params["clip_skip"] = _coerce_param(
                    "LoadCheckpoint", "clip_skip", raw)
                rep.folded.append(
                    f"{cls}#{fold_id} → {upstream.id}.clip_skip")
                return True
        rep.warnings.append(
            f"{cls}#{fold_id}: 找不到上游 LoadCheckpoint，clip_skip 丢失")
        return False
    return False


def _resolve_link(src: Any, api: dict, id_map: dict, wf: Workflow,
                  target: Optional[Node], rep: ComfyReport,
                  depth: int = 0) -> Optional[tuple]:
    """把 ComfyUI 的连线解析成本项目的 (节点 id, 端口名)。"""
    if depth > 16 or not _is_link(src):
        return None
    sid, slot = str(src[0]), int(src[1])
    if sid in id_map:
        nd = api.get(sid) or {}
        m = _MAP.get(nd.get("class_type") or "")
        if m is not None and 0 <= slot < len(m.outs):
            return id_map[sid], m.outs[slot]
        if m is not None:
            rep.warnings.append(
                f"节点 #{sid} {nd.get('class_type')} 只有 {len(m.outs)} 个输出，"
                f"引用了第 {slot} 个")
        return None
    src_cls = (api.get(sid) or {}).get("class_type") or ""
    if src_cls in _FOLD:
        _apply_fold(sid, api, id_map, wf, target, rep)
        return None
    nd = api.get(sid) or {}
    cls = nd.get("class_type") or ""
    # Reroute / 其它单输入直通节点：继续往上找
    if cls in ("Reroute", "PrimitiveNode"):
        for val in (nd.get("inputs") or {}).values():
            if _is_link(val):
                return _resolve_link(val, api, id_map, wf, target, rep, depth + 1)
        return None
    return None


def import_prompt(data: Any, name: str = "") -> tuple[Workflow, ComfyReport]:
    """导入 ComfyUI 工作流（API 或 UI 格式）。"""
    kind = detect_format(data)
    if kind == FORMAT_UI:
        api, rep = ui_to_api(data)
    else:
        api, rep = (dict(data), ComfyReport(kind=FORMAT_API))

    wf = Workflow(name=name or "ComfyUI 导入")
    wf.metadata.setdefault("source", "comfyui")
    id_map: dict[str, str] = {}

    # 第一遍：建节点 + 字面量参数
    for cid, nd in api.items():
        cls = (nd.get("class_type") or "").strip()
        if not cls or cls in IGNORED_TYPES:
            continue
        if cls in _FOLD:
            continue
        m = _MAP.get(cls)
        if m is None:
            rep.unsupported.append((cid, cls))
            continue
        if not node_exists(m.ours):
            rep.unsupported.append((cid, cls))
            rep.warnings.append(f"目标节点 {m.ours} 未注册（{cls}）")
            continue
        nid = f"c{cid}"
        node = Node(id=nid, type=m.ours, inputs={}, params={})
        wf.add(node)
        id_map[cid] = nid
        rep.mapped.append(f"{cls}#{cid} → {m.ours}")
        for cin, val in (nd.get("inputs") or {}).items():
            if _is_link(val):
                continue
            key = m.params.get(cin)
            if key is None:
                continue
            value = _coerce_param(m.ours, key, val)
            if key in _FILE_PARAMS and isinstance(value, str):
                base = _basename_any(value)
                if base and base != value:
                    rep.warnings.append(
                        f"{cls}#{cid}: {val} 带子目录，已按文件名 "
                        f"{base} 在本机模型目录里查找")
                    value = base
            node.params[key] = value

    # 第二遍：连线 + 折叠值
    for cid, nd in api.items():
        cls = (nd.get("class_type") or "").strip()
        target_id = id_map.get(cid)
        target = wf.node(target_id) if target_id else None
        m = _MAP.get(cls)
        if target is None or m is None:
            continue
        for cin, val in (nd.get("inputs") or {}).items():
            if not _is_link(val):
                continue
            resolved = _resolve_link(val, api, id_map, wf, target, rep)
            if resolved is None:
                continue
            src_id, port = resolved
            our_in = m.links.get(cin, cin)
            if our_in not in _known_inputs(m.ours):
                rep.warnings.append(f"{cls}#{cid}: 输入 {cin} 在 {m.ours} 上没有对应端口")
                continue
            try:
                wf.connect(src_id, port, target_id, our_in)
            except KeyError as e:
                rep.warnings.append(f"{cls}#{cid}: 连线失败 {e}")

    # 第三遍：CLIPTextEncode 的正面/负面角色
    _assign_text_roles(wf, api, id_map, rep)
    if wf.metadata.get("_notes"):
        rep.warnings.extend(wf.metadata.pop("_notes"))
    if not wf.nodes:
        rep.warnings.append("没有导入任何节点")
    return wf, rep


_KNOWN_INPUTS_CACHE: dict[str, set] = {}
_KNOWN_PARAMS_CACHE: dict[str, set] = {}


def _known_params(node_type: str) -> set:
    got = _KNOWN_PARAMS_CACHE.get(node_type)
    if got is None:
        try:
            from workflow.registry import get as _get
            got = set(_get(node_type).params)
        except Exception:
            got = set()
        _KNOWN_PARAMS_CACHE[node_type] = got
    return got


def _known_inputs(node_type: str) -> set:
    got = _KNOWN_INPUTS_CACHE.get(node_type)
    if got is None:
        try:
            from workflow.registry import get as _get
            got = set(_get(node_type).inputs)
        except Exception:
            got = set()
        _KNOWN_INPUTS_CACHE[node_type] = got
    return got


def _assign_text_roles(wf: Workflow, api: dict, id_map: dict,
                       rep: ComfyReport) -> None:
    """CLIPTextEncode 是正面还是负面，取决于它接到哪个输入口。"""
    role: dict[str, str] = {}
    for nd in api.values():
        for cin, val in (nd.get("inputs") or {}).items():
            if not _is_link(val):
                continue
            src = id_map.get(str(val[0]))
            if src is None:
                continue
            if cin == "positive":
                role[src] = "positive"
            elif cin == "negative":
                role.setdefault(src, "negative")
    for node in wf.nodes:
        if node.type not in ("TextEncode", "NegativeTextEncode"):
            continue
        want = role.get(node.id)
        if want == "negative" and node.type == "TextEncode":
            node.type = "NegativeTextEncode"
        elif want == "positive" and node.type == "NegativeTextEncode":
            node.type = "TextEncode"
        elif want is None:
            rep.warnings.append(
                f"{node.id}: 文本编码没有连到 positive/negative，按正面处理")


# ---------------------------------------------------------------
# 导出
# ---------------------------------------------------------------
def _reverse_map(ours: str) -> Optional[tuple[str, _ComfyMap]]:
    # 本项目把一个节点拆成"正面 / 负面"两个类型，ComfyUI 只有一个
    # CLIPTextEncode —— 导出时两边都落到它上面。
    alias = _EXPORT_ALIAS.get(ours, ours)
    for cls, m in _MAP.items():
        if m.ours == alias:
            return cls, m
    return None


_EXPORT_ALIAS = {
    "NegativeTextEncode": "TextEncode",
}


def _port_index(ours: str, port: str) -> int:
    r = _reverse_map(ours)
    if r is None:
        return 0
    _, m = r
    try:
        return m.outs.index(port)
    except ValueError:
        return 0


def export_prompt(wf: Workflow) -> tuple[dict, ComfyReport]:
    """导出成 ComfyUI API 格式（只导能完整表达的节点）。"""
    rep = ComfyReport(kind=FORMAT_API)
    id_of: dict[str, str] = {}
    for idx, node in enumerate(wf.nodes, start=1):
        id_of[node.id] = str(idx)
    next_id = len(wf.nodes) + 1

    out: dict[str, dict] = {}
    for node in wf.nodes:
        r = _reverse_map(node.type)
        if r is None:
            rep.unsupported.append((node.id, node.type))
            continue
        cls, m = r
        if m.note:
            rep.warnings.append(f"{node.id} {node.type}: {m.note}")
        inputs: dict[str, Any] = {}
        for cin, key in m.params.items():
            if key not in node.params:
                continue
            val = node.params[key]
            if node.type == "Sampler" and key == "sampler":
                val = _comfy_sampler_name(str(val))
            elif node.type == "Sampler" and key == "scheduler":
                val = _comfy_sigma_name(str(val))
            inputs[cin] = val
        for cin, our_in in m.links.items():
            ref = node.inputs.get(our_in)
            if not _is_link_like(ref):
                continue
            src_id, port = str(ref[0]), str(ref[1])
            src = wf.node(src_id)
            if src is None or src_id not in id_of:
                rep.warnings.append(f"{node.id}: 输入 {our_in} 的源节点不存在")
                continue
            inputs[cin] = [id_of[src_id], _port_index(src.type, port)]
        # txt2img 的采样器：ComfyUI 的 KSampler 必须接一个 LATENT，
        # 而本项目的 Sampler 直接带 width/height。导出时补一个
        # EmptyLatentImage，导入时会再折回去 —— 这样两个方向都无损。
        if (node.type == "Sampler" and "width" in node.params
                and "height" in node.params
                and not _is_link_like(node.inputs.get("image"))):
            eid = str(next_id)
            next_id += 1
            out[eid] = {"class_type": "EmptyLatentImage",
                        "inputs": {"width": int(node.params["width"]),
                                   "height": int(node.params["height"]),
                                   "batch_size": 1}}
            inputs["latent_image"] = [eid, 0]
            rep.folded.append(f"{node.id}.width/height → EmptyLatentImage#{eid}")
        out[id_of[node.id]] = {"class_type": cls, "inputs": inputs}
        rep.mapped.append(f"{node.type}#{node.id} → {cls}")
    if not out:
        rep.warnings.append("没有可导出的节点")
    return out, rep


def _is_link_like(val: Any) -> bool:
    return (isinstance(val, (list, tuple)) and len(val) == 2
            and isinstance(val[0], str))


# ---------------------------------------------------------------
# 便捷入口（界面用）
# ---------------------------------------------------------------
def load_file(path: str) -> tuple[Workflow, ComfyReport]:
    import json
    import os

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return import_prompt(data, name=os.path.splitext(os.path.basename(path))[0])


def save_file(wf: Workflow, path: str) -> ComfyReport:
    import json

    data, rep = export_prompt(wf)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return rep


__all__ = [
    "FORMAT_API", "FORMAT_UI", "ComfyReport",
    "detect_format", "is_api_format", "is_ui_format",
    "ui_to_api", "import_prompt", "export_prompt",
    "load_file", "save_file",
]
