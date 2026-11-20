# -*- coding: utf-8 -*-
"""内置工作流模板。

参考 ComfyUI 的"自带示例"，但这里的模板是**能直接跑的**：
每个模板只用内置节点、只连合法端口、参数取节点默认值再覆盖必要的几项，
所以用户点开就能按「运行」，也能当学习画布结构的起点。

用法：
    from workflow import templates
    wf = templates.build("txt2img")
    names = templates.list_templates()      # [(key, 中文名, 说明), ...]
"""

from __future__ import annotations

import logging

from workflow import registry
from workflow.schema import Node, Workflow

logger = logging.getLogger(__name__)


class TemplateError(RuntimeError):
    """模板引用了不存在的节点类型（说明代码与注册表不同步）。"""


def _defaults(ntype: str) -> dict:
    spec = registry.get(ntype)
    if spec is None:
        raise TemplateError(f"模板引用了未注册的节点类型：{ntype}")
    return dict(spec.param_defaults())


def _node(wf: Workflow, nid: str, ntype: str, x: float, y: float,
          params: dict | None = None) -> Node:
    merged = _defaults(ntype)
    merged.update(params or {})
    node = Node(nid, ntype, {}, merged)
    wf.add(node)
    wf.metadata.setdefault("ui_positions", {})[nid] = [float(x), float(y)]
    return node


def _link(wf: Workflow, src: str, sport: str, dst: str, dport: str):
    wf.connect(src, sport, dst, dport)


# ----------------------------------------------------------------------
def _txt2img() -> Workflow:
    wf = Workflow(name="文生图")
    _node(wf, "model", "LoadCheckpoint", 40, 40)
    _node(wf, "pos", "TextEncode", 330, 40,
          {"text": "a cat sitting on a windowsill, soft morning light"})
    _node(wf, "neg", "NegativeTextEncode", 330, 300,
          {"text": "lowres, blurry, watermark, text"})
    _node(wf, "sampler", "Sampler", 640, 40, {"steps": 28, "cfg": 7.0,
                                             "width": 512, "height": 768})
    _node(wf, "decode", "VAEDecode", 940, 40)
    _node(wf, "save", "SaveImage", 1180, 40, {"filename_prefix": "t2i"})
    _link(wf, "model", "model", "pos", "model")
    _link(wf, "model", "model", "neg", "model")
    _link(wf, "model", "model", "sampler", "model")
    _link(wf, "pos", "conditioning", "sampler", "positive")
    _link(wf, "neg", "conditioning", "sampler", "negative")
    _link(wf, "sampler", "latent", "decode", "latent")
    _link(wf, "model", "model", "decode", "model")
    _link(wf, "decode", "image", "save", "image")
    _link(wf, "sampler", "metadata", "save", "metadata")
    return wf


def _txt2img_hires() -> Workflow:
    wf = _txt2img()
    wf.name = "文生图 + 高清修复"
    # 高清修复插在解码之后、保存之前
    _node(wf, "hires", "HiresFix", 940, 260, {"scale": "1.5", "denoise": 0.35})
    # 把最简模板里的保存节点换成"先高清修复再保存"
    for link in list(wf.links):
        if "save" in (link[0], link[2]):
            wf.links.remove(link)
    wf.nodes = [n for n in wf.nodes if n.id != "save"]
    wf.metadata["ui_positions"].pop("save", None)
    _node(wf, "save2", "SaveImage", 1240, 260, {"filename_prefix": "t2i_hires"})
    _link(wf, "decode", "image", "hires", "image")
    _link(wf, "model", "model", "hires", "model")
    _link(wf, "pos", "conditioning", "hires", "positive")
    _link(wf, "neg", "conditioning", "hires", "negative")
    _link(wf, "hires", "image", "save2", "image")
    _link(wf, "hires", "metadata", "save2", "metadata")
    return wf


def _inpaint() -> Workflow:
    wf = Workflow(name="局部重绘")
    _node(wf, "model", "LoadCheckpoint", 40, 40)
    _node(wf, "img", "LoadImage", 40, 300, {"path": ""})
    _node(wf, "mask", "MaskCreate", 330, 300, {"mode": "矩形"})
    _node(wf, "pos", "TextEncode", 330, 40, {"text": "the same photo, refined"})
    _node(wf, "neg", "NegativeTextEncode", 330, 560,
          {"text": "lowres, blurry, watermark"})
    _node(wf, "paint", "Inpaint", 640, 40, {"steps": 24, "cfg": 6.5,
                                            "strength": 0.6})
    _node(wf, "save", "SaveImage", 940, 40, {"filename_prefix": "inpaint"})
    _link(wf, "model", "model", "pos", "model")
    _link(wf, "model", "model", "neg", "model")
    _link(wf, "img", "image", "mask", "image")
    _link(wf, "model", "model", "paint", "model")
    _link(wf, "img", "image", "paint", "image")
    _link(wf, "mask", "mask", "paint", "mask")
    _link(wf, "pos", "conditioning", "paint", "positive")
    _link(wf, "neg", "conditioning", "paint", "negative")
    _link(wf, "paint", "image", "save", "image")
    return wf


def _ipadapter() -> Workflow:
    wf = Workflow(name="参考图（IP-Adapter）")
    _node(wf, "model", "LoadCheckpoint", 40, 40)
    _node(wf, "ref", "LoadImage", 40, 300, {"path": ""})
    _node(wf, "ipa", "IPAdapterApply", 330, 40, {"weight": 0.8})
    _node(wf, "pos", "TextEncode", 330, 300,
          {"text": "the same character, standing in a garden"})
    _node(wf, "neg", "NegativeTextEncode", 330, 540,
          {"text": "lowres, blurry, watermark"})
    _node(wf, "sampler", "Sampler", 640, 40, {"steps": 28, "cfg": 7.0})
    _node(wf, "decode", "VAEDecode", 940, 40)
    _node(wf, "save", "SaveImage", 1180, 40, {"filename_prefix": "ipa"})
    _link(wf, "model", "model", "ipa", "model")
    _link(wf, "ref", "image", "ipa", "image")
    _link(wf, "ipa", "model", "sampler", "model")
    _link(wf, "ipa", "model", "pos", "model")
    _link(wf, "ipa", "model", "neg", "model")
    _link(wf, "pos", "conditioning", "sampler", "positive")
    _link(wf, "neg", "conditioning", "sampler", "negative")
    _link(wf, "sampler", "latent", "decode", "latent")
    _link(wf, "model", "model", "decode", "model")
    _link(wf, "decode", "image", "save", "image")
    _link(wf, "sampler", "metadata", "save", "metadata")
    return wf


def _upscale() -> Workflow:
    wf = Workflow(name="图片放大")
    _node(wf, "img", "LoadImage", 40, 40, {"path": ""})
    _node(wf, "up", "Upscale", 340, 40, {"scale": 2.0})
    _node(wf, "save", "SaveImage", 640, 40, {"filename_prefix": "upscaled"})
    _link(wf, "img", "image", "up", "image")
    _link(wf, "up", "image", "save", "image")
    _link(wf, "up", "metadata", "save", "metadata")
    return wf


_TEMPLATES = {
    "txt2img": ("文生图（最简）", "加载底模 → 正负提示词 → 采样 → 解码 → 保存", _txt2img),
    "txt2img_hires": ("文生图 + 高清修复", "在文生图基础上加 HiresFix（1.5× 重绘）", _txt2img_hires),
    "inpaint": ("局部重绘", "载入图片 + 遮罩 → Inpaint → 保存", _inpaint),
    "ipadapter": ("参考图（IP-Adapter）", "用一张参考图控制人物/画风", _ipadapter),
    "upscale": ("图片放大", "载入图片 → 放大 2× → 保存", _upscale),
}


def list_templates() -> list[tuple[str, str, str]]:
    """[(key, 名称, 说明), ...]，按固定顺序返回。"""
    return [(k, name, desc) for k, (name, desc, _fn) in _TEMPLATES.items()]


def build(key: str) -> Workflow:
    """按 key 生成工作流；未知 key 抛 KeyError。"""
    if key not in _TEMPLATES:
        raise KeyError(f"未知模板：{key}（可选：{', '.join(_TEMPLATES)}）")
    registry.load_builtin_nodes()
    wf = _TEMPLATES[key][2]()
    wf.metadata["template"] = key
    return wf
