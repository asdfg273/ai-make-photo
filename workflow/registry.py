# workflow/registry.py
"""节点注册表：节点类型 -> NodeSpec。

节点通过 `@register(...)` 装饰器自注册；`load_builtin_nodes()` 会 import
`nodes/` 下所有模块来触发注册。这样新增节点只需要新增文件，
不用改执行器、不用改界面。

刻意**不**提供"任意 Python 节点"入口：那会让调试、显存管理和错误定位
迅速失控（用户自己的代码炸了，报错会指向节点内部而不是工作流）。
将来的插件机制也要走这个注册表，而不是 eval。
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

from workflow.schema import NodeSpec, ResourceRequirement

logger = logging.getLogger(__name__)

_SPECS: dict[str, NodeSpec] = {}

# 内置节点模块（相对项目根的模块名）
BUILTIN_NODE_MODULES = (
    "nodes.model_nodes",
    "nodes.text_nodes",
    "nodes.sampling_nodes",
    "nodes.image_nodes",
    "nodes.control_nodes",
    "nodes.adapter_nodes",
    "nodes.detailer_nodes",
    "nodes.regional_nodes",
    "nodes.upscale_nodes",
    "nodes.tile_nodes",
    "nodes.postprocess_nodes",
    "nodes.video_nodes",
)

_LOADED = False


def register(spec: NodeSpec) -> NodeSpec:
    """注册一个节点类型。重复注册会覆盖并告警（热重载友好）。"""
    if not spec.type:
        raise ValueError("NodeSpec.type 不能为空")
    if spec.type in _SPECS:
        logger.debug(f"节点重复注册，覆盖: {spec.type}")
    _SPECS[spec.type] = spec
    return spec


def node(node_type: str, category: str = "通用", description: str = "",
         inputs: Optional[dict] = None, outputs: Optional[dict] = None,
         params: Optional[dict] = None,
         resources: Optional[ResourceRequirement] = None,
         force_links: tuple = (), cacheable: bool = True
         ) -> Callable:
    """装饰器：把 `run(ctx, node, inputs)` 变成已注册的节点。

    用法::

        @node("TextEncode", category="文本", outputs={"conditioning": PortSpec(DataType.TEXT_COND)})
        def run_text_encode(ctx, node, inputs):
            ...
            return {"conditioning": cond}
    """
    def deco(fn: Callable) -> Callable:
        register(NodeSpec(
            type=node_type, category=category, description=description,
            inputs=dict(inputs or {}), outputs=dict(outputs or {}),
            params=dict(params or {}),
            resources=resources or ResourceRequirement(),
            force_links=tuple(force_links), run=fn, cacheable=cacheable))
        return fn
    return deco


def exists(node_type: str) -> bool:
    return node_type in _SPECS


def get(node_type: str) -> NodeSpec:
    spec = _SPECS.get(node_type)
    if spec is None:
        raise KeyError(f"未注册的节点类型: {node_type}")
    return spec


def all_specs() -> dict[str, NodeSpec]:
    return dict(_SPECS)


def by_category() -> dict[str, list[NodeSpec]]:
    out: dict[str, list[NodeSpec]] = {}
    for spec in _SPECS.values():
        out.setdefault(spec.category, []).append(spec)
    for v in out.values():
        v.sort(key=lambda s: s.type)
    return out


def load_builtin_nodes(force: bool = False) -> int:
    """import 全部内置节点模块，返回注册到的节点数。"""
    global _LOADED
    if _LOADED and not force:
        return len(_SPECS)
    for mod in BUILTIN_NODE_MODULES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            # ⚠️ 必须区分两种情况，不能只看异常类型：
            #   · 节点模块本身不存在（渐进迁移期很常见）→ 可安全跳过
            #   · 模块存在，但它**内部**依赖的包缺失（如可选的 controlnet_aux）
            #     → 绝不能当成"模块缺失"静默跳过，否则那一整批节点会凭空消失，
            #       用户看到的只是"节点库里没有这个节点"，完全无从排查。
            if getattr(e, "name", "") == mod or (e.name or "").startswith(mod + "."):
                logger.warning(f"⚠️ 节点模块缺失，跳过: {mod} ({e})")
            else:
                logger.error(
                    f"❌ 节点模块 {mod} 内部依赖缺失（{e.name}）→ "
                    f"该模块的节点未注册。请安装依赖: pip install {e.name}")
        except Exception as e:
            logger.error(f"❌ 节点模块加载失败: {mod}: {e}", exc_info=True)
    _LOADED = True
    return len(_SPECS)


def clear() -> None:
    """清空注册表（测试用）。"""
    global _LOADED
    _SPECS.clear()
    _LOADED = False


def catalog() -> list[dict]:
    """给 UI / 文档用的节点目录。"""
    return [s.to_dict() for s in sorted(_SPECS.values(),
                                        key=lambda s: (s.category, s.type))]
