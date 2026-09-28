# nodes/__init__.py
"""内置工作流节点。

每个模块用 `@node(...)` 自注册；`workflow.registry.load_builtin_nodes()`
会 import 下面这些模块来触发注册。新增节点 = 新增一个函数，
不用改执行器、不用改界面（界面从注册表自动生成控件）。
"""
from __future__ import annotations

__all__ = [
    "model_nodes", "text_nodes", "sampling_nodes", "image_nodes",
    "control_nodes", "adapter_nodes", "detailer_nodes", "upscale_nodes",
    "video_nodes",
]
