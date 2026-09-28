# workflow/__init__.py
"""工作流引擎。

分层：
    schema / values   节点协议与端口数据类型（不依赖 torch）
    registry          节点注册表
    graph             拓扑排序 + 运行前校验
    memory_policy     能力配置驱动的显存策略
    cache             节点级缓存（支持只重跑某个节点）
    resource_manager  资源清单（只读本地，不联网）
    adapters          模型族适配器
    executor          执行器
    serializers       JSON 保存 / 加载 / 版本迁移
    builder           把现有简易界面状态编译成工作流图

设计约束：导入本包**不应**触发 torch/diffusers 的加载，
这样界面与静态工具可以随时使用（重依赖都在节点 run() 里按需 import）。
"""
from __future__ import annotations

from workflow.schema import (SCHEMA_VERSION, Cancelled, DataType, Node,
                             NodeError, NodeSpec, ParamSpec, PortSpec,
                             ResourceRequirement, Workflow, WorkflowError,
                             types_compatible)
from workflow import registry, graph, cache, serializers, adapters  # noqa: F401

__all__ = [
    "SCHEMA_VERSION", "Cancelled", "DataType", "Node", "NodeError",
    "NodeSpec", "ParamSpec", "PortSpec", "ResourceRequirement", "Workflow",
    "WorkflowError", "types_compatible",
    "registry", "graph", "cache", "serializers", "adapters",
    "executor", "memory_policy", "resource_manager", "builder",
]


def __getattr__(name):
    """惰性子模块：避免 `import workflow` 就把执行器/适配器全拖进来。"""
    if name in ("executor", "memory_policy", "resource_manager", "builder"):
        import importlib
        mod = importlib.import_module(f"workflow.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(name)
