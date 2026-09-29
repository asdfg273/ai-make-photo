# workflow/schema.py
"""工作流的基础数据类型与节点协议。

这是整个工作流系统**最重要**的部分：节点协议一旦定下来，界面、执行器、
缓存、显存策略、资源清单都围绕它展开。所以这里刻意只依赖标准库 +
已有的架构注册表，不 import torch/diffusers —— 这样 UI、校验器、
纯静态工具都能在不起重依赖的情况下使用。

三类核心对象
------------
1. `DataType`   端口上流动的数据种类（Model / Latent / Image ...）。
                连错类型是工作流最常见的错误，靠它做静态校验。
2. `PortSpec` / `ParamSpec`   一个节点有哪些接口、参数允许什么范围。
3. `NodeSpec`   节点类型定义（含资源需求），注册到 registry 后即可实例化。

`ResourceRequirement` 是显存策略的输入：执行器不再"看显卡名字"决定
是否 offload，而是汇总各节点的声明 + 设备能力配置来决策。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1


class DataType(str, Enum):
    """端口数据类型。字符串枚举，便于 JSON 序列化与人类阅读。"""

    MODEL = "MODEL"
    VAE = "VAE"
    TEXT_COND = "TEXT_COND"
    IMAGE = "IMAGE"
    LATENT = "LATENT"
    MASK = "MASK"
    CONTROL_IMAGE = "CONTROL_IMAGE"
    LORA = "LORA"
    SCHEDULER = "SCHEDULER"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    METADATA = "METADATA"
    # 区域提示词：一串 `core.regional.Region`（区域定义）。
    # 它不适合塞进 METADATA：METADATA 是"给人看的信息汇总"，
    # 而 REGIONS 是**要参与校验的强类型数据**（连错端口必须在运行前报出来）。
    REGIONS = "REGIONS"
    # 组合类型：SAMPLER 的输出既可能是 LATENT 也可能是 IMAGE，
    # 用 ANY 表示"不参与类型校验"的透传口（如元数据汇聚）。
    ANY = "ANY"


# 类型兼容规则：dst 端口声明 LATENT，src 给 ANY 也算通过（透传口）
def types_compatible(src: DataType, dst: DataType) -> bool:
    if src == dst:
        return True
    if DataType.ANY in (src, dst):
        return True
    # IMAGE 与 LATENT 不互通（必须显式经过 VAEEncode/VAEDecode），
    # 这是 ComfyUI 的核心约束，能挡掉绝大多数连错。
    return False


@dataclass(frozen=True)
class PortSpec:
    """一个输入/输出端口。"""

    type: DataType
    label: str = ""
    optional: bool = False

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", self.type.value)


@dataclass(frozen=True)
class ParamSpec:
    """节点参数（不通过连线传入的那些）。

    `kind` 取值：int / float / str / bool / choice / text / seed
    """

    kind: str
    default: Any = None
    label: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: tuple = ()
    help: str = ""
    # 选项来源："" | "checkpoints" | "loras" | "vae" | "images"
    # 给界面用的提示：非空表示这个参数是"用户机器上的本地文件"，界面可以
    # 用它（见 ui/workflow_editor/param_sources.py）列出候选、渲染成下拉框，
    # 免得用户自己背文件名；空字符串 = 自由文本，界面不做任何提示。
    # 只影响界面呈现，不参与 coerce / 执行语义，放最后以便向后兼容。
    source: str = ""

    def coerce(self, value):
        """把值转成该参数类型的规范形式；越界抛 ValueError。

        执行器在**运行前**会对全部参数调一次，把错误提前暴露，
        而不是等跑到一半才炸。
        """
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.kind == "int":
            v = int(value)
        elif self.kind in ("float",):
            v = float(value)
        elif self.kind in ("str", "text"):
            return "" if value is None else str(value)
        elif self.kind == "choice":
            s = "" if value is None else str(value)
            if self.choices and s not in self.choices:
                raise ValueError(
                    f"参数取值 {s!r} 不在允许范围 {list(self.choices)}")
            return s
        elif self.kind == "seed":
            v = int(value)
            # -1 表示随机
            if v < -1:
                raise ValueError(f"seed 不能小于 -1（得到 {v}）")
            return v
        else:
            return value

        if self.minimum is not None and v < self.minimum:
            raise ValueError(f"参数值 {v} 小于下限 {self.minimum}")
        if self.maximum is not None and v > self.maximum:
            raise ValueError(f"参数值 {v} 大于上限 {self.maximum}")
        return v


@dataclass(frozen=True)
class ResourceRequirement:
    """节点对运行环境的声明 —— 显存策略与资源清单的唯一输入。

    以前是否 offload 靠"看显卡名字/总显存"，加一张新卡就要改代码。
    改成节点声明后，8GB 卡自动启低显存策略、24GB 卡自动少卸载，
    新增 12/16/24GB 设备都不需要重写工作流。
    """

    requires: tuple = ()                 # 如 ("cuda", "model")
    estimated_vram_mb: int = 0           # 该节点峰值大致占用
    supports_cpu_offload: bool = True    # 能否靠 offload 降峰值
    supports_tiled: bool = False         # 能否分块处理
    exclusive_gpu: bool = False          # 是否要求独占（如整模常驻）
    needs_model: bool = False            # 是否依赖已加载的底模

    def to_dict(self) -> dict:
        return {
            "requires": list(self.requires),
            "estimated_vram_mb": self.estimated_vram_mb,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_tiled": self.supports_tiled,
            "exclusive_gpu": self.exclusive_gpu,
            "needs_model": self.needs_model,
        }


@dataclass
class NodeSpec:
    """节点类型定义。注册后由 registry 统一管理。"""

    type: str
    category: str = "通用"
    description: str = ""
    # 端口：名字 -> PortSpec
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    # 参数：名字 -> ParamSpec
    params: dict = field(default_factory=dict)
    resources: ResourceRequirement = field(default_factory=ResourceRequirement)
    # 要求输入必须来自连线（不可用字面量）；否则字面量优先
    force_links: tuple = ()
    run: Optional[Callable] = None       # fn(ctx, node, inputs) -> dict
    # 该节点是否可被"仅重跑此节点"复用上游缓存
    cacheable: bool = True

    def out_types(self) -> dict:
        return {k: v.type for k, v in self.outputs.items()}

    def in_types(self) -> dict:
        return {k: v.type for k, v in self.inputs.items()}

    def param_defaults(self) -> dict:
        return {k: v.default for k, v in self.params.items()}

    def to_dict(self) -> dict:
        """给 UI / 文档用的自描述（不含 run 回调）。"""
        return {
            "type": self.type,
            "category": self.category,
            "description": self.description,
            "inputs": {k: {"type": v.type.value, "label": v.label,
                           "optional": v.optional}
                       for k, v in self.inputs.items()},
            "outputs": {k: {"type": v.type.value, "label": v.label}
                        for k, v in self.outputs.items()},
            "params": {k: dataclasses.asdict(v) for k, v in self.params.items()},
            "resources": self.resources.to_dict(),
            "force_links": list(self.force_links),
        }


# ----------------------------------------------------------------------
#  工作流实例（可序列化的那一层，纯数据）
# ----------------------------------------------------------------------
@dataclass
class Node:
    """工作流里的一个节点实例。

    `inputs` 里的值有两种形态：
      · 字面量：{"text": "a cat"}
      · 连线  ：{"model": ["model_1", "model"]}   # [源节点 id, 源端口名]
    用"列表=连线、其它=字面量"这一个约定同时表达两者，JSON 友好。
    """

    id: str
    type: str
    inputs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    # 运行期状态（不参与序列化语义，但保存下来便于排错）
    status: str = "idle"          # idle | running | ok | error | skipped
    error: str = ""
    elapsed_ms: int = 0
    cached: bool = False

    def link_refs(self) -> dict:
        """{输入名: (源节点id, 源端口)}"""
        out = {}
        for k, v in self.inputs.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 \
                    and isinstance(v[0], str):
                out[k] = (v[0], v[1])
        return out

    def literals(self) -> dict:
        return {k: v for k, v in self.inputs.items()
                if k not in self.link_refs()}

    def to_dict(self, with_status: bool = False) -> dict:
        d = {"id": self.id, "type": self.type,
             "inputs": self.inputs, "params": self.params}
        if with_status:
            d["_status"] = {"status": self.status, "error": self.error,
                            "elapsed_ms": self.elapsed_ms, "cached": self.cached}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(id=d["id"], type=d["type"],
                   inputs=dict(d.get("inputs") or {}),
                   params=dict(d.get("params") or {}))


@dataclass
class Workflow:
    """一张完整的工作流图。"""

    name: str = "未命名工作流"
    nodes: list = field(default_factory=list)
    # 冗余保存连线（便于外部工具直接读 links，不必反解 inputs）
    links: list = field(default_factory=list)   # [(src_id, src_port, dst_id, dst_port)]
    schema_version: int = SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    # ---- 查询 ----
    def node(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_ids(self) -> list:
        return [n.id for n in self.nodes]

    def by_type(self, ntype: str) -> list:
        return [n for n in self.nodes if n.type == ntype]

    def add(self, node: Node) -> Node:
        if self.node(node.id) is not None:
            raise ValueError(f"节点 id 重复: {node.id}")
        self.nodes.append(node)
        return node

    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None:
        dst = self.node(dst_id)
        if dst is None:
            raise KeyError(f"目标节点不存在: {dst_id}")
        dst.inputs[dst_port] = [src_id, src_port]
        self.links.append((src_id, src_port, dst_id, dst_port))

    def to_dict(self, with_status: bool = False) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "nodes": [n.to_dict(with_status) for n in self.nodes],
            "links": [list(l) for l in self.links],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        wf = cls(name=d.get("name", "未命名工作流"),
                 schema_version=int(d.get("schema_version", 1)),
                 metadata=dict(d.get("metadata") or {}))
        for nd in d.get("nodes") or []:
            wf.nodes.append(Node.from_dict(nd))
        wf.links = [tuple(l) for l in (d.get("links") or [])]
        # 容错：links 与 inputs 不一致时以 links 为准补齐，反之亦然
        for (s, sp, di, dp) in wf.links:
            n = wf.node(di)
            if n is not None and dp not in n.inputs:
                n.inputs[dp] = [s, sp]
        return wf


# ----------------------------------------------------------------------
#  执行错误：必须能指出"是哪个节点错了"
# ----------------------------------------------------------------------
class WorkflowError(Exception):
    """工作流级错误（结构非法、无法执行）。"""


class NodeError(Exception):
    """单个节点执行失败。带上节点 id 与类型，方便界面高亮定位。"""

    def __init__(self, node_id: str, node_type: str, message: str,
                 node_inputs: dict | None = None):
        self.node_id = node_id
        self.node_type = node_type
        self.message = message
        self.node_inputs = node_inputs or {}
        super().__init__(f"[{node_id} / {node_type}] {message}")


class Cancelled(Exception):
    """用户取消。"""
