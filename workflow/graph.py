# workflow/graph.py
"""图算法与**运行前校验**。

执行器在真正跑之前必须先把能静态发现的问题全部报出来：
  · 节点类型未注册
  · 必填输入既没连线也没字面量
  · 连线两端端口不存在 / 类型不兼容
  · 参数越界或取值非法
  · 有环（无法拓扑排序）
  · 重复节点 id
这样用户看到的是"第 3 个节点的 steps 超范围"，而不是跑到一半崩在 diffusers 里。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from workflow import registry
from workflow.schema import (Node, NodeSpec, Workflow, WorkflowError,
                             types_compatible)


class ValidationIssue:
    __slots__ = ("level", "node_id", "field", "message")

    def __init__(self, level: str, node_id: str, message: str, field: str = ""):
        self.level = level          # "error" | "warning"
        self.node_id = node_id
        self.field = field
        self.message = message

    def __repr__(self):
        where = f"{self.node_id}.{self.field}" if self.field else self.node_id
        return f"<{self.level} {where}: {self.message}>"

    def to_dict(self) -> dict:
        return {"level": self.level, "node_id": self.node_id,
                "field": self.field, "message": self.message}


def _spec_of(node: Node) -> Optional[NodeSpec]:
    if not registry.exists(node.type):
        return None
    return registry.get(node.type)


def validate(wf: Workflow, *, strict: bool = True) -> List[ValidationIssue]:
    """全量静态校验。返回问题列表（error 阻断执行）。"""
    issues: List[ValidationIssue] = []

    # ---- 1. 重复 id ----
    seen = set()
    for n in wf.nodes:
        if n.id in seen:
            issues.append(ValidationIssue("error", n.id, "节点 id 重复"))
        seen.add(n.id)

    # ---- 2. 逐节点 ----
    for n in wf.nodes:
        spec = _spec_of(n)
        if spec is None:
            issues.append(ValidationIssue(
                "error", n.id, f"未注册的节点类型: {n.type}"))
            continue

        # 2a. 必填输入
        for port, pspec in spec.inputs.items():
            has_link = port in n.link_refs()
            has_lit = port in n.literals() and n.literals()[port] is not None
            if not has_link and not has_lit:
                if not pspec.optional:
                    issues.append(ValidationIssue(
                        "error", n.id, f"缺少必填输入 {port}({pspec.type.value})",
                        port))
            if port in spec.force_links and not has_link:
                issues.append(ValidationIssue(
                    "error", n.id, f"输入 {port} 必须来自连线", port))

        # 2b. 未声明的输入名（拼写错误极易发生）
        declared = set(spec.inputs) | set(spec.params)
        for k in n.inputs:
            if k not in declared:
                issues.append(ValidationIssue(
                    "warning", n.id, f"输入 {k!r} 未被节点声明（将被忽略）", k))

        # 2c. 参数取值
        for pname, pspec in spec.params.items():
            if pname not in n.params:
                continue
            try:
                pspec.coerce(n.params[pname])
            except Exception as e:
                issues.append(ValidationIssue(
                    "error", n.id, f"参数 {pname} 非法: {e}", pname))

        # 2d. 连线来源存在性 + 类型
        for port, (src_id, src_port) in n.link_refs().items():
            src = wf.node(src_id)
            if src is None:
                issues.append(ValidationIssue(
                    "error", n.id, f"输入 {port} 连到不存在的节点 {src_id}", port))
                continue
            sspec = _spec_of(src)
            if sspec is None:
                continue
            if src_port not in sspec.outputs:
                issues.append(ValidationIssue(
                    "error", n.id,
                    f"输入 {port} 连到 {src_id} 的不存在输出 {src_port}", port))
                continue
            if port in spec.inputs:
                a = sspec.outputs[src_port].type
                b = spec.inputs[port].type
                if not types_compatible(a, b):
                    issues.append(ValidationIssue(
                        "error", n.id,
                        f"类型不匹配: {src_id}.{src_port}({a.value}) → "
                        f"{port}({b.value})", port))

    # ---- 3. 有环检测（顺带给出参与环的节点）----
    try:
        topo_sort(wf)
    except WorkflowError as e:
        issues.append(ValidationIssue("error", "", str(e)))

    # ---- 4. 没有终点节点（warning）----
    # 每个节点都有下游 = 没有任何产出被消费，通常是忘了接 SaveImage/预览。
    if wf.nodes:
        consumed = {lk[0] for lk in wf.links}
        if not any(n.id not in consumed for n in wf.nodes):
            issues.append(ValidationIssue(
                "warning", "", "图中所有节点都有下游，没有终点节点"
                               "（通常意味着忘了接 SaveImage）"))

    return issues


def errors_only(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    return [i for i in issues if i.level == "error"]


def topo_sort(wf: Workflow) -> List[str]:
    """拓扑排序（Kahn）。有环抛 WorkflowError 并指出环上节点。"""
    ids = wf.node_ids()
    indeg: Dict[str, int] = {i: 0 for i in ids}
    adj: Dict[str, List[str]] = {i: [] for i in ids}

    for (src, _sp, dst, _dp) in wf.links:
        if src in indeg and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1

    q = deque([i for i in ids if indeg[i] == 0])
    order: List[str] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(ids):
        stuck = [i for i in ids if i not in set(order)]
        raise WorkflowError(f"工作流存在环路，涉及节点: {', '.join(stuck)}")
    return order


def upstream_of(wf: Workflow, node_id: str) -> List[str]:
    """某个节点的全部上游（含间接），按拓扑序返回。"""
    order = topo_sort(wf)
    parents: Dict[str, set] = {i: set() for i in wf.node_ids()}
    for (src, _sp, dst, _dp) in wf.links:
        if dst in parents:
            parents[dst].add(src)
    need = set()
    stack = list(parents.get(node_id, ()))
    while stack:
        cur = stack.pop()
        if cur in need:
            continue
        need.add(cur)
        stack.extend(parents.get(cur, ()))
    return [i for i in order if i in need]


def needed_for(wf: Workflow, targets: List[str]) -> List[str]:
    """执行 targets 所需的节点集合（targets + 其全部上游），拓扑序。"""
    keep = set(targets)
    for t in targets:
        keep.update(upstream_of(wf, t))
    return [i for i in topo_sort(wf) if i in keep]


def resolve_inputs(wf: Workflow, node: Node,
                   produced: Dict[str, dict]) -> Tuple[dict, List[str]]:
    """把节点的输入解析成实际值。

    连线优先取上游产出；没有连线时用字面量；再没有就用参数默认值。
    返回 (inputs, 缺失的必填输入名列表)。
    """
    spec = _spec_of(node)
    values: dict = {}
    missing: List[str] = []

    for port, pspec in (spec.inputs.items() if spec else ()):
        ref = node.link_refs().get(port)
        if ref is not None:
            src_id, src_port = ref
            up = produced.get(src_id) or {}
            if src_port in up:
                values[port] = up[src_port]
                continue
            if not pspec.optional:
                missing.append(port)
            continue
        lit = node.literals().get(port)
        if lit is not None:
            values[port] = lit
            continue
        if pspec.optional:
            continue
        missing.append(port)

    return values, missing
