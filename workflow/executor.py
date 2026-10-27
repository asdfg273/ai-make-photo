# workflow/executor.py
"""工作流执行器。

职责（对应需求里列的那些能力）
------------------------------
· 运行前校验节点连接与参数          -> graph.validate()
· 检查模型架构兼容性                -> adapters.supports()
· 按显存选择 offload / 低显存 / 分块 -> memory_policy.plan_for_nodes()
· 缓存可复用的模型与中间结果        -> cache.NodeCache
· 支持取消任务                      -> cancel_check
· 保存每个节点的运行状态            -> Node.status / error / elapsed_ms / cached
· 失败时指明是哪个节点              -> NodeError 带 node_id
· 允许只重跑某一个节点              -> run(targets=[...]) / run(only=[...])

执行器**不**关心节点内部怎么做，只负责：取值 → 调 run() → 收产出 → 记状态。
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from workflow import cache as cache_mod
from workflow import graph as graph_mod
from workflow import registry
from workflow.memory_policy import (DeviceCapabilities, MemoryPolicy, plan_for_nodes,
                                    resolve as resolve_policy)
from workflow.schema import (Cancelled, DataType, NodeError, Workflow,
                             WorkflowError)

logger = logging.getLogger(__name__)


@dataclass
class NodeContext:
    """传给节点 run() 的运行期上下文。

    节点只应通过它访问宿主（模型管理器、UI 桥），**不要**自己去 import
    main / 操作界面控件 —— 否则工作流就没法脱离界面单独跑（测试、
    批处理、将来的无头模式都指望这一点）。
    """
    host: Any = None                     # 通常是 AIDesktopApp，可为 None
    ai: Any = None                       # ModelManager
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    device: str = "cuda"
    cache: Any = None
    cancel_check: Optional[Callable[[], bool]] = None
    log: Callable[[str], None] = lambda msg: logger.info(msg)
    status: Callable[[str], None] = lambda msg: None
    progress: Callable[[int, int], None] = lambda a, b: None
    # 执行器写入：当前节点 id（节点可据此命名临时文件）
    current_node: str = ""
    # 执行器写入：本次正在跑的 Workflow（SaveImage 等节点可据此把整张图
    # 嵌进图片元数据 —— 否则"保存工作流"这个开关就是个摆设）
    workflow: Any = None
    node_order: list = field(default_factory=list)
    # 节点可往这里塞"要写进元数据"的东西
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 当前模型的能力描述（ModelCapabilities）。节点应当判断**能力**而不是
    # 判断模型名：`ctx.capabilities.require("controlnet")`。
    # 由 LoadCheckpoint 加载后写入；没加载模型时为 None。
    capabilities: Any = None
    # 界面可注入翻译函数；后台与既有调用默认保留中文。
    translate: Callable[[str], str] = lambda text: text

    def cancelled(self) -> bool:
        try:
            return bool(self.cancel_check and self.cancel_check())
        except Exception:
            return False

    def check_cancel(self):
        if self.cancelled():
            raise Cancelled("用户已取消")

    def config(self):
        return getattr(self.host, "config", None)


class Executor:
    """一次执行 = 一份 Executor。可重复 run() 以复用缓存。"""

    def __init__(self, ctx: NodeContext, *, use_cache: bool = True,
                 max_cache_entries: int = 24):
        self.ctx = ctx
        self.cache = ctx.cache or cache_mod.NodeCache(
            max_entries=max_cache_entries,
            enabled=use_cache and ctx.policy.cache_intermediates)
        ctx.cache = self.cache
        self.last_states: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    def _text(self, text: str, **values) -> str:
        return self.ctx.translate(text).format(**values)

    def _log(self, text: str, **values) -> None:
        self.ctx.log(self._text(text, **values))

    def validate(self, wf: Workflow) -> List[graph_mod.ValidationIssue]:
        return graph_mod.validate(wf, translate=self.ctx.translate)

    def plan(self, wf: Workflow) -> dict:
        specs = []
        for n in wf.nodes:
            if registry.exists(n.type):
                specs.append((n.id, registry.get(n.type)))
        # 复用上下文里已探测好的设备能力，避免每次 plan 都重新探测
        cap = (self.ctx.metadata or {}).get("_device_caps")
        return plan_for_nodes(self.ctx.policy, specs, cap=cap)

    # ------------------------------------------------------------------
    def run(self, wf: Workflow, *,
            targets: Optional[List[str]] = None,
            only: Optional[List[str]] = None,
            use_cache: Optional[bool] = None,
            dry_run: bool = False) -> Dict[str, Any]:
        """执行工作流。

        targets: 只跑到这些节点（含其上游）。
        only   : 只**执行**这些节点，其余用缓存里的结果（"重跑某个节点"）。
        dry_run: 只做校验与计划，不真跑。
        """
        t_start = time.time()
        registry.load_builtin_nodes()

        # ---- 1. 校验 ----
        issues = self.validate(wf)
        errs = [i for i in issues if i.level == "error"]
        if errs:
            return {"ok": False, "error": self._text("工作流校验未通过"),
                    "issues": [i.to_dict() for i in issues],
                    "failed_node": errs[0].node_id, "states": {},
                    "outputs": {}, "seconds": 0.0,
                    "plan": self.plan(wf)}

        # ---- 2. 执行顺序 ----
        try:
            order = graph_mod.topo_sort(wf, translate=self.ctx.translate)
        except WorkflowError as e:
            return {"ok": False, "error": str(e), "failed_node": "",
                    "issues": [i.to_dict() for i in issues], "states": {},
                    "outputs": {}, "seconds": 0.0, "plan": self.plan(wf)}

        # 只跑到 targets（含上游）
        if targets:
            keep = set(graph_mod.needed_for(wf, targets))
            order = [i for i in order if i in keep]
        only_set = set(only) if only else None

        plan = self.plan(wf)
        if dry_run:
            return {"ok": True, "error": "", "failed_node": "", "issues":
                    [i.to_dict() for i in issues], "states": {},
                    "outputs": {}, "seconds": 0.0, "plan": plan,
                    "order": order, "dry_run": True}

        self._log("▶️ 工作流开始：{name}（{count} 个节点）",
                  name=wf.name, count=len(order))
        self._log("🧠 显存策略：{policy}",
                  policy=self.ctx.policy.describe(translate=self.ctx.translate))

        produced: Dict[str, Dict[str, Any]] = {}
        states: Dict[str, dict] = {}
        cached_nodes: List[str] = []
        failed_node = ""
        error_msg = ""

        try:
            for nid in order:
                node = wf.node(nid)
                spec = registry.get(node.type)
                node.status, node.error, node.cached = "running", "", False
                t0 = time.time()

                # ---- 取值 ----
                inputs, missing = graph_mod.resolve_inputs(wf, node, produced)
                if missing:
                    raise NodeError(
                        nid, node.type,
                        self._text("缺少输入: {inputs}", inputs=", ".join(missing)),
                        node.inputs)

                # ---- 缓存 ----
                want_cache = (self.cache.enabled if use_cache is None
                              else bool(use_cache))
                key = cache_mod.make_key(node.type, inputs, node.params)

                # `only=[...]` 里的节点是用户**点名要重跑**的，必须跳过缓存强制执行。
                # ⚠️ 旧代码写的是 `if hit is not None and (skip_cached or True):`
                # —— `or True` 恒真（死代码），于是连 only 里的节点也照常吃缓存：
                # 用户点「🎯 只跑选中节点」会看到"完成"，实际上什么都没跑。
                force_exec = only_set is not None and nid in only_set
                if want_cache and spec.cacheable and not force_exec:
                    hit = self.cache.get(key)
                    if hit is not None:
                        produced[nid] = hit
                        node.status, node.cached = "ok", True
                        node.elapsed_ms = int((time.time() - t0) * 1000)
                        states[nid] = dict(node.to_dict(True)["_status"])
                        states[nid]["node_id"] = nid
                        cached_nodes.append(nid)
                        self._log("♻️ 复用缓存: {node} ({type})",
                                  node=nid, type=node.type)
                        continue
                elif force_exec:
                    self._log("🎯 强制重跑: {node} ({type})",
                              node=nid, type=node.type)

                self.ctx.check_cancel()
                self.ctx.current_node = nid
                self.ctx.workflow = wf
                self.ctx.node_order = order

                # ---- 执行 ----
                if spec.run is None:
                    raise NodeError(
                        nid, node.type, self._text("该节点没有实现 run()"))
                try:
                    out = spec.run(self.ctx, node, inputs)
                except Cancelled:
                    raise
                except NodeError:
                    raise
                except Exception as e:            # 统一包成 NodeError 便于定位
                    raise NodeError(nid, node.type,
                                    f"{type(e).__name__}: {e}", node.inputs) from e
                if out is None:
                    out = {}
                if not isinstance(out, dict):
                    raise NodeError(
                        nid, node.type,
                        self._text("run() 必须返回 dict，实际 {type}",
                                   type=type(out).__name__))

                produced[nid] = out
                node.status = "ok"
                node.elapsed_ms = int((time.time() - t0) * 1000)
                self.cache.bind(nid, key)
                if want_cache and spec.cacheable:
                    self.cache.put(key, out)
                states[nid] = dict(node.to_dict(True)["_status"])
                states[nid]["node_id"] = nid
                self.ctx.log(f"✅ {nid} ({node.type}) {node.elapsed_ms}ms")

        except Cancelled:
            for nid in order:
                node = wf.node(nid)
                if node.status == "running":
                    node.status = "skipped"
            elapsed = time.time() - t_start
            self._log("⏹️ 工作流已取消")
            return {"ok": False, "cancelled": True,
                    "error": self._text("已取消"),
                    "failed_node": "", "states": states,
                    "outputs": self._leaf_outputs(wf, order, produced),
                    "cached": cached_nodes, "cache_stats": self.cache.stats(),
                    "issues": [i.to_dict() for i in issues],
                    "seconds": elapsed, "plan": plan}
        except NodeError as e:
            failed_node, error_msg = e.node_id, self.ctx.translate(e.message)
            n = wf.node(e.node_id)
            if n is not None:
                n.status, n.error = "error", error_msg
            states[e.node_id] = dict(n.to_dict(True)["_status"]) if n else {
                "status": "error", "error": error_msg}
            states[e.node_id]["node_id"] = e.node_id
            self._log("❌ 节点失败 [{node} / {type}] {error}",
                      node=e.node_id, type=e.node_type, error=error_msg)
            # ⚠️ 不能用 `x.__traceback`：类体内的双下划线属性会被 Python
            # 名字改写成 `_Executor__traceback`，结果"记录失败原因"这行
            # 自己先抛 AttributeError，把真正的原因盖掉。必须 getattr。
            cause = e.__cause__ or e
            tb = getattr(cause, "__traceback__", None)
            logger.error("节点 %s 执行失败:\n%s", e.node_id,
                         "".join(traceback.format_exception(
                             type(cause), cause, tb)))
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._log("❌ 工作流异常: {error}", error=error_msg)
            logger.exception("工作流执行异常")

        elapsed = time.time() - t_start
        ok = not error_msg
        if ok:
            self._log("🏁 工作流完成，用时 {seconds}s",
                      seconds=f"{elapsed:.1f}")
        return {"ok": ok, "error": error_msg, "failed_node": failed_node,
                "states": states, "outputs": self._leaf_outputs(wf, order, produced),
                "cached": cached_nodes, "cache_stats": self.cache.stats(),
                "issues": [i.to_dict() for i in issues],
                "seconds": elapsed, "plan": plan,
                "images": self._node_images(wf, order, produced)}

    # ------------------------------------------------------------------
    @staticmethod
    def _node_images(wf: Workflow, order: List[str],
                     produced: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """每个节点产出的 IMAGE 值，供编辑器把图渲染进节点卡片。

        只取每个节点**第一个** IMAGE 输出（节点通常只有一个图像输出），
        并且只保留 PIL 图 / 文件路径字符串。这些对象本来就被缓存持有，
        这里只是多一份引用，不会额外占多少内存。
        """
        imgs: Dict[str, Any] = {}
        for nid in order:
            node = wf.node(nid)
            if node is None or not registry.exists(node.type):
                continue
            spec = registry.get(node.type)
            out_ports = getattr(spec, "outputs", {}) or {}
            vals = produced.get(nid) or {}
            for pname, pspec in out_ports.items():
                if pspec.type is not DataType.IMAGE:
                    continue
                v = vals.get(pname)
                if v is None:
                    continue
                try:
                    from PIL import Image as _I
                    if isinstance(v, _I.Image):
                        imgs[nid] = v
                        break
                except Exception:
                    pass
                if isinstance(v, str) and v:
                    imgs[nid] = v
                    break
        return imgs

    # ------------------------------------------------------------------
    @staticmethod
    def _leaf_outputs(wf: Workflow, order: List[str],
                      produced: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """把"没有被下游消费"的产出汇总，作为工作流结果返回。"""
        consumed = {lk[0] for lk in wf.links}
        out: Dict[str, Any] = {}
        for nid in order:
            if nid in consumed:
                continue
            for k, v in (produced.get(nid) or {}).items():
                out[f"{nid}.{k}"] = v
        return out


def make_context(host=None, ai=None, *, preset: str = "auto",
                 arch_id: str = "", overrides: Optional[dict] = None,
                 cancel_check=None, log=None, status=None,
                 progress=None, cap: Optional[DeviceCapabilities] = None,
                 translate=None) -> NodeContext:
    """便捷构造：探测设备 → 解析策略 → 组装 NodeContext。"""
    cap = cap or DeviceCapabilities.detect()
    if not arch_id and ai is not None:
        arch_id = getattr(ai, "current_arch_id", "") or ""
    policy = resolve_policy(preset, cap, arch_id=arch_id, overrides=overrides)
    ctx = NodeContext(
        host=host, ai=ai or getattr(host, "ai", None), policy=policy,
        device="cuda" if cap.has_cuda else "cpu",
        cancel_check=cancel_check,
        log=log or (lambda m: logger.info(m)),
        status=status or (lambda m: None),
        progress=progress or (lambda a, b: None),
        translate=translate or (lambda text: text),
    )
    # 让模型适配器也能拿到能力信息
    ctx.metadata["_device_caps"] = cap
    return ctx


def run_workflow(wf: Workflow, host=None, ai=None, **kw) -> Dict[str, Any]:
    """一步到位：建上下文 → 执行。"""
    ctx = make_context(host=host, ai=ai,
                       preset=kw.pop("preset", "auto"),
                       arch_id=kw.pop("arch_id", ""),
                       overrides=kw.pop("overrides", None),
                       cancel_check=kw.pop("cancel_check", None),
                       log=kw.pop("log", None),
                       status=kw.pop("status", None),
                       progress=kw.pop("progress", None),
                       translate=kw.pop("translate", None))
    return Executor(ctx, use_cache=kw.pop("use_cache", True)).run(wf, **kw)
