# workflow/serializers.py
"""工作流 JSON 的保存 / 加载 / 版本迁移。

文件格式（与需求里给的示例一致）：
    {
      "schema_version": 1,
      "name": "...",
      "nodes": [{"id": "...", "type": "...", "inputs": {...}}],
      "links": [["model_1", "model", "sampler_1", "model"]]
    }

加载时的容错原则
----------------
工作流文件可能来自旧版本、别的机器、或用户手改。加载**不应该**因为
一个未知节点类型就整份废掉：未知类型标记成 `_unknown`，等校验阶段再报，
这样用户能在编辑器里看到并修好它，而不是面对一个空画布。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from workflow import registry
from workflow.schema import SCHEMA_VERSION, Workflow

logger = logging.getLogger(__name__)

DEFAULT_DIRNAME = "workflows"


def workflows_dir() -> str:
    from utils import paths
    d = os.path.join(paths.PROJECT_ROOT, DEFAULT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _migrate(d: dict) -> dict:
    """按 schema_version 做迁移。目前只有 v1，留好扩展点。"""
    v = int(d.get("schema_version", 1) or 1)
    if v > SCHEMA_VERSION:
        logger.warning(
            "⚠️ 工作流版本 %s 高于程序支持的 %s，可能有字段被忽略",
            v, SCHEMA_VERSION)
    # 未来：if v == 1: d = _v1_to_v2(d)
    d["schema_version"] = SCHEMA_VERSION
    return d


def dumps(wf: Workflow, *, with_status: bool = False, indent: int = 2) -> str:
    return json.dumps(wf.to_dict(with_status), ensure_ascii=False,
                      indent=indent)


def loads(text: str) -> Workflow:
    d = _migrate(json.loads(text))
    wf = Workflow.from_dict(d)
    normalize_params(wf)
    _mark_unknown(wf)
    return wf


def normalize_params(wf: Workflow) -> int:
    """把写在 `inputs` 里、但其实是**节点参数**的键搬到 `params`。

    为什么必须做
    ------------
    对外约定的工作流格式（也是本文档里给的示例）是：

        {"id": "sampler_1", "type": "Sampler",
         "inputs": {"steps": 28, "cfg": 6.5, "seed": 123}}

    也就是说用户/AI 生成的 JSON 把"参数"写在 `inputs` 里。而内部
    `Node` 区分 `inputs`（连线/字面量）与 `params`（节点设置）。如果
    不做这一步，加载回来的节点 `params` 全空 —— Sampler 会按默认步数跑、
    LoadCheckpoint 直接报"没有指定模型"，也就是**文档格式根本跑不起来**。

    规则：`inputs[k]` 是普通值（不是 [源节点,端口] 连线）且 `k` 是该节点
    声明的参数名 → 搬进 `params`。是连线就留在 inputs。
    返回搬运的键数。
    """
    moved = 0
    for n in wf.nodes:
        if not registry.exists(n.type):
            continue                      # 未知类型等修复后再规范化
        spec = registry.get(n.type)
        if not spec.params:
            continue
        for k in list(n.inputs.keys()):
            if k not in spec.params:
                continue
            v = n.inputs[k]
            is_link = (isinstance(v, (list, tuple)) and len(v) == 2
                       and isinstance(v[0], str))
            if is_link:
                continue
            n.params.setdefault(k, v)
            n.inputs.pop(k, None)
            moved += 1
    return moved


def _mark_unknown(wf: Workflow) -> List[str]:
    unknown = []
    for n in wf.nodes:
        if not registry.exists(n.type):
            unknown.append(n.id)
    if unknown:
        logger.warning("⚠️ 工作流含未注册节点类型: %s（保留在图中待修复）",
                       ", ".join(unknown))
    return unknown


def save(wf: Workflow, path: str) -> str:
    """原子写（先写临时文件再 replace），避免写坏已有工作流。"""
    path = path if os.path.isabs(path) else os.path.join(workflows_dir(), path)
    if not path.lower().endswith(".json"):
        path += ".json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(dumps(wf))
    os.replace(tmp, path)
    logger.info(f"💾 工作流已保存: {path}")
    return path


def load(path: str) -> Workflow:
    path = path if os.path.isabs(path) else os.path.join(workflows_dir(), path)
    with open(path, encoding="utf-8") as f:
        return loads(f.read())


def _safe_name(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", (name or "workflow").strip())
    return (s[:60] or "workflow") + ".json"


def save_named(wf: Workflow, name: Optional[str] = None) -> str:
    return save(wf, _safe_name(name or wf.name))


def list_workflows() -> List[dict]:
    """列出本地工作流文件（给模板库 / 编辑器用）。"""
    out = []
    try:
        for fn in sorted(os.listdir(workflows_dir())):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(workflows_dir(), fn)
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                out.append({"file": fn, "path": p,
                            "name": d.get("name", fn),
                            "nodes": len(d.get("nodes") or []),
                            "schema_version": d.get("schema_version", 1),
                            "mtime": os.path.getmtime(p)})
            except Exception as e:
                out.append({"file": fn, "path": p, "name": fn,
                            "error": str(e)})
    except OSError:
        pass
    return out


def delete(path: str) -> bool:
    p = path if os.path.isabs(path) else os.path.join(workflows_dir(), path)
    try:
        os.remove(p)
        return True
    except OSError as e:
        logger.warning(f"⚠️ 删除工作流失败 {p}: {e}")
        return False
