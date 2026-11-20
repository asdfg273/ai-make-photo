# nodes/python_nodes.py
"""任意 Python 节点：让用户在工作流里直接写一段 Python。

为什么默认关闭
--------------
早期设计里这个入口是刻意不做的（见 `workflow/registry.py` 的历史说明）：
用户自己的代码炸了，报错会指向节点内部，排查成本高；而且能在界面里
执行任意代码，等于把"运行别人给的工作流"变成一个风险动作。现在按需求
开放，但保留三道闸：

1. **默认关闭**：`AppConfig.allow_python_nodes` 为 False 时，节点照常出现在
   节点库里（这样能看见、能连线、能存盘），但 `run()` 直接抛一条中文错误，
   告诉用户去哪儿打开。不打开就绝不执行任何用户代码。
2. **单次执行有超时**：默认 30 秒，超时就报错并放弃等待（线程是 daemon，
   不会被强制杀死 —— 这一点在错误信息里写明，避免用户以为"已经停了"）。
3. **可关掉的 import 与危险内置**：`allow_imports=False`（默认）时只放行
   math/random/json 等纯计算模块，并从 `builtins` 里摘掉 `open` / `exec` /
   `eval` / `compile` / `input` / `breakpoint` / `globals` / `locals` / `vars`
   —— 直接调这些会得到一条中文报错。

   ⚠️ **这不是沙箱。** 任何 Python 代码都能顺着 `().__class__.__mro__`、
   `[c for c in object.__subclasses__() ...]` 这类路径走回运行时（实测可以），
   `allow_imports=False` 只能挡住"顺手就写出来"的文件读写与动态执行，
   挡不住有意为之的代码。真正的隔离要单独进程 + 权限降级，本项目不做。
   所以"运行别人给的工作流"这件事，防线在**默认关闭 + 打开时二次确认**，
   而不是这道闸。

节点的变量约定
--------------
代码执行时的命名空间里有：

* `inputs`  —— dict，本节点收到的全部输入（连线或字面量）
* `value`   —— `value` 输入（ANY），没有就是 None
* `image`   —— `image` 输入（IMAGE），没有就是 None
* `ctx`     —— NodeContext（可用来 `ctx.log(...)`）
* `log`     —— 等价于 `ctx.log` 的简写

执行完读这些键：

* `result`（任意值，可缺省）→ 输出 `result`
* `image` （PIL 图像，可缺省）→ 输出 `image`；不覆盖时会原样透传输入图
* `text`  （字符串，可缺省）→ 输出 `text`；缺省时用标准输出
"""
from __future__ import annotations

import io
import logging
import threading

from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)

logger = logging.getLogger(__name__)

#: 关掉 import 时仍然放行的模块（纯计算，不碰文件系统/进程/网络）
SAFE_MODULES = frozenset({
    "math", "cmath", "random", "json", "re", "time", "datetime",
    "itertools", "functools", "collections", "statistics", "decimal",
    "fractions", "string", "textwrap", "typing", "dataclasses",
    "hashlib", "base64", "struct", "copy", "operator", "bisect", "heapq",
})

#: 关掉 import 时从 builtins 里摘掉的名字。
#: 只换 `__import__` 是假的：`open` 根本不在 import 名单里，`open("x","w")`
#: 照样能写文件（实测通过）。这几个是"顺手就能碰到文件/进程/动态执行"的入口。
BLOCKED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "input", "breakpoint",
    "globals", "locals", "vars",
})

#: 允许的默认超时（秒）
DEFAULT_TIMEOUT = 30.0

#: 节点 key，界面与文档引用它
NODE_TYPE = "PythonScript"


def python_nodes_allowed() -> bool:
    """读配置：用户是否显式打开了"任意 Python 节点"。"""
    try:
        from core.config_manager import AppConfig
        cfg = AppConfig.load()
        return bool(getattr(cfg, "allow_python_nodes", False))
    except Exception as e:            # 配置坏了就当没开 —— 安全侧
        logger.warning("读取 allow_python_nodes 失败（按关闭处理）: %s", e)
        return False


def _make_import_guard(allow_imports: bool):
    """返回一个受限制的 __import__（不限制就直接返回内置的）。"""
    if allow_imports:
        return __import__

    def _guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".")[0]
        if root not in SAFE_MODULES:
            raise ImportError(
                f"本节点已关闭 import：不允许导入 {name!r}"
                f"（可用：{', '.join(sorted(SAFE_MODULES))}）。"
                f"要导入任意模块请勾选参数「允许 import」")
        return __import__(name, globals, locals, fromlist, level)

    return _guarded


def _make_denied(name: str):
    """被摘掉的内置函数：调用时抛一条能照做的中文错误。"""
    def _denied(*args, **kwargs):
        raise NameError(
            f"本节点已禁用 {name}()：它能直接读写文件或动态执行代码。\n"
            f"   要使用它，请勾选节点参数「允许 import」——"
            f"那等于允许任意代码，请只在你信任的工作流里这么做。")
    _denied.__name__ = name
    return _denied


def _restricted_builtins() -> dict:
    """受限的 builtins：白名单 import + 摘掉危险内置。

    注意：**这不是沙箱**。`[c for c in object.__subclasses__()]` 之类的写法
    依然能走回运行时，这里只是去掉"顺手就能用"的入口，让误写在提交前失败，
    而不是让恶意代码跑不起来。
    """
    import builtins as _b

    safe = dict(vars(_b))
    for name in BLOCKED_BUILTINS:
        safe.pop(name, None)
    for name in BLOCKED_BUILTINS:
        safe[name] = _make_denied(name)
    safe["__import__"] = _make_import_guard(False)
    return safe


def _collector_print(buf):
    """把用户代码里的 print 收进 buf **而不动 sys.stdout**。

    为什么不用 contextlib.redirect_stdout：它是**全局**替换 sys.stdout，而这个
    节点的代码可能在超时后仍留在后台线程里 —— 那会把整个程序的 stdout 一直
    挂着，日志、界面输出全被吞掉（第一版就是这么翻车的）。所以改成查命名空间
    里的 print。
    """
    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is not None:
            return print(*args, sep=sep, end=end, file=file, flush=flush)
        buf.write(sep.join(str(a) for a in args) + end)
        return None
    return _print


def run_code(code: str, *, namespace: dict, allow_imports: bool,
             timeout: float = DEFAULT_TIMEOUT) -> tuple:
    """执行用户代码，返回 `(namespace, stdout, error)`。

    `timeout<=0` 表示不限时。超时只放弃等待，线程本身无法强杀 —— 调用方
    需要把这一点如实告诉用户。
    """
    if not allow_imports:
        namespace = dict(namespace)
        namespace["__builtins__"] = _restricted_builtins()
    else:
        namespace = dict(namespace)

    buf = io.StringIO()
    namespace["print"] = _collector_print(buf)
    box: dict = {"error": None}

    def _worker():
        try:
            exec(compile(code, "<workflow-python>", "exec"), namespace)
        except BaseException as e:      # noqa: BLE001 —— 用户代码什么都可能抛
            box["error"] = e

    th = threading.Thread(target=_worker, name="workflow-python", daemon=True)
    th.start()
    th.join(None if timeout is None or timeout <= 0 else float(timeout))
    if th.is_alive():
        return namespace, buf.getvalue(), TimeoutError(
            f"代码运行超过 {timeout:g} 秒，已放弃等待"
            f"（线程是 daemon，会继续在后台跑到自然结束）")
    return namespace, buf.getvalue(), box["error"]


@node(
    NODE_TYPE,
    category="高级",
    description="运行一段 Python 代码。默认关闭，需在「设置 → 推理性能」里"
                "打开「允许任意 Python 节点」；打开后工作流的可移植性由你负责。",
    inputs={
        "value": PortSpec(DataType.ANY, "任意值", optional=True),
        "image": PortSpec(DataType.IMAGE, "图像", optional=True),
    },
    outputs={
        "result": PortSpec(DataType.ANY, "结果"),
        "image": PortSpec(DataType.IMAGE, "图像"),
        "text": PortSpec(DataType.ANY, "文本"),
        "metadata": PortSpec(DataType.METADATA, "元数据", optional=True),
    },
    params={
        "code": ParamSpec("text", default="# 例：\n# result = len(inputs)\n"
                                          "# print('hello')\n",
                          label="代码", help="用 Python 写；执行后读 result / "
                                            "image / text 三个变量作为输出"),
        "timeout": ParamSpec("float", default=DEFAULT_TIMEOUT, label="超时(秒)",
                             minimum=0.0, maximum=3600.0, step=1.0,
                             help="0 = 不限时；超时只放弃等待，线程不会被强杀"),
        "allow_imports": ParamSpec("bool", default=False, label="允许 import",
                                   help="关闭时只放行 math/random/json 等"
                                        "纯计算模块"),
    },
    resources=ResourceRequirement(requires=(), estimated_vram_mb=0,
                                  supports_tiled=False),
    cacheable=False,
)
def run_python_script(ctx, node_obj, inputs):
    if not python_nodes_allowed():
        raise RuntimeError(
            "「任意 Python 节点」当前是关闭的。\n"
            "要使用它：设置 → 推理性能 → 勾选「允许任意 Python 节点」。\n"
            "（默认关闭是因为这类节点能执行任意代码，工作流文件也会因此"
            "变得不可移植。）")

    code = str(node_obj.params.get("code", "") or "")
    if not code.strip():
        raise ValueError("Python 节点没有代码，请在参数里填一段脚本")

    try:
        timeout = float(node_obj.params.get("timeout", DEFAULT_TIMEOUT) or 0.0)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    allow_imports = bool(node_obj.params.get("allow_imports", False))

    ctx.check_cancel()
    ctx.status("🐍 运行 Python 节点")

    ns: dict = {
        "__name__": "__workflow_python__",
        "__doc__": None,
        "inputs": dict(inputs),
        "value": inputs.get("value"),
        "image": inputs.get("image"),
        "ctx": ctx,
        "log": ctx.log,
        "result": None,
        "text": "",
    }
    ns, stdout, err = run_code(code, namespace=ns,
                               allow_imports=allow_imports, timeout=timeout)

    for line in (stdout or "").splitlines():
        if line.strip():
            ctx.log(f"🐍 {line}")
    if err is not None:
        name = type(err).__name__
        raise RuntimeError(f"Python 节点执行失败（{name}）: {err}") from err

    out_image = ns.get("image")
    if out_image is None:
        out_image = inputs.get("image")
    text = ns.get("text")
    if not text and stdout:
        text = stdout.strip()
    ctx.metadata.setdefault("python_nodes", []).append(
        {"node": node_obj.id, "chars": len(code)})
    return {
        "result": ns.get("result"),
        "image": out_image,
        "text": text or "",
        "metadata": {"python": {"chars": len(code), "stdout": (stdout or "")[:4000]}},
    }


__all__ = ["NODE_TYPE", "SAFE_MODULES", "DEFAULT_TIMEOUT",
           "python_nodes_allowed", "run_code", "run_python_script"]
