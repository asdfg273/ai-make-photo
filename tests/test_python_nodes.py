# tests/test_python_nodes.py
# ============================================================
#  「任意 Python 节点」的回归
# ------------------------------------------------------------
#  这个节点是全程序唯一能执行任意代码的地方，所以这里锁的是"闸门"而不是
#  功能好不好用：
#    1. 默认关闭时**一个字节的代码都不执行**；
#    2. 打开后能拿到 result / text / image，stdout 进日志；
#    3. 超时、import 限制都真的生效，且报错能看懂。
# ============================================================
import _bootstrap  # noqa: F401  （路径 + 配置隔离）

import builtins
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_bootstrap.isolate_config_globally()

from nodes import python_nodes as pn  # noqa: E402
from workflow.registry import get, load_builtin_nodes  # noqa: E402

load_builtin_nodes()

FAILS = []


def check(name):
    def deco(fn):
        try:
            fn()
        except Exception as e:
            import traceback
            FAILS.append((name, f"{e}"))
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")
        return fn
    return deco


class _Ctx:
    def __init__(self):
        self.logs = []
        self.statuses = []
        self.metadata = {}

    def log(self, msg):
        self.logs.append(str(msg))

    def status(self, msg):
        self.statuses.append(str(msg))

    def progress(self, a, b):
        pass

    def check_cancel(self):
        pass

    def cancelled(self):
        return False


def _node(**params):
    base = {"code": "result = 1", "timeout": 30.0, "allow_imports": False}
    base.update(params)
    return types.SimpleNamespace(id="py1", params=base)


def _enable(value: bool = True):
    from core.config_manager import AppConfig
    cfg = AppConfig.load()
    cfg.allow_python_nodes = value
    cfg.save()


@check("test_registered_and_not_cacheable")
def test_registered_and_not_cacheable():
    spec = get(pn.NODE_TYPE)
    assert spec.category == "高级", spec.category
    assert spec.cacheable is False, "能执行任意代码的节点不能进缓存"
    assert set(spec.params) == {"code", "timeout", "allow_imports"}, sorted(spec.params)
    assert "image" in spec.inputs and "result" in spec.outputs


@check("test_disabled_by_default_does_not_run_code")
def test_disabled_by_default_does_not_run_code():
    _enable(False)
    if hasattr(builtins, "_py_ran"):
        del builtins._py_ran
    ctx = _Ctx()
    node = _node(code="import builtins\nbuiltins._py_ran = True\nresult = 1")
    try:
        pn.run_python_script(ctx, node, {})
    except RuntimeError as e:
        assert "设置" in str(e) and "允许任意 Python 节点" in str(e), str(e)
    else:
        raise AssertionError("默认关闭时必须抛错")
    assert not hasattr(builtins, "_py_ran"), "关闭状态下竟然执行了用户代码！"


@check("test_enabled_runs_and_returns_result")
def test_enabled_runs_and_returns_result():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="result = sum(range(10))")
    out = pn.run_python_script(ctx, node, {})
    assert out["result"] == 45, out
    assert ctx.metadata["python_nodes"], ctx.metadata


@check("test_stdout_and_text")
def test_stdout_and_text():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="print('第一行')\nprint('second line')")
    out = pn.run_python_script(ctx, node, {})
    assert "第一行" in out["text"] and "second line" in out["text"], out["text"]
    assert any("第一行" in m for m in ctx.logs), ctx.logs


@check("test_inputs_are_visible_to_code")
def test_inputs_are_visible_to_code():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="result = value * 3\nimage = None")
    out = pn.run_python_script(ctx, node, {"value": 7})
    assert out["result"] == 21, out


@check("test_image_passes_through_untouched")
def test_image_passes_through_untouched():
    from PIL import Image

    _enable(True)
    ctx = _Ctx()
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    node = _node(code="result = image.size")
    out = pn.run_python_script(ctx, node, {"image": img})
    assert out["image"] is img, "没改 image 时应原样透传"
    assert out["result"] == (8, 8)


@check("test_import_guard_blocks_os")
def test_import_guard_blocks_os():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="import os\nresult = os.getcwd()", allow_imports=False)
    try:
        pn.run_python_script(ctx, node, {})
    except RuntimeError as e:
        assert "不允许导入" in str(e) and "允许 import" in str(e), str(e)
    else:
        raise AssertionError("关掉 import 时导入 os 应被拦下")


@check("test_safe_modules_still_importable")
def test_safe_modules_still_importable():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="import math, json\nresult = [math.floor(2.7), json.dumps([1])]",
                 allow_imports=False)
    out = pn.run_python_script(ctx, node, {})
    assert out["result"] == [2, "[1]"], out


@check("test_timeout_is_reported")
def test_timeout_is_reported():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="import time\nwhile True:\n    time.sleep(0.02)",
                 allow_imports=True, timeout=0.3)
    try:
        pn.run_python_script(ctx, node, {})
    except RuntimeError as e:
        msg = str(e)
        assert "超时" in msg or "超过" in msg, msg
        assert "daemon" in msg, "必须如实说明线程没有被强杀"
    else:
        raise AssertionError("超时应该抛错")


@check("test_syntax_error_is_readable")
def test_syntax_error_is_readable():
    _enable(True)
    ctx = _Ctx()
    node = _node(code="result = = 1")
    try:
        pn.run_python_script(ctx, node, {})
    except RuntimeError as e:
        assert "Python 节点执行失败" in str(e), str(e)
        assert "SyntaxError" in str(e), str(e)
    else:
        raise AssertionError("语法错误应该抛错")


@check("test_settings_dialog_exposes_toggle")
def test_settings_dialog_exposes_toggle():
    import io as _io
    text = _io.open(ROOT / "ui" / "settings_dialog.py", encoding="utf-8").read()
    assert "chk_allow_python" in text
    assert "cfg.allow_python_nodes" in text
    assert "QMessageBox.warning" in text, "从关到开必须再确认一次"



# ── 第三道闸：这些内置必须真的被摘掉（用户实测 open 能写文件）──
@check("dangerous_builtins_are_blocked")
def test_dangerous_builtins_are_blocked():
    import os as _os
    import tempfile

    from nodes import python_nodes as pn

    target = _os.path.join(tempfile.gettempdir(), "_pn_blocked_probe.txt")
    if _os.path.exists(target):
        _os.remove(target)
    ns, _out, err = pn.run_code(
        f"open({target!r}, 'w').write('x')\nresult = 'written'\n",
        namespace={}, allow_imports=False, timeout=5)
    assert isinstance(err, NameError), err
    assert "open" in str(err) and "允许 import" in str(err)
    assert not _os.path.exists(target), "allow_imports=False 仍然写出了文件"

    for name, code in (("exec", "exec('x=1')\nresult=x\n"),
                       ("eval", "result = eval('1+1')\n"),
                       ("compile", "result = compile('1', '<s>', 'eval')\n")):
        _ns, _o, e = pn.run_code(code, namespace={}, allow_imports=False, timeout=5)
        assert isinstance(e, NameError), (name, e)
        assert name in str(e), (name, str(e))


@check("blocked_builtins_are_listed")
def test_blocked_builtins_are_listed():
    from nodes import python_nodes as pn

    for name in ("open", "exec", "eval", "compile", "input", "breakpoint"):
        assert name in pn.BLOCKED_BUILTINS, name
    safe = pn._restricted_builtins()
    for name in pn.BLOCKED_BUILTINS:
        assert callable(safe[name]), name
    assert safe["len"] is len, "正常的 builtins 不该被动到"


@check("allow_imports_true_keeps_full_builtins")
def test_allow_imports_true_keeps_full_builtins():
    from nodes import python_nodes as pn

    ns, _o, e = pn.run_code("result = eval('2*3')\nimport math\nresult += int(math.pi)\n",
                            namespace={}, allow_imports=True, timeout=5)
    assert e is None, e
    assert ns.get("result") == 9


@check("escape_via_subclasses_is_documented_not_claimed")
def test_escape_via_subclasses_is_documented_not_claimed():
    """`allow_imports=False` 不是沙箱：子类遍历仍然能走回运行时。

    这个用例刻意**断言它仍然可行** —— 免得有人以为那道闸等于隔离，
    顺手把设置里的"会执行任意代码"警告删掉。
    """
    import io as _io

    from nodes import python_nodes as pn

    ns, _o, e = pn.run_code(
        "subs = [c for c in ().__class__.__base__.__subclasses__()]\n"
        "result = len(subs)\n",
        namespace={}, allow_imports=False, timeout=5)
    assert e is None, e
    assert (ns.get("result") or 0) > 0

    node_src = _io.open(ROOT / "nodes" / "python_nodes.py", encoding="utf-8").read()
    assert "这不是沙箱" in node_src
    assert "BLOCKED_BUILTINS" in node_src
    dlg = _io.open(ROOT / "ui" / "settings_dialog.py", encoding="utf-8").read()
    assert "会执行任意代码" in dlg
    assert "不是沙箱" in dlg


print("=" * 60)
if FAILS:
    print(f"总计：FAIL {len(FAILS)}")
    for name, err in FAILS:
        print(f"  - {name}: {err}")
    sys.exit(1)
print("全部通过")
sys.exit(0)
