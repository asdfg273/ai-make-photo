# tests/_bootstrap.py
"""测试脚本公共引导：让 print 不再因控制台编码中断。

背景
----
中文 Windows 上，`python tests/test_x.py` 的 stdout 一旦被**重定向**
（管道、日志、CI），编码就变成 GBK。此时脚本末尾的
`print("✅ 全部测试通过")` 会抛 UnicodeEncodeError，进程以 1 退出 ——
看起来像"测试失败"，其实所有断言都过了。

这里统一把 errors 放宽为 replace：中文在 GBK 下照常显示，
emoji 降级成 '?'，断言结果不再被编码问题掩盖。

用法（放在其他 import 之前或之后均可，越早越好）：
    import _bootstrap  # noqa: F401
"""
import contextlib
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")
    except Exception:
        pass

# 顺带保证项目根可导入（个别脚本漏了 sys.path 设置）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@contextlib.contextmanager
def isolated_config():
    """把 `AppConfig.config_file` 指到测试自己的文件（with 用法）。

    ⚠️ 测试里**必须**隔离任何 `AIDesktopApp()` 的创建与 `close()`：
    `closeEvent()` 会把配置写回 `AppConfig.config_file`，不隔离就会
    **覆盖用户真实的 app_config.json**（踩过：测试用的区域被写进用户配置，
    `use_hires` 也被从 True 改成 False）。

    用法：
        from _bootstrap import isolated_config
        with isolated_config():
            win = main.AIDesktopApp()
        ...
        with isolated_config():
            win.close()
    """
    from core.config_manager import AppConfig
    orig = AppConfig.config_file
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_tmp_test_config.json")
    AppConfig.config_file = tmp
    try:
        yield tmp
    finally:
        AppConfig.config_file = orig
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def isolate_config_globally():
    """整进程隔离配置：紧跟 `import _bootstrap` 调一次即可。

    给不便改写成 `with` 结构的老脚本用（它们在模块级建窗口、在 finally 里
    close，用上下文管理器要改一大片缩进）。写入的仍是 tests 下的临时文件，
    进程退出时删除。
    """
    import atexit
    from core.config_manager import AppConfig

    if getattr(AppConfig, "_bootstrap_isolated", False):
        return AppConfig.config_file

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_tmp_test_config_global.json")
    AppConfig.config_file = tmp
    AppConfig._bootstrap_isolated = True

    def _cleanup():
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass

    atexit.register(_cleanup)
    return tmp
