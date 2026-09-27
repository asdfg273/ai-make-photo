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
