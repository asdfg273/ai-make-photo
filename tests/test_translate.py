import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码 + 把项目根加进 sys.path）

# ⚠️ 原先这个文件第一行就直接 `from utils...`，而直接跑
# `python tests/test_translate.py` 时 sys.path 上是 `tests/` 而不是项目根，
# 于是立刻 `ModuleNotFoundError: No module named 'utils'` ——
# 看起来像"翻译功能坏了"，其实只是测试自己缺了引导。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompt_enhancer import get_enhancer  # noqa: E402

enhancer = get_enhancer()

# 测试用例
cases = [
    ("你好,今天天气真好", "ja"),
    ("我喜欢吃苹果", "ja"),
    ("こんにちは、元気ですか", "zh"),
    ("这是一只可爱的小猫", "ja"),
    ("我是一名程序员,喜欢写代码", "ja"),
]

for text, lang in cases:
    result = enhancer.translate(text, target_lang=lang)
    print(f"\n[{lang}] {text}\n  → {result}\n" + "-" * 50)
