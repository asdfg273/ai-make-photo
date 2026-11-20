# test_tts.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码 + 把项目根加进 sys.path）

# ⚠️ 直接跑 `python tests/test_tts.py` 时 sys.path 上是 `tests/` 而不是项目根，
# 缺了 _bootstrap 就会立刻 `ModuleNotFoundError: No module named 'utils'`。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from utils.tts_engine import TTSEngine  # noqa: E402

tts = TTSEngine()
path = tts.generate_chattts(
    "你好,我是 AI 绘画工作站的配音助手,很高兴为你服务。",
    seed=42,
)
print(f"生成完成: {path}")
