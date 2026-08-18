# ============ LangSegment 兼容补丁 ============
import sys
class _LangSegPatch:
    def setLangfilters(self, *a, **k): pass
    def getLangfilters(self): return []
    def getTexts(self, text): return [{"text": text, "lang": "zh"}]
    def printList(self, *a, **k): pass

try:
    import LangSegment
    for name in ["setLangfilters", "getLangfilters", "getTexts", "printList"]:
        if not hasattr(LangSegment, name):
            setattr(LangSegment, name, getattr(_LangSegPatch(), name))
except ImportError:
    sys.modules['LangSegment'] = _LangSegPatch()
# ==========================================

# test_sovits_import.py
"""验证 GPT-SoVITS 源码和依赖是否就绪"""
import sys
import os
from pathlib import Path

# 把 GPT-SoVITS 源码加入 PYTHONPATH
ROOT = Path(__file__).parent
SOVITS_ROOT = ROOT / "third_party" / "GPT-SoVITS"
sys.path.insert(0, str(SOVITS_ROOT))
sys.path.insert(0, str(SOVITS_ROOT / "GPT_SoVITS"))

os.chdir(SOVITS_ROOT)  # GPT-SoVITS 假设从自己的根目录运行

print("=" * 60)
print("🧪 测试 1: 基础依赖")
print("=" * 60)

tests = [
    ("pyopenjtalk (日语音素)", lambda: __import__("pyopenjtalk")),
    ("jieba_fast (中文分词)", lambda: __import__("jieba_fast")),
    ("LangSegment (语言分段)", lambda: __import__("LangSegment")),
    ("cn2an (中文数字)", lambda: __import__("cn2an")),
    ("pypinyin (拼音)", lambda: __import__("pypinyin")),
]

for name, fn in tests:
    try:
        fn()
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print()
print("=" * 60)
print("🧪 测试 2: GPT-SoVITS 核心模块")
print("=" * 60)

sovits_tests = [
    ("AR 模型", "AR.models.t2s_lightning_module"),
    ("SoVITS 模型", "module.models"),
    ("中文文本处理", "text.chinese"),
    ("日语文本处理", "text.japanese"),
    ("符号表", "text.symbols"),
]

for name, module in sovits_tests:
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:80]}")

print()
print("=" * 60)
print("🧪 测试 3: 权重文件")
print("=" * 60)

weights = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin",
]

for w in weights:
    p = SOVITS_ROOT / w
    if p.exists():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  ✅ {p.name} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ 缺失: {p.name}")

print()
print("✅ 验证完成")