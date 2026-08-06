# utils/chattts_patch.py
"""
🔧 ChatTTS 兼容性补丁
适配 transformers >= 4.45 (DynamicCache API 变化)
"""
from transformers.cache_utils import DynamicCache


def apply_chattts_patch():
    """在 import ChatTTS 之前调用"""
    # ChatTTS 访问 cache.layers,新版没有该属性
    # 用 key_cache 长度模拟旧 API
    if not hasattr(DynamicCache, "layers"):
        def _get_layers(self):
            # 返回一个假的 layers 列表,长度 = key_cache 长度
            return [None] * len(self.key_cache) if hasattr(self, "key_cache") else []
        DynamicCache.layers = property(_get_layers)
        print("🔧 ChatTTS DynamicCache 补丁已应用")