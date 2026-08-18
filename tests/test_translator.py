# test_translator.py
from utils.translator import get_translator

tr = get_translator()

# 中→日
zh_cases = [
    "你好,今天天气真好",
    "我喜欢吃苹果",
    "这是一只可爱的小猫",
    "我是一名程序员,喜欢写代码",
    "赛博朋克风格的城市夜景,霓虹灯闪烁",
]

print("\n=== 中 → 日 ===")
for text in zh_cases:
    result = tr.zh2ja(text)
    print(f"[原] {text}")
    print(f"[译] {result}")
    print("-" * 50)

# 日→中
ja_cases = [
    "こんにちは、元気ですか",
    "今日はいい天気ですね",
    "アニメが大好きです",
]

print("\n=== 日 → 中 ===")
for text in ja_cases:
    result = tr.ja2zh(text)
    print(f"[原] {text}")
    print(f"[译] {result}")
    print("-" * 50)