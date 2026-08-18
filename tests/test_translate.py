from utils.prompt_enhancer import get_enhancer

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