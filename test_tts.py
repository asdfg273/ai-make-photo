# test_tts.py
from utils.tts_engine import TTSEngine

tts = TTSEngine()
path = tts.generate_chattts(
    "你好,我是 AI 绘画工作站的配音助手,很高兴为你服务。",
    seed=42,
)
print(f"生成完成: {path}")