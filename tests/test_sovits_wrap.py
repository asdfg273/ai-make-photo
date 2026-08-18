from utils.sovits_tts import synth_once

# 测试 1: 用默认女声合成日语
synth_once(
    text="今日はとても楽しかったです。また明日も頑張りましょう!",
    output_path="output/test_wrap_default.wav",
    language="ja",
)

# 测试 2: 慢速合成
synth_once(
    text="ゆっくり話します。聞き取れますか?",
    output_path="output/test_wrap_slow.wav",
    language="ja",
    speed=0.85,
)

print("\n✅ 全部完成")
from utils.sovits_tts import release_tts
release_tts() 