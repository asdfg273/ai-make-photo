# scripts/download_ref_voice.py
"""
用 edge-tts 生成日语女声参考音频(GPT-SoVITS 零样本克隆用)
- edge-tts: 微软 Azure 免费在线 TTS,无需 API key
- 音色: ja-JP-NanamiNeural (日语女声,清晰自然)
- 输出: assets/voices/default_female_ja.wav + .txt
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_WAV = ROOT / "assets" / "voices" / "default_female_ja.wav"
DEST_TXT = ROOT / "assets" / "voices" / "default_female_ja.txt"

# 参考文本(5-10 秒,GPT-SoVITS 推荐长度)
REF_TEXT = "こんにちは、今日はいい天気ですね。散歩に行きましょう。"
VOICE = "ja-JP-NanamiNeural"  # 日语女声


async def synth():
    try:
        import edge_tts
    except ImportError:
        print("❌ 缺少 edge-tts,请先安装:")
        print("   pip install edge-tts")
        sys.exit(1)

    DEST_WAV.parent.mkdir(parents=True, exist_ok=True)

    # 先输出 mp3,再转 wav
    tmp_mp3 = DEST_WAV.with_suffix(".mp3")

    print(f"🎙️ 合成中... (voice={VOICE})")
    print(f"   文本: {REF_TEXT}")

    communicate = edge_tts.Communicate(REF_TEXT, VOICE)
    await communicate.save(str(tmp_mp3))

    # mp3 → wav (GPT-SoVITS 要 wav)
    print(f"🔄 转 wav...")
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    subprocess.run([
        ffmpeg_exe, "-y", "-i", str(tmp_mp3),
        "-ar", "32000", "-ac", "1",
        str(DEST_WAV)
    ], check=True, capture_output=True)
    tmp_mp3.unlink()

    DEST_TXT.write_text(REF_TEXT, encoding="utf-8")

    size_kb = DEST_WAV.stat().st_size / 1024
    print()
    print("=" * 60)
    print(f"✅ 完成!")
    print(f"   音频: {DEST_WAV} ({size_kb:.1f} KB)")
    print(f"   文本: {DEST_TXT}")
    print("=" * 60)


def main():
    if DEST_WAV.exists() and DEST_TXT.exists():
        print(f"✅ 参考音频已存在: {DEST_WAV}")
        print(f"   如需重新生成,请先删除该文件")
        return

    asyncio.run(synth())


if __name__ == "__main__":
    main()