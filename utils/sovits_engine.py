class SoVITSEngine:
    def synthesize(
        text: str,              # 要合成的文本
        ref_audio: str,         # 参考音频路径(3-10秒)
        ref_text: str,          # 参考音频的文字内容
        language: str = "ja",   # 语言:ja/zh/en
        output_path: str = None
    ) -> str: 