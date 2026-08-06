# utils/tts_engine.py
"""
🎙️ TTS 引擎 — 统一封装
Phase 1: ChatTTS (中文)
Phase 2: GPT-SoVITS (中日 + 克隆) [预留接口]
"""
from utils.chattts_patch import apply_chattts_patch
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_OUT_DIR = PROJECT_ROOT / "photo" / "audio"
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

def _preprocess_text(text: str) -> str:
    import re
    digit_map = str.maketrans("0123456789", "零一二三四五六七八九")
    text = text.translate(digit_map)

    # 常见符号 → 中文
    text = text.replace("%", "百分之").replace("+", "加").replace("-", "减")

    # 如果预处理后仍无中文字符,补个引导词避免空音频
    if not re.search(r"[\u4e00-\u9fff]", text):
        text = "内容:" + text

    return text

class TTSEngine:
    """统一 TTS 引擎,懒加载"""

    def __init__(self):
        self._chattts = None
        self._sovits = None  # Phase 2

    # ============================================================
    #  ChatTTS
    # ============================================================

    def _ensure_chattts(self):
        if self._chattts is not None:
            return

        model_dir = PROJECT_ROOT / "models" / "tts" / "ChatTTS"
        if not (model_dir / "asset" / "GPT.pt").is_file():
            print("📥 ChatTTS 缺失,开始自动下载...")
            from utils.model_downloader import install
            install("chattts")

        if not (model_dir / "asset" / "GPT.pt").is_file():
            raise RuntimeError("❌ ChatTTS 模型下载失败")

        print("🎙️ 加载 ChatTTS...")
        import ChatTTS
        chat = ChatTTS.Chat()
        chat.load(source="custom", custom_path=str(model_dir), compile=False)
        self._chattts = chat
        print("✅ ChatTTS 就绪")

    def generate_chattts(
        self,
        text: str,
        seed: int = 42,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        speaker_seed: int = None,
    ) -> str:
        """
        用 ChatTTS 生成语音
        Returns: 保存的 wav 文件路径
        """
        text = _preprocess_text(text)
        self._ensure_chattts()

        # 固定说话人音色
        torch.manual_seed(speaker_seed if speaker_seed is not None else seed)
        rand_spk = self._chattts.sample_random_speaker()

        params_infer = self._chattts.InferCodeParams(
            spk_emb=rand_spk,
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
        )

        print(f"🎙️ [ChatTTS] 合成: {text[:40]}...")
        wavs = self._chattts.infer(
            [text],
            params_infer_code=params_infer,
        )

        # 保存
        import soundfile as sf
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = AUDIO_OUT_DIR / f"tts_{ts}.wav"
        audio = wavs[0]
        # ChatTTS 输出可能是 [1, N] 或 [N]
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        sf.write(str(out_path), audio, 24000)
        print(f"💾 已保存: {out_path}")
        return str(out_path)

    # ============================================================
    #  GPT-SoVITS (Phase 2 预留)
    # ============================================================
    def generate_sovits(self, text: str, ref_audio: str, ref_text: str, lang: str = "zh"):
        raise NotImplementedError("Phase 2 会实现")

    # ============================================================
    #  统一入口
    # ============================================================
    def generate(self, text: str, engine: str = "chattts", **kwargs) -> str:
        if engine == "chattts":
            return self.generate_chattts(text, **kwargs)
        elif engine == "sovits":
            return self.generate_sovits(text, **kwargs)
        else:
            raise ValueError(f"未知 TTS 引擎: {engine}")

    def unload(self):
        """释放显存"""
        self._chattts = None
        self._sovits = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
