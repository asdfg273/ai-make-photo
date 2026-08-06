# utils/sovits_tts.py
"""
GPT-SoVITS 日语 TTS 封装(常驻显存版)
- 单例:首次调用加载,之后复用
- 支持自定义参考音频(零样本克隆)
- 显式 unload() 才释放
"""

import os
import sys
import gc
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ========== 路径常量 ==========
PROJECT_ROOT = Path(__file__).parent.parent
SOVITS_ROOT = PROJECT_ROOT / "third_party" / "GPT-SoVITS"
SOVITS_INNER = SOVITS_ROOT / "GPT_SoVITS"
PRETRAINED = SOVITS_INNER / "pretrained_models"

DEFAULT_GPT = PRETRAINED / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
DEFAULT_SOVITS = PRETRAINED / "gsv-v2final-pretrained" / "s2G2333k.pth"

DEFAULT_REF_AUDIO = PROJECT_ROOT / "assets" / "voices" / "default_female_ja.wav"
DEFAULT_REF_TEXT = "こんにちは、今日はいい天気ですね。散歩に行きましょう。"


def _inject_paths():
    for p in [str(SOVITS_ROOT), str(SOVITS_INNER), str(SOVITS_INNER / "eres2net")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _patch_torchaudio():
    import torchaudio
    if getattr(torchaudio.load, "_patched_sf", False):
        return

    def _load_sf(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr

    _load_sf._patched_sf = True
    torchaudio.load = _load_sf


def _ensure_nltk():
    """确保 GPT-SoVITS 需要的 nltk 数据已安装（英文 TTS 用）"""
    try:
        import nltk
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        return  # 已安装
    except LookupError:
        pass

    logger.info("📥 下载 nltk averaged_perceptron_tagger_eng ...")
    try:
        import nltk
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        logger.info("✅ nltk 资源下载成功")
    except Exception as e:
        logger.warning(
            "⚠️ nltk 资源下载失败(英文配音将跳过): %s。"
            "手动安装: python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"",
            e)


class SovitsTTS:
    def __init__(self, gpt_path=None, sovits_path=None, device="cuda"):
        self.gpt_path = str(gpt_path or DEFAULT_GPT)
        self.sovits_path = str(sovits_path or DEFAULT_SOVITS)
        self.device = device if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._orig_cwd = None

    def _load(self):
        if self._loaded:
            return
        _inject_paths()
        _patch_torchaudio()
        _ensure_nltk()
        self._orig_cwd = os.getcwd()
        os.chdir(str(SOVITS_ROOT))

        import GPT_SoVITS.inference_webui as iw
        for attr in ("sovits_path", "gpt_path"):
            if hasattr(iw, attr):
                setattr(iw, attr, "")
        iw.change_gpt_weights(self.gpt_path)
        iw.change_sovits_weights(self.sovits_path)
        self._loaded = True
        logger.info(f"✅ GPT-SoVITS 已加载 (device={self.device})")

    def synth(self, text, output_path, ref_audio=None, ref_text=None,
              language="ja", speed=1.0):
        self._load()
        # 保存 CWD（GPT-SoVITS 加载会 os.chdir 到其根目录, 用 finally 恢复）
        saved_cwd = os.getcwd()
        try:
            ref_wav = ref_audio or str(DEFAULT_REF_AUDIO)
            ref_txt = ref_text or DEFAULT_REF_TEXT
            if not Path(ref_wav).exists():
                raise FileNotFoundError(f"参考音频不存在: {ref_wav}")

            lang_map = {"ja": "日文", "jp": "日文", "zh": "中文", "en": "英文"}
            prompt_lang = "日文"
            text_lang = lang_map.get(language, "日文")

            # 如果是英文文本，确保 nltk 资源可用（GPT-SoVITS 内部分句需要）
            if text_lang == "英文":
                try:
                    import nltk
                    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
                except (LookupError, ImportError):
                    logger.warning("⚠️ nltk 英文分句资源缺失,强制切为 不切 模式")

            from GPT_SoVITS.inference_webui import get_tts_wav
            t0 = time.time()
            gen = get_tts_wav(
                ref_wav_path=ref_wav, prompt_text=ref_txt,
                prompt_language=prompt_lang, text=text, text_language=text_lang,
                how_to_cut="不切", top_k=15, top_p=1.0, temperature=1.0,
                ref_free=False, speed=speed, if_freeze=False, inp_refs=None,
            )
            sr, audio = next(gen)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), audio, sr)
            logger.info(f"✅ 合成: {out.name} ({len(audio)/sr:.2f}s, 用时{time.time()-t0:.1f}s)")
            return str(out)
        finally:
            os.chdir(saved_cwd)

    def unload(self):
        if not self._loaded:
            return
        try:
            if self._orig_cwd:
                os.chdir(self._orig_cwd)
        except Exception:
            pass
        try:
            import GPT_SoVITS.inference_webui as iw
            for attr in ("t2s_model", "vq_model", "bert_model", "ssl_model", "hps"):
                if hasattr(iw, attr):
                    setattr(iw, attr, None)
        except Exception as e:
            logger.warning(f"卸载失败: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("🗑️ GPT-SoVITS 已释放")


# ========== 全局单例 ==========
_global_tts: Optional[SovitsTTS] = None


def get_tts() -> SovitsTTS:
    """获取全局单例(常驻显存)"""
    global _global_tts
    if _global_tts is None:
        _global_tts = SovitsTTS()
    return _global_tts


def release_tts():
    """释放全局单例"""
    global _global_tts
    if _global_tts is not None:
        _global_tts.unload()
        _global_tts = None


def synth_once(text, output_path, ref_audio=None, ref_text=None,
               language="ja", speed=1.0) -> str:
    """便捷入口:走单例,不释放(常驻)"""
    return get_tts().synth(
        text=text, output_path=output_path,
        ref_audio=ref_audio, ref_text=ref_text,
        language=language, speed=speed,
    )