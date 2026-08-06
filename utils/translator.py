# utils/translator.py
"""
🌐 中日双向翻译引擎 (NLLB-200)
- facebook/nllb-200-distilled-600M
- 支持 200 种语言,懒加载
"""
import os
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    """基于 NLLB-200 的多语言翻译"""

    MODEL_ID = "facebook/nllb-200-distilled-600M"

    # NLLB 语言代码
    LANG_CODES = {
        "zh": "zho_Hans",   # 简体中文
        "ja": "jpn_Jpan",   # 日语
        "en": "eng_Latn",   # 英语
        "ko": "kor_Hang",   # 韩语
    }

    def __init__(self, device: str = None, cache_dir: str = "models_cache"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        os.makedirs(cache_dir, exist_ok=True)

    def _ensure_loaded(self):
        """懒加载模型"""
        if self._model is not None:
            return

        print(f"📥 加载翻译模型: {self.MODEL_ID}")
        t0 = time.time()

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, cache_dir=self.cache_dir
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"翻译模型下载失败: {e}\n"
                               f"请检查网络或手动下载 {self.MODEL_ID}")

        print(f"✅ 翻译模型就绪 ({time.time()-t0:.1f}s)")

    def translate(self, text: str, src_lang: str = "zh", tgt_lang: str = "ja") -> str:
        """
        翻译文本
        :param text: 原文
        :param src_lang: 源语言 ("zh"/"ja"/"en"/"ko")
        :param tgt_lang: 目标语言
        """
        if not text or not text.strip():
            return ""

        if src_lang not in self.LANG_CODES:
            raise ValueError(f"不支持的源语言: {src_lang}")
        if tgt_lang not in self.LANG_CODES:
            raise ValueError(f"不支持的目标语言: {tgt_lang}")

        self._ensure_loaded()

        src_code = self.LANG_CODES[src_lang]
        tgt_code = self.LANG_CODES[tgt_lang]

        # NLLB 需要设置源语言
        self._tokenizer.src_lang = src_code

        t0 = time.time()
        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt",
                                     truncation=True, max_length=512).to(self.device)

            # 强制目标语言的 token id
            forced_bos = self._tokenizer.convert_tokens_to_ids(tgt_code)

            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
            result = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        elapsed = time.time() - t0
        print(f"🌐 [{src_lang}→{tgt_lang}] {elapsed:.2f}s: {text[:30]}... → {result[:30]}...")
        return result

    def zh2ja(self, text: str) -> str:
        return self.translate(text, "zh", "ja")

    def ja2zh(self, text: str) -> str:
        return self.translate(text, "ja", "zh")

    def unload(self):
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_global_translator = None


def get_translator() -> Translator:
    global _global_translator
    if _global_translator is None:
        _global_translator = Translator()
    return _global_translator