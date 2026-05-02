import os
import json
import jieba
import threading
from functools import lru_cache
from deep_translator import GoogleTranslator

class TranslationService:
    def __init__(self, dict_file="zh_to_en_dict.json"):
        self.dict_file = dict_file
        self.lock = threading.Lock()
        self._translator = None
        self._translator_lock = threading.Lock()
        self.dictionary = {
            "高质量": "best quality", "大师作": "masterpiece", "细节": "highly detailed",
            "女孩": "1girl", "男孩": "1boy", "单人": "solo",
            "低画质": "low quality", "最差画质": "worst quality", "坏手": "bad hands"
        }
        self.load_dictionary()
        self._cache = {}

    @property
    def translator(self):
        if self._translator is None:
            with self._translator_lock:
                if self._translator is None:
                    self._translator = GoogleTranslator(source='zh-CN', target='en')
        return self._translator

    def load_dictionary(self):
        if os.path.exists(self.dict_file):
            try:
                with open(self.dict_file, 'r', encoding='utf-8') as f:
                    user_dict = json.load(f)
                    self.dictionary.update(user_dict)
            except Exception as e:
                print(f"词典加载失败: {e}")
        for word in self.dictionary.keys():
            jieba.add_word(word)

    def save_dictionary(self):
        with self.lock:
            try:
                with open(self.dict_file, 'w', encoding='utf-8') as f:
                    json.dump(self.dictionary, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"词典保存失败: {e}")

    @lru_cache(maxsize=1000)
    def _cached_translate(self, text):
        try:
            return self.translator.translate(text)
        except Exception:
            return None

    def translate(self, text):
        if not text.strip():
            return ""
        
        cache_key = text.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        has_new_word = False
        text = text.replace('，', ',').replace('。', ',').replace('\n', ',')
        segments = [seg.strip() for seg in text.split(',') if seg.strip()]
        final_parts = []
        
        stop_words = {"在", "的", "了", "着", "地", "个", "是"}
        
        for seg in segments:
            if all(ord(c) < 128 for c in seg): 
                final_parts.append(seg)
                continue
            if seg in self.dictionary: 
                final_parts.append(self.dictionary[seg])
                continue
                
            words = jieba.lcut(seg)
            seg_trans_words = []
            for word in words:
                word = word.strip()
                if not word or word in stop_words:
                    continue 
                if word in self.dictionary:
                    seg_trans_words.append(self.dictionary[word])
                else:
                    cached = self._cached_translate(word)
                    if cached:
                        seg_trans_words.append(cached)
                        self.dictionary[word] = cached
                        jieba.add_word(word) 
                        has_new_word = True
                    else:
                        seg_trans_words.append(word)
            if seg_trans_words:
                final_parts.append(" ".join(seg_trans_words))
            
        result = ", ".join(final_parts)
        self._cache[cache_key] = result
        if has_new_word:
            self.save_dictionary()
        return result