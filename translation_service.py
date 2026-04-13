# translation_service.py
import os
import json
import jieba
from deep_translator import GoogleTranslator

class TranslationService:
    def __init__(self, dict_file="zh_to_en_dict.json"):
        self.dict_file = dict_file
        self.dictionary = {
            "高质量": "best quality", "大师作": "masterpiece", "细节": "highly detailed",
            "女孩": "1girl", "男孩": "1boy", "单人": "solo",
            "低画质": "low quality", "最差画质": "worst quality", "坏手": "bad hands"
        }
        self.load_dictionary()

    def load_dictionary(self):
        if os.path.exists(self.dict_file):
            try:
                with open(self.dict_file, 'r', encoding='utf-8') as f:
                    user_dict = json.load(f)
                    self.dictionary.update(user_dict)
            except Exception as e: print("词典加载失败:", e)
        for word in self.dictionary.keys():
            jieba.add_word(word)

    def save_dictionary(self):
        try:
            with open(self.dict_file, 'w', encoding='utf-8') as f:
                json.dump(self.dictionary, f, ensure_ascii=False, indent=4)
        except: pass

    def translate(self, text):
        if not text.strip(): return ""
        has_new_word = False
        text = text.replace('，', ',').replace('。', ',').replace('\n', ',')
        segments = [seg.strip() for seg in text.split(',') if seg.strip()]
        final_parts = []
        
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
                if not word or word in ["在", "的", "了", "着", "地", "个", "是"]: continue 
                if word in self.dictionary:
                    seg_trans_words.append(self.dictionary[word])
                else:
                    try:
                        res = GoogleTranslator(source='zh-CN', target='en').translate(word)
                        if res:
                            seg_trans_words.append(res)
                            self.dictionary[word] = res
                            jieba.add_word(word) 
                            has_new_word = True
                        else: seg_trans_words.append(word)
                    except: seg_trans_words.append(word)
            if seg_trans_words: final_parts.append(" ".join(seg_trans_words))
            
        if has_new_word: self.save_dictionary()
        return ", ".join(final_parts)