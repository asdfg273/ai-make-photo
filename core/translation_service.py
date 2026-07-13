"""
🌐 中译英服务
优先级：缓存 → Qwen2-VL（智能整段翻译）→ 本地词典（逐词翻译）→ Google（兜底）
"""
import os
import json
import re
import threading
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
DICT_DIR = os.path.join("data", "dictionaries")
# 可选依赖 —— 没装也能跑
try:
    import jieba
    jieba.setLogLevel(logging.WARNING)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logger.warning("⚠️ jieba 未安装，中文分词降级到单字符模式")

try:
    from deep_translator import GoogleTranslator
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False


class TranslationService:
    """中译英翻译服务（Qwen 优先 + 词典兜底）"""

    def __init__(self, dict_file="zh_to_en_dict.json", qwen_enhancer=None):
        self.dict_file = dict_file
        self.lock = threading.Lock()
        self._cache = {}            # 整段翻译缓存
        self._dictionary = {}       # 词级翻译字典
        self.qwen_enhancer = qwen_enhancer   # 🆕 Qwen 实例（可选注入）

        # Google 翻译器（兜底，被墙也不会崩）
        self.google_translator = None
        if HAS_GOOGLE:
            try:
                self.google_translator = GoogleTranslator(source="zh-CN", target="en")
                logger.info("✅ Google 翻译器已初始化（兜底用）")
            except Exception as e:
                logger.warning(f"⚠️ Google 翻译初始化失败: {e}")
                logger.warning("→ 将仅使用本地词典翻译")

        self.load_dictionary()

    # ========== 字典持久化 ==========

    def load_dictionary(self):
        """
        加载多个 JSON 词典文件,合并到 self._dictionary
        优先级(后加载覆盖前面): general < style < outfit < nsfw < user_custom
        """
        self._dictionary = {}
    
        # 1️⃣ 先加载旧的单文件字典(兼容)
        if os.path.exists(self.dict_file):
            try:
                with open(self.dict_file, "r", encoding="utf-8") as f:
                    self._dictionary.update(json.load(f))
            except Exception as e:
                logger.warning(f"⚠️ 旧词典加载失败: {e}")
    
        # 2️⃣ 加载 data/dictionaries/ 下所有 JSON
        if os.path.isdir(DICT_DIR):
            # 定义加载顺序(用户自定义最后加载,优先级最高)
            priority_order = ["general", "style", "outfit", "nsfw", "user_custom"]
            loaded_files = []
        
            for name in priority_order:
                fpath = os.path.join(DICT_DIR, f"{name}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._dictionary.update(data)
                            loaded_files.append(f"{name}({len(data)})")
                    except Exception as e:
                        logger.warning(f"⚠️ 加载 {name}.json 失败: {e}")
        
            # 加载 priority_order 之外的自定义 JSON
            for fname in os.listdir(DICT_DIR):
                if fname.endswith(".json") and fname[:-5] not in priority_order:
                    fpath = os.path.join(DICT_DIR, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._dictionary.update(data)
                            loaded_files.append(f"{fname[:-5]}({len(data)})")
                    except Exception as e:
                        logger.warning(f"⚠️ 加载 {fname} 失败: {e}")
        
            if loaded_files:
                logger.info(f"📚 已加载词典: {', '.join(loaded_files)}")
        else:
            os.makedirs(DICT_DIR, exist_ok=True)
            logger.info(f"📁 已创建词典目录: {DICT_DIR}")
    
        logger.info(f"📖 词典总数: {len(self._dictionary)} 条")

    def save_dictionary(self):
        """新翻译的词写入 user_custom.json (不污染分类词典)"""
        try:
            os.makedirs(DICT_DIR, exist_ok=True)
            custom_path = os.path.join(DICT_DIR, "user_custom.json")
        
            # 读原有 user_custom
            existing = {}
            if os.path.exists(custom_path):
                with open(custom_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.update(self._dictionary)
        
            with open(custom_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning(f"⚠️ 词典保存失败: {e}")

    # ========== 工具函数 ==========

    @staticmethod
    def _has_chinese(text: str) -> bool:
        """检测是否含中文"""
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _is_pure_english(text: str) -> bool:
        """是否纯英文（含数字符号）"""
        return all(ord(c) < 128 for c in text)

    # ========== 核心翻译 ==========

    def translate(self, text: str, mode: str = "auto") -> str:
        """
        mode:
          - "dict"   : 纯词典
          - "ai"     : 纯 AI (Qwen)
          - "auto"   : 词典命中直接用, 未命中调 AI (推荐)
        """
        if not text or not text.strip():
            return ""

        # 缓存命中直接返回
        cache_key = f"{mode}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = ""
        used_ai = False  # 🆕 标记是否用了 Qwen

        if mode == "dict":
            result = self._translate_by_dict(text)

        elif mode == "ai":
            if self.qwen_enhancer:
                try:
                    result = self.qwen_enhancer.translate_zh_to_en(text)
                    used_ai = True
                except Exception as e:
                    logger.warning(f"⚠️ Qwen 翻译失败,降级到词典: {e}")
                    result = self._translate_by_dict(text)
            else:
                result = self._translate_by_dict(text)

        else:  # auto
            if text in self._dictionary:
                result = self._dictionary[text]
            else:
                dict_result = self._translate_by_dict(text)
                hit_rate = self._calc_hit_rate(text, dict_result)

                if hit_rate >= 0.8:
                    result = dict_result
                elif self.qwen_enhancer:
                    try:
                        result = self.qwen_enhancer.translate_zh_to_en(text)
                        used_ai = True
                    except Exception:
                        result = dict_result
                else:
                    result = dict_result

        # 写入缓存
        self._cache[cache_key] = result

        # 🆕 用了 AI 才释放，纯词典不动
        if used_ai and self.qwen_enhancer is not None:
            try:
                if getattr(self.qwen_enhancer, 'model', None) is not None:
                    self.qwen_enhancer.unload()
                    print("[TRANS] ✅ Qwen 已释放显存/内存")
            except Exception as e:
                print(f"[TRANS] ⚠️ Qwen 释放失败: {e}")

        return result

    def _calc_hit_rate(self, zh_text: str, en_result: str) -> float:
        """粗略估算词典命中率: 英文词数 / 中文分词数"""
        import jieba
        zh_words = [w for w in jieba.lcut(zh_text) if len(w) > 1]
        if not zh_words:
            return 1.0
        # 英文结果里逗号分隔的词数
        en_tokens = [t.strip() for t in en_result.split(",") if t.strip()]
        return min(len(en_tokens) / len(zh_words), 1.0)

    def _translate_by_dict(self, text: str) -> str:
        """词典 + jieba 分词翻译"""
        # 按逗号/句号切分大段
        segments = re.split(r"[,，。;；\n]+", text)
        translated_segments = []

        has_new_word = False

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # 纯英文片段原样保留
            if not self._has_chinese(seg):
                translated_segments.append(seg)
                continue

            # 整段命中字典
            if seg in self._dictionary:
                translated_segments.append(self._dictionary[seg])
                continue

            # 分词翻译
            if HAS_JIEBA:
                words = list(jieba.cut(seg))
            else:
                words = list(seg)   # 降级：单字切分

            trans_words = []
            for w in words:
                w = w.strip()
                if not w:
                    continue

                if not self._has_chinese(w):
                    trans_words.append(w)
                elif w in self._dictionary:
                    trans_words.append(self._dictionary[w])
                else:
                    # 单词级 Google 翻译（一次性写入字典）
                    if self.google_translator:
                        try:
                            tr = self.google_translator.translate(w)
                            if tr:
                                self._dictionary[w] = tr
                                trans_words.append(tr)
                                has_new_word = True
                                continue
                        except Exception:
                            pass
                    # 翻译失败：保留原文
                    trans_words.append(w)

            translated_segments.append(" ".join(trans_words))

        result = ", ".join(translated_segments)

        if has_new_word:
            self.save_dictionary()

        return result

    # ========== 批量接口 ==========

    def translate_batch(self, texts: list) -> list:
        """批量翻译"""
        return [self.translate(t) for t in texts]

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("🗑 翻译缓存已清空")

    def stats(self) -> dict:
        """统计信息"""
        return {
            "dict_size": len(self._dictionary),
            "cache_size": len(self._cache),
            "qwen_available": self.qwen_enhancer is not None,
            "google_available": self.google_translator is not None,
        }