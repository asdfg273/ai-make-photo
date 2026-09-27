"""
🌐 中译英服务
优先级：缓存 → Qwen2-VL（智能整段翻译）→ 本地词典（逐词翻译）→ Google（兜底）
"""
import os
import json
import re
import socket
import threading
import logging

logger = logging.getLogger(__name__)

from utils.paths import DATA_DIR, DICT_FILE
DICT_DIR = os.path.join(DATA_DIR, "dictionaries")
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

    def __init__(self, dict_file=None, qwen_enhancer=None):
        self.dict_file = dict_file or DICT_FILE
        self.lock = threading.Lock()
        self._cache = {}            # 整段翻译缓存
        self._dictionary = {}       # 词级翻译字典
        self._new_words = {}        # 本次运行新学到、待落盘的词（只写这些）
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
        """把**本次新增**的词写入 user_custom.json（不污染分类词典）。

        ⚠️ 旧实现是 `existing.update(self._dictionary)` —— 等于把 general/style/
        outfit/nsfw 等**全部内置词典**复制进用户词典。后果：
          · 用户词典文件被灌成几千条，用户根本删不掉自己的误译；
          · 内置词典后续更新会被这份副本永久盖住（优先级顺序名存实亡）。
        现在只落盘本次 Google/Qwen 兜底新产生的词。
        """
        try:
            new_words = dict(getattr(self, "_new_words", {}) or {})
            if not new_words:
                logger.debug("无新增词，跳过词典落盘")
                return
            os.makedirs(DICT_DIR, exist_ok=True)
            custom_path = os.path.join(DICT_DIR, "user_custom.json")

            existing = {}
            if os.path.exists(custom_path):
                with open(custom_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            before = len(existing)
            existing.update(new_words)

            # 原子写：避免崩溃/断电留下半截 JSON 导致整份用户词典损坏
            import tempfile as _tf
            fd, tmp_path = _tf.mkstemp(
                dir=DICT_DIR, prefix=".user_custom_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False,
                              indent=2, sort_keys=True)
                os.replace(tmp_path, custom_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.info(f"💾 用户词典已更新：+{len(existing) - before} 条 "
                        f"（共 {len(existing)} 条）")
            # 已落盘的词不再重复计
            with self.lock:
                self._new_words.clear()
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

    def translate(self, text: str, mode: str = "auto", target_lang: str = "en") -> str:
        """
        mode:
          - "dict"   : 纯词典
          - "ai"     : 纯 AI (Qwen)
          - "auto"   : 词典命中直接用, 未命中调 AI (推荐)
        target_lang:
          - "en"     : 中译英（默认，走 词典/Qwen 标签翻译流程）
          - 其它(如 "ja"): 配音等自然语言场景，直接走 Qwen 通用翻译
        """
        if not text or not text.strip():
            return ""

        # 非英文目标语言（如 SoVITS 配音翻译为日语）走 Qwen 通用翻译
        if target_lang != "en":
            if self.qwen_enhancer is None:
                logger.warning(f"⚠️ 无 Qwen 实例，无法翻译为 {target_lang}，返回原文")
                return text
            try:
                result = self.qwen_enhancer.translate(text, target_lang=target_lang)
            except Exception as e:
                logger.warning(f"⚠️ Qwen 翻译为 {target_lang} 失败，返回原文: {e}")
                return text
            # 不自动 unload：Qwen 是跨线程共享单例,此处卸载会与
            # 改写/识图线程形成竞态。显存释放由用户显式"卸载模型"触发。
            return result

        # 缓存命中直接返回
        cache_key = f"{mode}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # ── 分段翻译 ──
        # 整段一起翻有两个问题（实测）：
        #   · 一处低命中率会把**整段**拉去问 Qwen，浪费一次模型加载；
        #   · 长句里某个词失败，整段回退成原文 → 中文残留在提示词里，
        #     而中文 1 字≈1 token，会把 CLIP 预算吃光（见 prompt_budget）。
        # 按标点切段后逐段决定，命中率高的段完全不碰 AI。
        segs = [s for s in re.split(r"[,，。；;、\n]+", text) if s.strip()]
        if len(segs) > 1:
            parts = [self._translate_single(s.strip(), mode) for s in segs]
            result = ", ".join(p for p in parts if p)
        else:
            result = self._translate_single(text, mode)

        # 写入缓存（设上限,防无限增长）
        # 用锁保护：生成线程与后台改写线程会并发翻译，旧实现只 clear/写
        # 无锁字典，可能把别人的条目整片清掉。
        with self.lock:
            if len(self._cache) > 500:
                self._cache.clear()
            self._cache[cache_key] = result

        # 注意：不在这里自动 unload Qwen —— 它是跨线程共享单例,
        # 自动卸载会与并发的改写/识图线程形成竞态(P1-17)。
        return result

    def _translate_single(self, text: str, mode: str) -> str:
        """单段翻译（原 translate 的主体逻辑，供分段复用）。"""
        if not text or not text.strip():
            return ""
        text = text.strip()

        if mode == "dict":
            return self._translate_by_dict(text)

        if mode == "ai":
            if self.qwen_enhancer:
                try:
                    return self.qwen_enhancer.translate_zh_to_en(text)
                except Exception as e:
                    logger.warning(f"⚠️ Qwen 翻译失败,降级到词典: {e}")
                    return self._translate_by_dict(text)
            return self._translate_by_dict(text)

        # auto
        if text in self._dictionary:
            return self._dictionary[text]

        dict_result = self._translate_by_dict(text)
        hit_rate = self._calc_hit_rate(text, dict_result)
        if hit_rate >= 0.8:
            return dict_result
        if self.qwen_enhancer:
            try:
                return self.qwen_enhancer.translate_zh_to_en(text)
            except Exception as e:
                logger.warning(f"Qwen 翻译失败({e})，用词典结果")
                return dict_result
        return dict_result

    def _calc_hit_rate(self, zh_text: str, en_result: str) -> float:
        """粗略估算词典命中率: 英文词数 / 中文分词数"""
        # ⚠️ 旧实现硬 import jieba，绕过了模块顶部的 HAS_JIEBA 可选导入：
        #    没装 jieba 时这里会直接 ImportError（丢失整段翻译）。
        if not HAS_JIEBA:
            return 1.0
        try:
            import jieba
            zh_words = [w for w in jieba.lcut(zh_text) if len(w) > 1]
        except Exception:
            return 1.0
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
            google_ok = self.google_translator is not None and not getattr(
                self, '_google_disabled', False)
            fail_count = 0
            for w in words:
                w = w.strip()
                if not w:
                    continue

                if not self._has_chinese(w):
                    trans_words.append(w)
                elif w in self._dictionary:
                    trans_words.append(self._dictionary[w])
                else:
                    if google_ok:
                        old_to = socket.getdefaulttimeout()
                        socket.setdefaulttimeout(3)
                        try:
                            tr = self.google_translator.translate(w)
                            if tr:
                                self._dictionary[w] = tr
                                # 只记进"待落盘"集合：save_dictionary 只写这些，
                                # 不会把内置词典整份复制进 user_custom.json
                                self._new_words[w] = tr
                                trans_words.append(tr)
                                has_new_word = True
                                fail_count = 0
                                continue
                        except Exception as e:
                            fail_count += 1
                            if fail_count >= 2:
                                # 连续失败：本次会话不再尝试 Google
                                self._google_disabled = True
                                google_ok = False
                                logger.warning(
                                    f"⚠️ Google 翻译连续失败({e})，本次会话禁用，"
                                    f"改用词典+原文")
                        finally:
                            socket.setdefaulttimeout(old_to)
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
