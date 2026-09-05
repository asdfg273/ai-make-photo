import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import List

from utils.paths import CONFIG_FILE
import logging

logger = logging.getLogger(__name__)


# 默认可改键位（设置界面可编辑，存进 app_config.json）
DEFAULT_SHORTCUTS = {
    "generate":         "Ctrl+Return",   # 触发生成
    "interrupt":        "Escape",        # 中断生成
    "page_txt2img":     "Ctrl+1",        # 切到 文生图
    "page_img2img":     "Ctrl+2",        # 切到 图生图
    "page_video":       "Ctrl+3",        # 切到 动画
    "page_gallery":     "Ctrl+4",        # 切到 画廊
    "extension_market": "Ctrl+E",        # 扩展市场
}

# 快捷键动作的中文标签（设置界面展示用）
SHORTCUT_LABELS = {
    "generate":         "🎨 触发生成",
    "interrupt":        "⏹ 中断生成",
    "page_txt2img":     "1️⃣ 切到 文生图",
    "page_img2img":     "2️⃣ 切到 图生图",
    "page_video":       "3️⃣ 切到 动画",
    "page_gallery":     "4️⃣ 切到 画廊",
    "extension_market": "🛒 扩展市场",
}


@dataclass
class AppConfig:
    config_file = CONFIG_FILE   # 绝对路径,不依赖 CWD
    
    # 基础参数默认值
    default_steps: int = 30
    default_strength: float = 0.6
    default_lora_weight: float = 0.7
    adetailer_enabled: bool = False
    theme: str = "darkly"
    device_preference:    str   = "auto"
    default_width: int = 512
    default_height: int = 768
    default_batch: int = 1
    default_cfg: float = 7.0
    default_sampler: str = "DPM++ 2M Karras"
    qwen_model_key: str = "qwen2vl_2b"
    last_prompt: str = ""
    last_neg: str = ""

    use_adetailer: bool = False
    adetailer_strength: float = 0.35
    use_ad_hand: bool = False
    ad_hand_strength: float = 0.25
    ad_hand_blend: float = 0.65

    use_hires: bool = False
    hires_denoise: float = 0.45

    default_trans_mode: int = 2          # 翻译模式默认档位（0 词典/1 AI/2 混合）
    shortcuts: dict = field(default_factory=lambda: dict(DEFAULT_SHORTCUTS))

    output_format: str = "PNG"
    output_dir: str = "outputs/"
    
    # 历史记录
    recent_models: List[str] = field(default_factory=list)
    recent_prompts: List[str] = field(default_factory=list)

    @classmethod
    def load(cls):
        """高容错加载：忽略废弃字段，补全缺失字段"""
        if os.path.exists(cls.config_file):
            try:
                with open(cls.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    config = cls()
                    # 只更新存在的字段，完美向下兼容
                    for key, value in data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    return config
            except Exception as e:
                logger.warning(f"⚠️ 配置文件加载失败: {e}，将使用默认配置")
        return cls()

    def save(self):
        """保存配置到本地（原子写：先写临时文件再 os.replace，崩溃不写坏）"""
        try:
            cfg_dir = os.path.dirname(self.config_file) or "."
            fd, tmp_path = tempfile.mkstemp(
                dir=cfg_dir, prefix=".app_config_", suffix=".tmp")
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
                os.replace(tmp_path, self.config_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"⚠️ 保存配置失败: {e}")

    def add_recent_model(self, model_name):
        if model_name and model_name not in self.recent_models:
            self.recent_models.insert(0, model_name)
            self.recent_models = self.recent_models[:10]  # 最多存10个
            self.save()  # 立即保存

    def add_recent_prompt(self, prompt):
        if prompt and prompt not in self.recent_prompts:
            self.recent_prompts.insert(0, prompt[:200])
            self.recent_prompts = self.recent_prompts[:20] # 最多存20个
            self.save()  # 立即保存