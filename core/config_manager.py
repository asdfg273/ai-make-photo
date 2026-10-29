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

# 快捷键动作的界面标签（多语言由 ui.i18n 统一处理）
SHORTCUT_LABELS = {
    "generate":         "触发生成",
    "interrupt":        "中断生成",
    "page_txt2img":     "切到 文生图",
    "page_img2img":     "切到 图生图",
    "page_video":       "切到 动画",
    "page_gallery":     "切到 画廊",
    "extension_market": "扩展市场",
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
    ui_language: str = "zh"           # 界面语言: zh / ja / en
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

    # ── 高清修复方式与顺序 ──────────────────────────────────
    # hires_mode:
    #   "auto"   跟随显存自动选（<10GB → 分块）
    #   "latent" 整图 latent 放大 + 一次 img2img（快，但显存吃紧）
    #   "tiled"  分块重绘 + 边缘融合（8GB 卡推荐，能出 2K~4K）
    # hires_order:
    #   "hires_first"    基础图 → 高清 → 修脸（先提清晰度再修脸，脸更锐）
    #   "adetailer_first" 基础图 → 修脸 → 高清（先修脸再放大，脸更自然但可能被二次模糊）
    # 两种顺序对人脸效果不同，交给用户决定。
    hires_mode: str = "auto"
    hires_order: str = "hires_first"
    # 分块参数（hires_mode="tiled" 时生效）
    hires_tile_size: int = 640
    hires_tile_overlap: int = 48

    freeu_enabled: bool = False

    # DiT 推理优化：仅作用于 Flux / SD3；步间缓存另支持 SDXL。
    dit_quantization: str = "off"       # off / fp8 / nvfp4
    dit_group_offload: bool = True
    dit_offload_blocks: int = 1
    step_cache_enabled: bool = False
    step_cache_threshold: float = 0.18
    step_cache_max_skips: int = 2
    clip_model_path: str = ""          # 画廊语义检索的本地 CLIP 目录

    # ── 区域提示词 / 局部重绘 ──────────────────────────────
    # 每个区域是一个 dict（与 core.regional.Region 字段一致）：
    #   {"name","prompt","negative","weight","shape","rect","mask_path",
    #    "color","enabled","feather","controlnet","adetailer"}
    # rect 是**归一化**的 (x0,y0,x1,y1)；shape="mask" 时用 mask_path 的黑白图。
    # 生成时按区域逐个做蒙版重绘（权重=该区域重绘强度）。
    use_regions: bool = False
    regions: List[dict] = field(default_factory=list)

    # CLIP Skip：0=关闭（用最后一层）, 1=倒数第二层（社区常说 CLIP skip 2）
    # 动漫向底模（Pony/Illustrious/NoobAI/多数 SD1.5 二次元）多在倒数第二层
    # 上训练，用默认层会明显掉质量。
    clip_skip: int = 0

    # Qwen 提示词辅助模式：strict=严格翻译 / polish=适当润色 / keywords=关键词帮写
    enhance_mode: str = "strict"

    # 图生图底图细节预处理（CLAHE）：仅在检测到过暗/低对比时介入。
    # 关掉 = 完全按原图使用（不改变像素）。
    enhance_init_image: bool = True

    default_trans_mode: int = 2          # 翻译模式默认档位（0 词典/1 AI/2 混合）
    shortcuts: dict = field(default_factory=lambda: dict(DEFAULT_SHORTCUTS))

    output_format: str = "PNG"
    output_dir: str = "outputs/"

    # ── 水印 ──────────────────────────────────────────────
    # 文字与图片可同时配置；都有时**图片优先**（图片水印通常更正式），
    # 图片加载失败会自动退回文字。位置可选右下/左下/右上/左上/居中/平铺。
    watermark_enabled: bool = False
    watermark_text: str = ""
    watermark_image: str = ""            # 自定义图片水印路径
    watermark_position: str = "右下"
    watermark_opacity: float = 0.6       # 0.02 ~ 1.0
    watermark_scale: float = 1.0         # 0.2 ~ 5.0

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

    def add_recent_prompt(self, prompt: str):
        if not prompt or not prompt.strip():
            return
        prompt = prompt.strip()
        if prompt in self.recent_prompts:
            self.recent_prompts.remove(prompt)
        self.recent_prompts.insert(0, prompt)
        self.recent_prompts = self.recent_prompts[:20]  # ✅ 截断到 20 个
        self.save()
