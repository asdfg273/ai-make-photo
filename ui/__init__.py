# ui/__init__.py
# ============================================================
#  UI 模块公共 API
#  集中导出所有 UI 组件、Mixin、设计令牌
# ============================================================

# ── 自定义控件 ──
from ui.widgets import FloatSlider, GpuCanvas

# ── 启动/弹窗 ──
from ui.splash import SplashScreen, create_splash
from ui.disclaimer import check_global_disclaimer, check_voice_clone_consent, DisclaimerDialog

# ── 设计系统 ──
from ui.design_tokens import DESIGN_TOKENS, DARK_STYLE

# ── 提示词数据 ──
from ui.tooltips import PARAM_TOOLTIPS, tip

# ── 画廊 ──
from ui.gallery_panel import GalleryPanel, ImageViewerDialog

# ── 扩展市场 ──
from ui.extension_market import ExtensionMarketDialog

# ── Mixin（组合进主窗口）──
from ui.ui_builder import UIBuilderMixin
from ui.preset_manager import PresetManagerMixin, TooltipMixin
from ui.video_panel_mixin import VideoPanelMixin

# ── GPU 基础设施（桥梁，实际定义在 utils）──
from utils.gpu_init import enable_gpu_acceleration