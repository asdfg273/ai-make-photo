# ui/icons.py
# ============================================================
#  图标注册接口 — 全项目唯一的图标来源
#
#  设计目标：
#  - 当前版本主界面为纯文字（无内置图标），本模块默认全部返回 None，
#    不改变任何现有界面；
#  - 后续想自定义图标（PNG/SVG/ICO），只需在应用启动早期调用
#    register_icon() 或 load_icon_dir()，导航栏、菜单、按钮会自动生效。
#
#  用法示例（在 main.py 的 apply_theme 之后、setup_ui 之前调用）：
#
#      from ui.icons import register_icon, load_icon_dir
#      register_icon("txt2img", "icons/txt2img.svg")   # 导航页图标
#      register_icon("img2img", "icons/img2img.svg")
#      register_icon("generate", "icons/bolt.png")     # 任意按钮/动作 key
#      load_icon_dir("icons")      # 或整个目录批量注册（文件名即 key）
#
#  约定：
#  - key 与用途：导航页用 page_id（txt2img / img2img / video / gallery /
#    characters）；其他按钮/菜单动作自行约定字符串 key；
#  - 路径可以是相对项目根目录的相对路径，也可以是绝对路径；
#  - 文件缺失时静默返回 None 并记一条日志，不影响界面。
# ============================================================
import logging
import os

from PyQt6.QtGui import QIcon

logger = logging.getLogger(__name__)

# key (str) -> QIcon
_REGISTRY: dict = {}


def register_icon(key: str, source) -> bool:
    """注册一个图标。source 为 QIcon 实例或图片文件路径。

    返回是否注册成功（重复注册会覆盖旧值）。"""
    if not key:
        return False
    if isinstance(source, QIcon):
        if source.isNull():
            logger.warning(f"[icons] 注册失败（空图标）: {key}")
            return False
        _REGISTRY[key] = source
        return True
    if isinstance(source, str):
        if not os.path.exists(source):
            logger.warning(f"[icons] 注册失败（文件不存在）: {key} -> {source}")
            return False
        icon = QIcon(source)
        if icon.isNull():
            logger.warning(f"[icons] 注册失败（无法加载）: {key} -> {source}")
            return False
        _REGISTRY[key] = icon
        return True
    logger.warning(f"[icons] 注册失败（不支持的类型）: {key}")
    return False


def load_icon_dir(directory: str, recursive: bool = False) -> int:
    """批量注册目录下的图标文件，文件名（去扩展名）即 key。返回注册数。"""
    exts = (".png", ".svg", ".ico", ".jpg", ".jpeg", ".webp")
    count = 0
    if not os.path.isdir(directory):
        logger.warning(f"[icons] 目录不存在: {directory}")
        return 0
    for root, dirs, files in os.walk(directory):
        if not recursive:
            dirs[:] = []
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue
            if register_icon(stem, os.path.join(root, name)):
                count += 1
    logger.info(f"[icons] 从 {directory} 注册 {count} 个图标")
    return count


def get_icon(key: str):
    """按 key 取图标；未注册返回 None（调用方保持纯文字）。"""
    if not key:
        return None
    return _REGISTRY.get(key)


def has_icon(key: str) -> bool:
    return bool(key) and key in _REGISTRY


def apply_icon(widget, key: str) -> bool:
    """把 key 对应的图标挂到按钮/动作上；未注册则不改动 widget。

    适用于 QPushButton / QToolButton / QAction 等带 setIcon 的控件。"""
    icon = get_icon(key)
    if icon is None:
        return False
    setter = getattr(widget, "setIcon", None)
    if not callable(setter):
        return False
    setter(icon)
    return True


def registered_keys() -> list:
    """返回全部已注册 key（调试用）。"""
    return sorted(_REGISTRY.keys())
