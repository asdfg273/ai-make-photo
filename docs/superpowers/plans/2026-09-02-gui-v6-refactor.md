# GUI v6.0 重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 AI 绘画工作站的 PyQt6 GUI 从单体 `UIBuilderMixin` 重构为三栏工作台（导航栏 + 中央工作区 + 右侧参数面板 + 底部胶片条），引入 QDarkStyleSheet 深蓝灰主题，统一图片/动画画廊，版本号升级 v6.0。

**Architecture:** 新建模块化 UI 包（`theme.py` / `shell.py` / `core_panel.py` / `nav.py` / `pages/` / `widgets/` / `contracts.py`），生成核心控件在 shell 层全局单例创建一次，页面只建专属控件；业务 mixin 零改动；旧 `ui_builder.py` 全程保留作对照，全部验证通过后删除。设计文档：`docs/superpowers/specs/2026-09-02-gui-refactor-design.md`。

**Tech Stack:** PyQt6（>=6.7）、qdarkstyle（venv 已装）、Python venv（`venv/Scripts/python.exe`）。测试为纯 Python 脚本（项目无 pytest），用 `QT_QPA_PLATFORM=offscreen` 无头跑。

## Global Constraints

- 所有控件属性名原样保留，一个不改；控件清单见 Task 2 的 `contracts.py`
- 生成核心控件（模型/Prompt/尺寸/步数/CFG/采样器/生成停止按钮/进度/预览画布）只在 shell 层创建一次，页面禁止重建同名控件
- 方法契约：新 shell 必须提供 `append_log` / `set_status` / `set_progress` / `play_video` 同名方法
- 兼容别名集中安装：`btn_gen→btn_generate`、`btn_stop→btn_interrupt`、`scale_str→scale_strength`、`scale_hires→scale_hires_denoise`、`progress_total→progress_gen`、`progress→progress_gen`、`preview_canvas→lbl_preview`、`pose_canvas→lbl_cn_thumb`、`combo_loras`、`scale_loras`
- `self.gallery` 及其信号/方法签名不变
- 契约自检分级：关键缺失→生成按钮置灰+tooltip+错误占位；非关键→仅告警
- 胶片条/画廊刷新 200ms 防抖合并
- 业务 mixin（`utils/app_events.py`、`utils/app_generation.py`、`ui/preset_manager.py`、`ui/video_panel_mixin.py`）零改动
- 测试运行方式：`set QT_QPA_PLATFORM=offscreen && venv/Scripts/python.exe tests/test_ui_contract.py`（Windows cmd）;Git Bash 用 `QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_ui_contract.py`
- 迁移顺序：主题层 → 外壳骨架 → 生成核心区 → 文生图 → 图生图 → 共享折叠分组 → 动画 → 统一画廊 → 样式清场 → 版本号 → 删除旧文件
- 每阶段一个 commit，全程不破坏旧 UI 可用性（用环境变量 `AI_STUDIO_UI=v2` 切换新旧 UI，默认旧）
- 禁止改动任何生成/推理/模型加载逻辑

---

### Task 1: 主题中枢 theme.py

**Files:**
- Create: `ui/theme.py`
- Test: `tests/test_theme.py`

**Interfaces:**
- Consumes: qdarkstyle（venv 已安装）
- Produces: `apply_theme(app: QApplication) -> str`（返回实际应用的主题名 `"qdarkstyle"` 或 `"fusion-fallback"`）；`PALETTE: dict`（色板常量）；`APP_QSS: str`（项目自定义覆盖层）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_theme.py`：

```python
# tests/test_theme.py — 主题模块契约测试（无头）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from ui.theme import apply_theme, PALETTE, APP_QSS
    used = apply_theme(app)
    assert used in ("qdarkstyle", "fusion-fallback"), used
    assert isinstance(PALETTE, dict) and "accent" in PALETTE
    assert "QPushButton" in APP_QSS
    ss = app.styleSheet()
    assert len(ss) > 100, "样式表未生效"
    print(f"PASS test_theme (theme={used})")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_theme.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.theme'`

- [ ] **Step 3: 实现 theme.py**

创建 `ui/theme.py`：

```python
# ui/theme.py
# ============================================================
#  主题中枢 — qdarkstyle 深蓝灰基底 + 项目自定义覆盖层
#  全项目唯一主题入口；组件一律从 PALETTE 取色，杜绝硬编码
# ============================================================
import logging

logger = logging.getLogger(__name__)

PALETTE = {
    "accent":      "#4a9eff",   # qdarkstyle 原生蓝，导航选中/主按钮
    "accent_hi":   "#6cb2ff",
    "bg":          "#19232d",
    "bg_soft":     "#22303d",
    "fg":          "#dfe5ec",
    "fg_mute":     "#8fa1b3",
    "danger":      "#e06c75",
    "radius":      6,
    "space":       8,
}

# 项目自定义覆盖层：只做品牌微调 + 语义类，不重画控件
APP_QSS = """
/* 语义类：组件用 setProperty 打标，样式集中在此 */
QLabel[role="hint"]  { color: #8fa1b3; font-size: 12px; }
QLabel[role="title"] { color: #dfe5ec; font-weight: bold; font-size: 13px; }
QLabel[role="value"] { color: #4a9eff; font-weight: bold; }

QPushButton[kind="primary"] {
    background: #4a9eff; color: #0d1620; font-weight: bold;
    border: none; border-radius: 6px; padding: 10px 20px; font-size: 15px;
}
QPushButton[kind="primary"]:hover    { background: #6cb2ff; }
QPushButton[kind="primary"]:disabled { background: #37414b; color: #8fa1b3; }

/* 左侧导航栏 */
QToolButton#navBtn {
    color: #8fa1b3; border: none; border-radius: 6px;
    padding: 8px 4px; font-size: 12px;
}
QToolButton#navBtn:hover   { color: #dfe5ec; background: #22303d; }
QToolButton#navBtn:checked { color: #ffffff; background: #4a9eff; }

/* 折叠分组 */
QWidget#collapsibleHeader { background: #22303d; border-radius: 6px; }
QGroupBox { border-radius: 6px; }
"""


def apply_theme(app) -> str:
    """应用主题。返回实际使用的主题名。qdarkstyle 失败时回退 Fusion 深色。"""
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt6") + APP_QSS)
        logger.info("主题: qdarkstyle (dark)")
        return "qdarkstyle"
    except Exception as e:
        logger.warning(f"⚠️ qdarkstyle 加载失败，回退 Fusion 深色: {e}")
        from PyQt6.QtGui import QPalette, QColor
        app.setStyle("Fusion")
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(25, 35, 45))
        p.setColor(QPalette.ColorRole.WindowText, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Base, QColor(34, 48, 61))
        p.setColor(QPalette.ColorRole.Text, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Button, QColor(34, 48, 61))
        p.setColor(QPalette.ColorRole.ButtonText, QColor(223, 229, 236))
        p.setColor(QPalette.ColorRole.Highlight, QColor(74, 158, 255))
        app.setPalette(p)
        app.setStyleSheet(APP_QSS)
        return "fusion-fallback"
```

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_theme.py`
Expected: `PASS test_theme (theme=qdarkstyle)`

- [ ] **Step 5: Commit**

```bash
git add ui/theme.py tests/test_theme.py
git commit -m "feat(ui): 主题中枢 theme.py——qdarkstyle 基底 + 语义类覆盖层 + Fusion 兜底"
```

---

### Task 2: 控件契约 contracts.py + 契约测试基座

**Files:**
- Create: `ui/contracts.py`
- Create: `tests/test_ui_contract.py`

**Interfaces:**
- Consumes: 无（纯数据 + 检查函数）
- Produces:
  - `GLOBAL_WIDGETS: list[str]` — 全局单例控件名
  - `PAGE_WIDGETS: dict[str, list[str]]` — 各页专属控件名（key: `"txt2img"/"img2img"/"video"/"gallery"`）
  - `METHOD_CONTRACT: list[str]` — `["append_log", "set_status", "set_progress", "play_video"]`
  - `ALIASES: dict[str, str]` 与 `LIST_ALIASES: dict[str, list[str]]`
  - `CRITICAL: set[str]` — 关键控件/方法名
  - `install_aliases(host) -> None`
  - `check_contract(host) -> tuple[list[str], list[str]]` — 返回 `(critical_missing, minor_missing)`
  - `apply_degradation(host, critical_missing) -> None` — 关键缺失时置灰生成入口

- [ ] **Step 1: 写失败测试**

创建 `tests/test_ui_contract.py`（本任务先测 contracts 模块自身，后续任务扩展为整窗自检）：

```python
# tests/test_ui_contract.py — UI 契约测试（无头，纯脚本）
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton


def test_contract_lists():
    from ui.contracts import (GLOBAL_WIDGETS, PAGE_WIDGETS, METHOD_CONTRACT,
                              ALIASES, LIST_ALIASES, CRITICAL)
    assert "txt_prompt" in GLOBAL_WIDGETS
    assert "combo_model" in GLOBAL_WIDGETS
    assert "btn_generate" in GLOBAL_WIDGETS
    assert set(PAGE_WIDGETS) == {"txt2img", "img2img", "video", "gallery"}
    assert "play_video" in METHOD_CONTRACT
    assert ALIASES["btn_gen"] == "btn_generate"
    assert "btn_generate" in CRITICAL


def test_check_and_degrade():
    from ui.contracts import install_aliases, check_contract, apply_degradation
    app = QApplication.instance() or QApplication([])

    class FakeHost:  # 最小假宿主：只有部分控件
        pass

    host = FakeHost()
    host.txt_prompt = QLineEdit()
    host.btn_generate = QPushButton()
    crit, minor = check_contract(host)
    assert "combo_model" in crit            # 关键缺失被识别
    apply_degradation(host, crit)
    assert not host.btn_generate.isEnabled()  # 生成按钮被置灰
    # 别名安装
    install_aliases(host)
    assert host.btn_gen is host.btn_generate
    print("PASS test_contract_lists / test_check_and_degrade")


if __name__ == "__main__":
    test_contract_lists()
    test_check_and_degrade()
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.contracts'`

- [ ] **Step 3: 实现 contracts.py**

创建 `ui/contracts.py`。控件清单从 `ui/ui_builder.py` 现有代码核实（`grep -oE "self\.[a-zA-Z_0-9]+ = Q" ui/ui_builder.py`），分类如下：

```python
# ui/contracts.py
# ============================================================
#  UI 控件/方法契约 — 业务 mixin 依赖的命名契约集中在此
#  启动自检: check_contract(); 关键缺失降级: apply_degradation()
# ============================================================
import logging

logger = logging.getLogger(__name__)

# ── 全局单例控件（shell/core_panel 创建一次，页面禁止重建）──
GLOBAL_WIDGETS = [
    # 生成核心
    "combo_model", "combo_model_type", "combo_sampler", "combo_res",
    "txt_prompt", "txt_neg", "spin_steps", "spin_width", "spin_height",
    "spin_seed", "spin_count", "combo_img_format", "combo_device",
    "chk_auto_enhance", "chk_auto_features", "btn_enhance_prompt",
    "btn_vision_prompt", "lbl_model_info",
    # 生成控制
    "btn_generate", "btn_interrupt", "progress_gen", "lbl_status",
    "btn_preset_menu", "btn_save_preset", "btn_restore_preset",
    "combo_preset", "lbl_preset_badge",
    # 预览区
    "lbl_preview", "btn_open_editor", "btn_save_as",
    "btn_send_img2img", "btn_send_inpaint", "txt_log_image",
    # 画廊
    "gallery",
    # 共享折叠分组（LoRA/ControlNet/高级/X-Y）
    "combo_lora_0", "combo_lora_1", "combo_lora_2",
    "scale_lora_0", "scale_lora_1", "scale_lora_2",
    "btn_refresh_lora", "btn_insert_lora_all", "text_lora_info",
    "combo_cn_type", "btn_load_cn_img", "lbl_cn_thumb",
    "chk_reference_only", "chk_use_pose", "chk_pose_transfer",
    "chk_enable_hires", "chk_hires", "combo_hires_scale",
    "combo_hires_upscaler", "chk_use_adetailer", "chk_use_ad_hand",
    "combo_adetailer_model", "combo_ad_target", "combo_ad_hand",
    "chk_enable_xy", "combo_x_type", "combo_y_type",
    "entry_x_vals", "entry_y_vals",
    "chk_use_tiled", "spin_tiled_w", "spin_tiled_h",
    "spin_tile_overlap", "combo_tile_size", "btn_run_tiled",
    "chk_use_ipa", "combo_ipa_variant", "spin_ipa_scale", "lbl_ipa_image",
    "chk_use_preview", "spin_preview_interval",
]

# ── 页面专属控件 ──
PAGE_WIDGETS = {
    "txt2img": [],  # 专属区为空，核心控件全在全局区
    "img2img": [
        "btn_load_img", "btn_clear_img", "lbl_img_path", "lbl_ref_thumb",
        "scale_strength", "lbl_ref_fidelity", "scale_ref_fidelity",
    ],
    "video": [
        "btn_gen_video", "video_player", "video_widget", "video_list",
        "txt_video_prompt", "txt_video_neg", "txt_log_video",
        "combo_video_mode", "combo_video_fmt", "combo_video_sched",
        "chk_long_video", "chk_frame_interp", "combo_frame_interp",
        "chk_video_upscale", "chk_video_voice", "combo_tts_engine",
        "chk_make_comic", "cmb_motion_lora_pick", "motion_lora_container",
        "travel_container", "wrap_travel_segments", "wrap_travel_text",
        "txt_neg_prompt_travel", "combo_travel_mode",
        "wrap_chattts", "wrap_sovits", "combo_sovits_ref",
        "txt_sovits_reftext", "chk_sovits_auto_translate",
        "txt_video_voice", "audio_output", "lbl_video_status",
        "lbl_video_duration", "lbl_video_input", "lbl_video_placeholder",
        "btn_video_pause", "btn_video_stop", "btn_video_save",
        "btn_video_refresh", "lbl_dynamic_hint",
    ],
    "gallery": [],  # 复用全局 self.gallery
}

# ── 方法契约（业务 mixin 调用的、定义在 UI 层的方法）──
METHOD_CONTRACT = ["append_log", "set_status", "set_progress", "play_video"]

# ── 兼容别名 ──
ALIASES = {
    "btn_gen": "btn_generate",
    "btn_stop": "btn_interrupt",
    "scale_str": "scale_strength",
    "scale_hires": "scale_hires_denoise",
    "progress_total": "progress_gen",
    "progress": "progress_gen",
    "preview_canvas": "lbl_preview",
    "pose_canvas": "lbl_cn_thumb",
}
LIST_ALIASES = {
    "combo_loras": ["combo_lora_0", "combo_lora_1", "combo_lora_2"],
    "scale_loras": ["scale_lora_0", "scale_lora_1", "scale_lora_2"],
}

# ── 关键契约：缺失则禁用生成入口 ──
CRITICAL = {
    "btn_generate", "btn_interrupt", "txt_prompt", "txt_neg",
    "combo_model", "lbl_preview", "progress_gen",
    *METHOD_CONTRACT,
}


def install_aliases(host) -> None:
    """集中安装兼容别名。"""
    for alias, real in ALIASES.items():
        if hasattr(host, real):
            setattr(host, alias, getattr(host, real))
        else:
            logger.warning(f"⚠️ 别名跳过（目标缺失）: {alias} -> {real}")
    for alias, names in LIST_ALIASES.items():
        setattr(host, alias, [getattr(host, n, None) for n in names])


def check_contract(host) -> tuple[list[str], list[str]]:
    """返回 (critical_missing, minor_missing)。方法用 callable 检查。"""
    all_widgets = GLOBAL_WIDGETS + [w for ws in PAGE_WIDGETS.values() for w in ws]
    critical, minor = [], []
    for name in all_widgets:
        if hasattr(host, name) and getattr(host, name) is not None:
            continue
        (critical if name in CRITICAL else minor).append(name)
    for name in METHOD_CONTRACT:
        if not callable(getattr(host, name, None)):
            if name not in critical:
                critical.append(name)
    for alias, real in ALIASES.items():
        if hasattr(host, real) and getattr(host, alias, None) is not getattr(host, real):
            minor.append(f"alias:{alias}")
    return critical, minor


def apply_degradation(host, critical_missing: list[str]) -> None:
    """关键契约缺失：置灰生成入口并说明原因。"""
    if not critical_missing:
        return
    reason = "关键组件缺失: " + ", ".join(critical_missing[:6])
    logger.error(f"❌ 契约自检失败（关键），生成入口禁用 — {reason}")
    for btn_name in ("btn_generate", "btn_gen_video"):
        btn = getattr(host, btn_name, None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setToolTip(f"⚠️ {reason}")
```

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_contract_lists / test_check_and_degrade`

注：`GLOBAL_WIDGETS` 清单在实现时以 `ui/ui_builder.py` 实际控件名为准逐条核对（清单不全不会导致功能错误——契约自检只报告缺失，不会误判多余控件）。

- [ ] **Step 5: Commit**

```bash
git add ui/contracts.py tests/test_ui_contract.py
git commit -m "feat(ui): 控件/方法契约 contracts.py——分级自检 + 生成入口降级"
```

---

### Task 3: 可折叠分组组件 CollapsibleSection

**Files:**
- Create: `ui/widgets/__init__.py`（空文件）
- Create: `ui/widgets/collapsible.py`
- Test: `tests/test_collapsible.py`

**Interfaces:**
- Consumes: PyQt6
- Produces: `CollapsibleSection(title: str, collapsed: bool = True)` — `.content` 是要往里放控件的 `QWidget`（带 `QVBoxLayout`，变量名 `content_layout`）；`set_collapsed(bool)`；`is_collapsed() -> bool`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_collapsible.py
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication, QLabel

def main():
    app = QApplication([])
    from ui.widgets.collapsible import CollapsibleSection
    sec = CollapsibleSection("LoRA", collapsed=True)
    assert sec.is_collapsed() is True
    assert not sec.content.isVisible()
    sec.content_layout.addWidget(QLabel("x"))
    sec.set_collapsed(False)
    assert sec.content.isVisible()
    print("PASS test_collapsible")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_collapsible.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 collapsible.py**

```python
# ui/widgets/collapsible.py
# 可折叠分组：标题行(点击展开/收起) + 内容区。LoRA/ControlNet/高级/X-Y 用
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton
from PyQt6.QtCore import Qt


class CollapsibleSection(QWidget):
    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.ArrowType.RightArrow if collapsed
                               else Qt.ArrowType.DownArrow)
        self._btn.setStyleSheet(
            "QToolButton { background:#22303d; border:none; border-radius:6px;"
            " padding:8px; font-weight:bold; text-align:left; }")
        self._btn.toggled.connect(self._on_toggled)
        root.addWidget(self._btn)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content_layout.setSpacing(6)
        self.content.setVisible(not collapsed)
        root.addWidget(self.content)

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if checked
                               else Qt.ArrowType.RightArrow)

    def set_collapsed(self, collapsed: bool):
        self._btn.setChecked(not collapsed)

    def is_collapsed(self) -> bool:
        return not self._btn.isChecked()
```

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_collapsible.py`
Expected: `PASS test_collapsible`

- [ ] **Step 5: Commit**

```bash
git add ui/widgets/__init__.py ui/widgets/collapsible.py tests/test_collapsible.py
git commit -m "feat(ui): CollapsibleSection 可折叠分组组件"
```

---

### Task 4: 导航栏 NavRail + 页面基类 PageBase + 注册表

**Files:**
- Create: `ui/nav.py`
- Create: `ui/pages/__init__.py`（内含 `PAGES` 注册表占位）
- Create: `ui/pages/base.py`
- Test: `tests/test_nav.py`

**Interfaces:**
- Consumes: Task 1 `theme.PALETTE`
- Produces:
  - `PageBase(QWidget)`：类属性 `page_id: str`、`title: str`、`icon: str`（Unicode 符号）；方法 `build(host) -> None`（页面把控件挂到 host）；`workspace() -> QWidget`（中央工作区内容）；`params_widget() -> QWidget | None`（右侧专属区内容，None=无）
  - `NavRail(QWidget)`：`page_selected = pyqtSignal(str)`；`set_pages(pages: list[type[PageBase]])`；`select(page_id: str)`
  - `ui.pages.PAGES: list[type[PageBase]]`（本任务为空列表，后续任务逐个注册）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_nav.py
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from ui.nav import NavRail
    from ui.pages.base import PageBase

    class DummyPage(PageBase):
        page_id, title, icon = "dummy", "测试", "🧪"
        def build(self, host): pass

    rail = NavRail()
    got = []
    rail.page_selected.connect(got.append)
    rail.set_pages([DummyPage])
    rail.select("dummy")
    assert got == ["dummy"]
    assert rail._buttons["dummy"].isChecked()
    print("PASS test_nav")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_nav.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 base.py / nav.py / pages/__init__.py**

```python
# ui/pages/base.py
from PyQt6.QtWidgets import QWidget


class PageBase(QWidget):
    """页面基类。子类设 page_id/title/icon，实现 build()。
    扩展新功能 = 新建 page 文件 + ui/pages/__init__.py 的 PAGES 加一行。"""
    page_id: str = ""
    title: str = ""
    icon: str = ""

    def build(self, host) -> None:
        """构建页面控件；页面专属控件按契约名挂到 host。host 为主窗口。"""
        raise NotImplementedError

    def workspace(self) -> QWidget:
        """中央工作区内容。默认空。"""
        return QWidget()

    def params_widget(self) -> QWidget | None:
        """右侧参数面板的页面专属区。None = 本页无专属参数。"""
        return None
```

```python
# ui/nav.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QButtonGroup
from PyQt6.QtCore import pyqtSignal, Qt


class NavRail(QWidget):
    """左侧导航栏：读页面注册表自动生成按钮，单选。"""
    page_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(64)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 8, 6, 8)
        self._layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        self._layout.addStretch(1)

    def set_pages(self, pages: list) -> None:
        for cls in pages:
            btn = QToolButton()
            btn.setObjectName("navBtn")          # theme.py 的 QSS 钩子
            btn.setText(f"{cls.icon}\n{cls.title}")
            btn.setCheckable(True)
            btn.setToolTip(cls.title)
            btn.clicked.connect(
                lambda _=False, pid=cls.page_id: self.page_selected.emit(pid))
            self._buttons[cls.page_id] = btn
            self._group.addButton(btn)
            self._layout.insertWidget(self._layout.count() - 1, btn)

    def select(self, page_id: str) -> None:
        btn = self._buttons.get(page_id)
        if btn is not None:
            btn.setChecked(True)
            self.page_selected.emit(page_id)
```

```python
# ui/pages/__init__.py
# 页面注册表：新增页面 = 在此追加一行
PAGES: list = []  # [Txt2ImgPage, Img2ImgPage, VideoPage, GalleryPage] 后续任务逐个加入
```

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_nav.py`
Expected: `PASS test_nav`

- [ ] **Step 5: Commit**

```bash
git add ui/nav.py ui/pages/__init__.py ui/pages/base.py tests/test_nav.py
git commit -m "feat(ui): NavRail 导航栏 + PageBase 页面基类 + PAGES 注册表"
```

---

### Task 5: 外壳 shell.py 骨架 + 新旧 UI 开关

**Files:**
- Create: `ui/shell.py`
- Modify: `main.py:76`（mixin 列表）与 `main.py:117`（`self.setup_ui()` 调用处）
- Test: `tests/test_ui_contract.py`（扩展：整窗契约自检）

**Interfaces:**
- Consumes: Task 1 `apply_theme`；Task 2 `install_aliases/check_contract/apply_degradation`；Task 4 `NavRail/PAGES`
- Produces: `ShellMixin` — 提供 `setup_ui()`（与 `UIBuilderMixin` 同签名，可替换）；方法契约 `append_log(text, color)` / `set_status(text, color)` / `set_progress(value)` / `play_video(path)` 的初版（迁移完成前先委托到占位实现）；属性 `nav`、`center_stack`、`params_stack`、`filmstrip`、`status_bar_widget`

- [ ] **Step 1: 扩展契约测试（失败）**

在 `tests/test_ui_contract.py` 追加：

```python
def test_shell_skeleton():
    """外壳骨架：无模型加载，最小宿主验证 setup_ui 可跑通。"""
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    assert win.nav is not None and win.center_stack is not None
    assert callable(win.append_log) and callable(win.set_status)
    assert callable(win.set_progress) and callable(win.play_video)
    win.close()
    print("PASS test_shell_skeleton")
```

并在 `__main__` 块加 `test_shell_skeleton()` 调用。

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.shell'`

- [ ] **Step 3: 实现 shell.py 骨架**

```python
# ui/shell.py
# ============================================================
#  新主窗口外壳 v6.0 — NavRail + 中央Stacked + 右侧参数面板
#  生成核心控件全局单例；页面专属区随导航切换
# ============================================================
import logging
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
                             QStackedWidget, QProgressBar, QLabel, QStatusBar,
                             QScrollArea)
from PyQt6.QtCore import Qt

from ui.theme import PALETTE
from ui.nav import NavRail
from ui.pages import PAGES
from ui.contracts import install_aliases, check_contract, apply_degradation

logger = logging.getLogger(__name__)


class ShellMixin:
    """替代 UIBuilderMixin 的 setup_ui 实现。业务 mixin 零改动。"""

    def setup_ui(self):
        self.setMinimumSize(1320, 820)
        self.setWindowTitle("AI 绘画工作站 v6.0")
        # 主题由 main.py 的 apply_theme(QApplication) 统一应用

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── 主体三栏 ──
        body = QHBoxLayout()
        body.setSpacing(6)

        self.nav = NavRail()
        body.addWidget(self.nav)

        self.center_stack = QStackedWidget()
        body.addWidget(self.center_stack, 1)

        # 右侧参数面板：核心区(常驻) + 专属区(切换) + 折叠分组 + 生成按钮
        self.params_panel = self._build_params_panel()
        body.addWidget(self.params_panel)

        root.addLayout(body, 1)

        # ── 底部胶片条（Task 11 接入真实组件，先占位）──
        self.filmstrip = QLabel("胶片条占位")
        self.filmstrip.setFixedHeight(110)
        root.addWidget(self.filmstrip)

        # ── 页面注册与切换 ──
        self._pages: dict = {}
        self.nav.set_pages(PAGES)
        for cls in PAGES:
            page = cls()
            try:
                page.build(self)              # 页面专属控件挂到 self
            except Exception:
                logger.exception(f"❌ 页面构建失败: {cls.page_id}")
                self.center_stack.addWidget(QLabel(f"页面 {cls.title} 加载失败"))
                continue
            self._pages[cls.page_id] = page
            self.center_stack.addWidget(page.workspace())
            pw = page.params_widget()
            if pw is not None:
                self.params_stack.addWidget(pw)
        self.nav.page_selected.connect(self._on_page_selected)
        if PAGES:
            self.nav.select(PAGES[0].page_id)

        # ── 契约自检 + 分级降级 ──
        install_aliases(self)
        crit, minor = check_contract(self)
        if minor:
            logger.warning(f"⚠️ 契约自检（非关键缺失）: {minor}")
        apply_degradation(self, crit)

        self._build_statusbar_v6()

    # ---------- 页面切换 ----------
    def _on_page_selected(self, page_id: str):
        page = self._pages.get(page_id)
        if page is None:
            return
        self.center_stack.setCurrentWidget(page.workspace())
        pw = page.params_widget()
        if pw is not None:
            self.params_stack.setCurrentWidget(pw)
        # 只切换/隐藏"页面专属区"；生成核心区与生成按钮永远常驻
        self.params_scroll.setVisible(pw is not None)

    # ---------- 右侧面板（本任务先骨架，Task 6 填核心区）----------
    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(360)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.core_area = QVBoxLayout()       # Task 6 填充生成核心区
        lay.addLayout(self.core_area)
        self.params_stack = QStackedWidget() # 页面专属区
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setWidget(self.params_stack)
        lay.addWidget(self.params_scroll, 1)
        self.shared_groups = QVBoxLayout()   # Task 9 填充共享折叠分组
        lay.addLayout(self.shared_groups)
        self.gen_area = QVBoxLayout()        # Task 6 填充生成/停止按钮
        lay.addLayout(self.gen_area)
        return w

    # ---------- 状态栏 ----------
    def _build_statusbar_v6(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = getattr(self, "lbl_status", None) or QLabel("就绪")
        self.progress_gen = getattr(self, "progress_gen", None) or QProgressBar()
        self.progress_gen.setMaximumWidth(220)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.progress_gen)
        sb.showMessage("AI 绘画工作站 v6.0 已就绪")

    # ---------- 方法契约（Task 6/10 换成完整实现）----------
    def append_log(self, text: str, color: str = "#dfe5ec"):
        logger.info(text)

    def set_status(self, text: str, color: str = "#dfe5ec"):
        if getattr(self, "lbl_status", None) is not None:
            self.lbl_status.setText(text)

    def set_progress(self, value: int):
        if getattr(self, "progress_gen", None) is not None:
            self.progress_gen.setValue(value)

    def play_video(self, video_path: str):
        logger.info(f"play_video 占位: {video_path}")  # Task 10 实现
```

- [ ] **Step 4: main.py 接入新旧开关**

修改 `main.py`（仅两处，旧路径完整保留）：

```python
# main.py:51 附近，import 区
from ui.ui_builder import UIBuilderMixin
import os as _os
if _os.environ.get("AI_STUDIO_UI") == "v2":
    from ui.shell import ShellMixin as _UIMixin
else:
    _UIMixin = UIBuilderMixin
```

```python
# main.py:76 类定义处
class AIDesktopApp(QMainWindow, _UIMixin, EventMixin, GenerationMixin,
                   PresetManagerMixin, TooltipMixin, VideoPanelMixin):
```

并在 `main.py` 的 `QApplication` 创建后、窗口创建前加：

```python
if os.environ.get("AI_STUDIO_UI") == "v2":
    from ui.theme import apply_theme
    logger.info(f"🎨 主题: {apply_theme(app)}")
```

（旧路径继续用 `DARK_STYLE`，不动。）

- [ ] **Step 5: 运行测试确认通过**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_contract_lists / test_check_and_degrade` + `PASS test_shell_skeleton`

- [ ] **Step 6: 冒烟验证旧 UI 不受影响**

Run: `venv/Scripts/python.exe -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; import main; print('import main OK')"`
Expected: `import main OK`（无 AI_STUDIO_UI 环境变量时仍走旧 UI）

- [ ] **Step 7: Commit**

```bash
git add ui/shell.py main.py tests/test_ui_contract.py
git commit -m "feat(ui): shell.py 外壳骨架 + AI_STUDIO_UI=v2 新旧开关（默认旧）"
```

---

### Task 6: 生成核心区 core_panel.py（全局单例，契约成败所在）

**Files:**
- Create: `ui/core_panel.py`
- Modify: `ui/shell.py`（`_build_params_panel` 内调用 build_core / build_gen_area）
- Test: `tests/test_ui_contract.py`（扩展：关键控件断言）

**Interfaces:**
- Consumes: 旧 `ui/ui_builder.py` 的 `_build_tab_basic`（172-471 行）、`_build_gen_button_area`（1817-1851 行）、`_build_status_bar_widget`（1851-1872 行）
- Produces: `build_core(host, layout: QVBoxLayout) -> None`（把生成核心控件按原属性名挂到 host）；`build_gen_area(host, layout: QVBoxLayout) -> None`（生成/停止按钮 + 进度 + 状态）

- [ ] **Step 1: 扩展契约测试（失败）**

在 `tests/test_ui_contract.py` 追加：

```python
def test_core_widgets():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow
    from ui.contracts import check_contract, GLOBAL_WIDGETS

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    crit, minor = check_contract(win)
    for name in ("txt_prompt", "txt_neg", "combo_model", "combo_sampler",
                 "spin_steps", "spin_width", "spin_height", "btn_generate",
                 "btn_interrupt", "progress_gen"):
        assert name not in crit, f"关键控件缺失: {name}"
        assert getattr(win, name) is not None
    # 别名指向同一实例
    assert win.btn_gen is win.btn_generate
    assert win.preview_canvas is win.lbl_preview
    win.close()
    print("PASS test_core_widgets")
```

`__main__` 块加 `test_core_widgets()`。

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: FAIL — `AssertionError: 关键控件缺失: txt_prompt`

- [ ] **Step 3: 迁移生成核心控件**

创建 `ui/core_panel.py`，采用**搬移+重组**而非重写：

```python
# ui/core_panel.py
# ============================================================
#  生成核心区（全局单例）— 从 ui_builder._build_tab_basic 迁出
#  硬约束：所有属性名与原代码完全一致；页面禁止重建同名控件
# ============================================================

def build_core(host, layout) -> None:
    """把 _build_tab_basic 中的核心控件构建代码原样迁入。
    迁移规则（机械执行）：
    1. 从 ui/ui_builder.py 的 _build_tab_basic（172-471 行）中，把创建
       以下控件的语句块逐块复制到此函数，self → host：
       combo_model, combo_model_type, lbl_model_info, txt_prompt, txt_neg,
       btn_enhance_prompt, btn_vision_prompt, chk_auto_enhance,
       chk_auto_features, combo_res, spin_width, spin_height, spin_steps,
       combo_sampler, spin_seed, spin_count, combo_img_format, combo_device,
       btn_preset_menu, btn_save_preset, btn_restore_preset,
       combo_preset, lbl_preset_badge
    2. 删除 tab 外壳（QTabWidget/QScrollArea 包装），控件直接 add 进 layout
    3. 删除所有硬编码 setStyleSheet，改用 setProperty("role", ...)：
       - 提示性 QLabel → setProperty("role", "hint")
       - 标题性 QLabel → setProperty("role", "title")
    4. 信号连接（clicked/textChanged 等）原样保留
    5. 逐块搬移后对照 GLOBAL_WIDGETS 清单核对，禁止改名
    """
    raise NotImplementedError  # 实现时按 docstring 规则搬移
```

`build_gen_area` 同样从 `_build_gen_button_area`（1817-1851 行）迁入 `btn_generate`/`btn_interrupt`，`btn_generate` 打 `setProperty("kind", "primary")`；从 `_build_status_bar_widget`（1851-1872 行）迁入 `lbl_status`/`progress_gen`。

修改 `ui/shell.py` 的 `_build_params_panel`，在 `self.core_area` 创建后调用：

```python
        from ui.core_panel import build_core, build_gen_area
        build_core(self, self.core_area)
        ...
        build_gen_area(self, self.gen_area)
```

同时把 `_build_statusbar_v6` 里的 `getattr(... ) or QLabel(...)` 兜底保留（core 构建在先，会复用真控件）。

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_core_widgets`

- [ ] **Step 5: Commit**

```bash
git add ui/core_panel.py ui/shell.py tests/test_ui_contract.py
git commit -m "feat(ui): 生成核心区全局单例迁移（_build_tab_basic → core_panel）"
```

---

### Task 7: 文生图页（中央预览区迁移）

**Files:**
- Create: `ui/pages/txt2img_page.py`
- Modify: `ui/pages/__init__.py`（注册 `Txt2ImgPage`）
- Modify: `ui/shell.py`（`append_log` 完整实现，写 `txt_log_image`）
- Test: `tests/test_ui_contract.py`（扩展：页面切换断言）

**Interfaces:**
- Consumes: Task 4 `PageBase`；Task 5 `ShellMixin`；旧 `_build_right_panel`（1509-1606 行）
- Produces: `Txt2ImgPage(PageBase)`，`page_id="txt2img"`, `title="文生图"`, `icon="🎨"`；`workspace()` 内含 `host.lbl_preview`（GpuCanvas）、4 操作按钮行、`host.gallery` 暂留旧位（Task 11 再迁）、`host.txt_log_image`

- [ ] **Step 1: 扩展契约测试（失败）**

```python
def test_txt2img_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    assert "txt2img" in win._pages
    assert win.lbl_preview is not None
    assert win.txt_log_image is not None
    win.append_log("hello")            # 方法契约：写入 txt_log_image
    assert "hello" in win.txt_log_image.toPlainText()
    win.nav.select("txt2img")
    assert win.center_stack.currentWidget() is win._pages["txt2img"].workspace()
    win.close()
    print("PASS test_txt2img_page")
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: FAIL — `KeyError: 'txt2img'` 或断言失败

- [ ] **Step 3: 实现 txt2img_page.py**

```python
# ui/pages/txt2img_page.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QLabel, QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt
from ui.pages.base import PageBase
from ui.widgets import GpuCanvas  # 复用 ui/widgets.py 现有 GpuCanvas
from ui.gallery_panel import GalleryPanel


class Txt2ImgPage(PageBase):
    page_id, title, icon = "txt2img", "文生图", "🎨"

    def build(self, host):
        """中央工作区：预览画布 + 4 操作按钮 + 画廊（暂）+ 日志。
        从 ui_builder._build_right_panel（1509-1606 行）迁入，
        删除硬编码 setStyleSheet，属性名不变。"""
        w = QWidget()
        layout = QVBoxLayout(w)
        # ... 逐块迁入：lbl_preview(GpuCanvas)、btn_open_editor/btn_save_as/
        #     btn_send_img2img/btn_send_inpaint（信号连接原样保留）、
        #     gallery（GalleryPanel，信号连接原样）、txt_log_image
        self._workspace = w
        self._host = host

    def workspace(self) -> QWidget:
        return self._workspace

    def params_widget(self):
        return None  # 文生图无专属参数，核心区已覆盖
```

注意：`GpuCanvas` 在现有 `ui/widgets.py` 中，import 路径以实际为准；`lbl_preview.setText("等待生成...")` 等初始化语句一并迁入。

同时在 `ui/pages/__init__.py`：

```python
from ui.pages.txt2img_page import Txt2ImgPage
PAGES = [Txt2ImgPage]
```

`ui/shell.py` 的 `append_log` 换成完整实现（从旧 `append_log`（2457 行起）迁入：写 `self.txt_log_image`，带颜色 span；动画模式下写 `txt_log_video` 的分支保留，用 `getattr` 防御）。

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: 全部 PASS（含 `PASS test_txt2img_page`）

- [ ] **Step 5: Commit**

```bash
git add ui/pages/txt2img_page.py ui/pages/__init__.py ui/shell.py tests/test_ui_contract.py
git commit -m "feat(ui): 文生图页——中央预览区迁移 + append_log 完整实现"
```

---

### Task 8: 图生图页（专属区：参考图/蒙版/强度）

**Files:**
- Create: `ui/pages/img2img_page.py`
- Modify: `ui/pages/__init__.py`（追加注册）
- Test: `tests/test_ui_contract.py`（扩展）

**Interfaces:**
- Consumes: Task 4 `PageBase`；旧 `_build_tab_img2img`（995-1177 行）
- Produces: `Img2ImgPage(PageBase)`，`page_id="img2img"`；`params_widget()` 返回专属区（`btn_load_img`、`btn_clear_img`、`lbl_img_path`、`lbl_ref_thumb`、`scale_strength`、`lbl_ref_fidelity`、`scale_ref_fidelity`、`chk_reference_only` 等 i2i 控件）；`workspace()` 复用文生图同款预览（独立实例但挂不同属性名——若与 txt2img 共享 `lbl_preview`，则 `workspace()` 返回共享预览容器，禁止新建 `lbl_preview`）

- [ ] **Step 1: 扩展契约测试（失败）**

```python
def test_img2img_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    assert "img2img" in win._pages
    assert win.btn_load_img is not None
    assert win.scale_strength is not None
    assert win.scale_str is win.scale_strength   # 别名指向真控件
    win.nav.select("img2img")
    assert win.params_stack.currentWidget() is win._pages["img2img"].params_widget()
    win.close()
    print("PASS test_img2img_page")
```

- [ ] **Step 2: 运行确认失败 → Step 3: 迁移 `_build_tab_img2img` 专属控件 → Step 4: 通过**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_img2img_page`

- [ ] **Step 5: Commit**

```bash
git add ui/pages/img2img_page.py ui/pages/__init__.py tests/test_ui_contract.py
git commit -m "feat(ui): 图生图页——i2i 专属参数区迁移"
```

---

### Task 9: 共享折叠分组（LoRA / ControlNet / 高级 / X-Y）

**Files:**
- Create: `ui/shared_groups.py`
- Modify: `ui/shell.py`（`shared_groups` 布局填充；动画页时隐藏分组区）
- Test: `tests/test_ui_contract.py`（扩展）

**Interfaces:**
- Consumes: Task 3 `CollapsibleSection`；旧 `_build_tab_lora`（1177-1243 行）、`_build_tab_ctrl`（1243-1304 行）、`_build_tab_advanced`（1304-1468 行）、`_build_tab_xy`（1468-1509 行）
- Produces: `build_shared_groups(host, layout: QVBoxLayout) -> dict[str, CollapsibleSection]` — 4 个折叠分组，默认全部折叠；控件按原属性名挂 host

- [ ] **Step 1: 扩展契约测试（失败）**

```python
def test_shared_groups():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow
    from ui.contracts import check_contract

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    crit, minor = check_contract(win)
    assert crit == [], f"关键契约缺失: {crit}"
    for name in ("combo_lora_0", "scale_lora_0", "combo_cn_type",
                 "chk_enable_hires", "chk_enable_xy", "entry_x_vals"):
        assert getattr(win, name) is not None, name
    assert len(win.combo_loras) == 3        # 列表别名
    win.close()
    print("PASS test_shared_groups")
```

- [ ] **Step 2: 运行确认失败 → Step 3: 迁移四个 tab 的控件创建代码到 `build_shared_groups`，每组装进一个 `CollapsibleSection(标题, collapsed=True)`；`_toggle_adetailer`/`_toggle_hires`/`_toggle_cn`/`_toggle_xy` 等切换方法一并迁入 shell 或保留委托 → Step 4: 通过**

注意：这些 `_toggle_*` 方法被旧 builder 定义且被信号连接；迁移时把方法定义移到 `ShellMixin`，信号连接指向不变。

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_shared_groups`

- [ ] **Step 5: Commit**

```bash
git add ui/shared_groups.py ui/shell.py tests/test_ui_contract.py
git commit -m "feat(ui): 共享折叠分组——LoRA/ControlNet/高级/X-Y 迁移"
```

---

### Task 10: 动画页（专属参数 + 视频预览）

**Files:**
- Create: `ui/pages/video_page.py`
- Modify: `ui/pages/__init__.py`（追加注册）
- Modify: `ui/shell.py`（`play_video`/`pause`/`stop` 完整实现，从旧方法迁入）
- Test: `tests/test_ui_contract.py`（扩展）

**Interfaces:**
- Consumes: Task 4 `PageBase`；旧 `_build_tab_animation`（471-995 行）、`_build_video_right_panel`（1608-1720 行）、`play_video`/`stop_video`/`pause_video`（1744-1817 行）、`_on_video_mode_changed` 等视频辅助方法（2476-2576 行）
- Produces: `VideoPage(PageBase)`，`page_id="video"`；`params_widget()` = 动画参数组（`PAGE_WIDGETS["video"]` 清单全部控件）；`workspace()` = 视频预览（`video_player`/`video_widget`/`lbl_video_status` 等）

- [ ] **Step 1: 扩展契约测试（失败）**

```python
def test_video_page():
    from ui.shell import ShellMixin
    from PyQt6.QtWidgets import QMainWindow
    from ui.contracts import PAGE_WIDGETS

    class MiniApp(QMainWindow, ShellMixin):
        pass

    win = MiniApp()
    win.setup_ui()
    assert "video" in win._pages
    for name in ("btn_gen_video", "txt_video_prompt", "combo_video_mode",
                 "video_player", "txt_log_video"):
        assert getattr(win, name) is not None, name
    win.nav.select("video")
    assert win.params_stack.currentWidget() is win._pages["video"].params_widget()
    win.close()
    print("PASS test_video_page")
```

- [ ] **Step 2-4: 失败 → 迁移 → 通过**

迁移要点：`_build_tab_animation` 的 `VIDEO_TAB_QSS` 硬编码样式删除，改用语义 property；`play_video`/`stop_video`/`pause_video` 迁入 `ShellMixin`（替换 Task 5 的占位实现）；旧的 `_switch_to_video_mode/_switch_to_image_mode` 废弃——选页即选模式，`nav.page_selected` 里直接做模式标志切换（`self._video_mode = page_id == "video"`），供 `app_generation.py` 查询处兼容（保留 `right_stacked` 同名属性指向中央 stack 以减少业务层改动；如业务层有其他依赖，用别名兜底并记录进 `contracts.LIST_ALIASES`）。

Run: `venv/Scripts/python.exe tests/test_ui_contract.py`
Expected: `PASS test_video_page`

- [ ] **Step 5: Commit**

```bash
git add ui/pages/video_page.py ui/pages/__init__.py ui/shell.py tests/test_ui_contract.py
git commit -m "feat(ui): 动画页——参数组+视频预览迁移，选页即选模式"
```

---

### Task 11: 统一画廊页 + 胶片条 + 防抖防洪

**Files:**
- Modify: `ui/gallery_panel.py`（媒体类型维度 + 防抖，契约签名不变）
- Create: `ui/pages/gallery_page.py`
- Create: `ui/widgets/filmstrip.py`
- Modify: `ui/pages/__init__.py`（追加注册）
- Modify: `ui/shell.py`（胶片条占位换成真组件）
- Create: `scripts/smoke_ui.py`
- Test: `tests/test_gallery_unified.py`

**Interfaces:**
- Consumes: Task 4 `PageBase`；现有 `GalleryPanel`；`utils/paths.py` 的 `OUTPUT_DIR`（`photo/`）、`VIDEO_DIR`（`photo/videos/`）
- Produces:
  - `GalleryPanel.set_media_filter(mode: str) -> None`（`"image"/"video"/"all"`）— 新增方法，旧方法签名不变
  - `GalleryPanel.add_media(path: str, prepend: bool = False) -> None` — 图片/动画统一入口；`add_image` 保留为 `add_media` 的兼容包装
  - `GalleryPage(PageBase)`，`page_id="gallery"`
  - `FilmStrip(QWidget)`：`refresh(paths: list[str]) -> None`；`media_clicked = pyqtSignal(str)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gallery_unified.py
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication([])
    from ui.gallery_panel import GalleryPanel
    from ui.widgets.filmstrip import FilmStrip

    g = GalleryPanel()
    # 媒体过滤
    g.set_media_filter("video")
    assert g._media_filter == "video"
    g.set_media_filter("all")
    # 防抖：连续 add_media 不立刻刷 N 次
    refresh_calls = []
    orig = g._apply_filter
    g._apply_filter = lambda: (refresh_calls.append(1), orig())
    for i in range(20):                       # 模拟 X-Y 矩阵批量出图
        g.add_media(f"photo/fake_{i}.png", prepend=True)
    assert len(refresh_calls) <= 1, f"防抖失效: {len(refresh_calls)} 次刷新"
    time.sleep(0.3)
    app.processEvents()
    assert len(refresh_calls) <= 2            # 200ms 后最多补一次合并刷新

    fs = FilmStrip()
    fs.refresh(["photo/a.png", "photo/videos/b.mp4"])
    got = []
    fs.media_clicked.connect(got.append)
    print("PASS test_gallery_unified")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行确认失败**

Run: `venv/Scripts/python.exe tests/test_gallery_unified.py`
Expected: FAIL — `AttributeError: 'GalleryPanel' object has no attribute 'set_media_filter'`

- [ ] **Step 3: 实现统一画廊**

`ui/gallery_panel.py` 修改（契约不破）：

```python
# GalleryPanel.__init__ 追加：
        self._media_filter = "all"
        self._refresh_timer = QTimer(self)     # 200ms 防抖合并
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self._apply_filter)

# 新增方法：
    IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    VIDEO_EXT = {".mp4", ".gif", ".webm", ".mov"}

    def set_media_filter(self, mode: str):
        assert mode in ("image", "video", "all")
        self._media_filter = mode
        self._apply_filter()

    def add_media(self, path: str, prepend: bool = False):
        """统一入口：图片/动画都可。防抖合并刷新。"""
        self._all_items.insert(0 if prepend else len(self._all_items), path)
        self._refresh_timer.start()            # 合并 200ms 内的批量调用

    def add_image(self, path: str, prepend: bool = False):   # 兼容包装
        self.add_media(path, prepend=prepend)

    def reload_from_dir(self, directory: str, limit: int = 80):
        """扩展：同时扫描 VIDEO_DIR；过滤逻辑加媒体类型维度。"""
        # 在原实现基础上：收集 IMAGE_EXT ∪ VIDEO_EXT 文件；
        # _apply_filter 增加一层 self._media_filter 过滤；
        # 动画条目缩略图右上角画播放角标（QPixmap 合成小三角）
```

`ui/pages/gallery_page.py`：

```python
# 顶部工具条：[图片|动画|全部] QButtonGroup + GalleryPanel 现有搜索行整体复用
# workspace() = 工具条 + GalleryPanel 网格
#   注意：self.gallery 本体在 Task 7 已创建于文生图页工作区，本任务将其
#   setParent 重挂到画廊页（同一实例，契约不破），文生图页原位置移除
# params_widget() = 内嵌的 MetadataPanel（画廊页时右侧专属区显示元数据/大图，
#   核心区仍按全局约束常驻；现有浮窗式 meta_panel 改为内嵌，信号不变）
# 选中动画 → 双击用 host.play_video(path) 播放
```

`ui/widgets/filmstrip.py`：

```python
# FilmStrip(QWidget)：横向 QListWidget(IconMode)，高 110px
# refresh(paths)：重建缩略图（图片直读，视频读首帧——imageio/ffmpeg 已有依赖，
#   失败则显示占位图标）
# media_clicked(str)：点击发射路径，shell 接到后 nav.select("gallery") + 选中
```

`scripts/smoke_ui.py`（跳过模型加载的冒烟入口）：

```python
# scripts/smoke_ui.py — v2 GUI 冒烟：不加载模型，直接起壳
import os, sys
os.environ["AI_STUDIO_UI"] = "v2"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QApplication
from ui.theme import apply_theme
from ui.shell import ShellMixin
from PyQt6.QtWidgets import QMainWindow

class SmokeApp(QMainWindow, ShellMixin):
    pass

app = QApplication(sys.argv)
print("主题:", apply_theme(app))
win = SmokeApp()
win.setup_ui()
win.show()
sys.exit(app.exec())
```

- [ ] **Step 4: 运行确认通过**

Run: `venv/Scripts/python.exe tests/test_gallery_unified.py`
Expected: `PASS test_gallery_unified`

- [ ] **Step 5: 人工冒烟**

Run: `venv/Scripts/python.exe scripts/smoke_ui.py`
走查清单：四页切换 / 画廊三态过滤 / 搜索+收藏+废过滤叠加 / 胶片条点击跳画廊 / 折叠分组展开收起。

- [ ] **Step 6: Commit**

```bash
git add ui/gallery_panel.py ui/pages/gallery_page.py ui/widgets/filmstrip.py ui/pages/__init__.py ui/shell.py scripts/smoke_ui.py tests/test_gallery_unified.py
git commit -m "feat(ui): 统一画廊（图片/动画三态过滤）+ 胶片条 + 200ms 防抖防洪"
```

---

### Task 12: 样式清场（硬编码 setStyleSheet → 主题色板）

**Files:**
- Modify: `ui/gallery_panel.py`（57/59/68/133/137/142/149/154/161/169 等行）
- Modify: `ui/extension_market.py`（64-302 行各处）
- Modify: `ui/disclaimer.py`（136 行）
- Modify: `ui/splash.py`
- Test: 无新测试；靠契约测试回归 + 冒烟目检

- [ ] **Step 1: 逐文件替换规则**

1. 删除所有内联 `setStyleSheet` 中的硬编码色值（`#1e1e2e`/`#cdd6f4`/`#a6e3a1`/`#313244` 等 catppuccin 色、`#0a0a0a`/`#212327` 等旧 xAI 色）
2. 需要强调/弱化的 QLabel 改用 `setProperty("role", "title"|"hint"|"value")`
3. 组件特有微调（如进度条高度）保留但色值改引 `ui.theme.PALETTE`：
   `from ui.theme import PALETTE` → `f"color:{PALETTE['fg_mute']};"`
4. `ui/ui_builder.py` 本任务**不动**（Task 14 整体删除）

- [ ] **Step 2: 回归验证**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py && venv/Scripts/python.exe tests/test_gallery_unified.py`
Expected: 全部 PASS

- [ ] **Step 3: 人工冒烟目检**

Run: `venv/Scripts/python.exe scripts/smoke_ui.py`
走查：画廊、扩展市场对话框（菜单打开）、免责声明——风格统一为深蓝灰。

- [ ] **Step 4: Commit**

```bash
git add ui/gallery_panel.py ui/extension_market.py ui/disclaimer.py ui/splash.py
git commit -m "refactor(ui): 清场硬编码样式，统一走 theme.PALETTE 与语义类"
```

---

### Task 13: 版本号 v5.0 → v6.0

**Files:**
- Modify: `main.py:3`（注释）、`main.py:723`（`setApplicationVersion`）
- Modify: `ui/disclaimer.py:38`、`ui/splash.py:69`
- Modify: `README.md:1,7,458`、`项目说明.txt:1,7,456`

（`ui/ui_builder.py` 内的三处 v5.0 不单独改——Task 14 整体删除；shell.py 窗口标题已是 v6.0。）

- [ ] **Step 1: 批量替换并核对**

规则：仅替换版本字符串 `"v5.0"` → `"v6.0"`、`"5.0"` → `"6.0"`（仅限 `setApplicationVersion("5.0")` 一处）；参数值（如 `cfg: 5.0`、`size_gb > 5.0`）一律不动。

- [ ] **Step 2: 验证无遗漏**

Run: `grep -rn "v5\.0" --include="*.py" --include="*.md" --include="*.txt" . | grep -v ui_builder | grep -v .venv`
Expected: 无输出（除 git 历史与本文档）

- [ ] **Step 3: Commit**

```bash
git add main.py ui/disclaimer.py ui/splash.py README.md 项目说明.txt
git commit -m "chore: 版本号 v5.0 → v6.0"
```

---

### Task 14: 切换默认 + 删除旧 ui_builder.py

**Files:**
- Modify: `main.py`（移除 `AI_STUDIO_UI` 开关与 `_UIMixin` 别名，固定用 `ShellMixin`；旧 `UIBuilderMixin` import 删除）
- Delete: `ui/ui_builder.py`
- Modify: `ui/design_tokens.py`（标注 deprecated 或删除，确认无引用后删除）
- Test: 全量回归

- [ ] **Step 1: 确认零引用**

Run: `grep -rn "ui_builder\|UIBuilderMixin\|DARK_STYLE\|design_tokens" --include="*.py" . | grep -v ".venv" | grep -v "docs/"`
Expected: 仅 main.py 中的待删引用；`DARK_STYLE`/`design_tokens` 无活引用

- [ ] **Step 2: 切换并删除**

`main.py`：

```python
# 删除 Task 5 的开关块，固定为：
from ui.shell import ShellMixin

class AIDesktopApp(QMainWindow, ShellMixin, EventMixin, GenerationMixin,
                   PresetManagerMixin, TooltipMixin, VideoPanelMixin):
```

`apply_theme(app)` 调用改为无条件执行。删除 `ui/ui_builder.py`，`design_tokens.py` 若无引用一并删除。

- [ ] **Step 3: 全量回归**

Run: `venv/Scripts/python.exe tests/test_ui_contract.py && venv/Scripts/python.exe tests/test_gallery_unified.py && venv/Scripts/python.exe tests/test_theme.py && venv/Scripts/python.exe tests/test_nav.py && venv/Scripts/python.exe tests/test_collapsible.py`
Expected: 全部 PASS

Run: `venv/Scripts/python.exe scripts/smoke_ui.py`
Expected: 人工走查四页 + 画廊三态 + 胶片条 + 折叠分组全部可用；契约自检无关键缺失（日志无 ❌）

- [ ] **Step 4: 真实启动验证（含模型加载路径）**

Run: `venv/Scripts/python.exe main.py`
Expected: 正常启动到主界面，四页可切换，生成按钮可用（若本机模型环境完整则跑一次最小生成确认链路通）

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(ui): v6.0 切换默认 UI，删除旧 ui_builder.py（2626 行单体退役）"
```

---

## 风险与回滚

- 全程 `AI_STUDIO_UI=v2` 开关隔离新旧 UI，Task 14 前旧路径随时可用
- 每任务独立 commit，任一阶段 `git revert` 可回滚
- 契约测试是迁移正确性的主防线；冒烟脚本覆盖视觉/交互回归
- 最大风险点：Task 6 控件搬移漏属性 → 由契约测试的 `GLOBAL_WIDGETS` 全清单兜底；Task 10 视频页模式切换语义变化 → `right_stacked` 同名别名兜底
