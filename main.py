# main.py
# ============================================================
#  PyQt6 主入口 — AI 绘画工作站 v5.0
# ============================================================

import os
import sys
import threading
import warnings
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ GPU 加速必须在 QApplication 创建之前完成
from utils.ui_builder import enable_gpu_acceleration
enable_gpu_acceleration()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QFileDialog
)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtGui  import QCloseEvent, QIcon, QPixmap

from PIL import Image

from translation_service  import TranslationService
from config_manager       import AppConfig
from utils.ui_builder     import (
    UIBuilderMixin, FloatSlider, create_splash, DARK_STYLE
)
from utils.app_events     import EventMixin
from utils.app_generation import GenerationMixin
from utils.app_utils      import PROMPT_PRESETS, OUTPUT_DIR as APP_OUTPUT_DIR
from utils.system_utils   import log_system_info, logger


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "photo")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

torch = None  # 延迟导入


# ============================================================
#  AI 加载阶段专用信号桥
# ============================================================
class _AppBridge(QObject):
    ai_loaded  = pyqtSignal()
    status_msg = pyqtSignal(str, str)


# ============================================================
#  主窗口
# ============================================================
class AIDesktopApp(QMainWindow, UIBuilderMixin, EventMixin, GenerationMixin):

    def __init__(self):
        super().__init__()

        log_system_info()
        logger.info("🚀 AI 绘画工作站启动...")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ── 基础状态 ──────────────────────────────────────
        self.translator             = TranslationService()
        self.ai                     = None
        self.is_generating          = False
        self.cancel_flag            = False
        self.ref_image_path         = None
        self.mask_image_path        = None
        self.pose_image_path        = None
        self.current_generated_path = None
        self.last_generated_path    = None
        self._editor_window         = None  # 防止 GC
        self.cleanup_temp_files(verbose=False)

        # ⭐ 关键修复 1：在 setup_ui 之前注入预设字典
        self._prompt_presets = PROMPT_PRESETS

        # ── App 信号桥 ─────────────────────────────────────
        self._app_bridge = _AppBridge()
        self._app_bridge.ai_loaded.connect(self._on_ai_loaded)
        self._app_bridge.status_msg.connect(self._set_status)

        # ── 配置 & UI ──────────────────────────────────────
        self.config = AppConfig()
        self.config.load()

        self.setup_ui()
        self.apply_config_to_ui()

        # ── 生成信号桥 ─────────────────────────────────────
        self._init_gen_bridge()
        try:
            self._bridge.preview_signal.connect(self._on_new_image_saved)
        except Exception:
            pass

        # ── 初始按钮状态 ───────────────────────────────────
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("🚀 AI 引擎预热中...")
        self._set_status(
            "⏳ 正在后台加载大模型与底层环境，请稍候...", "#f9e2af")

        threading.Thread(
            target=self._async_init_ai, daemon=True).start()
        


    # ----------------------------------------------------------
    def _set_status(self, text: str, color: str = "#89dceb"):
        self.set_status(text, color)

    def _emit_status(self, text: str, color: str = "#89dceb"):
        self._app_bridge.status_msg.emit(text, color)

    # ----------------------------------------------------------
    def _async_init_ai(self):
        logger.info("👉 [预热] 后台导入重型库 (PyTorch / Diffusers)...")
        self._emit_status("⏳ 正在导入 PyTorch & Diffusers...", "#f9e2af")
        try:
            global torch
            import torch
            from model_manager import ModelManager
            self._emit_status(
                "⏳ 正在加载大模型，首次启动可能需要数分钟...", "#f9e2af")
            self.ai = ModelManager()
        except Exception as e:
            logger.error(f"❌ AI 引擎加载失败: {e}")
            self._emit_status(f"❌ 引擎加载失败: {e}", "#f38ba8")
            return
        self._app_bridge.ai_loaded.emit()

    # ----------------------------------------------------------
    def _on_ai_loaded(self):
        if hasattr(self, 'refresh_models'):
            self.refresh_models()
        if hasattr(self, 'refresh_lora_by_model'):
            self.refresh_lora_by_model()

        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("🚀 开始生成")
        self._set_status("✅ 系统就绪，等待生成指令...", "#a6e3a1")
        logger.info("✅ [预热] 引擎就绪！")

        if not hasattr(self, 'combo_device'):
            return

        available_devices = ["自动 (Auto)"]
        if torch is not None and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                available_devices.append(f"CUDA:{i} ({name})")
        if (torch is not None
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            available_devices.append("MPS (Apple Silicon)")
        available_devices.append("CPU (纯内存·极慢)")

        self.combo_device.clear()
        self.combo_device.addItems(available_devices)
        self.combo_device.setEnabled(True)

        pref = getattr(self.config, "device_preference", "自动 (Auto)")
        idx  = self.combo_device.findText(pref)
        self.combo_device.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, 'gallery'):
            self.gallery.reload_from_dir(OUTPUT_DIR, limit=80)
            self.gallery.image_selected.connect(self._on_gallery_pick)

    # ==========================================================
    #  ⭐ 关键修复 2：补全预设/修图/PNG 信息 的方法
    # ==========================================================

    # ---------- 预设提示词 ----------
    def apply_preset(self, idx=None):
        """从 combo_preset 选取预设，并把 p/n 追加到提示词框"""
        try:
            name = self.combo_preset.currentText()
        except Exception:
            return

        if not name or name == "（无）":
            return
        if name not in PROMPT_PRESETS:
            return

        preset = PROMPT_PRESETS[name]
        new_p  = preset.get("p", "")
        new_n  = preset.get("n", "")

        cur_p = self.txt_prompt.toPlainText().strip()
        cur_n = self.txt_neg.toPlainText().strip()

        # 追加方式：避免覆盖用户已写的内容
        merged_p = (new_p + (", " + cur_p if cur_p else "")).strip(", ").strip()
        merged_n = (new_n + (", " + cur_n if cur_n else "")).strip(", ").strip()

        self.txt_prompt.setPlainText(merged_p)
        self.txt_neg.setPlainText(merged_n)

        self._set_status(f"🎯 已套用预设: {name}", "#a6e3a1")
        logger.info(f"🎯 套用提示词预设: {name}")

    # ---------- 读取 PNG 内嵌生成参数 ----------
    def read_png_info(self):
        """从 PNG 文件中读取 SD 风格的 parameters 元数据并回填"""
        start_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else BASE_DIR
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PNG 文件", start_dir, "PNG (*.png)")
        if not path:
            return

        try:
            img = Image.open(path)
            params = (img.info or {}).get("parameters", "")
            if not params:
                QMessageBox.information(
                    self, "提示", "该 PNG 中未找到生成参数信息。")
                return

            # 解析 SD WebUI 风格：
            # 第 1 段 = positive
            # 以 "Negative prompt:" 开头的段 = negative
            # 剩余 = 参数表（暂仅显示）
            pos, neg, info_part = "", "", ""

            if "Negative prompt:" in params:
                pos_part, rest = params.split("Negative prompt:", 1)
                pos = pos_part.strip()

                # rest 的第一行是 negative，剩下的是参数
                rest_lines = rest.strip().split("\n", 1)
                neg = rest_lines[0].strip()
                info_part = rest_lines[1].strip() if len(rest_lines) > 1 else ""
            else:
                # 没有 Negative prompt 字段
                lines = params.split("\n", 1)
                pos = lines[0].strip()
                info_part = lines[1].strip() if len(lines) > 1 else ""

            self.txt_prompt.setPlainText(pos)
            self.txt_neg.setPlainText(neg)

            if hasattr(self, "append_log"):
                self.append_log(f"📥 已读取 PNG 参数: {os.path.basename(path)}",
                                "#a6e3a1")
                if info_part:
                    self.append_log(f"   {info_part}", "#7f849c")

            self._set_status("📥 已回填 PNG 元数据", "#a6e3a1")

            # 顺便把这张图设为当前预览
            try:
                pix = QPixmap(path)
                if hasattr(self, "lbl_preview"):
                    self.lbl_preview.set_pixmap(pix)
                self.last_generated_path    = path
                self.current_generated_path = path
                if hasattr(self, "btn_edit"):    self.btn_edit.setEnabled(True)
                if hasattr(self, "btn_upscale"): self.btn_upscale.setEnabled(True)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))

    # ==========================================================
    #  关闭：保存配置
    # ==========================================================
    def closeEvent(self, event: QCloseEvent):
        logger.info("💾 正在保存配置并退出...")
        try:
            self.config.default_steps    = self.spin_steps.value()
            self.config.default_width    = self.spin_width.value()
            self.config.default_height   = self.spin_height.value()
            self.config.default_batch    = self.spin_batch.value()
            self.config.default_cfg      = self.scale_cfg.float_value()
            self.config.default_strength = self.scale_strength.float_value()
            self.config.default_sampler  = self.combo_sampler.currentText()

            self.config.last_prompt = self.txt_prompt.toPlainText()
            self.config.last_neg    = self.txt_neg.toPlainText()

            self.config.use_adetailer      = self.chk_use_adetailer.isChecked()
            self.config.adetailer_strength = \
                self.scale_adetailer_strength.float_value()
            self.config.use_ad_hand        = self.chk_use_ad_hand.isChecked()
            self.config.ad_hand_strength   = self.scale_ad_hand.float_value()
            self.config.ad_hand_blend      = self.scale_ad_hand_blend.float_value()

            self.config.use_hires      = self.chk_hires.isChecked()
            self.config.hires_denoise  = self.scale_hires_denoise.float_value()

            self.config.output_format = self.combo_img_format.currentText()
            self.config.output_dir    = self.combo_output_dir.currentText()

            if hasattr(self, 'combo_device'):
                self.config.device_preference = self.combo_device.currentText()

            self.config.save()
            logger.info("✅ 配置保存成功。")
        except Exception as e:
            logger.warning(f"⚠️ 配置保存时出错（已忽略）: {e}")

        event.accept()

    def _on_gallery_pick(self, path: str):
        """画廊双击/菜单载入 → 触发预览更新。"""
        # 直接走 EventMixin.show_preview,保持和生成完后一致的行为
        try:
            self.show_preview(path)
        except Exception:
            pass

    def _on_new_image_saved(self, path: str):
        """preview_signal 触发时,额外把图片塞进画廊。"""
        if hasattr(self, 'gallery') and path and os.path.exists(path):
            self.gallery.add_image(path, prepend=True)

    def closeEvent(self, event):
        try:
            self.cleanup_temp_files(verbose=False)
        except Exception:
            pass
        super().closeEvent(event)


# ============================================================
#  程序入口
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("AI 绘画工作站")
    app.setApplicationVersion("5.0")
    app.setStyleSheet(DARK_STYLE)

    ico_path = os.path.join(BASE_DIR, "logo", "dzbut-9fc5g-001.ico")
    if os.path.exists(ico_path):
        app.setWindowIcon(QIcon(ico_path))

    splash = create_splash()
    splash.set_message("正在初始化主界面...")

    window = AIDesktopApp()

    def _show_main():
        window.showMaximized()
        splash.finish_loading(window)

    QTimer.singleShot(500, _show_main)
    sys.exit(app.exec())