# main.py
# ============================================================
#  PyQt6 主入口 — AI 绘画工作站 v5.0
# ============================================================
import warnings
warnings.simplefilter("always", RuntimeWarning)
import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 统一缓存目录
CACHE_ROOT = os.path.join(PROJECT_ROOT, "models_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

os.environ["HF_HOME"]              = os.path.join(CACHE_ROOT, "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "huggingface", "hub")
os.environ["TRANSFORMERS_CACHE"]   = os.path.join(CACHE_ROOT, "huggingface", "transformers")
os.environ["DIFFUSERS_CACHE"]      = os.path.join(CACHE_ROOT, "huggingface", "diffusers")
os.environ["MODELSCOPE_CACHE"]     = os.path.join(CACHE_ROOT, "modelscope")
os.environ["TORCH_HOME"]           = os.path.join(CACHE_ROOT, "torch")
os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from utils.system_utils import setup_logging, log_system_info, logger
setup_logging()
logger.info(f"📦 模型缓存目录: {CACHE_ROOT}")

import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from utils.gpu_init import enable_gpu_acceleration
enable_gpu_acceleration()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QLineEdit, QSpinBox, QPushButton, QLabel,
    QWidget, QGroupBox, QScrollArea, QComboBox, QCheckBox,
    QListWidget, QListWidgetItem, QTextEdit, QDoubleSpinBox
)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt, QSize, pyqtSlot,QMetaObject
from PyQt6.QtGui  import QCloseEvent, QIcon, QPixmap, QImage
from PIL import Image

from core.translation_service  import TranslationService
from core.config_manager       import AppConfig
from ui.ui_builder     import UIBuilderMixin
from ui.splash          import create_splash
from ui.design_tokens   import DARK_STYLE
from utils.app_events     import EventMixin
from utils.app_generation import GenerationMixin
from ui.preset_manager import PresetManagerMixin, TooltipMixin
from ui.video_panel_mixin import VideoPanelMixin
from core.presets import PROMPT_PRESETS
from utils.prompt_enhancer import get_enhancer
from utils.paths import OUTPUT_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch = None  # 延迟导入

# ============================================================
#  AI 加载阶段专用信号桥
# ============================================================
class _AppBridge(QObject):
    ai_loaded  = pyqtSignal()
    ai_failed  = pyqtSignal(str)
    status_msg = pyqtSignal(str, str)


# ============================================================
#  主窗口
# ============================================================
class AIDesktopApp(QMainWindow, UIBuilderMixin, EventMixin, GenerationMixin,
                   PresetManagerMixin, TooltipMixin, VideoPanelMixin):

    def __init__(self):
        super().__init__()
        log_system_info()
        logger.info("🚀 AI 绘画工作站启动...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ── 基础状态 ──
        self.translator = TranslationService()
        self.translator.qwen_enhancer = get_enhancer()
        logger.info("✅ 已将 Qwen 单例注入到翻译服务")
        self.ai = None
        self.is_generating = False
        self.cancel_flag = False
        self.ref_image_path = None
        self.mask_image_path = None
        self.pose_image_path = None
        self.current_generated_path = None
        self.last_generated_path = None
        self.ipa_image_path = None
        self._editor_window = None
        self.video_gen = None
        self.video_generator = None
        self._video_input_path = None
        self.tts_engine = None
        self.travel_segments = []

        self.cleanup_temp_files(verbose=False)
        self._prompt_presets = PROMPT_PRESETS

        # ── App 信号桥 ──
        self._app_bridge = _AppBridge()
        self._app_bridge.ai_loaded.connect(self._on_ai_loaded)
        self._app_bridge.ai_failed.connect(self._on_ai_failed)
        self._app_bridge.status_msg.connect(self._set_status)

        # ── 配置 & UI ──
        self.config = AppConfig.load()

        self.setup_ui()
        self.apply_config_to_ui()
        self.cleanup_temp_files(verbose=False)
        self._refresh_preset_combo()
        self.apply_tooltips()

        self.gallery.reuse_params_signal.connect(self.reuse_params_from_path)
        self.gallery.send_to_i2i_signal.connect(self.send_path_to_img2img)
        self.gallery.send_to_face_signal.connect(self.send_path_to_face_fix)
        self.gallery.send_to_editor_signal.connect(self.send_path_to_editor)
        self._setup_menu_and_statusbar()

        # ── 生成信号桥 ──
        self._init_gen_bridge()
        try:
            self._bridge.preview_signal.connect(self._on_new_image_saved)
            self._bridge.gallery_add_signal.connect(self._on_new_image_saved)
            self._bridge.live_preview_signal.connect(self.show_preview_image)
        except Exception as e:
            logger.error(f"[CONNECT] ❌ 预览信号连接失败: {e}")

        try:
            self.gallery.apply_params_signal.connect(self._on_apply_gallery_params)
        except Exception as e:
            logger.error(f"[CONNECT] ❌ 失败: {e}")

        # enhance_done_signal 已在 _init_gen_bridge() 中连接,此处不再重复连接
        self._ui_ready = True

        # ── 初始按钮状态 ──
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("🚀 AI 引擎预热中...")
        self._set_status("⏳ 正在后台加载大模型与底层环境,请稍候...", "#f9e2af")

        threading.Thread(target=self._async_init_ai, daemon=True).start()

        if hasattr(self, 'combo_model') and self.combo_model.count() > 0:
            self._set_status("🔄 正在加载默认模型...", "#f9e2af")
            QTimer.singleShot(500, self._preload_default_model)

    # ==========================================================
    #  视频生成 → 见 ui/video_panel_mixin.py (VideoPanelMixin)
    # ==========================================================

    # ==========================================================
    #  状态 / 加载
    # ==========================================================
    def _preload_default_model(self):
        """启动后预载默认模型信息（SD pipeline 权重在首次生成时惰性加载）"""
        try:
            if hasattr(self, 'combo_model') and self.combo_model.count() > 0:
                self.on_model_selected(0)   # 刷新模型信息 + LoRA + 备注
                self._set_status("✅ 默认模型就绪（权重将在首次生成时加载）", "#a6e3a1")
        except Exception as e:
            logger.warning(f"[PRELOAD] ⚠️ 默认模型信息加载失败: {e}")

    def _set_status(self, text: str, color: str = "#89dceb"):
        self.set_status(text, color)

    def _emit_status(self, text: str, color: str = "#89dceb"):
        self._app_bridge.status_msg.emit(text, color)

    def _on_ai_failed(self, err: str):
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("❌ 引擎加载失败")
        self._set_status(f"❌ 引擎加载失败: {err}", "#f38ba8")
        QMessageBox.critical(self, "AI 引擎加载失败",
                             f"无法加载 AI 引擎：\n\n{err}")

    def _async_init_ai(self):
        try:
            from utils.model_downloader import print_scan_report
            print_scan_report()
        except Exception as e:
            logger.warning(f"模型扫描失败(忽略): {e}")
        logger.info("👉 [预热] 后台导入重型库 (PyTorch / Diffusers)...")
        self._emit_status("⏳ 正在导入 PyTorch & Diffusers...", "#f9e2af")
        try:
            global torch
            import torch
            from utils.system_utils import log_gpu_info
            log_gpu_info()
            from core.model_manager import ModelManager
            self._emit_status(
                "⏳ 正在加载大模型,首次启动可能需要数分钟...", "#f9e2af")
            self.ai = ModelManager()
        except Exception as e:
            logger.exception("❌ AI 引擎加载失败")   # exception 会带完整 traceback
            self._app_bridge.ai_failed.emit(str(e))
            return
        self._app_bridge.ai_loaded.emit()

    def _on_ai_loaded(self):
        if hasattr(self, 'refresh_models'):
            self.refresh_models()
        if hasattr(self, 'refresh_lora_by_model'):
            self.refresh_lora_by_model()

        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("🚀 开始生成")
        self._set_status("✅ 系统就绪,等待生成指令...", "#a6e3a1")
        logger.info("✅ [预热] 引擎就绪!")
        QMetaObject.invokeMethod(self, "refresh_models", Qt.ConnectionType.QueuedConnection)

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
    #  预设 / PNG 读取
    # ==========================================================
    def apply_preset(self, idx=None):
        name = self._get_current_preset_name()
        if not name or name == "(无)":
            return
        try:
            from core.presets import PROMPT_PRESETS
        except ImportError:
            try:
                from utils.app_utils import PROMPT_PRESETS
            except Exception as e:
                logger.error(f"加载预设字典失败: {e}")
                return

        if name not in PROMPT_PRESETS:
            return
        preset = PROMPT_PRESETS[name]

        before_snap = {k: self._get_widget_value(k)
                       for k in self._CONTROL_LABELS.keys()}
        self._preset_before_snap = before_snap

        if preset.get("p"):
            cur = self.txt_prompt.toPlainText().strip()
            merged = (preset["p"] + (", " + cur if cur else "")).strip(", ")
            self.txt_prompt.setPlainText(merged)
        if preset.get("n"):
            cur = self.txt_neg.toPlainText().strip()
            merged = (preset["n"] + (", " + cur if cur else "")).strip(", ")
            self.txt_neg.setPlainText(merged)

        params = preset.get("params", {})
        if "steps"   in params: self._safe_set_int  ("spin_steps",    params["steps"])
        if "cfg"     in params: self._safe_set_float("scale_cfg",     params["cfg"])
        if "sampler" in params: self._safe_set_combo("combo_sampler", params["sampler"])
        if "resolution" in params: self._safe_set_combo("combo_res", params["resolution"])
        if "count"   in params: self._safe_set_int("spin_count", params["count"])
        if "seed"    in params: self._safe_set_int("spin_seed",  params["seed"])

        if "strength" in preset:
            self._safe_set_float("scale_strength", preset["strength"])

        if (af := preset.get("adetailer_face")) is not None:
            self._safe_set_check("chk_use_adetailer", af.get("enabled"))
            self._safe_set_combo("combo_ad_target", af.get("target"))
            self._safe_set_combo("combo_adetailer_model", af.get("model"))
            self._safe_set_float("scale_adetailer_strength", af.get("strength"))

        if (ah := preset.get("adetailer_hand")) is not None:
            self._safe_set_check("chk_use_ad_hand", ah.get("enabled"))
            self._safe_set_combo("combo_ad_hand", ah.get("target"))
            self._safe_set_float("scale_ad_hand", ah.get("strength"))
            self._safe_set_float("scale_ad_hand_blend", ah.get("blend"))

        if (hr := preset.get("hires")) is not None:
            self._safe_set_check("chk_hires", hr.get("enabled"))
            self._safe_set_combo("combo_hires_scale", hr.get("scale"))
            self._safe_set_float("scale_hires_denoise", hr.get("denoise"))
            self._safe_set_combo("combo_hires_upscaler", hr.get("upscaler"))

        if (cn := preset.get("controlnet")) is not None:
            self._safe_set_check("chk_use_pose", cn.get("enabled"))
            self._safe_set_combo("combo_cn_type", cn.get("type"))
            self._safe_set_float("scale_cn_strength", cn.get("strength"))

        for toggle in ("_toggle_adetailer", "_toggle_ad_hand",
                       "_toggle_hires", "_toggle_cn"):
            if hasattr(self, toggle):
                try: getattr(self, toggle)()
                except: pass

        after_snap = {k: self._get_widget_value(k)
                      for k in self._CONTROL_LABELS.keys()}
        changed = 0
        for k in self._CONTROL_LABELS.keys():
            b, a = before_snap.get(k), after_snap.get(k)
            if b != a and a is not None:
                self._flash_widget(k, "#a6e3a1")
                changed += 1
        self._flash_widget("combo_preset", "#cba6f7")

        if hasattr(self, "_update_preset_badge"):
            try: self._update_preset_badge(changed, [])
            except: pass
        self._set_status(
            f"🎯 已套用「{name}」· {changed} 项参数变化",
            "#a6e3a1" if changed else "#fab387"
        )
        logger.info(f"🎯 套用预设 [{name}] - {changed} 项变化")

    def read_png_info(self, path: str = None):
        if not path:
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

            pos, neg, info_part = "", "", ""

            if "Negative prompt:" in params:
                pos_part, rest = params.split("Negative prompt:", 1)
                pos = pos_part.strip()
                rest_lines = rest.strip().split("\n", 1)
                neg = rest_lines[0].strip()
                info_part = rest_lines[1].strip() if len(rest_lines) > 1 else ""
            else:
                lines = params.split("\n", 1)
                pos = lines[0].strip()
                info_part = lines[1].strip() if len(lines) > 1 else ""

            self.txt_prompt.setPlainText(pos)
            self.txt_neg.setPlainText(neg)

            if hasattr(self, "append_log"):
                self.append_log(f"📥 已读取 PNG 参数: {os.path.basename(path)}", "#a6e3a1")
                if info_part:
                    self.append_log(f"   {info_part}", "#7f849c")

            self._set_status("📥 已回填 PNG 元数据", "#a6e3a1")

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
    #  关闭
    # ==========================================================
    def closeEvent(self, event: QCloseEvent):
        if not getattr(self, '_ui_ready', False):
            logger.warning("⚠️ UI 未完成初始化, 跳过配置保存")
            event.accept()
            return
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
            self.config.adetailer_strength = self.scale_adetailer_strength.float_value()
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
            logger.warning(f"⚠️ 配置保存时出错(已忽略): {e}")

        if hasattr(self, '_video_thread') and self._video_thread.isRunning():
            self._video_thread.quit()
            self._video_thread.wait(2000)

        if hasattr(self, '_enhance_thread') and self._enhance_thread.isRunning():
            self._enhance_thread.quit()
            self._enhance_thread.wait(2000)

        t = getattr(self, '_gen_thread', None)
        if t and t.is_alive():
            from PyQt6.QtWidgets import QMessageBox
            r = QMessageBox.question(
                self, "生成进行中",
                "还有生成任务未完成，强制退出可能损坏输出文件。\n确定退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            logger.warning("⚠️ 用户强制退出，正在中断生成…")
            self.cancel_flag = True          # ← 关键：让 step_cb 抛 InterruptedError
            t.join(timeout=10)
            if t.is_alive():
                logger.warning("⚠️ 线程未在 10 秒内退出，强制关闭")

        event.accept()

    # ==========================================================
    #  画廊 / 预览
    # ==========================================================
    def _on_gallery_pick(self, path: str):
        try:
            self.show_preview(path)
        except Exception:
            pass

    def show_preview_image(self, path: str):
        logger.debug(f"[SHOW] 收到预览 path={path}")
        if not path or not os.path.exists(path):
            logger.error(f"[SHOW] ❌ 文件不存在")
            return
        try:
            from PyQt6.QtGui import QPixmap
            pix = QPixmap(path)
            if pix.isNull():
                logger.error(f"[SHOW] ❌ QPixmap 加载失败")
                return
            if hasattr(self, 'lbl_preview'):
                self.lbl_preview.set_pixmap(pix)
                logger.debug(f"[SHOW] ✅ 已刷新 lbl_preview")
        except Exception as e:
            import traceback
            logger.error(f"[SHOW] ❌ 异常: {e}")
            traceback.print_exc()

    def _on_new_image_saved(self, path: str):
        logger.debug(f"[GALLERY-RECV] 收到信号: {path}")
        if hasattr(self, 'gallery') and path and os.path.exists(path):
            try:
                self.gallery.add_image(path, prepend=True)
                logger.debug(f"[GALLERY-SIG] ✅ add_image 成功")
            except Exception as e:
                logger.error(f"[GALLERY-SIG] ❌ add_image 失败: {e}")

    def _on_apply_gallery_params(self, meta: dict):
        logger.debug(f"[APPLY] 收到 meta: {list(meta.keys())}")
        try:
            if meta.get("prompt") and hasattr(self, "txt_prompt"):
                self.txt_prompt.setPlainText(meta["prompt"])
            if meta.get("negative_prompt") and hasattr(self, "txt_neg"):
                self.txt_neg.setPlainText(meta["negative_prompt"])
            mapping = {
                "steps": ("spin_steps", int),
                "cfg_scale": ("scale_cfg", float),
                "cfg": ("scale_cfg", float),
                "seed": ("spin_seed", int),
                "width": ("spin_width", int),
                "height": ("spin_height", int),
            }
            for k, (widget_name, caster) in mapping.items():
                if k in meta and hasattr(self, widget_name):
                    try:
                        if caster is float:
                            self._safe_set_float(widget_name, float(meta[k]))
                        else:
                            self._safe_set_int(widget_name, int(meta[k]))
                    except Exception as e:
                        logger.debug(f"[APPLY] 跳过 {k}: {e}")
            if "sampler" in meta and hasattr(self, "combo_sampler"):
                try:
                    self._safe_set_combo("combo_sampler", str(meta["sampler"]))
                except Exception:
                    pass
            self._set_status("✅ 已套用画廊参数", "#a6e3a1")
        except Exception as e:
            logger.error(f"[APPLY] ❌ 异常: {e}")

    def apply_meta_params(self, meta: dict):
        try:
            if "prompt" in meta and meta["prompt"]:
                self.txt_prompt.setPlainText(meta["prompt"])
            if "negative_prompt" in meta and meta["negative_prompt"]:
                self.txt_neg.setPlainText(meta["negative_prompt"])
            if "steps" in meta:
                self._safe_set_int("spin_steps", int(meta["steps"]))
            if "cfg" in meta or "cfg_scale" in meta:
                cfg = meta.get("cfg") or meta.get("cfg_scale")
                self._safe_set_float("scale_cfg", float(cfg))
            if "sampler" in meta:
                self._safe_set_combo("combo_sampler", str(meta["sampler"]))
            if "seed" in meta:
                try:
                    self.spin_seed.setValue(int(meta["seed"]))
                except Exception:
                    pass
            self._set_status("✅ 已套用元数据参数", "#a6e3a1")
        except Exception as e:
            self._set_status(f"⚠️ 套用失败: {e}", "#f38ba8")

    def save_current_image_as(self):
        from PyQt6.QtWidgets import QFileDialog
        import shutil
        src = getattr(self, 'last_generated_path', None)
        if not src or not os.path.exists(src):
            self._set_status("⚠️ 当前没有可保存的图片", "#f9e2af")
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "另存为", os.path.basename(src),
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;所有文件 (*)"
        )
        if not dst:
            return
        try:
            shutil.copy2(src, dst)
            self._set_status(f"✅ 已另存为: {os.path.basename(dst)}", "#a6e3a1")
        except Exception as e:
            self._set_status(f"❌ 保存失败: {e}", "#f38ba8")

    def send_preview_to_img2img(self):
        src = getattr(self, 'last_generated_path', None)
        if not src or not os.path.exists(src):
            self._set_status("⚠️ 当前没有可发送的图片", "#f9e2af")
            return
        try:
            self.ref_image_path = src
            if hasattr(self, 'tabs'):
                for i in range(self.tabs.count()):
                    if "图生图" in self.tabs.tabText(i) or "img2img" in self.tabs.tabText(i).lower():
                        self.tabs.setCurrentIndex(i)
                        break
            if hasattr(self, 'refresh_ref_image_preview'):
                self.refresh_ref_image_preview()
            self._set_status("✅ 已发送到图生图", "#a6e3a1")
        except Exception as e:
            self._set_status(f"❌ 发送失败: {e}", "#f38ba8")

    def send_preview_to_inpaint(self):
        src = getattr(self, 'last_generated_path', None)
        if not src or not os.path.exists(src):
            self._set_status("⚠️ 当前没有可发送的图片", "#f9e2af")
            return
        try:
            self.inpaint_image_path = src
            if hasattr(self, 'tabs'):
                for i in range(self.tabs.count()):
                    if "重绘" in self.tabs.tabText(i) or "inpaint" in self.tabs.tabText(i).lower():
                        self.tabs.setCurrentIndex(i)
                        break
            if hasattr(self, 'refresh_inpaint_preview'):
                self.refresh_inpaint_preview()
            self._set_status("✅ 已发送到局部重绘", "#a6e3a1")
        except Exception as e:
            self._set_status(f"❌ 发送失败: {e}", "#f38ba8")

    def reuse_params_from_path(self, path: str):
        try:
            if hasattr(self, 'read_png_info'):
                self.read_png_info(path=path)
                self._set_status("🔁 已套用 PNG 参数", "#a6e3a1")
            else:
                self._set_status("⚠️ 未找到 read_png_info", "#f9e2af")
        except Exception as e:
            self._set_status(f"❌ 复用失败: {e}", "#f38ba8")

    def send_path_to_img2img(self, path: str):
        self.last_generated_path = path
        if hasattr(self, 'send_preview_to_img2img'):
            self.send_preview_to_img2img()
        self._set_status("🛠 已发送到 img2img", "#a6e3a1")

    def send_path_to_face_fix(self, path: str):
        """画廊 → 发送到修脸：打开修图编辑器，引擎就绪时自动运行 ADetailer"""
        if not path or not os.path.exists(path):
            self._set_status("⚠️ 图片不存在,无法修脸", "#f38ba8")
            return
        try:
            self.show_preview(path)
        except Exception:
            self.current_generated_path = path
            self.last_generated_path = path

        # 引擎已就绪 → 编辑器打开后自动跑一次 ADetailer
        engine_ready = bool(
            getattr(self, 'ai', None)
            and getattr(self.ai, 'img2img_pipe', None) is not None
        )
        self._editor_auto_adetailer = engine_ready
        self._set_status(
            "😀 已打开发修图编辑器" + ("，ADetailer 自动修复中..." if engine_ready
                                     else "，请点击左侧「ADetailer 人脸修复」"),
            "#a6e3a1")
        self.open_editor()

    def send_path_to_editor(self, path: str):
        """画廊 → 载入预览并打开修图编辑器"""
        if not path or not os.path.exists(path):
            self._set_status("⚠️ 图片不存在,无法编辑", "#f38ba8")
            return
        try:
            self.show_preview(path)   # 同时更新 current_generated_path
        except Exception:
            self.current_generated_path = path
            self.last_generated_path = path
        self._set_status("✏️ 正在打开修图编辑器...", "#89dceb")
        self.open_editor()

    # ==========================================================
    #  模型切换
    # ==========================================================
    def _on_model_type_changed(self, idx=None):
        from utils.model_scanner import scan_models
        if not hasattr(self, 'combo_model_type'):
            return
        mtype = self.combo_model_type.currentData()
        if not mtype:
            return

        models = scan_models(mtype)
        self.combo_model.blockSignals(True)
        self.combo_model.clear()

        if not models:
            self.combo_model.addItem(f"(此目录无模型,请放入 models/{mtype}/)")
            self.combo_model.setEnabled(False)
        else:
            self.combo_model.setEnabled(True)
            for m in models:
                label = f"{m['name']}  [{m['size_gb']}GB]"
                self.combo_model.addItem(label, m)

        self.combo_model.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self, idx=None):
        data = self.combo_model.currentData()
        note = ""
        if isinstance(data, dict):
            note = data.get("note", "")

        label = getattr(self, 'lbl_model_note', None) or getattr(self, 'lbl_model_info', None)
        if label is not None:
            if note:
                label.setText(f"📌 备注: {note}")
            else:
                label.setText("📌 备注: (无)")

    # ==========================================================
    #  视频面板辅助 → 见 ui/video_panel_mixin.py (VideoPanelMixin)
    # ==========================================================

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

    from ui.disclaimer import check_global_disclaimer
    if not check_global_disclaimer():
        logger.error("❌ 用户未同意免责声明,程序退出")
        sys.exit(0)

    splash = create_splash()
    splash.set_message("正在初始化主界面...")

    window = AIDesktopApp()

    def _show_main():
        window.showMaximized()
        splash.finish_loading(window)

    QTimer.singleShot(500, _show_main)
    sys.exit(app.exec())