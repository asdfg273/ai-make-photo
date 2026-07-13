# main.py
# ============================================================
#  PyQt6 主入口 — AI 绘画工作站 v5.0
# ============================================================

import os
import sys
import glob
import cv2

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

print(f"📦 模型缓存目录: {CACHE_ROOT}")

import threading
import warnings
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from utils.ui_builder import enable_gpu_acceleration
enable_gpu_acceleration()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QLineEdit, QSpinBox, QPushButton, QLabel,
    QWidget, QGroupBox, QScrollArea, QComboBox, QCheckBox,
    QListWidget, QListWidgetItem, QTextEdit, QDoubleSpinBox
)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt, QSize, pyqtSlot
from PyQt6.QtGui  import QCloseEvent, QIcon, QPixmap, QImage
from PIL import Image

from core.translation_service  import TranslationService
from core.config_manager       import AppConfig
from utils.ui_builder     import (
    UIBuilderMixin, FloatSlider, create_splash, DARK_STYLE
)
from utils.app_events     import EventMixin
from utils.app_generation import GenerationMixin
from utils.app_utils      import OUTPUT_DIR as APP_OUTPUT_DIR
from utils.system_utils   import log_system_info, logger
from utils.preset_manager import PresetManagerMixin, TooltipMixin
from core.presets import PROMPT_PRESETS
from utils.prompt_enhancer import get_enhancer
from utils.paths import OUTPUT_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch = None  # 延迟导入
try:
    from utils.model_downloader import print_scan_report
    print_scan_report()
except Exception as e:
    print(f"⚠️ 模型扫描失败(忽略): {e}")

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
                   PresetManagerMixin, TooltipMixin):

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
        self.travel_segments = []

        self.cleanup_temp_files(verbose=False)
        self._prompt_presets = PROMPT_PRESETS

        # ── App 信号桥 ──
        self._app_bridge = _AppBridge()
        self._app_bridge.ai_loaded.connect(self._on_ai_loaded)
        self._app_bridge.ai_failed.connect(self._on_ai_failed)
        self._app_bridge.status_msg.connect(self._set_status)

        # ── 配置 & UI ──
        self.config = AppConfig()
        self.config.load()

        self.setup_ui()
        self.apply_config_to_ui()
        self.cleanup_temp_files(verbose=False)
        self._refresh_preset_combo()
        self.apply_tooltips()

        self.gallery.reuse_params_signal.connect(self.reuse_params_from_path)
        self.gallery.send_to_i2i_signal.connect(self.send_path_to_img2img)
        self.gallery.send_to_face_signal.connect(self.send_path_to_face_fix)

        # ── 生成信号桥 ──
        self._init_gen_bridge()
        try:
            self._bridge.preview_signal.connect(self._on_new_image_saved)
            self._bridge.gallery_add_signal.connect(self._on_new_image_saved)
            self._bridge.live_preview_signal.connect(self.show_preview_image)
        except Exception as e:
            print(f"[CONNECT] ❌ 预览信号连接失败: {e}")

        try:
            self.gallery.apply_params_signal.connect(self._on_apply_gallery_params)
        except Exception as e:
            print(f"[CONNECT] ❌ 失败: {e}")

        try:
            self._bridge.enhance_done_signal.connect(self._on_enhance_done)
        except Exception:
            pass

        # ── 初始按钮状态 ──
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("🚀 AI 引擎预热中...")
        self._set_status("⏳ 正在后台加载大模型与底层环境,请稍候...", "#f9e2af")

        threading.Thread(target=self._async_init_ai, daemon=True).start()

        if hasattr(self, 'combo_model') and self.combo_model.count() > 0:
            self._set_status("🔄 正在加载默认模型...", "#f9e2af")
            QTimer.singleShot(500, self._preload_default_model)

    # ==========================================================
    #  视频生成
    # ==========================================================
    def on_generate_video(self):
        """触发视频生成流程 - 支持 4 种模式"""
        print("🔵 [DEBUG] on_generate_video 被调用")

        if getattr(self, 'is_generating', False):
            self._set_status("⚠️ 正在生成中,请等待", "#ff7a17")
            return

        try:
            self.is_generating = True
            self.btn_gen_video.setEnabled(False)
            self.btn_gen_video.setText("生成中...")

            # ---- 1. 模式识别 ----
            mode_idx = self.combo_video_mode.currentIndex()
            mode_str = ["txt2video", "img2video", "vid2vid", "prompt_travel"][mode_idx] \
                if 0 <= mode_idx <= 3 else "txt2video"
            mode_label = ["文生视频", "图生视频", "视频转绘", "提示词旅行"][mode_idx] \
                if 0 <= mode_idx <= 3 else "文生视频"

            print(f"🔵 [DEBUG] mode = {mode_str} ({mode_label})")
            print(f"🔵 [DEBUG] _video_input_path = {getattr(self, '_video_input_path', '未设置')}")

            # ---- 2. 提示词 ----
            prompt = self.txt_video_prompt.toPlainText().strip()
            negative = self.txt_video_neg.toPlainText().strip() or (
                "bad hands, bad fingers, extra fingers, missing fingers, "
                "deformed hands, orange tint, warm color cast, oversaturated, "
                "lowres, worst quality, low quality, jpeg artifacts, blurry"
            )

            if not prompt and mode_str != "prompt_travel":
                self._reset_video_button()
                self._set_status("⚠️ 请输入正面提示词", "#ff7a17")
                return

            # ---- 3. 输入文件校验 ----
            input_path = getattr(self, '_video_input_path', None)
            if mode_str in ("img2video", "vid2vid"):
                if not input_path or not os.path.exists(input_path):
                    self._reset_video_button()
                    self._set_status(f"⚠️ {mode_label}需要选择输入文件", "#ff7a17")
                    return

                ext = os.path.splitext(input_path)[1].lower()
                if mode_str == "img2video" and ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    self._reset_video_button()
                    self._set_status("⚠️ 图生视频请选择图片文件", "#ff7a17")
                    return
                if mode_str == "vid2vid" and ext not in (".mp4", ".mov", ".avi", ".webm", ".gif"):
                    self._reset_video_button()
                    self._set_status("⚠️ 视频转绘请选择视频文件", "#ff7a17")
                    return

            # ---- 4. 基本参数 ----
            num_frames = self.spin_video_frames.value()
            num_steps  = self.spin_video_steps.value()
            guidance   = self.spin_video_cfg.value()
            width      = self.spin_video_w.value()
            height     = self.spin_video_h.value()
            fps        = self.spin_video_fps.value()

            scheduler_map = {
                "EulerDiscrete (推荐)": "euler",
                "DPM++ 2M": "dpm++",
                "LCM (快速)": "lcm",
                "DDIM": "ddim",
            }
            scheduler = scheduler_map.get(
                self.combo_video_sched.currentText(), "euler")

            use_context_window = self.chk_long_video.isChecked() and num_frames > 32
            output_format = self.combo_video_fmt.currentText()

            # ---- 5. 提示词旅行分段 ----
            prompt_travel = []
            if mode_str == "prompt_travel":
                if hasattr(self, "travel_segments") and self.travel_segments:
                    for seg in self.travel_segments:
                        try:
                            if isinstance(seg, dict):
                                f = int(seg['frame_spin'].value())
                                p = seg['prompt_edit'].text().strip()
                            else:
                                f = int(seg[0].value())
                                p = seg[1].text().strip()
                            if p:
                                prompt_travel.append((f, p))
                        except Exception:
                            continue

                if not prompt_travel and hasattr(self, "txt_prompt_travel"):
                    for line in self.txt_prompt_travel.toPlainText().splitlines():
                        line = line.strip()
                        if '|' in line:
                            a, b = line.split('|', 1)
                            try:
                                prompt_travel.append((int(a.strip()), b.strip()))
                            except ValueError:
                                continue

                if len(prompt_travel) < 2:
                    self._reset_video_button()
                    self._set_status("⚠️ 提示词旅行至少需要 2 个关键帧", "#ff7a17")
                    return

                if not prompt:
                    prompt = prompt_travel[0][1]

            # ---- 6. 翻译 ----
            try:
                if prompt and getattr(self, "translator", None):
                    prompt = self.translator.translate(prompt)
                if negative and getattr(self, "translator", None):
                    negative = self.translator.translate(negative)
                if prompt_travel and getattr(self, "translator", None):
                    prompt_travel = [
                        (f, self.translator.translate(p)) for f, p in prompt_travel
                    ]
            except Exception as e:
                print(f"⚠️ 翻译失败(忽略): {e}")

            # ---- 7. 高级参数 ----
            strength = float(self.spin_video_strength.value()) \
                if hasattr(self, "spin_video_strength") else 0.75
            ip_adapter_scale = float(self.spin_video_ipa_scale.value()) \
                if hasattr(self, "spin_video_ipa_scale") else 0.7
            seed_val = int(self.spin_video_seed.value()) \
                if hasattr(self, "spin_video_seed") else -1

            # ---- 8. Motion LoRA (多选) ----
            motion_loras = self._collect_motion_loras() \
                if hasattr(self, "_collect_motion_loras") else []
            print(f"🔵 [DEBUG] motion_loras = {motion_loras}")

            # ---- 9. 状态 ----
            self._set_status(
                f"🎬 [{mode_label}] 开始生成: {num_frames}帧 {width}x{height}",
                "#ff7a17")
            self.lbl_video_status.setText(f"🎬 [{mode_label}] 生成中...")
            self.lbl_video_status.setStyleSheet("color:#ff7a17; padding:4px;")

            # ---- 10. 进度回调 ----
            def progress_callback(pipe, step, timestep, callback_kwargs):
                try:
                    progress = int((step / max(num_steps, 1)) * 100)
                    self._app_bridge.status_msg.emit(f"🎬 生成中 {progress}%", "#ff7a17")
                except Exception:
                    pass
                return callback_kwargs

            # ---- 11. 后台线程 ----
            def generate_task():
                try:
                    if not hasattr(self, 'video_generator') or self.video_generator is None:
                        from utils.video_gen import VideoGenerator
                        self.video_generator = VideoGenerator(self.ai)

                    result = self.video_generator.generate(
                        prompt=prompt,
                        negative=negative,
                        num_frames=num_frames,
                        num_steps=num_steps,
                        guidance=guidance,
                        width=width,
                        height=height,
                        fps=fps,
                        scheduler=scheduler,
                        motion_loras=motion_loras,
                        use_context_window=use_context_window,
                        prompt_travel=prompt_travel if prompt_travel else None,
                        output_format=output_format,
                        output_dir="photo/videos",
                        progress_callback=progress_callback,
                        mode=mode_str,
                        input_path=input_path,
                        strength=strength,
                        ip_adapter_scale=ip_adapter_scale,
                        seed=seed_val,
                    )

                    if isinstance(result, tuple) and len(result) == 2:
                        video_path, used_seed = result
                    else:
                        video_path, used_seed = result, seed_val

                    from PyQt6.QtCore import QMetaObject, Q_ARG, Qt
                    QMetaObject.invokeMethod(
                        self, "_on_video_generated",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, str(video_path)),
                        Q_ARG(int, int(used_seed)),
                    )

                except Exception as e:
                    import traceback
                    error_msg = f"❌ 视频生成失败: {str(e)}"
                    print(f"[VIDEO GEN ERROR]\n{traceback.format_exc()}")
                    from PyQt6.QtCore import QMetaObject, Q_ARG, Qt
                    QMetaObject.invokeMethod(
                        self, "_on_video_error",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(int, 0),
                        Q_ARG(str, error_msg),
                    )

            threading.Thread(target=generate_task, daemon=True).start()

        except Exception as e:
            import traceback
            print(f"[VIDEO PARAM ERROR]\n{traceback.format_exc()}")
            self._reset_video_button()
            self._set_status(f"⚠️ 参数校验失败: {e}", "#ff7a17")

    def _reset_video_button(self):
        """恢复视频生成按钮状态"""
        self.is_generating = False
        if hasattr(self, 'btn_gen_video'):
            self.btn_gen_video.setEnabled(True)
            self.btn_gen_video.setText("🎬 生成视频")

    @pyqtSlot(str, int)
    def _on_video_generated(self, video_path: str, seed: int):
        """视频生成完成回调 (主线程)"""
        try:
            self._reset_video_button()
            self.set_progress(100)

            self._set_status(
                f"✅ 视频生成完成: {os.path.basename(video_path)} (seed={seed})",
                "#dadbdf")
            if hasattr(self, 'lbl_video_status'):
                self.lbl_video_status.setText(f"✅ 已完成 (seed={seed})")
                self.lbl_video_status.setStyleSheet("color:#dadbdf; padding:4px;")

            if hasattr(self, 'play_video') and hasattr(self, 'video_widget'):
                try:
                    self.play_video(video_path)
                except Exception as e:
                    print(f"⚠️ 播放失败: {e}")

            if hasattr(self, 'video_list'):
                self._refresh_video_gallery()

            if hasattr(self, 'gallery'):
                if not hasattr(self, '_gallery_seen_paths'):
                    self._gallery_seen_paths = set()
                abs_path = os.path.abspath(video_path)
                if abs_path not in self._gallery_seen_paths:
                    self._gallery_seen_paths.add(abs_path)
                    try:
                        self.gallery.add_image(video_path, prepend=True)
                    except Exception:
                        pass
        except Exception as e:
            self._set_status(f"⚠️ 视频生成后处理失败: {e}", "#ff7a17")

    @pyqtSlot(int, str)
    def _on_video_error(self, code: int, error_msg: str):
        """视频生成失败回调 (主线程)"""
        self._reset_video_button()
        if hasattr(self, 'lbl_video_status'):
            self.lbl_video_status.setText("❌ 生成失败")
            self.lbl_video_status.setStyleSheet("color:#f38ba8; padding:4px;")
        self._set_status(error_msg, "#f38ba8")

    def _on_video_player_error(self, error, error_string):
        """QMediaPlayer 播放错误"""
        self._set_status(f"⚠️ 视频播放错误: {error_string}", "#ff7a17")
        if hasattr(self, 'video_stacked'):
            self.video_stacked.setCurrentIndex(0)
        elif hasattr(self, 'lbl_video_placeholder'):
            self.lbl_video_placeholder.show()
            if hasattr(self, 'video_widget'):
                self.video_widget.hide()

    # ==========================================================
    #  状态 / 加载
    # ==========================================================
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
        logger.info("👉 [预热] 后台导入重型库 (PyTorch / Diffusers)...")
        self._emit_status("⏳ 正在导入 PyTorch & Diffusers...", "#f9e2af")
        try:
            global torch
            import torch
            from core.model_manager import ModelManager
            self._emit_status(
                "⏳ 正在加载大模型,首次启动可能需要数分钟...", "#f9e2af")
            self.ai = ModelManager()
        except Exception as e:
            logger.error(f"❌ AI 引擎加载失败: {e}")
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

    def _on_enhance_done(self, result: str):
        if hasattr(self, 'btn_enhance_prompt'):
            self.btn_enhance_prompt.setEnabled(True)
            self.btn_enhance_prompt.setText("✨ 智能改写")
        if hasattr(self, 'btn_vision_prompt'):
            self.btn_vision_prompt.setEnabled(True)
            self.btn_vision_prompt.setText("📷 识图生成")

        if result.startswith("[识图失败]") or result.startswith("[改写失败]"):
            self._set_status(result, "#f38ba8")
            return

        self.txt_prompt.setPlainText(result)
        self._set_status("✨ 提示词生成完成!", "#a6e3a1")

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

    def read_png_info(self):
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
        print(f"[SHOW] 收到预览 path={path}", flush=True)
        if not path or not os.path.exists(path):
            print(f"[SHOW] ❌ 文件不存在", flush=True)
            return
        try:
            from PyQt6.QtGui import QPixmap
            pix = QPixmap(path)
            if pix.isNull():
                print(f"[SHOW] ❌ QPixmap 加载失败", flush=True)
                return
            if hasattr(self, 'lbl_preview'):
                self.lbl_preview.set_pixmap(pix)
                print(f"[SHOW] ✅ 已刷新 lbl_preview", flush=True)
        except Exception as e:
            import traceback
            print(f"[SHOW] ❌ 异常: {e}", flush=True)
            traceback.print_exc()

    def _on_new_image_saved(self, path: str):
        print(f"[GALLERY-RECV] 收到信号: {path}", flush=True)
        if hasattr(self, 'gallery') and path and os.path.exists(path):
            try:
                self.gallery.add_image(path, prepend=True)
                print(f"[GALLERY-SIG] ✅ add_image 成功")
            except Exception as e:
                print(f"[GALLERY-SIG] ❌ add_image 失败: {e}")

    def _on_apply_gallery_params(self, meta: dict):
        print(f"[APPLY] 收到 meta: {list(meta.keys())}", flush=True)
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
                        print(f"[APPLY] 跳过 {k}: {e}", flush=True)
            if "sampler" in meta and hasattr(self, "combo_sampler"):
                try:
                    self._safe_set_combo("combo_sampler", str(meta["sampler"]))
                except Exception:
                    pass
            self._set_status("✅ 已套用画廊参数", "#a6e3a1")
        except Exception as e:
            print(f"[APPLY] ❌ 异常: {e}", flush=True)

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
                try:
                    self.read_png_info(path=path)
                except TypeError:
                    self.last_generated_path = path
                    self.read_png_info()
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
        self.last_generated_path = path
        if hasattr(self, 'chk_use_adetailer'):
            self.chk_use_adetailer.setChecked(True)
        if hasattr(self, 'send_preview_to_img2img'):
            self.send_preview_to_img2img()
        self._set_status("😀 已发送到修脸 (ADetailer 已开启)", "#a6e3a1")

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
    #  视频面板辅助
    # ==========================================================
    def _set_video_duration(self, seconds: int):
        fps = self.spin_video_fps.value()
        frames = seconds * fps
        self.spin_video_frames.setValue(min(frames, 80))

    def _add_travel_segment(self):
        row = QHBoxLayout()

        idx = len(self.travel_segments) + 1
        lbl = QLabel(f"段 {idx}")
        lbl.setFixedWidth(40)
        row.addWidget(lbl)

        lbl_frame = QLabel("帧")
        row.addWidget(lbl_frame)
        spin_frame = QSpinBox()
        spin_frame.setRange(0, 79)
        spin_frame.setValue(0)
        spin_frame.setFixedWidth(50)
        spin_frame.setStyleSheet("QSpinBox { padding: 2px 4px; }")
        row.addWidget(spin_frame)

        txt_prompt = QLineEdit()
        txt_prompt.setPlaceholderText("输入该段的提示词...")
        row.addWidget(txt_prompt)

        btn_del = QPushButton("❌")
        btn_del.setFixedWidth(30)
        widgets = {'frame_spin': spin_frame, 'prompt_edit': txt_prompt, 'btn_del': btn_del, 'row': row}
        btn_del.clicked.connect(lambda _, w=widgets: self._remove_travel_segment(w))
        row.addWidget(btn_del)

        if hasattr(self, 'travel_container'):
            self.travel_container.addLayout(row)
        self.travel_segments.append(widgets)

        self._auto_distribute_frames()

    def _remove_travel_segment(self, widgets):
        try:
            row_layout = widgets['row']
            while row_layout.count():
                item = row_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            if widgets in self.travel_segments:
                self.travel_segments.remove(widgets)
            self._auto_distribute_frames()
        except Exception as e:
            print(f"⚠️ 移除旅行分段失败: {e}")

    def _auto_distribute_frames(self):
        if not self.travel_segments:
            return

        total_frames = self.spin_video_frames.value()
        num_segments = len(self.travel_segments)

        for i, seg in enumerate(self.travel_segments):
            spin_frame = seg['frame_spin'] if isinstance(seg, dict) else seg[0]
            frame_idx = int(i * total_frames / num_segments)
            spin_frame.blockSignals(True)
            spin_frame.setValue(frame_idx)
            spin_frame.blockSignals(False)

    def _refresh_video_gallery(self):
        if not hasattr(self, 'video_list'):
            return
        self.video_list.clear()
        video_dir = "photo/videos"

        if not os.path.exists(video_dir):
            return

        videos = []
        for ext in ["*.mp4", "*.gif"]:
            videos.extend(glob.glob(os.path.join(video_dir, ext)))

        videos.sort(key=os.path.getmtime, reverse=True)

        for vpath in videos[:50]:
            cap = cv2.VideoCapture(vpath)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(160, 90, Qt.AspectRatioMode.KeepAspectRatio)

            item = QListWidgetItem(QIcon(pixmap), os.path.basename(vpath))
            item.setData(Qt.ItemDataRole.UserRole, vpath)
            self.video_list.addItem(item)

    def _on_video_item_clicked(self, item):
        vpath = item.data(Qt.ItemDataRole.UserRole)
        try:
            os.startfile(vpath)
        except Exception as e:
            print(f"⚠️ 打开视频失败: {e}")

    def _add_motion_lora_item(self):
        """➕ 添加一个 Motion LoRA 到已选列表"""
        name = self.cmb_motion_lora_pick.currentText()
        if not name or name.startswith("--"):
            return
        # 去重
        for item in self.motion_lora_items:
            if item['name'] == name:
                print(f"⚠️ {name} 已添加")
                if hasattr(self, '_app_bridge'):
                    self._app_bridge.status_msg.emit(f"⚠️ {name} 已添加", "#ff7a17")
                return

        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider, QPushButton
        from PyQt6.QtCore import Qt

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        lbl_name = QLabel(name)
        lbl_name.setFixedWidth(100)
        lbl_name.setStyleSheet("color:#ddd; font-size:12px;")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 150)          # 0.00 - 1.50
        slider.setValue(80)              # 默认 0.80
        slider.setFixedHeight(20)

        lbl_val = QLabel("0.80")
        lbl_val.setFixedWidth(40)
        lbl_val.setStyleSheet("color:#8be9fd; font-size:12px;")
        slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(f"{v/100:.2f}"))

        btn_del = QPushButton("✕")
        btn_del.setFixedSize(24, 24)
        btn_del.setStyleSheet("QPushButton{color:#ff5555;background:transparent;border:none;font-weight:bold;}"
                              "QPushButton:hover{background:#4a2020;border-radius:4px;}")
        btn_del.clicked.connect(lambda _, n=name: self._remove_motion_lora_item(n))

        layout.addWidget(lbl_name)
        layout.addWidget(slider, 1)
        layout.addWidget(lbl_val)
        layout.addWidget(btn_del)

        self.motion_lora_container.addWidget(row)
        self.motion_lora_items.append({
            'name': name, 'widget': row, 'slider': slider, 'label': lbl_val
        })
        print(f"✅ 已添加 Motion LoRA: {name}")

    def _remove_motion_lora_item(self, name):
        """✕ 移除某个 Motion LoRA"""
        for i, item in enumerate(self.motion_lora_items):
            if item['name'] == name:
                item['widget'].setParent(None)
                item['widget'].deleteLater()
                self.motion_lora_items.pop(i)
                print(f"🗑️ 已移除 Motion LoRA: {name}")
                return

    def _collect_motion_loras(self):
        """收集所有已选 Motion LoRA → [{'name':str, 'weight':float}, ...]"""
        result = []
        for item in self.motion_lora_items:
            result.append({
                'name': item['name'],
                'weight': item['slider'].value() / 100.0,
            })
        return result

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