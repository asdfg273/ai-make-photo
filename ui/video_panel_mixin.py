# ui/video_panel_mixin.py
# ============================================================
#  视频面板控制器 Mixin — 从 main.py 提取
#  负责视频生成(4 种模式)、Motion LoRA、提示词旅行、TTS 配音
# ============================================================

import os
import re
import glob
import cv2
import threading
import traceback
import subprocess
import shutil
import tempfile

from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit,
    QListWidgetItem, QWidget, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSlot, QMetaObject, Q_ARG
from PyQt6.QtGui import QIcon, QPixmap, QImage

from utils.system_utils import logger


class VideoPanelMixin:
    """视频面板相关方法。

    要求宿主类 (AIDesktopApp) 提供以下属性:
    - self.ai (ModelManager)
    - self.translator (TranslationService)
    - self.video_generator (VideoGenerator)
    - self.tts_engine
    - self._app_bridge (_AppBridge)
    - self.is_generating (bool)
    - self._video_input_path (str)
    - self.travel_segments (list)
    - self.motion_lora_items (list)
    - self.motion_lora_container (QWidget)
    - self.travel_container (QWidget / QLayout)
    - self.gallery (GalleryPanel)
    - self._gallery_seen_paths (set)

    以及以下方法 (由其他 Mixin/主类提供):
    - self._set_status(text, color)
    - self.set_progress(value)
    - self.set_status(text, color)
    - self.play_video(path)
    """

    # ==========================================================
    #  视频生成主流程
    # ==========================================================
    def on_generate_video(self):
        """触发视频生成流程 - 支持 4 种模式"""
        print("🔵 [DEBUG] on_generate_video 被调用")

        if getattr(self, 'is_generating', False):
            self._set_status("⚠️ 正在生成中,请等待", "#ff7a17")
            return
        self.is_generating = True

        try:
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

            # ---- 8. Motion LoRA ----
            motion_loras = self._collect_motion_loras() \
                if hasattr(self, "_collect_motion_loras") else []
            print(f"🔵 [DEBUG] motion_loras = {motion_loras}")

            # ---- 8.5 配音参数收集 ----
            voice_params = None
            if hasattr(self, "chk_video_voice") and self.chk_video_voice.isChecked():
                engine = self.combo_tts_engine.currentText() if hasattr(self, "combo_tts_engine") else "GPT-SoVITS"
                if "SoVITS" in engine:
                    ref_data = self.combo_sovits_ref.currentData()
                    voice_params = {
                        "engine": "sovits",
                        "ref_audio": ref_data if isinstance(ref_data, str) else None,
                        "ref_text": self.txt_sovits_reftext.toPlainText().strip() or None,
                        "speed": self.spin_sovits_speed.value(),
                        "auto_translate": self.chk_sovits_auto_translate.isChecked(),
                    }
                else:
                    voice_params = {"engine": "chattts"}
            # 背景配乐（可选）
            # TODO: lbl_bg_music_path 控件从未创建,此分支永远为空 ——
            # 待补文件选择控件后再启用,先保留参数透传逻辑
            if voice_params and hasattr(self, "chk_bg_music") and self.chk_bg_music.isChecked():
                bg_path = getattr(self, "lbl_bg_music_path", None)
                if bg_path and os.path.exists(str(bg_path)):
                    voice_params["bg_music"] = str(bg_path)

            if voice_params:
                # 优先使用配音文本框内容，为空时回退到提示词
                voice_text = ""
                if hasattr(self, "txt_video_voice"):
                    voice_text = self.txt_video_voice.toPlainText().strip()
                if not voice_text:
                    voice_text = prompt
                    logger.info("📝 配音文本框为空,使用提示词作为配音文本")
                else:
                    logger.info(f"📝 使用配音文本框内容: {voice_text[:30]}...")
                # SoVITS: 可选自动翻译为日语
                if voice_params.get("engine") == "sovits":
                    if voice_params.get("auto_translate") and getattr(self, "translator", None):
                        try:
                            voice_text = self.translator.translate(voice_text, target_lang="ja")
                            logger.info(f"📝 已翻译为日语: {voice_text[:30]}...")
                        except Exception as e:
                            logger.warning(f"⚠️ 翻译失败,用原文: {e}")
                voice_params["text"] = voice_text
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
                    if self.ai.txt2img_pipe is None:
                        raw = self._ui_read(lambda: self.combo_model.currentText(), "")
                        # 剥掉 " [1.99GB]" 之类的后缀
                        model_name = re.sub(r"\s*\[.*?\]\s*$", "", raw).strip()
                        if not model_name:
                            raise RuntimeError("请先在图片面板选择一个 SD 底模")

                        print(f"⏳ 视频生成前置:加载 SD 底模 {model_name} ...")
                        self.ai.load_model(model_name)
                        print("✅ 底模就绪,继续视频生成")

                    if hasattr(self, 'video_generator') and self.video_generator is not None:
                        if getattr(self.video_generator, '_base_model_name', None) != self.ai.current_model_name:
                            print("🔄 底模已更换,重建 VideoGenerator")
                            self.video_generator = None

                    if not hasattr(self, 'video_generator') or self.video_generator is None:
                        from utils.video_gen import VideoGenerator
                        self.video_generator = VideoGenerator(self.ai)
                        self.video_generator._base_model_name = self.ai.current_model_name

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

                    final_path = str(video_path)
                    if voice_params:
                        try:
                            orig_path = str(video_path)
                            final_path = self._add_voice_to_video(orig_path, voice_params)
                            # 配音成功 → 删除未配音原版，避免画廊出现两份
                            if final_path != orig_path and os.path.exists(final_path):
                                try:
                                    os.remove(orig_path)
                                    logger.info(f"🗑️ 已删除未配音版本: {os.path.basename(orig_path)}")
                                except Exception as e:
                                    logger.warning(f"⚠️ 删除原视频失败: {e}")

                        except Exception as ve:
                            print(f"⚠️ 配音合成失败(保留原视频): {ve}")
                            traceback.print_exc()

                    QMetaObject.invokeMethod(
                        self, "_on_video_generated",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, final_path),
                        Q_ARG(int, int(used_seed)),
                    )

                except Exception as e:
                    error_msg = f"❌ 视频生成失败: {str(e)}"
                    print(f"[VIDEO GEN ERROR]\n{traceback.format_exc()}")
                    QMetaObject.invokeMethod(
                        self, "_on_video_gen_error",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(int, 0),
                        Q_ARG(str, error_msg),
                    )

            threading.Thread(target=generate_task, daemon=True).start()

        except Exception as e:
            print(f"[VIDEO PARAM ERROR]\n{traceback.format_exc()}")
            self._reset_video_button()
            self._set_status(f"⚠️ 参数校验失败: {e}", "#ff7a17")

    # ==========================================================
    #  TTS 配音 + FFmpeg 合并
    # ==========================================================
    def _add_voice_to_video(self, video_path: str, voice_params: dict) -> str:
        """双引擎 TTS + 合并视频,失败返回原路径"""
        # 转为绝对路径，防止 TTS 加载时 os.chdir 导致相对路径失效
        video_path = os.path.abspath(video_path)

        engine = voice_params.get("engine", "chattts")
        text = voice_params.get("text", "")
        if not text.strip():
            return video_path

        try:
            audio_path = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name

            if engine == "sovits":
                from utils.sovits_tts import synth_once
                ref = voice_params.get("ref_audio")
                synth_once(
                    text=text,
                    output_path=audio_path,
                    ref_audio=ref if ref else None,
                    ref_text=voice_params.get("ref_text", ""),
                    language=voice_params.get("language", "ja"),
                    speed=voice_params.get("speed", 1.0),
                )
                logger.info(f"🎙️ SoVITS 合成完成: {audio_path}")
            else:
                audio_path = self.tts_engine.generate_chattts(
                    text, speaker_seed=voice_params.get("speaker_seed", 2222)
                )
                logger.info(f"🎙️ ChatTTS 合成完成: {audio_path}")
        except Exception as e:
            logger.error(f"❌ TTS 合成失败: {e}")
            traceback.print_exc()
            return video_path

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                logger.info(f"📦 使用 imageio_ffmpeg: {ffmpeg}")
            except Exception as e:
                logger.error(f"❌ ffmpeg 不可用 (shutil.which + imageio_ffmpeg 均失败): {e}")
                return video_path
        else:
            logger.info(f"📦 使用系统 ffmpeg: {ffmpeg}")

        def probe_dur(path):
            r = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
            m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
            return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 0.0

        v_dur, a_dur = probe_dur(video_path), probe_dur(audio_path)
        base, _ = os.path.splitext(video_path)
        out_path = f"{base}_voiced.mp4"

        if a_dur > v_dur + 0.1 and v_dur > 0:
            cmd = [ffmpeg, "-y", "-stream_loop", "-1", "-i", video_path,
                   "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-preset", "fast", "-t", str(a_dur), out_path]
        else:
            cmd = [ffmpeg, "-y", "-i", video_path, "-i", audio_path,
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                   "-shortest", out_path]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            logger.info(f"✅ 配音合并完成 → {os.path.basename(out_path)}")
        else:
            logger.error(f"❌ ffmpeg 合并失败 (code={r.returncode}): {r.stderr[:300]}")
            # 清理临时音频
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass
            return video_path

        # 清理临时 TTS 音频文件（配音成功后不再需要）
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"🗑️ 已清理临时音频: {os.path.basename(audio_path)}")
        except Exception as e:
            logger.warning(f"⚠️ 清理临时音频失败: {e}")

        # 可选：混合背景配乐
        bg_music = voice_params.get("bg_music", "")
        if bg_music and os.path.exists(bg_music):
            merged_path = out_path
            final_path = out_path.replace(".mp4", "_bgm.mp4")
            bg_cmd = [
                ffmpeg, "-y", "-i", merged_path, "-i", bg_music,
                "-filter_complex",
                "[0:a]volume=1.0[a0];[1:a]volume=0.3[a1];[a0][a1]amix=inputs=2:duration=shortest[a]",
                "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                "-shortest", final_path,
            ]
            r2 = subprocess.run(bg_cmd, capture_output=True, text=True)
            if r2.returncode == 0:
                logger.info(f"✅ 背景音混合完成 → {os.path.basename(final_path)}")
                return final_path
            else:
                logger.warning(f"⚠️ 背景音混合失败 (code={r2.returncode}), 使用配音版")

        return out_path

    # ==========================================================
    #  按钮 / 回调
    # ==========================================================
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
    def _on_video_gen_error(self, code: int, error_msg: str):
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
        video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "photo", "videos")

        if not os.path.exists(video_dir):
            return

        videos = []
        for ext in ["*.mp4", "*.gif"]:
            videos.extend(glob.glob(os.path.join(video_dir, ext)))

        videos.sort(key=os.path.getmtime, reverse=True)

        # 去重：同一 base name 只保留配音版（_voiced/_bgm/_dubbed）
        seen_bases = {}
        for vpath in videos:
            base = os.path.splitext(os.path.basename(vpath))[0]
            clean_base = re.sub(r'(_voiced|_bgm|_dubbed)$', '', base)
            is_voiced = any(tag in base for tag in ('_voiced', '_bgm', '_dubbed'))
            if clean_base not in seen_bases or is_voiced:
                seen_bases[clean_base] = vpath
        videos = list(seen_bases.values())
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

    # ==========================================================
    #  Motion LoRA
    # ==========================================================
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