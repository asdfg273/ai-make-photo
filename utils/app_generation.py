# app_generation.py
# ============================================================
#  PyQt6 GenerationMixin — 修复版 (含 A-1/2/3/4 升级)
# ============================================================

import os
import threading
import datetime
import traceback
import random
import gc
import time
import torch

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from PyQt6.QtCore import QObject, pyqtSignal

from utils.app_utils       import OUTPUT_DIR, parse_dynamic_prompt
from utils.system_utils    import (performance_timer,
                                   generate_unique_filename, logger)
from utils.image_processor import make_comic_strip, process_adetailer


# ============================================================
#  线程安全信号桥
# ============================================================
class _GenBridge(QObject):
    status_signal   = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, int)
    sub_prog_signal = pyqtSignal(int, int)
    preview_signal  = pyqtSignal(str)
    preview_img_sig = pyqtSignal(object)
    done_signal     = pyqtSignal()
    error_signal    = pyqtSignal(str)
    cancel_signal   = pyqtSignal()
    log_signal      = pyqtSignal(str)
    image_signal    = pyqtSignal(object)


# ============================================================
class GenerationMixin:

    def _init_gen_bridge(self):
        self._bridge = _GenBridge()
        self._bridge.status_signal.connect(self._on_status)
        self._bridge.progress_signal.connect(self._on_progress)
        self._bridge.sub_prog_signal.connect(self._on_sub_progress)
        self._bridge.preview_signal.connect(self.show_preview)
        self._bridge.preview_img_sig.connect(self._on_preview_img)
        self._bridge.done_signal.connect(self._on_gen_done)
        self._bridge.error_signal.connect(self._on_error_to_log)
        self._bridge.cancel_signal.connect(self._on_cancelled)

    # ----------------------- 槽 -----------------------
    def _on_status(self, text: str, color: str):
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText(text)
            self.lbl_status.setStyleSheet(
                f"color:{color}; font-size:13px; font-weight:bold;")

    def _on_progress(self, val: int, maximum: int):
        if hasattr(self, 'progress_total') and self.progress_total:
            self.progress_total.setMaximum(max(maximum, 1))
            self.progress_total.setValue(val)

    def _on_sub_progress(self, val: int, maximum: int):
        if hasattr(self, 'progress') and self.progress:
            self.progress.setMaximum(max(maximum, 1))
            self.progress.setValue(val)

    def _on_preview_img(self, pil_img):
        if hasattr(self, 'update_preview_ui'):
            self.update_preview_ui(pil_img)

    def _on_gen_done(self):
        if hasattr(self, 'btn_gen'):     self.btn_gen.setEnabled(True)
        if hasattr(self, 'btn_stop'):    self.btn_stop.setEnabled(False)
        if hasattr(self, 'btn_edit'):    self.btn_edit.setEnabled(True)
        if hasattr(self, 'btn_upscale'): self.btn_upscale.setEnabled(True)

    def _on_gen_error(self, msg: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "生成错误", msg)

    def _on_cancelled(self):
        self._bridge.status_signal.emit("🛑 已打断", "#ff5555")

    # ----------------------- 控件读值 -----------------------
    def _cbo(self, w) -> str:
        return w.currentText() if w else ""

    def _chk(self, w) -> bool:
        if w is None: return False
        if hasattr(w, 'isChecked'): return bool(w.isChecked())
        if hasattr(w, 'get'):       return bool(w.get())
        return False

    def _sld(self, w) -> float:
        if w is None: return 0.0
        if hasattr(w, 'float_value'): return w.float_value()
        return float(w.value())

    def _spn(self, w) -> int:
        return int(w.value()) if w else 0

    def _txt(self, w) -> str:
        return w.toPlainText().strip() if w else ""

    def _edt(self, w) -> str:
        return w.text().strip() if w else ""

    # ----------------------------------------------------------
    def apply_adetailer(self, base_image, prompt, negative_prompt, seed,
                        target="现实脸部", strength=None):
        """简洁封装，正确匹配 process_adetailer 的真实签名"""
        if strength is None:
            strength = self._sld(
                getattr(self, 'scale_adetailer_strength', None))
        return process_adetailer(
            base_image, self.ai.inpaint_pipe,
            prompt, negative_prompt,
            strength=strength, target=target,
        )

    # ==================================================================
    #  启动生成
    # ==================================================================
    def start_generation(self):
        if not getattr(self, 'combo_model', None) or \
                self._cbo(self.combo_model) in ("未找到模型", ""):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告",
                                "请先在 models 文件夹中放入模型！")
            return

        if self._chk(getattr(self, 'chk_use_pose', None)) and \
                not getattr(self, 'pose_image_path', None):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告",
                                "启用了动作控制，但没有载入真人动作图！")
            return
        if getattr(self, 'is_generating', False):
            self._bridge.status_signal.emit("⚠️ 已有任务在运行", "#f38ba8")
            return
        self.is_generating = True
        self.cancel_flag   = False

        if hasattr(self, 'btn_gen'):  self.btn_gen.setEnabled(False)
        if hasattr(self, 'btn_stop'): self.btn_stop.setEnabled(True)
        if hasattr(self, 'btn_edit'): self.btn_edit.setEnabled(False)

        threading.Thread(target=self.generation_task, daemon=True).start()

    # ==================================================================
    #  主生成任务
    # ==================================================================
    def generation_task(self):
        try:
            print("\n🚀 [任务开始]")

            # --- 设备 ---
            device_str = self._cbo(getattr(self, 'combo_device', None))
            if   "CUDA" in device_str: target_device = "cuda"
            elif "MPS"  in device_str: target_device = "mps"
            elif "CPU"  in device_str: target_device = "cpu"
            else: target_device = "cuda" if torch.cuda.is_available() else "cpu"

            if getattr(self.ai, 'device', None) != target_device:
                print(f"🔄 切换设备: {self.ai.device} -> {target_device}")
                self.ai.device = target_device
                if hasattr(self.ai, 'clear_memory'):
                    self.ai.clear_memory()

            # --- 基础参数 ---
            model_name   = self._cbo(self.combo_model)
            raw_prompt   = self._txt(getattr(self, 'txt_prompt', None))
            raw_neg      = self._txt(getattr(self, 'txt_neg', None))
            en_neg       = self.translator.translate(raw_neg) if raw_neg else ""

            steps        = self._spn(getattr(self, 'spin_steps', None)) or 30
            strength     = self._sld(getattr(self, 'scale_str', None))
            cfg          = self._sld(getattr(self, 'scale_cfg', None))
            res_text     = self._cbo(getattr(self, 'combo_res', None))
            if 'x' in res_text:
                width, height = map(int, res_text.split('x'))
            else:
                width, height = 512, 512
            sampler_name = self._cbo(getattr(self, 'combo_sampler', None))

            # --- 动态提示词 ---
            parsed_raw_prompts = parse_dynamic_prompt(raw_prompt)
            base_count         = self._spn(
                getattr(self, 'spin_count', None)) or 1

            if len(parsed_raw_prompts) > 1:
                self._bridge.status_signal.emit(
                    f"📖 侦测到动态组合，将生成 "
                    f"{len(parsed_raw_prompts)} 页分镜...", "#ffd700")
                total_generate_count = len(parsed_raw_prompts)
            else:
                total_generate_count = base_count
                parsed_raw_prompts   = [parsed_raw_prompts[0]] * base_count

            en_prompts = [
                self.translator.translate(p) if p else ""
                for p in parsed_raw_prompts
            ]

            # --- 加载模型 ---
            self._bridge.status_signal.emit(
                "🧠 正在加载底层大模型...", "#ffd700")
            self.ai.load_model(model_name)

            # --- LoRA ---
            lora_config_list = []
            lora_meta_info   = []
            if hasattr(self, 'combo_loras') and hasattr(self, 'scale_loras'):
                for i in range(min(len(self.combo_loras),
                                   len(self.scale_loras))):
                    lname   = self._cbo(self.combo_loras[i])
                    lweight = self._sld(self.scale_loras[i])
                    if lname and lname != "无":
                        lora_config_list.append((lname, float(lweight)))
                        lora_meta_info.append(f"{lname}:{lweight:.2f}")

            sub_dir = "sdxl" if getattr(self.ai, 'is_sdxl', False) else "sd1.5"
            print(f"👉 LoRA 组合: {lora_config_list}")
            self.ai.apply_multiple_loras(lora_config_list, sub_dir=sub_dir)

            # --- ControlNet ---
            pose_image = None
            use_pose   = self._chk(getattr(self, 'chk_use_pose', None))
            if use_pose:
                cn_type = self._cbo(getattr(self, 'combo_cn_type', None))
                self._bridge.status_signal.emit(
                    f"⚙️ 解析 {cn_type} 参考图特征...", "#ffd700")
                self.ai.prepare_controlnet(control_type=cn_type)
                raw_img    = Image.open(
                    self.pose_image_path).convert("RGB")
                pose_image = self.ai.get_control_image(
                    raw_img, control_type=cn_type)
                self._bridge.preview_img_sig.emit(pose_image.copy())

            # --- 切采样器 ---
            self.ai.switch_sampler(sampler_name)

            # --- X/Y 炼丹分支 ---
            if self._chk(getattr(self, 'chk_enable_xy', None)):
                self._bridge.status_signal.emit(
                    "📊 进入 X/Y 炼丹模式...", "#ffd700")
                generator = torch.Generator(self.ai.device).manual_seed(
                    random.randint(1, 2_147_483_647))
                base_kwargs = self.ai.encode_prompt(en_prompts[0], en_neg)
                base_kwargs.update({
                    "num_inference_steps": steps,
                    "guidance_scale": cfg,
                    "generator": generator,
                })
                # 矩阵任务也需要 meta 信息
                xy_meta = {
                    "prompt": raw_prompt, "neg": raw_neg,
                    "en_prompt": en_prompts[0], "en_neg": en_neg,
                    "steps": steps, "sampler": sampler_name,
                    "cfg": cfg, "width": width, "height": height,
                    "model": model_name, "lora": lora_meta_info,
                }
                self.run_xy_plot_task(
                    base_kwargs, width, height,
                    pose_image=pose_image if use_pose else None,
                    meta=xy_meta,
                )
                return

            # --- 进度条 ---
            self._bridge.progress_signal.emit(0, total_generate_count)

            generated_images_list = []

            # ==================== 生成循环 ====================
            for i in range(total_generate_count):
                if getattr(self, 'cancel_flag', False):
                    break

                self._bridge.progress_signal.emit(i, total_generate_count)

                current_raw_prompt = parsed_raw_prompts[i]
                current_en_prompt  = en_prompts[i]
                current_seed       = random.randint(1, 2_147_483_647)
                generator          = torch.Generator(
                    self.ai.device).manual_seed(current_seed)

                self._bridge.status_signal.emit(
                    f"🔥 第 {i+1}/{total_generate_count} 张 "
                    f"(Seed: {current_seed}) ...", "#00ffff")
                self._bridge.sub_prog_signal.emit(0, steps)

                embed_kwargs = self.ai.encode_prompt(
                    current_en_prompt, en_neg)

                # ── ETA 步骤回调 (A-4) ──
                t0 = time.time()

                def step_cb(pipe, step_index, timestep, callback_kwargs,
                            _steps=steps, _t0=t0):
                    if getattr(self, 'cancel_flag', False):
                        raise InterruptedError()
                    done = step_index + 1
                    self._bridge.sub_prog_signal.emit(done, _steps)
                    # 节流:首帧/末帧/每3步刷新 ETA
                    if done == 1 or done == _steps or done % 3 == 0:
                        elapsed = time.time() - _t0
                        eta = ((elapsed / done) * (_steps - done)
                               if done > 0 else 0)
                        self._bridge.status_signal.emit(
                            f"🎨 第 {done}/{_steps} 步 · "
                            f"已用 {elapsed:.1f}s · "
                            f"预估剩余 {eta:.1f}s",
                            "#89dceb"
                        )
                    return self.on_generation_step(
                        pipe, step_index, timestep, callback_kwargs)

                kwargs = {
                    "num_inference_steps": steps,
                    "guidance_scale":      cfg,
                    "width":               width,
                    "height":              height,
                    "generator":           generator,
                    "callback_on_step_end": step_cb,
                    "callback_on_step_end_tensor_inputs": ["latents"],
                }
                kwargs.update(embed_kwargs)

                with torch.inference_mode():

                    # ── 阶段 1: 基础生成 ──
                    with performance_timer("🎨 阶段 1: 基础图像生成"):
                        if use_pose and pose_image:
                            image = self.ai.controlnet_pipe(
                                **kwargs, image=pose_image).images[0]

                        elif getattr(self, 'mask_image_path', None):
                            if getattr(self.ai, 'inpaint_pipe', None) is None:
                                self._bridge.status_signal.emit(
                                    "⏳ 正在加载 inpaint 管线...", "#fab387")
                                self.ai.load_model(model_name)
                                if getattr(self.ai, 'inpaint_pipe', None) is None:
                                    raise RuntimeError(
                                        "inpaint 管线加载失败,请确认模型兼容 inpaint。")

                            # 诊断日志
                            print(f"🎨 [INPAINT 分支] ref={self.ref_image_path}")
                            print(f"🎨 [INPAINT 分支] mask={self.mask_image_path}")

                            init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                            mask_img = Image.open(self.mask_image_path).convert("L").resize((width, height))

                            # 再检查一次遮罩非空
                            mn, mx = mask_img.getextrema()
                            print(f"🎨 [INPAINT 分支] 遮罩 extrema=({mn},{mx})")
                            if mx == 0:
                                print("⚠ 遮罩为全黑,退化为 img2img")
                                image = self.ai.img2img_pipe(
                                    **kwargs, image=init_img, strength=strength).images[0]
                            else:
                                image = self.ai.inpaint_pipe(
                                    **kwargs, image=init_img, mask_image=mask_img, strength=strength,
                                ).images[0]

                        elif getattr(self, 'ref_image_path', None):
                            init_img = Image.open(
                                self.ref_image_path
                            ).convert("RGB").resize((width, height))
                            image = self.ai.img2img_pipe(
                                **kwargs,
                                image    = init_img,
                                strength = strength,
                            ).images[0]

                        else:
                            image = self.ai.txt2img_pipe(**kwargs).images[0]

                    # ── 阶段 2: Hires.fix ──
                    if self._chk(getattr(self, 'chk_hires', None)):
                        self._bridge.status_signal.emit(
                            "✨ 阶段 2: Hires.fix 高清放大...", "#ff1493")
                        with performance_timer("Hires.fix 高清放大"):
                            hires_str = self._sld(
                                getattr(self, 'scale_hires', None))
                            image = self.ai.img2img_pipe(
                                **kwargs,
                                image    = image,
                                strength = hires_str,
                            ).images[0]

                    # ── 阶段 3: ADetailer 脸部 ──
                    if self._chk(getattr(self, 'chk_use_adetailer', None)):
                        self._bridge.status_signal.emit(
                            "✨ 阶段 3: ADetailer 脸部精修...", "#17a2b8")
                        face_target = self._cbo(
                            getattr(self, 'combo_ad_target', None)) or "现实脸部"
                        face_str    = self._sld(
                            getattr(self, 'scale_adetailer_strength', None))
                        with performance_timer("ADetailer 脸部精修"):
                            image = process_adetailer(
                                image, self.ai.inpaint_pipe,
                                current_en_prompt, en_neg,
                                strength=face_str, target=face_target)

                    # ── 阶段 4: ADetailer 手部 ──
                    if self._chk(getattr(self, 'chk_use_ad_hand', None)):
                        self._bridge.status_signal.emit(
                            "✨ 阶段 4: ADetailer 手部精修...", "#17a2b8")
                        hand_target = self._cbo(
                            getattr(self, 'combo_ad_hand', None)) or "现实手部"
                        hand_str    = self._sld(
                            getattr(self, 'scale_ad_hand', None))
                        blend_ratio = self._sld(
                            getattr(self, 'scale_ad_hand_blend', None))
                        blend_ratio = max(0.0, min(1.0,
                            blend_ratio if blend_ratio > 0 else 0.65))

                        with performance_timer("ADetailer 手部精修"):
                            original_image = image.copy()
                            repaired_image = process_adetailer(
                                image, self.ai.inpaint_pipe,
                                current_en_prompt, en_neg,
                                strength=hand_str, target=hand_target)
                            image = Image.blend(
                                original_image, repaired_image,
                                alpha=blend_ratio)

                # ── 保存 + 写 PNG 元数据 (A-3) ──
                generated_images_list.append(image)

                meta = {
                    "prompt":    current_raw_prompt,
                    "neg":       raw_neg,
                    "en_prompt": current_en_prompt,
                    "en_neg":    en_neg,
                    "steps":     steps,
                    "sampler":   sampler_name,
                    "cfg":       cfg,
                    "seed":      current_seed,
                    "width":     width,
                    "height":    height,
                    "model":     model_name,
                    "lora":      lora_meta_info,
                }
                filename  = generate_unique_filename(
                    prefix=f"v4_{current_seed}")
                save_path = os.path.join(OUTPUT_DIR, filename)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                self._save_with_meta(image, save_path, meta)
                self.last_generated_path = save_path

                self._bridge.preview_signal.emit(save_path)

            # ── 连环画 ──
            if (self._chk(getattr(self, 'chk_make_comic', None))
                    and len(generated_images_list) > 1
                    and not getattr(self, 'cancel_flag', False)):
                self.generate_comic_strip(generated_images_list)
            else:
                if not getattr(self, 'cancel_flag', False):
                    self._bridge.status_signal.emit(
                        "✅ 批量生成任务全部完成！", "#00ff00")

            self._bridge.progress_signal.emit(
                total_generate_count, total_generate_count)

        except InterruptedError:
            self._bridge.cancel_signal.emit()

        except Exception:
            err = traceback.format_exc()
            print(err)
            self._bridge.error_signal.emit(err)

        finally:
            self.is_generating = False
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._bridge.done_signal.emit()
            try:
                self.cleanup_temp_files(verbose=True)
            except Exception:
                pass
            self._bridge.done_signal.emit()

    # ==================================================================
    #  中断生成
    # ==================================================================
    def stop_generation(self):
        if getattr(self, 'is_generating', False):
            self.cancel_flag = True
            self._bridge.status_signal.emit(
                "🛑 正在打断，请等待当前步骤结束...", "#ff5555")
            if hasattr(self, 'btn_stop'):
                self.btn_stop.setEnabled(False)

    # ==================================================================
    #  步进回调
    # ==================================================================
    def on_generation_step(self, pipe, step_index, timestep, callback_kwargs):
        if getattr(self, 'cancel_flag', False):
            raise InterruptedError("用户中断")
        return callback_kwargs

    # ==================================================================
    #  保存图片 + 写入参数元数据 (WebUI 兼容) (A-3)
    # ==================================================================
    def _save_with_meta(self, img, save_path: str, meta: dict):
        """将图片写入磁盘,并在 PNG 中嵌入 'parameters' 文本块。"""
        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".png":
                pnginfo = PngInfo()
                params_text = (
                    f"{meta.get('prompt','')}\n"
                    f"Negative prompt: {meta.get('neg','')}\n"
                    f"Steps: {meta.get('steps','')}, "
                    f"Sampler: {meta.get('sampler','')}, "
                    f"CFG scale: {meta.get('cfg','')}, "
                    f"Seed: {meta.get('seed','')}, "
                    f"Size: {meta.get('width','')}x{meta.get('height','')}, "
                    f"Model: {meta.get('model','')}"
                )
                # 若有 LoRA 追加
                lora = meta.get('lora')
                if lora:
                    if isinstance(lora, (list, tuple)):
                        lora = ",".join(lora)
                    params_text += f", LoRAs: {lora}"
                pnginfo.add_text("parameters", params_text)
                img.save(save_path, pnginfo=pnginfo)
            else:
                img.save(save_path, quality=95)
        except Exception as e:
            try:
                img.save(save_path)
            except Exception:
                pass
            print(f"⚠️ 元数据写入失败: {e}")

    # ==================================================================
    #  X/Y 矩阵
    # ==================================================================
    def run_xy_plot_task(self, base_kwargs, width, height,
                         pose_image=None, meta=None):
        try:
            x_type = self._cbo(getattr(self, 'combo_x_type', None))
            y_type = self._cbo(getattr(self, 'combo_y_type', None))
            x_raw  = self._edt(getattr(self, 'entry_x_vals', None))
            y_raw  = self._edt(getattr(self, 'entry_y_vals', None))

            x_vals = [v.strip() for v in x_raw.split(',') if v.strip()]
            y_vals = [v.strip() for v in y_raw.split(',') if v.strip()]
            if not x_vals or not y_vals:
                self._bridge.error_signal.emit(
                    "X/Y 轴的值不能为空，请用英文逗号分隔。")
                return

            total_cells = len(x_vals) * len(y_vals)
            self._bridge.progress_signal.emit(0, total_cells)

            use_hires = self._chk(getattr(self, 'chk_enable_hires', None))
            hires_str = self._sld(getattr(self, 'scale_hires', None))

            cell_images = []
            counter     = 0

            for yi, y_v in enumerate(y_vals):
                row_images = []
                for xi, x_v in enumerate(x_vals):
                    if getattr(self, 'cancel_flag', False):
                        self._bridge.cancel_signal.emit()
                        return

                    counter += 1
                    self._bridge.status_signal.emit(
                        f"📊 XY 矩阵 {counter}/{total_cells}  "
                        f"X={x_type}:{x_v}  Y={y_type}:{y_v}", "#00ffff")

                    kwargs = dict(base_kwargs)
                    kwargs["width"]  = width
                    kwargs["height"] = height

                    self._apply_xy_param(kwargs, x_type, x_v)
                    self._apply_xy_param(kwargs, y_type, y_v)

                    _steps_ = kwargs.get("num_inference_steps", 30)
                    self._bridge.sub_prog_signal.emit(0, _steps_)

                    t0 = time.time()

                    def step_cb(pipe, step_index, timestep,
                                callback_kwargs,
                                _steps=_steps_, _t0=t0):
                        if getattr(self, 'cancel_flag', False):
                            raise InterruptedError()
                        done = step_index + 1
                        self._bridge.sub_prog_signal.emit(done, _steps)
                        if done == 1 or done == _steps or done % 3 == 0:
                            elapsed = time.time() - _t0
                            eta = ((elapsed / done) * (_steps - done)
                                   if done > 0 else 0)
                            self._bridge.status_signal.emit(
                                f"🎨 XY {counter}/{total_cells} · "
                                f"步 {done}/{_steps} · "
                                f"剩余 {eta:.1f}s",
                                "#89dceb"
                            )
                        return callback_kwargs

                    kwargs["callback_on_step_end"] = step_cb
                    kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

                    with torch.inference_mode():
                        if pose_image is not None:
                            img = self.ai.controlnet_pipe(
                                **kwargs, image=pose_image).images[0]
                        else:
                            img = self.ai.txt2img_pipe(**kwargs).images[0]

                        if use_hires:
                            img = self.ai.img2img_pipe(
                                **kwargs, image=img,
                                strength=hires_str).images[0]

                    row_images.append(img)
                    self._bridge.progress_signal.emit(counter, total_cells)

                cell_images.append(row_images)

            # 拼接 XY Grid
            grid_img = self._compose_xy_grid(
                cell_images, x_vals, y_vals, x_type, y_type)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, f"xy_grid_{ts}.png")

            if meta:
                grid_meta = dict(meta)
                grid_meta["seed"] = f"xy_{x_type}_{y_type}"
                self._save_with_meta(grid_img, save_path, grid_meta)
            else:
                grid_img.save(save_path)

            self.last_generated_path = save_path
            self._bridge.preview_signal.emit(save_path) 
            self._bridge.status_signal.emit(
                "✅ X/Y 矩阵生成完成！", "#00ff00")

        except InterruptedError:
            self._bridge.cancel_signal.emit()
        except Exception:
            self._bridge.error_signal.emit(traceback.format_exc())

    def _apply_xy_param(self, kwargs: dict, axis_type: str, val: str):
        """根据轴类型将字符串值写入 kwargs"""
        try:
            if axis_type == "Steps":
                kwargs["num_inference_steps"] = int(float(val))
            elif axis_type == "CFG Scale":
                kwargs["guidance_scale"] = float(val)
            elif axis_type == "Sampler":
                self.ai.switch_sampler(val)
            elif axis_type == "Seed":
                kwargs["generator"] = torch.Generator(
                    self.ai.device).manual_seed(int(float(val)))
            elif axis_type == "LoRA 权重":
                if hasattr(self, 'combo_loras') and self.combo_loras:
                    lname = self._cbo(self.combo_loras[0])
                    if lname and lname != "无":
                        sub_dir = ("sdxl"
                                   if getattr(self.ai, 'is_sdxl', False)
                                   else "sd1.5")
                        self.ai.apply_multiple_loras(
                            [(lname, float(val))], sub_dir=sub_dir)
        except Exception as e:
            print(f"⚠ 应用 XY 参数失败 ({axis_type}={val}): {e}")

    def _compose_xy_grid(self, cell_images, x_vals, y_vals,
                         x_type, y_type):
        """把矩阵拼成一张大图，带行列标签"""
        if not cell_images or not cell_images[0]:
            return Image.new("RGB", (256, 256), "black")

        cell_w, cell_h = cell_images[0][0].size
        margin_top  = 60
        margin_left = 180

        rows = len(y_vals)
        cols = len(x_vals)
        total_w = margin_left + cell_w * cols
        total_h = margin_top  + cell_h * rows

        grid = Image.new("RGB", (total_w, total_h), "#1e1e2e")
        try:
            font = ImageFont.truetype("msyh.ttc", 18)
        except Exception:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(grid)

        # 列标签
        for ci, xv in enumerate(x_vals):
            tx = margin_left + ci * cell_w + cell_w // 2
            draw.text((tx, 20), f"{x_type}={xv}",
                      fill="#cba6f7", font=font, anchor="mm")

        # 行标签
        for ri, yv in enumerate(y_vals):
            ty = margin_top + ri * cell_h + cell_h // 2
            draw.text((margin_left // 2, ty), f"{y_type}={yv}",
                      fill="#a6e3a1", font=font, anchor="mm")

        # 粘贴单元
        for ri, row in enumerate(cell_images):
            for ci, img in enumerate(row):
                grid.paste(
                    img,
                    (margin_left + ci * cell_w,
                     margin_top  + ri * cell_h)
                )
        return grid

    # ==================================================================
    #  连环画
    # ==================================================================
    def generate_comic_strip(self, image_list):
        try:
            self._bridge.status_signal.emit(
                "🖼 正在拼合分镜连环画...", "#ffd700")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, f"comic_{ts}.png")
            comic = make_comic_strip(image_list)
            comic.save(save_path)
            self.last_generated_path = save_path   
            self._bridge.preview_signal.emit(save_path)
            self._bridge.status_signal.emit(
                "✅ 连环画拼合完成！", "#00ff00")
        except Exception:
            self._bridge.error_signal.emit(traceback.format_exc())

    # ==================================================================
    #  高清放大 (后处理按钮) — BUG FIX
    # ==================================================================
    def start_upscale(self):
        if not getattr(self, 'last_generated_path', None):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "提示",
                                "请先生成一张图片再进行放大！")
            return
        if getattr(self, 'is_generating', False):
            return

        self.is_generating = True
        self.cancel_flag   = False
        if hasattr(self, 'btn_gen'):     self.btn_gen.setEnabled(False)
        if hasattr(self, 'btn_upscale'): self.btn_upscale.setEnabled(False)
        if hasattr(self, 'btn_stop'):    self.btn_stop.setEnabled(True)

        threading.Thread(target=self._upscale_task, daemon=True).start()

    def _resolve_device(self, raw: str) -> str:
        """把下拉框里的任意格式('AUTO' / '自动 (auto)' / '显卡 (cuda)' 等)
           解析为 torch 能识别的 'cuda' / 'mps' / 'cpu'。"""
        import re
        s = (raw or "").lower()

        # 抽出括号里的英文值,如 "自动 (auto)" → "auto"
        m = re.search(r'\(([^)]+)\)', s)
        if m:
            s = m.group(1).strip()

        # 去掉多余空格/中文字符
        s = s.strip()

        if "cuda" in s or "gpu" in s or "显卡" in s:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if "mps" in s or "苹果" in s:
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if "cpu" in s:
            return "cpu"
        if "auto" in s or "自动" in s or s == "":
            if torch.cuda.is_available():
                return "cuda"
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"

        # 未知 → 安全兜底
        return "cuda" if torch.cuda.is_available() else "cpu"


    def _upscale_task(self):
        try:
            # ─────────── ★ 兜底:确保 img2img 管线已就绪 ───────────
            if getattr(self.ai, "img2img_pipe", None) is None:
                self._bridge.status_signal.emit(
                    "⏳ 正在加载图生图管线(首次放大)...", "#fab387")

                model_name = self._cbo(getattr(self, 'combo_model', None))
                if not model_name or model_name in (
                        "（未选择）", "（无模型）", "无", ""):
                    raise RuntimeError(
                        "未选择主模型,无法放大。\n"
                        "请先在「基础」标签里选择 SD 模型。")

                device = self._resolve_device(
                    self._cbo(getattr(self, 'combo_device', None)))


                # 调用模型管理器加载(方法名按你的 ModelManager 实际为准)
                try:
                    if hasattr(self.ai, "device"):
                        self.ai.device = device
                except Exception:
                    pass

                # 兼容多种 ModelManager 签名
                import inspect
                if hasattr(self.ai, "load_model"):
                    sig = inspect.signature(self.ai.load_model)
                    params = sig.parameters
                    call_kwargs = {}
                    if "device" in params:
                        call_kwargs["device"] = device
                    try:
                        self.ai.load_model(model_name, **call_kwargs)
                    except TypeError:
                        # 兜底:只传模型名
                        self.ai.load_model(model_name)
                elif hasattr(self.ai, "load"):
                    try:
                        self.ai.load(model_name)
                    except TypeError:
                        self.ai.load(model_name, device=device)
                else:
                    raise RuntimeError(
                        "ModelManager 没有 load_model 方法,无法自动加载。")

                if getattr(self.ai, "img2img_pipe", None) is None:
                    raise RuntimeError(
                        "img2img 管线加载失败,请检查模型文件是否完整。")

                self._bridge.status_signal.emit(
                    "✅ 图生图管线已就绪,开始放大...", "#a6e3a1")

            # ─────────── 原有放大逻辑 ───────────
            self._bridge.status_signal.emit(
                "🔍 正在进行高清放大...", "#ffd700")
            src_path = self.last_generated_path
            img      = Image.open(src_path).convert("RGB")

            scale_text = self._cbo(
                getattr(self, 'combo_hires_scale', None)) or "2.0"
            scale      = float(scale_text)
            denoise    = self._sld(
                getattr(self, 'scale_hires_denoise', None))

            new_w = int(img.width  * scale)
            new_h = int(img.height * scale)
            enlarged = img.resize((new_w, new_h), Image.LANCZOS)

            raw_prompt = self._txt(getattr(self, 'txt_prompt', None))
            raw_neg    = self._txt(getattr(self, 'txt_neg', None))
            en_prompt  = (self.translator.translate(raw_prompt)
                          if raw_prompt else "")
            en_neg     = (self.translator.translate(raw_neg)
                          if raw_neg else "")

            embed_kwargs = self.ai.encode_prompt(en_prompt, en_neg)

            up_steps = 25
            up_cfg   = 6.5
            t0       = time.time()
            self._bridge.sub_prog_signal.emit(0, up_steps)

            def step_cb(pipe, step_index, timestep, callback_kwargs,
                        _steps=up_steps, _t0=t0):
                if getattr(self, 'cancel_flag', False):
                    raise InterruptedError()
                done = step_index + 1
                self._bridge.sub_prog_signal.emit(done, _steps)
                if done == 1 or done == _steps or done % 3 == 0:
                    elapsed = time.time() - _t0
                    eta = ((elapsed / done) * (_steps - done)
                           if done > 0 else 0)
                    self._bridge.status_signal.emit(
                        f"🔍 放大 {done}/{_steps} · "
                        f"已用 {elapsed:.1f}s · "
                        f"剩余 {eta:.1f}s",
                        "#ffd700"
                    )
                return callback_kwargs

            kwargs = {
                "num_inference_steps": up_steps,
                "guidance_scale":      up_cfg,
                "strength":            denoise,
                "image":               enlarged,
                "callback_on_step_end": step_cb,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
            kwargs.update(embed_kwargs)

            # ⚠ 修复:用 img2img_pipe,不是 txt2img_pipe
            with torch.inference_mode():
                img = self.ai.img2img_pipe(**kwargs).images[0]

            ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUTPUT_DIR, f"upscale_{ts}.png")
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            meta = {
                "prompt":    raw_prompt,
                "neg":       raw_neg,
                "en_prompt": en_prompt,
                "en_neg":    en_neg,
                "steps":     up_steps,
                "sampler":   self._cbo(
                    getattr(self, 'combo_sampler', None)) or "",
                "cfg":       up_cfg,
                "seed":      "upscale",
                "width":     img.width,
                "height":    img.height,
                "model":     self._cbo(
                    getattr(self, 'combo_model', None)) or "",
            }
            self._save_with_meta(img, out_path, meta)
            self.last_generated_path = out_path
            self._bridge.preview_signal.emit(out_path)
            self._bridge.status_signal.emit(
                "✅ 高清放大完成！", "#00ff00")

        except InterruptedError:
            self._bridge.cancel_signal.emit()
        except Exception:
            self._bridge.error_signal.emit(traceback.format_exc())
        finally:
            self.is_generating = False
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._bridge.done_signal.emit()

    # ==================================================================
    #  动态提示词解析 (备用版)
    # ==================================================================
    def parse_dynamic_prompt_fallback(self, prompt_text: str):
        """
        兼容 {A|B|C} 多组笛卡尔展开。
        若 utils.app_utils.parse_dynamic_prompt 已存在则用那个。
        """
        import re
        if not prompt_text:
            return [""]

        pattern = re.compile(r"\{([^{}]+)\}")
        matches = pattern.findall(prompt_text)
        if not matches:
            return [prompt_text]

        from itertools import product
        groups = [[x.strip() for x in m.split('|')] for m in matches]
        final_prompts = []
        for combo in product(*groups):
            temp_prompt = prompt_text
            for replacement in combo:
                temp_prompt = pattern.sub(
                    replacement, temp_prompt, count=1)
            final_prompts.append(temp_prompt)
        return final_prompts

    def _on_error_to_log(self, err_text: str):
        """错误信息打到日志框,不弹窗,便于复制"""
        self.set_status("❌ 生成出错 — 详情见下方日志", "#f38ba8")

        self.append_log("─" * 60, "#f38ba8")
        self.append_log("❌ [错误堆栈]", "#f38ba8")
        for line in err_text.rstrip().splitlines():
            safe = (line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace(" ", "&nbsp;"))
            self.append_log(safe, "#f38ba8")
        self.append_log("─" * 60, "#f38ba8")

        # 兜底恢复按钮状态
        try:
            self.is_generating = False
            if hasattr(self, 'btn_generate'):
                self.btn_generate.setEnabled(True)
            if hasattr(self, 'btn_interrupt'):
                self.btn_interrupt.setEnabled(False)
        except Exception:
            pass