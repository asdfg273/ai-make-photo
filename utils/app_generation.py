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
    live_preview_signal = pyqtSignal(str) 
    preview_img_sig = pyqtSignal(object)
    done_signal     = pyqtSignal()
    error_signal    = pyqtSignal(str)
    cancel_signal   = pyqtSignal()
    log_signal      = pyqtSignal(str)
    image_signal    = pyqtSignal(object)
    gallery_add_signal  = pyqtSignal(str)
    enhance_done_signal  = pyqtSignal(str) 


# ============================================================
class GenerationMixin:

    def _init_gen_bridge(self):
        self._bridge = _GenBridge()
        self._bridge.status_signal.connect(self._on_status)
        self._bridge.progress_signal.connect(self._on_progress)
        self._bridge.enhance_done_signal.connect(self._on_enhance_done)
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
        from utils.vram_manager import VRAMManager
        VRAMManager.cleanup()
        VRAMManager.print_status()

        ctx = None
        try:
            print("\n🚀 [任务开始]", flush=True)

            # 1. 准备上下文 (设备/参数/翻译)
            ctx = self._gt_prepare_context()
            if ctx is None:
                return

            # 2. 加载底模
            self._gt_load_model(ctx)

            # 3. 配置 IP-Adapter
            self._gt_setup_ipa(ctx)

            # 4. 应用 LoRA
            self._gt_apply_loras(ctx)

            # 5. Pose Transfer (会修改 ctx['use_pose']/'pose_image')
            self._gt_run_pose_transfer(ctx)

            # 6. 切采样器
            print(f"🟢 step 14: 切采样器 = {ctx['sampler_name']}", flush=True)
            self.ai.switch_sampler(ctx['sampler_name'])

            # 7. X/Y 炼丹分支 (返回 True 表示已处理)
            if self._gt_try_xy_plot(ctx):
                return

            # 8. 主生成循环
            self._gt_main_loop(ctx)

        except InterruptedError:
            print("⏸ [generation_task] 用户中断", flush=True)
            try: self._bridge.cancel_signal.emit()
            except Exception: pass

        except Exception:
            err = traceback.format_exc()
            print("❌ [generation_task] 异常:", flush=True)
            print(err, flush=True)
            try: self._bridge.error_signal.emit(err)
            except Exception: pass

        finally:
            self._gt_cleanup()

    def _gt_prepare_context(self):
        """收集所有参数到一个 ctx 字典,贯穿整个流程。"""
        import torch
        from utils.app_utils import parse_dynamic_prompt

        # --- 设备 ---
        device_str = self._cbo(getattr(self, 'combo_device', None))
        if   "CUDA" in device_str: target_device = "cuda"
        elif "MPS"  in device_str: target_device = "mps"
        elif "CPU"  in device_str: target_device = "cpu"
        else: target_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🟢 step 2: 设备 = {target_device}", flush=True)

        if getattr(self.ai, 'device', None) != target_device:
            print(f"🔄 切换设备: {self.ai.device} -> {target_device}", flush=True)
            self.ai.device = target_device
            if hasattr(self.ai, 'clear_memory'):
                self.ai.clear_memory()

        model_data = self.combo_model.currentData() if hasattr(self, 'combo_model') else None
        if isinstance(model_data, dict) and 'name' in model_data:
            model_name = model_data['name']
            model_type = model_data.get('type', '')  # 顺便记录类型
            print(f"🟢 模型: [{model_type}] {model_name}", flush=True)
        else:
            # 兜底:剥离 [xxGB] 后缀
            raw = self._cbo(getattr(self, 'combo_model', None))
            model_name = raw.split('  [')[0].strip() if raw else ""
            model_type = ''
            print(f"🟢 模型(兜底): {model_name}", flush=True)

        raw_prompt = self._txt(getattr(self, 'txt_prompt', None))
        raw_neg    = self._txt(getattr(self, 'txt_neg', None))

        mode_idx = self.combo_trans_mode.currentIndex() if hasattr(self, 'combo_trans_mode') else 2
        trans_mode = ["dict", "ai", "auto"][mode_idx]
        print(f"🌐 翻译模式: {trans_mode}", flush=True)

        # --- AI 改写 ---
        raw_prompt = self._gt_apply_prompt_enhance(raw_prompt)

        # --- 翻译反向词 ---
        en_neg = self.translator.translate(raw_neg, mode=trans_mode) if raw_neg else ""

        # --- 数值参数 ---
        steps    = self._spn(getattr(self, 'spin_steps', None)) or 30
        strength = self._sld(getattr(self, 'scale_str', None))
        cfg      = self._sld(getattr(self, 'scale_cfg', None))
        res_text = self._cbo(getattr(self, 'combo_res', None))
        if 'x' in res_text:
            width, height = map(int, res_text.split('x'))
        else:
            width, height = 512, 512
        sampler_name = self._cbo(getattr(self, 'combo_sampler', None))
        cn_strength = self._sld(getattr(self, 'scale_cn_strength', None)) or 1.0

        raw_pt = self._sld(getattr(self, 'slider_pt_cn', None))
        if raw_pt is None or raw_pt == 0:
            pt_cn_strength = 0.65
        elif raw_pt > 5:
            pt_cn_strength = raw_pt / 100.0
        else:
            pt_cn_strength = raw_pt

        # --- 动态提示词 (Wildcards 批量) ---
        parsed_raw_prompts = parse_dynamic_prompt(raw_prompt)
        base_count = self._spn(getattr(self, 'spin_count', None)) or 1
        combo_count = len(parsed_raw_prompts)

        if combo_count > 1:
            total_generate_count = combo_count * base_count

            SAFETY_THRESHOLD = 12
            if total_generate_count > SAFETY_THRESHOLD:
                if not self._gt_confirm_batch(combo_count, base_count, total_generate_count):
                    print(f"⏸ 用户取消批量队列 ({total_generate_count} 张)", flush=True)
                    self._bridge.status_signal.emit("⏸ 已取消批量生成", "#f38ba8")
                    return None

            expanded = []
            for p in parsed_raw_prompts:
                expanded.extend([p] * base_count)
            parsed_raw_prompts = expanded

            self._bridge.status_signal.emit(
                f"📖 批量队列启动: {combo_count} 组合 × {base_count} 张 = {total_generate_count} 张",
                "#ffd700"
            )
            print(f"🎯 [批量] {combo_count} 组合 × {base_count} 张/组 = 共 {total_generate_count} 张", flush=True)
        else:
            total_generate_count = base_count
            parsed_raw_prompts   = [parsed_raw_prompts[0]] * base_count

        extracted_features = ""
        ref_img = getattr(self.ai, 'ipa_ref_image', None)
        auto_extract = self._chk(getattr(self, 'chk_auto_features', None))

        if ref_img is not None and auto_extract:
            try:
                from utils.prompt_enhancer import PromptEnhancer
                enhancer = (PromptEnhancer.instance()
                            if hasattr(PromptEnhancer, 'instance')
                            else PromptEnhancer())

                self._bridge.status_signal.emit(
                    "🔍 正在分析参考图角色特征...", "#7ed957")

                extracted_features = enhancer.extract_character_features(ref_img)
                if extracted_features:
                    print(f"✨ 角色特征:\n   {extracted_features}", flush=True)
            except Exception as e:
                print(f"⚠️ 特征提取失败(跳过): {e}", flush=True)
                extracted_features = ""

        # ── 翻译每个 prompt,并把特征拼到最前面 ──
        en_prompts = []
        for p in parsed_raw_prompts:
            en = self.translator.translate(p, mode=trans_mode) if p else ""
            if extracted_features:
                en = f"{extracted_features}, {en}" if en else extracted_features
            en_prompts.append(en)

        if extracted_features:
            print(f"📝 注入特征后第一条 prompt:\n   {en_prompts[0][:200]}...", flush=True)

        # ── 返回 ctx ──
        return {
            # === 设备/模型 ===
            'device':            target_device,
            'model_name':        model_name,
            'model_type':        model_type,   

            # === 提示词 ===
            'raw_prompt':        raw_prompt,
            'raw_neg':           raw_neg,
            'en_neg':            en_neg,
            'parsed_raw_prompts': parsed_raw_prompts,
            'en_prompts':        en_prompts,

            # === 数值 ===
            'steps':             steps,
            'strength':          strength,
            'cfg':               cfg,
            'width':             width,
            'height':            height,
            'sampler_name':      sampler_name,
            'cn_strength':       cn_strength,
            'pt_cn_strength':    pt_cn_strength,
            'total_count':       total_generate_count,

            # === IP-Adapter ===
            'use_ipa':           False,
            'ipa_pil_image':     None,
            'ipa_scale':         0.9,
            'ipa_variant':       "plus",

            # === 流程开关 ===
            'use_img2img':       False,
            'use_pose':          False,
            'pose_image':        None,
            'pose_transfer_used': False,
            'skip_img2img':      False,
            'init_image_path':   None,
            'use_inpaint':       False,

            # === 角色一致性 ===
            'extracted_features':  extracted_features,
            'use_reference_only':  False,
            'ref_fidelity':        0.7,

            # === 其他 ===
            'lora_meta_info':    [],
        }

    def _gt_apply_prompt_enhance(self, raw_prompt):
        if not self._chk(getattr(self, 'chk_auto_enhance', None)):
            return raw_prompt
        try:
            from utils.prompt_enhancer import PromptEnhancer
            enhancer = PromptEnhancer()
            enhancer.load()
            new_prompt = enhancer.enhance(raw_prompt)
            if new_prompt and new_prompt.strip():
                self._bridge.status_signal.emit(
                    f"✨ 已智能改写: {new_prompt[:60]}...", "#b48ead")
                return new_prompt
        except Exception as e:
            print(f"⚠️ 智能改写失败: {e}", flush=True)
        return raw_prompt

    def _gt_load_model(self, ctx):
        self._bridge.status_signal.emit("🧠 正在加载底层大模型...", "#ffd700")
        print(f"🟢 step 9: 调用 ai.load_model({ctx['model_name']})", flush=True)
        self.ai.load_model(ctx['model_name'])

    def _gt_setup_ipa(self, ctx):
        use_ipa = (self.chk_use_ipa.isChecked()
                   if hasattr(self, 'chk_use_ipa') else False)
        ipa_image_path = getattr(self, 'ipa_image_path', None)
        ipa_scale = (self.spin_ipa_scale.value()
                     if hasattr(self, 'spin_ipa_scale') else 0.6)
        ipa_variant_text = (self.combo_ipa_variant.currentText()
                            if hasattr(self, 'combo_ipa_variant') else "plus")
        ipa_variant = "plus" if "plus" in ipa_variant_text else "standard"

        # 校验
        if use_ipa and not ipa_image_path:
            self._bridge.status_signal.emit(
                "⚠️ IP-Adapter 已开启但未加载参考图,自动跳过", "#fab387")
            use_ipa = False
        if use_ipa and ipa_image_path and not os.path.exists(ipa_image_path):
            self._bridge.status_signal.emit(
                f"⚠️ 角色参考图丢失: {ipa_image_path},自动跳过", "#fab387")
            use_ipa = False

        ipa_pil_image = None
        if use_ipa:
            ipa_pil_image = Image.open(ipa_image_path).convert("RGB")
            print(f"🟢 IP-Adapter 参考图已加载 {ipa_pil_image.size}", flush=True)

            self._bridge.status_signal.emit(
                "🎭 正在加载 IP-Adapter (角色一致性)...", "#fab387")
            ok = self.ai.prepare_ip_adapter(variant=ipa_variant)
            if ok:
                self.ai.set_ip_adapter_scale(ipa_scale)
                # 诊断
                try:
                    from diffusers.models.attention_processor import (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
                    )
                    ipa_classes = (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    ipa_count = bad_count = 0
                    for n, p in self.ai.txt2img_pipe.unet.attn_processors.items():
                        if 'attn2' in n:
                            if isinstance(p, ipa_classes): ipa_count += 1
                            else: bad_count += 1
                    print(f"🟢 [诊断] IPA attn2: {ipa_count} 正确 / {bad_count} 错误",
                          flush=True)
                    if bad_count > 0 or ipa_count == 0:
                        self._bridge.status_signal.emit(
                            "⚠️ IP-Adapter 安装异常,自动卸载", "#fab387")
                        try: self.ai.unload_ip_adapter()
                        except Exception: pass
                        use_ipa = False
                        ipa_pil_image = None
                except Exception as e:
                    print(f"⚠️ 诊断 IPA 异常: {e}", flush=True)
            else:
                try: self.ai.unload_ip_adapter()
                except Exception: pass
                use_ipa = False
                ipa_pil_image = None
        else:
            if getattr(self.ai, 'ip_adapter_loaded', False):
                self.ai.unload_ip_adapter()

        ctx['use_ipa']        = use_ipa
        ctx['ipa_pil_image']  = ipa_pil_image
        ctx['ipa_scale']      = ipa_scale
        ctx['ipa_variant']    = ipa_variant
        ctx['ipa_image_path'] = ipa_image_path

    def _gt_apply_loras(self, ctx):
        lora_config_list = []
        lora_meta_info   = []
        if hasattr(self, 'combo_loras') and hasattr(self, 'scale_loras'):
            for i in range(min(len(self.combo_loras), len(self.scale_loras))):
                lname   = self._cbo(self.combo_loras[i])
                lweight = self._sld(self.scale_loras[i])
                if lname and lname != "无":
                    lora_config_list.append((lname, float(lweight)))
                    lora_meta_info.append(f"{lname}:{lweight:.2f}")

        sub_dir = "sdxl" if getattr(self.ai, 'is_sdxl', False) else "sd1.5"
        if lora_config_list:
            self.ai.apply_multiple_loras(lora_config_list, sub_dir=sub_dir)
            print(f"🟢 LoRA 已应用: {lora_config_list}", flush=True)
        ctx['lora_meta_info'] = lora_meta_info

    def _gt_run_pose_transfer(self, ctx):
    # 🔧 必须先定义这两个变量
        use_pose_transfer = self._chk(getattr(self, 'chk_pose_transfer', None))
        use_pose_manual   = self._chk(getattr(self, 'chk_use_pose', None))

        try:
            import torch
            ie = getattr(self.ai.txt2img_pipe, "image_encoder", None)
            if ie is not None:
                # 如果被 CPU offload hook 挂住,先摘掉
                if hasattr(ie, "_hf_hook"):
                    try:
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(ie, recurse=True)
                    except Exception:
                        pass
                ie.to(device="cuda", dtype=torch.float16)
                print(f"🔧 image_encoder → cuda / float16")
        except Exception as e:
            print(f"⚠️ image_encoder 迁移失败: {e}")

        # ── 分支 A: 手动 ControlNet 模式(用户自己上传骨架图) ──
        if use_pose_manual and not use_pose_transfer:
            ctx['use_pose']   = True
            ctx['pose_image'] = getattr(self, 'pose_image', None)
            if ctx['pose_image'] is not None:
                self.ai.prepare_controlnet("openpose")
                # 🔧 同时开 IPA 时,要同步到 controlnet_pipe
                if ctx['use_ipa']:
                    self.ai.sync_ipa_to_controlnet()
                    try:
                        self.ai.controlnet_pipe.set_ip_adapter_scale(ctx['ipa_scale'])
                    except Exception:
                        pass
            # ⚠️ 手动模式用基础 CN 滑块的值,不覆盖
            return

        # ── 分支 B: 不开 Pose Transfer 直接返回 ──
        if not use_pose_transfer:
            return

        # ── 分支 C: Pose Transfer 三阶段 ──
        print("🎬 进入 Pose Transfer 模式", flush=True)
        ctx['pose_transfer_used'] = True

        # 必须有角色参考图
        char_ref_path = ctx.get('ipa_image_path') or getattr(self, 'ref_image_path', None)
        if not char_ref_path:
            raise RuntimeError(
                "❌ Pose Transfer 模式需要角色参考图!\n"
                "请先在【图生图】Tab 上传 IP-Adapter 角色参考图。"
            )

        # 自动开 IPA(如果用户没勾)
        if not ctx['use_ipa']:
            self._bridge.status_signal.emit(
                "🎭 自动启用 IP-Adapter (Pose Transfer 必需)...", "#fab387")
            ctx['ipa_pil_image'] = Image.open(char_ref_path).convert("RGB")
            ok = self.ai.prepare_ip_adapter(variant="plus")
            if not ok:
                raise RuntimeError("Pose Transfer 模式需要 IP-Adapter,但加载失败")
            ctx['ipa_scale']   = 0.9  
            ctx['use_ipa']     = True
            ctx['ipa_variant'] = "plus"
            self.ai.set_ip_adapter_scale(ctx['ipa_scale'])

        # ── Stage 1: 生成动作参考图 ──
        self._bridge.status_signal.emit(
            "🎬 [1/3] Pose Transfer: 生成动作参考图...", "#ffd700")
        print("🎬 [Stage 1/3] 生成动作参考图...", flush=True)

        stage1_prompt = (
            f"masterpiece, best quality, 1girl, solo, full body, simple white background, "
            f"{ctx['en_prompts'][0]}"
        )
        stage1_gen = torch.Generator(self.ai.device).manual_seed(
            random.randint(1, 2_147_483_647))

        # 临时关 IPA 生成纯净姿势参考图
        saved_scale = ctx['ipa_scale']
        try:
            self.ai.txt2img_pipe.set_ip_adapter_scale(0.0)
        except Exception:
            pass

        dummy_ipa = Image.new("RGB", (224, 224), (0, 0, 0))
        with torch.inference_mode():
            pose_ref_img = self.ai.txt2img_pipe(
                prompt=stage1_prompt,
                negative_prompt=ctx['en_neg'],
                width=ctx['width'], height=ctx['height'],
                num_inference_steps=min(20, ctx['steps']),
                guidance_scale=ctx['cfg'],
                generator=stage1_gen,
                ip_adapter_image=dummy_ipa,
            ).images[0]

        try:
            self.ai.txt2img_pipe.set_ip_adapter_scale(saved_scale)
        except Exception:
            pass

        try:
            self._bridge.preview_img_sig.emit(pose_ref_img.copy())
        except Exception:
            pass

        # ── Stage 2: 提取骨架 + 同步 IPA 到 ControlNet pipe ──
        self._bridge.status_signal.emit(
            "🦴 [2/3] 提取 OpenPose 骨架...", "#ffd700")
        self.ai.prepare_controlnet("openpose")
        skeleton_img = self.ai.get_control_image(pose_ref_img, "openpose")

        # 🔧 关键:同步 IPA(否则 Stage 3 会报 tuple 错误)
        if ctx['use_ipa']:
            sync_ok = self.ai.sync_ipa_to_controlnet()
            if not sync_ok:
                self._bridge.status_signal.emit(
                    "⚠️ IPA 同步到 CN 失败,Stage 3 不使用角色一致性", "#fab387")
                ctx['use_ipa'] = False
                ctx['ipa_pil_image'] = None
            else:
                try:
                    self.ai.controlnet_pipe.set_ip_adapter_scale(ctx['ipa_scale'])
                except Exception:
                    pass

        ctx['cn_strength'] = ctx['pt_cn_strength']
        print(f"🎯 Pose Transfer 参数: CN={ctx['cn_strength']:.2f}, "
              f"IPA={ctx['ipa_scale']:.2f}", flush=True)

        try:
            self._bridge.preview_img_sig.emit(skeleton_img.copy())
        except Exception:
            pass

        # ── Stage 3: 配置上下文,主循环里完成最终生成 ──
        ctx['use_pose']     = True
        ctx['pose_image']   = skeleton_img
        ctx['skip_img2img'] = True
        self._bridge.status_signal.emit(
            "🎨 [3/3] 准备最终生成...", "#ffd700")



    def _gt_safe_pipe_call(self, pipe, ctx, **call_kwargs):
        """
        统一 pipeline 调用入口:
        1. 自动处理 IPA 占位图 (UNet 被污染时)
        2. 支持 Compel 长提示词 (>77 tokens)
        """
        unet_has_ipa = (
            pipe is not None
            and getattr(pipe, 'unet', None) is not None
            and getattr(pipe.unet, 'encoder_hid_proj', None) is not None
        )

        if unet_has_ipa:
            user_provided = call_kwargs.get("ip_adapter_image") is not None
            if not user_provided:
                # 喂 dummy + scale=0
                call_kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (0, 0, 0))
                try:
                    pipe.set_ip_adapter_scale(0.0)
                    print("🔧 [safe_call] UNet 含 IPA 但未启用 → dummy + scale=0", flush=True)
                except Exception:
                    pass
            else:
                # 用户启用了 IPA,设置正确 scale
                try:
                    pipe.set_ip_adapter_scale(ctx.get('ipa_scale', 0.6))
                except Exception:
                    pass

        if 'prompt' in call_kwargs and call_kwargs['prompt']:
            compel = None
        
            # 根据 pipe 类型选择对应的 Compel 实例
            if pipe == self.ai.txt2img_pipe:
                compel = getattr(self.ai, 'compel_txt2img', None)
            elif pipe == self.ai.img2img_pipe:
                compel = getattr(self.ai, 'compel_img2img', None)
            elif pipe == self.ai.controlnet_pipe:
                compel = getattr(self.ai, 'compel_controlnet', None)
        
            if compel is not None:
                try:
                    # 正向提示词
                    prompt_embeds = compel(call_kwargs['prompt'])
                    call_kwargs['prompt_embeds'] = prompt_embeds
                    call_kwargs.pop('prompt')  # 删掉原始 prompt
                
                    # 负向提示词
                    if 'negative_prompt' in call_kwargs and call_kwargs['negative_prompt']:
                        neg_embeds = compel(call_kwargs['negative_prompt'])
                        call_kwargs['negative_prompt_embeds'] = neg_embeds
                        call_kwargs.pop('negative_prompt')
                
                    print(f"✅ Compel 已处理长提示词 ({prompt_embeds.shape[1]} tokens)", flush=True)
                except Exception as e:
                    print(f"⚠️ Compel 处理失败,降级为截断: {e}", flush=True)

        try:
            output = pipe(**call_kwargs)
        
            # 恢复 IPA scale (供下次调用)
            if unet_has_ipa and ctx.get('use_ipa'):
                try:
                    pipe.set_ip_adapter_scale(ctx['ipa_scale'])
                except Exception:
                    pass
        
            return output
        
        except InterruptedError:
            raise
        except Exception as e:
            print(f"❌ Pipeline 调用失败: {e}", flush=True)
            raise

    def _gt_try_xy_plot(self, ctx):
        if not self._chk(getattr(self, 'chk_enable_xy', None)):
            return False

        print("🟢 进入 XY 分支", flush=True)
        self._bridge.status_signal.emit("📊 进入 X/Y 炼丹模式...", "#ffd700")
        generator = torch.Generator(self.ai.device).manual_seed(
            random.randint(1, 2_147_483_647))

        if ctx['use_ipa']:
            base_kwargs = {"prompt": ctx['en_prompts'][0],
                           "negative_prompt": ctx['en_neg']}
        else:
            base_kwargs = self.ai.encode_prompt(ctx['en_prompts'][0], ctx['en_neg'])

        base_kwargs.update({
            "num_inference_steps": ctx['steps'],
            "guidance_scale":      ctx['cfg'],
            "generator":           generator,
        })
        if ctx['use_ipa'] and ctx['ipa_pil_image'] is not None:
            base_kwargs["ip_adapter_image"] = ctx['ipa_pil_image']

        xy_meta = {
            "prompt": ctx['raw_prompt'], "neg": ctx['raw_neg'],
            "en_prompt": ctx['en_prompts'][0], "en_neg": ctx['en_neg'],
            "steps": ctx['steps'], "sampler": ctx['sampler_name'],
            "cfg": ctx['cfg'], "width": ctx['width'], "height": ctx['height'],
            "model": ctx['model_name'], "lora": ctx['lora_meta_info'],
        }
        self.run_xy_plot_task(
            base_kwargs, ctx['width'], ctx['height'],
            pose_image=ctx['pose_image'] if ctx['use_pose'] else None,
            meta=xy_meta,
        )
        return True

    def _gt_main_loop(self, ctx):
        self._bridge.progress_signal.emit(0, ctx['total_count'])
        generated_paths = []

        if (ctx['use_ipa'] and getattr(self, 'ref_image_path', None)
                and not ctx['skip_img2img']):
            self._bridge.status_signal.emit(
                "💡 IP-Adapter 已启用,自动忽略图生图参考图(避免冲突)", "#fab387")

        for i in range(ctx['total_count']):
            if getattr(self, 'cancel_flag', False):
                break

            print(f"🟢 [{i+1}/{ctx['total_count']}] 开始", flush=True)
            self._bridge.progress_signal.emit(i, ctx['total_count'])

            current_seed = random.randint(1, 2_147_483_647)
            self._bridge.status_signal.emit(
                f"🔥 第 {i+1}/{ctx['total_count']} 张 (Seed: {current_seed}) ...",
                "#00ffff")
            self._bridge.sub_prog_signal.emit(0, ctx['steps'])

            try:
                image = self._gt_generate_one(ctx, i, current_seed)
            except InterruptedError:
                raise
            except Exception as e:
                print(f"❌ 第 {i+1} 张生成失败: {e}", flush=True)
                traceback.print_exc()
                continue

            # 保存
            save_path = self._gt_save_image(image, ctx, i, current_seed)
            if not save_path:
                continue

            self.last_generated_path = save_path
            generated_paths.append(save_path)
            try: self._bridge.preview_img_sig.emit(image.copy())
            except Exception: pass
            try: self._bridge.gallery_add_signal.emit(save_path)
            except Exception: pass
            self._bridge.progress_signal.emit(i + 1, ctx['total_count'])

        if not getattr(self, 'cancel_flag', False):
            self._bridge.status_signal.emit(
                f"✅ 全部完成! 共生成 {len(generated_paths)} 张", "#00ff00")
        self._bridge.progress_signal.emit(ctx['total_count'], ctx['total_count'])

    def _gt_generate_one(self, ctx, i, current_seed):
        generator = torch.Generator(self.ai.device).manual_seed(current_seed)
        en_prompt = ctx['en_prompts'][i]
        en_neg    = ctx['en_neg']

        # prompt embeds vs 普通 prompt
        if ctx['use_ipa'] and ctx['ipa_pil_image'] is not None:
            embed_kwargs = {"prompt": en_prompt, "negative_prompt": en_neg}
        else:
            embed_kwargs = self.ai.encode_prompt(en_prompt, en_neg)

        # ETA 回调
        cur_prompt = ctx['en_prompts'][i] if i < len(ctx['en_prompts']) else ""
        prompt_preview = cur_prompt[:35] + "..." if len(cur_prompt) > 35 else cur_prompt

        t0 = time.time()
        def step_cb(pipe, step_index, timestep, callback_kwargs,
                    _steps=ctx['steps'], _t0=t0, _preview=prompt_preview):
            if getattr(self, 'cancel_flag', False):
                raise InterruptedError()
            done = step_index + 1
            self._bridge.sub_prog_signal.emit(done, _steps)
            if done == 1 or done == _steps or done % 3 == 0:
                elapsed = time.time() - _t0
                eta = (elapsed / done) * (_steps - done) if done > 0 else 0
                self._bridge.status_signal.emit(
                    f"🔥 [{i+1}/{ctx['total_count']}] {done}/{_steps} | "
                    f"ETA {eta:.1f}s | Seed:{current_seed} | {_preview}",
                    "#00ffff"
                )
            return self.on_generation_step(pipe, step_index, timestep, callback_kwargs)

        kwargs = {
            "num_inference_steps": ctx['steps'],
            "guidance_scale":      ctx['cfg'],
            "width":               ctx['width'],
            "height":              ctx['height'],
            "generator":           generator,
            "callback_on_step_end": step_cb,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        kwargs.update(embed_kwargs)
        if ctx['use_ipa'] and ctx['ipa_pil_image'] is not None:
            kwargs["ip_adapter_image"] = ctx['ipa_pil_image']

        with torch.inference_mode():
            # ── 阶段 1: 基础图像生成 ──
            with performance_timer("🎨 阶段 1: 基础图像生成"):
                image = self._gt_run_base_pipe(ctx, kwargs)

            # ── 阶段 2: Hires.fix ──
            if self._chk(getattr(self, 'chk_hires', None)):
                if ctx['use_ipa']:
                    self._bridge.status_signal.emit(
                        "⚠️ IP-Adapter 模式跳过 Hires.fix", "#fab387")
                else:
                    self._bridge.status_signal.emit(
                        "✨ 阶段 2: Hires.fix...", "#ff1493")
                    with performance_timer("Hires.fix"):
                        hires_str = self._sld(getattr(self, 'scale_hires', None))
                        hires_kwargs = {k: v for k, v in kwargs.items()
                                        if k not in ('width','height','ip_adapter_image')}
                        image = self._gt_safe_pipe_call(
                            self.ai.img2img_pipe, ctx,
                            **hires_kwargs, image=image, strength=hires_str
                        ).images[0]

            # ── 阶段 3: ADetailer 脸部 ──
            if self._chk(getattr(self, 'chk_use_adetailer', None)):
                self._bridge.status_signal.emit(
                    "✨ 阶段 3: ADetailer 脸部精修...", "#17a2b8")
                face_target = self._cbo(getattr(self, 'combo_ad_target', None)) or "现实脸部"
                face_str    = self._sld(getattr(self, 'scale_adetailer_strength', None))
    
                # 🔧 ADetailer 前临时禁用 IPA (避免污染)
                saved_scale, has_ipa = self._adetailer_disable_ipa()
    
                try:
                    with performance_timer("ADetailer 脸部"):
                        image = process_adetailer(
                            image, self.ai.inpaint_pipe,
                            en_prompt, en_neg,
                            strength=face_str, target=face_target)
                finally:
                    self._adetailer_restore_ipa(saved_scale, has_ipa)

            # ── 阶段 4: ADetailer 手部 ──
            if self._chk(getattr(self, 'chk_use_ad_hand', None)):
                self._bridge.status_signal.emit(
                    "✨ 阶段 4: ADetailer 手部精修...", "#17a2b8")
                hand_target = self._cbo(getattr(self, 'combo_ad_hand', None)) or "现实手部"
                hand_str    = self._sld(getattr(self, 'scale_ad_hand', None))
                blend_ratio = self._sld(getattr(self, 'scale_ad_hand_blend', None))
                blend_ratio = max(0.0, min(1.0, blend_ratio if blend_ratio > 0 else 0.65))
    
                # 🔧 ADetailer 前临时禁用 IPA
                saved_scale, has_ipa = self._adetailer_disable_ipa()
    
                try:
                    with performance_timer("ADetailer 手部"):
                        original_image = image.copy()
                        repaired_image = process_adetailer(
                            image, self.ai.inpaint_pipe,
                            en_prompt, en_neg,
                            strength=hand_str, target=hand_target)
                        image = Image.blend(original_image, repaired_image,
                                            alpha=blend_ratio)
                finally:
                    self._adetailer_restore_ipa(saved_scale, has_ipa)

        # 确保 RGB
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"image 类型异常: {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _gt_run_base_pipe(self, ctx, kwargs):
        """根据上下文选择合适的 pipeline 调用,统一走 _gt_safe_pipe_call。"""
        use_pose          = ctx['use_pose']
        use_img2img       = ctx['use_img2img']
        use_inpaint       = ctx['use_inpaint']
        use_reference     = ctx.get('use_reference_only', False)
        skip_img2img      = ctx.get('skip_img2img', False)
        use_img2img       = ctx.get('use_img2img', False)
        init_img          = ctx.get('init_image')
        inpaint_img       = ctx.get('inpaint_image')
        inpaint_mask      = ctx.get('inpaint_mask')
        pose_image        = ctx.get('pose_image')
        strength          = ctx.get('strength', 0.7)
        cn_strength       = ctx.get('cn_strength', 0.6)
        ref_image         = getattr(self.ai, 'ipa_ref_image', None)
        ref_fidelity      = ctx.get('ref_fidelity', 0.7)

        ip_kwargs = {}
        if getattr(self.ai, 'ip_adapter_loaded', False) and ref_image is not None:
            ip_kwargs['ip_adapter_image'] = ref_image

        output = None

        # ── 路径 0: Reference-Only (单图角色一致性最强方案) ──
        if use_reference and ref_image is not None and not use_pose and not use_inpaint:
            ref_pipe = getattr(self.ai, 'reference_pipe', None)
            if ref_pipe is None:
                print("⏳ 首次使用 Reference-Only,正在准备...", flush=True)
                self.ai.prepare_reference_only()
                ref_pipe = self.ai.reference_pipe
        
            # Reference-Only 不接受 ip_adapter_image 等参数,得过滤
            ref_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ('prompt', 'negative_prompt', 'num_inference_steps',
                                   'guidance_scale', 'width', 'height', 'generator',
                                   'prompt_embeds', 'negative_prompt_embeds')}
        
            print(f"🪞 [Reference-Only] fidelity={ref_fidelity:.2f}", flush=True)
            output = ref_pipe(
                ref_image=ref_image,
                reference_attn=True,
                reference_adain=True,
                style_fidelity=ref_fidelity,
                **ref_kwargs,
            )

        # ── 路径 1: ControlNet (Pose Transfer 或手动) ──
        elif use_pose and pose_image is not None:
            output = self._gt_safe_pipe_call(
                self.ai.controlnet_pipe, ctx,
                **kwargs, **ip_kwargs,
                image=pose_image,
                controlnet_conditioning_scale=cn_strength,
            )

        # ── 路径 2: Inpaint ──
        elif use_inpaint and inpaint_img is not None and inpaint_mask is not None:
            output = self._gt_safe_pipe_call(
                self.ai.inpaint_pipe, ctx,
                **kwargs, **ip_kwargs,
                image=inpaint_img, mask_image=inpaint_mask, strength=strength,
            )

        # ── 路径 3: 跳过 img2img (Pose Transfer 已用 CN) ──
        elif skip_img2img:
            output = self._gt_safe_pipe_call(
                self.ai.txt2img_pipe, ctx, **kwargs, **ip_kwargs)

        # ── 路径 4: img2img ──
        elif use_img2img and init_img is not None:
            output = self._gt_safe_pipe_call(
                self.ai.img2img_pipe, ctx,
                **kwargs, **ip_kwargs,
                image=init_img, strength=strength,
            )

        # ── 路径 5: 纯 txt2img ──
        else:
            output = self._gt_safe_pipe_call(
                self.ai.txt2img_pipe, ctx, **kwargs, **ip_kwargs)

        # ── 统一提取 image ──
        if output is None:
            raise RuntimeError("pipeline 返回 None")
        if hasattr(output, 'images'):
            return output.images[0]
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def _gt_save_image(self, image, ctx, i, current_seed):
        """成功返回 save_path,失败返回 None。"""
        # 元数据
        meta = None
        try:
            from PIL.PngImagePlugin import PngInfo
            meta = PngInfo()
            def _add(key, val):
                try:
                    s = "" if val is None else str(val)
                    if len(s) > 8000: s = s[:8000] + "...(truncated)"
                    meta.add_text(key, s)
                except Exception as e:
                    print(f"⚠️ 元数据 [{key}] 写入失败: {e}", flush=True)

            _add("prompt",      ctx['parsed_raw_prompts'][i])
            _add("negative",    ctx['raw_neg'])
            _add("en_prompt",   ctx['en_prompts'][i])
            _add("en_negative", ctx['en_neg'])
            _add("model",       ctx['model_name'])
            _add("sampler",     ctx['sampler_name'])
            _add("steps",       ctx['steps'])
            _add("cfg",         ctx['cfg'])
            _add("size",        f"{ctx['width']}x{ctx['height']}")
            _add("seed",        current_seed)
            _add("lora",        ",".join(ctx['lora_meta_info']))
            _add("use_ipa",     bool(ctx['use_ipa']))
            if ctx['use_ipa']:
                _add("ipa_scale",   f"{ctx['ipa_scale']:.2f}")
                _add("ipa_variant", ctx['ipa_variant'])
            _add("use_pose",      bool(ctx['use_pose']))
            _add("cn_strength",   f"{ctx['cn_strength']:.2f}")
            _add("pose_transfer", bool(ctx['pose_transfer_used']))
            _add("timestamp",     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            a1111 = (
                f"{ctx['en_prompts'][i]}\n"
                f"Negative prompt: {ctx['en_neg']}\n"
                f"Steps: {ctx['steps']}, Sampler: {ctx['sampler_name']}, "
                f"CFG scale: {ctx['cfg']}, Seed: {current_seed}, "
                f"Size: {ctx['width']}x{ctx['height']}, Model: {ctx['model_name']}"
            )
            _add("parameters", a1111)
        except Exception as e:
            print(f"⚠️ 构建 PngInfo 失败: {e}", flush=True)
            meta = None

        # 文件名
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = "".join(c if c.isalnum() or c in "._-" else "_"
                             for c in (ctx['model_name'] or "model"))[:40]
        filename = f"{ts}_{safe_model}_{i+1:02d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        try:
            if meta is not None:
                image.save(save_path, format="PNG", pnginfo=meta,
                           optimize=False, compress_level=4)
            else:
                image.save(save_path, format="PNG", optimize=False, compress_level=4)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
                print(f"💾 已保存: {save_path} "
                      f"({os.path.getsize(save_path)//1024} KB)", flush=True)
                return save_path
            print(f"⚠️ 保存的文件大小异常: {save_path}", flush=True)
        except Exception as e:
            print(f"⚠️ 主保存失败: {e},降级保存...", flush=True)
            try:
                image.save(save_path, format="PNG")
                if os.path.exists(save_path):
                    print(f"💾 [降级] 已保存: {save_path}", flush=True)
                    return save_path
            except Exception as e2:
                print(f"❌ 降级保存也失败: {e2}", flush=True)
                traceback.print_exc()
        return None

    def _gt_cleanup(self):
        print("🏁 [generation_task] 结束", flush=True)
        self.is_generating = False
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception: pass
        try: self.cleanup_temp_files(verbose=True)
        except Exception: pass
        try: self._bridge.done_signal.emit()
        except Exception: pass

    def _adetailer_disable_ipa(self):
        """
        ADetailer 前临时禁用 IPA,返回 (saved_scale, has_ipa)
        用于配合 _adetailer_restore_ipa 恢复
        """
        inpaint_pipe = self.ai.inpaint_pipe
        has_ipa = False
        saved_scale = None
    
        try:
            for proc in inpaint_pipe.unet.attn_processors.values():
                if 'IPAdapter' in type(proc).__name__:
                    has_ipa = True
                    break
        except Exception:
            pass
    
        if has_ipa:
            try:
                saved_scale = getattr(self.ai, '_ipa_scale', 0.7)
                inpaint_pipe.set_ip_adapter_scale(0.0)
                print(f"  🔧 [ADetailer] 临时禁用 IPA (saved scale={saved_scale:.2f})",
                      flush=True)
            except Exception as e:
                print(f"  ⚠️ [ADetailer] 禁用 IPA 失败: {e}", flush=True)
    
        return saved_scale, has_ipa


    def _adetailer_restore_ipa(self, saved_scale, has_ipa):
        """ADetailer 跑完后恢复 IPA scale"""
        if not has_ipa or saved_scale is None:
            return
        try:
            self.ai.inpaint_pipe.set_ip_adapter_scale(saved_scale)
            print(f"  ✅ [ADetailer] IPA scale 已恢复 ({saved_scale:.2f})",
                  flush=True)
        except Exception as e:
            print(f"  ⚠️ [ADetailer] 恢复 IPA 失败: {e}", flush=True)

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
    
        try:
            enable = (
                hasattr(self, "chk_use_preview")
                and self.chk_use_preview.isChecked()
            )
        
            if not enable:
                return callback_kwargs
        
            interval = (
                self.spin_preview_interval.value()
                if hasattr(self, "spin_preview_interval")
                else 10
            )
            step_now = step_index + 1
        
            if step_now % interval != 0:
                return callback_kwargs
        
            latents = callback_kwargs.get("latents")
        
            if latents is None:
                return callback_kwargs
        
            import torch
            from PIL import Image
            import numpy as np
        
            with torch.no_grad():
                scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                lat = latents / scale
                img = pipe.vae.decode(lat).sample
                img = (img / 2 + 0.5).clamp(0, 1)
                img = img.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                img = (img * 255).astype(np.uint8)
                pil = Image.fromarray(img)
            
            
                print(f"[PREVIEW-5] 准备发图 size={pil.size}", flush=True)

                try:
                    import os, tempfile
                    if not hasattr(self, "_preview_tmp_dir"):
                        self._preview_tmp_dir = os.path.join(
                            tempfile.gettempdir(), "ai_preview"
                        )
                        os.makedirs(self._preview_tmp_dir, exist_ok=True)
    
                    tmp_path = os.path.join(
                        self._preview_tmp_dir, "live_preview.png"
                    )
                    pil.save(tmp_path, "PNG")
    
                    if hasattr(self, "_bridge"):
                        self._bridge.live_preview_signal.emit(tmp_path)
                        print(f"[PREVIEW-6] live_preview_signal 已发送: {tmp_path}", flush=True)

                except Exception as e:
                    print(f"❌ [PREVIEW-SAVE-EXC] {e}", flush=True)

        except Exception as e:
            import traceback
            print(f"❌ [PREVIEW-EXC] {e}", flush=True)
            traceback.print_exc()
    
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
            print(f"[EMIT-1341] preview_signal: {save_path}", flush=True)
            self._bridge.preview_signal.emit(save_path)
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
            print(f"[EMIT-1432] preview_signal: {save_path}", flush=True)
            self._bridge.preview_signal.emit(save_path)
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

    #各类辅助函数

    def _gt_confirm_batch(self, combo_count: int, per_count: int, total: int) -> bool:
        """
        子线程请求主线程弹出确认对话框。
        使用 QMetaObject.invokeMethod + BlockingQueuedConnection 阻塞等待返回。
        （无 context 的 QTimer.singleShot 会在调用线程执行，子线程无事件循环，
          对话框永远不会弹出 —— 已废弃该写法。）
        """
        from PyQt6.QtCore import QMetaObject, Qt, QThread

        result_holder = {'ok': False}

        def _ask_on_main():
            from PyQt6.QtWidgets import QMessageBox
            msg = (
                f"⚠️ 批量队列确认\n\n"
                f"侦测到 {combo_count} 个 prompt 组合\n"
                f"每个组合生成 {per_count} 张\n"
                f"共计 {total} 张图\n\n"
                f"预估耗时: 约 {total * 8} 秒\n\n"
                f"是否继续？"
            )
            reply = QMessageBox.question(
                self, "批量队列确认", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            result_holder['ok'] = (reply == QMessageBox.StandardButton.Yes)

        if QThread.currentThread() is self.thread():
            _ask_on_main()
        else:
            QMetaObject.invokeMethod(
                self, _ask_on_main,
                Qt.ConnectionType.BlockingQueuedConnection)
        return result_holder['ok']

    def _start_batch_queue(self, prompt_list: list):
        """批量生成队列"""
        self._batch_queue = prompt_list.copy()
        self._batch_total = len(prompt_list)
        self._batch_done = 0
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        """跑队列里的下一个"""
        if not self._batch_queue:
            self._set_status(f"✅ 批量完成: {self._batch_total} 张", "#a6e3a1")
            return
    
        next_prompt = self._batch_queue.pop(0)
        self._batch_done += 1
    
        # 更新 UI 提示词框（让用户看到当前在跑哪个）
        self.txt_prompt.setPlainText(next_prompt)
    
        self._set_status(
            f"🎯 队列 [{self._batch_done}/{self._batch_total}]: {next_prompt[:30]}...",
            "#89b4fa"
        )
    
        # 调用原有单次生成，结束后回调跑下一个
        self._start_single_generation(next_prompt, on_done=self._run_next_in_queue)

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