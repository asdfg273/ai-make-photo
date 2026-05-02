# app_generation.py

import os
import threading
import datetime
import traceback
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from ttkbootstrap.constants import END, NORMAL, DISABLED

# 导入你的自定义工具
from utils.app_utils import OUTPUT_DIR, parse_dynamic_prompt
from utils.system_utils import performance_timer, generate_unique_filename, logger
from utils.image_processor import make_comic_strip, process_adetailer

class GenerationMixin:
    """专职负责：多线程生图、X/Y 矩阵生成、高清放大、后处理分发"""

    def apply_adetailer(self, base_image, prompt, negative_prompt, seed):
        is_anime = (self.combo_adetailer_model.get() == "二次元脸")
        strength = float(self.scale_adetailer_strength.get())
        
        def status_cb(msg, color):
            self.after(0, lambda: self.lbl_status.config(text=msg, foreground=color))
            
        return process_adetailer(
            base_image=base_image, prompt=prompt, negative_prompt=negative_prompt,
            seed=seed, is_anime=is_anime, strength=strength,
            img2img_pipe=self.ai.img2img_pipe, device=self.ai.device, status_callback=status_cb
        )

    def start_generation(self):
        if not getattr(self, 'combo_model', None) or self.combo_model.get() == "未找到模型":
            from tkinter import messagebox
            messagebox.showwarning("警告", "请先在 models 文件夹中放入模型！")
            return
        if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() and not getattr(self, 'pose_image_path', None):
            from tkinter import messagebox
            messagebox.showwarning("警告", "启用了动作控制，但没有载入真人动作图！")
            return
            
        self.is_generating = True
        self.cancel_flag = False
        self.btn_gen.config(state=DISABLED)
        self.btn_stop.config(state=NORMAL)
        self.btn_edit.config(state=DISABLED)
        threading.Thread(target=self.generation_task, daemon=True).start()

    def generation_task(self):
        try:
            print("\n🚀 [任务开始] 正在初始化生成参数...")
            device_str = self.combo_device.get()
            target_device = "cpu" 
            
            if "CUDA" in device_str:
                target_device = "cuda"
            elif "MPS" in device_str:
                target_device = "mps"
            elif "CPU" in device_str:
                target_device = "cpu"
            else:
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if getattr(self.ai, 'device', None) != target_device:
                print(f"🔄 正在切换计算设备: {self.ai.device} -> {target_device}")
                self.ai.device = target_device
                if hasattr(self.ai, 'clear_memory'):
                    self.ai.clear_memory()
            
            model_name = self.combo_model.get()
            
            raw_prompt = self.txt_prompt.get("1.0", END).strip()
            raw_neg = self.txt_neg.get("1.0", END).strip()
            en_neg = self.translator.translate(raw_neg) if raw_neg else ""
            
            parsed_raw_prompts = parse_dynamic_prompt(raw_prompt)
            base_count = int(self.spin_count.get())
            
            if len(parsed_raw_prompts) > 1:
                self.after(0, lambda: self.lbl_status.config(text=f"📖 侦测到动态组合，将生成 {len(parsed_raw_prompts)} 页分镜...", foreground="info"))
                total_generate_count = len(parsed_raw_prompts)
            else:
                total_generate_count = base_count
                parsed_raw_prompts = [parsed_raw_prompts[0]] * base_count

            en_prompts = [self.translator.translate(p) if p else "" for p in parsed_raw_prompts]
            
            steps = int(self.spin_steps.get())
            strength = self.scale_str.get()
            width, height = map(int, self.combo_res.get().split('x'))

            self.after(0, lambda: self.lbl_status.config(text="🧠 正在加载底层大模型...", foreground="yellow"))
            self.ai.load_model(model_name)
            
            lora_config_list = []
            lora_meta_info = [] 
            for i in range(3):
                lname = self.combo_loras[i].get()
                lweight = self.scale_loras[i].get()
                if lname and lname != "无":
                    lora_config_list.append((lname, float(lweight)))
                    lora_meta_info.append(f"{lname}:{lweight}")

            sub_dir = "sdxl" if self.ai.is_sdxl else "sd1.5"
            print(f"👉 准备融合 LoRA 组合: {lora_config_list}")
            self.ai.apply_multiple_loras(lora_config_list, sub_dir=sub_dir)
            
            pose_image = None
            cn_type = "openpose"
            if getattr(self, 'var_use_pose', None) and self.var_use_pose.get():
                cn_type = self.combo_cn_type.get()
                self.after(0, lambda: self.lbl_status.config(text=f"⚙️ 正在解析 {cn_type} 参考图特征...", foreground="yellow"))
                self.ai.prepare_controlnet(control_type=cn_type)
                raw_img = Image.open(self.pose_image_path).convert("RGB")
                pose_image = self.ai.get_control_image(raw_img, control_type=cn_type)
                self.after(0, lambda p=pose_image: self.show_pose_preview(p.copy()))

            generated_images_list = [] 
            self.after(0, lambda: self.progress_total.configure(value=0, maximum=total_generate_count))

            sampler_name = self.combo_sampler.get()
            self.ai.switch_sampler(sampler_name)

            for i in range(total_generate_count):
                if getattr(self, 'cancel_flag', False): break
                self.after(0, lambda idx=i: self.progress_total.configure(value=idx))
            
                current_raw_prompt = parsed_raw_prompts[i]
                current_en_prompt = en_prompts[i]
                current_seed = random.randint(1, 2147483647)
                generator = torch.Generator(self.ai.device).manual_seed(current_seed)

                self.after(0, lambda idx=i, s=current_seed: self.lbl_status.config(text=f"🔥 第 {idx+1}/{total_generate_count} 张 (Seed: {s}) ...", foreground="cyan"))
                self.after(0, lambda: self.progress.configure(value=0, maximum=steps))

                embed_kwargs = self.ai.encode_prompt(current_en_prompt, en_neg)

                def step_cb(pipe, step_index, timestep, callback_kwargs):
                    if getattr(self, 'cancel_flag', False): raise InterruptedError()
                    self.after(0, lambda: self.progress.configure(value=step_index + 1))
                    return self.on_generation_step(pipe, step_index, timestep, callback_kwargs)

                kwargs = {
                    "num_inference_steps": int(self.spin_steps.get()),
                    "guidance_scale": float(self.scale_cfg.get()), 
                    "width": width,
                    "height": height,
                    "generator": generator,
                    "callback_on_step_end": step_cb,                  
                    "callback_on_step_end_tensor_inputs": ["latents"]      
                }
                kwargs.update(embed_kwargs)

                with torch.no_grad():
                    # 1. 拦截 X/Y 图表任务（它内部有自己的循环，直接交接出去）
                    if getattr(self, "var_enable_xy", None) and self.var_enable_xy.get():
                        self.run_xy_plot_task(kwargs, width, height, pose_image=pose_image if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() else None)
                        return
    
                    # 2. 基础生成阶段（带监控计时）
                    with performance_timer("🎨 阶段 1: 基础图像生成"):
                        if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() and pose_image:
                            image = self.ai.controlnet_pipe(**kwargs, image=pose_image).images[0]
                        elif getattr(self, 'mask_image_path', None): 
                            init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                            mask_img = Image.open(self.mask_image_path).convert("L").resize((width, height))
                            image = self.ai.inpaint_pipe(**kwargs, image=init_img, mask_image=mask_img, strength=strength).images[0]
                        elif getattr(self, 'ref_image_path', None):
                            init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                            image = self.ai.img2img_pipe(**kwargs, image=init_img, strength=strength).images[0]
                        else:
                            image = self.ai.txt2img_pipe(**kwargs).images[0]

                    # 3. 高清重绘阶段（带监控计时）
                    if getattr(self, "var_enable_hires", None) and self.var_enable_hires.get():
                        self.after(0, lambda: self.lbl_status.config(text="✨ 构图完成！正在进行 Hires. fix 高清重绘...", foreground="#00FFFF"))
                        with performance_timer("✨ 阶段 2: Hires.fix 高清重绘"):
                            hires_scale = float(self.combo_hires_scale.get())
                            denoise_strength = float(self.scale_hires_denoise.get())
                            target_w, target_h = int(width * hires_scale), int(height * hires_scale)
            
                            base_upscaled = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                            hires_kwargs = kwargs.copy()
                            hires_kwargs["image"] = base_upscaled
                            hires_kwargs["strength"] = denoise_strength
            
                            image = self.ai.img2img_pipe(**hires_kwargs).images[0]

                # 4. ADetailer 面部修复（注意：ADetailer 内部有独立的流程，可能不需要 no_grad 或者模型自身有管理，放在外层更安全）
                if getattr(self, 'var_use_adetailer', None) and self.var_use_adetailer.get() and not getattr(self, 'cancel_flag', False):
                    self.after(0, lambda: self.lbl_status.config(text="🧑‍🎨 正在进行 ADetailer 脸部修复...", foreground="yellow"))
                    with performance_timer("🧑‍🎨 阶段 3: ADetailer 面部精修"):
                        image = self.apply_adetailer(image, current_en_prompt, en_neg, current_seed)

                # 5. 收尾工作：存入列表与写入元数据
                generated_images_list.append(image) 

                lora_str = ", ".join(lora_meta_info) if lora_meta_info else "无"
                metadata_str = f"{current_raw_prompt}\nNegative prompt: {raw_neg}\nSteps: {steps}, Size: {width}x{height}, Seed: {current_seed}, Model: {model_name}, LoRAs: {lora_str}"
                pnginfo = PngInfo()
                pnginfo.add_text("parameters", metadata_str)

                filename = generate_unique_filename(prefix="AI_Gen")
                save_path = os.path.join(OUTPUT_DIR, filename)
                image.save(save_path, pnginfo=pnginfo)
                filename = generate_unique_filename(prefix=f"v4_{current_seed}")
                save_path = os.path.join(OUTPUT_DIR, filename) 
            
                image.save(save_path, pnginfo=pnginfo)
                self.after(0, lambda p=save_path: self.show_preview(p))

            if getattr(self, "var_make_comic", None) and self.var_make_comic.get() and len(generated_images_list) > 1 and not getattr(self, 'cancel_flag', False):
                self.generate_comic_strip(generated_images_list)
            else:
                if not getattr(self, 'cancel_flag', False):
                    self.after(0, lambda: self.lbl_status.config(text="✅ 批量生成任务全部完成！", foreground="#00FF00"))

            self.after(0, lambda: self.progress_total.configure(value=total_generate_count))
        
        except InterruptedError:
            self.after(0, lambda: self.lbl_status.config(text="🛑 已打断", foreground="red"))
        except Exception as e:
            print(f"\n❌ [致命错误] 生成线程在后台崩溃，详细报告如下：\n{traceback.format_exc()}")
            self.after(0, lambda: self.lbl_status.config(text="❌ 生成崩溃，请查看终端红字报错", foreground="red"))
        finally:
            self.is_generating, self.cancel_flag = False, False
            self.after(0, lambda: self.btn_gen.config(state=NORMAL))
            if hasattr(self, 'btn_stop'):
                self.after(0, lambda: self.btn_stop.config(state=DISABLED))
            print("🧹 [系统维护] 正在清理碎片...")
            import gc
            gc.collect()
            if 'torch' in globals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def on_generation_step(self, pipe, step_index, timestep, callback_kwargs):
        if step_index % 4 == 0:
            try:
                latents = callback_kwargs["latents"]
                latents_cpu = latents[0].detach().cpu().float()
                weights = torch.tensor([
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ])
                img_tensor = torch.einsum("chw, ck -> khw", latents_cpu, weights)
                img_tensor = ((img_tensor + 1.0) / 2.0).clamp(0, 1)
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                
                preview_img = Image.fromarray(img_np)
                
                # 兼容旧代码的方法名（你可能把更新界面的方法叫 update_preview_ui 或 show_preview）
                if hasattr(self, "update_preview_ui"):
                    self.after(0, self.update_preview_ui, preview_img)
            except Exception as e:
                pass
        return callback_kwargs

    def start_upscale(self):
        if not self.current_generated_path or not os.path.exists(self.current_generated_path): return
        self.btn_upscale.config(state=DISABLED)
        threading.Thread(target=self.upscale_task, daemon=True).start()

    def upscale_task(self):
        try:
            self.after(0, lambda: self.lbl_status.config(text="🔍 正在进行 AI 高清重绘放大 (2倍)...", foreground="yellow"))
            img = Image.open(self.current_generated_path).convert("RGB")
            target_w, target_h = img.width * 2, img.height * 2
            img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            generator = torch.Generator(self.ai.device).manual_seed(random.randint(1, 2147483647))
            
            with torch.no_grad():
                upscaled_img = self.ai.img2img_pipe(
                    prompt=self.txt_prompt.get("1.0", END).strip(),
                    negative_prompt=self.txt_neg.get("1.0", END).strip(),
                    image=img_resized,
                    strength=0.25, 
                    num_inference_steps=int(self.spin_steps.get()),
                    generator=generator
                ).images[0]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"upscaled_{timestamp}.png")
            upscaled_img.save(save_path)
            
            self.after(0, lambda: self.show_preview(save_path))
            self.after(0, lambda: self.lbl_status.config(text="✨ 高清放大完成！", foreground="green"))
            
        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text="❌ 放大失败，请查看控制台", foreground="red"))
        finally:
            self.after(0, lambda: self.btn_upscale.config(state=NORMAL))

    def run_xy_plot_task(self, base_kwargs, width, height, pose_image=None):
        x_param = self.combo_x_type.get()
        y_param = self.combo_y_type.get()
        x_vals = [v.strip() for v in self.entry_x_vals.get().split(",") if v.strip()]
        y_vals = [v.strip() for v in self.entry_y_vals.get().split(",") if v.strip()]
        
        total = len(x_vals) * len(y_vals)
        self.after(0, lambda: self.lbl_status.config(text=f"📊 X/Y 炼丹中... 准备生成 {total} 张对比图！", foreground="yellow"))
        
        grid_images = []
        for y_idx, y_val in enumerate(y_vals):
            row_images = []
            for x_idx, x_val in enumerate(x_vals):
                current_kwargs = base_kwargs.copy()
                
                for p_type, p_val in [(x_param, x_val), (y_param, y_val)]:
                    if "Steps" in p_type: current_kwargs["num_inference_steps"] = int(p_val)
                    if "CFG" in p_type: current_kwargs["guidance_scale"] = float(p_val)
                    if "Seed" in p_type: current_kwargs["generator"] = torch.Generator(self.ai.device).manual_seed(int(p_val))

                self.after(0, lambda x=x_val, y=y_val, c=len(row_images)+y_idx*len(x_vals)+1: 
                           self.lbl_status.config(text=f"📊 正在炼制第 {c}/{total} 张... [X: {x}, Y: {y}]"))
                
                with torch.no_grad():
                    if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() and pose_image:
                        img = self.ai.controlnet_pipe(**current_kwargs, width=width, height=height, image=pose_image).images[0]
                    else:
                        img = self.ai.txt2img_pipe(**current_kwargs, width=width, height=height).images[0]

                if getattr(self, "var_enable_hires", None) and self.var_enable_hires.get():
                    self.after(0, lambda: self.lbl_status.config(text="✨ 构图完成！正在进行 XY-Hires. fix 高清重绘...", foreground="#00FFFF"))
                    hires_scale = float(self.combo_hires_scale.get())
                    denoise_strength = float(self.scale_hires_denoise.get())
                    target_w, target_h = int(width * hires_scale), int(height * hires_scale)
                
                    base_upscaled = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    hires_kwargs = current_kwargs.copy()
                    hires_kwargs["image"] = base_upscaled
                    hires_kwargs["strength"] = denoise_strength 
                    
                    img = self.ai.img2img_pipe(**hires_kwargs).images[0]
                    
                row_images.append(img)
            grid_images.append(row_images)

        margin_top, margin_left = 60, 120
        grid_w = margin_left + len(x_vals) * width
        grid_h = margin_top + len(y_vals) * height
        
        grid_canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(grid_canvas)
        try: font = ImageFont.truetype("arial.ttf", 30)
        except: font = ImageFont.load_default()

        for i, x_val in enumerate(x_vals):
            draw.text((margin_left + i * width + width//2 - 40, 15), f"{x_param[:4]}: {x_val}", fill="black", font=font)
        for j, y_val in enumerate(y_vals):
            draw.text((15, margin_top + j * height + height//2 - 20), f"{y_param[:4]}:\n{y_val}", fill="black", font=font)
            for i, img in enumerate(grid_images[j]):
                grid_canvas.paste(img, (margin_left + i * width, margin_top + j * height))
                
        save_path = os.path.join(OUTPUT_DIR, f"XY_Plot_{int(datetime.datetime.now().timestamp())}.png")
        grid_canvas.save(save_path)
        
        if hasattr(self, "update_preview_ui"):
            self.after(0, lambda: self.update_preview_ui(grid_canvas))
        else:
            self.after(0, lambda: self.show_preview(save_path))
            
        self.after(0, lambda: self.lbl_status.config(text=f"🎉 X/Y 炼丹完成！已保存为: {os.path.basename(save_path)}", foreground="#00FF00"))

    def generate_comic_strip(self, image_list):
        if not image_list or len(image_list) < 2: return
        self.after(0, lambda: self.lbl_status.config(text="🎞️ 正在拼合生成分镜连环画...", foreground="yellow"))

        try:
            comic_bg = make_comic_strip(image_list)
            file_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"Comic_Storyboard_{file_time}.png")
            comic_bg.save(save_path)

            self.after(0, lambda p=save_path: self.show_preview(p))
            self.after(0, lambda: self.lbl_status.config(text=f"🎉 分镜连环画拼合完成！已保存为: {os.path.basename(save_path)}", foreground="#00FF00"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text="⚠️ 连环画拼合失败，单张图已保存。", foreground="orange"))