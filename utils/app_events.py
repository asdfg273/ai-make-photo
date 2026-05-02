import tkinter as tk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont
import os
import threading
import warnings
from utils.app_utils import OUTPUT_DIR, PROMPT_PRESETS
from photo_turn.pro_editor_tk import ProImageEditor
class EventMixin:
    """专门负责处理所有按钮点击、滑块拖动、下拉框刷新的逻辑"""
    def apply_config_to_ui(self):
        self.spin_steps.set(self.config.default_steps)
        self.scale_str.set(self.config.default_strength)
        self.lbl_str_val.config(text=f"{self.config.default_strength:.2f}")
        
        try:
            # 如果配置里存了新的多 LoRA 列表
            if hasattr(self.config, 'lora_names') and isinstance(self.config.lora_names, list):
                for i in range(min(3, len(self.config.lora_names))):
                    self.combo_loras[i].set(self.config.lora_names[i])
                    self.scale_loras[i].set(self.config.lora_weights[i])
            else:
                # 兼容你以前的单 LoRA 老配置文件，把老配置加载到第一个槽位上
                self.combo_loras[0].set(getattr(self.config, 'default_lora', '无'))
                self.scale_loras[0].set(getattr(self.config, 'default_lora_weight', 0.7))
        except Exception as e:
            print("LoRA 配置加载跳过:", e)
        
        self.var_use_adetailer.set(self.config.adetailer_enabled)
        try:
            self.combo_device.set(self.config.device_preference)
        except Exception:
            self.combo_device.current(0)

    def on_closing(self):
        """窗口关闭时触发，统一保存当前界面上的参数，绝不卡顿"""
        try:
            self.config.default_steps = int(self.spin_steps.get())
            self.config.default_strength = float(self.scale_str.get())
            self.config.lora_names = [combo.get() for combo in self.combo_loras]
            self.config.lora_weights = [scale.get() for scale in self.scale_loras]
            self.config.adetailer_enabled = self.var_use_adetailer.get()
            self.config.device_preference = self.combo_device.get()
            self.config.save()
            print("💾 退出前已自动保存最后使用的参数配置。")
        except Exception as e:
            print("保存配置异常:", e)
            
        self.quit()  # 真正关闭程序
        self.destroy()

    def refresh_lora_by_model(self):
        current_model = self.combo_model.get().lower()
        if not current_model: return
        is_sdxl = "xl" in current_model 
        
        # 统一使用底层引擎去获取列表，不用再自己写 os.listdir 了，更安全
        loras = self.ai.get_available_loras("sdxl" if is_sdxl else "sd1.5")
        
        # 批量更新 3 个下拉框
        for combo in self.combo_loras:
            combo['values'] = loras
            # 如果当前选择的 LoRA 已经不在本地硬盘了（或者刚切换了模型架构），就重置为"无"
            if combo.get() not in loras:
                combo.set("无")
                
        # 更新提示词 Label（保留这行）
        self.text_lora_info.config(state="normal")
        self.text_lora_info.delete("1.0", "end")
        self.text_lora_info.insert("1.0", "LoRA说明: (未选择)")
        self.text_lora_info.config(state="disabled", fg="gray")

    def update_preview_ui(self, preview_img):
        img_tk = ImageTk.PhotoImage(preview_img.resize((512, 512), Image.Resampling.LANCZOS))
        self.preview_canvas.config(image=img_tk, text="")
        self.preview_canvas.image = img_tk

    def apply_preset(self, event):
        preset_name = self.combo_preset.get()
        if preset_name in PROMPT_PRESETS:
            self.txt_prompt.delete("1.0", END)
            self.txt_prompt.insert(END, PROMPT_PRESETS[preset_name]["p"])
            self.txt_neg.delete("1.0", END)
            self.txt_neg.insert(END, PROMPT_PRESETS[preset_name]["n"])
            self.lbl_status.config(text=f"✨ 已应用预设: {preset_name}", foreground="cyan")

    def read_png_info(self):
        path = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
        if not path: return
        try:
            img = Image.open(path)
            info = img.info.get("parameters", "")
            if not info:
                messagebox.showinfo("提示", "这张图片没有包含AI生成参数。")
                return
            lines = info.split('\n')
            if len(lines) >= 1:
                self.txt_prompt.delete("1.0", END)
                self.txt_prompt.insert(END, lines[0])
            if len(lines) >= 2 and lines[1].startswith("Negative prompt:"):
                self.txt_neg.delete("1.0", END)
                self.txt_neg.insert(END, lines[1].replace("Negative prompt: ", "").strip())
            messagebox.showinfo("解析成功", f"已成功提取参数！")
        except Exception as e: messagebox.showerror("错误", str(e))

    def refresh_models(self):
        models = self.ai.get_available_models()
        self.combo_model['values'] = models if models else ["未找到模型"]
        if models: 
            self.combo_model.current(0)
            self.load_model_info()  
            
        loras = self.ai.get_available_loras("sd1.5" if not self.ai.is_sdxl else "sdxl")
        
        for combo in self.combo_loras:
            combo['values'] = loras
            if combo.get() not in loras:
                combo.set("无")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path: self.set_reference_image(path)

    def set_reference_image(self, path, mask_path=None):
        self.ref_image_path = path
        self.mask_image_path = mask_path
        status = "已加载底图: " + os.path.basename(path)
        if mask_path: status = "🎨 已挂载局部重绘与遮罩"
        self.lbl_img_path.config(text=status, foreground="cyan")

    def clear_reference(self):
        self.ref_image_path = None
        self.mask_image_path = None
        self.lbl_img_path.config(text="未选择参考图", foreground="gray")

    def load_pose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.pose_image_path = path
            self.lbl_pose_path.config(text="已加载动作图: " + os.path.basename(path), foreground="yellow")
            self.var_use_pose.set(True) 

    def stop_generation(self):
        if self.is_generating:
            self.cancel_flag = True
            self.lbl_status.config(text="⚠️ 正在强行刹车，请稍候...", foreground="red")

    # 👇 核心修复区：统一并完善了调用专业修图器的逻辑
    def open_editor(self):
        """点击 '专业修图/画遮罩' 按钮的回调"""
        if not getattr(self, 'current_generated_path', None) or not os.path.exists(self.current_generated_path):
            messagebox.showwarning("提示", "请先在预览区选中一张生成的图片！")
            return

        def on_editor_saved(edited_pil_img, mask_pil_img):
            # 1. 确保存储目录存在
            os.makedirs(os.path.join(OUTPUT_DIR, "temp"), exist_ok=True)
            
            # 2. 保存底图和纯黑白遮罩图
            ref_path = os.path.abspath(os.path.join(OUTPUT_DIR, "temp", "inpaint_ref.png"))
            mask_path = os.path.abspath(os.path.join(OUTPUT_DIR, "temp", "inpaint_mask.png"))
            
            edited_pil_img.save(ref_path)
            mask_pil_img.save(mask_path)
            
            # 3. 赋值给主程序的图生图变量
            self.set_reference_image(ref_path, mask_path)
            
            # 4. 🔥 解决工作流冲突：取消 ControlNet 勾选，优先进行局部重绘
            if self.var_use_pose.get():
                self.var_use_pose.set(False)
                print("👉 [工作流调整] 已自动关闭 ControlNet，优先执行局部重绘")
            
            self.lbl_status.config(text="✅ 遮罩已准备完毕！点击生成将自动进入【局部重绘】模式", foreground="#00FF00")
            self.show_preview(ref_path) 
            
        # 唤醒子窗口，模态阻塞主窗口
        editor = ProImageEditor(self, self.current_generated_path, callback_on_save=on_editor_saved)
        editor.grab_set()

    def show_preview(self, img_path):
        try:
            img = Image.open(img_path)
            img.thumbnail((450, 600))
            photo = ImageTk.PhotoImage(img)
            self.preview_canvas.config(image=photo, text="")
            self.preview_canvas.image = photo
            self.current_generated_path = img_path
            self.btn_edit.config(state=NORMAL)
            self.btn_upscale.config(state=NORMAL) 
        except Exception as e: print("预览加载失败", e)
        
    def show_pose_preview(self, img):
        try:
            img.thumbnail((450, 600))
            photo = ImageTk.PhotoImage(img)
            self.pose_canvas.config(image=photo, text="")
            self.pose_canvas.image = photo
        except Exception as e: print("骨架预览失败", e)

    def load_model_info(self, event=None):
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
            
        # 2. 强制系统垃圾回收
        import gc
        gc.collect()
        
        # 👇 局部导入 torch，绝不拖慢软件秒开的速度！
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        model_name = self.combo_model.get()
        if not model_name or model_name == "未找到模型": return
        txt_path = os.path.join("models", model_name.replace(".safetensors", ".txt").replace(".ckpt", ".txt"))
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.lbl_model_info.config(text=f"📌 备忘: {f.read().strip()[:80]}", foreground="cyan")
            except: pass
        else: self.lbl_model_info.config(text="💡 提示: 可新建同名txt记录", foreground="gray")

    def load_lora_info(self, event=None):
        current_model = self.combo_model.get().lower()
        sub_dir = "sdxl" if "xl" in current_model else "sd1.5"
        
        info_texts = []
        has_lora = False
        
        for i, combo in enumerate(self.combo_loras):
            lora_name = combo.get()
            if lora_name and lora_name != "无":
                has_lora = True
                txt_path = os.path.join("loras", sub_dir, lora_name.replace(".safetensors", ".txt").replace(".ckpt", ".txt"))
                
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            memo = f.read().strip() 
                            info_texts.append(f"[槽{i+1}] {memo}")
                    except: pass
                else:
                    info_texts.append(f"[槽{i+1}] (无备忘录txt)")

        self.text_lora_info.config(state="normal")
        self.text_lora_info.delete("1.0", "end")
        
        if not has_lora:
            self.text_lora_info.insert("1.0", "💡 未使用 LoRA 插件")
            self.text_lora_info.config(fg="gray")
        else:
            if info_texts:
                final_text = "📌 备忘录:\n" + "\n".join(info_texts)
                self.text_lora_info.insert("1.0", final_text)
                self.text_lora_info.config(fg="#E066FF")
            else:
                self.text_lora_info.insert("1.0", "💡 提示: 可在 loras 目录下新建同名txt记录说明")
                self.text_lora_info.config(fg="gray")
                
        self.text_lora_info.config(state="disabled")

    def open_gallery_to_edit(self):
        photo_dir = os.path.abspath("photo")
        if not os.path.exists(photo_dir): os.makedirs(photo_dir)
            
        file_path = filedialog.askopenfilename(
            title="选择要进行高级调色的图片", 
            initialdir=photo_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            def on_editor_saved(edited_pil_image, mask_pil_image):
                self.lbl_status.config(text="修图完成！已自动加载为参考图。", foreground="green")
                timestamp = datetime.datetime.now().strftime('%H%M%S')
                out_path = os.path.join(photo_dir, f"pro_edited_{timestamp}.png")
                edited_pil_image.save(out_path)
                
                # 设置给重绘参数
                mask_path = None
                if mask_pil_image:
                    mask_path = os.path.join(photo_dir, f"pro_edited_mask_{timestamp}.png")
                    mask_pil_image.save(mask_path)
                    
                self.set_reference_image(out_path, mask_path)
                self.show_preview(out_path)

            self.lbl_status.config(text="正在打开专业级图片编辑器...", foreground="yellow")
            ProImageEditor(self, file_path, callback_on_save=on_editor_saved)

    def on_model_selected(self, event=None):
        self.load_model_info(event)  
        self.refresh_lora_by_model() 