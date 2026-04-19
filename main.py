# main.py

import os
import threading
import subprocess
import sys
import datetime
import random
import re
import torch
import cv2  
import numpy as np 
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import urllib.request
from PIL.PngImagePlugin import PngInfo
import ttkbootstrap as tb
from ttkbootstrap.constants import * 
from ttkbootstrap.constants import *
from ttkbootstrap.widgets.scrolled import ScrolledFrame 
from tkinter import messagebox, filedialog
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from translation_service import TranslationService
from model_manager import ModelManager
from photo_turn.pro_editor_tk import ProImageEditor

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
OUTPUT_DIR = "photo"

PROMPT_PRESETS = {
    "默认精美": {"p": "杰作, 最高画质, 极其精致的细节, 绝美光影, 8k分辨率, 电影级打光", "n": "低画质, 畸形, 扭曲, 糟糕的人体结构, 错误的比例, 模糊, 水印, 签名"},
    "二次元动漫": {"p": "杰作, 极致画质, 动漫风格, 绚丽的色彩, 京阿尼风格, 精致的五官, 唯美背景", "n": "真实照片, 3d, 丑陋, 崩坏, 多余的手指, 恐怖, 猎奇"},
    "电影级真实": {"p": "RAW照片, 极其真实的自然光, 电影感构图, 8k uhd, dslr, 柔和焦点, 胶片颗粒, 毛孔细节", "n": "动漫, 卡通, 画作, 插画, 虚假, 塑料质感, 过度平滑"},
    "赛博朋克": {"p": "赛博朋克风格, 霓虹灯, 未来城市, 高科技机甲, 杰作, 赛博格, 强烈的色彩对比", "n": "古风, 乡村, 简单背景, 模糊, 阳光明媚"},
    "唯美水墨": {"p": "中国水墨画风格, 意境深远, 留白, 极其优美的毛笔线条, 大师级杰作, 传统色彩", "n": "写实, 3d, 鲜艳的霓虹色, 机械, 现代"}
}

def parse_dynamic_prompt(prompt_text):
    prompt_text = re.sub(r'<lora:[^>]+>', '', prompt_text)
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, prompt_text)
    if not matches:
        return [prompt_text]
    options_list = [match.split('|') for match in matches]
    combinations = list(itertools.product(*options_list))
    final_prompts = []
    for combo in combinations:
        temp_prompt = prompt_text
        for replacement in combo:
            temp_prompt = re.sub(pattern, replacement.strip(), temp_prompt, count=1)
        final_prompts.append(temp_prompt)
    return final_prompts

class AIDesktopApp(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly", title="AI 绘画引擎 Pro V4.0", size=(1250, 850))
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
        self.translator = TranslationService()
        self.ai = ModelManager()
        
        self.is_generating = False
        self.cancel_flag = False
        self.ref_image_path = None
        self.mask_image_path = None
        self.pose_image_path = None
        self.current_generated_path = None
        
        self.setup_ui()
        self.refresh_models()
        self.refresh_lora_by_model()
    def setup_ui(self):
        # 🌟 修复 UI 核心：使用 ScrolledFrame 替代普通 Frame
        self.left_panel = ScrolledFrame(self, padding=10, width=360)
        self.left_panel.pack(side=LEFT, fill=Y, expand=False)
        
        # 为了方便代码调用，提取内部的容器
        container = self.left_panel

        # === 1. 模型与 LoRA ===
        tb.Label(container, text="📦 主模型 (画风)", font=("微软雅黑", 10, "bold")).pack(anchor=W)
        self.combo_model = tb.Combobox(container, state="readonly", width=35)
        self.combo_model.pack(fill=X, pady=2)
        self.lbl_model_info = tb.Label(container, text="模型说明: (未找到备忘录)", font=("微软雅黑", 9), foreground="gray", wraplength=250)
        self.lbl_model_info.pack(anchor=W, pady=(0, 5))
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        lora_frame = tb.Frame(container)
        lora_frame.pack(fill=X, pady=5)
        tb.Label(lora_frame, text="🧩 LoRA 插件", font=("微软雅黑", 10, "bold"), foreground="orchid").grid(row=0, column=0, sticky=W)
        self.combo_lora = tb.Combobox(lora_frame, state="readonly", width=18)
        self.combo_lora.grid(row=0, column=1, padx=5)
        
        tb.Label(lora_frame, text="权重").grid(row=1, column=0, pady=5, sticky=W)
        lora_scale_frame = tb.Frame(lora_frame)
        lora_scale_frame.grid(row=1, column=1, pady=5, sticky=EW)
        self.lbl_lora_val = tb.Label(lora_scale_frame, text="0.70", width=4, bootstyle="info")
        self.lbl_lora_val.pack(side=RIGHT, padx=(5,0))
        self.scale_lora = tb.Scale(
            lora_scale_frame, from_=0.1, to=1.5, value=0.7,
            command=lambda v: self.lbl_lora_val.config(text=f"{float(v):.2f}")
        )
        self.scale_lora.pack(side=LEFT, fill=X, expand=True)

        self.lbl_lora_info = tb.Label(container, text="LoRA说明: (未找到备忘录)", font=("微软雅黑", 9), foreground="gray", wraplength=250)
        self.lbl_lora_info.pack(anchor=W, pady=(0, 5))
        self.combo_lora.bind("<<ComboboxSelected>>", self.load_lora_info)
        tb.Button(container, text="🔄 刷新模型", bootstyle="outline-secondary", command=self.refresh_models).pack(fill=X, pady=2)

        # === 2. 预设与提示词 ===
        preset_frame = tb.Frame(container)
        preset_frame.pack(fill=X, pady=(10, 5))
        tb.Label(preset_frame, text="📖 魔法预设:", font=("微软雅黑", 10, "bold"), foreground="cyan").pack(side=LEFT)
        self.combo_preset = tb.Combobox(preset_frame, state="readonly", values=list(PROMPT_PRESETS.keys()))
        self.combo_preset.pack(side=LEFT, fill=X, expand=True, padx=(5,0))
        self.combo_preset.bind("<<ComboboxSelected>>", self.apply_preset)

        tb.Label(container, text="✨ 正向提示词", font=("微软雅黑", 10)).pack(anchor=W, pady=(5,2))
        self.txt_prompt = tb.Text(container, height=4, width=40)
        self.txt_prompt.pack(fill=X)
        self.txt_prompt.insert(END, PROMPT_PRESETS["默认精美"]["p"])

        tb.Label(container, text="🚫 反向防崩坏词", font=("微软雅黑", 10)).pack(anchor=W, pady=(5,2))
        self.txt_neg = tb.Text(container, height=3, width=40)
        self.txt_neg.pack(fill=X)
        self.txt_neg.insert(END, PROMPT_PRESETS["默认精美"]["n"])

        # === 3. 参数设置 ===
        params_frame = tb.Frame(container)
        params_frame.pack(fill=X, pady=5)
        
        tb.Label(params_frame, text="分辨率").grid(row=0, column=0, pady=2, sticky=W)
        self.combo_res = tb.Combobox(params_frame, values=["512x512", "512x768", "768x512"], width=10, state="readonly")
        self.combo_res.current(0)
        self.combo_res.grid(row=0, column=1, pady=2, padx=5)

        tb.Label(params_frame, text="生成步数").grid(row=1, column=0, pady=2, sticky=W)
        self.spin_steps = tb.Spinbox(params_frame, from_=10, to=100, width=10)
        self.spin_steps.set(30)
        self.spin_steps.grid(row=1, column=1, pady=2, padx=5)
        
        tb.Label(params_frame, text="重绘幅度").grid(row=2, column=0, pady=2, sticky=W)
        str_scale_frame = tb.Frame(params_frame)
        str_scale_frame.grid(row=2, column=1, pady=2, padx=5, sticky=EW)
        self.lbl_str_val = tb.Label(str_scale_frame, text="0.60", width=4, bootstyle="info")
        self.lbl_str_val.pack(side=RIGHT)
        self.scale_str = tb.Scale(
            str_scale_frame, from_=0.1, to=1.0, value=0.6, bootstyle="info",
            command=lambda v: self.lbl_str_val.config(text=f"{float(v):.2f}")
        )
        self.scale_str.pack(side=LEFT, fill=X, expand=True)

        tb.Label(params_frame, text="生成数量").grid(row=3, column=0, pady=2, sticky=W)
        self.spin_count = tb.Spinbox(params_frame, from_=1, to=20, width=10)
        self.spin_count.set(1)
        self.spin_count.grid(row=3, column=1, pady=2, padx=5)

        # 🌟 V4.0 新增：ADetailer 开关
        adetailer_frame = tb.Frame(params_frame)
        adetailer_frame.grid(row=4, column=0, columnspan=2, pady=(5,0), sticky=W)
        self.var_use_adetailer = tb.BooleanVar(value=False)
        tb.Checkbutton(adetailer_frame, text="🧑‍🎨 ADetailer 修复:", variable=self.var_use_adetailer, bootstyle="success-round-toggle").pack(side=LEFT)
        self.combo_adetailer_model = tb.Combobox(adetailer_frame, values=["二次元动漫", "真人脸部"], width=10, state="readonly")
        self.combo_adetailer_model.current(0)
        self.combo_adetailer_model.pack(side=LEFT, padx=5)

        self.var_make_comic = tb.BooleanVar(value=False)
        tb.Checkbutton(params_frame, text="🎞️ 漫画排版(生成后拼合)", variable=self.var_make_comic, bootstyle="info-round-toggle").grid(row=5, column=0, columnspan=2, pady=(5,5), sticky=W)

        # === 4. ControlNet 姿势控制 ===
        cn_frame = tb.Labelframe(container, text="🏋️‍♂️ ControlNet 强力控制", padding=5, bootstyle="warning")
        cn_frame.pack(fill=X, pady=5)
        
        self.var_use_pose = tb.BooleanVar(value=False)
        tb.Checkbutton(cn_frame, text="开启 ControlNet", variable=self.var_use_pose, bootstyle="warning-round-toggle").pack(anchor=W, pady=2)
        
        # 下拉框：选择控制模式
        self.combo_cn_type = tb.Combobox(cn_frame, state="readonly", values=["openpose", "canny", "depth"], width=15)
        self.combo_cn_type.current(0)
        self.combo_cn_type.pack(fill=X, pady=2)

        tb.Button(cn_frame, text="📸 载入参考图 (动作/线稿/结构)", bootstyle="warning-outline", command=self.load_pose_image).pack(fill=X, pady=(5,0))
        self.lbl_pose_path = tb.Label(cn_frame, text="未选择参考图", foreground="gray")
        self.lbl_pose_path.pack(anchor=W)


        # === 5. 图生图与解析工具 ===
        tb.Button(container, text="🔍 提取图片参数 (PNG Info)", bootstyle="secondary", command=self.read_png_info).pack(fill=X, pady=(5,2))
        tb.Button(container, text="🖼️ 载入底图/遮罩 (图生图)", bootstyle="info-outline", command=self.select_image).pack(fill=X, pady=2)
        self.lbl_img_path = tb.Label(container, text="未选择参考图", foreground="gray")
        self.lbl_img_path.pack(anchor=W)
        self.btn_clear_ref = tb.Button(container, text="❌ 清除底图/遮罩", bootstyle="link-danger", command=self.clear_reference)
        self.btn_clear_ref.pack(anchor=W, pady=(0, 5))

        # === 6. 操作按钮 ===
        self.btn_gen = tb.Button(container, text="🚀 开始生成", bootstyle="success", command=self.start_generation)
        self.btn_gen.pack(fill=X, pady=5, ipady=5)
        self.btn_open_gallery = tb.Button(container, text="📂 打开历史图库并编辑", bootstyle="info-outline", command=self.open_gallery_to_edit)
        self.btn_open_gallery.pack(fill=X, pady=5)
        self.btn_stop = tb.Button(container, text="🛑 强制中断", bootstyle="danger", state=DISABLED, command=self.stop_generation)
        self.btn_stop.pack(fill=X, pady=(2, 20)) # 底部留一点边距

        # === 右侧预览区 ===
        right_panel = tb.Frame(self, padding=20)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        self.lbl_status = tb.Label(right_panel, text="V4.0 就绪。新增 ADetailer 面部崩坏自动修复！", font=("微软雅黑", 12), foreground="cyan")
        self.lbl_status.pack(pady=5)

        self.progress = tb.Progressbar(right_panel, bootstyle="success-striped", mode="determinate")
        self.progress.pack(fill=X, pady=5)

        preview_frame = tb.Frame(right_panel)
        preview_frame.pack(fill=BOTH, expand=True, pady=10)
        
        self.pose_canvas = tb.Label(preview_frame, text="[ 骨架预览区 ]", background="#1a1a1a", anchor=CENTER)
        self.pose_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        self.preview_canvas = tb.Label(preview_frame, text="[ 生成图片区 ]", background="#2b2b2b", anchor=CENTER)
        self.preview_canvas.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        self.btn_edit = tb.Button(right_panel, text="🎨 将此图送入编辑器 (支持局部涂鸦)", bootstyle="warning", state=DISABLED, command=self.open_editor)
        self.btn_edit.pack(pady=5)
        self.btn_upscale = tb.Button(right_panel, text="🔍 一键 AI 高清放大 (耗时较长)", bootstyle="info-outline", state=DISABLED, command=self.start_upscale)
        self.btn_upscale.pack(pady=2)

    # --- 工具函数 (保持原样) ---
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
        loras = self.ai.get_available_loras()
        self.combo_lora['values'] = loras
        if loras: self.combo_lora.current(0)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path: self.set_reference_image(path)

    def set_reference_image(self, path, mask_path=None):
        self.ref_image_path = path
        self.mask_image_path = mask_path
        status = "已加载: " + os.path.basename(path)
        if mask_path: status += " (含遮罩)"
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

    def open_editor(self):
        if self.current_generated_path and os.path.exists(self.current_generated_path):
            ImageEditorWindow(self, self.current_generated_path, callback_use_as_ref=self.set_reference_image)

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
        lora_name = self.combo_lora.get()
        if not lora_name or lora_name == "无":
            self.lbl_lora_info.config(text="💡 未使用 LoRA 插件", foreground="gray")
            return
            
        current_model = self.combo_model.get().lower()
        sub_dir = "sdxl" if "xl" in current_model else "sd1.5"
        
        txt_path = os.path.join("loras", sub_dir, lora_name.replace(".safetensors", ".txt").replace(".ckpt", ".txt"))
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.lbl_lora_info.config(text=f"📌 备忘: {f.read().strip()[:80]}", foreground="#E066FF")
            except: pass
        else: 
            self.lbl_lora_info.config(text="💡 提示: 可新建同名txt记录", foreground="gray")

    # 🌟 V4.0 核心算法：低显存 ADetailer 脸部修复
    def apply_adetailer(self, base_image, prompt, negative_prompt, seed):
        try:
            cv_img = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 1. 动态选择脸部识别模型
            is_anime = (self.combo_adetailer_model.get() == "二次元动漫")
            if is_anime:
                xml_path = "lbpcascade_animeface.xml"
                # 自动下载动漫脸模型 (免去手动找文件的麻烦)
                if not os.path.exists(xml_path):
                    print("⏳ 首次使用，正在自动下载二次元脸部识别模型...")
                    url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
                    urllib.request.urlretrieve(url, xml_path)
                face_cascade = cv2.CascadeClassifier(xml_path)
                # 动漫脸参数微调，提高检测率
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
            else:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            if len(faces) == 0:
                print("ADetailer: 未检测到人脸，跳过。")
                return base_image 

            result_image = base_image.copy()
            
            for (x, y, w, h) in faces:
                # 2. 扩大选区框，防止下巴或头发被截断
                margin = int(w * 0.4)
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(base_image.width, x + w + margin), min(base_image.height, y + h + margin)
                crop_w, crop_h = x2 - x1, y2 - y1
                
                face_crop = base_image.crop((x1, y1, x2, y2))
                face_crop_512 = face_crop.resize((512, 512), Image.LANCZOS)

                # 3. 针对不同风格强化修复词
                extra_tag = "highly detailed anime face, perfect eyes, masterpiece" if is_anime else "beautiful detailed face, highly detailed eyes, perfectly symmetrical face"
                enhanced_prompt = prompt + ", " + extra_tag
                generator = torch.Generator(self.ai.device).manual_seed(seed)
                
                with torch.no_grad():
                    fixed_face_512 = self.ai.img2img_pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=face_crop_512,
                        strength=0.35,  
                        num_inference_steps=25,
                        generator=generator
                    ).images[0]
                
                fixed_face = fixed_face_512.resize((crop_w, crop_h), Image.LANCZOS)
                
                # 4. 🌟 边缘羽化融合算法 (消除拼接大方块印记)
                mask = Image.new("L", (crop_w, crop_h), 0)
                draw = ImageDraw.Draw(mask)
                blur_radius = int(crop_w * 0.15) # 边缘15%作为过渡区
                draw.rectangle([blur_radius, blur_radius, crop_w-blur_radius, crop_h-blur_radius], fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(blur_radius)) # 高斯模糊产生渐变遮罩
                
                result_image.paste(fixed_face, (x1, y1), mask) # 使用渐变遮罩无缝贴回
                
                print(f"ADetailer: 成功修复并完美融合脸部 ({x1},{y1})")

            return result_image
        except Exception as e:
            print("ADetailer 执行失败，返回原图:", e)
            return base_image

    # --- 核心生成逻辑 ---
    def start_generation(self):
        if not self.combo_model.get() or self.combo_model.get() == "未找到模型":
            messagebox.showwarning("警告", "请先在 models 文件夹中放入模型！")
            return
        if self.var_use_pose.get() and not self.pose_image_path:
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
            model_name = self.combo_model.get()
            lora_name = self.combo_lora.get()
            lora_weight = self.scale_lora.get()
            
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

            self.after(0, lambda: self.lbl_status.config(text="🧠 正在加载模型与组件...", foreground="yellow"))
            self.ai.load_model(model_name)
            sub_dir = "sdxl" if "xl" in model_name.lower() else "sd1.5"
            self.ai.apply_lora(lora_name, sub_dir=sub_dir)
            
            pose_image = None
            cn_type = "openpose"
            if self.var_use_pose.get():
                cn_type = self.combo_cn_type.get()
                self.after(0, lambda: self.lbl_status.config(text=f"⚙️ 正在解析 {cn_type} 参考图特征...", foreground="yellow"))
                self.ai.prepare_controlnet(control_type=cn_type)
                raw_img = Image.open(self.pose_image_path).convert("RGB")
                # 调用新的 get_control_image
                pose_image = self.ai.get_control_image(raw_img, control_type=cn_type)
                self.after(0, lambda p=pose_image: self.show_pose_preview(p.copy()))

            generated_images_list = [] 

            print(f"👉 [调试信息] 准备进入画图循环！")
            print(f"👉 [调试信息] 总共需要画的张数: {total_generate_count}, 提示词: {current_en_prompt if 'current_en_prompt' in locals() else '未定义'}")

            for i in range(total_generate_count):
                if self.cancel_flag: break
            
                current_raw_prompt = parsed_raw_prompts[i]
                current_en_prompt = en_prompts[i]
                current_seed = random.randint(1, 2147483647)
                generator = torch.Generator(self.ai.device).manual_seed(current_seed)

                self.after(0, lambda idx=i, s=current_seed: self.lbl_status.config(text=f"🔥 第 {idx+1}/{total_generate_count} 张 (Seed: {s}) ...", foreground="cyan"))
                self.after(0, lambda: self.progress.configure(value=0, maximum=steps))

                print(f"👉 [探针 1] 正在解析无限提示词...")
                embed_kwargs = self.ai.encode_prompt(current_en_prompt, en_neg)
                print(f"👉 [探针 2] 提示词解析完毕！")

                # 🔥 核心修复：适配最新版 Diffusers 的进度条回调机制
                def step_cb(pipe, step_index, timestep, callback_kwargs):
                    if self.cancel_flag: raise InterruptedError()
                    self.after(0, lambda: self.progress.configure(value=step_index + 1))
                    return callback_kwargs

                kwargs = {
                    "num_inference_steps": steps, 
                    "generator": generator,
                    "callback_on_step_end": step_cb, # 必须使用新的参数名
                }
                kwargs.update(embed_kwargs)

                if lora_name and lora_name != "无":
                    kwargs["cross_attention_kwargs"] = {"scale": lora_weight}

                print(f"👉 [探针 3] 准备进入底层画图管线，预计步数: {steps}")
                with torch.no_grad():
                    if self.var_use_pose.get():
                        print("👉 [探针 4] 正在执行 ControlNet...")
                        image = self.ai.controlnet_pipe(**kwargs, width=width, height=height, image=pose_image).images[0]
                    elif getattr(self, 'mask_image_path', None): 
                        print("👉 [探针 4] 正在执行 局部重绘...")
                        init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                        mask_img = Image.open(self.mask_image_path).convert("L").resize((width, height))
                        image = self.ai.inpaint_pipe(**kwargs, image=init_img, mask_image=mask_img, strength=strength).images[0]
                    elif getattr(self, 'ref_image_path', None):
                        print("👉 [探针 4] 正在执行 图生图...")
                        init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                        image = self.ai.img2img_pipe(**kwargs, image=init_img, strength=strength).images[0]
                    else:
                        print("👉 [探针 4] 正在执行 文生图...")
                        image = self.ai.txt2img_pipe(**kwargs, width=width, height=height).images[0]

                print("👉 [探针 5] 管线执行完毕！准备保存图片...")

                # 🌟 V4.0 ADetailer 挂载点
                if self.var_use_adetailer.get() and not self.cancel_flag:
                    self.after(0, lambda: self.lbl_status.config(text="🧑‍🎨 正在进行 ADetailer 脸部修复...", foreground="yellow"))
                    image = self.apply_adetailer(image, current_en_prompt, en_neg, current_seed)

                generated_images_list.append(image) 

                metadata_str = f"{current_raw_prompt}\nNegative prompt: {raw_neg}\nSteps: {steps}, Size: {width}x{height}, Seed: {current_seed}, Model: {model_name}"
                pnginfo = PngInfo()
                pnginfo.add_text("parameters", metadata_str)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join("photo", f"v4_{timestamp}_{current_seed}.png") 
            
                image.save(save_path, pnginfo=pnginfo)
                print(f"👉 [探针 6] 图片已成功保存至: {save_path}")
                self.after(0, lambda p=save_path: self.show_preview(p))
        
        except InterruptedError:
            self.after(0, lambda: self.lbl_status.config(text="🛑 已打断", foreground="red"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text="❌ 发生错误，请查看控制台", foreground="red"))
            print("详细错误:", e)
        finally:
            self.is_generating, self.cancel_flag = False, False
            self.after(0, lambda: self.btn_gen.config(state=NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=DISABLED))

    def start_upscale(self):
        if not self.current_generated_path or not os.path.exists(self.current_generated_path):
            return
        self.btn_upscale.config(state=DISABLED)
        threading.Thread(target=self.upscale_task, daemon=True).start()

    def upscale_task(self):
        try:
            self.after(0, lambda: self.lbl_status.config(text="🔍 正在进行 AI 高清重绘放大 (2倍)...", foreground="yellow"))
            img = Image.open(self.current_generated_path).convert("RGB")
            
            target_w, target_h = img.width * 2, img.height * 2
            img_resized = img.resize((target_w, target_h), Image.LANCZOS)
            
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
            print("放大失败:", e)
            self.after(0, lambda: self.lbl_status.config(text="❌ 放大失败，请查看控制台", foreground="red"))
        finally:
            self.after(0, lambda: self.btn_upscale.config(state=NORMAL))

    def open_gallery_to_edit(self):
        photo_dir = os.path.abspath("photo")
        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)
            
        file_path = filedialog.askopenfilename(
            title="选择要进行高级调色的图片", 
            initialdir=photo_dir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            # 定义回调函数：当编辑器点击"保存"时执行
            def on_editor_saved(edited_pil_image):
                self.lbl_status.config(text="修图完成！已自动加载为参考图。", foreground="green")
                
                # 保存为新文件
                timestamp = datetime.datetime.now().strftime('%H%M%S')
                out_path = os.path.join(photo_dir, f"pro_edited_{timestamp}.png")
                edited_pil_image.save(out_path)
                
                # 传回给主界面的参考图
                self.pose_image_path = out_path
                self.lbl_pose_path.config(text=f"已加载: {os.path.basename(out_path)}", foreground="cyan")
                self.show_pose_preview(edited_pil_image)
            
            # 唤醒子窗口，处于同一个进程，安全且极速！
            self.lbl_status.config(text="正在打开专业级图片编辑器...", foreground="yellow")
            ProImageEditor(self, file_path, callback_on_save=on_editor_saved)

    def on_editor_saved(self, edited_img_path, mask_img_path):
        # 编辑器保存后的回调处理：自动加载为参考图
        self.pose_image_path = edited_img_path
        self.lbl_pose_path.config(text=f"已加载编辑图: {os.path.basename(edited_img_path)}", foreground="cyan")
        
        # 自动在界面上预览该图
        img = Image.open(edited_img_path).convert("RGB")
        self.show_pose_preview(img)
        
        # 如果有遮罩，可以自动切换到局部重绘模式 (可选，看您的原先逻辑)
        if mask_img_path:
            messagebox.showinfo("提示", "检测到 AI 遮罩，建议使用 Inpaint(局部重绘) 管线或传入相应参数进行重绘。")

    def on_model_selected(self, event=None):
        self.load_model_info(event)  # 保留你原来的备忘录读取功能
        self.refresh_lora_by_model() # 触发超酷的 LoRA 智能隔离过滤

    # 👇 智能联动 2：根据当前大模型类型，读取不同的 LoRA 文件夹
    def refresh_lora_by_model(self):
        current_model = self.combo_model.get().lower()
        if not current_model: return
        
        # 智能判定：模型名字里带 "xl" 就是 SDXL 模型，否则默认当做 SD1.5
        is_sdxl = "xl" in current_model 
        
        # ⚠️ 注意：这里的 "models/lora" 请改成你实际存放 LoRA 的主文件夹路径！
        base_lora_dir = "loras" 
        target_dir = os.path.join(base_lora_dir, "sdxl" if is_sdxl else "sd1.5")
        
        lora_list = ["无"]
        if os.path.exists(target_dir):
            lora_list += [f for f in os.listdir(target_dir) if f.endswith(('.safetensors', '.ckpt', '.pt'))]
        
        # 更新下拉框并重置状态
        self.combo_lora.config(values=lora_list)
        self.combo_lora.set("无")
        self.lbl_lora_info.config(text="LoRA说明: (未选择)")
        print(f"👉 [智能联动] 已自动切换为 {'SDXL' if is_sdxl else 'SD 1.5'} 的 LoRA 列表。")

if __name__ == "__main__":
    app = AIDesktopApp()
    app.mainloop()