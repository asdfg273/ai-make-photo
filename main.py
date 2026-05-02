# main.py

import os
import threading
import subprocess
import sys
import datetime
import traceback
import random
import re
import cv2  
import numpy as np 
from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont
import urllib.request
from PIL.PngImagePlugin import PngInfo
import ttkbootstrap as tb
from ttkbootstrap.constants import * 
from ttkbootstrap.widgets.scrolled import ScrolledFrame 
from tkinter import messagebox, filedialog
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from translation_service import TranslationService
from photo_turn.pro_editor_tk import ProImageEditor
from config_manager import AppConfig
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 替换原来的相对路径为绝对路径
OUTPUT_DIR = os.path.join(BASE_DIR, "photo")
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
        self.ai = None  
        
        self.is_generating = False
        self.cancel_flag = False
        self.ref_image_path = None
        self.mask_image_path = None
        self.pose_image_path = None
        self.current_generated_path = None
        self.config = AppConfig.load()
        self.setup_ui()    
        # 2. 将配置应用到 UI
        self.apply_config_to_ui()
        
        # 3. 拦截窗口关闭事件 (右上角的 X)，在退出前保存配置！
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        if hasattr(self, "btn_gen"):
            self.btn_gen.config(state=DISABLED, text="🚀 AI 引擎预热中...")
        self.lbl_status.config(text="⏳ 正在后台加载大模型与底层环境，请稍候...", foreground="yellow")
        
        threading.Thread(target=self.async_init_ai, daemon=True).start()

    def async_init_ai(self):
        print("👉 [系统预热] 后台开始导入重型库(PyTorch/Diffusers)...")
        global torch  
        import torch
        from model_manager import ModelManager 
        
        print("👉 [系统预热] 依赖导入完成，正在加载大模型...")
        self.ai = ModelManager()  
        self.after(0, self.on_ai_loaded)

    def on_ai_loaded(self):
        self.refresh_models()
        self.refresh_lora_by_model()
        if hasattr(self, "btn_gen"):
            self.btn_gen.config(state=NORMAL, text="🚀 开始生成")
        self.lbl_status.config(text="✅ 系统就绪。等待生成指令...", foreground="#00FF00")
        print("👉 [系统预热] 引擎就绪！")
        available_devices = ["自动 (Auto)"]
        if torch.cuda.is_available():
            available_devices.append("CUDA (Nvidia GPU)")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append("MPS (Apple Silicon Mac)")
        available_devices.append("CPU (纯内存-极慢)")
        
        # 更新下拉框的值，并解除禁用状态
        self.combo_device.config(state="readonly", values=available_devices)
        
        # 尝试恢复保存的配置（如果没有配置过，就默认选0自动）
        try:
            self.combo_device.set(self.config.device_preference)
        except Exception:
            self.combo_device.current(0)

    def setup_ui(self):
        self.left_panel = tb.Frame(self, width=350)
        self.left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)
        
        # === 全局底部按钮区 ===
        action_frame = tb.Frame(self.left_panel)
        action_frame.pack(side=BOTTOM, fill=X)
        
        self.btn_gen = tb.Button(action_frame, text="🚀 开始生成", bootstyle="success", command=self.start_generation)
        self.btn_gen.pack(fill=X, pady=(10, 5), ipady=8) 
        self.btn_open_gallery = tb.Button(action_frame, text="📂 打开历史图库并编辑", bootstyle="info-outline", command=self.open_gallery_to_edit)
        self.btn_open_gallery.pack(fill=X, pady=2)
        self.btn_stop = tb.Button(action_frame, text="🛑 强制中断", bootstyle="danger", state=DISABLED, command=self.stop_generation)
        self.btn_stop.pack(fill=X, pady=(2, 5))

        # === 选项卡容器 ===
        self.notebook = tb.Notebook(self.left_panel, bootstyle="info")
        self.notebook.pack(side=TOP, fill=BOTH, expand=True)
        
        tab_basic_outer = tb.Frame(self.notebook)
        tab_adv_outer = tb.Frame(self.notebook)
        tab_img_outer = tb.Frame(self.notebook)
        
        self.notebook.add(tab_basic_outer, text="⚙️ 基础设置")
        self.notebook.add(tab_adv_outer, text="🧪 高级炼丹")
        self.notebook.add(tab_img_outer, text="🎮 图像控制")

        tab_basic = ScrolledFrame(tab_basic_outer, padding=10)
        tab_basic.pack(fill=BOTH, expand=True)
        tab_adv = ScrolledFrame(tab_adv_outer, padding=10)
        tab_adv.pack(fill=BOTH, expand=True)
        tab_img = ScrolledFrame(tab_img_outer, padding=10)
        tab_img.pack(fill=BOTH, expand=True)

        # ==========================================
        # 🟢 选项卡 1：基础设置
        # ==========================================
        hw_frame = tb.Frame(tab_basic)
        hw_frame.pack(fill=X, pady=(0, 10), padx=8)
        
        tb.Label(hw_frame, text="💻 硬件加速", font=("微软雅黑", 10, "bold"), foreground="orange").pack(side=LEFT)
        
        # 初始只放一个占位符，状态设为 disabled 防止用户在加载完之前点击
        self.combo_device = tb.Combobox(hw_frame, state="disabled", values=["检测中 (Loading...)"])
        self.combo_device.current(0)
        self.combo_device.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))
        
        def on_device_changed(event):
            self.lbl_status.config(text="⚠️ 硬件设备已切换，请点击下方 [🔄 刷新模型与LoRA] 使其生效。", foreground="yellow")
            
        self.combo_device.bind("<<ComboboxSelected>>", on_device_changed)
            
        tb.Label(tab_basic, text="📦 主模型 (画风)", font=("微软雅黑", 10, "bold")).pack(anchor=W)
        self.combo_model = tb.Combobox(tab_basic, state="readonly")
        self.combo_model.pack(fill=X, pady=2, padx=8)
        self.lbl_model_info = tb.Label(tab_basic, text="模型说明: (未找到备忘录)", font=("微软雅黑", 9), foreground="gray", wraplength=260, justify="left")
        self.lbl_model_info.pack(anchor=W, pady=(0, 5), padx=8)
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_selected)
        
        lora_container = tb.LabelFrame(tab_basic, text="🧩 LoRA 组合插件 (支持多重融合)")
        lora_container.pack(fill=X, pady=(5, 10))

        self.combo_loras = []
        self.scale_loras = []

        # 循环创建 3 个槽位，每个槽位包含：标签 + 下拉框 + 滑块 + 权重数值
        for i in range(3):
            row_frame = tb.Frame(lora_container)
            row_frame.pack(fill=X, pady=3)
            
            tb.Label(row_frame, text=f"槽位 {i+1}").pack(side=LEFT)
            
            # 1. 下拉框
            combo = tb.Combobox(row_frame, state="readonly", width=14)
            combo.pack(side=LEFT, padx=5)
            combo.bind("<<ComboboxSelected>>", self.load_lora_info)
            
            # 2. 动态数字标签 (比如显示 0.80)
            val_lbl = tb.Label(row_frame, text="0.80", width=4, font=("Arial", 8))
            val_lbl.pack(side=RIGHT, padx=(2, 0))
            
            # 3. 权重滑块 (范围 0.0 到 2.0)
            scale = tb.Scale(row_frame, from_=0.0, to=2.0, orient=HORIZONTAL)
            scale.set(0.8) # 默认权重设定为 0.8
            scale.pack(side=RIGHT, fill=X, expand=True)
            
            # 滑动时动态更新旁边的数字
            def update_val(val, l=val_lbl):
                l.config(text=f"{float(val):.2f}")
            scale.config(command=update_val)
            
            self.combo_loras.append(combo)
            self.scale_loras.append(scale)

        # 4. 备忘录文本框
        self.text_lora_info = tb.Text(lora_container, height=5, font=("微软雅黑", 9), wrap="word")
        self.text_lora_info.pack(fill=X, pady=(5, 5), padx=5)
        self.text_lora_info.insert("1.0", "📌 备忘录:\n(选择LoRA后自动显示)")
        self.text_lora_info.config(state="disabled", fg="#E066FF", bg="#2b2b2b", borderwidth=0)
        tb.Button(tab_basic, text="🔄 刷新模型与LoRA", bootstyle="outline-secondary", command=self.refresh_models).pack(fill=X, pady=2, padx=8)

        preset_frame = tb.Frame(tab_basic)
        preset_frame.pack(fill=X, pady=(10, 5), padx=8)
        tb.Label(preset_frame, text="📖 魔法预设:", font=("微软雅黑", 10, "bold"), foreground="cyan").pack(side=LEFT)
        self.combo_preset = tb.Combobox(preset_frame, state="readonly", values=list(PROMPT_PRESETS.keys()))
        self.combo_preset.pack(side=LEFT, fill=X, expand=True, padx=(5,0))
        self.combo_preset.bind("<<ComboboxSelected>>", self.apply_preset)

        tb.Label(tab_basic, text="✨ 正向提示词", font=("微软雅黑", 10)).pack(anchor=W, pady=(5,2), padx=8)
        self.txt_prompt = tb.Text(tab_basic, height=4)
        self.txt_prompt.pack(fill=X, padx=8)
        self.txt_prompt.insert(END, PROMPT_PRESETS["默认精美"]["p"])

        tb.Label(tab_basic, text="🚫 反向防崩坏词", font=("微软雅黑", 10)).pack(anchor=W, pady=(5,2), padx=8)
        self.txt_neg = tb.Text(tab_basic, height=3)
        self.txt_neg.pack(fill=X, padx=8)
        self.txt_neg.insert(END, PROMPT_PRESETS["默认精美"]["n"])

        params_frame = tb.Frame(tab_basic)
        params_frame.pack(fill=X, pady=10, padx=8)
        tb.Label(params_frame, text="分辨率").grid(row=0, column=0, pady=4, sticky=W)
        self.combo_res = tb.Combobox(params_frame, values=["512x512", "512x768", "768x512"], state="readonly")
        self.combo_res.current(0)
        self.combo_res.grid(row=0, column=1, pady=4, padx=10, sticky=EW)

        tb.Label(params_frame, text="生成步数").grid(row=1, column=0, pady=4, sticky=W)
        self.spin_steps = tb.Spinbox(params_frame, from_=10, to=100)
        self.spin_steps.set(30)
        self.spin_steps.grid(row=1, column=1, pady=4, padx=10, sticky=EW)

        frame_cfg = tb.Frame(tab_basic)  
        frame_cfg.pack(fill=X, pady=5)
        
        tb.Label(frame_cfg, text="🎨 引导系数(CFG):").pack(side=LEFT)
        self.lbl_cfg_val = tb.Label(frame_cfg, text="7.0", width=4, foreground="gray")
        self.lbl_cfg_val.pack(side=RIGHT)
        
        self.scale_cfg = tb.Scale(frame_cfg, from_=1.0, to=15.0, orient=HORIZONTAL)
        self.scale_cfg.set(7.0)  # 默认黄金数值 7.0
        self.scale_cfg.pack(side=RIGHT, fill=X, expand=True, padx=5)
        def update_cfg(val):
            self.lbl_cfg_val.config(text=f"{float(val):.1f}")
        self.scale_cfg.config(command=update_cfg)

        frame_sampler = tb.Frame(tab_basic)
        frame_sampler.pack(fill=X, pady=5)
        
        tb.Label(frame_sampler, text="🎲 采样算法:").pack(side=LEFT)
        
        self.combo_sampler = tb.Combobox(
            frame_sampler, 
            state="readonly", 
            values=["Euler a", "Euler", "DPM++ 2M Karras", "默认"]
        )
        self.combo_sampler.current(0)  # 默认选中你最爱用的 Euler a
        self.combo_sampler.pack(side=RIGHT, fill=X, expand=True, padx=5)

        tb.Label(params_frame, text="生成数量").grid(row=2, column=0, pady=4, sticky=W)
        self.spin_count = tb.Spinbox(params_frame, from_=1, to=20)
        self.spin_count.set(1)
        self.spin_count.grid(row=2, column=1, pady=4, padx=10, sticky=EW)
        params_frame.columnconfigure(1, weight=1)

        # ==========================================
        # 🟡 选项卡 2：高级炼丹
        # ==========================================
        self.frame_hires = tb.Labelframe(tab_adv, text=" 🔍 高清修复 (Hires. fix) ")
        self.frame_hires.pack(fill=X, padx=10, pady=(10, 5))
        self.var_enable_hires = tb.BooleanVar(value=False)
        
        def toggle_hires():
            state = "readonly" if self.var_enable_hires.get() else DISABLED
            self.combo_hires_scale.config(state=state)
            self.scale_hires_denoise.config(state=NORMAL if self.var_enable_hires.get() else DISABLED)

        tb.Checkbutton(self.frame_hires, text="开启 Hires.fix 画质飙升", variable=self.var_enable_hires, bootstyle="round-toggle", command=toggle_hires).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky=W)
        
        tb.Label(self.frame_hires, text="放大倍数:").grid(row=1, column=0, padx=10, pady=5, sticky=W)
        self.combo_hires_scale = tb.Combobox(self.frame_hires, values=["1.5", "2.0", "2.5"], width=8, state=DISABLED)
        self.combo_hires_scale.current(0)
        self.combo_hires_scale.grid(row=1, column=1, padx=10, pady=5, sticky=W)

        tb.Label(self.frame_hires, text="重绘幅度 (Denoising):").grid(row=2, column=0, padx=10, pady=(5, 10), sticky=W)
        self.scale_hires_denoise = tb.Scale(self.frame_hires, from_=0.1, to=0.8, value=0.4, state=DISABLED)
        self.scale_hires_denoise.grid(row=2, column=1, padx=10, pady=(5, 10), sticky=EW)
        self.frame_hires.columnconfigure(1, weight=1)

        # --- 炼丹九宫格 ---
        self.frame_xy = tb.Labelframe(tab_adv, text=" 炼丹九宫格 (X/Y Plot) ")
        self.frame_xy.pack(fill=X, padx=10, pady=5)
        
        self.var_enable_xy = tb.BooleanVar(value=False)

        def toggle_xy():
            state_combo = "readonly" if self.var_enable_xy.get() else "disabled"
            state_entry = "normal" if self.var_enable_xy.get() else "disabled"
            
            self.combo_x_type.config(state=state_combo)
            self.entry_x_vals.config(state=state_entry)
            self.combo_y_type.config(state=state_combo)
            self.entry_y_vals.config(state=state_entry)

        tb.Checkbutton(self.frame_xy, text="开启 X/Y 轴网格生成", variable=self.var_enable_xy, bootstyle="round-toggle", command=toggle_xy).grid(row=0, column=0, columnspan=3, sticky=W, padx=10, pady=(10, 5))

        xy_options = ["迭代步数 (Steps)", "提示词引导 (CFG)", "随机种子 (Seed)"]
        
        tb.Label(self.frame_xy, text="X:").grid(row=1, column=0, sticky=W, padx=(10, 2), pady=2)
        self.combo_x_type = tb.Combobox(self.frame_xy, values=xy_options, width=12)
        self.combo_x_type.current(1)
        self.combo_x_type.grid(row=1, column=1, sticky=W, padx=2, pady=2)
        
        self.entry_x_vals = tb.Entry(self.frame_xy, width=15) 
        self.entry_x_vals.insert(0, "6, 7, 8") 
        self.entry_x_vals.grid(row=1, column=2, sticky=EW, padx=(2, 10), pady=2)

        tb.Label(self.frame_xy, text="Y:").grid(row=2, column=0, sticky=W, padx=(10, 2), pady=(2, 10))
        self.combo_y_type = tb.Combobox(self.frame_xy, values=xy_options, width=12)
        self.combo_y_type.current(0)
        self.combo_y_type.grid(row=2, column=1, sticky=W, padx=2, pady=(2, 10))
        
        self.entry_y_vals = tb.Entry(self.frame_xy, width=15)
        self.entry_y_vals.insert(0, "20, 30, 40")
        self.entry_y_vals.grid(row=2, column=2, sticky=EW, padx=(2, 10), pady=(2, 10))
        
        self.frame_xy.columnconfigure(1, weight=0)
        self.frame_xy.columnconfigure(2, weight=1)
        toggle_xy()

        # --- 图像后处理面板 ---
        self.frame_post = tb.Labelframe(tab_adv, text=" 图像后处理与排版 ")
        self.frame_post.pack(fill=X, padx=10, pady=(5, 10))

        self.var_use_adetailer = tb.BooleanVar(value=False)
        
        # 👇 修改点 1：完善 toggle_adetailer 函数，加入对滑块的禁用/启用控制
        def toggle_adetailer():
            is_enabled = self.var_use_adetailer.get()
            self.combo_adetailer_model.config(state="readonly" if is_enabled else DISABLED)
            self.scale_adetailer_strength.config(state=NORMAL if is_enabled else DISABLED)

        tb.Checkbutton(self.frame_post, text="开启 ADetailer 面部修复", variable=self.var_use_adetailer, bootstyle="round-toggle", command=toggle_adetailer).pack(anchor=W, padx=10, pady=(10, 2))

        self.combo_adetailer_model = tb.Combobox(self.frame_post, values=["二次元脸", "真人写实"], width=12, state=DISABLED)
        self.combo_adetailer_model.current(0)
        self.combo_adetailer_model.pack(anchor=W, padx=35, pady=(0, 5)) # 微调了 pady
        
        # 👇 修改点 2：新增面部修复重绘强度滑块
        tb.Label(self.frame_post, text="面部重绘强度 (越小越像原图):").pack(anchor=W, padx=35, pady=(0, 2))
        self.scale_adetailer_strength = tb.Scale(self.frame_post, from_=0.1, to=0.7, value=0.35, state=DISABLED)
        self.scale_adetailer_strength.pack(anchor=W, padx=35, pady=(0, 10), fill=X)

        self.var_make_comic = tb.BooleanVar(value=False)
        tb.Checkbutton(self.frame_post, text="生成后拼合为连环画", variable=self.var_make_comic, bootstyle="round-toggle").pack(anchor=W, padx=10, pady=(0, 10))

        # ==========================================
        # 🟣 选项卡 3：图像控制
        # ==========================================
        i2i_frame = tb.Labelframe(tab_img, text="🖼️ 图生图与遮罩")
        i2i_frame.pack(fill=X, padx=5, pady=8)
        tb.Button(i2i_frame, text="🔍 提取图片参数 (PNG Info)", bootstyle="secondary", command=self.read_png_info).pack(fill=X, pady=2, padx=10)
        tb.Button(i2i_frame, text="📂 载入底图/遮罩", bootstyle="info-outline", command=self.select_image).pack(fill=X, pady=5, padx=10)
        self.lbl_img_path = tb.Label(i2i_frame, text="未选择参考图", foreground="gray")
        self.lbl_img_path.pack(anchor=W, padx=10)
        self.btn_clear_ref = tb.Button(i2i_frame, text="❌ 清除底图/遮罩", bootstyle="link-danger", command=self.clear_reference)
        self.btn_clear_ref.pack(anchor=W, pady=2, padx=10)
        
        row_str = tb.Frame(i2i_frame)
        row_str.pack(fill=X, pady=5, padx=10)
        tb.Label(row_str, text="重绘幅度").pack(side=LEFT)
        self.lbl_str_val = tb.Label(row_str, text="0.60", width=4, bootstyle="info")
        self.lbl_str_val.pack(side=RIGHT)
        self.scale_str = tb.Scale(row_str, from_=0.1, to=1.0, value=0.6, bootstyle="info", command=lambda v: self.lbl_str_val.config(text=f"{float(v):.2f}"))
        self.scale_str.pack(side=LEFT, fill=X, expand=True, padx=5)

        cn_frame = tb.Labelframe(tab_img, text="🏋️‍♂️ ControlNet 强力控制", bootstyle="warning")
        cn_frame.pack(fill=X, padx=5, pady=8)
        self.var_use_pose = tb.BooleanVar(value=False)
        tb.Checkbutton(cn_frame, text="开启 ControlNet", variable=self.var_use_pose, bootstyle="warning-round-toggle").pack(anchor=W, pady=5, padx=10)
        self.combo_cn_type = tb.Combobox(cn_frame, state="readonly", values=["openpose", "canny", "depth"])
        self.combo_cn_type.current(0)
        self.combo_cn_type.pack(fill=X, pady=5, padx=10)
        tb.Button(cn_frame, text="📸 载入骨架/线稿", bootstyle="warning-outline", command=self.load_pose_image).pack(fill=X, pady=2, padx=10)
        self.lbl_pose_path = tb.Label(cn_frame, text="未选择参考图", foreground="gray")
        self.lbl_pose_path.pack(anchor=W, padx=10, pady=(0,5))

        # === 右侧预览区 ===
        right_panel = tb.Frame(self, padding=20)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        self.lbl_status = tb.Label(right_panel, text="V4.0 就绪。极速引擎与高级选项卡已挂载！", font=("微软雅黑", 12), foreground="cyan")
        self.lbl_status.pack(pady=(5, 10))
        
        # === 进度条区域 ===
        progress_frame = tb.Frame(right_panel)
        progress_frame.pack(fill=X, pady=5)
        
        # 1. 单张步数进度
        lbl_prg_step = tb.Label(progress_frame, text="单张生成步数:", font=("微软雅黑", 9), foreground="gray")
        lbl_prg_step.pack(anchor=W)
        self.progress = tb.Progressbar(progress_frame, bootstyle="success-striped", mode="determinate")
        self.progress.pack(fill=X, pady=(2, 8))
        
        # 2. 批量总任务进度
        lbl_prg_total = tb.Label(progress_frame, text="批量任务总进度:", font=("微软雅黑", 9), foreground="gray")
        lbl_prg_total.pack(anchor=W)
        self.progress_total = tb.Progressbar(progress_frame, bootstyle="info-striped", mode="determinate")
        self.progress_total.pack(fill=X, pady=(2, 5))

        preview_frame = tb.Frame(right_panel)
        preview_frame.pack(fill=BOTH, expand=True, pady=10)
        self.pose_canvas = tb.Label(preview_frame, text="[ 骨架预览区 ]", background="#1a1a1a", anchor=CENTER)
        self.pose_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        self.preview_canvas = tb.Label(preview_frame, text="[ 生成图片区 ]", background="#2b2b2b", anchor=CENTER)
        self.preview_canvas.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        # 👇 修改点：统一挂载到完善后的 open_editor 逻辑
        self.btn_edit = tb.Button(right_panel, text="🎨 将此图送入编辑器 (支持局部涂鸦)", bootstyle="warning", state=DISABLED, command=self.open_editor)
        self.btn_edit.pack(pady=5)
        self.btn_upscale = tb.Button(right_panel, text="🔍 一键 AI 高清放大 (耗时较长)", bootstyle="info-outline", state=DISABLED, command=self.start_upscale)
        self.btn_upscale.pack(pady=2)

    # --- 工具函数 ---
    def apply_preset(self, event):
        preset_name = self.combo_preset.get()
        if preset_name in PROMPT_PRESETS:
            self.txt_prompt.delete("1.0", END)
            self.txt_prompt.insert(END, PROMPT_PRESETS[preset_name]["p"])
            self.txt_neg.delete("1.0", END)
            self.txt_neg.insert(END, PROMPT_PRESETS[preset_name]["n"])
            self.lbl_status.config(text=f"✨ 已应用预设: {preset_name}", foreground="cyan")

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
            
        # 2. 强制系统垃圾回收（非常重要！）
        import gc
        gc.collect()
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

    def apply_adetailer(self, base_image, prompt, negative_prompt, seed):
        try:
            cv_img = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            is_anime = (self.combo_adetailer_model.get() == "二次元脸")
            if is_anime:
                xml_path = "lbpcascade_animeface.xml"
                if not os.path.exists(xml_path):
                    url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
                    urllib.request.urlretrieve(url, xml_path)
                face_cascade = cv2.CascadeClassifier(xml_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
            else:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            if len(faces) == 0:
                self.after(0, lambda: self.lbl_status.config(text="🧑‍🎨 ADetailer: 未检测到明显人脸，跳过修复。", foreground="gray"))
                return base_image 

            self.after(0, lambda: self.lbl_status.config(text=f"🧑‍🎨 ADetailer: 侦测到 {len(faces)} 张人脸，正在逐一精修...", foreground="yellow"))
            result_image = base_image.copy()
            
            for idx, (x, y, w, h) in enumerate(faces):
                try:
                    self.after(0, lambda i=idx+1, t=len(faces): self.lbl_status.config(text=f"🧑‍🎨 ADetailer: 正在修复第 {i}/{t} 张脸...", foreground="yellow"))
                    
                    margin_x, margin_y = int(w * 0.4), int(h * 0.4)
                    x1, y1 = max(0, x - margin_x), max(0, y - int(margin_y * 1.5)) 
                    x2, y2 = min(base_image.width, x + w + margin_x), min(base_image.height, y + h + margin_y)
                    crop_w, crop_h = x2 - x1, y2 - y1
                    
                    face_crop = base_image.crop((x1, y1, x2, y2))
                    face_crop_512 = face_crop.resize((512, 512), Image.Resampling.LANCZOS)

                    extra_tag = "highly detailed anime face, perfect eyes, masterpiece" if is_anime else "beautiful detailed face, highly detailed eyes, perfectly symmetrical face, raw photo"
                    enhanced_prompt = prompt + ", " + extra_tag
                    generator = torch.Generator(self.ai.device).manual_seed(seed)
                    
                    with torch.no_grad():
                        fixed_face_512 = self.ai.img2img_pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=negative_prompt,
                            image=face_crop_512,
                            strength=float(self.scale_adetailer_strength.get()), 
                            num_inference_steps=25,
                            generator=generator
                        ).images[0]
                    
                    fixed_face = fixed_face_512.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
                    
                    mask = Image.new("L", (crop_w, crop_h), 0)
                    draw = ImageDraw.Draw(mask)
                    blur_radius = max(5, min(50, int(min(crop_w, crop_h) * 0.15))) 
                    draw.rectangle([blur_radius, blur_radius, crop_w-blur_radius, crop_h-blur_radius], fill=255)
                    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius)) 
                    
                    result_image.paste(fixed_face, (x1, y1), mask) 
                except Exception as inner_e:
                    continue 

            return result_image
        except Exception as e:
            return base_image

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
                # 自动模式
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 将设备告诉底层模型引擎
            if getattr(self.ai, 'device', None) != target_device:
                print(f"🔄 正在切换计算设备: {self.ai.device} -> {target_device}")
                self.ai.device = target_device
                # 如果设备变了，强制清理显存
                if hasattr(self.ai, 'clear_memory'):
                    self.ai.clear_memory()
            
            model_name = self.combo_model.get()
            
            # === 解析提示词 ===
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

            # === 【核心顺序修正】第一步：先加载底层大模型 ===
            self.after(0, lambda: self.lbl_status.config(text="🧠 正在加载底层大模型...", foreground="yellow"))
            self.ai.load_model(model_name)
            
            # === 【核心顺序修正】第二步：大模型就绪后，再加载 3 槽位 LoRA ===
            lora_config_list = []
            lora_meta_info = [] # 用于保存图片元数据
            for i in range(3):
                lname = self.combo_loras[i].get()
                lweight = self.scale_loras[i].get()
                if lname and lname != "无":
                    lora_config_list.append((lname, float(lweight)))
                    lora_meta_info.append(f"{lname}:{lweight}")

            sub_dir = "sdxl" if self.ai.is_sdxl else "sd1.5"
            print(f"👉 准备融合 LoRA 组合: {lora_config_list}")
            self.ai.apply_multiple_loras(lora_config_list, sub_dir=sub_dir)
            
            # === 准备 ControlNet ===
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

            # === 开始生成循环 ===
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
                    "prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
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
                    if getattr(self, "var_enable_xy", None) and self.var_enable_xy.get():
                        self.run_xy_plot_task(kwargs, width, height, pose_image=pose_image if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() else None)
                        return
                    
                    if getattr(self, 'var_use_pose', None) and self.var_use_pose.get() and pose_image:
                        image = self.ai.controlnet_pipe(**kwargs, width=width, height=height, image=pose_image).images[0]
                    elif getattr(self, 'mask_image_path', None): 
                        init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                        mask_img = Image.open(self.mask_image_path).convert("L").resize((width, height))
                        image = self.ai.inpaint_pipe(**kwargs, image=init_img, mask_image=mask_img, strength=strength).images[0]
                    elif getattr(self, 'ref_image_path', None):
                        init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                        image = self.ai.img2img_pipe(**kwargs, image=init_img, strength=strength).images[0]
                    else:
                        image = self.ai.txt2img_pipe(**kwargs, width=width, height=height).images[0]

                    if getattr(self, "var_enable_hires", None) and self.var_enable_hires.get():
                        self.after(0, lambda: self.lbl_status.config(text="✨ 构图完成！正在进行 Hires. fix 高清重绘...", foreground="#00FFFF"))
                        hires_scale = float(self.combo_hires_scale.get())
                        denoise_strength = float(self.scale_hires_denoise.get())
                        target_w, target_h = int(width * hires_scale), int(height * hires_scale)
                        
                        base_upscaled = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        hires_kwargs = kwargs.copy()
                        hires_kwargs["image"] = base_upscaled
                        hires_kwargs["strength"] = denoise_strength
                        
                        image = self.ai.img2img_pipe(**hires_kwargs).images[0]

                if getattr(self, 'var_use_adetailer', None) and self.var_use_adetailer.get() and not getattr(self, 'cancel_flag', False):
                    self.after(0, lambda: self.lbl_status.config(text="🧑‍🎨 正在进行 ADetailer 脸部修复...", foreground="yellow"))
                    image = self.apply_adetailer(image, current_en_prompt, en_neg, current_seed)

                generated_images_list.append(image) 

                # === 保存带参数的图像 ===
                lora_str = ", ".join(lora_meta_info) if lora_meta_info else "无"
                metadata_str = f"{current_raw_prompt}\nNegative prompt: {raw_neg}\nSteps: {steps}, Size: {width}x{height}, Seed: {current_seed}, Model: {model_name}, LoRAs: {lora_str}"
                pnginfo = PngInfo()
                pnginfo.add_text("parameters", metadata_str)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(OUTPUT_DIR, f"v4_{timestamp}_{current_seed}.png") 
            
                image.save(save_path, pnginfo=pnginfo)
                self.after(0, lambda p=save_path: self.show_preview(p))

            # === 后处理逻辑 ===
            if getattr(self, "var_make_comic", None) and self.var_make_comic.get() and len(generated_images_list) > 1 and not getattr(self, 'cancel_flag', False):
                self.generate_comic_strip(generated_images_list)
            else:
                if not getattr(self, 'cancel_flag', False):
                    self.after(0, lambda: self.lbl_status.config(text="✅ 批量生成任务全部完成！", foreground="#00FF00"))

            self.after(0, lambda: self.progress_total.configure(value=total_generate_count))
        
        except InterruptedError:
            self.after(0, lambda: self.lbl_status.config(text="🛑 已打断", foreground="red"))
        except Exception as e:
            # 🌟 终极除错系统：如果再出错，控制台绝对会立刻标红抛出完整堆栈！
            import traceback
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
                self.after(0, self.update_preview_ui, preview_img)
            except Exception as e:
                pass
        return callback_kwargs

    def update_preview_ui(self, preview_img):
        img_tk = ImageTk.PhotoImage(preview_img.resize((512, 512), Image.Resampling.LANCZOS))
        self.preview_canvas.config(image=img_tk, text="")
        self.preview_canvas.image = img_tk

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
                    if self.var_use_pose.get() and pose_image:
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
        self.after(0, lambda: self.update_preview_ui(grid_canvas))
        self.after(0, lambda: self.lbl_status.config(text=f"🎉 X/Y 炼丹完成！已保存为: {os.path.basename(save_path)}", foreground="#00FF00"))

    def generate_comic_strip(self, image_list):
        if not image_list or len(image_list) < 2: return
        self.after(0, lambda: self.lbl_status.config(text="🎞️ 正在拼合生成分镜连环画...", foreground="yellow"))

        try:
            border_size, line_width = 25, 4  
            img_w, img_h = image_list[0].size

            num_imgs = len(image_list)
            cols = 2 if num_imgs >= 4 else 1
            rows = (num_imgs + cols - 1) // cols
            footer_height = 40 

            bg_w = cols * img_w + (cols + 1) * border_size
            bg_h = rows * img_h + (rows + 1) * border_size + footer_height

            comic_bg = Image.new("RGB", (bg_w, bg_h), "white")
            draw = ImageDraw.Draw(comic_bg)

            for i, img in enumerate(image_list):
                row, col = i // cols, i % cols
                if row == rows - 1 and num_imgs % cols != 0:
                    paste_x = (bg_w - img_w) // 2
                else:
                    paste_x = border_size + col * (img_w + border_size)
                
                paste_y = border_size + row * (img_h + border_size)
                comic_bg.paste(img.resize((img_w, img_h), Image.Resampling.LANCZOS), (paste_x, paste_y))
            
                box = [paste_x - line_width, paste_y - line_width, 
                       paste_x + img_w + line_width - 1, paste_y + img_h + line_width - 1]
                draw.rectangle(box, outline="black", width=line_width)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((border_size, bg_h - footer_height + 10), f"AI Storyboard generated at {timestamp}", fill=(100,100,100))

            file_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(OUTPUT_DIR, f"Comic_Storyboard_{file_time}.png")
            comic_bg.save(save_path)

            self.after(0, lambda p=save_path: self.show_preview(p))
            self.after(0, lambda: self.lbl_status.config(text=f"🎉 分镜连环画拼合完成！已保存为: {os.path.basename(save_path)}", foreground="#00FF00"))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.config(text="⚠️ 连环画拼合失败，单张图已保存。", foreground="orange"))

if __name__ == "__main__":
    app = AIDesktopApp()
    app.mainloop()