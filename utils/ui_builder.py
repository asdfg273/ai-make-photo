# ui_builder.py

import ttkbootstrap as tb
from tkinter import messagebox
import os
from PIL import Image, ImageTk
from ttkbootstrap.scrolled import ScrolledFrame 
from ttkbootstrap.constants import * 
from tkinter import messagebox, filedialog
from utils.app_utils import PROMPT_PRESETS

class UIMixin:
    """这是一个 UI 混入类，负责把界面的积木拼好"""
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