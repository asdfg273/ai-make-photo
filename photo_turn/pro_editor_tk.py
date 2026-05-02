# photo_turn/pro_editor_tk.py
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import threading
import time
import cv2
import numpy as np
import os
import sys


class CropOverlay:
    def __init__(self, canvas, start_x, start_y):
        self.canvas = canvas
        self.start_x = start_x
        self.start_y = start_y
        self.rect_id = None
        self.end_x = start_x
        self.end_y = start_y
        self._create_overlay()
    
    def _create_overlay(self):
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.end_x, self.end_y,
            outline='white', width=2, tags='crop_overlay',
            dash=(5, 5)
        )
    
    def update(self, x, y):
        self.end_x = x
        self.end_y = y
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, x, y)
    
    def delete(self):
        self.canvas.delete(self.rect_id)
    
    def get_bounds(self):
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)
        return (x1, y1, x2, y2)


class ProImageEditor(tb.Toplevel):
    def __init__(self, parent, image_path, callback_on_save=None):
        super().__init__(master=parent, title="✨ 专业级修图与 AI 遮罩引擎 (Tkinter)", size=(1400, 900))
        self.callback_on_save = callback_on_save
        self.image_path = image_path
        
        self.original_full_img = Image.open(image_path).convert("RGB")
        self.original_full_img.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        
        self.base_img = self.original_full_img.copy()
        self.current_img = self.base_img.copy()
        self.filter_base_img = self.base_img.copy()
        self.original_img = self.original_full_img.copy()
        
        self.mask_img = Image.new("L", self.base_img.size, 0)
        
        self.display_tk_img = None
        self.last_display_size = None
        
        self.history = []
        self.future = []
        
        self.crop_mode = False
        self.crop_overlay = None
        self.crop_start_pos = None
        
        self.text_mode = False
        self.current_text_string = ""
        self.text_element = None
        self.text_color = "white"
        self.text_size = 40
        self.dragging_text = False
        self.text_offset_x = 0
        self.text_offset_y = 0
        
        self.draw_mode = False
        self.is_mask_brush = False
        self.brush_color = "white"
        self.is_eraser = False
        
        self.adjust_vars = {}
        self.adjust_pending = False
        self.adjust_timer = None
        self.resize_timer = None
        
        self._adetailer_running = False
        
        self.setup_ui()
        self.push_history(self.current_img, self.mask_img)
        self.update_canvas(self.current_img, force=True)
        
        self._bind_shortcuts()

    def _bind_shortcuts(self):
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())
        self.bind_all("<Control-s>", lambda e: self.save_and_return())
        self.bind_all("<Escape>", lambda e: self._cancel_any_mode())
        
    def _cancel_any_mode(self):
        if self.crop_mode:
            self.toggle_crop()
        if self.text_mode:
            self.toggle_text()
        if self.draw_mode:
            self.draw_mode = False
            self.is_mask_brush = False
            self.is_eraser = False
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
        
    def setup_ui(self):
        # 顶部工具栏 - 类似PS
        top_toolbar = tb.Frame(self, height=50, padding=5)
        top_toolbar.pack(side=TOP, fill=X)
        
        # 左侧工具栏
        btn_save = tb.Button(top_toolbar, text="✅ 保存", 
                            bootstyle="success", command=self.save_and_return)
        btn_save.pack(side=LEFT, padx=2)
        
        tb.Button(top_toolbar, text="↩️ 撤销", 
                bootstyle="secondary", command=self.undo).pack(side=LEFT, padx=2)
        tb.Button(top_toolbar, text="↪️ 重做", 
                bootstyle="secondary", command=self.redo).pack(side=LEFT, padx=2)
        
        tb.Separator(top_toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        
        tb.Button(top_toolbar, text="🔄 重置为原图", 
                bootstyle="outline", command=self.reset_adjustments).pack(side=LEFT, padx=2)
        
        # 画布区域
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<ButtonPress-3>", self.on_mouse_right_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # 左侧滚动面板
        left_container = tb.Frame(self, width=340)
        left_container.pack(side=LEFT, fill=Y, expand=False)
        
        # 创建滚动面板 - 使用标准Tkinter实现
        left_canvas = tk.Canvas(left_container, bg="#2b2b2b")
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(left_container, orient=VERTICAL, command=left_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        left_canvas.configure(yscrollcommand=scrollbar.set)
        left_canvas.bind('<Configure>', lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        
        left_panel = tb.Frame(left_canvas, padding=10)
        left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        frame_ai = ttk.LabelFrame(left_panel, text="🤖 AI 重绘/局部调整", padding=8)
        frame_ai.pack(fill=X, pady=5)
        
        row_ai = tb.Frame(frame_ai)
        row_ai.pack(fill=X, pady=2)
        
        self.btn_mask = tb.Button(row_ai, text="🖍️ 遮罩画笔", 
                                bootstyle="danger-outline", command=self.toggle_mask_brush)
        self.btn_mask.pack(side=LEFT, expand=True, padx=2)
        
        btn_adetailer = tb.Button(row_ai, text="👤 ADetailer自动提脸", 
                                 bootstyle="warning-outline", command=self.run_adetailer)
        btn_adetailer.pack(side=LEFT, expand=True, padx=2)
        
        frame_draw = ttk.LabelFrame(left_panel, text="🖌️ 绘图 & 裁剪", padding=8)
        frame_draw.pack(fill=X, pady=5)
        
        row_tools = tb.Frame(frame_draw)
        row_tools.pack(fill=X, pady=2)
        self.btn_brush = tb.Button(row_tools, text="画笔", 
                                    bootstyle="outline", command=self.toggle_brush)
        self.btn_brush.pack(side=LEFT, expand=True, padx=2)
        
        self.btn_eraser = tb.Button(row_tools, text="橡皮", 
                                    bootstyle="outline", command=self.toggle_eraser)
        self.btn_eraser.pack(side=LEFT, expand=True, padx=2)
        
        self.btn_crop = tb.Button(row_tools, text="✂️ 裁剪", 
                                  bootstyle="outline", command=self.toggle_crop)
        self.btn_crop.pack(side=LEFT, expand=True, padx=2)
        
        row_color = tb.Frame(frame_draw)
        row_color.pack(fill=X, pady=5)
        
        btn_color_picker = tb.Button(row_color, text="🎨 画笔颜色", 
                                    bootstyle="info-outline", command=self.pick_color)
        btn_color_picker.pack(side=LEFT, expand=True, padx=2)
        
        self.color_preview = tk.Canvas(row_color, width=30, height=30, bg="white", highlightthickness=1, highlightbackground="gray")
        self.color_preview.pack(side=RIGHT, padx=5)
        
        tb.Label(frame_draw, text="画笔大小", font=("微软雅黑", 8), 
                 foreground="gray").pack(anchor="w", pady=(5,0))
        self.scale_brush_size = tb.Scale(frame_draw, from_=1, to=150, orient=HORIZONTAL)
        self.scale_brush_size.set(30)
        self.scale_brush_size.pack(fill=X, pady=2)
        
        row_brush_shape = tb.Frame(frame_draw)
        row_brush_shape.pack(fill=X, pady=2)
        tb.Label(row_brush_shape, text="形状:", font=("微软雅黑", 8), 
                 foreground="gray").pack(side=LEFT)
        self.brush_shape_var = tk.StringVar(value="round")
        tb.Radiobutton(row_brush_shape, text="圆形", variable=self.brush_shape_var, 
                      value="round").pack(side=LEFT, padx=10)
        tb.Radiobutton(row_brush_shape, text="方形", variable=self.brush_shape_var, 
                      value="square").pack(side=LEFT, padx=10)
        
        tb.Label(frame_draw, text="硬度", font=("微软雅黑", 8), 
                 foreground="gray").pack(anchor="w", pady=(5,0))
        self.scale_brush_hardness = tb.Scale(frame_draw, from_=0, to=100, orient=HORIZONTAL)
        self.scale_brush_hardness.set(100)
        self.scale_brush_hardness.pack(fill=X, pady=2)
        
        frame_text = ttk.LabelFrame(left_panel, text="🔤 文字工具", padding=8)
        frame_text.pack(fill=X, pady=5)
        
        self.btn_text = tb.Button(frame_text, text="添加文字", 
                                  bootstyle="primary-outline", command=self.toggle_text)
        self.btn_text.pack(fill=X, pady=2)
        
        row_text_color = tb.Frame(frame_text)
        row_text_color.pack(fill=X, pady=2)
        btn_text_color = tb.Button(row_text_color, text="🖊️ 文字颜色", 
                                  bootstyle="primary-outline", command=self.pick_text_color)
        btn_text_color.pack(side=LEFT, expand=True, padx=2)
        self.text_color_preview = tk.Canvas(row_text_color, width=30, height=30, 
                                           bg="white", highlightthickness=1, highlightbackground="gray")
        self.text_color_preview.pack(side=RIGHT, padx=5)
        
        tb.Label(frame_text, text="文字大小", font=("微软雅黑", 8), 
               foreground="gray").pack(anchor="w", pady=(5,0))
        self.scale_text_size = tb.Scale(frame_text, from_=10, to=200, orient=HORIZONTAL)
        self.scale_text_size.set(40)
        self.scale_text_size.pack(fill=X, pady=2)
        
        frame_transform = ttk.LabelFrame(left_panel, text="🔄 变换操作", padding=8)
        frame_transform.pack(fill=X, pady=5)
        
        row_flip = tb.Frame(frame_transform)
        row_flip.pack(fill=X, pady=2)
        tb.Button(row_flip, text="↔️ 水平翻转", 
                 bootstyle="outline", command=lambda: self.flip_image("horizontal")).pack(side=LEFT, expand=True, padx=2)
        tb.Button(row_flip, text="↕️ 垂直翻转", 
                 bootstyle="outline", command=lambda: self.flip_image("vertical")).pack(side=LEFT, expand=True, padx=2)
        
        row_rotate = tb.Frame(frame_transform)
        row_rotate.pack(fill=X, pady=2)
        tb.Button(row_rotate, text="↻ 左转90°", 
                 bootstyle="outline", command=lambda: self.rotate_image(-90)).pack(side=LEFT, expand=True, padx=2)
        tb.Button(row_rotate, text="↺ 右转90°", 
                 bootstyle="outline", command=lambda: self.rotate_image(90)).pack(side=LEFT, expand=True, padx=2)
        
        frame_adj = ttk.LabelFrame(left_panel, text="📊 图像调整", padding=8)
        frame_adj.pack(fill=X, pady=5)
        
        self.adjust_vars = {}
        adjust_configs = [
            ("brightness", "🔆 亮度", -100, 100, 0),
            ("contrast", "◐ 对比度", -100, 100, 0),
            ("saturation", "🎨 饱和度", -100, 100, 0),
            ("sharpness", "🔪 锐化", 0, 100, 0),
            ("temperature", "🌡️ 色温", -100, 100, 0),
        ]
        
        for key, label, min_v, max_v, default in adjust_configs:
            row = tb.Frame(frame_adj)
            row.pack(fill=X, pady=2)
            
            lbl = tb.Label(row, text=label, width=12, font=("微软雅黑", 9))
            lbl.pack(side=LEFT)
            
            var = tk.IntVar(value=default)
            self.adjust_vars[key] = var
            
            val_lbl = tb.Label(row, text=str(default), width=4, font=("微软雅黑", 9))
            val_lbl.pack(side=RIGHT)
            
            def make_update_callback(v, label_widget):
                def callback(*args):
                    label_widget.config(text=str(v.get()))
                return callback
            
            scale = tb.Scale(row, from_=min_v, to=max_v, orient=HORIZONTAL, variable=var,
                           command=lambda v, k=key: self.on_adjust_change(k, int(float(v))))
            scale.pack(side=LEFT, fill=X, expand=True, padx=(5, 0))
            
            var.trace_add("write", make_update_callback(var, val_lbl))
        
        frame_filter = ttk.LabelFrame(left_panel, text="🎨 预设滤镜", padding=8)
        frame_filter.pack(fill=X, pady=5)
        
        self.filter_combo = tb.Combobox(frame_filter, 
                                        values=["无", "黑白", "复古", "冷色调", 
                                               "暖色调", "胶片颗粒", "模糊", "浮雕", 
                                               "边缘检测", "轮廓", "锐化", "油画"], 
                                        state="readonly")
        self.filter_combo.current(0)
        self.filter_combo.pack(fill=X, pady=(0, 5))
        
        row_filter = tb.Frame(frame_filter)
        row_filter.pack(fill=X)
        self.blur_scale = tb.Scale(row_filter, from_=0, to=20, orient=HORIZONTAL, value=5)
        self.blur_scale.pack(side=LEFT, fill=X, expand=True)
        tb.Label(row_filter, text="模糊", width=6).pack(side=RIGHT)
        
        btn_apply_filter = tb.Button(frame_filter, text="✨ 应用滤镜", 
                                   bootstyle="info", command=self.apply_selected_filter)
        btn_apply_filter.pack(fill=X, pady=(5, 0))
        
        # 底部状态栏
        self.lbl_status = tb.Label(left_container, text="✅ 就绪 | ESC取消", 
                                  font=("微软雅黑", 9), foreground="gray", padding=5)
        self.lbl_status.pack(side=BOTTOM, fill=X)

    def on_canvas_resize(self, event):
        if self.resize_timer:
            self.after_cancel(self.resize_timer)
        self.resize_timer = self.after(50, lambda: self.update_canvas(self.current_img))

    def on_mouse_right_click(self, event):
        print(f"右键点击: crop_mode={self.crop_mode}, crop_overlay={self.crop_overlay is not None}")
        if self.text_mode and self.text_element:
            self._commit_text_to_image()
        elif self.crop_mode and self.crop_overlay:
            print("触发裁剪操作")
            self._apply_crop()
        else:
            print(f"右键点击未触发操作: text_mode={self.text_mode}, text_element={self.text_element is not None}")
        
    def pick_color(self):
        self.grab_set()
        try:
            color_result = colorchooser.askcolor(
                title="选择画笔颜色",
                parent=self,
                initialcolor=self.brush_color
            )
            if color_result[1]:
                self.brush_color = color_result[1]
                self.color_preview.config(bg=self.brush_color)
                self.lbl_status.config(text=f"✅ 已选择颜色: {color_result[1]}", 
                                       foreground="cyan")
        finally:
            self.grab_release()
        
    def pick_text_color(self):
        self.grab_set()
        try:
            color_result = colorchooser.askcolor(
                title="选择文字颜色",
                parent=self,
                initialcolor=self.text_color
            )
            if color_result[1]:
                self.text_color = color_result[1]
                self.text_color_preview.config(bg=self.text_color)
                if self.text_element:
                    self.canvas.itemconfig(self.text_element, fill=self.text_color)
                self.lbl_status.config(text=f"✅ 已选择文字颜色: {color_result[1]}", 
                                       foreground="cyan")
        finally:
            self.grab_release()
            
    def on_adjust_change(self, key, value):
        if self.adjust_timer:
            self.after_cancel(self.adjust_timer)
        self.adjust_timer = self.after(150, self.apply_adjustments)

    def apply_adjustments(self):
        try:
            if self.base_img is None:
                return
                
            img = self.base_img.copy()
            
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[-1]
                img = img.convert("RGB")
            
            brightness = self.adjust_vars["brightness"].get()
            contrast = self.adjust_vars["contrast"].get()
            saturation = self.adjust_vars["saturation"].get()
            sharpness = self.adjust_vars["sharpness"].get()
            temperature = self.adjust_vars["temperature"].get()
            
            if brightness != 0:
                img = ImageEnhance.Brightness(img).enhance(1.0 + brightness / 100.0)
            
            if contrast != 0:
                img = ImageEnhance.Contrast(img).enhance(1.0 + contrast / 100.0)
            
            if saturation != 0:
                img = ImageEnhance.Color(img).enhance(1.0 + saturation / 100.0)
            
            if sharpness != 0:
                img = ImageEnhance.Sharpness(img).enhance(1.0 + sharpness / 50.0)
            
            if temperature != 0:
                r, g, b = img.split()
                factor_r = 1.0 + temperature / 200.0
                factor_b = 1.0 - temperature / 200.0
                r = r.point(lambda v: min(255, max(0, int(v * factor_r))))
                b = b.point(lambda v: min(255, max(0, int(v * factor_b))))
                img = Image.merge("RGB", (r, g, b))
            
            if has_alpha:
                img = img.convert("RGBA")
                img.putalpha(alpha)
            
            self.current_img = img
            self.update_canvas(self.current_img, force=True)
        except Exception as e:
            print(f"图像调整错误: {e}")
            import traceback
            traceback.print_exc()

    def reset_adjustments(self):
        for key, var in self.adjust_vars.items():
            var.set(0)
        self.current_img = self.original_img.copy()
        self.base_img = self.original_img.copy()
        self.filter_base_img = self.original_img.copy()
        self.update_canvas(self.current_img, force=True)

    def apply_selected_filter(self):
        try:
            filter_name = self.filter_combo.get()
            print(f"应用滤镜: {filter_name}")
            
            if filter_name == "无":
                return
                
            img = self.filter_base_img.copy()
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[-1]
                img = img.convert("RGB")
            
            if filter_name == "黑白":
                gray = ImageOps.grayscale(img).convert("RGB")
                img = ImageEnhance.Contrast(gray).enhance(1.2)
            
            elif filter_name == "复古":
                img = ImageEnhance.Color(img).enhance(0.6)
                r, g, b = img.split()
                r = r.point(lambda v: min(255, int(v * 1.1) + 20))
                g = g.point(lambda v: min(255, int(v * 1.05) + 10))
                b = b.point(lambda v: max(0, int(v * 0.85)))
                img = Image.merge("RGB", (r, g, b))
                img = ImageEnhance.Contrast(img).enhance(0.95)
            
            elif filter_name == "冷色调":
                r, g, b = img.split()
                r = r.point(lambda v: max(0, int(v * 0.85)))
                b = b.point(lambda v: min(255, int(v * 1.2)))
                img = Image.merge("RGB", (r, g, b))
            
            elif filter_name == "暖色调":
                r, g, b = img.split()
                r = r.point(lambda v: min(255, int(v * 1.2)))
                g = g.point(lambda v: min(255, int(v * 1.05)))
                b = b.point(lambda v: max(0, int(v * 0.85)))
                img = Image.merge("RGB", (r, g, b))
            
            elif filter_name == "胶片颗粒":
                px = img.load()
                w, h = img.size
                for _ in range(int(w * h * 0.03)):
                    x = np.random.randint(0, w)
                    y = np.random.randint(0, h)
                    noise = np.random.randint(-40, 40)
                    r, g, b = px[x, y]
                    px[x, y] = (
                        max(0, min(255, r + noise)),
                        max(0, min(255, g + noise)),
                        max(0, min(255, b + noise)),
                    )
            
            elif filter_name == "模糊":
                radius = self.blur_scale.get()
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
            elif filter_name == "浮雕":
                img = img.filter(ImageFilter.EMBOSS)
            
            elif filter_name == "边缘检测":
                img = img.filter(ImageFilter.FIND_EDGES)
            
            elif filter_name == "轮廓":
                img = img.filter(ImageFilter.CONTOUR)
            
            elif filter_name == "锐化":
                img = img.filter(ImageFilter.SHARPEN)
                img = ImageEnhance.Sharpness(img).enhance(2.0)
            
            elif filter_name == "油画":
                img = img.filter(ImageFilter.ModeFilter(5))
                img = ImageEnhance.Color(img).enhance(1.2)
            
            if has_alpha:
                img = img.convert("RGBA")
                img.putalpha(alpha)
            
            self.push_history(self.current_img, self.mask_img)
            self.current_img = img
            self.base_img = img.copy()
            self.push_history(self.current_img, self.mask_img)
            self.update_canvas(self.current_img, force=True)
            self.lbl_status.config(text=f"✅ 已应用滤镜: {filter_name}", foreground="green")
        except Exception as e:
            print(f"滤镜应用错误: {e}")
            import traceback
            traceback.print_exc()

    def run_adetailer(self):
        if self._adetailer_running:
            messagebox.showinfo("提示", "ADetailer 正在运行中，请稍候...")
            return
            
        self._adetailer_running = True
        threading.Thread(target=self._adetailer_worker, daemon=True).start()

    def _adetailer_worker(self):
        try:
            self.after(0, lambda: self.lbl_status.config(text="🔍 正在检测人脸...", 
                                                     foreground="yellow"))
            
            cv_img = cv2.cvtColor(np.array(self.current_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                self.after(0, lambda: messagebox.showinfo("提示", "未检测到明显的人脸。"))
                self._adetailer_running = False
                return
            
            self.after(0, lambda: self.lbl_status.config(text=f"👤 检测到 {len(faces)} 张人脸，正在处理...", 
                                                     foreground="cyan"))
            
            result_img = self.current_img.copy()
            result_mask = self.mask_img.copy()
            
            for idx, (x, y, w, h) in enumerate(faces):
                self.after(0, lambda i=idx+1, t=len(faces): self.lbl_status.config(
                    text=f"🧑‍🎨 ADetailer: 正在修复第 {i}/{t} 张脸...", 
                    foreground="yellow"))
                
                margin_x, margin_y = int(w * 0.4), int(h * 0.4)
                x1, y1 = max(0, x - margin_x), max(0, y - int(margin_y * 1.5))
                x2, y2 = min(result_img.width, x + w + margin_x), min(result_img.height, y + h + margin_y)
                crop_w, crop_h = x2 - x1, y2 - y1
                
                face_crop = result_img.crop((x1, y1, x2, y2))
                face_crop_512 = face_crop.resize((512, 512), Image.LANCZOS)
                
                try:
                    from model_manager import ModelManager
                    manager = ModelManager()
                    enhanced_face = manager.img2img_pipe(
                        prompt="highly detailed face, perfect eyes, symmetrical face, beautiful skin, masterpiece, best quality",
                        negative_prompt="blurry, low quality, distorted face, ugly",
                        image=face_crop_512,
                        strength=0.35,
                        num_inference_steps=25
                    ).images[0]
                    
                    fixed_face = enhanced_face.resize((crop_w, crop_h), Image.LANCZOS)
                    
                    mask = Image.new("L", (crop_w, crop_h), 0)
                    draw = ImageDraw.Draw(mask)
                    blur_radius = max(5, min(50, int(min(crop_w, crop_h) * 0.15)))
                    draw.rectangle([blur_radius, blur_radius, crop_w-blur_radius, crop_h-blur_radius], fill=255)
                    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
                    
                    result_img.paste(fixed_face, (x1, y1), mask)
                    
                    draw_mask = ImageDraw.Draw(result_mask)
                    draw_mask.ellipse([x1, y1, x2, y2], fill=255)
                    
                except Exception as inner_e:
                    continue
            
            self.after(0, lambda: self.lbl_status.config(text="✅ ADetailer 处理完成", 
                                                     foreground="green"))
            self.after(0, lambda: self.on_adetailer_complete(result_img, result_mask))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", f"ADetailer 处理失败: {str(e)}"))
        finally:
            self._adetailer_running = False

    def on_adetailer_complete(self, result_img, result_mask):
        self.push_history(self.current_img, self.mask_img)
        self.current_img = result_img
        self.mask_img = result_mask
        self.update_canvas(self.current_img)
        messagebox.showinfo("成功", "ADetailer 人脸修复完成！")
        
    def _reset_buttons(self):
        self.btn_mask.config(bootstyle="danger-outline")
        self.btn_brush.config(bootstyle="outline")
        self.btn_eraser.config(bootstyle="outline")
        self.btn_crop.config(bootstyle="outline")
        self.btn_text.config(bootstyle="primary-outline")
        
    def toggle_mask_brush(self):
        if self.draw_mode and self.is_mask_brush:
            self.draw_mode = False
            self.is_mask_brush = False
            self.is_eraser = False
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
            return
            
        self.draw_mode = True
        self.is_mask_brush = True
        self.is_eraser = False
        self.text_mode = False
        self.crop_mode = False
        self.text_element = None
        if self.crop_overlay:
            self.crop_overlay.delete()
            self.crop_overlay = None
        
        self._reset_buttons()
        self.btn_mask.config(bootstyle="danger")
        self.canvas.config(cursor="crosshair")
        self.lbl_status.config(text="🔴 遮罩画笔模式", foreground="cyan")

    def toggle_brush(self):
        if self.draw_mode and not self.is_mask_brush and not self.is_eraser:
            self.draw_mode = False
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
            return
            
        self.draw_mode = True
        self.is_mask_brush = False
        self.is_eraser = False
        self.text_mode = False
        self.crop_mode = False
        self.text_element = None
        if self.crop_overlay:
            self.crop_overlay.delete()
            self.crop_overlay = None
        
        self._reset_buttons()
        self.btn_brush.config(bootstyle="solid")
        self.canvas.config(cursor="pencil")
        self.lbl_status.config(text="🖌️ 画笔模式", foreground="cyan")

    def toggle_eraser(self):
        if self.draw_mode and self.is_eraser:
            self.draw_mode = False
            self.is_eraser = False
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
            return
            
        self.draw_mode = True
        self.is_mask_brush = False
        self.is_eraser = True
        self.text_mode = False
        self.crop_mode = False
        self.text_element = None
        if self.crop_overlay:
            self.crop_overlay.delete()
            self.crop_overlay = None
        
        self._reset_buttons()
        self.btn_eraser.config(bootstyle="solid")
        self.canvas.config(cursor="dot")
        self.lbl_status.config(text="🧽 橡皮擦模式", foreground="cyan")
        
    def toggle_crop(self):
        if self.crop_mode:
            self.crop_mode = False
            if self.crop_overlay:
                self.crop_overlay.delete()
                self.crop_overlay = None
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
            return
            
        self.crop_mode = True
        self.text_mode = False
        self.text_element = None
        self.draw_mode = False
        self.is_mask_brush = False
        self.is_eraser = False
        
        self._reset_buttons()
        self.btn_crop.config(bootstyle="solid")
        self.canvas.config(cursor="cross")
        self.lbl_status.config(text="✂️ 裁剪模式: 拖动选择区域，右键裁剪", foreground="cyan")

    def toggle_text(self):
        if self.text_mode:
            self.text_mode = False
            self.text_element = None
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 就绪", foreground="gray")
            return
            
        self.draw_mode = False
        self.is_mask_brush = False
        self.is_eraser = False
        self.crop_mode = False
        if self.crop_overlay:
            self.crop_overlay.delete()
            self.crop_overlay = None
        
        self._reset_buttons()
        self.text_mode = True
        self.btn_text.config(bootstyle="primary")
        
        self.text_element = None
        self.lbl_status.config(text="📝 请输入文字...", foreground="cyan")
        self.update()
        
        text_content = simpledialog.askstring("输入文字", "请输入要添加的文字：", parent=self)
        
        if text_content and text_content.strip():
            self.current_text_string = text_content.strip()
            self.text_size = self.scale_text_size.get()
            self.canvas.config(cursor="crosshair")
            self.lbl_status.config(text="✅ 文字已就绪，点击画布添加，拖拽移动，右键保存", foreground="green")
        else:
            self.text_mode = False
            self.btn_text.config(bootstyle="primary-outline")
            self.canvas.config(cursor="arrow")
            self.lbl_status.config(text="✅ 已取消加字", foreground="gray")

    def on_mouse_press(self, event):
        if self.crop_mode:
            self.crop_start_pos = (event.x, event.y)
            self.crop_overlay = CropOverlay(self.canvas, event.x, event.y)
            return
            
        if self.text_mode:
            if self.text_element:
                bbox = self.canvas.bbox(self.text_element)
                if bbox and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                    self.dragging_text = True
                    self.text_offset_x = event.x - bbox[0]
                    self.text_offset_y = event.y - bbox[1]
                    self.lbl_status.config(text="✋ 拖拽移动文字", foreground="cyan")
                    return
            if not self.text_element:
                self.add_text_at_position(event.x, event.y)
            return
            
        if self.draw_mode:
            self.current_stroke = [(event.x, event.y)]
            self.last_x, self.last_y = event.x, event.y

    def on_mouse_drag(self, event):
        if self.crop_mode and self.crop_overlay:
            self.crop_overlay.update(event.x, event.y)
            return
            
        if self.dragging_text and self.text_element:
            x = event.x - self.text_offset_x
            y = event.y - self.text_offset_y
            self.canvas.coords(self.text_element, x, y)
            return
            
        if self.draw_mode and hasattr(self, 'current_stroke'):
            self.current_stroke.append((event.x, event.y))
            brush_size = int(self.scale_brush_size.get())
            
            if self.is_mask_brush:
                color = "red"
                stipple = "gray50"
            else:
                color = self.brush_color
                stipple = ""
            
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y, 
                fill=color, width=brush_size, capstyle="round", joinstyle="round", 
                tags="temp_stroke", stipple=stipple
            )
            self.last_x, self.last_y = event.x, event.y

    def on_mouse_release(self, event):
        if self.dragging_text and self.text_element:
            self.dragging_text = False
            self.lbl_status.config(text="✋ 位置已更新，右键保存", foreground="cyan")
            return
            
        if self.crop_mode and self.crop_overlay:
            self.lbl_status.config(text="✂️ 区域已选择，右键裁剪或ESC取消", foreground="cyan")
            return
            
        if self.draw_mode and hasattr(self, 'current_stroke'):
            if not self.current_stroke: return
            
            try:
                brush_size = int(self.scale_brush_size.get())
                
                if self.is_mask_brush:
                    draw_mask = ImageDraw.Draw(self.mask_img)
                    overlay = Image.new("RGBA", self.current_img.size, (0,0,0,0))
                    draw_overlay = ImageDraw.Draw(overlay)
                    
                    self._draw_smooth_line(self.current_stroke, draw_mask, brush_size, fill=255)
                    self._draw_smooth_line(self.current_stroke, draw_overlay, brush_size, fill=(255, 0, 0, 128))
                    
                    self.current_img = Image.alpha_composite(self.current_img.convert("RGBA"), overlay).convert("RGB")
                    
                elif self.is_eraser:
                    # 橡皮擦：从原始图像恢复像素
                    for i in range(len(self.current_stroke) - 1):
                        cx1, cy1 = self.current_stroke[i]
                        cx2, cy2 = self.current_stroke[i+1]
                        rx1, ry1, scale = self._get_real_xy(cx1, cy1)
                        rx2, ry2, _ = self._get_real_xy(cx2, cy2)
                        real_width = max(1, int(brush_size / scale))
                        
                        x1, x2 = int(min(rx1, rx2) - real_width), int(max(rx1, rx2) + real_width)
                        y1, y2 = int(min(ry1, ry2) - real_width), int(max(ry1, ry2) + real_width)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(self.current_img.width, x2), min(self.current_img.height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            orig_region = self.original_full_img.crop((x1, y1, x2, y2))
                            orig_region = orig_region.resize((x2-x1, y2-y1))
                            self.current_img.paste(orig_region, (x1, y1))
                else:
                    draw_img = ImageDraw.Draw(self.current_img)
                    self._draw_smooth_line(self.current_stroke, draw_img, brush_size, fill=self.brush_color)
                
                self.canvas.delete("temp_stroke")
                self.push_history(self.current_img, self.mask_img)
                self.update_canvas(self.current_img, force=True)
            except Exception as e:
                print(f"画笔提交错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if hasattr(self, 'current_stroke'):
                    del self.current_stroke
            
    def _apply_crop(self):
        try:
            if not self.crop_overlay:
                return
                
            x1, y1, x2, y2 = self.crop_overlay.get_bounds()
            rx1, ry1, scale = self._get_real_xy(x1, y1)
            rx2, ry2, _ = self._get_real_xy(x2, y2)
            
            rx1, rx2 = int(min(rx1, rx2)), int(max(rx1, rx2))
            ry1, ry2 = int(min(ry1, ry2)), int(max(ry1, ry2))
            
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(self.current_img.width, rx2), min(self.current_img.height, ry2)
            
            if rx2 - rx1 < 10 or ry2 - ry1 < 10:
                messagebox.showinfo("提示", "选择的区域太小了")
                return
            
            if not messagebox.askyesno("确认裁剪", "确定要裁剪选中区域吗？"):
                return
            
            cropped = self.current_img.crop((rx1, ry1, rx2, ry2))
            cropped_mask = self.mask_img.crop((rx1, ry1, rx2, ry2))
            
            self.current_img = cropped
            self.mask_img = cropped_mask
            self.base_img = self.current_img.copy()
            self.original_full_img = self.current_img.copy()
            
            self.crop_overlay.delete()
            self.crop_overlay = None
            self.crop_mode = False
            
            self._reset_buttons()
            self.canvas.config(cursor="arrow")
            self.push_history(self.current_img, self.mask_img)
            self.update_canvas(self.current_img, force=True)
            self.lbl_status.config(text="✅ 裁剪完成", foreground="green")
        except Exception as e:
            print(f"裁剪错误: {e}")
            import traceback
            traceback.print_exc()
        
    def add_text_at_position(self, x, y):
        text_content = self.current_text_string
        
        self.canvas.delete("text_preview") if self.canvas.find_withtag("text_preview") else None
        
        self.text_element = self.canvas.create_text(
            x, y, text=text_content, fill=self.text_color, 
            font=("微软雅黑", 18), tags="text_preview")
        
        self.lbl_status.config(text="✋ 文字已添加，拖拽移动，右键保存", foreground="green")
        self.canvas.config(cursor="fleur")

    def _commit_text_to_image(self):
        if not self.text_element:
            return
            
        bbox = self.canvas.bbox(self.text_element)
        if not bbox:
            return
            
        rx1, ry1, _ = self._get_real_xy(bbox[0], bbox[1])
        
        font = self._get_font(self.text_size)
        
        draw = ImageDraw.Draw(self.current_img)
        draw.text((rx1, ry1), self.current_text_string, fill=self.text_color, font=font)
        
        self.push_history(self.current_img, self.mask_img)
        self.canvas.delete(self.text_element)
        self.text_element = None
        self.current_text_string = ""
        self.text_mode = False
        self._reset_buttons()
        self.canvas.config(cursor="arrow")
        self.update_canvas(self.current_img)
        self.lbl_status.config(text="✅ 文字已保存", foreground="green")

    def _get_font(self, size=40):
        try:
            if sys.platform == "win32":
                font_path = "C:/Windows/Fonts/msyh.ttc"
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
                font_path = "C:/Windows/Fonts/simsun.ttc"
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
                font_path = "C:/Windows/Fonts/arial.ttf"
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def _draw_smooth_line(self, points, draw_obj, ui_brush_size, fill):
        try:
            if len(points) < 2: return
            
            for i in range(len(points) - 1):
                cx1, cy1 = points[i]
                cx2, cy2 = points[i+1]
                rx1, ry1, scale = self._get_real_xy(cx1, cy1)
                rx2, ry2, _ = self._get_real_xy(cx2, cy2)
                real_width = max(1, int(ui_brush_size / scale))
                
                draw_obj.line([(rx1, ry1), (rx2, ry2)], fill=fill, width=real_width, joint="curve")
                r = real_width / 2.0
                draw_obj.ellipse([rx1-r, ry1-r, rx1+r, ry1+r], fill=fill)
        except Exception as e:
            print(f"绘制错误: {e}")

    def _get_real_xy(self, x, y):
        self.canvas.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        scale = min(cw/iw, ch/ih)
        pad_x = (cw - iw * scale) / 2
        pad_y = (ch - ih * scale) / 2
        return max(0, min(iw, (x - pad_x) / scale)), max(0, min(ih, (y - pad_y) / scale)), scale
        
    def push_history(self, img=None, mask=None):
        if img is None:
            img = self.current_img
        if mask is None:
            mask = self.mask_img
        self.history.append((img.copy(), mask.copy()))
        self.future.clear()
        if len(self.history) > 15:
            self.history.pop(0)

    def flip_image(self, direction="horizontal"):
        try:
            print(f"执行翻转: {direction}")
            self.push_history(self.current_img, self.mask_img)
            
            if direction == "horizontal":
                self.current_img = ImageOps.mirror(self.current_img)
                self.mask_img = ImageOps.mirror(self.mask_img)
                action = "水平翻转"
            else:
                self.current_img = ImageOps.flip(self.current_img)
                self.mask_img = ImageOps.flip(self.mask_img)
                action = "垂直翻转"
                
            self.base_img = self.current_img.copy()
            self.update_canvas(self.current_img, force=True)
            self.lbl_status.config(text=f"✅ 已{action}", foreground="green")
        except Exception as e:
            print(f"翻转错误: {e}")
            import traceback
            traceback.print_exc()
        
    def rotate_image(self, angle=90):
        try:
            print(f"执行旋转: {angle}度")
            self.push_history(self.current_img, self.mask_img)
            
            self.current_img = self.current_img.rotate(angle, expand=True)
            self.mask_img = self.mask_img.rotate(angle, expand=True)
            self.base_img = self.current_img.copy()
            self.original_full_img = self.current_img.copy()
            
            self.update_canvas(self.current_img, force=True)
            self.lbl_status.config(text=f"✅ 已旋转{angle}°", foreground="green")
        except Exception as e:
            print(f"旋转错误: {e}")
            import traceback
            traceback.print_exc()
        
    def undo(self):
        if len(self.history) > 1:
            current_state = self.history.pop()
            self.future.append(current_state)
            
            prev_img, prev_mask = self.history[-1]
            self.current_img = prev_img.copy()
            self.mask_img = prev_mask.copy()
            self.base_img = self.current_img.copy()
            self.update_canvas(self.current_img)
            
            self.lbl_status.config(text="✅ 已撤销", foreground="green")

    def redo(self):
        if self.future:
            next_state = self.future.pop()
            self.history.append(next_state)
            
            self.current_img = next_state[0].copy()
            self.mask_img = next_state[1].copy()
            self.base_img = self.current_img.copy()
            self.update_canvas(self.current_img)
            
            self.lbl_status.config(text="✅ 已重做", foreground="green")

    def update_canvas(self, pil_img, force=False, preserve_temp=False):
        self.canvas.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 800
        
        iw, ih = pil_img.size
        scale = min(cw/iw, ch/ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        
        current_size = (new_w, new_h)
        if not force and self.last_display_size == current_size and not self.text_element:
            return
        
        self.last_display_size = current_size
        
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.display_tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.display_tk_img, anchor=CENTER)
        
        if self.text_element:
            try:
                self.canvas.tag_raise(self.text_element)
            except:
                self.text_element = None

    def save_and_return(self):
        if self.text_element:
            self._commit_text_to_image()
        if self.callback_on_save:
            self.callback_on_save(self.current_img, self.mask_img)
        self.destroy()
