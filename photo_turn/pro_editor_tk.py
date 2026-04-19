# photo_turn/pro_editor_tk.py
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import threading
import time

class ProImageEditor(tb.Toplevel):
    def __init__(self, parent, image_path, callback_on_save=None):
        super().__init__(master=parent, title="✨ 专业级修图引擎 (Tkinter原生版)", size=(1200, 800))
        self.callback_on_save = callback_on_save
        self.image_path = image_path
        
        # 核心数据
        self.original_full_img = Image.open(image_path).convert("RGB")
        self.base_img = self.original_full_img.copy()
        self.base_img.thumbnail((1000, 800))
        self.current_img = self.base_img.copy()
        self.display_tk_img = None
        self.history = [self.base_img.copy()]
        
        # 状态
        self.crop_mode = False
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect = None
        
        # 文字悬浮状态
        self.text_mode = False
        self.current_text_string = ""
        self.text_x = 0
        self.text_y = 0
        self.is_dragging_text = False
        self.text_drag_offset_x = 0
        self.text_drag_offset_y = 0
        
        # 防抖参数
        self.last_adjust_time = 0
        self.adjust_pending = False
        
        self.setup_ui()
        self.update_canvas(self.current_img)

    def setup_ui(self):
        left_panel = tb.Frame(self, padding=10)
        left_panel.pack(side=LEFT, fill=Y)

        btn_save = tb.Button(left_panel, text="✅ 保存并传回主界面", bootstyle="success", command=self.save_and_return)
        btn_save.pack(fill=X, pady=(0, 15))

        # --- 1. 基础调整面板 ---
        frame_adj = tb.LabelFrame(left_panel, text="基础调整")
        frame_adj.pack(fill=X, pady=5, ipadx=5, ipady=5)

        self.sliders = {}
        for name, label, min_v, max_v in [
            ("brightness", "亮度", -100, 100),
            ("contrast", "对比度", -100, 100),
            ("color", "饱和度", -100, 100),
            ("sharpness", "锐化", 0, 100)
        ]:
            tb.Label(frame_adj, text=label).pack(anchor=W)
            scale = tb.Scale(frame_adj, from_=min_v, to=max_v, orient=HORIZONTAL, command=self.on_slider_change)
            scale.set(0)
            scale.pack(fill=X, pady=(0, 10))
            self.sliders[name] = scale

        tb.Button(frame_adj, text="🔄 重置滑块", bootstyle="outline", command=self.reset_sliders).pack(fill=X)

        # --- 2. 预设滤镜 ---
        frame_filter = tb.LabelFrame(left_panel, text="预设滤镜")
        frame_filter.pack(fill=X, pady=5, ipadx=5, ipady=5)
        
        self.cbo_filter = tb.Combobox(frame_filter, values=["原图", "黑白", "复古", "冷色调", "暖色调", "高斯模糊"])
        self.cbo_filter.current(0)
        self.cbo_filter.pack(fill=X, pady=5)
        tb.Button(frame_filter, text="🎨 应用滤镜", bootstyle="info", command=self.apply_filter).pack(fill=X)

        # --- 3. 变换与裁剪 ---
        frame_transform = tb.LabelFrame(left_panel, text="变换与裁剪")
        frame_transform.pack(fill=X, pady=5, ipadx=5, ipady=5)

        row1 = tb.Frame(frame_transform)
        row1.pack(fill=X, pady=2)
        tb.Button(row1, text="↺ 左转", command=lambda: self.apply_transform("rotate_left")).pack(side=LEFT, expand=True, padx=2)
        tb.Button(row1, text="↻ 右转", command=lambda: self.apply_transform("rotate_right")).pack(side=LEFT, expand=True, padx=2)

        self.btn_crop = tb.Button(frame_transform, text="✂️ 框选裁剪", bootstyle="warning-outline", command=self.toggle_crop)
        self.btn_crop.pack(fill=X, pady=5)

        # --- 4. 画笔与文字 ---
        frame_draw = tb.LabelFrame(left_panel, text="画笔与文字")
        frame_draw.pack(fill=X, pady=5, ipadx=5, ipady=5)

        row_draw = tb.Frame(frame_draw)
        row_draw.pack(fill=X, pady=2)
        
        self.btn_brush = tb.Button(row_draw, text="🖌️ 画笔", bootstyle="outline", command=self.toggle_brush)
        self.btn_brush.pack(side=LEFT, expand=True, padx=2)
        tb.Button(row_draw, text="🎨 颜色", bootstyle="info-outline", command=self.choose_color).pack(side=LEFT, expand=True, padx=2)
        
        self.btn_text = tb.Button(row_draw, text="🔤 加字", bootstyle="primary-outline", command=self.toggle_text)
        self.btn_text.pack(side=LEFT, expand=True, padx=2)

        tb.Label(frame_draw, text="画笔粗细 / 字体大小").pack(anchor=W, pady=(5, 0))
        self.scale_brush_size = tb.Scale(frame_draw, from_=2, to=100, orient=HORIZONTAL, command=self.on_brush_size_change)
        self.scale_brush_size.set(15) 
        self.scale_brush_size.pack(fill=X)

        # 初始化画笔状态变量
        self.draw_mode = False
        self.brush_color = "white"

        tb.Button(left_panel, text="↩️ 撤销上一步", bootstyle="secondary", command=self.undo).pack(fill=X, side=BOTTOM)

        # --- 画布 ---
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=RIGHT, fill=BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    # --- 图像处理核心 ---
    def process_image(self):
        b = self.sliders["brightness"].get() / 100.0
        c = self.sliders["contrast"].get() / 100.0
        col = self.sliders["color"].get() / 100.0
        s = self.sliders["sharpness"].get() / 50.0

        img = self.base_img.copy()
        if b != 0: img = ImageEnhance.Brightness(img).enhance(1.0 + b)
        if c != 0: img = ImageEnhance.Contrast(img).enhance(1.0 + c)
        if col != 0: img = ImageEnhance.Color(img).enhance(1.0 + col)
        if s != 0: img = ImageEnhance.Sharpness(img).enhance(1.0 + s)

        self.current_img = img
        self.update_canvas(self.current_img)
        self.adjust_pending = False

    def on_slider_change(self, event=None):
        self.last_adjust_time = time.time()
        if not self.adjust_pending:
            self.adjust_pending = True
            self.after(150, self.check_and_process)

    def check_and_process(self):
        if time.time() - self.last_adjust_time >= 0.1:
            threading.Thread(target=self.process_image).start()
        else:
            self.after(50, self.check_and_process)

    def apply_filter(self):
        fname = self.cbo_filter.get()
        img = self.current_img.copy()
        if fname == "黑白":
            img = ImageOps.grayscale(img).convert("RGB")
        elif fname == "复古":
            img = ImageEnhance.Color(img).enhance(0.5)
            r, g, b = img.split()
            r = r.point(lambda i: min(255, int(i * 1.1) + 20))
            g = g.point(lambda i: min(255, int(i * 1.05) + 10))
            b = b.point(lambda i: max(0, int(i * 0.85)))
            img = Image.merge("RGB", (r, g, b))
        elif fname == "冷色调":
            r, g, b = img.split()
            r = r.point(lambda i: max(0, int(i * 0.85)))
            b = b.point(lambda i: min(255, int(i * 1.2)))
            img = Image.merge("RGB", (r, g, b))
        elif fname == "暖色调":
            r, g, b = img.split()
            r = r.point(lambda i: min(255, int(i * 1.2)))
            b = b.point(lambda i: max(0, int(i * 0.85)))
            img = Image.merge("RGB", (r, g, b))
        elif fname == "高斯模糊":
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        self.push_history(img)

    def apply_transform(self, action):
        if action == "rotate_left":
            img = self.current_img.rotate(90, expand=True)
        else:
            img = self.current_img.rotate(-90, expand=True)
        self.push_history(img)

    def toggle_crop(self):
        if getattr(self, 'text_mode', False): self.toggle_text()
        if self.draw_mode: self.toggle_brush()
        
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.btn_crop.config(bootstyle="warning", text="取消裁剪")
            self.canvas.config(cursor="cross")
        else:
            self.btn_crop.config(bootstyle="warning-outline", text="✂️ 框选裁剪")
            self.canvas.config(cursor="arrow")
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)

    def toggle_brush(self):
        if getattr(self, 'text_mode', False): self.toggle_text()
        if self.crop_mode: self.toggle_crop()
        
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.btn_brush.config(bootstyle="solid")
            self.canvas.config(cursor="pencil")
        else:
            self.btn_brush.config(bootstyle="outline")
            self.canvas.config(cursor="arrow")

    def choose_color(self):
        color = colorchooser.askcolor(title="选择画笔/文字颜色", parent=self)[1]
        if color: 
            self.brush_color = color
            # 如果处于加字模式，实时更新悬浮文字的颜色
            if getattr(self, 'text_mode', False):
                self.render_floating_text()

    def on_brush_size_change(self, event=None):
        # 拖动滑块时，实时更新悬浮文字的大小
        if getattr(self, 'text_mode', False):
            self.render_floating_text()

    # --- 悬浮加字核心系统 ---
    def toggle_text(self):
        if self.crop_mode: self.toggle_crop()
        if self.draw_mode: self.toggle_brush()
        
        self.text_mode = not getattr(self, 'text_mode', False)
        if self.text_mode:
            text = simpledialog.askstring("添加文字", "请输入内容：", parent=self)
            if not text:
                self.text_mode = False
                return
            self.current_text_string = text
            self.btn_text.config(bootstyle="primary", text="📌 贴上文字")
            self.canvas.config(cursor="fleur") # 变成移动指针
            
            # 初始位置放在画面中央
            self.text_x = self.canvas.winfo_width() / 2
            self.text_y = self.canvas.winfo_height() / 2
            self.render_floating_text()
        else:
            # 退出加字模式，并将文字烘焙到底图
            self.btn_text.config(bootstyle="primary-outline", text="🔤 加字")
            self.canvas.config(cursor="arrow")
            self.bake_text()

    def render_floating_text(self):
        self.canvas.delete("floating_text")
        if self.text_mode and self.current_text_string:
            size = int(self.scale_brush_size.get())
            self.canvas.create_text(
                self.text_x, self.text_y,
                text=self.current_text_string,
                fill=self.brush_color,
                font=("Microsoft YaHei", size, "bold"),
                tags="floating_text",
                anchor=CENTER
            )

    def bake_text(self):
        if not getattr(self, 'current_text_string', ""): return
        
        img = self.current_img.copy()
        draw = ImageDraw.Draw(img)
        ui_size = int(self.scale_brush_size.get())
        
        rx, ry, scale = self._get_real_xy(self.text_x, self.text_y)
        # 缩放真实字体大小，保持和UI视觉一致
        real_size = max(1, int(ui_size / scale))
        
        try: font = ImageFont.truetype("msyh.ttc", real_size)
        except: font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), self.current_text_string, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        
        # 烘焙到真实PIL图像的对应位置
        draw.text((rx - tw/2, ry - th/2), self.current_text_string, font=font, fill=self.brush_color)
        
        self.canvas.delete("floating_text")
        self.current_text_string = ""
        self.push_history(img)

    # --- 辅助与历史记录 ---
    def push_history(self, img):
        self.base_img = img.copy()
        self.current_img = img.copy()
        self.history.append(img.copy())
        if len(self.history) > 10:
            self.history.pop(0)
        self.reset_sliders(silent=True)
        self.update_canvas(self.current_img)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            prev_img = self.history[-1]
            self.base_img = prev_img.copy()
            self.current_img = prev_img.copy()
            self.reset_sliders(silent=True)
            self.update_canvas(self.current_img)

    def reset_sliders(self, silent=False):
        for scale in self.sliders.values():
            scale.set(0)
        if not silent:
            self.current_img = self.base_img.copy()
            self.update_canvas(self.current_img)

    def update_canvas(self, pil_img):
        self.canvas.update()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1: cw, ch = 800, 800
        iw, ih = pil_img.size
        scale = min(cw/iw, ch/ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.display_tk_img, anchor=CENTER)
        # 如果加字模式在激活中，重绘画布后文字也要浮在最上面
        if getattr(self, 'text_mode', False):
            self.render_floating_text()

    def _get_real_xy(self, x, y):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        scale = min(cw/iw, ch/ih)
        pad_x = (cw - iw * scale) / 2
        pad_y = (ch - ih * scale) / 2
        real_x = max(0, min(iw, (x - pad_x) / scale))
        real_y = max(0, min(ih, (y - pad_y) / scale))
        return real_x, real_y, scale

    # --- 鼠标事件分配 ---
    def on_mouse_press(self, event):
        if getattr(self, 'text_mode', False):
            # 获取文字边界框，判定是否点中了文字
            bbox = self.canvas.bbox("floating_text")
            if bbox:
                # 放宽判定区域，方便拖拽
                if (bbox[0] - 20) <= event.x <= (bbox[2] + 20) and (bbox[1] - 20) <= event.y <= (bbox[3] + 20):
                    self.is_dragging_text = True
                    self.text_drag_offset_x = event.x - self.text_x
                    self.text_drag_offset_y = event.y - self.text_y
                    
        elif self.crop_mode:
            self.crop_start_x, self.crop_start_y = event.x, event.y
            if self.crop_rect: self.canvas.delete(self.crop_rect)
            self.crop_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="yellow", dash=(4,4), width=2)
            
        elif self.draw_mode:
            self.current_stroke = [(event.x, event.y)]
            self.last_x, self.last_y = event.x, event.y
            brush_size = int(self.scale_brush_size.get())
            r_ui = brush_size / 2.0
            self.canvas.create_oval(
                event.x - r_ui, event.y - r_ui,
                event.x + r_ui, event.y + r_ui,
                fill=self.brush_color, outline=self.brush_color,
                tags="temp_stroke" 
            )

    def on_mouse_drag(self, event):
        if getattr(self, 'text_mode', False) and getattr(self, 'is_dragging_text', False):
            # 拖动悬浮文字
            self.text_x = event.x - self.text_drag_offset_x
            self.text_y = event.y - self.text_drag_offset_y
            self.render_floating_text()
            
        elif self.crop_mode and self.crop_rect:
            self.canvas.coords(self.crop_rect, self.crop_start_x, self.crop_start_y, event.x, event.y)
            
        elif self.draw_mode and hasattr(self, 'current_stroke'):
            if abs(event.x - self.last_x) < 1 and abs(event.y - self.last_y) < 1:
                return
            self.current_stroke.append((event.x, event.y))
            brush_size = int(self.scale_brush_size.get())
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y, 
                fill=self.brush_color, width=brush_size, 
                capstyle="round", joinstyle="round", 
                tags="temp_stroke"
            )
            self.last_x, self.last_y = event.x, event.y

    def on_mouse_release(self, event):
        if getattr(self, 'text_mode', False):
            self.is_dragging_text = False
            
        elif self.crop_mode and self.crop_rect:
            x1, y1, x2, y2 = self.canvas.coords(self.crop_rect)
            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20: 
                rx1, ry1, _ = self._get_real_xy(x1, y1)
                rx2, ry2, _ = self._get_real_xy(x2, y2)
                cropped = self.current_img.crop((min(rx1, rx2), min(ry1, ry2), max(rx1, rx2), max(ry1, ry2)))
                self.push_history(cropped)
            self.toggle_crop()
            
        elif self.draw_mode and hasattr(self, 'current_stroke'):
            if not self.current_stroke: return
            draw = ImageDraw.Draw(self.current_img)
            brush_size = int(self.scale_brush_size.get())
            
            if len(self.current_stroke) == 1:
                cx, cy = self.current_stroke[0]
                rx, ry, scale = self._get_real_xy(cx, cy)
                real_width = max(1, int(brush_size / scale))
                r = real_width / 2.0
                draw.ellipse([rx-r, ry-r, rx+r, ry+r], fill=self.brush_color)
            else:
                for i in range(len(self.current_stroke) - 1):
                    cx1, cy1 = self.current_stroke[i]
                    cx2, cy2 = self.current_stroke[i+1]
                    rx1, ry1, scale = self._get_real_xy(cx1, cy1)
                    rx2, ry2, _ = self._get_real_xy(cx2, cy2)
                    real_width = max(1, int(brush_size / scale))
                    r = real_width / 2.0
                    draw.line([(rx1, ry1), (rx2, ry2)], fill=self.brush_color, width=real_width)
                    draw.ellipse([rx1-r, ry1-r, rx1+r, ry1+r], fill=self.brush_color)
                last_cx, last_cy = self.current_stroke[-1]
                rx_last, ry_last, _ = self._get_real_xy(last_cx, last_cy)
                draw.ellipse([rx_last-r, ry_last-r, rx_last+r, ry_last+r], fill=self.brush_color)

            self.canvas.delete("temp_stroke")
            self.push_history(self.current_img)
            del self.current_stroke
            self.last_x = self.last_y = None

    def save_and_return(self):
        # 保存前，如果文字还悬浮着没有贴上，自动帮用户贴上
        if getattr(self, 'text_mode', False):
            self.bake_text()
            
        if self.callback_on_save:
            self.callback_on_save(self.current_img)
        self.destroy()