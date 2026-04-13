# image_editor.py
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
import ttkbootstrap as tb
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import datetime

class ImageEditorWindow(tb.Toplevel):
    def __init__(self, parent, image_path, callback_use_as_ref=None):
        super().__init__(master=parent, title="🎨 全能魔法图片编辑器", size=(1000, 750))
        self.image_path = image_path
        self.callback_use_as_ref = callback_use_as_ref
        
        # 加载图片并限制最大尺寸（防止撑爆屏幕）
        self.original_img = Image.open(image_path).convert("RGB")
        self.original_img.thumbnail((800, 600))
        self.img_width, self.img_height = self.original_img.size
        
        # 核心图像数据 (working_img是真实修改的图，mask_img是专门给AI用的黑白遮罩)
        self.working_img = self.original_img.copy()
        self.mask_img = Image.new("L", (self.img_width, self.img_height), 0)
        
        self.draw_img = ImageDraw.Draw(self.working_img)
        self.draw_mask = ImageDraw.Draw(self.mask_img)
        
        # 当前工具状态
        self.mode = "mask"  # 可选: "mask"(遮罩), "brush"(画笔), "crop"(裁剪), "text"(文字)
        self.brush_size = 20
        self.brush_color = "#ff0000" # 默认红色画笔
        
        self.last_x, self.last_y = None, None
        self.crop_start_x, self.crop_start_y = None, None
        self.crop_rect_id = None
        
        self.setup_ui()
        self.update_canvas()
        
    def setup_ui(self):
        # 🌟 修复版工具栏：分左右两排，绝对不会再被挤出屏幕！
        toolbar = tb.Frame(self, padding=5)
        toolbar.pack(fill=X, side=TOP)

        # 左侧工具组
        left_tools = tb.Frame(toolbar)
        left_tools.pack(side=LEFT)

        self.btn_mask = tb.Button(left_tools, text="✨ AI局部重绘(遮罩)", bootstyle="info", command=lambda: self.set_mode('mask'))
        self.btn_mask.pack(side=LEFT, padx=2)
        
        self.btn_draw = tb.Button(left_tools, text="🖌️ 普通画笔", bootstyle="outline-secondary", command=lambda: self.set_mode('draw'))
        self.btn_draw.pack(side=LEFT, padx=2)
        
        self.btn_crop = tb.Button(left_tools, text="✂️ 框选裁剪", bootstyle="outline-secondary", command=lambda: self.set_mode('crop'))
        self.btn_crop.pack(side=LEFT, padx=2)

        self.btn_text = tb.Button(left_tools, text="🔤 点击加字", bootstyle="outline-secondary", command=lambda: self.set_mode('text'))
        self.btn_text.pack(side=LEFT, padx=2)

        tb.Label(left_tools, text=" | 画笔粗细:").pack(side=LEFT, padx=5)
        self.scale_width = tb.Scale(left_tools, from_=2, to=50, value=15, length=80)
        self.scale_width.pack(side=LEFT, padx=2)

        self.btn_color = tb.Button(left_tools, text="🎨 颜色", bootstyle="outline-secondary", command=self.choose_color)
        self.btn_color.pack(side=LEFT, padx=5)

        # 右侧操作组 (强制靠右，永远不会被吃掉)
        right_tools = tb.Frame(toolbar)
        right_tools.pack(side=RIGHT)

        self.btn_undo = tb.Button(right_tools, text="🧹 撤销全部", bootstyle="danger-outline", command=self.undo_all)
        self.btn_undo.pack(side=LEFT, padx=5)

        self.btn_save = tb.Button(right_tools, text="✅ 保存并发送至主界面", bootstyle="success", command=self.save_and_return)
        self.btn_save.pack(side=LEFT, padx=2)

        # 下方的画布区域保持不变
        self.canvas_frame = tb.Frame(self)
        self.canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ================= 工具逻辑 =================
    def set_mode(self, new_mode):
        self.mode = new_mode
        # 更新按钮高亮状态
        self.btn_mask.configure(bootstyle="outline-info" if new_mode != "mask" else "info")
        self.btn_brush.configure(bootstyle="outline-primary" if new_mode != "brush" else "primary")
        self.btn_crop.configure(bootstyle="outline-primary" if new_mode != "crop" else "primary")
        self.btn_text.configure(bootstyle="outline-primary" if new_mode != "text" else "primary")
        
        # 更改鼠标光标
        if new_mode == "crop": self.canvas.config(cursor="crosshair")
        elif new_mode == "text": self.canvas.config(cursor="xterm")
        else: self.canvas.config(cursor="circle")

    def update_brush_size(self, val):
        self.brush_size = int(float(val))

    def choose_color(self):
        color_code = colorchooser.askcolor(title="选择颜色")[1]
        if color_code:
            self.brush_color = color_code

    def update_canvas(self):
        self.tk_img = ImageTk.PhotoImage(self.working_img)
        self.canvas.config(width=self.working_img.width, height=self.working_img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def reset_image(self):
        self.working_img = self.original_img.copy()
        self.mask_img = Image.new("L", (self.working_img.width, self.working_img.height), 0)
        self.draw_img = ImageDraw.Draw(self.working_img)
        self.draw_mask = ImageDraw.Draw(self.mask_img)
        self.update_canvas()

    # ================= 鼠标事件 =================
    def on_press(self, event):
        x, y = event.x, event.y
        if self.mode in ["mask", "brush"]:
            self.last_x, self.last_y = x, y
        elif self.mode == "crop":
            self.crop_start_x, self.crop_start_y = x, y
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = self.canvas.create_rectangle(x, y, x, y, outline="yellow", dash=(4, 4), width=2)
        elif self.mode == "text":
            text = simpledialog.askstring("添加文字", "请输入要添加的文字:")
            if text:
                # 尽量加载系统字体，如果没有则用默认（默认字体不支持中文）
                try: font = ImageFont.truetype("msyh.ttc", self.brush_size * 2) # 尝试加载微软雅黑
                except: font = ImageFont.load_default()
                
                self.draw_img.text((x, y), text, fill=self.brush_color, font=font)
                self.canvas.create_text(x, y, text=text, fill=self.brush_color, font=("微软雅黑", self.brush_size), anchor=tk.NW)

    def on_drag(self, event):
        x, y = event.x, event.y
        if self.mode == "mask" and self.last_x:
            # 画板上画粉色半透明（提示用户），实际遮罩画纯白
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=self.brush_size, fill="#ff00ff", capstyle=tk.ROUND, stipple="gray50")
            self.draw_mask.line([self.last_x, self.last_y, x, y], fill=255, width=self.brush_size, joint="curve")
            self.last_x, self.last_y = x, y
            
        elif self.mode == "brush" and self.last_x:
            # 真实画笔，直接画在原图上
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=self.brush_size, fill=self.brush_color, capstyle=tk.ROUND)
            self.draw_img.line([self.last_x, self.last_y, x, y], fill=self.brush_color, width=self.brush_size, joint="curve")
            self.last_x, self.last_y = x, y
            
        elif self.mode == "crop" and self.crop_start_x:
            self.canvas.coords(self.crop_rect_id, self.crop_start_x, self.crop_start_y, x, y)

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        
        if self.mode == "crop" and self.crop_start_x:
            x, y = event.x, event.y
            # 确保选取框有大小
            if abs(x - self.crop_start_x) > 10 and abs(y - self.crop_start_y) > 10:
                left = min(self.crop_start_x, x)
                top = min(self.crop_start_y, y)
                right = max(self.crop_start_x, x)
                bottom = max(self.crop_start_y, y)
                
                if messagebox.askyesno("裁剪确认", "确定要裁剪这部分区域吗？"):
                    # 裁剪原图和遮罩图
                    self.working_img = self.working_img.crop((left, top, right, bottom))
                    self.mask_img = self.mask_img.crop((left, top, right, bottom))
                    # 重新绑定画笔引擎
                    self.draw_img = ImageDraw.Draw(self.working_img)
                    self.draw_mask = ImageDraw.Draw(self.mask_img)
                    self.update_canvas()
            
            if self.crop_rect_id:
                self.canvas.delete(self.crop_rect_id)
                self.crop_rect_id = None
            self.crop_start_x, self.crop_start_y = None, None

    # ================= 保存与发送 =================
    def save_and_send(self):
        if not os.path.exists("photo"): os.makedirs("photo")
        timestamp = datetime.datetime.now().strftime('%H%M%S')
        
        # 保存被修改过的基础图片
        new_img_path = f"photo/edited_{timestamp}.png"
        self.working_img.save(new_img_path)
        
        # 检查是否画了 AI 局部重绘遮罩
        mask_path = None
        if self.mask_img.getbbox():  # 如果遮罩不是全黑的
            mask_path = f"photo/mask_{timestamp}.png"
            self.mask_img.save(mask_path)
            
        if self.callback_use_as_ref:
            # 传回主界面
            self.callback_use_as_ref(new_img_path, mask_path)
            
        self.destroy()