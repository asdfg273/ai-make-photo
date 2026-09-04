# photo_turn/mixin_tools.py

import os
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps
from PyQt6.QtWidgets import QColorDialog, QInputDialog
from PyQt6.QtCore    import Qt
import logging

logger = logging.getLogger(__name__)


class ToolsEventsMixin:

    # ----------------------------------------------------------
    #  模式切换
    # ----------------------------------------------------------
    def _exit_modes(self):
        """关闭所有互斥模式"""
        self.draw_mode     = False
        self.is_eraser     = False
        self.is_mask_brush = False
        self.crop_mode     = False
        self.text_mode     = False
        self.pick_mode     = False

    def _set_status(self, text, color="#cdd6f4"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color:{color}; font-size:13px;")

    def _refresh_brush_cursor(self):
        """根据当前工具状态刷新画布上的画笔光标圈"""
        show = self.draw_mode or self.is_mask_brush
        if not show:
            self.canvas.set_brush_cursor(0, "#ffffff")
            return
        if self.is_mask_brush:
            color = "#ff3232"
        elif self.is_eraser:
            color = "#ffffff"
        else:
            color = self.brush_color
        self.canvas.set_brush_cursor(
            max(1, self.sld_brush_size.value()), color)

    def toggle_brush(self):
        was = self.draw_mode and not self.is_eraser and not self.is_mask_brush
        self._exit_modes()
        self.draw_mode = not was
        color = "#89b4fa" if self.draw_mode else "#cdd6f4"
        state = "开启 ✅" if self.draw_mode else "关闭"
        self._set_status(f"🖌 画笔模式 {state}（Shift+单击画直线）", color)
        self._refresh_brush_cursor()

    def toggle_eraser(self):
        was = self.is_eraser
        self._exit_modes()
        self.is_eraser = not was
        self.draw_mode = self.is_eraser
        color = "#f38ba8" if self.is_eraser else "#cdd6f4"
        state = "开启 ✅" if self.is_eraser else "关闭"
        self._set_status(f"🧽 橡皮模式 {state}", color)
        self._refresh_brush_cursor()

    def toggle_mask_brush(self):
        was = self.is_mask_brush
        self._exit_modes()
        self.is_mask_brush = not was
        self.draw_mode     = self.is_mask_brush
        color = "#f38ba8" if self.is_mask_brush else "#cdd6f4"
        state = "开启 ✅" if self.is_mask_brush else "关闭"
        self._set_status(f"🔴 遮罩画笔 {state}", color)
        self._refresh_brush_cursor()

    def toggle_crop(self):
        was = self.crop_mode
        self._exit_modes()
        self.crop_mode = not was
        state = "请拖拽选择裁剪区域 ✅" if self.crop_mode else "已取消"
        self._set_status(f"✂ {state}", "#89b4fa")
        self._refresh_brush_cursor()

    def toggle_text(self):
        if not self.text_mode:
            text, ok = QInputDialog.getText(self, "输入文字", "请输入要添加的文字:")
            if ok and text.strip():
                self._exit_modes()
                self.current_text_string = text.strip()
                self.text_mode    = True
                self.text_element = None
                self._set_status("📝 点击画布放置文字", "#cba6f7")
        else:
            self.text_mode = False
            self._set_status("📝 文字模式已关闭")
        self._refresh_brush_cursor()

    def toggle_eyedropper(self):
        """吸管:单击画布取样画笔颜色后自动退出"""
        self._exit_modes()
        self.pick_mode = True
        self._set_status("💧 点击画布任意位置取色", "#94e2d5")
        self._refresh_brush_cursor()

    def _cancel_any_mode(self):
        self._exit_modes()
        if hasattr(self, "_cancel_filter_preview"):
            self._cancel_filter_preview()
        if self.crop_overlay:
            self.crop_overlay.hide()
            self.crop_overlay = None
        # 退出裁剪时清掉预览框
        self.update_canvas(self.current_img, force=True)
        self._set_status("⎋ 已退出所有模式")
        self._refresh_brush_cursor()

    # ----------------------------------------------------------
    #  颜色选择
    # ----------------------------------------------------------
    def pick_color(self):
        color = QColorDialog.getColor(parent=self, title="选择画笔颜色")
        if color.isValid():
            self.brush_color = color.name()
            self.color_preview.setStyleSheet(
                f"background:{self.brush_color}; border-radius:4px;")
            self._refresh_brush_cursor()

    def pick_text_color(self):
        color = QColorDialog.getColor(parent=self, title="选择文字颜色")
        if color.isValid():
            self.text_color = color.name()
            self.text_color_preview.setStyleSheet(
                f"background:{self.text_color}; border-radius:4px;")

    def _sample_color(self, x: int, y: int):
        """吸管取样"""
        w, h = self.current_img.size
        if 0 <= x < w and 0 <= y < h:
            px = self.current_img.convert("RGB").getpixel((x, y))
            self.brush_color = "#%02x%02x%02x" % px
            self.color_preview.setStyleSheet(
                f"background:{self.brush_color}; border-radius:4px;")
            self._set_status(f"💧 已取色 {self.brush_color}", "#94e2d5")
        else:
            self._set_status("💧 取色位置超出图像范围", "#f9e2af")
        self.pick_mode = False

    # ----------------------------------------------------------
    #  遮罩操作
    # ----------------------------------------------------------
    def toggle_mask_preview(self):
        """遮罩红色叠加预览开关"""
        self._mask_preview_on = not getattr(self, "_mask_preview_on", True)
        if self.is_mask_brush:
            self._overlay_mask()
        else:
            self.update_canvas(self.current_img, force=True)
        state = "显示" if self._mask_preview_on else "隐藏"
        self._set_status(f"🙈 遮罩预览已{state}", "#89dceb")

    def clear_mask(self):
        if self.mask_img.getextrema() == (0, 0):
            self._set_status("🧹 遮罩已经是空的", "#f9e2af")
            return
        self.push_history()
        self.mask_img = Image.new("L", self.current_img.size, 0)
        self.update_canvas(self.current_img, force=True)
        self._set_status("🧹 遮罩已清除", "#a6e3a1")

    def invert_mask(self):
        if self.mask_img.getextrema() == (0, 0):
            self._set_status("🔁 遮罩为空,无需反转", "#f9e2af")
            return
        self.push_history()
        self.mask_img = ImageOps.invert(self.mask_img)
        self._overlay_mask()
        self._set_status("🔁 遮罩已反转", "#a6e3a1")

    def feather_mask(self, radius=None):
        """遮罩羽化(柔化边缘,重绘过渡更自然)"""
        from PIL import ImageFilter
        if self.mask_img.getextrema() == (0, 0):
            self._set_status("🌫 遮罩为空,请先涂抹", "#f9e2af")
            return
        if radius is None:
            sld = getattr(self, "sld_mask_feather", None)
            radius = sld.value() if sld is not None else 4
        if radius <= 0:
            return
        self.push_history()
        self.mask_img = self.mask_img.filter(
            ImageFilter.GaussianBlur(radius))
        self._overlay_mask()
        self._set_status(f"🌫 遮罩已羽化 {radius}px", "#a6e3a1")

    def grow_mask(self, px: int = 4):
        """遮罩向外扩张"""
        from PIL import ImageFilter
        if self.mask_img.getextrema() == (0, 0):
            self._set_status("➕ 遮罩为空,请先涂抹", "#f9e2af")
            return
        self.push_history()
        self.mask_img = self.mask_img.filter(
            ImageFilter.MaxFilter(2 * px + 1))
        self._overlay_mask()
        self._set_status(f"➕ 遮罩已扩张 {px}px", "#a6e3a1")

    def shrink_mask(self, px: int = 4):
        """遮罩向内收缩"""
        from PIL import ImageFilter
        if self.mask_img.getextrema() == (0, 0):
            self._set_status("➖ 遮罩为空,请先涂抹", "#f9e2af")
            return
        self.push_history()
        self.mask_img = self.mask_img.filter(
            ImageFilter.MinFilter(2 * px + 1))
        self._overlay_mask()
        self._set_status(f"➖ 遮罩已收缩 {px}px", "#a6e3a1")

    # ----------------------------------------------------------
    #  鼠标事件
    # ----------------------------------------------------------
    def on_mouse_press(self, x: int, y: int):
        if getattr(self, "pick_mode", False):
            self._sample_color(x, y)
        elif self.draw_mode:
            self._filter_anchor = None   # 手绘改动后,滤镜锚点失效
            self.push_history()
            # Shift+单击:从上一笔末端拉直线(PS 同款)
            from PyQt6.QtWidgets import QApplication
            shift = bool(QApplication.keyboardModifiers()
                         & Qt.KeyboardModifier.ShiftModifier)
            prev  = getattr(self, "_stroke_end", None)
            if shift and prev is not None:
                self._draw_line(prev, (x, y))
            else:
                self._draw_at(x, y)
            self._last_draw_pos = (x, y)
        elif self.text_mode:
            self.text_element = (x, y)
            self._render_text_preview(x, y)
        elif self.crop_mode:
            self._crop_start = (x, y)

    def on_mouse_drag(self, x: int, y: int):
        if self.draw_mode and self._last_draw_pos:
            self._draw_line(self._last_draw_pos, (x, y))
            self._last_draw_pos = (x, y)
        elif self.text_mode and self.text_element:
            self.text_element = (x, y)
            self._render_text_preview(x, y)
        elif self.crop_mode and hasattr(self, "_crop_start") and self._crop_start:
            self._crop_preview(self._crop_start, (x, y))

    def on_mouse_release(self, x: int, y: int):
        if self.draw_mode:
            self._last_draw_pos = None
            self._stroke_end    = (x, y)   # 供 Shift+单击直线使用
        elif self.crop_mode and getattr(self, "_crop_start", None):
            x0, y0 = self._crop_start
            self._do_crop(x0, y0, x, y)
            self.crop_mode   = False
            self._crop_start = None

    def on_mouse_right_click(self, x: int, y: int):
        if self.text_element:
            self._commit_text_to_image()

    # ----------------------------------------------------------
    #  绘图核心 — 统一笔画管线(画笔/橡皮/遮罩共用,支持柔边)
    # ----------------------------------------------------------
    def _brush_feather(self, r: int) -> float:
        """由硬度滑块换算羽化半径:100=硬边,0=整笔刷半径羽化"""
        sld = getattr(self, "sld_brush_hardness", None)
        hv = sld.value() if sld is not None else 100
        return r * (100 - max(0, min(100, hv))) / 100.0

    def _build_stroke(self, points, size, feather: float) -> Image.Image:
        """硬笔画 → 可选羽化的笔画遮罩(L)"""
        from PIL import ImageFilter
        r = size // 2
        stroke = Image.new("L", self.current_img.size, 0)
        d = ImageDraw.Draw(stroke)
        if len(points) == 1:
            x, y = points[0]
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            (x0, y0), (x1, y1) = points
            d.line([x0, y0, x1, y1], fill=255, width=size)
            for px, py in points:
                d.ellipse([px - r, py - r, px + r, py + r], fill=255)
        if feather > 0.5:
            stroke = stroke.filter(ImageFilter.GaussianBlur(feather))
        return stroke

    def _brush_opacity_alpha(self) -> int:
        """0~255 的画笔不透明度"""
        sld = getattr(self, "sld_brush_opacity", None)
        if sld is None:
            return 255
        return int(255 * max(0, min(100, sld.value())) / 100)

    def _apply_stroke(self, points, size):
        """统一笔画应用入口"""
        r      = size // 2
        stroke = self._build_stroke(points, size, self._brush_feather(r))

        if self.is_eraser:
            # 从原图还原
            src = self.original_img.convert(self.current_img.mode)
            self.current_img.paste(src, (0, 0), stroke)
            self.update_canvas(self.current_img, force=True)

        elif self.is_mask_brush:
            # 软边并集叠加到遮罩
            self.mask_img = ImageChops.lighter(self.mask_img, stroke)
            self._overlay_mask()

        else:
            # 普通画笔:颜色经笔画遮罩(×不透明度)贴上
            alpha = self._brush_opacity_alpha()
            if alpha < 255:
                stroke = stroke.point(lambda v: v * alpha // 255)
            color_img = Image.new("RGB", self.current_img.size,
                                  self._hex_to_rgb(self.brush_color))
            base = self.current_img.convert("RGB")
            base.paste(color_img, (0, 0), stroke)
            self.current_img = base
            self.update_canvas(self.current_img, force=True)

    def _draw_at(self, x: int, y: int):
        size = max(1, self.sld_brush_size.value())
        self._apply_stroke([(x, y)], size)

    def _draw_line(self, p0: tuple, p1: tuple):
        size = max(1, self.sld_brush_size.value())
        self._apply_stroke([p0, p1], size)

    def _overlay_mask(self):
        """遮罩半透明叠加（红色预览）,可用开关临时隐藏"""
        if not getattr(self, "_mask_preview_on", True):
            self.update_canvas(self.current_img, force=True)
            return
        import numpy as np
        base    = self.current_img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        arr     = np.array(overlay)
        marr    = np.array(self.mask_img)
        arr[marr > 0] = [255, 50, 50, 120]
        overlay = Image.fromarray(arr, "RGBA")
        preview = Image.alpha_composite(base, overlay).convert("RGB")
        self.update_canvas(preview, force=True)

    # ----------------------------------------------------------
    #  文字工具
    # ----------------------------------------------------------
    def _get_pil_font(self, size: int) -> ImageFont.FreeTypeFont:
        size = max(10, size)
        for fp in [
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑（支持中文,优先）
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def _text_stroke_kwargs(self) -> dict:
        """文字描边参数:按文字亮度自动选黑/白描边色"""
        sw = self.spin_text_stroke.value() \
            if hasattr(self, "spin_text_stroke") else 0
        if sw <= 0:
            return {}
        r, g, b = self._hex_to_rgb(self.text_color)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return {"stroke_width": sw,
                "stroke_fill": "#000000" if lum > 140 else "#ffffff"}

    def _render_text_preview(self, x: int, y: int):
        preview = self.current_img.copy()
        draw    = ImageDraw.Draw(preview)
        font    = self._get_pil_font(self.spin_text_size.value())
        draw.text((x, y), self.current_text_string,
                  fill=self.text_color, font=font,
                  **self._text_stroke_kwargs())
        self.update_canvas(preview, force=True)

    def _commit_text_to_image(self):
        if not self.text_element:
            return
        x, y = self.text_element
        self._filter_anchor = None   # 写入文字后,滤镜锚点失效
        self.push_history()
        draw = ImageDraw.Draw(self.current_img)
        font = self._get_pil_font(self.spin_text_size.value())
        draw.text((x, y), self.current_text_string,
                  fill=self.text_color, font=font,
                  **self._text_stroke_kwargs())
        self.text_element = None
        self.text_mode    = False
        self.update_canvas(self.current_img, force=True)
        self._set_status("✅ 文字已写入图层", "#a6e3a1")

    # ----------------------------------------------------------
    #  裁剪工具
    # ----------------------------------------------------------
    def _apply_crop_aspect(self, x0, y0, x1, y1):
        """按选中的宽高比约束裁剪终点(以宽为主导,保持拖拽方向)"""
        combo = getattr(self, "combo_crop_aspect", None)
        a = combo.currentText() if combo else "自由"
        if ":" not in a:
            return x1, y1
        ar_w, ar_h = map(int, a.split(":"))
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            return x1, y1
        tw = abs(dx)
        th = round(tw * ar_h / ar_w)
        # (x0, y0) 是锚点，向右/下时新矩形右下角 = 锚点 + (tw, th)
        # 向左/上时新矩形左上角 = 锚点 - (tw, th)
        nx = x0 + tw if dx >= 0 else x0 - tw
        ny = y0 + th if dy >= 0 else y0 - th
        return int(nx), int(ny)

    def _crop_preview(self, start, end):
        """拖拽时实时显示选区框"""
        x0, y0 = start
        x1, y1 = self._apply_crop_aspect(x0, y0, *end)
        preview = self.current_img.copy()
        d = ImageDraw.Draw(preview)
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        d.rectangle(box, outline=(255, 220, 80), width=3)
        self.update_canvas(preview, force=True)

    def _do_crop(self, x0, y0, x1, y1):
        x1, y1 = self._apply_crop_aspect(x0, y0, x1, y1)
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            self.update_canvas(self.current_img, force=True)
            self._set_status("✂ 选区过小,已取消裁剪", "#f9e2af")
            return
        
        self._filter_anchor = None
        self.push_history()
        
        # clamp 到图像边界，防止裁出黑边
        w, h = self.current_img.size
        left   = max(0, min(x0, x1))
        top    = max(0, min(y0, y1))
        right  = min(w, max(x0, x1))
        bottom = min(h, max(y0, y1))
        box = (left, top, right, bottom)
        
        self.current_img   = self.current_img.crop(box)
        self.base_img      = self.current_img.copy()
        self.original_img  = self.original_img.crop(box)
        self.mask_img      = self.mask_img.crop(box)
        self.update_canvas(self.current_img, force=True)
        self._set_status(f"✅ 裁剪完成 → {self.current_img.size}", "#a6e3a1")

    # ----------------------------------------------------------
    #  变换
    # ----------------------------------------------------------
    def flip_image(self, direction="horizontal"):
        self._filter_anchor = None
        self.push_history()
        method = (Image.Transpose.FLIP_LEFT_RIGHT
                  if direction == "horizontal"
                  else Image.Transpose.FLIP_TOP_BOTTOM)
        self.current_img  = self.current_img.transpose(method)
        self.base_img     = self.current_img.copy()
        self.original_img = self.original_img.transpose(method)
        self.mask_img     = self.mask_img.transpose(method)
        self.update_canvas(self.current_img, force=True)
        name = "水平" if direction == "horizontal" else "垂直"
        self._set_status(f"✅ {name}翻转完成", "#a6e3a1")

    def rotate_image(self, angle: int):
        self._filter_anchor = None
        self.push_history()
        self.current_img = self.current_img.rotate(
            angle, expand=True, resample=Image.Resampling.BICUBIC)
        self.base_img     = self.current_img.copy()
        self.original_img = self.original_img.rotate(
            angle, expand=True, resample=Image.Resampling.BICUBIC)
        self.mask_img     = self.mask_img.rotate(
            angle, expand=True, resample=Image.Resampling.NEAREST)
        self.update_canvas(self.current_img, force=True)
        self._set_status(f"✅ 已旋转 {angle}°", "#a6e3a1")

    def rotate_image_any(self):
        """任意角度旋转(取角度输入框的值)"""
        angle = self.spin_rotate_angle.value() \
            if hasattr(self, "spin_rotate_angle") else 0
        if angle % 360 == 0:
            self._set_status("↻ 角度为 0,无需旋转", "#f9e2af")
            return
        self.rotate_image(angle)

    def resize_to_long_edge(self, target: int):
        """等比缩放到指定长边(出图前对齐 SD 常用尺寸)"""
        w, h = self.current_img.size
        if max(w, h) == target:
            self._set_status(f"📐 长边已是 {target}px", "#f9e2af")
            return
        scale  = target / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        self._filter_anchor = None
        self.push_history()
        self.current_img  = self.current_img.resize(
            (nw, nh), Image.Resampling.LANCZOS)
        self.base_img     = self.current_img.copy()
        self.original_img = self.original_img.resize(
            (nw, nh), Image.Resampling.LANCZOS)
        self.mask_img     = self.mask_img.resize(
            (nw, nh), Image.Resampling.NEAREST)
        self.update_canvas(self.current_img, force=True)
        self._set_status(f"📐 已缩放 → {nw}×{nh}", "#a6e3a1")

    def resize_dialog(self):
        """等比缩放对话框"""
        choices = ["512", "768", "1024", "1536", "2048"]
        cur = str(max(self.current_img.size))
        idx = choices.index(cur) if cur in choices else 2
        val, ok = QInputDialog.getItem(
            self, "调整尺寸", "目标长边 (等比缩放):", choices, idx, False)
        if ok and val:
            self.resize_to_long_edge(int(val))

    # ----------------------------------------------------------
    #  工具函数
    # ----------------------------------------------------------
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return (255, 0, 0)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
