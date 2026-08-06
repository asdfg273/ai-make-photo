# photo_turn/mixin_tools.py

import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
from PyQt6.QtWidgets import QColorDialog, QInputDialog
from PyQt6.QtCore    import Qt


class ToolsEventsMixin:

    # ----------------------------------------------------------
    #  模式切换
    # ----------------------------------------------------------
    def toggle_brush(self):
        self.is_eraser     = False
        self.is_mask_brush = False
        self.draw_mode     = not self.draw_mode
        self.text_mode     = False
        self.crop_mode     = False
        color = "#89b4fa" if self.draw_mode else "#cdd6f4"
        state = "开启 ✅" if self.draw_mode else "关闭"
        self.lbl_status.setText(f"🖌 画笔模式 {state}")
        self.lbl_status.setStyleSheet(f"color:{color}; font-size:13px;")

    def toggle_eraser(self):
        prev              = self.is_eraser
        self.is_eraser    = not prev
        self.draw_mode    = self.is_eraser
        self.is_mask_brush = False
        self.text_mode    = False
        self.crop_mode    = False
        color = "#f38ba8" if self.is_eraser else "#cdd6f4"
        state = "开启 ✅" if self.is_eraser else "关闭"
        self.lbl_status.setText(f"🧽 橡皮模式 {state}")
        self.lbl_status.setStyleSheet(f"color:{color}; font-size:13px;")

    def toggle_mask_brush(self):
        prev               = self.is_mask_brush
        self.is_mask_brush = not prev
        self.draw_mode     = self.is_mask_brush
        self.is_eraser     = False
        self.text_mode     = False
        self.crop_mode     = False
        color = "#f38ba8" if self.is_mask_brush else "#cdd6f4"
        state = "开启 ✅" if self.is_mask_brush else "关闭"
        self.lbl_status.setText(f"🔴 遮罩画笔 {state}")
        self.lbl_status.setStyleSheet(f"color:{color}; font-size:13px;")

    def toggle_crop(self):
        self.crop_mode = not self.crop_mode
        self.draw_mode = False
        self.text_mode = False
        state = "请拖拽选择裁剪区域 ✅" if self.crop_mode else "已取消"
        self.lbl_status.setText(f"✂ {state}")
        self.lbl_status.setStyleSheet("color:#89b4fa; font-size:13px;")

    def toggle_text(self):
        if not self.text_mode:
            text, ok = QInputDialog.getText(self, "输入文字", "请输入要添加的文字:")
            if ok and text.strip():
                self.current_text_string = text.strip()
                self.text_mode  = True
                self.draw_mode  = False
                self.crop_mode  = False
                self.text_element = None
                self.lbl_status.setText("📝 点击画布放置文字")
                self.lbl_status.setStyleSheet("color:#cba6f7; font-size:13px;")
        else:
            self.text_mode = False
            self.lbl_status.setText("📝 文字模式已关闭")
            self.lbl_status.setStyleSheet("color:#cdd6f4; font-size:13px;")

    def _cancel_any_mode(self):
        self.draw_mode     = False
        self.is_mask_brush = False
        self.is_eraser     = False
        self.crop_mode     = False
        self.text_mode     = False
        if self.crop_overlay:
            self.crop_overlay.hide()
            self.crop_overlay = None
        self.lbl_status.setText("⎋ 已退出所有模式")
        self.lbl_status.setStyleSheet("color:#cdd6f4; font-size:13px;")

    # ----------------------------------------------------------
    #  颜色选择
    # ----------------------------------------------------------
    def pick_color(self):
        color = QColorDialog.getColor(parent=self, title="选择画笔颜色")
        if color.isValid():
            self.brush_color = color.name()
            self.color_preview.setStyleSheet(
                f"background:{self.brush_color}; border-radius:4px;")

    def pick_text_color(self):
        color = QColorDialog.getColor(parent=self, title="选择文字颜色")
        if color.isValid():
            self.text_color = color.name()
            self.text_color_preview.setStyleSheet(
                f"background:{self.text_color}; border-radius:4px;")

    # ----------------------------------------------------------
    #  鼠标事件
    # ----------------------------------------------------------
    def on_mouse_press(self, x: int, y: int):
        if self.draw_mode:
            self.push_history()
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
        elif self.crop_mode and hasattr(self, "_crop_start"):
            self._crop_preview(self._crop_start, (x, y))

    def on_mouse_release(self, x: int, y: int):
        if self.draw_mode:
            self._last_draw_pos = None
        elif self.crop_mode and hasattr(self, "_crop_start"):
            x0, y0 = self._crop_start
            self._do_crop(x0, y0, x, y)
            self.crop_mode   = False
            self._crop_start = None

    def on_mouse_right_click(self, x: int, y: int):
        if self.text_element:
            self._commit_text_to_image()

    # ----------------------------------------------------------
    #  绘图核心
    # ----------------------------------------------------------
    def _make_circle_mask(self, size: tuple, cx: int, cy: int, r: int) -> Image.Image:
        """生成一个圆形遮罩，用于橡皮擦和画笔"""
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        return mask

    def _draw_at(self, x: int, y: int):
        size = max(1, self.sld_brush_size.value())
        r    = size // 2

        if self.is_eraser:
            # ✅ 真正的橡皮擦：从 original_img 还原像素
            mask = self._make_circle_mask(
                self.current_img.size, x, y, r)
            src  = self.original_img.convert(self.current_img.mode)
            self.current_img.paste(src, (0, 0), mask)
            self.update_canvas(self.current_img, force=True)

        elif self.is_mask_brush:
            draw = ImageDraw.Draw(self.mask_img)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            self._overlay_mask()

        else:
            color = self._hex_to_rgb(self.brush_color)
            draw  = ImageDraw.Draw(self.current_img)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            self.update_canvas(self.current_img, force=True)

    def _draw_line(self, p0: tuple, p1: tuple):
        size = max(1, self.sld_brush_size.value())
        r    = size // 2
        x0, y0 = p0
        x1, y1 = p1

        if self.is_eraser:
            # ✅ 橡皮擦连线：沿线段每隔 max(1, r//2) 像素还原一个圆
            import math
            dist    = math.hypot(x1 - x0, y1 - y0)
            steps   = max(1, int(dist / max(1, r // 2)))
            src     = self.original_img.convert(self.current_img.mode)
            for i in range(steps + 1):
                t  = i / steps
                px = int(x0 + (x1 - x0) * t)
                py = int(y0 + (y1 - y0) * t)
                mask = self._make_circle_mask(
                    self.current_img.size, px, py, r)
                self.current_img.paste(src, (0, 0), mask)
            self.update_canvas(self.current_img, force=True)

        elif self.is_mask_brush:
            draw = ImageDraw.Draw(self.mask_img)
            draw.line([x0, y0, x1, y1], fill=255, width=size)
            for px, py in [p0, p1]:
                draw.ellipse([px-r, py-r, px+r, py+r], fill=255)
            self._overlay_mask()

        else:
            color = self._hex_to_rgb(self.brush_color)
            draw  = ImageDraw.Draw(self.current_img)
            draw.line([x0, y0, x1, y1], fill=color, width=size)
            for px, py in [p0, p1]:
                draw.ellipse([px-r, py-r, px+r, py+r], fill=color)
            self.update_canvas(self.current_img, force=True)

    def _overlay_mask(self):
        """遮罩半透明叠加（红色预览）"""
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
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑（支持中文）
            "C:/Windows/Fonts/simhei.ttf",
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

    def _render_text_preview(self, x: int, y: int):
        preview = self.current_img.copy()
        draw    = ImageDraw.Draw(preview)
        font    = self._get_pil_font(self.spin_text_size.value())
        draw.text((x, y), self.current_text_string,
                  fill=self.text_color, font=font)
        self.update_canvas(preview, force=True)

    def _commit_text_to_image(self):
        if not self.text_element:
            return
        x, y = self.text_element
        self.push_history()
        draw = ImageDraw.Draw(self.current_img)
        font = self._get_pil_font(self.spin_text_size.value())
        draw.text((x, y), self.current_text_string,
                  fill=self.text_color, font=font)
        self.text_element = None
        self.text_mode    = False
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText("✅ 文字已写入图层")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

    # ----------------------------------------------------------
    #  裁剪工具
    # ----------------------------------------------------------
    def _crop_preview(self, start, end):
        pass

    def _do_crop(self, x0, y0, x1, y1):
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return
        self.push_history()
        box = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        self.current_img = self.current_img.crop(box)
        self.base_img    = self.current_img.copy()
        self.mask_img    = self.mask_img.crop(box)
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText(f"✅ 裁剪完成 → {self.current_img.size}")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

    # ----------------------------------------------------------
    #  变换
    # ----------------------------------------------------------
    def flip_image(self, direction="horizontal"):
        self.push_history()
        method = (Image.Transpose.FLIP_LEFT_RIGHT
                  if direction == "horizontal"
                  else Image.Transpose.FLIP_TOP_BOTTOM)
        self.current_img = self.current_img.transpose(method)
        self.base_img    = self.current_img.copy()
        self.update_canvas(self.current_img, force=True)
        name = "水平" if direction == "horizontal" else "垂直"
        self.lbl_status.setText(f"✅ {name}翻转完成")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

    def rotate_image(self, angle: int):
        self.push_history()
        self.current_img = self.current_img.rotate(
            angle, expand=True, resample=Image.Resampling.BICUBIC)
        self.base_img = self.current_img.copy()
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText(f"✅ 已旋转 {angle}°")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

    # ----------------------------------------------------------
    #  工具函数
    # ----------------------------------------------------------
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return (255, 0, 0)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))