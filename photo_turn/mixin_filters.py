# photo_turn/mixin_filters.py

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from PyQt6.QtCore import QTimer


class FiltersMixin:

    def on_adjust_change(self, key=None, value=None):
        """调色滑块防抖 — ✅ 不触发滤镜"""
        if self.adjust_timer and self.adjust_timer.isActive():
            self.adjust_timer.stop()
        self.adjust_timer = QTimer(self)
        self.adjust_timer.setSingleShot(True)
        # ✅ 只连 apply_adjustments，不连 apply_selected_filter
        self.adjust_timer.timeout.connect(self.apply_adjustments)
        self.adjust_timer.start(150)

    def apply_adjustments(self):
        try:
            if self.base_img is None:
                return
            img       = self.base_img.copy()
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[-1]
                img   = img.convert("RGB")

            brightness  = self.adjust_vars["brightness"].value()
            contrast    = self.adjust_vars["contrast"].value()
            saturation  = self.adjust_vars["saturation"].value()
            sharpness   = self.adjust_vars["sharpness"].value()
            temperature = self.adjust_vars["temperature"].value()

            if brightness != 0:
                img = ImageEnhance.Brightness(img).enhance(
                    1.0 + brightness / 100.0)
            if contrast != 0:
                img = ImageEnhance.Contrast(img).enhance(
                    1.0 + contrast / 100.0)
            if saturation != 0:
                img = ImageEnhance.Color(img).enhance(
                    1.0 + saturation / 100.0)
            if sharpness != 0:
                img = ImageEnhance.Sharpness(img).enhance(
                    1.0 + sharpness / 50.0)
            if temperature != 0:
                r, g, b  = img.split()
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
            print(f"[调色错误] {e}")

    def reset_adjustments(self):
        for widget in self.adjust_vars.values():
            widget.blockSignals(True)
            widget.setValue(0)
            widget.blockSignals(False)
        self.current_img     = self.original_img.copy()
        self.base_img        = self.original_img.copy()
        self.filter_base_img = self.original_img.copy()
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText("🔄 调色已重置")
        self.lbl_status.setStyleSheet("color:#cdd6f4; font-size:13px;")

    def apply_selected_filter(self):
        try:
            filter_name = self.filter_combo.currentText()
            if filter_name == "无":
                return   # ✅ 直接返回，不打印不处理

            print(f"[滤镜] 应用: {filter_name}")
            img       = self.filter_base_img.copy()
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[-1]
                img   = img.convert("RGB")

            if filter_name == "黑白":
                gray = ImageOps.grayscale(img).convert("RGB")
                img  = ImageEnhance.Contrast(gray).enhance(1.2)

            elif filter_name == "复古":
                img = ImageEnhance.Color(img).enhance(0.6)
                r, g, b = img.split()
                r = r.point(lambda v: min(255, int(v * 1.1) + 20))
                g = g.point(lambda v: min(255, int(v * 1.05) + 10))
                b = b.point(lambda v: max(0,   int(v * 0.85)))
                img = Image.merge("RGB", (r, g, b))
                img = ImageEnhance.Contrast(img).enhance(0.95)

            elif filter_name == "冷色调":
                r, g, b = img.split()
                r = r.point(lambda v: max(0,   int(v * 0.85)))
                b = b.point(lambda v: min(255, int(v * 1.2)))
                img = Image.merge("RGB", (r, g, b))

            elif filter_name == "暖色调":
                r, g, b = img.split()
                r = r.point(lambda v: min(255, int(v * 1.2)))
                g = g.point(lambda v: min(255, int(v * 1.05)))
                b = b.point(lambda v: max(0,   int(v * 0.85)))
                img = Image.merge("RGB", (r, g, b))

            elif filter_name == "胶片颗粒":
                px   = img.load()
                w, h = img.size
                rng  = np.random.default_rng()
                xs   = rng.integers(0, w, int(w * h * 0.03))
                ys   = rng.integers(0, h, int(w * h * 0.03))
                ns   = rng.integers(-40, 40, len(xs))
                for x, y, n in zip(xs, ys, ns):
                    r2, g2, b2 = px[x, y]
                    px[x, y] = (
                        max(0, min(255, r2 + n)),
                        max(0, min(255, g2 + n)),
                        max(0, min(255, b2 + n)),
                    )

            elif filter_name == "模糊":
                radius = self.blur_scale.value()
                img    = img.filter(ImageFilter.GaussianBlur(radius=radius))

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

            self.push_history()
            self.current_img = img
            self.base_img    = img.copy()
            self.update_canvas(self.current_img, force=True)
            self.lbl_status.setText(f"✅ 已应用滤镜: {filter_name}")
            self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

        except Exception as e:
            print(f"[滤镜错误] {e}")
            import traceback; traceback.print_exc()