# photo_turn/mixin_filters.py

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from PyQt6.QtCore import QTimer
import logging

logger = logging.getLogger(__name__)


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
            exposure    = self.adjust_vars.get("exposure")
            exposure    = exposure.value() if exposure else 0
            hue         = self.adjust_vars.get("hue")
            hue         = hue.value() if hue else 0

            if exposure != 0:
                # gamma 曝光:正值提亮
                gamma = 2 ** (-exposure / 100.0)
                lut   = np.clip((np.arange(256) / 255.0) ** gamma * 255,
                                0, 255).astype(np.uint8)
                img   = Image.fromarray(lut[np.asarray(img)])
            if brightness != 0:
                img = ImageEnhance.Brightness(img).enhance(
                    1.0 + brightness / 100.0)
            if contrast != 0:
                img = ImageEnhance.Contrast(img).enhance(
                    1.0 + contrast / 100.0)
            if saturation != 0:
                img = ImageEnhance.Color(img).enhance(
                    1.0 + saturation / 100.0)
            if hue != 0:
                # HSV 色相环旋转:PIL HSV 的 H 通道 0-255 对应 0-360°
                shift = int(hue * 1.8 / 360 * 256)   # ±100 → ±180°
                hsv   = np.asarray(img.convert("HSV")).copy()
                hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
                img   = Image.fromarray(hsv, "HSV").convert("RGB")
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
            logger.warning(f"[调色错误] {e}")

    def reset_adjustments(self):
        for widget in self.adjust_vars.values():
            widget.blockSignals(True)
            widget.setValue(0)
            widget.blockSignals(False)
        self.current_img = self.original_img.copy()
        self.base_img    = self.original_img.copy()
        self._filter_anchor = None   # 回到原图,滤镜锚点失效
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText("🔄 调色已重置")
        self.lbl_status.setStyleSheet("color:#cdd6f4; font-size:13px;")

    def _revert_filters(self):
        """选择「无」时:还原到滤镜叠加前的基图"""
        anchor = getattr(self, "_filter_anchor", None)
        if anchor is None:
            self.lbl_status.setText("✨ 当前没有叠加中的滤镜")
            self.lbl_status.setStyleSheet("color:#f9e2af; font-size:13px;")
            return
        self.push_history()
        self.current_img     = anchor.copy()
        self.base_img        = anchor.copy()
        self._filter_anchor  = None
        self.update_canvas(self.current_img, force=True)
        self.lbl_status.setText("✅ 已还原到滤镜叠加前的状态")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")

    # ----------------------------------------------------------
    #  滤镜实现(全部 numpy/PIL,输入 RGB 图)
    # ----------------------------------------------------------
    @staticmethod
    def _f_bw(img):
        gray = ImageOps.grayscale(img).convert("RGB")
        return ImageEnhance.Contrast(gray).enhance(1.2)

    @staticmethod
    def _f_sepia(img):
        img = ImageEnhance.Color(img).enhance(0.6)
        r, g, b = img.split()
        r = r.point(lambda v: min(255, int(v * 1.1) + 20))
        g = g.point(lambda v: min(255, int(v * 1.05) + 10))
        b = b.point(lambda v: max(0,   int(v * 0.85)))
        img = Image.merge("RGB", (r, g, b))
        return ImageEnhance.Contrast(img).enhance(0.95)

    @staticmethod
    def _f_cool(img):
        r, g, b = img.split()
        r = r.point(lambda v: max(0,   int(v * 0.85)))
        b = b.point(lambda v: min(255, int(v * 1.2)))
        return Image.merge("RGB", (r, g, b))

    @staticmethod
    def _f_warm(img):
        r, g, b = img.split()
        r = r.point(lambda v: min(255, int(v * 1.2)))
        g = g.point(lambda v: min(255, int(v * 1.05)))
        b = b.point(lambda v: max(0,   int(v * 0.85)))
        return Image.merge("RGB", (r, g, b))

    @staticmethod
    def _f_grain(img):
        """胶片颗粒 — numpy 向量化(替代逐像素循环)"""
        arr   = np.asarray(img).astype(np.int16)
        h, w  = arr.shape[:2]
        rng   = np.random.default_rng()
        noise = rng.integers(-40, 41, (h, w, 1))
        sel   = rng.random((h, w, 1)) < 0.25
        arr   = np.where(sel, arr + noise, arr)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def _f_invert(img):
        return ImageOps.invert(img)

    @staticmethod
    def _f_posterize(img):
        return ImageOps.posterize(img, 4)

    @staticmethod
    def _f_vignette(img):
        """晕影:四角径向压暗"""
        w, h  = img.size
        y, x  = np.ogrid[:h, :w]
        dist  = np.sqrt(((x - w / 2) / (w / 2)) ** 2
                        + ((y - h / 2) / (h / 2)) ** 2)
        mask  = 1.0 - 0.65 * np.clip((dist - 0.55) / 0.75, 0, 1)
        arr   = np.asarray(img).astype(np.float32) * mask[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _f_pixelate(self, img):
        """像素化(马赛克),块大小取自强度滑块"""
        w, h  = img.size
        block = max(2, self.blur_scale.value() * 2)
        small = img.resize((max(1, w // block), max(1, h // block)),
                           Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.NEAREST)

    @staticmethod
    def _f_sketch(img):
        """铅笔素描(色块除法)"""
        import cv2
        gray  = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        inv   = 255 - gray
        blur  = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        return Image.fromarray(sketch).convert("RGB")

    @staticmethod
    def _f_cartoon(img):
        """卡通风格化(双边滤波 + 自适应边缘)"""
        import cv2
        arr   = np.asarray(img)
        color = cv2.bilateralFilter(arr, 9, 150, 150)
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 5), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        out   = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(out)

    def apply_selected_filter(self):
        try:
            filter_name = self.filter_combo.currentText()
            if filter_name == "无":
                self._revert_filters()
                return

            logger.info(f"[滤镜] 应用: {filter_name}")
            # 第一次叠加滤镜前,锚定基图(供「无」还原)
            if getattr(self, "_filter_anchor", None) is None:
                self._filter_anchor = self.base_img.copy()
            img       = self.base_img.copy()
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[-1]
                img   = img.convert("RGB")

            if filter_name == "黑白":
                img = self._f_bw(img)
            elif filter_name == "复古":
                img = self._f_sepia(img)
            elif filter_name == "冷色调":
                img = self._f_cool(img)
            elif filter_name == "暖色调":
                img = self._f_warm(img)
            elif filter_name == "胶片颗粒":
                img = self._f_grain(img)
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
            elif filter_name == "负片":
                img = self._f_invert(img)
            elif filter_name == "像素化":
                img = self._f_pixelate(img)
            elif filter_name == "晕影":
                img = self._f_vignette(img)
            elif filter_name == "色调分离":
                img = self._f_posterize(img)
            elif filter_name == "素描":
                img = self._f_sketch(img)
            elif filter_name == "卡通":
                img = self._f_cartoon(img)
            else:
                logger.warning(f"[滤镜] 未知滤镜: {filter_name}")
                return

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
            logger.warning(f"[滤镜错误] {e}")
            logger.debug("滤镜堆栈", exc_info=True)
