# image_processor.py
"""
图像处理核心类：封装 Pillow 的调整、滤镜、裁剪、旋转等操作。
- 所有方法均为纯函数，不修改输入图像。
- 调用前请确保图像已转换为适当的 mode（通常为 RGB 或 RGBA）。
"""
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class ImageProcessor:

    # -------------------- 基础调整 --------------------
    @staticmethod
    def process_adjustments(pil_image: Image.Image, params: dict) -> Image.Image:
        """
        params:
            brightness: -100 ~ 100  -> factor 0.0 ~ 2.0
            contrast:   -100 ~ 100  -> factor 0.0 ~ 2.0
            saturation: -100 ~ 100  -> factor 0.0 ~ 2.0
            sharpness:    0 ~ 100   -> factor 1.0 ~ 3.0
            temperature:-100 ~ 100  -> RGB 乘性调整
        """
        img = pil_image.copy()

        # 保留 alpha 通道，处理主体 RGB
        has_alpha = img.mode == "RGBA"
        if has_alpha:
            alpha = img.split()[-1]
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 亮度
        b = params.get("brightness", 0)
        if b != 0:
            img = ImageEnhance.Brightness(img).enhance(1.0 + b / 100.0)

        # 对比度
        c = params.get("contrast", 0)
        if c != 0:
            img = ImageEnhance.Contrast(img).enhance(1.0 + c / 100.0)

        # 饱和度
        s = params.get("saturation", 0)
        if s != 0:
            img = ImageEnhance.Color(img).enhance(1.0 + s / 100.0)

        # 锐化 0 ~ 100 映射到 1.0 ~ 3.0
        sh = params.get("sharpness", 0)
        if sh != 0:
            img = ImageEnhance.Sharpness(img).enhance(1.0 + sh / 50.0)

        # 色温：暖色（+）提升 R 降低 B；冷色（-）反之
        t = params.get("temperature", 0)
        if t != 0:
            r, g, bch = img.split()
            factor_r = 1.0 + t / 200.0   # +100 -> 1.5
            factor_b = 1.0 - t / 200.0   # -100 -> 1.5
            r = r.point(lambda v: min(255, int(v * factor_r)))
            bch = bch.point(lambda v: min(255, int(v * factor_b)))
            img = Image.merge("RGB", (r, g, bch))

        if has_alpha:
            img = img.convert("RGBA")
            img.putalpha(alpha)

        return img

    # -------------------- 预设滤镜 --------------------
    @staticmethod
    def apply_filter(pil_image: Image.Image, filter_name: str,
                     blur_radius: float = 2.0) -> Image.Image:
        img = pil_image.copy()
        has_alpha = img.mode == "RGBA"
        if has_alpha:
            alpha = img.split()[-1]
            img_rgb = img.convert("RGB")
        else:
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img

        name = filter_name.lower()

        if name in ("黑白", "grayscale"):
            gray = ImageOps.grayscale(img_rgb).convert("RGB")
            img_rgb = ImageEnhance.Contrast(gray).enhance(1.2)

        elif name in ("复古", "vintage"):
            # 先降饱和，叠加黄褐色调
            img_rgb = ImageEnhance.Color(img_rgb).enhance(0.6)
            r, g, b = img_rgb.split()
            r = r.point(lambda v: min(255, int(v * 1.1) + 20))
            g = g.point(lambda v: min(255, int(v * 1.05) + 10))
            b = b.point(lambda v: max(0, int(v * 0.85)))
            img_rgb = Image.merge("RGB", (r, g, b))
            img_rgb = ImageEnhance.Contrast(img_rgb).enhance(0.95)

        elif name in ("冷色调", "cool"):
            r, g, b = img_rgb.split()
            r = r.point(lambda v: max(0, int(v * 0.85)))
            b = b.point(lambda v: min(255, int(v * 1.2)))
            img_rgb = Image.merge("RGB", (r, g, b))

        elif name in ("暖色调", "warm"):
            r, g, b = img_rgb.split()
            r = r.point(lambda v: min(255, int(v * 1.2)))
            g = g.point(lambda v: min(255, int(v * 1.05)))
            b = b.point(lambda v: max(0, int(v * 0.85)))
            img_rgb = Image.merge("RGB", (r, g, b))

        elif name in ("胶片颗粒", "grain"):
            # 添加随机噪声
            px = img_rgb.load()
            w, h = img_rgb.size
            for _ in range(int(w * h * 0.03)):   # 覆盖约 3% 像素
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                noise = random.randint(-40, 40)
                r, g, b = px[x, y]
                px[x, y] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise)),
                )

        elif name in ("高斯模糊", "blur"):
            img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        else:
            # 未知滤镜，返回原图
            pass

        if has_alpha:
            img_rgb = img_rgb.convert("RGBA")
            img_rgb.putalpha(alpha)

        return img_rgb

    # -------------------- 旋转与镜像 --------------------
    @staticmethod
    def rotate_left(pil_image: Image.Image) -> Image.Image:
        return pil_image.rotate(90, expand=True)

    @staticmethod
    def rotate_right(pil_image: Image.Image) -> Image.Image:
        return pil_image.rotate(-90, expand=True)

    @staticmethod
    def flip_horizontal(pil_image: Image.Image) -> Image.Image:
        return pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    @staticmethod
    def flip_vertical(pil_image: Image.Image) -> Image.Image:
        return pil_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # -------------------- 裁剪 --------------------
    @staticmethod
    def crop(pil_image: Image.Image, box: tuple) -> Image.Image:
        """box: (left, top, right, bottom)，以像素为单位"""
        left, top, right, bottom = box
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(pil_image.width, int(right))
        bottom = min(pil_image.height, int(bottom))
        if right <= left or bottom <= top:
            return pil_image.copy()
        return pil_image.crop((left, top, right, bottom))