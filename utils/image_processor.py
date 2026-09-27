# image_processor.py
import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import logging

logger = logging.getLogger(__name__)

def make_comic_strip(image_list):
    """纯净的漫画排版函数，只负责计算和绘图，返回 PIL Image"""
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

    return comic_bg

ADETAILER_MODELS = {
    # 动漫
    "二次元脸部": {
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
        "name": "face_yolov8n.pt",
        "conf": 0.30,
        "default_strength": 0.25,
    },
    "二次元手部": {
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
        "name": "hand_yolov8n.pt",
        "conf": 0.30,
        "default_strength": 0.45,
    },
    # 现实(共用 YOLOv8 模型,YOLOv8 同时支持真人和动漫)
    "现实脸部": {
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
        "name": "face_yolov8n.pt",
        "conf": 0.35,
        "default_strength": 0.30,
    },
    "现实手部": {
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
        "name": "hand_yolov8n.pt",
        "conf": 0.35,
        "default_strength": 0.45,
    },
    # 全身分割
    "全身人物": {
        "url": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt",
        "name": "person_yolov8n-seg.pt",
        "conf": 0.30,
        "default_strength": 0.20,
    },
}


# ── Prompt 模板 ──
ADETAILER_PROMPTS = {
    "二次元脸部": {
        "pos": "perfect anime face, highly detailed eyes, symmetric eyes, beautiful face, clean linework, masterpiece",
        "neg": "bad face, deformed eyes, asymmetric eyes, blurry face, poorly drawn face, extra eyes, lowres",
    },
    "二次元手部": {
        "pos": "perfect anime hands, five fingers, detailed fingers, natural pose, masterpiece",
        "neg": "bad hands, extra fingers, missing fingers, fused fingers, deformed hands, mutation",
    },
    "现实脸部": {
        "pos": "perfect realistic face, detailed skin texture, sharp eyes, photorealistic, 8k",
        "neg": "bad face, deformed, asymmetric, blurry, low quality, plastic skin",
    },
    "现实手部": {
        "pos": "perfect realistic hands, five fingers, detailed skin, natural anatomy, photorealistic",
        "neg": "bad hands, extra fingers, missing fingers, fused fingers, deformed hands, mutation",
    },
    "全身人物": {
        "pos": "perfect body, anatomically correct, high quality, masterpiece",
        "neg": "deformed body, bad anatomy, low quality, lowres",
    },
}


def _get_adetailer_model_path(cfg):
    """
    只计算 ADetailer 模型应该存放的位置，不执行下载。
    """
    from utils import paths

    model_dir = Path(paths.MODEL_DIR) / "adetailer"

    configured_path = (
        cfg.get("local_path")
        or cfg.get("name")
    )

    if not configured_path:
        raise ValueError(
            "ADetailer 模型缺少 local_path 或 name 配置"
        )

    configured_path = Path(str(configured_path)).expanduser()

    if configured_path.is_absolute():
        return configured_path

    # 防止配置中写入 models/adetailer/xxx.pt 后重复拼接目录
    if configured_path.parts[:2] == (
        "models",
        "adetailer",
    ):
        return Path(*configured_path.parts)

    return model_dir / configured_path


def _resolve_local_adetailer_model(cfg):
    """
    只查找本地 ADetailer 模型。
    生图流程只能调用这个函数，不能调用下载器。
    """
    model_path = _get_adetailer_model_path(cfg)

    if model_path.is_file():
        logger.info(
            "✅ 使用本地 ADetailer 模型：%s",
            model_path,
        )
        return str(model_path)

    if model_path.is_dir():
        logger.info(
            "✅ 使用本地 ADetailer 模型目录：%s",
            model_path,
        )
        return str(model_path)

    logger.warning(
        "⚠️ 未找到本地 ADetailer 模型：%s",
        model_path,
    )

    return None


_YOLO_CACHE = {}


def _yolo_detect(model_path, cv_image, conf=0.3):
    """YOLO 通用检测,返回 [(x, y, w, h), ...]（模型按路径缓存,不重复加载）"""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("⚠️ 缺少 ultralytics 库,请运行: pip install ultralytics")
        return []

    try:
        model = _YOLO_CACHE.get(model_path)
        if model is None:
            model = YOLO(model_path)
            _YOLO_CACHE[model_path] = model
        results = model(cv_image, verbose=False, conf=conf)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = int(x2 - x1)
                h = int(y2 - y1)
                # 过滤太小的框(避免误检)
                if w < 20 or h < 20:
                    continue
                boxes.append((int(x1), int(y1), w, h))
        return boxes
    except Exception as e:
        logger.warning(f"⚠️ YOLO 检测异常: {e}")
        return []


def process_adetailer(
    base_image,
    inpaint_pipe,
    prompt,
    negative_prompt,
    strength=None,
    target="二次元脸部",
    padding_ratio=0.15,
    mask_blur=8,
    detail_size=512,
    ipa_loaded=None,
    ipa_scale=None,
):
    """
    使用本地检测模型进行 ADetailer 局部重绘。

    适合 8GB 显存：
    1. 先检测目标区域；
    2. 只裁剪目标区域进行重绘；
    3. 将重绘结果羽化贴回原图；
    4. ADetailer 执行期间临时关闭 IPA；
    5. 不自动下载任何模型。

    参数：
        detail_size:
            局部重绘的最长边。5060 8GB 建议使用 512。
            如果显存充足，可以改成 640。

        ipa_loaded:
            当前 inpaint_pipe 使用的共享 UNet 是否已经加载 IPA。
            如果传 None，则从 inpaint_pipe._ai_ipa_loaded 读取。

        ipa_scale:
            IPA 原来的 scale。
            如果传 None，则从 inpaint_pipe._ai_ipa_scale 读取，
            默认使用 0.6。
    """
    original_image = base_image

    try:
        if target not in ADETAILER_MODELS:
            logger.warning(
                "⚠️ 未知 ADetailer 目标：%s，支持：%s",
                target,
                list(ADETAILER_MODELS.keys()),
            )
            return original_image

        if inpaint_pipe is None:
            logger.warning("⚠️ inpaint_pipe 不存在，跳过 ADetailer")
            return original_image

        cfg = ADETAILER_MODELS[target]
        prompts_cfg = ADETAILER_PROMPTS.get(
            target,
            {
                "pos": "",
                "neg": "",
            },
        )

        if strength is None:
            strength = cfg.get("default_strength", 0.35)

        try:
            strength = float(strength)
        except Exception:
            strength = 0.35

        # 防止 strength 为 0 或 1 导致结果异常
        strength = max(0.05, min(0.95, strength))

        # 只从本地查找模型，不自动下载
        model_path = _resolve_local_adetailer_model(cfg)

        if not model_path:
            logger.warning(
                "⚠️ 未找到 ADetailer 模型 [%s]，"
                "请先在模型管理界面点击下载：%s",
                target,
                _get_adetailer_model_path(cfg),
            )
            return original_image

        # 准备原图
        if not isinstance(base_image, Image.Image):
            logger.warning("⚠️ base_image 不是 PIL.Image，跳过 ADetailer")
            return original_image

        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")

        result_image = base_image.copy()
        width, height = result_image.size

        cv_image = cv2.cvtColor(
            np.asarray(result_image),
            cv2.COLOR_RGB2BGR,
        )

        # 目标检测
        boxes = _yolo_detect(
            model_path,
            cv_image,
            conf=cfg.get("conf", 0.35),
        )

        if not boxes:
            logger.info(
                "🤷‍♂️ 未在图像中检测到 [%s]，跳过修复",
                target,
            )
            return result_image

        logger.info(
            "🔍 [ADetailer] 检测到 %d 个 [%s] 目标",
            len(boxes),
            target,
        )

        # Prompt 不要无限制拼接，防止局部重绘提示词过长
        short_prompt = (prompt or "").strip()[:160]
        preset_positive = prompts_cfg.get("pos", "").strip()
        preset_negative = prompts_cfg.get("neg", "").strip()
        user_negative = (negative_prompt or "").strip()

        positive_parts = [
            item
            for item in (
                preset_positive,
                short_prompt,
            )
            if item
        ]

        negative_parts = [
            item
            for item in (
                preset_negative,
                user_negative,
            )
            if item
        ]

        target_positive = ", ".join(positive_parts)
        target_negative = ", ".join(negative_parts)

        # IPA 状态必须由明确状态控制
        if ipa_loaded is None:
            ipa_loaded = bool(
                getattr(inpaint_pipe, "_ai_ipa_loaded", False)
            )

        if ipa_scale is None:
            ipa_scale = getattr(
                inpaint_pipe,
                "_ai_ipa_scale",
                0.6,
            )

        try:
            ipa_scale = float(ipa_scale)
        except Exception:
            ipa_scale = 0.6

        def set_ipa_scale(pipe, scale):
            """
            同时兼容：
            - pipeline.set_ip_adapter_scale()
            - pipe.unet.set_ip_adapter_scale()
            """
            setter = getattr(
                pipe,
                "set_ip_adapter_scale",
                None,
            )

            if callable(setter):
                setter(scale)
                pipe._ai_ipa_scale = scale
                return True

            unet = getattr(pipe, "unet", None)
            setter = getattr(
                unet,
                "set_ip_adapter_scale",
                None,
            )

            if callable(setter):
                setter(scale)
                pipe._ai_ipa_scale = scale
                return True

            return False

        def round_to_8(value):
            value = int(round(float(value) / 8.0) * 8)
            return max(64, value)

        # 检测框按面积从大到小处理
        boxes = sorted(
            boxes,
            key=lambda item: int(item[2]) * int(item[3]),
            reverse=True,
        )

        for index, box in enumerate(boxes, start=1):
            try:
                x, y, box_w, box_h = [
                    int(value)
                    for value in box[:4]
                ]

                if box_w <= 0 or box_h <= 0:
                    continue

                # 修正检测框越界
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

                box_x2 = min(width, x + box_w)
                box_y2 = min(height, y + box_h)

                box_w = box_x2 - x
                box_h = box_y2 - y

                if box_w <= 4 or box_h <= 4:
                    continue

                # 给局部重绘保留上下文
                pad_x = max(4, int(box_w * padding_ratio))
                pad_y = max(4, int(box_h * padding_ratio))

                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(width, box_x2 + pad_x)
                crop_y2 = min(height, box_y2 + pad_y)

                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1

                if crop_w <= 8 or crop_h <= 8:
                    continue

                local_image = result_image.crop(
                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                    )
                )

                # 局部蒙版只覆盖检测到的目标，
                # 保留周围区域作为参考，融合更自然
                local_mask = Image.new(
                    "L",
                    (crop_w, crop_h),
                    0,
                )

                draw = ImageDraw.Draw(local_mask)

                mask_pad_x = max(2, int(box_w * 0.06))
                mask_pad_y = max(2, int(box_h * 0.06))

                mask_x1 = max(
                    0,
                    x - mask_pad_x - crop_x1,
                )
                mask_y1 = max(
                    0,
                    y - mask_pad_y - crop_y1,
                )
                mask_x2 = min(
                    crop_w,
                    box_x2 + mask_pad_x - crop_x1,
                )
                mask_y2 = min(
                    crop_h,
                    box_y2 + mask_pad_y - crop_y1,
                )

                draw.rectangle(
                    (
                        mask_x1,
                        mask_y1,
                        mask_x2,
                        mask_y2,
                    ),
                    fill=255,
                )

                if mask_blur and mask_blur > 0:
                    local_mask = local_mask.filter(
                        ImageFilter.GaussianBlur(
                            radius=int(mask_blur)
                        )
                    )

                # 将局部区域缩放到适合 8GB 显存的尺寸
                longest_edge = max(crop_w, crop_h)
                render_scale = float(detail_size) / float(
                    longest_edge
                )

                render_w = round_to_8(crop_w * render_scale)
                render_h = round_to_8(crop_h * render_scale)

                local_image_render = local_image.resize(
                    (render_w, render_h),
                    Image.Resampling.LANCZOS,
                )

                local_mask_render = local_mask.resize(
                    (render_w, render_h),
                    Image.Resampling.LANCZOS,
                )

                # ADetailer 禁止使用 IPA 参考图。
                # 如果共享 UNet 已加载 IPA，则传占位图，
                # 同时把 IPA scale 临时设为 0。
                ip_kwargs = {}
                old_ipa_scale = ipa_scale
                ipa_scale_changed = False

                try:
                    if ipa_loaded:
                        placeholder = Image.new(
                            "RGB",
                            (224, 224),
                            (0, 0, 0),
                        )

                        ip_kwargs["ip_adapter_image"] = (
                            placeholder
                        )

                        try:
                            ipa_scale_changed = set_ipa_scale(
                                inpaint_pipe,
                                0.0,
                            )
                        except Exception as ipa_error:
                            logger.warning(
                                "⚠️ [ADetailer] 暂时关闭 IPA 失败：%s",
                                ipa_error,
                            )

                    logger.info(
                        "🩹 [ADetailer] 局部重绘 %d/%d，尺寸 %dx%d",
                        index,
                        len(boxes),
                        render_w,
                        render_h,
                    )

                    output = inpaint_pipe(
                        prompt=target_positive,
                        negative_prompt=target_negative,
                        image=local_image_render,
                        mask_image=local_mask_render,
                        strength=strength,
                        num_inference_steps=24,
                        guidance_scale=6.5,
                        width=render_w,
                        height=render_h,
                        **ip_kwargs,
                    )

                    if not output or not getattr(
                        output,
                        "images",
                        None,
                    ):
                        logger.warning(
                            "⚠️ [ADetailer] 管线没有返回图像"
                        )
                        continue

                    generated_crop = output.images[0]

                    if generated_crop.mode != "RGB":
                        generated_crop = generated_crop.convert(
                            "RGB"
                        )

                    # 缩回原局部区域大小
                    generated_crop = generated_crop.resize(
                        (crop_w, crop_h),
                        Image.Resampling.LANCZOS,
                    )

                    # 用羽化蒙版贴回原图
                    result_image.paste(
                        generated_crop,
                        (crop_x1, crop_y1),
                        local_mask,
                    )

                    logger.info(
                        "✅ [ADetailer] %s %d/%d 修复完成",
                        target,
                        index,
                        len(boxes),
                    )

                finally:
                    # 无论生成成功或失败，都恢复 IPA scale
                    if ipa_loaded and ipa_scale_changed:
                        try:
                            set_ipa_scale(
                                inpaint_pipe,
                                old_ipa_scale,
                            )
                        except Exception as restore_error:
                            logger.warning(
                                "⚠️ [ADetailer] 恢复 IPA scale 失败：%s",
                                restore_error,
                            )

            except Exception as item_error:
                logger.warning(
                    "⚠️ [ADetailer] 第 %d 个目标处理失败：%s",
                    index,
                    item_error,
                    exc_info=True,
                )
                continue

        return result_image

    except Exception as error:
        logger.warning(
            "⚠️ [ADetailer] 致命错误：%s",
            error,
            exc_info=True,
        )
        return original_image