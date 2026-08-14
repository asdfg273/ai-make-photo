# image_processor.py
import os
import datetime
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import requests

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


def _download_adetailer_model(url, save_path):
    """下载 ADetailer 模型(支持 hf-mirror 加速)。"""
    if os.path.exists(save_path):
        return True
    try:
        # 走 hf-mirror 加速
        url_mirror = url.replace("huggingface.co", "hf-mirror.com")
        print(f"  📥 下载 ADetailer 模型: {os.path.basename(save_path)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for attempt_url in [url_mirror, url]:
            try:
                r = requests.get(attempt_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  ✅ 下载完成: {save_path}")
                return True
            except Exception as e:
                print(f"  ⚠️ 从 {attempt_url} 下载失败: {e}")
                continue
        return False
    except Exception as e:
        print(f"  ❌ 模型下载失败: {e}")
        return False


def _yolo_detect(model_path, cv_image, conf=0.3):
    """YOLO 通用检测,返回 [(x, y, w, h), ...]"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ 缺少 ultralytics 库,请运行: pip install ultralytics")
        return []
    
    try:
        model = YOLO(model_path)
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
        print(f"⚠️ YOLO 检测异常: {e}")
        return []


def process_adetailer(base_image, inpaint_pipe, prompt, negative_prompt,
                     strength=None, target="二次元脸部",
                     padding_ratio=0.15, mask_blur=8):
    """
    工业级 ADetailer 局部重绘
    
    Args:
        base_image: PIL.Image 输入图
        inpaint_pipe: 已加载的 SD 修复管线
        prompt: 用户原始正向提示词
        negative_prompt: 用户原始负向提示词
        strength: 重绘强度 (None 则用预设默认值)
        target: 目标类型,见 ADETAILER_MODELS 的键
        padding_ratio: 检测框外扩比例 (0.15 = 外扩 15%)
        mask_blur: 蒙版边缘羽化(像素)
    
    Returns:
        PIL.Image: 修复后的图
    """
    try:
        # ── 1. 校验 target ──
        if target not in ADETAILER_MODELS:
            print(f"⚠️ 未知 ADetailer 目标: {target},支持: {list(ADETAILER_MODELS.keys())}")
            return base_image
        
        cfg = ADETAILER_MODELS[target]
        prompts_cfg = ADETAILER_PROMPTS.get(target, {"pos": "", "neg": ""})
        
        # 用预设默认 strength
        if strength is None:
            strength = cfg["default_strength"]
        
        # ── 2. 准备模型 ──
        from utils import paths
        model_dir = os.path.join(paths.CACHE_DIR, "adetailer")
        model_path = os.path.join(model_dir, cfg["name"])
        if not _download_adetailer_model(cfg["url"], model_path):
            print(f"⚠️ ADetailer 模型不可用,跳过 {target}")
            return base_image
        
        # ── 3. 准备图像 ──
        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")
        cv_image = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        W, H = base_image.size
        
        # ── 4. 目标检测 ──
        boxes = _yolo_detect(model_path, cv_image, conf=cfg["conf"])
        if not boxes:
            print(f"🤷‍♂️ 未在图中检测到 [{target}],跳过修复。")
            return base_image
        
        print(f"🔍 [ADetailer] 成功定位 {len(boxes)} 处 [{target}],开始进行局部重绘...")
        
        # ── 5. 构建 prompt(融合用户原 prompt + 模板) ──
        # 取用户 prompt 前 80 字符作为上下文(避免太长被截断)
        short_prompt = prompt[:80] if prompt else ""
        target_pos = f"{prompts_cfg['pos']}, {short_prompt}"
        target_neg = f"{prompts_cfg['neg']}, {negative_prompt}" if negative_prompt else prompts_cfg['neg']
        
        # ── 6. 逐个目标修复 ──
        result_image = base_image.copy()
        for idx, (x, y, w, h) in enumerate(boxes, 1):
            try:
                # 6.1 外扩检测框(给修复留余量)
                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(W, x + w + pad_x)
                y2 = min(H, y + h + pad_y)
                
                # 6.2 构建蒙版
                mask = Image.new("L", (W, H), 0)
                mask_arr = np.array(mask)
                mask_arr[y1:y2, x1:x2] = 255
                mask = Image.fromarray(mask_arr)
                # 边缘羽化(让修复区域过渡自然)
                if mask_blur > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                
                # 6.3 调用 inpaint pipeline
                # 🔧 检查 IPA,如有则传占位图避免 NoneType 报错
                ip_kwargs = {}
                if hasattr(inpaint_pipe, '_load_ip_adapter_weights') or \
                   hasattr(inpaint_pipe, 'image_encoder') and inpaint_pipe.image_encoder is not None:
                    # 用纯黑占位图(IPA scale=0 时不会真的产生影响)
                    placeholder = Image.new("RGB", (224, 224), (0, 0, 0))
                    ip_kwargs['ip_adapter_image'] = placeholder
                    print(f"  🩹 [ADetailer] 检测到 IPA,将传入占位图")
                
                output = inpaint_pipe(
                    prompt=target_pos,
                    negative_prompt=target_neg,
                    image=result_image,
                    mask_image=mask,
                    strength=strength,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    width=W,
                    height=H,
                    **ip_kwargs,
                )
                result_image = output.images[0]
                print(f"  ✅ [ADetailer] {target} {idx}/{len(boxes)} 修复完成")
            except Exception as e:
                print(f"  ⚠️ [ADetailer] 第 {idx} 处修复异常: {e}")
                continue
        
        return result_image
    
    except Exception as e:
        print(f"⚠️ ADetailer 致命错误: {e}")
        import traceback
        traceback.print_exc()
        return base_image