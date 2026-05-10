# image_processor.py
import os
import datetime
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

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

def process_adetailer(base_image, inpaint_pipe, prompt, negative_prompt, strength=0.35, target="现实脸部"):
    """
    工业级 ADetailer 流水线：
    支持 真人/二次元 面部与手部，自动处理8倍数对齐、提示词截断截流与无缝融合。
    """
    try:
        # 1. 智能精简提示词，防止爆显存和超 77 Token 限制
        # 只保留原提示词的前 20 个词作为背景参考
        short_prompt = " ".join(prompt.split(",")[:4]) 
        
        if "二次元手部" in target:
            target_prompt = "perfect anime hands, highly detailed, five fingers, flawless, " + short_prompt
            target_neg = "bad hands, missing fingers, extra fingers, deformed, mutated, " + negative_prompt
            model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"
            model_name = "hand_yolov8n.pt"
        elif "现实手部" in target:
            target_prompt = "perfect realistic human hands, highly detailed, 5 fingers, pores, " + short_prompt
            target_neg = "bad hands, missing fingers, extra fingers, deformed, mutated, " + negative_prompt
            model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"
            model_name = "hand_yolov8n.pt"
        elif "二次元脸部" in target:
            target_prompt = "perfect anime face, highly detailed eyes, beautiful face, masterpiece, " + short_prompt
            target_neg = "bad face, deformed eyes, ugly, poorly drawn face, " + negative_prompt
            # 专属二次元人脸检测模型！
            model_url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
            model_name = "lbpcascade_animeface.xml"
        else:
            target_prompt = "perfect human face, highly detailed, realistic skin, beautiful, " + short_prompt
            target_neg = "bad face, deformed, ugly, poorly drawn face, " + negative_prompt
            model_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            model_name = "haarcascade_frontalface_default.xml"

        # 2. 准备模型模型
        model_path = os.path.join("models", model_name)
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_path):
            print(f"📥 正在下载 {target} 检测模型: {model_name} ...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"⚠️ 下载失败: {e}，跳过该通道。")
                return base_image

        open_cv_image = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        boxes = []

        # 3. 目标检测 (区分 YOLO 和 OpenCV Cascade)
        if "hand" in model_name:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(model_path)
                results = yolo_model(open_cv_image, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            except ImportError:
                print("⚠️ 缺少 ultralytics 库，请在终端运行: pip install ultralytics")
                return base_image
        else:
            detector = cv2.CascadeClassifier(model_path)
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            # 针对二次元脸稍微调低阈值，提高识别率
            min_neighbors = 3 if "anime" in model_name else 5
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(30, 30))

        if len(boxes) == 0:
            print(f"🤷‍♂️ 未在图中检测到 [{target}]，跳过修复。")
            return base_image

        print(f"🔍 [ADetailer] 成功定位 {len(boxes)} 处 [{target}]，开始进行局部重绘...")
        result_image = base_image.copy()

        # 4. 执行逐个区域修复
        for (x, y, w, h) in boxes:
            try:
                # 扩大框选范围，提供更多周围上下文
                padding = int(max(w, h) * 0.3)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(result_image.width, x + w + padding)
                y2 = min(result_image.height, y + h + padding)

                crop_img = result_image.crop((x1, y1, x2, y2))
                
                # 🌟 核心魔法：强制将尺寸调整为 512x512，完美规避 "images do not match" 错误！
                orig_size = crop_img.size
                crop_img_512 = crop_img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # 创建全白的 512x512 遮罩
                mask_512 = Image.new("L", (512, 512), 255)
                # 边缘羽化，让接缝处不可见
                mask_512 = mask_512.filter(ImageFilter.GaussianBlur(10))

                # 进行重绘
                fixed_crop_512 = inpaint_pipe(
                    prompt=target_prompt,
                    negative_prompt=target_neg,
                    image=crop_img_512,
                    mask_image=mask_512,
                    strength=strength,
                    num_inference_steps=20, # ADetailer 固定20步即可
                    guidance_scale=7.5
                ).images[0]

                # 将修好的 512x512 缩放回原本的尺寸
                fixed_crop_orig = fixed_crop_512.resize(orig_size, Image.Resampling.LANCZOS)
                mask_orig = mask_512.resize(orig_size, Image.Resampling.LANCZOS)

                # 无缝贴回原图
                result_image.paste(fixed_crop_orig, (x1, y1), mask_orig)
            except Exception as e:
                print(f"⚠️ 局部重绘警告: {e}")
                continue

        return result_image

    except Exception as e:
        print(f"⚠️ ADetailer 致命错误: {e}")
        return base_image