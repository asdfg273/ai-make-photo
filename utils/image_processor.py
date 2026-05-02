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

def process_adetailer(base_image, prompt, negative_prompt, seed, is_anime, strength, img2img_pipe, device, status_callback=None):
    """纯净的人脸修复函数，将 UI 状态更新通过 status_callback 回调抛出"""
    try:
        cv_img = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        if is_anime:
            xml_path = "lbpcascade_animeface.xml"
            if not os.path.exists(xml_path):
                url = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
                urllib.request.urlretrieve(url, xml_path)
            face_cascade = cv2.CascadeClassifier(xml_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            if status_callback: status_callback("🧑‍🎨 ADetailer: 未检测到明显人脸，跳过修复。", "gray")
            return base_image 

        if status_callback: status_callback(f"🧑‍🎨 ADetailer: 侦测到 {len(faces)} 张人脸，正在逐一精修...", "yellow")
        result_image = base_image.copy()
        
        for idx, (x, y, w, h) in enumerate(faces):
            try:
                if status_callback: status_callback(f"🧑‍🎨 ADetailer: 正在修复第 {idx+1}/{len(faces)} 张脸...", "yellow")
                
                margin_x, margin_y = int(w * 0.4), int(h * 0.4)
                x1, y1 = max(0, x - margin_x), max(0, y - int(margin_y * 1.5)) 
                x2, y2 = min(base_image.width, x + w + margin_x), min(base_image.height, y + h + margin_y)
                crop_w, crop_h = x2 - x1, y2 - y1
                
                face_crop = base_image.crop((x1, y1, x2, y2))
                face_crop_512 = face_crop.resize((512, 512), Image.Resampling.LANCZOS)

                extra_tag = "highly detailed anime face, perfect eyes, masterpiece" if is_anime else "beautiful detailed face, highly detailed eyes, perfectly symmetrical face, raw photo"
                enhanced_prompt = prompt + ", " + extra_tag
                generator = torch.Generator(device).manual_seed(seed)
                
                with torch.no_grad():
                    fixed_face_512 = img2img_pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=face_crop_512,
                        strength=strength, 
                        num_inference_steps=25,
                        generator=generator
                    ).images[0]
                
                fixed_face = fixed_face_512.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
                
                mask = Image.new("L", (crop_w, crop_h), 0)
                draw = ImageDraw.Draw(mask)
                blur_radius = max(5, min(50, int(min(crop_w, crop_h) * 0.15))) 
                draw.rectangle([blur_radius, blur_radius, crop_w-blur_radius, crop_h-blur_radius], fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(blur_radius)) 
                
                result_image.paste(fixed_face, (x1, y1), mask) 
            except Exception as inner_e:
                continue 

        return result_image
    except Exception as e:
        return base_image