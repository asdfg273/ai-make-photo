# model_manager.py

import os
import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.current_model_name = None
        self.current_lora_name = None
        
        # 各种工作流管道
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.controlnet_pipe = None  # 👈 新增：ControlNet 管道
        
        self.pose_detector = None    # 👈 新增：骨架提取器

    def get_available_models(self):
        if not os.path.exists("models"): return []
        return [f for f in os.listdir("models") if f.endswith((".safetensors", ".ckpt"))]

    def get_available_loras(self):
        if not os.path.exists("loras"): return ["无"]
        loras = [f for f in os.listdir("loras") if f.endswith(".safetensors")]
        return ["无"] + loras

    def load_model(self, model_name):
        if model_name == self.current_model_name:
            return
            
        # 🌟 核心修复：必须先告诉程序 model_path 在哪里，再去检查它的大小！
        model_path = os.path.join("models", model_name) 
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
        
        try:
            if file_size_gb > 5.0:
                # 如果模型文件大于 5GB，大概率是 SDXL！使用 XL 管道加载
                print("检测到巨型模型，启用 SDXL 架构引擎...")
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path, 
                    torch_dtype=torch.float32, 
                    use_safetensors=True
                )
            else:
                # 普通 SD 1.5 架构
                self.pipe = StableDiffusionPipeline.from_single_file(
                    model_path, 
                    torch_dtype=torch.float32, 
                    use_safetensors=True
                )
        except Exception as e:
            raise Exception(f"架构加载失败: {str(e)}")

        model_path = os.path.join("models", model_name)
        
        # 1. 基础管道
        self.txt2img_pipe = StableDiffusionPipeline.from_single_file(
            model_path, torch_dtype=self.dtype, safety_checker=None
        ).to(self.device)
        
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**self.txt2img_pipe.components)
        self.inpaint_pipe = StableDiffusionInpaintPipeline(**self.txt2img_pipe.components)
        
        self.current_model_name = model_name
        self.current_lora_name = None

    def apply_lora(self, lora_name):
        if not self.txt2img_pipe: return
        
        # 如果选了“无”，就卸载之前的 LoRA
        if lora_name == "无" or not lora_name:
            self.txt2img_pipe.unload_lora_weights()
            self.current_lora_name = None
            return
            
        if lora_name == self.current_lora_name:
            return
            
        lora_path = os.path.join("loras", lora_name)
        self.txt2img_pipe.unload_lora_weights()
        self.txt2img_pipe.load_lora_weights(lora_path)
        self.current_lora_name = lora_name

    # 👇 新增：准备 ControlNet 管道
    def prepare_controlnet(self):
        # 如果还没加载 ControlNet 模型，自动从 HuggingFace 缓存加载 (初次会下载大约 1.4GB)
        if not self.controlnet_pipe:
            print("正在装载 ControlNet (首次运行会自动下载权重文件)...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose", 
                torch_dtype=self.dtype
            )
            # 组合成带有姿势控制的生图管道
            self.controlnet_pipe = StableDiffusionControlNetPipeline(
                **self.txt2img_pipe.components, 
                controlnet=controlnet
            ).to(self.device)
            
        if not self.pose_detector:
            print("正在装载 OpenPose 骨架提取器...")
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    # 👇 新增：提取骨架并返回
    def get_pose_image(self, input_image):
        if not self.pose_detector:
            self.prepare_controlnet()
        return self.pose_detector(input_image)
