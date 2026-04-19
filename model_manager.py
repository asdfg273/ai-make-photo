import os
import gc
import torch
import cv2           
import numpy as np  
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,             
    StableDiffusionXLImg2ImgPipeline,      
    StableDiffusionXLInpaintPipeline,      
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType


class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.current_model_name = None
        self.current_lora_name = None
        self.is_sdxl = False  
        
        # 各种工作流管道
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.controlnet_pipe = None
        self.pose_detector = None
        self.depth_estimator = None
        self.loaded_controlnets = {}  # 缓存不同的 ControlNet 模型
        self.current_cn_type = None

    def get_available_models(self):
        if not os.path.exists("models"): return []
        return [f for f in os.listdir("models") if f.endswith((".safetensors", ".ckpt"))]

    def get_available_loras(self):
        if not os.path.exists("loras"): return ["无"]
        loras = [f for f in os.listdir("loras") if f.endswith(".safetensors")]
        return ["无"] + loras

    # 🧹 新增：彻底清理内存的方法
    def clear_memory(self):
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.controlnet_pipe = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # 🚀 新增：应用极限显存优化
    def apply_optimizations(self, pipe):
        if self.device == "cuda":
            # 1. 注意力机制优化 (加速生成，降低显存)
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                pipe.enable_attention_slicing()
            
            # 2. 核心！模型层级 CPU 卸载 (极大降低 VRAM 占用，低显存福音)
            # 注意：使用了这个就不需要再手动调用 .to("cuda") 了！
            pipe.enable_model_cpu_offload()

            # 3. VAE 切片与分块 (生成大图、高分辨率修复、漫画网格时防止爆显存)
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

    def load_model(self, model_name):
        if model_name == self.current_model_name:
            return
            
        model_path = os.path.join("models", model_name) 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 切换模型前，彻底释放旧模型的显存
        self.clear_memory()

        file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
        self.is_sdxl = file_size_gb > 5.0
        
        try:
            print(f"正在加载 {'SDXL' if self.is_sdxl else 'SD 1.5'} 模型...")
            
            # 🌟 修复BUG：动态选择基础和派生管道类，并只加载一次！
            if self.is_sdxl:
                pipe_class = StableDiffusionXLPipeline
                img2img_class = StableDiffusionXLImg2ImgPipeline
                inpaint_class = StableDiffusionXLInpaintPipeline
            else:
                pipe_class = StableDiffusionPipeline
                img2img_class = StableDiffusionImg2ImgPipeline
                inpaint_class = StableDiffusionInpaintPipeline

            # 1. 只进行一次物理加载，直接使用 self.dtype (节省内存)
            self.txt2img_pipe = pipe_class.from_single_file(
                model_path, 
                torch_dtype=self.dtype, 
                use_safetensors=True,
                safety_checker=None
            )
            
            # 对主管道应用优化
            self.apply_optimizations(self.txt2img_pipe)
            
            # 2. 组件秒级共享（不增加额外内存开销）
            self.img2img_pipe = img2img_class(**self.txt2img_pipe.components)
            self.inpaint_pipe = inpaint_class(**self.txt2img_pipe.components)
            
            # 对子管道也应用优化（因为卸载机制绑定在特定工作流上）
            self.apply_optimizations(self.img2img_pipe)
            self.apply_optimizations(self.inpaint_pipe)

            self.current_model_name = model_name
            self.current_lora_name = None
            print("模型加载与内存优化完成！")
            
        except Exception as e:
            raise Exception(f"架构加载失败: {str(e)}")

    def apply_lora(self, lora_name, sub_dir="sd1.5"):
        if not self.txt2img_pipe: return
        pipe = self.txt2img_pipe
        
        # 1. 立即清空旧的 LoRA，防止污染显存和报错
        pipe.unload_lora_weights()
        if hasattr(pipe, 'active_lora'):
            del pipe.active_lora
            
        if not lora_name or lora_name == "无":
            return
            
        # 2. 拼接带有子文件夹的绝对路径
        lora_base_path = os.path.join(os.getcwd(), "loras", sub_dir)
        
        try:
            print(f"👉 [LoRA] 准备加载: {lora_name} (路径: {lora_base_path})")
            # 使用 adapter_name 防止多个 LoRA 互相覆盖
            pipe.load_lora_weights(lora_base_path, weight_name=lora_name, adapter_name="lora_1")
            pipe.active_lora = lora_name
        except Exception as e:
            print(f"❌ [LoRA 报错] 无法加载插件: {e}")

    def prepare_controlnet(self, control_type="openpose"):
        # 如果已经加载了该类型，直接跳过
        if self.current_cn_type == control_type and self.controlnet_pipe is not None:
            return

        print(f"🔄 正在配置 ControlNet: {control_type} ... (初次加载会自动下载)")
        
        # 预设模型库
        model_id_map = {
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth"
        }
        cn_model_id = model_id_map.get(control_type, model_id_map["openpose"])

        # 显存优化：按需加载 ControlNet
        if control_type not in self.loaded_controlnets:
            self.loaded_controlnets[control_type] = ControlNetModel.from_pretrained(
                cn_model_id, torch_dtype=torch.float16
            ).to(self.device)

        controlnet = self.loaded_controlnets[control_type]

        # 借用主模型的组件（极大节省显存）
        self.controlnet_pipe = StableDiffusionControlNetPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            controlnet=controlnet
        ).to(self.device)

        self.current_cn_type = control_type

        # 动态加载对应预处理器
        if control_type == "openpose" and not self.pose_detector:
            from controlnet_aux import OpenposeDetector
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        elif control_type == "depth" and not self.depth_estimator:
            from transformers import pipeline
            self.depth_estimator = pipeline('depth-estimation')

    def get_control_image(self, input_image, control_type="openpose"):
        if not self.current_cn_type == control_type:
            self.prepare_controlnet(control_type)
            
        if control_type == "openpose":
            return self.pose_detector(input_image)
            
        elif control_type == "canny":
            # Canny 线稿提取 (极速，零显存占用)
            image = np.array(input_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            return Image.fromarray(image)
            
        elif control_type == "depth":
            # 深度图提取
            return self.depth_estimator(input_image)['depth']

    def encode_prompt(self, prompt, negative_prompt):
        if not self.txt2img_pipe: return {}
        pipe = self.txt2img_pipe
        
        # ⚠️ 必须保留的救命代码：唤醒文本编码器到 GPU，防止静默假死！
        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            pipe.text_encoder.to(self.device)
        if self.is_sdxl and hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2:
            pipe.text_encoder_2.to(self.device)

        # 退回稳定导入
        from compel import Compel, ReturnedEmbeddingsType
        
        if self.is_sdxl:
            # 经典稳定写法（如果有黄色警告直接无视它，能出图才是王道！）
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device
            )
            prompt_embeds, pooled = compel(prompt)
            neg_embeds, neg_pooled = compel(negative_prompt)
            
            # 对齐长度防崩溃
            prompt_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, neg_embeds])
            
            return {
                "prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled,
                "negative_prompt_embeds": neg_embeds, "negative_pooled_prompt_embeds": neg_pooled
            }
        else:
            compel = Compel(
                tokenizer=pipe.tokenizer, 
                text_encoder=pipe.text_encoder,
                device=self.device
            )
            prompt_embeds = compel(prompt)
            neg_embeds = compel(negative_prompt)
            
            prompt_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, neg_embeds])
            return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": neg_embeds}