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
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    EulerAncestralDiscreteScheduler,    
    EulerDiscreteScheduler,            
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from threading import Lock
from PIL import Image
from utils.system_utils import SingletonMeta

class ModelManager(metaclass=SingletonMeta):
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.current_model_name = None
        self.current_lora_name = None
        self.is_sdxl = False  
        
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.controlnet_pipe = None
        self.pose_detector = None
        self.depth_estimator = None
        self.loaded_controlnets = {}
        self.current_cn_type = None
        
        self._compel_cache = {}
        self._model_cache = {}

    def get_available_models(self):
        if not os.path.exists("models"):
            return []
        return [f for f in os.listdir("models") if f.endswith((".safetensors", ".ckpt"))]

    def get_available_loras(self, model_type="sd1.5"):
        base_dir = os.path.join("loras", model_type)
        if not os.path.exists(base_dir):
            return ["无"]
        loras = [f for f in os.listdir(base_dir) if f.endswith((".safetensors", ".ckpt", ".pt"))]
        return ["无"] + loras

    def refresh_lora_by_model(self):
        current_model = self.combo_model.get().lower()
        if not current_model: return
        is_sdxl = "xl" in current_model 
        model_type = "sdxl" if is_sdxl else "sd1.5"
    
        lora_list = self.ai.get_available_loras(model_type)
        self.combo_lora.config(values=lora_list)
        self.combo_lora.set("无")
        self.lbl_lora_info.config(text="LoRA说明: (未选择)")

    def clear_memory(self):
        # 1. 斩断所有引用
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.controlnet_pipe = None
        self.loaded_controlnets.clear() # 清空已加载的 ControlNet
        self._compel_cache.clear()      # 清空提示词缓存
        
        # 2. 强制 Python 垃圾回收
        gc.collect()
        
        # 3. 强制清空显卡底层的碎片
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  

    def apply_optimizations(self, pipe):
        if self.device != "cuda":
            return
            
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pipe.enable_attention_slicing()
            
        pipe.enable_model_cpu_offload()
        
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    def load_model(self, model_name):
        if getattr(self, 'current_model_name', None) == model_name:
            print("⚡ 模型未改变，极速跳过加载底层模型。")
            return
            
        print(f"🔄 开始加载全新大模型: {model_name} ...")
        if model_name == self.current_model_name:
            return
            
        model_path = os.path.join("models", model_name) 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        self.clear_memory()

        file_size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
        self.is_sdxl = any(keyword in model_name.lower() for keyword in ["xl", "sdxl"])
        if not self.is_sdxl:
            self.is_sdxl = file_size_gb > 5.0  
        
        try:
            print(f"正在加载 {'SDXL' if self.is_sdxl else 'SD 1.5'} 模型...")
            
            if self.is_sdxl:
                pipe_class = StableDiffusionXLPipeline
                img2img_class = StableDiffusionXLImg2ImgPipeline
                inpaint_class = StableDiffusionXLInpaintPipeline
            else:
                pipe_class = StableDiffusionPipeline
                img2img_class = StableDiffusionImg2ImgPipeline
                inpaint_class = StableDiffusionInpaintPipeline

            self.txt2img_pipe = pipe_class.from_single_file(
                model_path, 
                torch_dtype=self.dtype, 
                use_safetensors=True,
                safety_checker=None,
                low_cpu_mem_usage=True  # 🌟 核心魔法：阻止一口气吞掉所有内存！
            )
            
            self.apply_optimizations(self.txt2img_pipe)
            
            self.img2img_pipe = img2img_class(**self.txt2img_pipe.components)
            self.inpaint_pipe = inpaint_class(**self.txt2img_pipe.components)
            
            self.apply_optimizations(self.img2img_pipe)
            self.apply_optimizations(self.inpaint_pipe)
            
            self.current_model_name = model_name
            self.current_lora_name = None
            self._compel_cache.clear()
            print("模型加载与内存优化完成！")
            self.current_model_name = model_name
            
        except Exception as e:
            raise Exception(f"架构加载失败: {str(e)}")

    def apply_multiple_loras(self, lora_list, sub_dir="sd1.5"):
        """
        现代版多 LoRA 加载机制 (极速秒切 + 独立权重)
        lora_list 格式: [("lora1.safetensors", 0.8), ("lora2.safetensors", 0.5)]
        """
        # 1. 卸载之前所有的 LoRA 插件，保持底模纯净
        try:
            self.txt2img_pipe.unload_lora_weights()
            print("🧹 已清空旧的 LoRA 缓存。")
        except:
            pass

        if not lora_list:
            return

        adapter_names = []
        adapter_weights = []

        # 2. 像插卡带一样，逐个挂载 LoRA
        for i, (lora_name, weight) in enumerate(lora_list):
            lora_path = os.path.join("loras", sub_dir, lora_name)
            if os.path.exists(lora_path):
                adapter_name = f"lora_slot_{i}" # 给每个槽位起个名字
                try:
                    # 只挂载，不硬融合，速度极快
                    self.txt2img_pipe.load_lora_weights(
                        os.path.dirname(lora_path), 
                        weight_name=os.path.basename(lora_path), 
                        adapter_name=adapter_name
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(weight) # 👈 这里完美读取了你界面上滑块设置的独立权重！
                    print(f"✅ 挂载插件: {lora_name} (独立权重: {weight})")
                except Exception as e:
                    print(f"❌ [跳过] 插件 {lora_name} 不兼容或损坏: {e}")

        # 3. 一次性激活所有挂载的 LoRA，并精确分配权重！
        if adapter_names:
            try:
                self.txt2img_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                print(f"⚡ 闪电启动完成！已激活插件通道: {adapter_names}，对应权重: {adapter_weights}")
            except Exception as e:
                print(f"⚠️ 激活多 LoRA 权重时发生异常: {e}")

    def prepare_controlnet(self, control_type="openpose"):
        # 1. 检查是否已经加载了同类型的 ControlNet，避免重复加载
        if self.current_cn_type == control_type and getattr(self, 'controlnet_pipe', None) is not None:
            return
            
        print(f"🔄 正在配置 ControlNet: {control_type} ... (初次加载会自动下载网络与预处理器)")
        
        # 2. 模型 ID 映射 (兼容 SD1.5 和 SDXL)
        model_id_map = {
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "canny": "lllyasviel/sd-controlnet-canny",
            "depth": "lllyasviel/sd-controlnet-depth"
        }
        sdxl_model_id_map = {
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
            "canny": "diffusers/controlnet-canny-sdxl-1.0",
            "depth": "diffusers/controlnet-depth-sdxl-1.0"
        }
        
        # 根据当前主模型是否是 SDXL 来选择对应的 ControlNet
        cn_model_id = sdxl_model_id_map[control_type] if getattr(self, 'is_sdxl', False) else model_id_map[control_type]
        
        # 3. 加载 ControlNet 权重
        if control_type not in self.loaded_controlnets:
            self.loaded_controlnets[control_type] = ControlNetModel.from_pretrained(
                cn_model_id, torch_dtype=self.dtype
            ).to(self.device)
            
        controlnet = self.loaded_controlnets[control_type]
        
        # 4. 动态构建 Pipeline (支持 SDXL)
        pipe_class = StableDiffusionXLControlNetPipeline if getattr(self, 'is_sdxl', False) else StableDiffusionControlNetPipeline
        
        self.controlnet_pipe = pipe_class(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            controlnet=controlnet,
            # SDXL 额外参数 (若不是 SDXL, getattr 会返回 None, 兼容性极强)
            text_encoder_2=getattr(self.txt2img_pipe, 'text_encoder_2', None),
            tokenizer_2=getattr(self.txt2img_pipe, 'tokenizer_2', None)
        ).to(self.device)
        
        # 开启显存优化
        if hasattr(self, 'apply_optimizations'):
            self.apply_optimizations(self.controlnet_pipe)
            
        self.current_cn_type = control_type
        
        if control_type == "openpose" and not getattr(self, 'pose_detector', None):
            print("⏳ 正在加载 OpenPose 骨架提取器...")
            try:
                from controlnet_aux import OpenposeDetector
                self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            except ImportError:
                print("⚠️ 缺少 controlnet_aux 库，请运行: pip install controlnet_aux")
                
        elif control_type == "depth" and not getattr(self, 'depth_estimator', None):
            print("⏳ 正在加载 Depth 深度图提取器...")
            try:
                from transformers import pipeline
                self.depth_estimator = pipeline('depth-estimation')
            except ImportError:
                print("⚠️ 缺少 transformers 相关依赖。")

    def get_control_image(self, input_image, control_type="openpose"):
        if not self.current_cn_type == control_type:
            self.prepare_controlnet(control_type)
            
        if control_type == "openpose":
            return self.pose_detector(input_image)
        elif control_type == "canny":
            image = np.array(input_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            return Image.fromarray(image)
        elif control_type == "depth":
            return self.depth_estimator(input_image)['depth']
        return input_image

    def encode_prompt(self, prompt, negative_prompt):
        if not self.txt2img_pipe:
            return {}
        pipe = self.txt2img_pipe
        
        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            pipe.text_encoder.to(self.device)
        if self.is_sdxl and hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2:
            pipe.text_encoder_2.to(self.device)

        cache_key = (prompt, negative_prompt, self.is_sdxl)
        if cache_key in self._compel_cache:
            return self._compel_cache[cache_key]
        
        if self.is_sdxl:
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device
            )
            prompt_embeds, pooled = compel(prompt)
            neg_embeds, neg_pooled = compel(negative_prompt)
            prompt_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, neg_embeds])
            
            result = {
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
            result = {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": neg_embeds}
        
        self._compel_cache[cache_key] = result
        if len(self._compel_cache) > 100:
            oldest = next(iter(self._compel_cache))
            del self._compel_cache[oldest]
            
        return result

    def switch_sampler(self, sampler_name):
            """动态切换底层采样器(Scheduler)"""
            if not hasattr(self, 'txt2img_pipe') or self.txt2img_pipe is None:
                return
            
            # 获取当前模型的底层配置参数
            config = self.txt2img_pipe.scheduler.config
        
            try:
                if sampler_name == "Euler a":
                    new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
                elif sampler_name == "Euler":
                    new_scheduler = EulerDiscreteScheduler.from_config(config)
                elif sampler_name == "DPM++ 2M Karras":
                    new_scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
                else:
                    return # 保持模型默认，不切换

                # 将新算法应用到所有管线
                self.txt2img_pipe.scheduler = new_scheduler
                if hasattr(self, 'img2img_pipe') and self.img2img_pipe:
                    self.img2img_pipe.scheduler = new_scheduler
                if hasattr(self, 'inpaint_pipe') and self.inpaint_pipe:
                    self.inpaint_pipe.scheduler = new_scheduler
                
                print(f"⚙️ 采样器算法已无缝切换为: {sampler_name}")
            except Exception as e:
                print(f"⚠️ 采样器切换失败: {e}")

    def switch_sampler(self, sampler_name):
            """根据 UI 传来的名称，切换底层的扩散调度器 (Sampler)"""
            if not hasattr(self, 'txt2img_pipe') or self.txt2img_pipe is None:
                return
            
            # 获取当前模型的 config，确保兼容性
            config = self.txt2img_pipe.scheduler.config
        
            try:
                if "欧拉A" in sampler_name or "Euler a" in sampler_name:
                    new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
                elif "欧拉" in sampler_name or "Euler" in sampler_name:
                    new_scheduler = EulerDiscreteScheduler.from_config(config)
                elif "DPM++ 2M" in sampler_name:
                    new_scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
                elif "DDIM" in sampler_name:
                    new_scheduler = DDIMScheduler.from_config(config)
                else:
                    # 默认保底
                    new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
                
                # 把新采样器挂载到所有的管道上
                self.txt2img_pipe.scheduler = new_scheduler
                if hasattr(self, 'img2img_pipe') and self.img2img_pipe is not None:
                    self.img2img_pipe.scheduler = new_scheduler
                if hasattr(self, 'inpaint_pipe') and self.inpaint_pipe is not None:
                    self.inpaint_pipe.scheduler = new_scheduler
                
                print(f"🔄 引擎: 采样器已成功切换为 -> {sampler_name}")
            
            except Exception as e:
                print(f"⚠️ 切换采样器失败: {e}，将使用原默认采样器")