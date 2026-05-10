# model_manager.py
# ============================================================
#  模型管理器 — 多 Pipeline / LoRA / ControlNet / Sampler
# ============================================================

import os
import gc
import torch
import cv2
import numpy as np
from threading import Lock

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
    DDIMScheduler,
)

from controlnet_aux import OpenposeDetector
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image

from utils.system_utils import SingletonMeta


# ============================================================
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

        self.device = self._normalize_device("auto")
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        self.current_model_name = None
        self.current_lora_name  = None
        self.is_sdxl            = False

        self.txt2img_pipe       = None
        self.img2img_pipe       = None
        self.inpaint_pipe       = None
        self.controlnet_pipe    = None

        self.pose_detector      = None
        self.depth_estimator    = None
        self.loaded_controlnets = {}
        self.current_cn_type    = None

        self._compel_cache = {}
        self._model_cache  = {}

    # ------------------------------------------------------------
    #  资源枚举
    # ------------------------------------------------------------
    def get_available_models(self):
        if not os.path.exists("models"):
            return []
        return [f for f in os.listdir("models")
                if f.endswith((".safetensors", ".ckpt"))]

    def get_available_loras(self, model_type="sd1.5"):
        base_dir = os.path.join("loras", model_type)
        if not os.path.exists(base_dir):
            return ["无"]
        loras = [f for f in os.listdir(base_dir)
                 if f.endswith((".safetensors", ".ckpt", ".pt"))]
        return ["无"] + loras

    # ------------------------------------------------------------
    #  显存清理
    # ------------------------------------------------------------
    def clear_memory(self):
        self.txt2img_pipe     = None
        self.img2img_pipe     = None
        self.inpaint_pipe     = None
        self.controlnet_pipe  = None
        self.loaded_controlnets.clear()
        self._compel_cache.clear()

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ------------------------------------------------------------
    #  显存 / 性能优化 (A-2)
    # ------------------------------------------------------------
    def apply_optimizations(self, pipe, name: str = ""):
        """
        完整优化：Model CPU Offload + VAE tiling/slicing +
        Attention slicing + xformers。用于主 txt2img 管线。
        """
        applied = []

        # 1) CPU 卸载 (仅 CUDA 有效，共享组件的其他 pipe 不要重复开)
        if self.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
                applied.append("cpu_offload")
            except Exception as e:
                print(f"⚠️ enable_model_cpu_offload 失败: {e}")

        # 2) 剩余轻量优化
        for item in self._apply_light_optimizations(pipe, _return=True):
            applied.append(item)

        if applied:
            print(f"⚙️ [{name or 'pipe'}] 已启用优化: {', '.join(applied)}")
        return pipe

    def _apply_light_optimizations(self, pipe, name: str = "",
                                   _return: bool = False):
        """
        共享 components 的管线用这个，只做 VAE / Attention / xformers，
        不再重复 cpu_offload，避免冲突。
        """
        applied = []

        # VAE 分块解码 (大图关键)
        try:
            pipe.enable_vae_tiling()
            applied.append("vae_tiling")
        except Exception:
            pass

        # VAE 按 batch 切片
        try:
            pipe.enable_vae_slicing()
            applied.append("vae_slicing")
        except Exception:
            pass

        # Attention 分片 - 优先 "auto"，不支持则退到默认
        try:
            pipe.enable_attention_slicing("auto")
            applied.append("attn_slicing(auto)")
        except Exception:
            try:
                pipe.enable_attention_slicing()
                applied.append("attn_slicing")
            except Exception:
                pass

        # xformers (装了才用，PyTorch 2.x SDPA 也已是默认快通道)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            applied.append("xformers")
        except Exception:
            pass

        if _return:
            return applied
        if applied:
            print(f"⚙️ [{name or 'pipe'}] 已启用优化: {', '.join(applied)}")
        return pipe

    # ------------------------------------------------------------
    #  加载底模
    # ------------------------------------------------------------
    def load_model(self, model_name):
        # 同模型直接跳过
        if (self.current_model_name == model_name
                and self.txt2img_pipe is not None):
            print("⚡ 模型未改变，跳过加载。")
            return

        print(f"🔄 正在卸载旧模型，准备加载新模型: {model_name} ...")
        # 彻底释放旧管线
        self.clear_memory()

        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 根据文件名 + 文件大小判断是否 SDXL
        file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
        self.is_sdxl = any(k in model_name.lower()
                           for k in ["xl", "sdxl", "pony",
                                     "turbo", "lightning"])
        if not self.is_sdxl:
            self.is_sdxl = file_size_gb > 4.2

        try:
            print(f"⏳ 正在加载 {'SDXL' if self.is_sdxl else 'SD 1.5'} 模型...")

            if self.is_sdxl:
                pipe_class    = StableDiffusionXLPipeline
                img2img_class = StableDiffusionXLImg2ImgPipeline
                inpaint_class = StableDiffusionXLInpaintPipeline
            else:
                pipe_class    = StableDiffusionPipeline
                img2img_class = StableDiffusionImg2ImgPipeline
                inpaint_class = StableDiffusionInpaintPipeline

            self.txt2img_pipe = pipe_class.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None,
                low_cpu_mem_usage=True,
            )

            # 主管线全套优化 (含 cpu_offload)
            self.apply_optimizations(self.txt2img_pipe, name="txt2img")

            # 共享 components 的衍生管线 —— 轻量优化即可
            self.img2img_pipe = img2img_class(**self.txt2img_pipe.components)
            self.inpaint_pipe = inpaint_class(**self.txt2img_pipe.components)

            self._apply_light_optimizations(self.img2img_pipe, name="img2img")
            self._apply_light_optimizations(self.inpaint_pipe, name="inpaint")

            self.current_model_name = model_name
            self.current_lora_name  = None
            self._compel_cache.clear()
            print("✅ 模型加载与显存优化完成！")

        except Exception as e:
            raise Exception(f"架构加载失败: {str(e)}")

    # ------------------------------------------------------------
    #  多 LoRA 挂载
    # ------------------------------------------------------------
    def apply_multiple_loras(self, lora_list, sub_dir="sd1.5"):
        """
        现代版多 LoRA 加载机制 (极速秒切 + 独立权重)
        lora_list 格式: [("lora1.safetensors", 0.8),
                        ("lora2.safetensors", 0.5)]
        """
        # 1. 卸载之前所有的 LoRA 插件，保持底模纯净
        try:
            self.txt2img_pipe.unload_lora_weights()
            print("🧹 已清空旧的 LoRA 缓存。")
        except Exception:
            pass

        if not lora_list:
            return

        adapter_names   = []
        adapter_weights = []

        # 2. 逐个挂载
        for i, (lora_name, weight) in enumerate(lora_list):
            lora_path = os.path.join("loras", sub_dir, lora_name)
            if os.path.exists(lora_path):
                adapter_name = f"lora_slot_{i}"
                try:
                    self.txt2img_pipe.load_lora_weights(
                        os.path.dirname(lora_path),
                        weight_name=os.path.basename(lora_path),
                        adapter_name=adapter_name,
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(weight)
                    print(f"✅ 挂载插件: {lora_name} "
                          f"(独立权重: {weight})")
                except Exception as e:
                    print(f"❌ [跳过] 插件 {lora_name} 不兼容或损坏: {e}")

        # 3. 一次性激活全部 LoRA，并分配精确权重
        if adapter_names:
            try:
                self.txt2img_pipe.set_adapters(
                    adapter_names, adapter_weights=adapter_weights)
                print(f"⚡ 已激活插件通道: {adapter_names}，"
                      f"对应权重: {adapter_weights}")
            except Exception as e:
                print(f"⚠️ 激活多 LoRA 权重时异常: {e}")

    # ------------------------------------------------------------
    #  ControlNet
    # ------------------------------------------------------------
    def prepare_controlnet(self, control_type="openpose"):
        # 已加载同类型 → 直接复用
        if (self.current_cn_type == control_type
                and getattr(self, 'controlnet_pipe', None) is not None):
            return

        print(f"🔄 正在配置 ControlNet: {control_type} ... "
              f"(初次加载会自动下载)")

        model_id_map = {
            "openpose": "lllyasviel/sd-controlnet-openpose",
            "canny":    "lllyasviel/sd-controlnet-canny",
            "depth":    "lllyasviel/sd-controlnet-depth",
        }
        sdxl_model_id_map = {
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
            "canny":    "diffusers/controlnet-canny-sdxl-1.0",
            "depth":    "diffusers/controlnet-depth-sdxl-1.0",
        }

        cn_model_id = (sdxl_model_id_map[control_type]
                       if getattr(self, 'is_sdxl', False)
                       else model_id_map[control_type])

        if control_type not in self.loaded_controlnets:
            self.loaded_controlnets[control_type] = \
                ControlNetModel.from_pretrained(
                    cn_model_id, torch_dtype=self.dtype
                ).to(self.device)

        controlnet = self.loaded_controlnets[control_type]

        pipe_class = (StableDiffusionXLControlNetPipeline
                      if getattr(self, 'is_sdxl', False)
                      else StableDiffusionControlNetPipeline)

        self.controlnet_pipe = pipe_class(
            vae           = self.txt2img_pipe.vae,
            text_encoder  = self.txt2img_pipe.text_encoder,
            tokenizer     = self.txt2img_pipe.tokenizer,
            unet          = self.txt2img_pipe.unet,
            scheduler     = self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            controlnet    = controlnet,
            text_encoder_2=getattr(self.txt2img_pipe, 'text_encoder_2', None),
            tokenizer_2   =getattr(self.txt2img_pipe, 'tokenizer_2', None),
        ).to(self.device)

        # 轻量优化 (共享 components,不重复 cpu_offload)
        self._apply_light_optimizations(self.controlnet_pipe,
                                        name="controlnet")

        self.current_cn_type = control_type

        # 预处理器
        if control_type == "openpose" and not self.pose_detector:
            print("⏳ 正在加载 OpenPose 骨架提取器...")
            try:
                self.pose_detector = OpenposeDetector.from_pretrained(
                    "lllyasviel/Annotators")
            except Exception as e:
                print(f"⚠️ 加载 OpenPose 失败: {e}")

        elif control_type == "depth" and not self.depth_estimator:
            print("⏳ 正在加载 Depth 深度图提取器...")
            try:
                from transformers import pipeline
                self.depth_estimator = pipeline('depth-estimation')
            except Exception as e:
                print(f"⚠️ 加载 Depth 失败: {e}")

    def get_control_image(self, input_image, control_type="openpose"):
        if self.current_cn_type != control_type:
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

    # ------------------------------------------------------------
    #  提示词编码 (Compel,支持 SD1.5 / SDXL)
    # ------------------------------------------------------------
    def encode_prompt(self, prompt, negative_prompt):
        if not self.txt2img_pipe:
            return {}

        pipe = self.txt2img_pipe

        if hasattr(pipe, "text_encoder") and pipe.text_encoder:
            pipe.text_encoder.to(self.device)
        if (self.is_sdxl
                and hasattr(pipe, "text_encoder_2")
                and pipe.text_encoder_2):
            pipe.text_encoder_2.to(self.device)

        cache_key = (prompt, negative_prompt, self.is_sdxl)
        if cache_key in self._compel_cache:
            return self._compel_cache[cache_key]

        if self.is_sdxl:
            compel = Compel(
                tokenizer    = [pipe.tokenizer, pipe.tokenizer_2],
                text_encoder = [pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=
                    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=self.device,
            )
            prompt_embeds, pooled     = compel(prompt)
            neg_embeds,    neg_pooled = compel(negative_prompt)
            prompt_embeds, neg_embeds = \
                compel.pad_conditioning_tensors_to_same_length(
                    [prompt_embeds, neg_embeds])

            result = {
                "prompt_embeds":                prompt_embeds,
                "pooled_prompt_embeds":         pooled,
                "negative_prompt_embeds":       neg_embeds,
                "negative_pooled_prompt_embeds":neg_pooled,
            }
        else:
            compel = Compel(
                tokenizer    = pipe.tokenizer,
                text_encoder = pipe.text_encoder,
                device       = self.device,
            )
            prompt_embeds = compel(prompt)
            neg_embeds    = compel(negative_prompt)
            prompt_embeds, neg_embeds = \
                compel.pad_conditioning_tensors_to_same_length(
                    [prompt_embeds, neg_embeds])
            result = {
                "prompt_embeds":          prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
            }

        self._compel_cache[cache_key] = result
        # LRU 上限 100
        if len(self._compel_cache) > 100:
            oldest = next(iter(self._compel_cache))
            del self._compel_cache[oldest]

        return result

    # ------------------------------------------------------------
    #  采样器切换
    # ------------------------------------------------------------
    def switch_sampler(self, sampler_name):
        """根据 UI 传来的名称，切换底层的扩散调度器 (Sampler)"""
        if (not hasattr(self, 'txt2img_pipe')
                or self.txt2img_pipe is None):
            return

        config = self.txt2img_pipe.scheduler.config

        try:
            if "欧拉A" in sampler_name or "Euler a" in sampler_name:
                new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
            elif "欧拉" in sampler_name or "Euler" in sampler_name:
                new_scheduler = EulerDiscreteScheduler.from_config(config)
            elif "DPM++ 2M" in sampler_name:
                new_scheduler = DPMSolverMultistepScheduler.from_config(
                    config, use_karras_sigmas=True)
            elif "DDIM" in sampler_name:
                new_scheduler = DDIMScheduler.from_config(config)
            else:
                # 默认保底
                new_scheduler = EulerAncestralDiscreteScheduler.from_config(config)

            self.txt2img_pipe.scheduler = new_scheduler
            if getattr(self, 'img2img_pipe', None) is not None:
                self.img2img_pipe.scheduler = new_scheduler
            if getattr(self, 'inpaint_pipe', None) is not None:
                self.inpaint_pipe.scheduler = new_scheduler
            if getattr(self, 'controlnet_pipe', None) is not None:
                self.controlnet_pipe.scheduler = new_scheduler

            print(f"🔄 采样器已切换为 -> {sampler_name}")

        except Exception as e:
            print(f"⚠️ 切换采样器失败: {e}，将使用原默认采样器")

    def _normalize_device(self, raw) -> str:
        import re, torch
        s = str(raw or "").lower()
        m = re.search(r'\(([^)]+)\)', s)
        if m:
            s = m.group(1).strip()
        s = s.strip()

        if "cuda" in s or "gpu" in s or "显卡" in s:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if "mps" in s:
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if "cpu" in s:
            return "cpu"
        # auto / 自动 / 空 / 未知
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"