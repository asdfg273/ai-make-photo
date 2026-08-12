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
from compel import Compel

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
    def __init__(self):
        if getattr(self, '_initialized', False):
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
        self.compel_txt2img = None
        self.compel_img2img = None
        self.compel_controlnet = None
        # ★ IP-Adapter 状态
        self.ip_adapter_loaded   = False
        self.ip_adapter_variant  = None

        self._compel_cache = {}
        self._model_cache  = {}

    # ------------------------------------------------------------
    #  资源枚举
    # ------------------------------------------------------------
    def get_available_models(self, model_type="sd15"):
        from utils.model_scanner import scan_models
        return [m["name"] for m in scan_models(model_type)]

    def get_available_loras(self, model_type="sd1.5"):
        from utils import paths
        base_dir = os.path.join(paths.LORA_DIR, model_type)
        if not os.path.exists(base_dir):
            return ["无"]
        return ["无"] + [f for f in os.listdir(base_dir)
                         if f.endswith((".safetensors", ".ckpt", ".pt"))]


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
                if not getattr(self, 'ip_adapter_loaded', False):
                    try:
                        pipe.enable_attention_slicing()
                    except Exception:
                        pass
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
        # 同模型跳过
        if (self.current_model_name == model_name
                and self.txt2img_pipe is not None):
            print("⚡ 模型未改变，跳过加载。")
            return
    
        # === 智能查找:支持 models/ 根目录 + 子文件夹 ===
        MODELS_ROOT = "models"
        candidates = [
            os.path.join(MODELS_ROOT, model_name),                 # 根目录
            os.path.join(MODELS_ROOT, "sd15", model_name),         # 子目录
            os.path.join(MODELS_ROOT, "sdxl", model_name),
            os.path.join(MODELS_ROOT, "pony", model_name),
            os.path.join(MODELS_ROOT, "flux", model_name),
        ]
    
        # 优先取存在的路径
        model_path = None
        for c in candidates:
            if os.path.isfile(c):
                model_path = c
                break
    
        # 还找不到 → 递归扫描兜底
        if model_path is None:
            for root, _, files in os.walk(MODELS_ROOT):
                if model_name in files:
                    model_path = os.path.join(root, model_name)
                    break
    
        if model_path is None:
            raise FileNotFoundError(f"找不到模型文件: {model_name}")
    
        print(f"📂 模型路径: {model_path}", flush=True)

        # ========== 模型类型自动识别 ==========
        file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
        name_lower = model_name.lower()
    
        # SDXL 识别
        self.is_sdxl = any(k in name_lower for k in [
            "xl", "sdxl", "pony", "turbo", "lightning", "illustrious", "noobai"
        ])
        if not self.is_sdxl:
            self.is_sdxl = 4.2 < file_size_gb < 8.0
    
        # SD3 / Flux 识别（更大的现代模型）
        self.is_sd3 = "sd3" in name_lower or "stable-diffusion-3" in name_lower
        self.is_flux = "flux" in name_lower
    
        # 模型类型标签
        if self.is_flux:
            model_type = "Flux"
        elif self.is_sd3:
            model_type = "SD3"
        elif self.is_sdxl:
            model_type = "SDXL"
        else:
            model_type = "SD 1.5"
    
        print(f"📦 检测模型类型: {model_type} ({file_size_gb:.2f}GB)")

        try:
            # ========== 选择 Pipeline 类 ==========
            if self.is_flux:
                from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline
                pipe_class    = FluxPipeline
                img2img_class = FluxImg2ImgPipeline
                inpaint_class = FluxInpaintPipeline
            elif self.is_sd3:
                from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
                pipe_class    = StableDiffusion3Pipeline
                img2img_class = StableDiffusion3Img2ImgPipeline
                inpaint_class = StableDiffusion3Pipeline  # SD3 没有专门的 inpaint
            elif self.is_sdxl:
                pipe_class    = StableDiffusionXLPipeline
                img2img_class = StableDiffusionXLImg2ImgPipeline
                inpaint_class = StableDiffusionXLInpaintPipeline
            else:
                pipe_class    = StableDiffusionPipeline
                img2img_class = StableDiffusionImg2ImgPipeline
                inpaint_class = StableDiffusionInpaintPipeline

            print(f"⏳ 正在加载 {model_type} 模型...")

            # ========== 加载主 Pipeline ==========
            self.txt2img_pipe = pipe_class.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None,
                low_cpu_mem_usage=True,
            )
        
            # ========== 智能显存优化 ==========
            from utils.vram_manager import VRAMManager
            VRAMManager.apply_optimal_strategy(
                self.txt2img_pipe, 
                is_sdxl=(self.is_sdxl or self.is_sd3 or self.is_flux)
            )
            VRAMManager.print_status()

            # ========== 衍生 Pipeline（共享底层组件）==========
            try:
                self.img2img_pipe = img2img_class(**self.txt2img_pipe.components)
                self._apply_light_optimizations(self.img2img_pipe, name="img2img")
            except Exception as e:
                print(f"⚠️ img2img pipe 创建失败: {e}")
                self.img2img_pipe = None
        
            try:
                self.inpaint_pipe = inpaint_class(**self.txt2img_pipe.components)
                self._apply_light_optimizations(self.inpaint_pipe, name="inpaint")
            except Exception as e:
                print(f"⚠️ inpaint pipe 创建失败: {e}")
                self.inpaint_pipe = None

            self.current_model_name = model_name
            self.current_lora_name  = None
            self._compel_cache.clear()
            print(f"✅ {model_type} 模型加载与显存优化完成！")

            # ========== Compel 长提示词（仅 SD1.5/SDXL）==========
            if not (self.is_sd3 or self.is_flux):
                self.compel_txt2img = Compel(
                    tokenizer=self.txt2img_pipe.tokenizer,
                    text_encoder=self.txt2img_pipe.text_encoder,
                )
                if self.img2img_pipe:
                    self.compel_img2img = Compel(
                        tokenizer=self.img2img_pipe.tokenizer,
                        text_encoder=self.img2img_pipe.text_encoder,
                    )
                if getattr(self, 'controlnet_pipe', None):
                    self.compel_controlnet = Compel(
                        tokenizer=self.controlnet_pipe.tokenizer,
                        text_encoder=self.controlnet_pipe.text_encoder,
                    )
                print("✅ Compel 长提示词支持已启用", flush=True)
            else:
                # SD3/Flux 原生支持长提示词，不需要 Compel
                self.compel_txt2img = None
                self.compel_img2img = None
                print(f"✅ {model_type} 原生支持长提示词")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"模型加载失败: {str(e)}")

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
        CN_ALIAS = {
            "openpose": "openpose", "OpenPose": "openpose", "Openpose": "openpose",
            "OPENPOSE": "openpose", "姿势": "openpose", "骨架": "openpose",
            "canny": "canny", "Canny": "canny", "CANNY": "canny",
            "线稿": "canny", "边缘": "canny",
            "depth": "depth", "Depth": "depth", "DEPTH": "depth",
            "深度": "depth", "深度图": "depth",
        }

        raw_type = str(control_type).strip()
        control_type = CN_ALIAS.get(raw_type, raw_type.lower())

        # 已加载同类型 → 仍需检查 image_encoder 是否需要同步
        if (self.current_cn_type == control_type
                and getattr(self, 'controlnet_pipe', None) is not None):
            # 🔧 即便复用,也要确保 image_encoder 同步 (IP-Adapter 可能后加载)
            self._sync_ipa_components_to_controlnet()
            return

        print(f"🔄 正在配置 ControlNet: {control_type} "
              f"(原始输入: {raw_type})...")

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

        use_map = (sdxl_model_id_map
                   if getattr(self, 'is_sdxl', False)
                   else model_id_map)

        if control_type not in use_map:
            raise ValueError(
                f"❌ 未知 ControlNet 类型: '{raw_type}'\n"
                f"   归一化后: '{control_type}'\n"
                f"   可用类型: {list(use_map.keys())}"
            )

        if control_type not in self.loaded_controlnets:
            local_dir = os.path.abspath(
                f"controlnets/{'sdxl_' if getattr(self, 'is_sdxl', False) else ''}{control_type}"
            )
            if os.path.exists(os.path.join(local_dir, "config.json")):
                print(f"📂 从本地加载 ControlNet: {local_dir}")
                cn_source = local_dir
                local_only = True
            else:
                cn_model_id = use_map[control_type]
                print(f"🌐 从在线下载 ControlNet: {cn_model_id} "
                      f"(via {os.environ.get('HF_ENDPOINT', 'huggingface.co')})")
                cn_source = cn_model_id
                local_only = False

            try:
                self.loaded_controlnets[control_type] = \
                    ControlNetModel.from_pretrained(
                        cn_source,
                        torch_dtype=self.dtype,
                        local_files_only=local_only,
                    ).to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"❌ ControlNet 加载失败: {e}\n\n"
                    f"💡 解决办法:\n"
                    f"   1. 检查网络是否通畅,镜像 https://hf-mirror.com 能否访问\n"
                    f"   2. 或手动下载模型放到: {local_dir}\n"
                    f"      只需 config.json + diffusion_pytorch_model.safetensors\n"
                    f"   3. 或在 UI 中关闭 ControlNet,改用普通图生图"
                ) from e

        controlnet = self.loaded_controlnets[control_type]

        pipe_class = (StableDiffusionXLControlNetPipeline
                      if getattr(self, 'is_sdxl', False)
                      else StableDiffusionControlNetPipeline)

        common_kwargs = dict(
            vae          = self.txt2img_pipe.vae,
            text_encoder = self.txt2img_pipe.text_encoder,
            tokenizer    = self.txt2img_pipe.tokenizer,
            unet         = self.txt2img_pipe.unet,
            scheduler    = self.txt2img_pipe.scheduler,
            controlnet   = controlnet,
        )

        if getattr(self, 'is_sdxl', False):
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                text_encoder_2 = self.txt2img_pipe.text_encoder_2,
                tokenizer_2    = self.txt2img_pipe.tokenizer_2,
            ).to(self.device)
        else:
            self.controlnet_pipe = pipe_class(
                **common_kwargs,
                safety_checker          = None,
                feature_extractor       = None,
                requires_safety_checker = False,
            ).to(self.device)

        # 轻量优化
        self._apply_light_optimizations(self.controlnet_pipe, name="controlnet")

        self.current_cn_type = control_type
        print(f"✅ ControlNet ({control_type}) 加载完成")

        self._sync_ipa_components_to_controlnet()

        # ===== 预处理器(检测器) =====
        if control_type == "openpose" and not self.pose_detector:
            print("⏳ 正在加载 OpenPose 骨架提取器...")
            local_annot = os.path.abspath("controlnets/Annotators")
            if os.path.exists(os.path.join(local_annot, "body_pose_model.pth")):
                print(f"📂 从本地加载 OpenPose 检测器: {local_annot}")
                try:
                    self.pose_detector = OpenposeDetector.from_pretrained(local_annot)
                except Exception as e:
                    print(f"⚠️ 本地 OpenPose 加载失败: {e}")
            else:
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

    def _sync_ipa_components_to_controlnet(self):
        """
        把 txt2img_pipe 上的 IP-Adapter 组件 (image_encoder/feature_extractor)
        同步给 controlnet_pipe。
    
        IP-Adapter 加载在 txt2img_pipe 上时,会注入 image_encoder。
        controlnet_pipe 共享了 unet (含 attn processor),但 image_encoder 是
        pipeline 级别的属性,不会自动共享 → 需要手动赋值。
        """
        if getattr(self, 'controlnet_pipe', None) is None:
            return
        if getattr(self, 'txt2img_pipe', None) is None:
            return

        src = self.txt2img_pipe
        dst = self.controlnet_pipe

        # 1. image_encoder (IP-Adapter 的视觉编码器)
        src_img_enc = getattr(src, 'image_encoder', None)
        if src_img_enc is not None:
            dst.image_encoder = src_img_enc
            print("🔧 [sync] image_encoder → controlnet_pipe", flush=True)

        # 2. feature_extractor (CLIP 图像预处理器)
        src_feat = getattr(src, 'feature_extractor', None)
        if src_feat is not None:
            dst.feature_extractor = src_feat
            print("🔧 [sync] feature_extractor → controlnet_pipe", flush=True)

        # 3. 同步 IP-Adapter scale (UNet 共享,所以会自动生效,这里只为保险)
        try:
            if getattr(self, 'ip_adapter_loaded', False):
                current_scale = getattr(self, 'current_ipa_scale', 0.6)
                dst.set_ip_adapter_scale(current_scale)
                print(f"🔧 [sync] IPA scale={current_scale} → controlnet_pipe",
                      flush=True)
        except Exception as e:
            print(f"⚠️ [sync] 设置 controlnet_pipe IPA scale 失败: {e}", flush=True)


    def get_control_image(self, input_image, control_type="openpose"):
        control_type = str(control_type).strip().lower()
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

    def prepare_ip_adapter(self, variant="plus"):
        """
        加载 IP-Adapter 到 txt2img / img2img 管线
        🔧 inpaint_pipe 不装载 IPA,避免 ADetailer 局部重绘冲突
           (ADetailer 只需修手/脸细节,不需要参考角色特征)
        """
        if getattr(self, 'ip_adapter_loaded', False) \
           and getattr(self, 'ip_adapter_variant', None) == variant:
            print("✅ IP-Adapter 已加载,复用", flush=True)
            return True

        weight_name = (
            "ip-adapter-plus_sd15.safetensors"
            if variant == "plus"
            else "ip-adapter_sd15.safetensors"
        )
        print(f"🎭 加载 IP-Adapter ({variant}) ...", flush=True)

        pipes = []
        for attr in ("txt2img_pipe", "img2img_pipe"):   
            pipe = getattr(self, attr, None)
            if pipe is not None:
                pipes.append((attr, pipe))

        # ── 重置 attn processor (只对装载 IPA 的 pipe) ──
        print("  → 重置 attn processor (关闭 attention slicing) ...", flush=True)
        try:
            from diffusers.models.attention_processor import (
                AttnProcessor2_0, AttnProcessor
            )
            import torch.nn.functional as F
            proc_class = (
                AttnProcessor2_0
                if hasattr(F, 'scaled_dot_product_attention')
                else AttnProcessor
            )
            for attr, pipe in pipes:
                try:
                    if hasattr(pipe, 'disable_attention_slicing'):
                        pipe.disable_attention_slicing()
                except Exception:
                    pass
                try:
                    pipe.unet.set_attn_processor(proc_class())
                except Exception as e:
                    print(f"    ⚠️ {attr} 重置 processor 失败: {e}", flush=True)
        except Exception as e:
            print(f"  ⚠️ 重置 processor 异常: {e}", flush=True)

        # ── 装载 IP-Adapter (仅 txt2img / img2img) ──
        try:
            for attr, pipe in pipes:
                print(f"  → 装载 IP-Adapter 到 {attr} ...", flush=True)
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=weight_name,
                )

            # 🆕 inpaint_pipe 只同步 encoder/feature_extractor,不装载 IPA
            # 这样它能正常推理,且 ADetailer 不会触发 IPA 冲突
            inpaint_pipe = getattr(self, 'inpaint_pipe', None)
            if inpaint_pipe is not None:
                try:
                    # 同步基础组件 (复用显存)
                    if getattr(self.txt2img_pipe, 'image_encoder', None) is not None:
                        inpaint_pipe.image_encoder = self.txt2img_pipe.image_encoder
                    if getattr(self.txt2img_pipe, 'feature_extractor', None) is not None:
                        inpaint_pipe.feature_extractor = self.txt2img_pipe.feature_extractor

                    from diffusers.models.attention_processor import (
                        AttnProcessor2_0, AttnProcessor
                    )
                    import torch.nn.functional as F
                    proc_class = (
                        AttnProcessor2_0
                        if hasattr(F, 'scaled_dot_product_attention')
                        else AttnProcessor
                    )

                    # 检查 inpaint 是否和 txt2img 共享 UNet
                    shares_unet = (inpaint_pipe.unet is self.txt2img_pipe.unet)
                    if shares_unet:
                        print("  ⚠️ inpaint_pipe 与 txt2img 共享 UNet,"
                              "ADetailer 时需在调用层处理 IPA 占位", flush=True)
                    else:
                        # 独立 UNet,直接重置回干净的 processor
                        inpaint_pipe.unet.set_attn_processor(proc_class())
                        print("  ✅ inpaint_pipe 已重置为标准 attn_processor "
                              "(无 IPA 干扰)", flush=True)
                except Exception as e:
                    print(f"  ⚠️ inpaint_pipe 同步失败: {e}", flush=True)

            # 🆕 设置所有状态标志(统一两套变量名)
            self.ip_adapter_loaded  = True
            self.ip_adapter_variant = variant
            self._ipa_loaded        = True
            self._ipa_variant       = variant
            self._ipa_scale         = 0.7

            print("✅ IP-Adapter 加载完成 (inpaint_pipe 已排除)", flush=True)

            # 🆕 如果 controlnet_pipe 已存在,自动同步
            if getattr(self, 'controlnet_pipe', None) is not None:
                try:
                    self.sync_ipa_to_controlnet()
                except Exception as e:
                    print(f"⚠️ 自动同步 IPA 到 controlnet 失败: {e}", flush=True)

            return True

        except Exception as e:
            print(f"❌ IP-Adapter 加载失败: {e}", flush=True)
            import traceback
            traceback.print_exc()

            try:
                self.unload_ip_adapter()
            except Exception as e2:
                print(f"⚠️ 清理失败: {e2}", flush=True)

            self.ip_adapter_loaded  = False
            self.ip_adapter_variant = None
            self._ipa_loaded        = False
            self._ipa_variant       = None
            return False


    def unload_ip_adapter(self):
        """彻底卸载 IP-Adapter，恢复 UNet 到普通模式（覆盖全部 4 个 pipeline）"""
        print("🧹 卸载 IP-Adapter ...", flush=True)
        pipes = [
            self.txt2img_pipe,
            self.img2img_pipe,
            self.inpaint_pipe,
            self.controlnet_pipe,
        ]
        for pipe in pipes:
            if pipe is None:
                continue
            try:
                # 方式1: diffusers 0.31+ 提供的官方接口
                if hasattr(pipe, 'unload_ip_adapter'):
                    pipe.unload_ip_adapter()
                else:
                    # 方式2: 手动恢复 attn processor（兼容旧版 diffusers）
                    from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
                    proc_class = AttnProcessor2_0 if hasattr(
                        __import__('torch.nn.functional', fromlist=['scaled_dot_product_attention']),
                        'scaled_dot_product_attention') else AttnProcessor
                    pipe.unet.set_attn_processor(proc_class())
                    pipe.unet.encoder_hid_proj = None
                    pipe.unet.config.encoder_hid_dim_type = None
            except Exception as e:
                print(f"  ⚠️ 卸载异常: {e}", flush=True)

        self.ip_adapter_loaded  = False
        self.ip_adapter_variant = None
        self._ipa_loaded        = False
        self._ipa_variant       = None
        print("✅ IP-Adapter 卸载完成", flush=True)


    def set_ip_adapter_scale(self, scale: float = 0.6):
        """调整 IP-Adapter 影响强度 0.0~1.5（覆盖全部 4 个 pipeline）"""
        if not self.ip_adapter_loaded:
            return
        self._ipa_scale = float(scale)
        scale = max(0.0, min(1.5, float(scale)))
        targets = [
            self.txt2img_pipe, self.img2img_pipe,
            self.inpaint_pipe, self.controlnet_pipe,
        ]
        for pipe in targets:
            if pipe is None:
                continue
            try:
                if hasattr(pipe, 'set_ip_adapter_scale'):
                    pipe.set_ip_adapter_scale(scale)
            except Exception:
                pass
        print(f"🎛️ IP-Adapter scale = {scale}", flush=True)

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

    def unload_all(self):
        """彻底释放所有管线和内存"""
        print("🧹 开始释放所有模型...")

        # 释放管线
        for attr in ['txt2img_pipe', 'img2img_pipe', 'inpaint_pipe',
                     'controlnet_pipe']:
            pipe = getattr(self, attr, None)
            if pipe is not None:
                try:
                    # 解绑组件,加速 GC
                    for sub in ['unet', 'vae', 'text_encoder',
                                'text_encoder_2', 'tokenizer',
                                'tokenizer_2', 'scheduler', 'safety_checker',
                                'feature_extractor']:
                        if hasattr(pipe, sub):
                            try:
                                setattr(pipe, sub, None)
                            except Exception:
                                pass
                    del pipe
                except Exception:
                    pass
                setattr(self, attr, None)

        # 释放 ControlNet
        if hasattr(self, 'loaded_controlnets'):
            for k in list(self.loaded_controlnets.keys()):
                try:
                    del self.loaded_controlnets[k]
                except Exception:
                    pass
            self.loaded_controlnets = {}

        # 释放检测器
        for attr in ['pose_detector', 'depth_estimator']:
            if getattr(self, attr, None) is not None:
                try:
                    del self.__dict__[attr]
                except Exception:
                    pass
                setattr(self, attr, None)

        # 释放 IP-Adapter 参考图
        if hasattr(self, 'ipa_ref_image'):
            self.ipa_ref_image = None

        # 释放缓存
        if hasattr(self, '_compel_cache'):
            self._compel_cache = {}
        if hasattr(self, '_model_cache'):
            self._model_cache = {}

        self.current_model_name = None
        self.current_lora_name = None
        self.current_cn_type = None

        # 强制 GC
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✅ 所有模型已释放")

    def sync_ipa_to_controlnet(self):
        """
        给 controlnet_pipe 独立装载 IP-Adapter（不依赖共享 UNet）。
        """
        if self.controlnet_pipe is None:
            print("⚠️ controlnet_pipe 未初始化", flush=True)
            return False

        if self.txt2img_pipe is None or not getattr(self, '_ipa_loaded', False):
            print("⚠️ txt2img_pipe 未加载 IPA，无法同步", flush=True)
            return False

        print("🔄 给 controlnet_pipe 独立装载 IP-Adapter...", flush=True)

        try:
            # ── 1. 关闭 attention slicing（IPA 不兼容） ──
            try:
                self.controlnet_pipe.disable_attention_slicing()
                print("  → 关闭 controlnet_pipe attention_slicing", flush=True)
            except Exception:
                pass

            # ── 2. 共享 image_encoder 和 feature_extractor（节省显存） ──
            if hasattr(self.txt2img_pipe, 'image_encoder') and self.txt2img_pipe.image_encoder is not None:
                self.controlnet_pipe.image_encoder = self.txt2img_pipe.image_encoder
                print("  → 共享 image_encoder", flush=True)

            if hasattr(self.txt2img_pipe, 'feature_extractor') and self.txt2img_pipe.feature_extractor is not None:
                self.controlnet_pipe.feature_extractor = self.txt2img_pipe.feature_extractor
                print("  → 共享 feature_extractor", flush=True)

            # ── 3. 关键：独立装载 IP-Adapter 到 controlnet_pipe ──
            variant = getattr(self, '_ipa_variant', 'plus')
            weight_map = {
                'plus':      'ip-adapter-plus_sd15.safetensors',
                'plus-face': 'ip-adapter-plus-face_sd15.safetensors',
                'base':      'ip-adapter_sd15.safetensors',
            }
            weight_name = weight_map.get(variant, 'ip-adapter-plus_sd15.safetensors')

            # 优先用本地缓存
            local_dir = os.path.join("models_cache", "ip_adapter")
            if os.path.exists(os.path.join(local_dir, "models", weight_name)):
                print(f"  → 从本地装载: {weight_name}", flush=True)
                self.controlnet_pipe.load_ip_adapter(
                    local_dir,
                    subfolder="models",
                    weight_name=weight_name,
                )
            else:
                print(f"  → 从 HuggingFace 装载: {weight_name}", flush=True)
                self.controlnet_pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=weight_name,
                )

            # ── 4. 设置 scale ──
            scale = getattr(self, '_ipa_scale', 0.7)
            self.controlnet_pipe.set_ip_adapter_scale(scale)
            print(f"  → set scale = {scale}", flush=True)

            # ── 5. 验证 ──
            ok_count = 0
            err_count = 0
            for name, proc in self.controlnet_pipe.unet.attn_processors.items():
                if 'attn2' in name:
                    if 'IPAdapter' in type(proc).__name__:
                        ok_count += 1
                    else:
                        err_count += 1

            print(f"✅ ControlNet IPA 装载完成: {ok_count} 正确 / {err_count} 错误", flush=True)
            return ok_count > 0 and err_count == 0

        except Exception as e:
            print(f"❌ IPA 装载失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _enable_memory_efficient(self, pipe, name="pipe"):
        """统一启用 VAE 分块解码 + 切片，省显存出大图"""
        if pipe is None:
            return
        try:
            pipe.enable_vae_tiling()
            print(f"  ✅ [{name}] VAE Tiling 已启用（支持大图）")
        except Exception as e:
            print(f"  ⚠️ [{name}] VAE Tiling 启用失败: {e}")
        try:
            pipe.enable_vae_slicing()
            print(f"  ✅ [{name}] VAE Slicing 已启用（省显存）")
        except Exception as e:
            print(f"  ⚠️ [{name}] VAE Slicing 启用失败: {e}")
        # xformers 加速（如果装了）
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print(f"  ✅ [{name}] xformers 加速已启用")
        except Exception:
            pass  # 没装就算了

    def prepare_reference_only(self):
        """加载 Reference-Only ControlNet (无需额外模型)"""
        # Reference-Only 不需要下载额外模型
        # 它通过修改 attention 实现,集成在 diffusers 的 community pipeline 里
    
        from diffusers import StableDiffusionReferencePipeline
    
        self.reference_pipe = StableDiffusionReferencePipeline(
            unet=self.txt2img_pipe.unet,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.reference_pipe.to(self.device)
