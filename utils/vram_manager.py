# utils/vram_manager.py
import torch
import logging
logger = logging.getLogger(__name__)

class VRAMManager:
    """智能显存管理 —— 根据可用显存自动选择最优策略"""

    @staticmethod
    def _safe_enable(pipe, name: str) -> bool:
        """兼容 diffusers <0.40 的 pipe.enable_vae_xxx() 与 >=0.40 的 pipe.vae.enable_xxx()"""
        vae = getattr(pipe, "vae", None)
        short = name.replace("enable_vae_", "enable_")   # enable_vae_slicing -> enable_slicing
        if vae is not None and hasattr(vae, short):
            getattr(vae, short)()
            return True
        if hasattr(pipe, name):
            getattr(pipe, name)()
            return True
        logger.debug(f"⏭️ {name} 不可用，跳过")
        return False
    
    @staticmethod
    def get_vram_info():
        """获取显存信息（GB）"""
        if not torch.cuda.is_available():
            return {"total": 0, "free": 0, "used": 0, "has_cuda": False}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free, _ = torch.cuda.mem_get_info(0)
        free_gb = free / 1024**3
        used = total - free_gb
        return {
            "total": total,
            "free": free_gb,
            "used": used,
            "has_cuda": True,
            "name": torch.cuda.get_device_name(0)
        }
    
    @staticmethod
    def apply_optimal_strategy(pipe, is_sdxl=False):
        """根据显存自动应用最优策略"""
        info = VRAMManager.get_vram_info()
        
        if not info["has_cuda"]:
            logger.info("📌 CPU 模式")
            pipe.to("cpu")
            return "cpu"
        
        total = info["total"]
        logger.info(f"🎮 GPU: {info['name']} ({total:.1f}GB)")

        # fp16 权重实际占用：SD1.5 ≈ 2.5GB，SDXL ≈ 7GB
        # 阈值 = 权重 + 激活值余量
        gpu_full   = 16.0 if is_sdxl else 6.0   # 全量驻留，无需切片
        gpu_sliced = 11.0 if is_sdxl else 4.0   #驻留但开切片省激活值
        offload_min = 5.0 if is_sdxl else 2.5   # 权重放不下，按层搬运

        if total >= gpu_full:
            pipe.to("cuda")
            pipe.disable_attention_slicing()   # 清掉可能残留的切片
            strategy = "🚀 全速模式 (全GPU)"
        elif total >= gpu_sliced:
            pipe.to("cuda")
            pipe.disable_attention_slicing()            
            VRAMManager._safe_enable(pipe, "enable_vae_slicing") 
            strategy = "⚡ 标准模式 (全GPU+VAE切片)"
        elif total >= offload_min:
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            if is_sdxl:
                VRAMManager._safe_enable(pipe, "enable_vae_slicing")
            strategy = "💾 节能模式 (CPU Offload)"
        else:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing()
            strategy = "🐢 极限模式 (Sequential Offload)"
        
        logger.info(f"✅ 策略: {strategy}")
        return strategy
    
    @staticmethod
    def cleanup():
        """手动清理显存"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("🧹 显存已清理")
    
    @staticmethod
    def print_status():
        """打印当前显存状态"""
        info = VRAMManager.get_vram_info()
        if info["has_cuda"]:
            logger.info(
                f"📊 显存: {info['used']:.2f}/{info['total']:.2f}GB "
                f"(剩 {info['free']:.2f}GB)"
            )
            alloc    = torch.cuda.memory_allocated()  / 1024**3
            reserved = torch.cuda.memory_reserved()   / 1024**3
            logger.info(f"🔬 allocated {alloc:.2f}GB / reserved {reserved:.2f}GB")

    @staticmethod
    def tune_for_resolution(pipe, width, height, is_sdxl=False):
        """
        根据实际出图分辨率二次调整显存策略。
        apply_optimal_strategy 在加载时只看模型类型，
        此处补上分辨率这一维度。
        """
        import torch
        if not torch.cuda.is_available():
            return "CPU 模式"

        pixels = width * height
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # 像素量阈值：512x512=0.26M, 768x768=0.59M, 1024x1024=1.05M
        base = 0.55 if is_sdxl else 0.70      # SDXL 同尺寸开销更大，阈值提前
        heavy = pixels >= (base * 1_048_576)

        notes = []
        try:
            pipe.disable_vae_tiling()
        except Exception:
            pass
        try:
            VRAMManager._safe_enable(pipe, "enable_vae_slicing")      # 按 batch 逐张，无副作用
        except Exception:
            pass
        if heavy and total < 12.0:
            # torch 2.x SDPA 已是 O(N) 显存，切片反而退化成朴素实现
            try:
                pipe.disable_attention_slicing()
            except Exception:
                pass
            try:
                vae = getattr(pipe, "vae", None)
                pixels = width * height          # 用 tune_for_resolution already 拿到的宽高
                if pixels >= 1024 * 1024:
                    VRAMManager._safe_enable(pipe, "enable_vae_tiling")
                else:
                    try:
                        pipe.disable_vae_tiling()
                    except Exception:
                        pass
                VRAMManager._safe_enable(pipe, "enable_vae_slicing")       # slicing 按 batch 切，不产生空间 tile 边界
                notes.append("VAE 分块+切片")
            except Exception:
                pass
            # 实时检查剩余显存；1024 的 UNet 激活约需 3-4 GB
            # 不够就切 CPU offload，避免换页
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            pixels_m = pixels / 1_048_576          # 单位：百万像素
            est_activation = pixels_m * 3.2        # 1M 像素 ≈ 3.2 GB 激活？
            if free_gb < est_activation:
                try:
                    pipe.enable_model_cpu_offload()
                    notes.append("CPU offload (显存不足)")
                except Exception:
                    pass

        tag = "大图节省" if heavy else "小图全速"
        msg = f"{tag} ({width}x{height})" + (f" → {' + '.join(notes)}" if notes else "")
        print(f"🎯 分辨率策略: {msg}")
        vae = getattr(pipe, "vae", None)
        logger.info(f"🔬 VAE tiling={getattr(vae, 'use_tiling', '?')} "
                    f"slicing={getattr(vae, 'use_slicing', '?')} ")
        return msg
