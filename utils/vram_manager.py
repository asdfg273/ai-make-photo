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
        free = info["free"]
        budget = min(total, free + torch.cuda.memory_reserved() / 1024**3)
        logger.info(f"🎮 GPU: {info['name']} "
                    f"({total:.1f}GB 总量 / {budget:.1f}GB 可支配)")

        # fp16 权重实际占用：SD1.5 ≈ 2.5GB，SDXL ≈ 5GB(UNet)
        gpu_full    = 16.0 if is_sdxl else 6.0
        gpu_sliced  = 11.0 if is_sdxl else 4.0
        offload_min = 6.5 if is_sdxl else 2.5    # UNet 5GB + 激活值余量

        if budget >= gpu_full:
            pipe.to("cuda")
            pipe.disable_attention_slicing()
            strategy = "🚀 全速模式 (全GPU)"

        elif budget >= gpu_sliced:
            pipe.to("cuda")
            pipe.disable_attention_slicing()
            VRAMManager._safe_enable(pipe, "enable_vae_slicing")
            strategy = "⚡ 标准模式 (全GPU+VAE切片)"

        elif budget >= offload_min:
            pipe.enable_model_cpu_offload()
            # 关键：offload 下 VAE 必须保持 fp16 且开切片，
            # 否则 decode 时的 dtype 转换会把 VAE 钉死在 GPU
            VRAMManager._safe_enable(pipe, "enable_vae_slicing")
            VRAMManager._safe_enable(pipe, "enable_vae_tiling")
            pipe.enable_attention_slicing()
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
        根据出图分辨率调整 VAE 策略。
        注意：本函数不再改动 offload 档位——offload 由
        apply_optimal_strategy 在加载时唯一决定，此处重复调用
        会叠加 accelerate hook，导致显存不降反升。
        """
        import torch
        if not torch.cuda.is_available():
            return "CPU 模式"

        pixels = width * height
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # 像素量参考：512x512=0.26M, 768x768=0.59M, 1024x1024=1.05M
        base = 0.55 if is_sdxl else 0.70      # SDXL 同尺寸开销更大，阈值提前
        heavy = pixels >= (base * 1_048_576)

        notes = []

        # VAE slicing：按 batch 逐张解码，无空间边界副作用，恒开
        VRAMManager._safe_enable(pipe, "enable_vae_slicing")

        # VAE tiling：只有真大图才需要，小图开了会引入 tile 接缝
        if pixels >= 1024 * 1024:
            VRAMManager._safe_enable(pipe, "enable_vae_tiling")
            notes.append("VAE 分块")
        else:
            try:
                pipe.disable_vae_tiling()
            except Exception:
                pass

        # torch 2.x SDPA 已是 O(N) 显存，attention slicing 反而退化成朴素实现
        if heavy and total < 12.0:
            try:
                pipe.disable_attention_slicing()
                notes.append("关闭 attn 切片")
            except Exception:
                pass

        tag = "大图节省" if heavy else "小图全速"
        msg = f"{tag} ({width}x{height})" + (f" → {' + '.join(notes)}" if notes else "")
        logger.info(f"🎯 分辨率策略: {msg}")

        vae = getattr(pipe, "vae", None)
        logger.info(f"🔬 VAE tiling={getattr(vae, 'use_tiling', '?')} "
                    f"slicing={getattr(vae, 'use_slicing', '?')}")
        return msg

