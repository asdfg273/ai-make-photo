# utils/vram_manager.py
import torch
import logging
logger = logging.getLogger(__name__)

class VRAMManager:
    """智能显存管理 —— 根据可用显存自动选择最优策略"""
    
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
        
        # 策略选择
        if total >= 16:
            # 高端卡：直接 GPU，全速
            pipe.to("cuda")
            pipe.enable_vae_tiling()
            strategy = "🚀 高端模式 (全GPU)"
        elif total >= 10:
            # 中端卡：GPU + 少量优化
            pipe.to("cuda")
            pipe.enable_vae_tiling()
            pipe.enable_attention_slicing()
            strategy = "⚡ 标准模式 (GPU+优化)"
        elif total >= 6:
            # 你的 5060 (8GB) → 这里
            pipe.enable_model_cpu_offload()  # 自动 GPU/CPU 切换
            pipe.enable_vae_tiling()
            pipe.enable_attention_slicing()
            if is_sdxl:
                pipe.enable_vae_slicing()
            strategy = "💾 节能模式 (CPU Offload + 内存兜底)"
        else:
            # 低端卡：Sequential Offload
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_tiling()
            pipe.enable_attention_slicing()
            strategy = "🐢 极限模式 (Sequential Offload)"
        
        # xformers（如果有）
        try:
            pipe.enable_xformers_memory_efficient_attention()
            strategy += " + xFormers"
        except Exception:
            pass
        
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