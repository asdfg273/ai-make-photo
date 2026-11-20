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
    def cleanup():
        """手动清理显存"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            # offload 场景下先同步，确保待回收的张量不再被 kernel 引用
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
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
    def _set_vae_option(pipe, option: str, enabled: bool) -> bool:
        """按能力检测 VAE slicing / tiling 接口，不依赖版本号。"""
        if option not in ("slicing", "tiling"):
            raise ValueError(f"不支持的 VAE 选项: {option}")

        action = "enable" if enabled else "disable"
        vae = getattr(pipe, "vae", None)

        candidates = (
            (vae, f"{action}_{option}"),
            (pipe, f"{action}_vae_{option}"),
        )

        for obj, method_name in candidates:
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue

            try:
                method()
                return True
            except Exception:
                logger.warning(
                    "调用 %s 失败，尝试兼容接口",
                    method_name,
                    exc_info=True,
                )

        logger.debug("VAE %s=%s 未能设置", option, enabled)
        return False

    @staticmethod
    def _set_attention_slicing(pipe, enabled: bool) -> bool:
        """按能力检测 attention slicing 接口。

        ⚠️ 实测结论（RTX 5060 Laptop 8G，diffusers 0.40 + torch 2.11 SDPA）：
        **不要开** —— 开了反而更慢、峰值显存还更高：

            512x768 :  ON 3.43s / peak 2.67GB   OFF 2.51s / peak 2.61GB
            1024x1024: ON 111.8s / peak 6.26GB   OFF 9.53s / peak 2.62GB

        torch 2.x 的 SDPA 已经是显存高效实现，attention slicing 会把它拆成
        逐 head 循环，失去 SDPA 的 kernel 融合 → 慢 12~20 倍。
        本方法仅保留为显式关闭的入口（IP-Adapter 安装时会用到）。
        """
        action = "enable" if enabled else "disable"
        method = getattr(pipe, f"{action}_attention_slicing", None)
        if not callable(method):
            return False
        try:
            method()
            return True
        except Exception:
            logger.debug("attention slicing %s 失败", action, exc_info=True)
            return False

    @staticmethod
    def disable_attention_slicing(pipe) -> bool:
        """确保 attention slicing 关闭（推荐的默认状态）。"""
        return VRAMManager._set_attention_slicing(pipe, False)

    @staticmethod
    def _module_gb(mod) -> float:
        """模块参数占用的显存（GiB）。测不到返回 0。"""
        if mod is None:
            return 0.0
        try:
            total = 0
            for p in mod.parameters():
                total += p.numel() * p.element_size()
            return total / 1024 ** 3
        except Exception:
            return 0.0

    # 各架构「激活值余量」经验需求 (GiB)：(全GPU, CPU-Offload)
    #
    # 全GPU 一列 = 权重全部常驻时还要留多少给激活值；
    # Offload 一列 = 只有主干（UNet/transformer）+VAE 常驻时还要留多少。
    # 数值偏保守：宁可多 offload 一点，也不要溢出到共享内存
    # （一旦溢出就是 20 倍减速，见 apply_optimal_strategy 内的注释）。
    _ACT_NEEDS = {
        "sd15":  (1.2, 0.6),
        "sd21":  (1.2, 0.6),
        "sdxl":  (2.0, 1.0),
        "pony":  (2.0, 1.0),
        "sd3":   (2.0, 1.0),
        # Flux 主干 transformer 本身就比 8GB 大得多，只有量化版才可能
        # 走 CPU-Offload；全 bf16 会落到 Sequential。
        "flux":  (2.5, 1.2),
        # MOSS-Image：5B DiT（bf16 约 10GB）+ Qwen 系文本编码器。
        # 官方在 1024²/50 步测到 3.65GiB 显存 / 14GiB 内存（minimal 档），
        # 说明它是靠 offload 把权重留在 CPU 内存里的；这里按「主干外的
        # 组件几乎都得 offload」估，激活值需求比 SDXL 略高（4K token 提示词）。
        "mossimage": (3.0, 1.4),
    }

    @classmethod
    def _activation_needs(cls, arch_id: str, is_sdxl: bool) -> tuple:
        if arch_id and arch_id in cls._ACT_NEEDS:
            return cls._ACT_NEEDS[arch_id]
        return (2.0, 1.0) if is_sdxl else (1.2, 0.6)

    @staticmethod
    def apply_optimal_strategy(pipe, is_sdxl=False, weights_gb=0.0,
                               arch_id=""):
        """
        对新加载的管线选择设备策略。

        应在模型组件迁移到 GPU 之前调用一次。
        不用于运行时切换已有 offload hook 的管线。
        不修改 attention processor。
        """
        info = VRAMManager.get_vram_info()

        if not info["has_cuda"]:
            logger.info("CPU 模式")
            pipe.to("cpu")
            return "cpu"

        # 与 get_vram_info() 保持一致，当前使用 GPU 0。
        device_id = 0
        gib = 1024 ** 3

        reserved = torch.cuda.memory_reserved(device_id) / gib
        allocated = torch.cuda.memory_allocated(device_id) / gib

        # 仅估算未被活跃张量占用的缓存，不把活跃权重算作空闲。
        cached_unused = max(0.0, reserved - allocated)
        budget = min(info["total"], info["free"] + cached_unused)

        logger.info(
            "GPU: %s，总量 %.2f GiB，空闲 %.2f GiB，"
            "未使用缓存 %.2f GiB，估算预算 %.2f GiB",
            info["name"],
            info["total"],
            info["free"],
            cached_unused,
            budget,
        )

        # VAE slicing 主要降低多张图片解码时的峰值。
        VRAMManager._set_vae_option(pipe, "slicing", True)

        # ── 关键：决策必须看「装完权重后还剩多少」，而不是「装之前有多空」──
        #
        # 旧实现拿载入前的空闲显存（8GB 卡实测 6.88GiB）去比按物理总量缩放的
        # 阈值，于是 SDXL（fp16 权重 6.57GiB）被判成「全速模式(全GPU)」。
        # 实测后果（RTX 5060 Laptop 8G，768x1024，3 步）：
        #     全GPU   45.0 s/步，peak 9.55 GiB ← 超过物理 7.93GiB
        #     Offload  2.3 s/步，peak 5.13 GiB
        # 峰值超过物理显存后 Windows/WDDM 会**静默**溢出到共享系统内存，
        # 每一步都要走 PCIe，慢 20 倍，且会把系统提交内存耗尽
        # （连读模型文件头都失败，报 "页面文件太小 os error 1455"）。
        #
        # 用真实的模块体积算出「留给激活值的余量」再选档位。
        #
        # ⚠️ 主干不一定是 .unet：Flux/SD3 在 diffusers 里叫 .transformer，
        #    只认 .unet 会让它们量到 0 而掉进旧的兜底逻辑。
        backbone = getattr(pipe, "unet", None)
        if backbone is None:
            backbone = getattr(pipe, "transformer", None)
        main_gb = VRAMManager._module_gb(backbone)

        vae_gb = VRAMManager._module_gb(getattr(pipe, "vae", None))
        # SD3 有第三个编码器（T5），Flux 的 T5 在 text_encoder_2
        te_gb = sum(VRAMManager._module_gb(getattr(pipe, a, None))
                    for a in ("text_encoder", "text_encoder_2", "text_encoder_3"))

        total = float(info["total"])

        # ⚠️ 不是所有管线都支持 offload hooks：Anima 用的 ModularPipeline 连
        #    `enable_model_cpu_offload` / `enable_sequential_cpu_offload` 都
        #    没有（MRO 是 ConfigMixin，不是 DiffusionPipeline）。旧实现在
        #    分支里直接 call，一旦走到就是 AttributeError，模型加载失败。
        has_offload = (hasattr(pipe, "enable_model_cpu_offload")
                       and hasattr(pipe, "enable_sequential_cpu_offload"))

        def _offload_else_full(level: int, label: str,
                               tiling: bool = False) -> str:
            """能 offload 就 offload，不能就老实说清楚、退回全 GPU。"""
            if tiling:
                VRAMManager._set_vae_option(pipe, "tiling", True)
            if has_offload:
                if level == 1:
                    pipe.enable_model_cpu_offload(gpu_id=device_id)
                else:
                    pipe.enable_sequential_cpu_offload(gpu_id=device_id)
                return label
            pipe.to("cuda:0")
            logger.warning(
                "⚠️ %s 不支持 CPU offload（如 ModularPipeline），只能全 GPU 运行；"
                "显存不够时请降低分辨率或步数", type(pipe).__name__)
            return "🚀 全速模式 (全GPU，该管线不支持 offload)"

        def _tier(frac: float, floor: float) -> float:
            """比例阈值，不超过物理总量也不低于下限。"""
            return min(total, max(floor, total * frac))

        if main_gb > 0.0:
            # 激活值（含 attention 峰值）经验需求，GiB —— 按架构取
            need_full, need_offload = VRAMManager._activation_needs(arch_id,
                                                                    is_sdxl)

            # 全 GPU：主干 + VAE + 文本编码器全部常驻
            headroom_full = budget - (main_gb + vae_gb + te_gb)
            # CPU Offload：文本编码器用完即挪走，主干 + VAE 常驻
            headroom_offload = budget - (main_gb + vae_gb)

            logger.info(
                "权重体积: 主干 %.2f + VAE %.2f + 文本编码器 %.2f = %.2f GiB | "
                "全GPU余量 %.2f GiB (需 %.2f) / Offload余量 %.2f GiB (需 %.2f)",
                main_gb, vae_gb, te_gb, main_gb + vae_gb + te_gb,
                headroom_full, need_full, headroom_offload, need_offload,
            )

            if headroom_full >= need_full:
                pipe.to("cuda:0")
                strategy = "🚀 全速模式 (全GPU)"
            elif headroom_offload >= need_offload:
                strategy = _offload_else_full(
                    1, "⚡ 标准模式 (CPU Offload，权重按需上卡)")
            else:
                strategy = _offload_else_full(
                    2, "🐢 极限模式 (Sequential Offload)", tiling=True)
        else:
            # 主干体积测不到（unet / transformer 都没有，如 Anima 的
            # ModularPipeline）→ 退回按预算的旧逻辑，并用检查点体积扣减
            if weights_gb:
                budget = max(0.0, budget - float(weights_gb))
            if is_sdxl:
                gpu_full = _tier(0.88, 5.0)
                gpu_sliced = _tier(0.60, 3.5)
                offload_min = _tier(0.28, 2.0)
            else:
                gpu_full = _tier(0.55, 3.0)
                gpu_sliced = _tier(0.38, 2.0)
                offload_min = _tier(0.20, 1.2)

            if budget >= gpu_full:
                pipe.to("cuda:0")
                strategy = "🚀 全速模式 (全GPU)"
            elif budget >= gpu_sliced:
                pipe.to("cuda:0")
                strategy = "⚡ 标准模式 (全GPU+VAE切片)"
            elif budget >= offload_min:
                strategy = _offload_else_full(
                    1, "💾 节能模式 (CPU Offload)", tiling=True)
            else:
                strategy = _offload_else_full(
                    2, "🐢 极限模式 (Sequential Offload)", tiling=True)

        logger.info("策略: %s", strategy)
        return strategy

    @staticmethod
    def tune_for_resolution(pipe, width, height, is_sdxl=False):
        """
        只调整 VAE 策略。

        不改变 offload 档位，不重装 hook，
        不启用或关闭 attention slicing。
        """
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("生成宽度和高度必须大于 0")

        info = VRAMManager.get_vram_info()
        if not info["has_cuda"]:
            return "CPU 模式"

        pixels = width * height

        # 除总像素外，也考虑长条形图片。
        large_image = (
            pixels >= 1024 * 1024
            or max(width, height) > 1024
        )

        # 8GB SDXL 保留 VAE 分块。
        # 真正是否进入分块解码，还由 VAE 自身的尺寸阈值决定。
        low_vram_sdxl = is_sdxl and info["total"] < 10.0
        use_tiling = low_vram_sdxl or large_image

        VRAMManager._set_vae_option(pipe, "slicing", True)
        VRAMManager._set_vae_option(pipe, "tiling", use_tiling)

        vae = getattr(pipe, "vae", None)
        actual_tiling = getattr(vae, "use_tiling", "未知")
        actual_slicing = getattr(vae, "use_slicing", "未知")

        msg = (
            f"VAE 策略 ({width}x{height})："
            f"tiling={actual_tiling}, slicing={actual_slicing}"
        )
        logger.info(msg)
        return msg
