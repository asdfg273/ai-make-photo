# workflow/memory_policy.py
"""显存策略：**能力配置** 驱动，而不是"看显卡名字/总显存"。

为什么要改
----------
原来的 `VRAMManager` 靠"卡多大 + 当前空闲"选档位，加一张新卡就要改阈值。
工作流引擎需要的是一份**能力配置**：允许 offload 吗？允许分块吗？要不要
常驻模型？再叠加各节点自己声明的 `ResourceRequirement`，得出每个节点的
执行方式。这样 6GB / 8GB / 12GB / 16GB / 24GB 都走同一套代码。

四档预设（对应产品里的"配置"）
------------------------------
    saver        节省显存  6~8GB，允许 CPU offload 与分块
    balanced     平衡模式  8~16GB，优先速度与稳定性
    performance  高性能    16GB+，尽量常驻模型与附加组件
    custom       自定义    用户手动指定

`auto` 会按设备总显存挑一档 —— 这只是**默认值**，用户可以覆盖。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SAVER = "saver"
BALANCED = "balanced"
PERFORMANCE = "performance"
CUSTOM = "custom"

PRESET_LABELS = {
    SAVER: "节省显存（6~8GB，允许 CPU offload 与分块）",
    BALANCED: "平衡模式（8~16GB，优先速度与稳定性）",
    PERFORMANCE: "高性能（16GB+，尽量常驻模型与附加组件）",
    CUSTOM: "自定义（手动设置 offload / 缓存 / 精度 / 批量）",
}


@dataclass
class DeviceCapabilities:
    """设备能力探测结果（一次探测，全程复用）。"""
    has_cuda: bool = False
    total_gb: float = 0.0
    free_gb: float = 0.0
    name: str = "cpu"
    supports_bf16: bool = False
    supports_fp16: bool = True
    allow_tf32: bool = True

    @staticmethod
    def detect() -> "DeviceCapabilities":
        cap = DeviceCapabilities()
        try:
            import torch
            cap.has_cuda = bool(torch.cuda.is_available())
            if cap.has_cuda:
                p = torch.cuda.get_device_properties(0)
                cap.total_gb = p.total_memory / 1024 ** 3
                try:
                    free, _ = torch.cuda.mem_get_info(0)
                    cap.free_gb = free / 1024 ** 3
                except Exception:
                    cap.free_gb = cap.total_gb
                cap.name = torch.cuda.get_device_name(0)
                cap.supports_bf16 = bool(
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception as e:
            logger.debug(f"设备能力探测失败，按 CPU 处理: {e}")
        return cap


@dataclass
class MemoryPolicy:
    """一份**解析后**的策略：执行器与节点只看这个，不再看显卡型号。"""
    preset: str = BALANCED
    # 资源放置
    cpu_offload: bool = False          # 模型/组件按需上卡
    sequential_offload: bool = False   # 逐层 offload（最省但最慢）
    keep_model_loaded: bool = True     # 跨节点复用已加载模型
    # VAE / 大图
    vae_tiling: bool = True
    vae_slicing: bool = True
    # 分块
    allow_tiled: bool = True
    tiled_threshold_mp: float = 1.0    # 超过多少百万像素才自动分块
    # 缓存
    cache_intermediates: bool = True
    cache_max_entries: int = 24
    # 精度
    dtype: str = "float16"
    # 批量
    max_batch: int = 1
    # 诊断
    notes: list = field(default_factory=list)

    def describe(self, *, translate=None) -> str:
        tr = translate or (lambda text: text)
        bits = [tr(PRESET_LABELS.get(self.preset, self.preset))]
        if self.sequential_offload:
            bits.append(tr("逐层 offload"))
        elif self.cpu_offload:
            bits.append("CPU offload")
        else:
            bits.append(tr("全 GPU"))
        if self.allow_tiled:
            bits.append(tr("分块≥{threshold}MP").format(
                threshold=f"{self.tiled_threshold_mp:.1f}"))
        if self.cache_intermediates:
            bits.append(tr("缓存{count}").format(count=self.cache_max_entries))
        return " · ".join(bits)


def pick_preset(cap: DeviceCapabilities) -> str:
    """按设备总显存挑默认预设。只是默认，用户可覆盖。

    ⚠️ 阈值用「名义容量」而不是「GiB 读数」：
    标称 8GB 的卡实际报告 7.93GiB（显存被驱动预留掉了 1%~7%），
    如果按 8.5 卡阈值，用户手里的"8GB 卡"会被判成"节省显存"档 ——
    与产品定义（6~8GB 才是节省档）不符。
    取 7.0 作为分界：6GB 卡（≈5.9）进节省档，8GB 卡（≈7.9）进平衡档。
    """
    if not cap.has_cuda:
        return SAVER
    gb = cap.total_gb
    if gb < 7.0:
        return SAVER
    if gb < 15.0:
        return BALANCED
    return PERFORMANCE


def resolve(preset: str = "auto",
            cap: Optional[DeviceCapabilities] = None,
            *,
            arch_id: str = "",
            overrides: Optional[dict] = None) -> MemoryPolicy:
    """把预设 + 设备能力 + 模型架构 解析成具体策略。

    `arch_id` 只影响少数与架构强相关的项（如 SDXL 在 8GB 上必须 offload），
    **不**为每个模型写一套策略。
    """
    cap = cap or DeviceCapabilities.detect()
    if preset in (None, "", "auto"):
        preset = pick_preset(cap)

    if preset == PERFORMANCE:
        pol = MemoryPolicy(preset=PERFORMANCE, cpu_offload=False,
                           keep_model_loaded=True, vae_tiling=False,
                           vae_slicing=False, allow_tiled=True,
                           tiled_threshold_mp=2.5, cache_max_entries=8,
                           dtype="bfloat16" if cap.supports_bf16 else "float16",
                           max_batch=4, notes=["尽量常驻，减少重复加载"])
    elif preset == SAVER:
        pol = MemoryPolicy(preset=SAVER, cpu_offload=True,
                           keep_model_loaded=True, vae_tiling=True,
                           vae_slicing=True, allow_tiled=True,
                           tiled_threshold_mp=0.8, cache_max_entries=16,
                           dtype="float16", max_batch=1,
                           notes=["允许 CPU offload 与分块"])
    elif preset == CUSTOM:
        pol = MemoryPolicy(preset=CUSTOM, cpu_offload=False,
                           keep_model_loaded=True, vae_tiling=True,
                           vae_slicing=True, allow_tiled=True,
                           tiled_threshold_mp=1.0, cache_max_entries=16,
                           dtype="float16", max_batch=1,
                           notes=["用户自定义"])
    else:
        pol = MemoryPolicy(preset=BALANCED, cpu_offload=False,
                           keep_model_loaded=True, vae_tiling=True,
                           vae_slicing=True, allow_tiled=True,
                           tiled_threshold_mp=1.0, cache_max_entries=24,
                           dtype="float16", max_batch=2,
                           notes=["优先速度与稳定性"])

    # ---- 设备兜底：没卡就别谈常驻 ----
    if not cap.has_cuda:
        pol.cpu_offload = True
        pol.sequential_offload = True
        pol.keep_model_loaded = False
        pol.cache_intermediates = False
        pol.notes.append("未检测到 CUDA，全部走 CPU")

    # ---- 架构相关的少量硬约束 ----
    # 实测：SDXL fp16 权重约 6.6GiB，8GB 卡全 GPU 会溢出到共享内存，
    # 45s/步 → 2.3s/步 的差距就是从这里来的。
    heavy = arch_id in ("sdxl", "pony", "sd3", "flux", "anima")
    if heavy and cap.has_cuda and cap.total_gb < 10.0:
        if not pol.cpu_offload:
            pol.cpu_offload = True
            pol.notes.append(f"{arch_id} 在 {cap.total_gb:.1f}GB 卡上启用 CPU offload")
    # 显存很紧时进一步降级
    if cap.has_cuda and cap.total_gb < 6.5:
        pol.sequential_offload = True
        pol.keep_model_loaded = False
        pol.cache_intermediates = False
        pol.notes.append("显存低于 6.5GB，启用逐层 offload")

    if overrides:
        for k, v in overrides.items():
            if hasattr(pol, k) and v is not None:
                setattr(pol, k, v)
        pol.notes.append("应用了自定义覆盖")

    return pol


def plan_for_nodes(policy: MemoryPolicy, specs, cap=None, *, translate=None) -> dict:
    """把图里所有节点的资源声明汇总成一份执行计划摘要。

    `cap` 是设备能力（不给则现场探测）。用它判断"图里要 CUDA 但机器没有"，
    而不是去猜 `policy.notes` —— ⚠️ 旧代码写的是
        `if needs_cuda and not policy.notes:`
    而**每一档预设都会往 notes 里写东西**，`not policy.notes` 几乎永远为假，
    于是"需要 CUDA 但没有设备"这条警告从来没触发过（无卡机器上静默降到 CPU）。

    返回: {"total_peak_mb":…, "needs_cuda":…, "tiled_nodes":[…],
           "offload_nodes":[…], "exclusive_nodes":[…], "warnings":[…]}
    """
    tr = translate or (lambda text: text)
    total = 0
    needs_cuda = False
    tiled, offloadable, exclusive, warnings = [], [], [], []
    needs_exclusive = False

    for nid, spec in specs:
        res = spec.resources
        total += max(0, res.estimated_vram_mb)
        if "cuda" in res.requires:
            needs_cuda = True
        if res.supports_tiled:
            tiled.append(nid)
        if res.supports_cpu_offload:
            offloadable.append(nid)
        else:
            exclusive.append(nid)
        if res.exclusive_gpu:
            needs_exclusive = True

    if cap is None:
        cap = DeviceCapabilities.detect()

    if needs_cuda and not cap.has_cuda:
        warnings.append(tr(
            "该工作流需要 GPU（CUDA），但当前环境未检测到可用 CUDA 设备；"
            "执行会退回 CPU，速度极慢甚至失败"))
    if needs_cuda and cap.has_cuda and total > cap.total_gb * 1024:
        fallback = tr("将依赖分块/offload") if (
            policy.allow_tiled or policy.cpu_offload) else tr("可能 OOM")
        warnings.append(tr(
            "节点声明的显存合计约 {total} MB，超过设备总量 {capacity} MB，{fallback}"
        ).format(total=total, capacity=f"{cap.total_gb * 1024:.0f}",
                 fallback=fallback))
    if needs_exclusive and policy.cpu_offload:
        warnings.append(tr(
            "图里有要求独占 GPU 的节点，但当前策略启用了 CPU offload，速度会明显变慢"))

    return {
        "total_peak_mb": total,
        "needs_cuda": needs_cuda,
        "has_cuda": bool(cap.has_cuda),
        "tiled_nodes": tiled,
        "offload_nodes": offloadable,
        "exclusive_nodes": exclusive,
        "policy": policy.describe(translate=tr),
        "warnings": warnings,
    }
