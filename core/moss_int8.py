"""MOSS-Image DiT 的 int8 权重量化（纯 torch，不依赖任何额外内核）。

为什么需要
----------
bf16 的 DiT 是 9.47GB，装不进 7.93GB 显存，只能分块 offload：每步都要把权重从内存
搬到显卡，慢，而且那 9.5GB 权重会常驻**内存**（开 IDE / 浏览器就爆）。之前试过
bitsandbytes 的 8bit（`core/moss_accel.py` + `quantization_config("bnb8")`），加载
正常、但采样必崩在 `addmm_cuda not implemented for 'Char'` —— 因为模型自带的
`pipeline_moss_image.py` 用 `weight.dtype` 推计算精度，int8 权重会把激活也带成 int8。

这里换一条**完全不依赖第三方内核**的路：
  * 权重按**输出通道**（每行一个 scale）量化成 int8 —— 体积减半到 4.73GB，可以整块
    住进显存；
  * forward 时把这一层的权重反量化回 bf16 再做 bf16 matmul；
  * 反量化是逐层进行的，临时张量只有一层大小（约 23MB），不会把显存吃掉；
  * `weight` 以 property 暴露成**反量化后的 bf16 张量**，所以模型里那些
    `x.to(self.xxx.weight.dtype)` 会照常拿到 bf16 —— 不需要打任何补丁。

为什么不走真 fp8 matmul：本机（sm_120 + cuBLAS 12.8）调 `torch._scaled_mm` 直接报
`CUBLAS_STATUS_NOT_SUPPORTED`，实测过了。而"int8 存储 + 逐层反量化"用的全是普通
tensor 运算，任何 CUDA 卡都能跑。

代价：每层多一次 4.7GB 读 + 9.5GB 写（显存带宽级别），实测单层 bf16 matmul 6.27ms
→ 6.90ms，+10%；换来的是**权重常驻显存**、内存里不再留那 9.5GB。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 配置里 `moss_quant` 的合法取值。
QUANT_MODES = ("off", "int8")

#: 这些层不量化（输出投影对精度最敏感，且只占很小一部分）。
_SKIP_SUFFIXES = ("proj_out", "final_linear", "time_embedding")

INT8_MAX = 127.0

#: `torch._int_mm` 是否可用（模块首次量化时探一次，之后复用）。
W8A8_OK = False


def available() -> tuple[bool, str]:
    """能不能用（只需要一个能跑 bf16 matmul 的 CUDA 卡）。"""
    try:
        import torch
    except Exception as e:  # pragma: no cover - torch 必然存在
        return False, f"import torch 失败: {e}"
    if not torch.cuda.is_available():
        return False, "没有可用的 CUDA 设备"
    try:
        x = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda:0")
        y = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda:0")
        _ = x @ y
        _ = (torch.randn(8, 8, device="cuda:0") / 127.0).round().to(torch.int8)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def w8a8_available() -> tuple[bool, str]:
    """`torch._int_mm` 是否可用（int8 矩阵乘内核，sm_80+ 才有）。"""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        return False, f"import torch 失败: {e}"
    if not torch.cuda.is_available():
        return False, "没有可用的 CUDA 设备"
    if not hasattr(torch, "_int_mm"):
        return False, "本机 torch 没有 torch._int_mm"
    try:
        # 注意：`_int_mm` 对行数有要求（实测要 > 16），用 32 行做冒烟最稳。
        a = torch.randint(-8, 8, (32, 64), dtype=torch.int8, device="cuda:0")
        b = torch.randint(-8, 8, (64, 32), dtype=torch.int8, device="cuda:0")
        out = torch._int_mm(a, b)
        assert out.dtype == torch.int32 and tuple(out.shape) == (32, 32)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def quantize_weight(weight, *, chunk_rows: int = 2048):
    """把 (out, in) 的 bf16 权重压成 (int8 权重, 每行 scale)。

    **按行分块**做，避免一次性 `weight.float()`（4.85B 参数就是 19GB 临时内存，
    实测会把进程内存顶到 18.7GB —— 量化完还没出图就先 OOM 了）。
    """
    import torch

    rows, cols = weight.shape
    q = torch.empty((rows, cols), dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.bfloat16)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        chunk = weight[start:stop].detach().float()
        s = chunk.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / INT8_MAX
        q[start:stop] = (chunk / s).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        scale[start:stop] = s.to(torch.bfloat16)
        del chunk, s
    return q, scale


class Int8WeightLinear:
    """把某个 nn.Linear 的权重换成 int8 存储的包装层。

    注意：这不是 nn.Module 子类 —— 我们用 `__class__` 切换的方式替换已有模块的
    行为（见 `quantize_module`），这样父模块里保存的引用、名字、参数顺序统统不变，
    量化前后整个 state_dict 结构只在参数 dtype/形状上变。
    """

    @staticmethod
    def make(linear):
        import torch

        if getattr(linear, "_moss_int8", False):
            return linear
        q, scale = quantize_weight(linear.weight)
        device = linear.weight.device
        linear.weight_q = torch.nn.Parameter(q.to(device), requires_grad=False)
        linear.weight_scale = torch.nn.Parameter(scale.to(device), requires_grad=False)
        linear._moss_int8 = True
        linear._moss_w8a8 = bool(W8A8_OK)
        # 用 property 顶掉 weight：模型里 x.to(self.xxx.weight.dtype) 会拿到 bf16
        cls = type(linear)
        if "weight" not in cls.__dict__:
            cls = type(
                f"Int8{cls.__name__}",
                (cls,),
                {
                    "weight": property(
                        lambda self: self.weight_q.to(self.weight_scale.dtype)
                        * self.weight_scale
                    ),
                    "__doc__": "int8 权重 + 逐层反量化的 nn.Linear",
                },
            )
            linear.__class__ = cls
        # 关键：必须把原来那份 bf16 权重**从 _parameters 里摘掉**，否则显存/内存里
        # 会同时留着 bf16(9.47GB) 和 int8(4.73GB) 两份 —— 实测直接翻倍成 14.2GB。
        linear._parameters.pop("weight", None)
        return linear


def quantized_linear_forward(x, linear):
    """W8A8：激活按行量化 → `torch._int_mm`（int32 累加）→ 反缩放回 bf16。

    比"反量化权重再做 bf16 matmul"更快（实测单层 5.7ms → 4.2ms），也比它省显存。
    尺寸不满足 `_int_mm` 要求时自动退回反量化路径。
    """
    import torch
    import torch.nn.functional as F

    wq = linear.weight_q
    ws = linear.weight_scale
    bias = linear.bias if getattr(linear, "bias", None) is not None else None
    if linear._moss_w8a8 and x.dim() >= 2 and x.shape[-1] % 8 == 0:
        try:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            sxx = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / INT8_MAX
            xq = (flat / sxx).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            acc = torch._int_mm(xq, wq.t())
            scale = sxx * ws.t()
            out = acc.to(torch.bfloat16) * scale.to(torch.bfloat16)
            if bias is not None:
                out = out + bias
            return out.reshape(*shape[:-1], wq.shape[0])
        except Exception as e:  # 内核不支持 / 尺寸不合法 → 退回反量化
            logger.debug("int8 W8A8 内核失败（%s），改用反量化路径", e)
            linear._moss_w8a8 = False
    w = wq.to(x.dtype) * ws.to(x.dtype)
    return F.linear(x, w, bias)


def _trim_working_set():
    """让 Windows 把用不上的页交还系统（量化会把整份权重读一遍，页会留在工作集里）。"""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.EmptyWorkingSet(ctypes.c_void_p(kernel32.GetCurrentProcess()))
    except Exception as e:
        logger.debug("释放工作集失败（忽略）: %s", e)


def quantize_module(module, *, skip_suffixes=_SKIP_SUFFIXES) -> dict:
    """把 module 里所有 nn.Linear 换成 int8 存储，返回统计信息。"""
    import torch

    global W8A8_OK
    if module is None:
        return {"count": 0, "skipped": 0, "before_mb": 0.0, "after_mb": 0.0,
                "w8a8": False}
    ok, _why = w8a8_available()
    W8A8_OK = bool(ok)
    count = skipped = 0
    before = after = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, torch.nn.Linear):
            continue
        if getattr(child, "_moss_int8", False):
            continue
        if any(name.endswith(s) for s in skip_suffixes):
            skipped += 1
            continue
        n = child.weight.numel()
        before += n * child.weight.element_size()
        try:
            Int8WeightLinear.make(child)
        except Exception as e:
            logger.warning("int8 量化跳过 %s: %s", name, e)
            skipped += 1
            continue
        after += n * 1
        count += 1
        # 每 64 层交还一次工作集：文本编码器有 358 层、13.7GB，量化时会把整份
        # 权重读进工作集；不主动交还的话峰值会顶到 24GB（本机只有 31.7GB）。
        if count % 64 == 0:
            _trim_working_set()
    # forward 也一起换掉（在类上打补丁，逐层反量化）
    if count:
        _patch_forward()
        _trim_working_set()
    return {
        "count": count,
        "skipped": skipped,
        "before_mb": before / 1024 ** 2,
        "after_mb": after / 1024 ** 2,
        "w8a8": bool(W8A8_OK),
    }


def _patch_forward():
    from torch.nn.modules.linear import Linear

    if getattr(Linear.forward, "_moss_int8_patched", False):
        return
    original = Linear.forward

    def forward(self, input):  # noqa: A002 - 与 torch 签名一致
        if getattr(self, "_moss_int8", False):
            return quantized_linear_forward(input, self)
        return original(self, input)

    forward._moss_int8_patched = True
    forward.__doc__ = original.__doc__
    Linear.forward = forward


def describe(stats: dict) -> str:
    if not stats or not stats.get("count"):
        return "int8 权重量化：未生效（没有可量化的 Linear 层）"
    path = "W8A8 内核" if stats.get("w8a8") else "逐层反量化"
    return (
        f"int8 权重量化：{stats['count']} 层（跳过 {stats['skipped']} 层），"
        f"{stats['before_mb'] / 1024:.2f}GB → {stats['after_mb'] / 1024:.2f}GB，"
        f"走「{path}」"
    )
