"""MOSS-Image 8bit 常驻显存的补丁（本地生成，不下载任何东西）。

## 为什么需要它

MOSS-Image 自带的 `pipeline_moss_image.py` 用 **权重的 dtype** 推导计算精度::

    time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype))

一旦把 DiT 量化成 int8（`bitsandbytes_8bit`），这一行会把**激活**也转成 int8，
于是 `nn.SiLU` 直接报 `NotImplementedError: "silu_cuda" not implemented for
'Char'`。这种写法在官方文件里共 8 处，全部是 `.weight.dtype`。

本模块把官方文件读进来，逐处替换成"取权重 dtype，非浮点则回落到 bf16"的辅助
函数，然后在内存里 exec 出一份模块，最后把**已加载实例**的方法重新绑定过去 ——

* 不改磁盘上模型目录里的任何文件；
* 不改 diffusers 的模块缓存；
* 关掉开关就是原生行为（重新加载模型即可）。

## 收益（本机 RTX 5060 Laptop 8GB 实测）

================  ================  ==============  ==========
方案              DiT 权重          1024² 每步       系统内存
================  ================  ==============  ==========
bf16 + 分块 offload 常驻内存 9.5GB    5.2s/步          +9.5GB
8bit 常驻显存     显存 5.0GB         6.1s/步          -9.5GB
================  ================  ==============  ==========

即：**每步慢约 18%，但省下 9.5GB 系统内存**，显存峰值会顶到 7.8/7.9GB。
对小内存机器（开 IDE/浏览器就爆）这个交换是值得的，所以做成默认关闭、由用户决定。
"""
from __future__ import annotations

import logging
import os
import types
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: 补丁版本，官方文件变化或替换规则调整时 +1
PATCH_VERSION = 1

#: 官方文件里需要修补的方法所在类
PATCHED_CLASSES = ("MOSSImageTransformer", "_TextBlock", "_TextProjection")

_HELPER_SRC = '''

def _moss_ctype(module):
    """取模块的计算精度：整型权重（量化）一律回落到 bf16。

    MOSS-Image 官方管线用 ``weight.dtype`` 推计算精度，量化后连激活都会被
    转成 int8，SiLU 之类的算子会直接拒绝。这里只保留"浮点"语义。
    """
    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", torch.bfloat16)
    return dtype if getattr(dtype, "is_floating_point", False) else torch.bfloat16

'''

#: 官方文件里 8 处 `weight.dtype` 的精确替换（顺序无关，每处必须命中 1 次）
_REPLACEMENTS = (
    ("x.to(self.proj.weight.dtype)",
     "x.to(_moss_ctype(self.proj))"),
    ("dtype = self.proj['q'].weight.dtype",
     "dtype = _moss_ctype(self.proj['q'])"),
    ("embedding.to(self.time_embedding[0].weight.dtype)",
     "embedding.to(_moss_ctype(self.time_embedding[0]))"),
    ("time.to(self.shared_block_modulation.weight.dtype)",
     "time.to(_moss_ctype(self.shared_block_modulation))"),
    ("time.to(self.video_head_modulation.weight.dtype)",
     "time.to(_moss_ctype(self.video_head_modulation))"),
    ("video.to(self.video_embedder.weight.dtype)",
     "video.to(_moss_ctype(self.video_embedder))"),
    ("x[:, :height * width].to(self.final_norm_video.weight.dtype)",
     "x[:, :height * width].to(_moss_ctype(self.final_norm_video))"),
    ("x.to(self.final_linear_video.weight.dtype)",
     "x.to(_moss_ctype(self.final_linear_video))"),
)


def describe() -> str:
    """给界面用的一句话说明。"""
    return ("8bit 常驻显存：DiT 权重留在显存里不再每步搬运，"
            "可释放约 9.5GB 系统内存；代价是每步慢约 18%、显存顶到 7.8GB 左右")


def bnb8_available() -> Tuple[bool, str]:
    """跑一次 8×8 的 int8 矩阵乘，确认 bitsandbytes 的内核真的能用。

    本机实测（bnb 0.49.2 + torch 2.11 + RTX 5060 / sm_120）：
    ``MatMul8bitLt`` 会掉进 bitsandbytes 的 Python 兜底实现并报

    * ``NotImplementedError: "addmm_cuda" not implemented for 'Char'``
      （threshold > 0 走 mixed-int8 路线），或
    * ``RuntimeError: expected scalar type Float but found Char``
      （把 threshold 改成 0 走 int8_scaled_mm 路线）。

    也就是说**权重能加载成 int8，但一采样就崩**。这种情况必须提前拦住，
    否则用户会看到一条与"开个开关"毫无关系的报错。
    """
    try:
        import torch
        import bitsandbytes as bnb
    except Exception as e:
        return False, f"没有 bitsandbytes: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "没有可用的 CUDA 设备"
    try:
        layer = bnb.nn.Linear8bitLt(8, 8, bias=False, has_fp16_weights=False)
        layer = layer.to("cuda:0")
        x = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda:0")
        layer(x)
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def patch_source(src: str) -> Tuple[Optional[str], str]:
    """把官方管线源码改成量化安全的版本。

    返回 ``(新源码, "")`` 或 ``(None, 失败原因)``。任何一处替换不唯一就
    整体放弃 —— 宁可退回原生 bf16，也不生成一份"改了一半"的代码。
    """
    if "def _moss_ctype" in src:
        return None, "这份管线文件已经被修补过了"
    out = src
    for i, (old, new) in enumerate(_REPLACEMENTS):
        count = out.count(old)
        if count != 1:
            return None, (f"第 {i + 1} 处替换命中 {count} 次（应为 1 次）："
                          f"{old[:60]}… —— 官方管线文件可能改版了")
        out = out.replace(old, new)
    if ".weight.dtype" in out:
        return None, "修补后仍有 .weight.dtype 残留"
    marker = "\nclass MOSSImageTransformer"
    if marker not in out:
        return None, "找不到 MOSSImageTransformer 类定义，无法插入辅助函数"
    out = out.replace(marker, _HELPER_SRC + marker, 1)
    return out, ""


def build_module(model_dir: str) -> Tuple[Optional[types.ModuleType], str]:
    """读取官方管线文件 → 修补 → 在内存里 exec 成模块。

    返回 ``(module, "")`` 或 ``(None, 原因)``；``model_dir`` 里没有
    ``pipeline_moss_image.py`` 时也返回原因而不是抛异常。
    """
    script = os.path.join(model_dir or "", "pipeline_moss_image.py")
    if not os.path.isfile(script):
        return None, f"目录里没有 pipeline_moss_image.py: {script}"
    try:
        with open(script, "r", encoding="utf-8") as fh:
            src = fh.read()
    except OSError as e:
        return None, f"读取管线文件失败: {e}"

    patched, why = patch_source(src)
    if patched is None:
        return None, why

    mod = types.ModuleType("moss_image_patched")
    mod.__file__ = script
    try:
        exec(compile(patched, script, "exec"), mod.__dict__)
    except Exception as e:  # 语法/导入问题都要能降级
        return None, f"修补后的管线代码无法编译: {type(e).__name__}: {e}"
    return mod, ""


def apply(pipe, module: types.ModuleType) -> int:
    """把 ``module`` 里同名类的方法绑定到已加载的实例上。

    diffusers 会用**模型目录里的** `pipeline_moss_image.py` 来建组件，所以
    只换管线类是不够的；而且量化给模块套了 `_old_forward` 钩子，必须一起换，
    否则调用的还是旧方法。
    """
    if pipe is None or module is None:
        return 0
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return 0

    patched_classes = 0
    rebound_modules = 0
    for m in transformer.modules():
        cls = type(m)
        patched_cls = getattr(module, cls.__name__, None)
        fn = getattr(patched_cls, "forward", None)
        if fn is None or "forward" not in getattr(patched_cls, "__dict__", {}):
            continue
        if cls.__name__ not in PATCHED_CLASSES:
            continue
        if cls.forward is not fn:
            cls.forward = fn
            patched_classes += 1
        old = getattr(m, "_old_forward", None)
        if old is not None:
            m._old_forward = types.MethodType(fn, m)
            rebound_modules += 1

    logger.info("🩹 MOSS-Image 管线已修补：%d 个类、%d 个模块（版本 v%d）",
                patched_classes, rebound_modules, PATCH_VERSION)
    return patched_classes
