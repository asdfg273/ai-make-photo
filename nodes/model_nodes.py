# nodes/model_nodes.py
"""模型相关节点：LoadCheckpoint / LoadVAE / LoadLoRA。

节点约定（所有节点模块统一遵守）
--------------------------------
· `run(ctx, node, inputs)` 返回 **dict**，键名必须与 NodeSpec.outputs 一致；
· 需要重依赖（torch/diffusers）时在函数**内部** import —— 这样 `import nodes`
  不会拖慢界面启动；
· 出错就抛异常，执行器会包成带 node_id 的 NodeError 并定位到本节点；
· 不要直接操作界面控件，只通过 `ctx`（将来才能无头跑）。
"""
from __future__ import annotations

import logging
import os

from workflow import adapters as adapters_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import LoraRef, ModelRef, VaeRef

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
@node(
    "LoadCheckpoint",
    category="模型",
    description="加载底模（.safetensors/.ckpt）。可填文件名或绝对路径。",
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "vae": PortSpec(DataType.VAE, "VAE"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "path": ParamSpec("str", default="", label="模型",
                          source="checkpoints",
                          help="models/ 下的文件名，或绝对路径；"
                               "界面会列出本地已下载的底模供下拉选择"),
        "clip_skip": ParamSpec("int", default=0, label="CLIP Skip",
                               minimum=0, maximum=12),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=6400,
                                  supports_cpu_offload=True,
                                  exclusive_gpu=False, needs_model=True),
)
def run_load_checkpoint(ctx, node, inputs):
    path = (node.params.get("path") or inputs.get("path") or "").strip()
    if not path:
        raise ValueError(ctx.translate("没有指定模型（path 为空）"))
    if ctx.ai is None:
        raise RuntimeError(ctx.translate("没有可用的模型管理器（ctx.ai 为空）"))

    ai = ctx.ai
    name = path
    if os.path.isabs(path) and os.path.exists(path):
        name = os.path.basename(path)

    ctx.status(f"⏳ 加载模型 {name} ...")
    ai.load_model(name)
    ctx.check_cancel()

    arch = getattr(ai, "current_arch_id", "") or "sd15"
    adapter = adapters_mod.get_adapter(arch)

    # 可选：应用 clip skip / 采样器
    cs = int(node.params.get("clip_skip") or 0)
    if cs > 0:
        try:
            from utils.clip_skip import apply_clip_skip_to_pipes, supports_clip_skip
            ok, why = supports_clip_skip(ai)
            if ok:
                apply_clip_skip_to_pipes(ai, cs)
            else:
                logger.info(f"ℹ️ CLIP Skip 不适用: {why}")
        except Exception as e:
            logger.warning(f"⚠️ CLIP Skip 应用失败（忽略）: {e}")
    # ⚠️ 这里**不再**切换采样器：采样器是 Sampler 节点的参数，不属于"加载底模"。
    # 老工作流里残留的 `sampler` 参数会被忽略（不再生效），避免两个地方都能改
    # 采样器、还互相覆盖。
    sampler = (node.params.get("sampler") or "").strip()
    if sampler:
        logger.info("ℹ️ LoadCheckpoint 上的 sampler 参数已废弃，"
                    "采样器请在 Sampler 节点里设置（本次忽略）")

    model = ModelRef(
        name=getattr(ai, "current_model_name", name) or name,
        path=getattr(ai, "current_model_path", "") or "",
        arch_id=arch, pipe=ai.txt2img_pipe,
        img2img_pipe=getattr(ai, "img2img_pipe", None),
        inpaint_pipe=getattr(ai, "inpaint_pipe", None),
        adapter=adapter, vae=getattr(ai.txt2img_pipe, "vae", None),
        scheduler=getattr(ai.txt2img_pipe, "scheduler", None),
        is_sdxl=bool(getattr(ai, "is_sdxl", False)),
        clip_skip=cs, dtype=getattr(ai, "dtype", None), loaded=True,
        meta={"capabilities": adapters_mod.capabilities_for(arch).to_dict()})
    caps = adapters_mod.capabilities_for(arch)
    meta = {"model": model.name, "model_path": model.path,
            "architecture": arch, "adapter": adapter.describe(),
            "capabilities": caps.to_dict()}
    ctx.metadata.update(meta)
    # 把能力描述挂到上下文，供后续节点直接判断能力（而不是判断模型名）
    try:
        ctx.capabilities = caps
    except Exception:
        pass

    vae = VaeRef(name=os.path.basename(model.path or "vae"), path=model.path,
                 vae=model.vae)
    return {"model": model, "vae": vae, "metadata": meta}


# ----------------------------------------------------------------------
@node(
    "LoadVAE",
    category="模型",
    description="加载独立 VAE，并替换到模型管线上。",
    inputs={"model": PortSpec(DataType.MODEL, "模型", optional=True)},
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "vae": PortSpec(DataType.VAE, "VAE")},
    params={"path": ParamSpec("str", default="", label="VAE 文件",
                              source="vae",
                              help="models/ 或 weights/ 下的文件名，"
                                   "也可以直接填绝对路径；"
                                   "界面会列出本地 VAE 供下拉选择")},
    resources=ResourceRequirement(requires=("model",),
                                  estimated_vram_mb=300,
                                  supports_cpu_offload=True),
)
def run_load_vae(ctx, node, inputs):
    import torch
    from diffusers import AutoencoderKL

    model = inputs.get("model")
    path = (node.params.get("path") or "").strip()
    if not path:
        raise ValueError(ctx.translate("没有指定 VAE 文件"))
    if not os.path.exists(path):
        # 到 models/ 与 weights/ 下找
        from utils import paths
        found = None
        for root in (paths.MODEL_DIR, paths.WEIGHTS_DIR):
            for dirpath, _d, files in os.walk(root):
                for f in files:
                    if f == path or f.lower() == path.lower():
                        found = os.path.join(dirpath, f)
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(ctx.translate("找不到 VAE 文件: {path}").format(path=path))
        path = found

    dtype = getattr(ctx.ai, "dtype", torch.float16) if ctx.ai else torch.float16
    vae = AutoencoderKL.from_single_file(path, torch_dtype=dtype)
    if ctx.device == "cuda":
        vae.to("cuda")
    vae.eval()

    if model is not None:
        for attr in ("pipe", "img2img_pipe", "inpaint_pipe"):
            p = getattr(model, attr, None)
            if p is not None:
                try:
                    p.vae = vae
                except Exception:
                    pass
        model.vae = vae
    return {"model": model, "vae": VaeRef(name=os.path.basename(path),
                                          path=path, vae=vae)}


# ----------------------------------------------------------------------
@node(
    "LoadLoRA",
    category="模型",
    description="加载若干 LoRA 并挂到模型上（可串联多个）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型")},
    outputs={"model": PortSpec(DataType.MODEL, "模型"),
             "lora": PortSpec(DataType.LORA, "LoRA"),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "lora_1": ParamSpec("str", default="", label="LoRA 1",
                            source="loras",
                            help="loras/ 下的文件名（可含子目录），"
                                 "也可以填绝对路径；"
                                 "界面会列出本地 LoRA 供下拉选择"),
        "weight_1": ParamSpec("float", default=0.8, label="权重 1",
                              minimum=-2.0, maximum=2.0, step=0.05),
        "lora_2": ParamSpec("str", default="", label="LoRA 2",
                            source="loras",
                            help="loras/ 下的文件名（可含子目录），"
                                 "也可以填绝对路径；"
                                 "界面会列出本地 LoRA 供下拉选择"),
        "weight_2": ParamSpec("float", default=0.8, label="权重 2",
                              minimum=-2.0, maximum=2.0, step=0.05),
        "lora_3": ParamSpec("str", default="", label="LoRA 3",
                            source="loras",
                            help="loras/ 下的文件名（可含子目录），"
                                 "也可以填绝对路径；"
                                 "界面会列出本地 LoRA 供下拉选择"),
        "weight_3": ParamSpec("float", default=0.8, label="权重 3",
                              minimum=-2.0, maximum=2.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("model",),
                                  estimated_vram_mb=200,
                                  supports_cpu_offload=True),
)
def run_load_lora(ctx, node, inputs):
    model = inputs.get("model")
    if model is None:
        raise ValueError(ctx.translate("缺少输入 model"))
    if ctx.ai is None:
        raise RuntimeError(ctx.translate("没有可用的模型管理器"))

    pairs = []
    for i in (1, 2, 3):
        nm = (node.params.get(f"lora_{i}") or "").strip()
        if not nm or nm in ("无", "none", "None"):
            continue
        pairs.append((nm, float(node.params.get(f"weight_{i}") or 0.8)))
    if not pairs:
        return {"model": model, "lora": None,
                "metadata": {"loras": []}}

    # 复用 ModelManager 的 LoRA 应用逻辑（它已处理子目录/架构匹配）
    lora_list = [{"name": n, "weight": w} for n, w in pairs]
    try:
        ctx.ai.apply_multiple_loras(lora_list)
    except TypeError:
        # 兼容旧的 (name, weight) 元组签名
        ctx.ai.apply_multiple_loras(pairs)
    ctx.check_cancel()

    refs = [LoraRef(name=n, weight=w, applied=True) for n, w in pairs]
    info = [f"<lora:{n}:{w}>" for n, w in pairs]
    model.meta["loras"] = info
    ctx.metadata["loras"] = info
    return {"model": model, "lora": refs, "metadata": {"loras": info}}
