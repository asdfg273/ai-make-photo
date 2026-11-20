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

    # 建立"干净的基线"：清掉上一轮可能残留的 LoRA。
    #
    # ⚠️ 为什么必须清：`load_model()` 在**同一个模型**上会短路跳过（不重新加载），
    # 而 LoRA 是挂在管线上的副作用状态。于是"没有 LoadLoRA 节点的工作流"
    # 会继承上一个工作流挂的 LoRA —— 同一张图跑两次结果不一样，
    # 用户会以为"关掉 LoRA 没生效"。实测：带 LoRA 出的图 vs 之后不带 LoRA
    # 出的图差异 **0.0**（完全相同），显式清空后才恢复到 33.3 的正常差异。
    #
    # 语义上也该如此：LoadCheckpoint = 干净的底模；LoRA 由 LoadLoRA 负责加。
    # 拓扑序保证 LoadCheckpoint 一定排在 LoadLoRA 之前。
    try:
        ai.apply_multiple_loras([])
    except Exception as e:
        logger.debug(f"清理残留 LoRA 失败（忽略）: {e}")

    # IP-Adapter 多参考图同样必须清空 —— 一模一样的串味问题：
    # `ModelRef.meta` 是跨节点传递的同一份 dict，若不清，
    # "这轮没接参考图"的工作流会继承上一轮的参考图，
    # 同一组参数跑两次结果不同，用户完全无从判断。
    # （清理放在 ModelRef 建立**之后**，见下面的 model.meta）

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
    # 干净基线：清掉累积的 IP-Adapter 参考图（见上面 LoRA 那段同样的理由）。
    # ⚠️ 必须在 ModelRef 建好之后清：参考图就存在 model.meta 里。
    try:
        from nodes import ipa_nodes
        ipa_nodes.clear_references(model)
    except Exception as e:
        logger.debug(f"清理残留参考图失败（忽略）: {e}")
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
    # 必须传 [(name, weight)] 元组形式。
    # 以前传的是 [{"name":..,"weight":..}]，而 apply_multiple_loras 内部是
    # `for (lora_name, weight) in lora_list` —— 对字典解包拿到的是**键**
    # （"name"/"weight"），于是它去找名为 "name" 的文件、找不到就 continue，
    # **LoRA 被静默跳过**，出图和不挂 LoRA 完全一样；而下面的
    # `except TypeError` 兜底也救不了（2 键字典解包不抛 TypeError）。
    lora_pairs = [(n, w) for n, w in pairs]
    try:
        ctx.ai.apply_multiple_loras(lora_pairs)
    except TypeError:
        # 兼容旧的 (name, weight) 元组签名
        ctx.ai.apply_multiple_loras(pairs)
    ctx.check_cancel()

    # 挂完核实一下：没生效就必须说出来，不能再静默
    applied_ok = True
    try:
        _pipe = getattr(ctx.ai, "txt2img_pipe", None)
        _active = list(_pipe.get_active_adapters() or []) if _pipe else []
        if not _active:
            applied_ok = False
            logger.error("LoRA 未真正挂上（active_adapters 为空）: %s",
                         lora_pairs)
    except Exception:
        pass

    refs = [LoraRef(name=n, weight=w, applied=applied_ok)
            for n, w in pairs]
    info = [f"<lora:{n}:{w}>" for n, w in pairs]
    model.meta["loras"] = info
    model.meta["loras_applied"] = applied_ok
    ctx.metadata["loras"] = info
    if not applied_ok:
        ctx.metadata["warnings"] = list(
            ctx.metadata.get("warnings") or []) + [
            "LoRA 没有真正挂上（文件可能不在当前架构的 loras/ 子目录下）"]
    return {"model": model, "lora": refs, "metadata": {"loras": info}}
