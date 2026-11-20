# nodes/sampling_nodes.py
"""采样与潜空间节点：Sampler / VAEEncode / VAEDecode。

为什么拆成"采样出 latent + 单独解码"
------------------------------------
这样"换 VAE 重解码""只重跑解码"“接 Hires 前先看 latent”都变成改一条线的事，
而不是重跑整个流程。Sampler 优先请求 `output_type="latent"`；
若某条管线不支持，则退回"直接解码"并同时给出 image，
下游 VAEDecode 发现已有 image 就透传 —— 保证图仍然跑得通。
"""
from __future__ import annotations

import logging
import random

from workflow import adapters as adapters_mod
from workflow.registry import node
from workflow.schema import (DataType, ParamSpec, PortSpec,
                             ResourceRequirement)
from workflow.values import Latent, as_image

logger = logging.getLogger(__name__)

# 采样器与 sigma 风格的**唯一来源**是 core/samplers.py：界面下拉、节点参数、
# 以及 model_manager 真正构造调度器的地方都读它，避免"界面能选、底层悄悄
# 换成 Euler a"（这正是修之前的状态）。
from core import samplers as samplers_mod  # noqa: E402

_SAMPLERS = samplers_mod.sampler_names()
_SCHEDULERS = samplers_mod.sigma_styles()


def _resolve_seed(seed: int) -> int:
    if seed is None or int(seed) < 0:
        return random.randint(1, 2_147_483_647)
    return int(seed)


def _generator(device: str, seed: int):
    import torch
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def _extract_latent(res):
    """从管线返回值里取 latent；取不到返回 None。"""
    try:
        import torch
    except Exception:
        return None
    lat = getattr(res, "latents", None)
    if isinstance(lat, torch.Tensor):
        return lat
    imgs = getattr(res, "images", None)
    if isinstance(imgs, torch.Tensor):
        return imgs
    return None


# ----------------------------------------------------------------------
#  IP-Adapter：多参考图 → 管线入参
# ----------------------------------------------------------------------
def _is_offloaded(module) -> bool:
    """模块是否已被 accelerate 接管（CPU offload）。

    被接管后权重可能是 meta 占位张量，手动 `.to()` 会抛
    `NotImplementedError: Cannot copy out of meta tensor; no data!`
    —— 这个报错跟"解码"毫无关系，用户完全看不出原因。
    与 `ModelManager._is_offloaded` 同一判定（`_hf_hook` 是否存在），
    这里复制一份是为了让采样节点不反向依赖模型管理器。
    """
    return module is not None and getattr(module, "_hf_hook", None) is not None


def _place_vae(vae, device: str) -> bool:
    """把 VAE 安置到指定设备；**不安全的搬动一律跳过**。

    返回是否真的搬了。跳过（返回 False）的两种情况：
      · 已被 accelerate 接管（`_hf_hook`）：手动 `.to()` 会绕过 hook 的
        权重簿记，下次 forward 时它又搬回去 —— 白搬一趟还污染状态；
      · 权重是 meta 占位张量：`.to()` 直接抛 `NotImplementedError:
        Cannot copy out of meta tensor; no data!`（实测遇到过一次，
        报错完全指不到原因）。

    为什么要在这里做：SD15 的 VAE 通常不在 offload 计划里，但
    `enable_model_cpu_offload()` / sequential offload 一旦启用，
    或者显存吃紧时 accelerate 仍可能接管它。那时任何 `.to()` 都会炸。
    """
    if vae is None or not device:
        return False
    if _is_offloaded(vae):
        logger.debug("ℹ️ VAE 已被 offload 接管，跳过手动搬运")
        return False
    try:
        if str(getattr(vae, "device", "")) == "meta":
            logger.debug("ℹ️ VAE 权重是 meta 占位张量，跳过手动搬运")
            return False
        vae.to(device)
        return True
    except Exception as e:
        logger.debug(f"ℹ️ VAE 搬到 {device} 失败（忽略，交给 hook）: {e}")
        return False


def _vae_scale_factor(pipe) -> int:
    """潜空间下采样倍率（转发到 `ipa_multi.vae_scale_factor`，唯一实现）。"""
    from core import ipa_multi as ipa_mod
    return ipa_mod.vae_scale_factor(pipe)


def _dedupe_ip_adapter_warnings(warnings, ctx):
    """合并与上游重复的降级警告（角色节点当时已经写过同样一条）。

    同一个"缺 plus-face"在元数据里出现两次，只会让用户以为出了两个问题。
    判定依据：降级警告文本里同时出现「角色中文名」和「需要的档位名」，且这个
    (档位, 角色) 组合上游已经报过 —— 按**精确组合**判，不做"档位名是否出现
    过"这种模糊匹配（那会把另一条真实警告吞掉）。
    """
    from core import ipa_multi as ipa_mod

    reported = {(str(d.get("variant")), str(d.get("role")))
                for d in (ctx.metadata.get("ipa_degraded") or [])}
    if not reported:
        return list(warnings)
    out = []
    for w in warnings:
        dup = any(
            (variant, role) in reported
            and ipa_mod.role_label(role) in w
            and f"「{variant}」" in w
            for (variant, role) in reported)
        if dup:
            logger.info(f"ℹ️ 降级警告与上游重复，已合并: {w[:48]}…")
            continue
        out.append(w)
    return out


def apply_ip_adapter_kwargs(ctx, model, kw, *, width: int, height: int,
                            pipe=None):
    """把累积的多参考图变成管线入参（**一次安装 N 个适配器**）。

    为什么要放在采样前一刻才做（而不是在每个 IP-Adapter 节点里安装）：
    只有到了这里才知道**一共有几条**参考图。链式串接
    IPAdapterIdentity → IPAdapterStyle → IPAdapterComposition 时，
    在第一个节点就装权重只能装一个；采样前统一装才能做到"一次安装、
    每适配器一个 scale"，而不是跑三遍采样。

    返回 `(kw, info_dict)`；info 里带本轮实际用的档位/权重/警告，
    供节点写进元数据。任何失败都抛 RuntimeError（**带中文原因**），
    因为"用户明确要了 IP-Adapter 但没生效"必须让人看见。

    ⚠️ 另一个必须做的检查：只要 UNet 上装了 IPA（`encoder_hid_proj` 是
    `MultiIPAdapterImageProjection`），**每次调用都必须给 ip_adapter_image**。
    否则 diffusers 会在 UNet 里抛：
        TypeError: argument of type 'NoneType' is not iterable
        （unet_2d_condition.py:965 `if "image_embeds" not in added_cond_kwargs`
          而 added_cond_kwargs 此时是 None）
    实测：装好 base 后不给图 → 立刻抛上面这个 TypeError；给了图 → 正常出图。
    所以这里宁可抛一条中文错误，也不让用户看到那个莫名其妙的 TypeError。
    """
    from core import ipa_multi as ipa_mod
    from nodes import ipa_nodes

    ai = ctx.ai
    refs = ipa_nodes.pending_references(model) if model is not None else []
    refs = ipa_mod.enabled_refs(refs)

    if not refs:
        # 没接参考图：若 UNet 上仍残留 IPA（同一会话上一轮装的），
        # 直接生成会抛 TypeError —— 这里给一条能看懂的错误。
        for p in (getattr(ai, "txt2img_pipe", None),
                  getattr(ai, "img2img_pipe", None),
                  getattr(ai, "inpaint_pipe", None)):
            if p is None:
                continue
            proj = getattr(getattr(p, "unet", None), "encoder_hid_proj", None)
            if type(proj).__name__ == "MultiIPAdapterImageProjection":
                if getattr(ai, "ip_adapter_loaded", False):
                    raise RuntimeError(
                        "❌ 当前底模上已经装好 IP-Adapter，但本次采样没有任何"
                        "参考图。\n"
                        "   装了 IP-Adapter 的管线**每次调用都必须提供参考图**"
                        "（否则 diffusers 会在 UNet 里报一个与 IP-Adapter "
                        "无关的 TypeError: 'NoneType' object is not iterable）。\n"
                        "   请接一个 IPAdapterIdentity / IPAdapterStyle / "
                        "IPAdapterComposition 节点，或先在界面上取消 IP-Adapter。")
                break

    if not refs:
        return kw, {}

    if ai is None:
        raise RuntimeError(
            "缺少模型管理器（ctx.ai 为空），无法安装 IP-Adapter。")

    # ── 七步流程走唯一实现：core.ipa_multi.build_runtime ──
    # 这里只负责"节点的语义"：失败抛中文错误（主界面那条是降级返回 None），
    # 以及把降级警告与上游 `ipa_degraded` 去重、把"起止步不生效"如实说出来。
    ctx.status(f"🎭 安装 {len(refs)} 个 IP-Adapter "
               f"（{ipa_mod.describe_references(refs)}）")
    ctx.check_cancel()
    rt = ipa_mod.build_runtime(
        refs, ai, width=width, height=height, pipe=pipe,
        on_missing="raise", device="cpu")
    if not rt.ok:
        raise RuntimeError(
            f"❌ 挂载 IP-Adapter 失败：{rt.error or '未知原因'}\n"
            "   常见原因：权重未安装 —— 请在**扩展市场**安装"
            "（asset: ip_adapter_sd15 / ip_adapter_sdxl），\n"
            "   或手动把 ip-adapter-*.safetensors 放到 models/ip_adapter/ 下\n"
            "（还需 image_encoder/ 目录）。\n"
            "   也可能是变体与当前底模架构不匹配（SD1.5 ⇄ SDXL）。\n"
            "   本节点不会自动下载任何权重。")
    warnings = _dedupe_ip_adapter_warnings(rt.warnings, ctx)

    # 先如实说明"起止步目前不生效"（diffusers 0.40 没有逐参考图步数窗口）。
    _meta = getattr(model, "meta", None) or {}
    ranges = list(_meta.get("ipa_ranges") or [])
    odd = [f"{a:.2f}~{b:.2f}" for (a, b) in ranges if (a, b) != (0.0, 1.0)]
    if odd:
        warnings.append(
            f"参考图起止步 {', '.join(odd)} 已记录，但当前 diffusers 0.40 的 "
            f"IP-Adapter 只支持整段生效（不支持逐参考图步数窗口，"
            f"也没有逐步回调可改 scale）→ 本次采样中这些参考图**全程生效**。")

    # ── 管线入参 ──
    # ⚠️ 装了 IPA 的管线**每次调用都必须给 ip_adapter_image**，且长度必须等于
    #    适配器数量（否则 diffusers 会在 UNet 里报 TypeError / ValueError）。
    kw["ip_adapter_image"] = rt.images
    if rt.masks is not None:
        existing = dict(kw.get("cross_attention_kwargs") or {})
        existing["ip_adapter_masks"] = rt.masks
        kw["cross_attention_kwargs"] = existing

    info = dict(rt.info)
    if warnings:
        info["ip_adapter_warnings"] = list(warnings)
        existing_w = list(ctx.metadata.get("warnings") or [])
        existing_w.extend(warnings)
        ctx.metadata["warnings"] = existing_w
    ctx.metadata.update({k: v for k, v in info.items()
                         if k != "ip_adapter_warnings"})
    for w in warnings:
        logger.warning(f"⚠️ {w}")
        try:
            ctx.log(f"⚠️ {w}")
        except Exception:
            pass
    return kw, info


@node(
    "Sampler",
    category="采样",
    description="采样：文生图 / 图生图 / 局部重绘（有 image 即图生图，再有 mask 即重绘）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "positive": PortSpec(DataType.TEXT_COND, "正面条件"),
            "negative": PortSpec(DataType.TEXT_COND, "负面条件", optional=True),
            "image": PortSpec(DataType.IMAGE, "初始图", optional=True),
            "mask": PortSpec(DataType.MASK, "遮罩", optional=True)},
    outputs={"latent": PortSpec(DataType.LATENT, "潜空间"),
             "image": PortSpec(DataType.IMAGE, "图像", optional=True),
             "metadata": PortSpec(DataType.METADATA, "信息")},
    params={
        "steps": ParamSpec("int", default=28, label="步数",
                           minimum=1, maximum=150),
        "cfg": ParamSpec("float", default=7.0, label="CFG",
                         minimum=0.0, maximum=30.0, step=0.5),
        "seed": ParamSpec("seed", default=-1, label="种子",
                          help="-1 表示随机"),
        "width": ParamSpec("int", default=512, label="宽",
                           minimum=64, maximum=8192),
        "height": ParamSpec("int", default=768, label="高",
                            minimum=64, maximum=8192),
        "sampler": ParamSpec("choice", default="Euler a", label="采样器",
                             choices=_SAMPLERS,
                             help="真正切换 diffusers 调度器；缺依赖的采样器"
                                  "（如没装 torchsde 的 DPM++ SDE）不会出现在"
                                  "候选里"),
        "scheduler": ParamSpec("choice", default="默认", label="sigma 风格",
                               choices=_SCHEDULERS,
                               help="默认 / Karras / Exponential / Beta；"
                                    "只对支持的调度器生效（DDIM/PNDM/LCM 会被"
                                    "自动忽略）"),
        "eta": ParamSpec("float", default=0.0, label="eta（随机性）",
                         minimum=0.0, maximum=1.0, step=0.1,
                         help="只对 DDIM / LMS 的图生图生效；0=确定性，"
                              "1=完全重新采样"),
        "strength": ParamSpec("float", default=0.55, label="重绘强度",
                              minimum=0.0, maximum=1.0, step=0.05),
    },
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=2600,
                                  supports_cpu_offload=True,
                                  supports_tiled=True,
                                  exclusive_gpu=True, needs_model=True),
)
def run_sampler(ctx, node, inputs):
    import torch

    model = inputs.get("model")
    if model is None:
        raise ValueError(ctx.translate("缺少输入 model"))
    pos = inputs.get("positive")
    neg = inputs.get("negative")
    image = as_image(inputs.get("image"))
    mask = inputs.get("mask")

    adapter = model.adapter
    kind = "txt2img"
    if image is not None:
        kind = "inpaint" if mask is not None else "img2img"
    pipe = adapter.pipe(model, kind)
    if pipe is None:
        raise RuntimeError(ctx.translate(
            "模型没有可用的 {kind} 管线（该架构可能不支持）").format(kind=kind))
    if kind != "txt2img" and not getattr(adapter.caps, kind, True):
        # 抛专用异常：UI/预检据此区分"能力不支持"与"运行出错"
        raise adapters_mod.UnsupportedFeatureError(
            {"img2img": "图生图", "inpaint": "局部重绘"}.get(kind, kind),
            model.arch_id)

    steps = int(node.params.get("steps", 28))
    cfg = float(node.params.get("cfg", adapter.default_guidance()))
    width = int(node.params.get("width", 512))
    height = int(node.params.get("height", 768))
    seed = _resolve_seed(node.params.get("seed", -1))
    sampler = (node.params.get("sampler") or "").strip()
    scheduler = (node.params.get("scheduler") or "").strip()

    if sampler:
        try:
            ctx.ai.switch_sampler(sampler, scheduler or None)
        except Exception as e:
            # 缺依赖 / 未知名字：明确报出来，但**不让整张图白跑**——
            # 沿用当前调度器并在元数据里留一条警告。
            msg = f"采样器「{sampler}」不可用（{e}），本次沿用当前调度器"
            logger.warning(f"⚠️ {msg}")
            ctx.metadata.setdefault("warnings", []).append(msg)

    kw = adapter.sample_kwargs(pos, steps=steps, cfg=cfg, seed=seed,
                               width=width, height=height)
    # 负面条件优先用独立输入
    if neg is not None and getattr(neg, "negative_embeds", None) is not None:
        kw["negative_prompt_embeds"] = neg.negative_embeds
        if getattr(neg, "negative_pooled", None) is not None:
            kw["negative_pooled_prompt_embeds"] = neg.negative_pooled

    if kind in ("img2img", "inpaint"):
        kw["image"] = image
        kw["strength"] = float(node.params.get("strength", 0.55))
        if kind == "inpaint":
            kw["mask_image"] = mask
        # 图生图尺寸跟输入图，避免被拉伸
        kw.pop("width", None)
        kw.pop("height", None)
        width, height = image.width, image.height

    gen = _generator(ctx.device, seed)
    kw["generator"] = gen

    # ── IP-Adapter：把累积的多参考图变成"一次安装 N 个适配器 + 每图独立强度" ──
    # 放在这里（采样前一刻）的原因见 apply_ip_adapter_kwargs 的文档。
    # 失败直接抛（带中文原因）—— 用户明确要了 IP-Adapter，静默跳过比报错更糟。
    kw, ipa_info = apply_ip_adapter_kwargs(
        ctx, model, kw, width=width, height=height, pipe=pipe)

    _eta = float(node.params.get("eta", 0.0) or 0.0)
    _sdef = samplers_mod.find_sampler(sampler)
    if kind in ("img2img", "inpaint") and _sdef is not None and _sdef.uses_eta:
        kw["eta"] = _eta        # 其它采样器/文生图不接受 eta，不塞
    ctx.status(f"🎨 采样 {steps} 步 · {width}x{height} · seed {seed}"
               f" · {samplers_mod.describe(sampler, scheduler)}")
    latent = None
    out_image = None

    def _call(**extra):
        with torch.inference_mode():
            return pipe(**{**kw, **extra})

    try:
        res = _call(output_type="latent")
        latent = _extract_latent(res)
    except (TypeError, ValueError) as e:
        # 管线不认 output_type="latent"：下面用 latent is None 走直接解码
        logger.info(f"ℹ️ 该管线不支持 output_type=latent，改为直接解码: {e}")
    ctx.check_cancel()

    if latent is None:
        res = _call()
        imgs = getattr(res, "images", None)
        if imgs is None:
            raise RuntimeError(ctx.translate("管线没有返回图像"))
        if isinstance(imgs, torch.Tensor):
            latent = imgs
        else:
            out_image = imgs[0] if isinstance(imgs, (list, tuple)) else imgs

    meta = {"steps": steps, "cfg": cfg, "seed": seed, "sampler": sampler,
            "width": width, "height": height, "workflow_kind": kind,
            "strength": float(node.params.get("strength", 0.55))
            if kind != "txt2img" else ""}
    if ipa_info:
        meta.update({k: v for k, v in ipa_info.items()})
    ctx.metadata.update({k: v for k, v in meta.items()
                         if k != "ip_adapter_warnings"})
    ctx.progress(steps, steps)

    return {"latent": Latent(tensor=latent, is_noise=False),
            "image": out_image, "metadata": meta}


@node(
    "VAEEncode",
    category="潜空间",
    description="把图像编码到潜空间（图生图 / 重绘 / 分块放大的入口）。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "image": PortSpec(DataType.IMAGE, "图像")},
    outputs={"latent": PortSpec(DataType.LATENT, "潜空间")},
    params={},
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=1200,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_vae_encode(ctx, node, inputs):
    import torch

    model = inputs.get("model")
    image = as_image(inputs.get("image"))
    if model is None or image is None:
        raise ValueError(ctx.translate("缺少 model 或 image"))

    vae = getattr(model, "vae", None) or getattr(model.pipe, "vae", None)
    if vae is None:
        raise RuntimeError(ctx.translate("模型没有 VAE，无法编码"))

    # ⚠️ VAE 若已被 accelerate 接管（CPU offload）或权重是 meta 占位张量，
    # 手动 `.to()` 会抛 `NotImplementedError: Cannot copy out of meta tensor;
    # no data!` —— 这个报错与"编码"毫无关系，用户根本看不出原因。
    _place_vae(vae, getattr(ctx, "device", "") or "")

    import numpy as np
    with torch.inference_mode():
        arr = np.asarray(image.convert("RGB"), dtype="float32") / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(device=vae.device, dtype=vae.dtype)
        lat = vae.encode(t).latent_dist.sample()
        if hasattr(vae.config, "scaling_factor"):
            lat = lat * vae.config.scaling_factor
    return {"latent": Latent(tensor=lat)}


@node(
    "VAEDecode",
    category="潜空间",
    description="把潜空间解码成图像。若上游已给出图像则直接透传。",
    inputs={"model": PortSpec(DataType.MODEL, "模型"),
            "latent": PortSpec(DataType.LATENT, "潜空间", optional=True),
            "image": PortSpec(DataType.IMAGE, "图像", optional=True)},
    outputs={"image": PortSpec(DataType.IMAGE, "图像")},
    params={},
    resources=ResourceRequirement(requires=("cuda", "model"),
                                  estimated_vram_mb=1400,
                                  supports_cpu_offload=True,
                                  supports_tiled=True),
)
def run_vae_decode(ctx, node, inputs):
    import torch

    passthrough = as_image(inputs.get("image"))
    if passthrough is not None:
        return {"image": passthrough}     # 上游已解码，无需再来一次

    model = inputs.get("model")
    lat = inputs.get("latent")
    if model is None or lat is None or getattr(lat, "tensor", None) is None:
        raise ValueError(ctx.translate("缺少 latent（也没有可透传的 image）"))

    vae = getattr(model, "vae", None) or getattr(model.pipe, "vae", None)
    if vae is None:
        raise RuntimeError(ctx.translate("模型没有 VAE，无法解码"))

    t = lat.tensor
    if hasattr(vae.config, "scaling_factor"):
        t = t / vae.config.scaling_factor
    # ⚠️ 同 run_vae_encode：被 offload 接管的 VAE 不能手动 `.to()`
    # （meta 张量会抛 NotImplementedError，报错完全指不到原因）。
    _place_vae(vae, getattr(ctx, "device", "") or "")
    with torch.inference_mode():
        t = t.to(device=vae.device, dtype=vae.dtype)
        img = vae.decode(t).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).round()
    from PIL import Image
    return {"image": Image.fromarray(arr.astype("uint8"), "RGB")}
