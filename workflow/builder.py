# workflow/builder.py
"""把现有"引导式界面"的状态编译成工作流图。

这是"简易模式 = 自动生成的工作流"的落地点：
用户在界面上点 文生图 → 选模型 → 提示词 → LoRA → ControlNet → 高清 → 修脸，
这里生成等价的一张图；高级模式下看到的、能改的就是这张图。

为什么要复用 `_gt_prepare_context()`
-----------------------------------
那个函数已经把「界面状态 → 生成参数」的全部规则算清楚了（提示词解析、
翻译、LoRA 标签、ControlNet 力度、IPA 开关、尺寸、种子…）。
builder 只做**翻译**：dict → 图。否则等于把同一套规则再写一遍，
迟早两处不一致 —— 这正是本项目之前"两套输出目录""两套元数据"的病根。

⚠️ 构造节点时必须写 `Node(id, type, inputs={}, params={...})`
-----------------------------------------------------------
`Node` 的签名是 `Node(id, type, inputs={}, params={})` —— **第三个位置参数是
inputs**。之前这里写成了 `Node(id, type, {所有参数})`，于是全部参数都进了
inputs、`params` 全空，生成的工作流跑起来一律用默认值（提示词变空、步数 28、
种子随机、水印开关失效）。只有 `LoadCheckpoint.path` 因为节点里有
`inputs.get("path")` 兜底才侥幸没丢。

现在两处保险：
  1. 每个节点都显式 `inputs={}, params={...}`；
  2. 末尾再调一次 `serializers.normalize_params()`（幂等），
     将来若有人又写错位置也能自动归位。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from workflow import registry
from workflow.schema import Node, Workflow

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE = "lowscore, bad anatomy, worst quality, low quality"


def _txt(w, default="") -> str:
    """从各种控件上取字符串（QLineEdit/QTextEdit/QComboBox 都兼容）。"""
    if w is None:
        return default
    for meth in ("toPlainText", "text", "currentText"):
        f = getattr(w, meth, None)
        if callable(f):
            try:
                return str(f())
            except Exception:
                continue
    return default


def _num(w, default=0.0, float_value=True) -> float:
    if w is None:
        return default
    try:
        if float_value and hasattr(w, "float_value"):
            return float(w.float_value())
        if hasattr(w, "value"):
            return float(w.value())
    except Exception:
        pass
    return default


def _checked(w, default=False) -> bool:
    try:
        return bool(w.isChecked())
    except Exception:
        return default


def _add(host_wf: Workflow, node_id: str, ntype: str,
         params: Optional[Dict[str, Any]] = None) -> Node:
    """统一的加节点入口：**参数一律走 params**，inputs 留给连线。"""
    return host_wf.add(Node(node_id, ntype, {}, dict(params or {})))


def _add_image_source(wf: Workflow, node_id: str, path: str) -> Optional[str]:
    """为「需要一张图」的输入建一个 LoadImage 源；拿不到路径返回 None。

    ControlNetPreprocess / IPAdapterApply 的 `image` 是**必填**输入。
    没有这个源节点，只要用户勾了姿态控制或 IP-Adapter，生成出来的图
    就一定校验失败（"缺少必填输入 image"），整条链路走不通。
    """
    p = (path or "").strip()
    if not p:
        return None
    if not os.path.isabs(p):
        try:
            from utils import paths
            cand = os.path.join(paths.PROJECT_ROOT, p)
            p = cand if os.path.exists(cand) else p
        except Exception:
            pass
    if not os.path.exists(p):
        logger.warning(f"[builder] 参考图不存在，跳过图片源: {p}")
        return None
    _add(wf, node_id, "LoadImage", {"path": p})
    return node_id


def _parse_lora_tag(tag: str):
    """`<lora:名称:0.8>` → ("名称", 0.8)。解析不了就原样返回、权重 1.0。"""
    s = str(tag or "").strip()
    if s.startswith("<lora:") and s.endswith(">"):
        body = s[len("<lora:"):-1]
        parts = body.split(":")
        if len(parts) >= 2:
            try:
                return parts[0], float(parts[1])
            except ValueError:
                return parts[0], 1.0
        if parts:
            return parts[0], 1.0
    return s, 1.0


def _controlnet_kind(host, c) -> str:
    """推断当前 ControlNet 类型（优先模型管理器上的实际值）。"""
    ai = getattr(host, "ai", None)
    for src in (getattr(ai, "current_cn_type", None), c.get("cn_type")):
        if src:
            return str(src)
    return "openpose" if c.get("use_pose") else ""


def _host_image_path(host, *attrs) -> str:
    for a in attrs:
        v = getattr(host, a, None)
        if v and isinstance(v, str) and os.path.exists(v):
            return v
    return ""


def _resource_ref_params(name_or_path: str) -> Dict[str, Any]:
    """给模型节点附上**资源引用字段**（resource_id 等），让工作流可跨机器迁移。

    为什么不能只存路径：绝对路径换台电脑就失效。存 `resource_id` +
    `filename` + `architecture` + `sha256` 之后，工作流在任何机器上都能反查
    "我依赖哪个模型"，缺失时也能明确告诉用户缺什么。
    `local_path` 默认写 None —— 这正是"可迁移"的关键。
    `resource_store` 不可用时返回空 dict（只留 `path`），不影响生成。
    """
    if not name_or_path:
        return {}
    try:
        from workflow import resource_store as RS
        base = os.path.basename(str(name_or_path))
        r = RS.by_filename(base) or RS.by_filename(str(name_or_path))
        if r is None:
            return {"filename": base}
        return RS.to_ref(r)
    except Exception as e:
        logger.debug(f"[builder] 资源引用生成失败（忽略）: {e}")
        return {}


def _hires_mode(host, cfg) -> str:
    """决定高清修复走「整图」还是「分块」。

    优先级：界面下拉 > config > 自动（按显存）。
    8GB 卡整图 img2img 放大 2x 非常容易爆显存/疯狂换页，所以自动档
    在总量 <10GB 时选**分块** —— 这正是分块高清存在的意义。
    """
    ui = _txt(getattr(host, "combo_hires_mode", None), "")
    if "分块" in ui:
        return "tiled"
    if "整图" in ui:
        return "latent"
    mode = str(getattr(cfg, "hires_mode", "auto") or "auto").lower()
    if mode in ("latent", "tiled"):
        return mode
    try:
        from workflow.memory_policy import DeviceCapabilities
        if DeviceCapabilities.detect().total_gb < 10.0:
            return "tiled"
    except Exception:
        pass
    return "latent"


def _hires_order(host, cfg) -> str:
    """高清与修脸的先后顺序。"""
    ui = _txt(getattr(host, "combo_hires_order", None), "")
    if "先修脸" in ui:
        return "adetailer_first"
    if "先高清" in ui:
        return "hires_first"
    order = str(getattr(cfg, "hires_order", "hires_first") or "hires_first")
    return order if order in ("hires_first", "adetailer_first") else "hires_first"


def build_from_context(host, c: Dict[str, Any], *,
                       name: str = "简易模式流程",
                       kind: Optional[str] = None) -> Workflow:
    """把 `_gt_prepare_context()` 的产物编译成工作流图。"""
    registry.load_builtin_nodes()          # normalize_params 需要注册表
    wf = Workflow(name=name)
    wf.metadata["source"] = "easy_mode"
    wf.metadata["generated"] = True
    warnings = []

    prompts = c.get("parsed_raw_prompts") or [""]
    en_prompts = c.get("en_prompts") or prompts
    raw_p = prompts[0] if prompts else ""
    final_p = en_prompts[0] if en_prompts else raw_p
    raw_n = c.get("raw_neg", "") or ""
    final_n = c.get("en_neg", "") or raw_n

    cfg = getattr(host, "config", None)

    model_id = "model_1"
    cp_params: Dict[str, Any] = {
        "path": c.get("model_name", ""),
        "clip_skip": int(getattr(cfg, "clip_skip", 0) or 0),
    }
    # 注意：**不再**往 LoadCheckpoint 写 `sampler`。采样器属于 Sampler 节点，
    # 两个地方都能设会互相覆盖（用户已经指出这个框该删）。
    # 附上资源引用（resource_id / architecture / sha256 / local_path=None），
    # 这样工作流存的是"依赖哪个模型"而不是"我的硬盘上在哪"
    cp_params.update(_resource_ref_params(c.get("model_name", "")))
    _add(wf, model_id, "LoadCheckpoint", cp_params)

    # ---- LoRA ----
    # 优先用 _gt_aux_state 给出的 (名字, 权重) 列表（与真实生成同一套读数）；
    # 没有时退回 lora_meta_info 的 "名字:权重" 字符串形式
    loras = list(c.get("lora_list") or [])
    if not loras:
        for t in (c.get("lora_meta_info") or []):
            nm, wt = _parse_lora_tag(t)
            if nm and nm != "无":
                loras.append((nm, wt))
    lora_id = None
    if loras:
        lora_id = "lora_1"
        params: Dict[str, Any] = {}
        for i, item in enumerate(loras[:3], start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                nm, wt = str(item[0]), float(item[1])
            else:
                nm, wt = _parse_lora_tag(item)
            params[f"lora_{i}"] = nm
            params[f"weight_{i}"] = wt
        _add(wf, lora_id, "LoadLoRA", params)
        wf.connect(model_id, "model", lora_id, "model")

    src = lora_id or model_id

    # ---- 初始图（图生图）----
    init_path = c.get("init_image_path") or ""
    if init_path:
        init_src = _add_image_source(wf, "init_image_1", init_path)
        if init_src is None:
            warnings.append("图生图参考图路径无效，已按文生图生成")

    # ---- ControlNet（姿态/骨架等）----
    cn_kind = _controlnet_kind(host, c)
    if c.get("use_pose") and cn_kind:
        pose_path = (_host_image_path(host, "pose_image_path")
                     or c.get("pose_image_path") or init_path)
        pre_src = _add_image_source(wf, "control_image_1", pose_path)
        if pre_src is None:
            # 关键：image 是必填输入。拿不到参考图就**不要**生成这两个节点，
            # 否则工作流必然校验失败、用户点运行只会看到"校验未通过"。
            warnings.append(
                f"ControlNet「{cn_kind}」已启用，但找不到参考图/姿势图，"
                "本次未加入工作流（请先选择图片）")
            logger.warning("[builder] %s", warnings[-1])
        else:
            cn_id = "control_1"
            pre_id = "control_pre_1"
            weight = float(c.get("cn_strength", 0.6) or 0.6)
            _add(wf, pre_id, "ControlNetPreprocess",
                 {"kind": cn_kind, "weight": weight})
            _add(wf, cn_id, "ControlNetApply", {"weight": weight})
            wf.connect(pre_src, "image", pre_id, "image")
            wf.connect(pre_id, "control", cn_id, "control")
            wf.connect(src, "model", cn_id, "model")
            src = cn_id

    # ---- IP-Adapter ----
    # ① 多参考图（主界面「🎭 多参考图」）优先：每张图编译成一个角色节点
    #    （IPAdapterIdentity / IPAdapterStyle / IPAdapterComposition），
    #    串联起来由采样层**一次**装 N 个适配器。
    # ② 没有多参考图时才走旧的「单张 IPAdapterApply」。
    #    ⚠️ 不能两个都发：`load_ip_adapter()` 第二次调用是**覆盖**语义，
    #    两条路同时存在会把多参考图整体冲掉（界面上看起来"设了没用"）。
    used_multi_refs = False
    ref_entries = [r for r in (c.get("ipa_refs")
                               or getattr(cfg, "ipa_refs", None) or [])
                   if isinstance(r, dict) and r.get("enabled", True)]
    use_refs = bool(c.get("use_ipa_refs")
                    or (getattr(cfg, "use_ipa_refs", False) and ref_entries))
    if use_refs and ref_entries:
        try:
            from core import ipa_multi as ipa_mod
        except Exception as e:                       # pragma: no cover - 防御
            logger.warning(f"[builder] 读取多参考图角色表失败: {e}")
            ipa_mod = None
        if ipa_mod is not None:
            ref_src = src
            added = 0
            for _i, _r in enumerate(ref_entries, start=1):
                role = ipa_mod.normalize_role(_r.get("role"))
                label = ipa_mod.role_label(role)
                node_type = {
                    "identity": "IPAdapterIdentity",
                    "style": "IPAdapterStyle",
                    "composition": "IPAdapterComposition",
                }.get(role, "IPAdapterStyle")
                _path = str(_r.get("path") or "")
                _src_id = _add_image_source(wf, f"ipa_ref_image_{_i}", _path)
                if _src_id is None:
                    warnings.append(
                        f"多参考图第 {_i} 张（{label}）找不到图片文件，"
                        "本次未加入工作流（请重新选择图片）")
                    logger.warning("[builder] %s", warnings[-1])
                    continue
                try:
                    _weight = float(_r.get("weight",
                                           ipa_mod.DEFAULT_WEIGHT))
                except (TypeError, ValueError):
                    _weight = float(ipa_mod.DEFAULT_WEIGHT)
                _id = f"ipa_ref_{_i}"
                _add(wf, _id, node_type, {
                    "weight": max(0.0, min(2.0, _weight)),
                    "enabled": True,
                    "start_at": 0.0, "end_at": 1.0})
                wf.connect(ref_src, "model", _id, "model")
                wf.connect(_src_id, "image", _id, "image")
                ref_src = _id
                added += 1
            if added:
                src = ref_src
                used_multi_refs = True
                wf.metadata["ipa_refs"] = added
                # 与界面摘要同一套说法（"3 张参考：身份(0.50) 风格(0.70)"）
                try:
                    from ui.ipa_editor import summary_text
                    wf.metadata["ipa_refs_summary"] = summary_text(ref_entries)
                except Exception:
                    pass
            else:
                warnings.append("多参考图已启用，但没有一张参考图路径可用，"
                                "本次未加入工作流（请重新选择图片）")
                logger.warning("[builder] %s", warnings[-1])

    if c.get("use_ipa") and not used_multi_refs:
        ipa_path = (_host_image_path(host, "ipa_image_path", "ref_image_path")
                    or init_path)
        ipa_src = _add_image_source(wf, "ipa_image_1", ipa_path)
        if ipa_src is None:
            warnings.append("IP-Adapter 已启用，但找不到参考图，"
                            "本次未加入工作流（请先选择参考图）")
            logger.warning("[builder] %s", warnings[-1])
        else:
            ipa_id = "ipa_1"
            _add(wf, ipa_id, "IPAdapterApply", {
                "variant": c.get("ipa_variant", "plus") or "plus",
                "weight": float(c.get("ipa_scale", 0.6) or 0.6)})
            wf.connect(src, "model", ipa_id, "model")
            wf.connect(ipa_src, "image", ipa_id, "image")
            src = ipa_id

    # ---- 文本条件 ----
    pos_id, neg_id = "prompt_1", "neg_1"
    _add(wf, pos_id, "TextEncode", {"text": raw_p, "text_final": final_p})
    wf.connect(src, "model", pos_id, "model")
    _add(wf, neg_id, "NegativeTextEncode",
         {"text": raw_n or DEFAULT_NEGATIVE, "text_final": final_n})
    wf.connect(src, "model", neg_id, "model")

    # ---- 采样 ----
    sampler_id = "sampler_1"
    _add(wf, sampler_id, "Sampler", {
        "steps": int(c.get("steps", 28) or 28),
        "cfg": float(c.get("cfg", 7.0) or 7.0),
        "seed": int(c.get("seed", -1) if c.get("seed") is not None else -1),
        "width": int(c.get("width", 512) or 512),
        "height": int(c.get("height", 768) or 768),
        "sampler": c.get("sampler_name", "") or "Euler a",
        "strength": float(c.get("strength", 0.55) or 0.55),
    })
    wf.connect(src, "model", sampler_id, "model")
    wf.connect(pos_id, "conditioning", sampler_id, "positive")
    wf.connect(neg_id, "conditioning", sampler_id, "negative")
    if init_path:
        init_src = wf.node("init_image_1")
        if init_src is not None:
            wf.connect("init_image_1", "image", sampler_id, "image")

    # ---- 解码 ----
    decode_id = "decode_1"
    _add(wf, decode_id, "VAEDecode", {})
    wf.connect(src, "model", decode_id, "model")
    wf.connect(sampler_id, "latent", decode_id, "latent")
    # Sampler 在"管线不支持 latent 输出"时会直接给 image，一并接上做透传
    wf.connect(sampler_id, "image", decode_id, "image")

    cur = decode_id

    # ---- 后处理顺序：高清 与 修脸 谁先谁后由用户决定 ----
    # 两种顺序对人脸效果不同：
    #   先高清后修脸 → 放大后再修脸，脸更锐利
    #   先修脸后高清 → 先修好脸再放大，脸更自然（放大时可能被二次柔化）
    hires_on = _checked(getattr(host, "chk_hires", None),
                        bool(getattr(cfg, "use_hires", False)))
    ad_face = _checked(getattr(host, "chk_use_adetailer", None))
    ad_hand = _checked(getattr(host, "chk_use_ad_hand", None))
    ad_on = bool(ad_face or ad_hand)

    def _add_hires():
        nonlocal cur
        h_id = "hires_1"
        scale = _txt(getattr(host, "combo_hires_scale", None), "2.0")
        denoise = _num(getattr(host, "scale_hires_denoise", None),
                       float(getattr(cfg, "hires_denoise", 0.45) or 0.45))
        mode = _hires_mode(host, cfg)
        steps = int(c.get("steps", 24) or 24)
        seed = int(c.get("seed", -1) if c.get("seed") is not None else -1)
        if mode == "tiled":
            # 分块：8GB 卡也能出 2K~4K（整图 img2img 很容易爆显存）
            _add(wf, h_id, "TiledHiresFix", {
                "scale": scale,
                "tile_size": int(getattr(cfg, "hires_tile_size", 640) or 640),
                "overlap": int(getattr(cfg, "hires_tile_overlap", 48) or 48),
                "denoise": denoise,
                "upscale_method": "Lanczos",
                "blend": "羽化",
                "color_correct": True,
                "vram_mode": "自动",
                "steps": steps, "cfg": float(c.get("cfg", 7.0) or 7.0),
                "seed": seed,
            })
        else:
            _add(wf, h_id, "HiresFix", {
                "scale": scale, "denoise": denoise,
                "upscaler": _txt(getattr(host, "combo_hires_upscaler", None),
                                 "Latent"),
                "steps": steps, "cfg": float(c.get("cfg", 7.0) or 7.0),
                "seed": seed,
            })
        wf.connect(src, "model", h_id, "model")
        wf.connect(cur, "image", h_id, "image")
        wf.connect(pos_id, "conditioning", h_id, "positive")
        wf.connect(neg_id, "conditioning", h_id, "negative")
        cur = h_id

    def _add_adetailer():
        nonlocal cur
        d_id = "adetailer_1"
        _add(wf, d_id, "ADetailer", {
            "target": "二次元脸部" if ad_face else "二次元手部",
            "strength": _num(getattr(host, "scale_adetailer_strength", None),
                             float(getattr(cfg, "adetailer_strength", 0.35) or 0.35)),
            "both_face_and_hand": bool(ad_face and ad_hand),
        })
        wf.connect(src, "model", d_id, "model")
        wf.connect(cur, "image", d_id, "image")
        wf.connect(pos_id, "conditioning", d_id, "positive")
        wf.connect(neg_id, "conditioning", d_id, "negative")
        cur = d_id

    order = _hires_order(host, cfg)
    if order == "adetailer_first":
        if ad_on:
            _add_adetailer()
        if hires_on:
            _add_hires()
    else:
        if hires_on:
            _add_hires()
        if ad_on:
            _add_adetailer()
    wf.metadata["post_order"] = order

    # ---- 区域提示词 / 局部重绘 ----
    # 主界面里设了区域就编译成 RegionalPrompt → RegionalRedraw，
    # 这样"简易模式生成的图"和"高级模式看到的图"仍然是同一件事。
    if getattr(cfg, "use_regions", False):
        regs = [r for r in (getattr(cfg, "regions", None) or [])
                if isinstance(r, dict) and r.get("enabled", True)]
        if regs:
            try:
                r_id, rd_id = "regions_1", "regional_1"
                _add(wf, r_id, "RegionalPrompt",
                     {"regions_json": json.dumps({"regions": regs},
                                                 ensure_ascii=False),
                      "preset_name": "简易模式区域"})
                _add(wf, rd_id, "RegionalRedraw", {
                    "steps": int(c.get("steps", 24) or 24),
                    "cfg": float(c.get("cfg", 7.0) or 7.0),
                    "seed": int(c.get("seed", -1)
                                if c.get("seed") is not None else -1),
                    "global_negative": raw_n,
                })
                wf.connect(r_id, "regions", rd_id, "regions")
                wf.connect(src, "model", rd_id, "model")
                wf.connect(cur, "image", rd_id, "image")
                wf.connect(pos_id, "conditioning", rd_id, "positive")
                wf.connect(neg_id, "conditioning", rd_id, "negative")
                cur = rd_id
                wf.metadata["regions"] = len(regs)
            except Exception as e:
                logger.warning(f"[builder] 区域节点生成失败（跳过）: {e}")

    # ---- 保存 ----
    save_id = "save_1"
    _add(wf, save_id, "SaveImage", {
        "filename_prefix": "", "format": "PNG", "watermark": True,
        "save_workflow": True})
    wf.connect(cur, "image", save_id, "image")

    # ---- 兜底：万一将来又有人把参数写进 inputs，这里自动归位 ----
    try:
        from workflow import serializers
        moved = serializers.normalize_params(wf)
        if moved:
            logger.info(f"[builder] normalize_params 归位了 {moved} 个参数")
    except Exception as e:
        logger.warning(f"[builder] normalize_params 失败（忽略）: {e}")

    wf.metadata["final_node"] = save_id
    wf.metadata["prompt"] = raw_p
    wf.metadata["seed"] = c.get("seed")
    if warnings:
        wf.metadata["warnings"] = warnings
    return wf


def build_from_ui(host, *, name: str = "简易模式流程") -> Workflow:
    """便捷入口：先让界面算出 ctx，再**补齐辅助项**，最后编译成图。

    ⚠️ 为什么必须补这一步（这是曾经的 P0 静默降级）
    ------------------------------------------------
    `_gt_prepare_context()` 返回的 ctx 里，这三项是**硬编码初值**：

        'use_ipa': False   'use_pose': False   'lora_meta_info': []

    真实值要等 `_gt_setup_ipa` / `_gt_apply_loras` / `_gt_run_pose_transfer`
    写进去，而那三个函数**只在 `start_generation` 主流程里被调用**。
    所以只调 `_gt_prepare_context()` 的话，界面勾了 LoRA / ControlNet /
    IP-Adapter，生成的图里一个对应节点都没有 —— 而且不报错不警告，
    用户以为用了，实际根本没跑。

    现在通过 `_gt_aux_state()`（纯读、无副作用）把这三项补齐：
    不加载权重、不跑推理，只是把"用户想要什么"读出来。
    """
    prep = getattr(host, "_gt_prepare_context", None)
    if not callable(prep):
        raise RuntimeError("宿主没有 _gt_prepare_context()，无法从界面编译工作流")
    c = prep()

    aux = getattr(host, "_gt_aux_state", None)
    if callable(aux):
        try:
            st = aux(c)
            c["use_ipa"] = st["use_ipa"]
            c["use_pose"] = st["use_pose"]
            c["ipa_image_path"] = st["ipa_image_path"]
            c["ipa_scale"] = st["ipa_scale"]
            c["ipa_variant"] = st["ipa_variant"]
            # 多参考图：与真实生成读的是同一份（_gt_aux_state）
            c["use_ipa_refs"] = st.get("use_ipa_refs", False)
            c["ipa_refs"] = st.get("ipa_refs", []) or []
            c["pose_image_path"] = st["pose_image_path"]
            c["lora_meta_info"] = st["lora_meta_info"]
            c["lora_list"] = st["lora_list"]
            logger.debug(
                "[builder] 辅助项补齐: lora=%s cn=%s ipa=%s ipa_refs=%s",
                len(st["lora_list"]), st["use_pose"], st["use_ipa"],
                len(st.get("ipa_refs") or []))
        except Exception as e:
            logger.warning(f"[builder] 辅助项读取失败（可能漏掉 LoRA/CN/IPA）: {e}")
    else:
        logger.warning("[builder] 宿主没有 _gt_aux_state()，"
                       "LoRA/ControlNet/IP-Adapter 可能不会进工作流")

    return build_from_context(host, c, name=name)


def describe(wf: Workflow) -> str:
    """给日志/弹窗用的一行式说明：按拓扑序把节点类型连成一条链。

    用拓扑序而不是 `wf.nodes` 的插入顺序，这样编辑器里拖过的图
    也能显示真实执行顺序。
    """
    try:
        from workflow import graph as G
        order = [nid for nid in G.topo_sort(wf)]
        types = []
        for nid in order:
            n = wf.node(nid)
            if n is not None:
                types.append(n.type)
    except Exception:
        types = [n.type for n in wf.nodes]
    return f"{wf.name}: " + " → ".join(types)
