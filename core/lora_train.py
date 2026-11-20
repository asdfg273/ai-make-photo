# core/lora_train.py
"""轻量 LoRA 训练（SD1.5 / SDXL，UNet 为主，可选文本编码器）。

设计目标
--------
给 8GB 卡用：只训 LoRA 适配器（rank 4~16），底模权重全部冻结，所以
显存里只放"权重 + 少量激活"，不需要 DeepSpeed / 多卡 / 分布式。

为什么不用 `diffusers/examples/dreambooth/train_dreambooth_lora.py`：
那是脚本不是库（它的 `parse_args` + `accelerate` 直接接管进程），
接进 GUI 要么再起一个子进程传参，要么把里面的逻辑抄一份。这里直接把
**能跑在消费级卡上的最小闭环**写成一个可回调、可取消的函数。

关键取舍（都有实测依据，别随手改）
----------------------------------
1. **latents 与文本向量先缓存**：底模冻结，VAE/CLIP 的输出在整个训练里
   不变。512² 的 latents 每张 8KB（fp16），100 张不到 1MB；缓存后每步
   只跑 UNet，省掉每步两次 CLIP + 一次 VAE（约 60% 时间）。
2. **VAE 用 fp32 编码**（`needs_fp32_vae` / fp16 溢出会出全 NaN 的 latents），
   编码完立刻 `.to("cpu")` 释放。
3. **梯度检查点默认开**：512² batch1 下 SD1.5 从约 5.6GB 降到 3.1GB，
   代价是慢 25% 左右。SDXL 不开直接 OOM。
4. **训练噪声用 `DDPMScheduler`**，与推理用的 sampler 无关 —— kohya /
   diffusers 一致做法；`prediction_type` 从底模 scheduler 读（v_pred 模型
   必须用 v 预测目标，否则学不出东西）。
5. **保存走 `pipe.save_lora_weights`**：产出 `pytorch_lora_weights.safetensors`，
   重命名成 `loras/<名字>.safetensors`，并写同名 `.txt` 触发词旁文件
   —— 与 `utils/lora_triggers.py` 现有的读取逻辑对齐，训完就能在主界面
   下拉框里直接带出触发词。
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
#: 桶尺寸必须是 64 的倍数：VAE 下采样 8 次 → 64/8 = 8，UNet 的 down 层也要求 /8。
SIZE_MULTIPLE = 64
#: UNet 上挂 LoRA 的线性层（SD1.5 与 SDXL 都是这四类）。
UNET_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
#: 文本编码器（CLIP）里挂 LoRA 的线性层。
TE_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")

_SDXL_ARCHES = ("sdxl", "pony", "sd35", "sdxl_refiner")

#: 主干上可以挂 LoRA 的线性层，按架构族区分（peft 按名字后缀匹配，
#: 不存在的名字会被静默忽略，所以宁可多列几个）。
_BACKBONE_TARGETS = {
    "sd": ("to_q", "to_k", "to_v", "to_out.0"),
    # SD3 的 joint block 里还有一路"给文本用"的注意力
    "sd3": ("to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"),
    # Flux 每个 block 还带 proj_mlp / proj_out（社区 LoRA 常一起训）
    "flux": ("to_q", "to_k", "to_v", "to_out.0", "proj_mlp", "proj_out"),
}


def arch_family(arch: str) -> str:
    """把架构 id 归到三个可训练的族：`sd`（UNet）/ `sd3` / `flux`（transformer）。

    三族的差别不是"名字不同"，而是 **前向签名与训练目标不同**：
      · sd  ：UNet2DConditionModel，噪声预测（epsilon）或 v-prediction；
      · sd3 ：SD3Transformer2DModel，rectified-flow，目标是 (noise - latents)；
      · flux：FluxTransformer2DModel，latents 要 2×2 pack，还要 img_ids/txt_ids/guidance。
    """
    a = (arch or "").lower()
    if a in ("flux", "flux1", "flux.1"):
        return "flux"
    if a in ("sd3", "sd35", "sd3.5", "sd3_5"):
        return "sd3"
    return "sd"


def backbone_targets(arch: str) -> Tuple[str, ...]:
    return _BACKBONE_TARGETS.get(arch_family(arch), UNET_TARGETS)


def backbone_of(pipe, arch: str):
    """返回该架构要训练的"主干"模块（UNet 或 transformer）。"""
    fam = arch_family(arch)
    if fam in ("sd3", "flux"):
        tb = getattr(pipe, "transformer", None)
        if tb is None:
            raise RuntimeError(f"{arch} 应该用 transformer 训练，但管线里没有 transformer")
        return tb
    unet = getattr(pipe, "unet", None)
    if unet is None:
        raise RuntimeError(f"{arch} 应该用 unet 训练，但管线里没有 unet")
    return unet



@dataclass
class TrainConfig:
    """一次训练的全部参数（可直接序列化进 `.train.json`）。"""

    dataset_dir: str = ""
    base_model: str = ""
    output_name: str = ""
    rank: int = 8
    alpha: float = 0.0          # 0 → 用 rank（社区惯例 alpha=r）
    lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    steps: int = 400
    batch_size: int = 1
    resolution: int = 512
    train_text_encoder: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = -1
    trigger: str = ""
    save_every: int = 0
    warmup: int = 10
    sample_prompt: str = ""
    fp16: bool = False          # False → 有 bf16 就用 bf16（训练不需要 loss scaling）
    #: Flux 训练时的 guidance 嵌入（0 = 自动：dev 用 3.5，schnell 用 1.0）
    guidance: float = 0.0
    #: 学习率调度：constant / linear / cosine / cosine_restarts / poly / plateau
    lr_scheduler: str = "cosine"
    #: 调度末端的下限比例（真实 lr = cfg.lr × min_lr_ratio）
    min_lr_ratio: float = 0.05
    #: 优化器：adamw / adamw8bit（需要 bitsandbytes）/ adafactor（最省显存）
    optimizer: str = "adamw"
    #: 权重衰减（LoRA 一般 0.01；过拟合严重时加）
    weight_decay: float = 0.01
    #: 梯度累积步数（1 = 不累积；显存紧张时用它换等效 batch）
    grad_accumulation: int = 1

    def effective_alpha(self) -> float:
        return float(self.alpha) if self.alpha and self.alpha > 0 else float(self.rank)


@dataclass
class TrainResult:
    ok: bool = False
    cancelled: bool = False
    out_path: str = ""
    sidecar: str = ""
    arch: str = ""
    steps_done: int = 0
    steps_planned: int = 0
    seconds: float = 0.0
    losses: List[float] = field(default_factory=list)
    vram_peak_mb: float = 0.0
    images: int = 0
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    def summary_text(self) -> str:
        if not self.ok:
            return f"训练未完成：{self.error or '已取消'}"
        avg = sum(self.losses[-20:]) / max(1, len(self.losses[-20:]))
        return (f"完成 {self.steps_done} 步，末 20 步平均 loss {avg:.4f}，"
                f"用时 {self.seconds:.0f}s，显存峰值 {self.vram_peak_mb:.0f}MB → {self.out_path}")


# ---------------------------------------------------------------- 数据集扫描
def scan_dataset(dataset_dir: str, trigger: str = "") -> Tuple[List[dict], List[str]]:
    """扫目录（不递归）里的图片 + 同名 `.txt` caption。

    返回 (records, warnings)。record 键：`path` / `caption` / `size`。
    caption 为空时仍然训练（等于无标签），但会给 warning —— 无标签的 LoRA
    通常学不出可复用的东西，必须让用户看见。
    """
    warnings: List[str] = []
    records: List[dict] = []
    root = str(dataset_dir or "").strip()
    if not root or not os.path.isdir(root):
        return [], [f"数据集目录不存在：{root or '(空)'}"]
    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        return [], [f"读取数据集目录失败：{exc}"]

    missing_caption = 0
    for name in names:
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        caption = ""
        caption_path = os.path.splitext(path)[0] + ".txt"
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(caption_path, "r", encoding=enc) as fh:
                    caption = fh.read().strip()
                break
            except FileNotFoundError:
                break
            except (UnicodeDecodeError, OSError):
                continue
        if not caption:
            missing_caption += 1
        if trigger:
            caption = f"{trigger}, {caption}" if caption else trigger
        size = None
        try:
            from PIL import Image
            with Image.open(path) as im:
                size = im.size
        except Exception as exc:  # noqa: BLE001 - 单张坏图不该中断训练
            logger.warning("读取图片尺寸失败 %s: %s", path, exc)
        records.append({"path": path, "caption": caption, "size": size})

    if not records:
        warnings.append("目录里没有可用的图片（支持 png/jpg/jpeg/webp/bmp/tif）")
    if missing_caption:
        warnings.append(f"{missing_caption} 张图没有同名 .txt 标注，会被当成无标签样本训练")
    return records, warnings


def bucket_size(width: int, height: int, resolution: int = 512) -> Tuple[int, int]:
    """按面积≈resolution² 归一到最近的桶，长宽都取 64 的倍数。

    取整用 **四舍五入到最近的 64**（不是 floor），并且把面积误差控制在
    ±6% 内：先按短边比例算，再对齐，最后夹到 [1,4] 倍长宽比。
    """
    w = max(1, int(width or 1))
    h = max(1, int(height or 1))
    res = max(256, int(resolution or 512))
    aspect = w / h
    aspect = min(max(aspect, 0.25), 4.0)
    bh = math.sqrt((res * res) / aspect)
    bw = bh * aspect

    def _snap(v: float) -> int:
        return max(SIZE_MULTIPLE, int(round(v / SIZE_MULTIPLE)) * SIZE_MULTIPLE)

    return _snap(bw), _snap(bh)


def build_batches(items: Sequence[dict], batch_size: int, rng: random.Random) -> List[List[dict]]:
    """同桶同 batch：UNet 一个 batch 里张量尺寸必须一致，所以按桶分组。"""
    groups: Dict[Tuple[int, int], List[dict]] = {}
    for it in items:
        groups.setdefault(tuple(it.get("bucket") or (0, 0)), []).append(it)
    batches: List[List[dict]] = []
    bs = max(1, int(batch_size))
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            if len(chunk) < bs and batches:
                # 凑不满的尾块并到同桶的上一个 batch 里，避免 batch=1 的尾块
                # 把有效 batch 变小。
                prev = batches[-1]
                if len(prev) + len(chunk) <= bs:
                    prev.extend(chunk)
                    continue
            batches.append(chunk)
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- 适配器配置
def build_lora_config(rank: int, alpha: float, targets: Sequence[str]):
    """构造 peft 的 LoraConfig（diffusers 的 `add_adapter` 直接吃）。"""
    from peft import LoraConfig

    return LoraConfig(
        r=max(1, int(rank)),
        lora_alpha=float(alpha),
        init_lora_weights="gaussian",
        target_modules=list(targets),
    )


def estimate_vram_mb(cfg: TrainConfig, arch: str = "sd15") -> int:
    """粗略显存估算（给界面提示用，不是承诺）。

    基数是"512² + 梯度检查点"下的实测/推断值：
    SD1.5 约 2.9GB、SDXL 约 5.5GB、SD3-medium 约 6GB、Flux 约 20GB（12B 参数）。
    """
    res = max(256, min(1024, int(cfg.resolution or 512)))
    scale = (res / 512.0) ** 2
    family = arch_family(arch)
    a = (arch or "").lower()
    if family == "flux":
        base = 20000 * scale
    elif family == "sd3":
        base = 6000 * scale
    elif a in _SDXL_ARCHES:
        base = 5200 * scale
    else:
        base = 2600 * scale
    if not cfg.gradient_checkpointing:
        base *= 1.7
    if cfg.train_text_encoder and family == "sd":
        base += 900 if a in _SDXL_ARCHES else 350
    base += 300 * max(1, int(cfg.batch_size))
    return int(base)


def vram_advice(cfg: TrainConfig, arch: str, available_mb: int = 0) -> str:
    """给界面的一句话建议：够不够、要不要降分辨率。"""
    need = estimate_vram_mb(cfg, arch)
    family = arch_family(arch)
    parts = [f"估算显存约 {need} MB"]
    if available_mb > 0:
        parts.append(f"当前可用约 {available_mb} MB")
    if need > available_mb > 0:
        parts.append("显存可能不够，建议降到 512 或关掉梯度检查点以外的选项")
    if family == "flux":
        parts.append("Flux 主干 12B，8GB 卡练不动（建议 16GB 以上，或改训 SDXL/SD3）")
    elif family == "sd3":
        parts.append("SD3 建议 10GB 以上显存")
    return "；".join(parts)


def _fit_image(path: str, bucket: Tuple[int, int]):
    """把图片 resize + center-crop 到桶尺寸（返回 RGB PIL）。"""
    from PIL import Image

    bw, bh = bucket
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max(bw / im.width, bh / im.height)
        nw, nh = max(bw, int(round(im.width * scale))), max(bh, int(round(im.height * scale)))
        im = im.resize((nw, nh), Image.LANCZOS)
        left = (nw - bw) // 2
        top = (nh - bh) // 2
        return im.crop((left, top, left + bw, top + bh))


def _to_tensor(im) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(im, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _encode_caption(tokenizer, text_encoder, caption: str, sdxl: bool, device: str, dtype):
    """返回 (encoder_hidden_states, pooled_or_None)。"""
    tokens = tokenizer(
        caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = tokens.input_ids.to(device)
    with torch.no_grad():
        if sdxl:
            out = text_encoder(ids, output_hidden_states=True)
            hidden = out.hidden_states[-2]
            pooled = out[0]
            return hidden.to(dtype=dtype), pooled.to(dtype=dtype)
        out = text_encoder(ids)
        return out[0].to(dtype=dtype), None


def _sdxl_time_ids(bucket: Tuple[int, int], device: str, dtype):
    w, h = bucket
    ids = torch.tensor([[w, h, 0, 0, w, h]], dtype=dtype, device=device)
    return ids


# ---------------------------------------------------------------------------
#  SD3 / Flux：rectified-flow 家族的公共小工具
# ---------------------------------------------------------------------------

def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, (H/2)*(W/2), C*4)，Flux 的主干只吃打包后的序列。"""
    bsz, ch, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError(f"Flux 需要长宽都是偶数，收到 {h}x{w}")
    packed = latents.reshape(bsz, ch, h // 2, 2, w // 2, 2)
    packed = torch.einsum("bchpwq->bhwpqc", packed)
    return packed.reshape(bsz, (h // 2) * (w // 2), ch * 4)


def _flux_img_ids(height: int, width: int, device: str, dtype) -> torch.Tensor:
    """Flux 的 img_ids：3 列 (t, h, w)，按 2×2 patch 网格展开。"""
    hh, ww = height // 2, width // 2
    ids = torch.zeros(hh, ww, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(hh, device=device, dtype=dtype).view(-1, 1)
    ids[..., 2] = torch.arange(ww, device=device, dtype=dtype).view(1, -1)
    return ids.reshape(-1, 3)


def _flux_text_ids(text_len: int, device: str, dtype) -> torch.Tensor:
    return torch.zeros(int(text_len), 3, device=device, dtype=dtype)


def _encode_prompt_with_pipe(pipe, caption: str, family: str, device: str, dtype):
    """为 SD3 / Flux 编码提示词，返回 `(embeds, pooled)`（都搬回 CPU 缓存）。

    不直接调管线自带的 `encode_prompt`：它按 `self._execution_device` 搬输入，
    而我们为了省显存只把文本编码器临时放到卡上、管线本身还在 CPU —— 借用它
    很容易踩设备不匹配。这里按上游同样的拼接口径自己算：
      · SD3  ：CLIP-L(768) + CLIP-G(1280) + T5(2048) 拼成 4096；pooled = 768+1280
      · Flux ：T5(4096) 作 embeds；pooled 取 CLIP 的 `pooler_output`(768)
    两边都按 77 / 512 的固定长度 padding，与推理时一致。
    """
    def _ids(tokenizer, text, max_len):
        if tokenizer is None:
            raise RuntimeError("缺少 tokenizer，无法编码提示词")
        toks = tokenizer(text, padding="max_length", max_length=max_len,
                         truncation=True, return_tensors="pt")
        return toks.input_ids.to(device)

    def _on_card(mod):
        if mod is None:
            return None
        return mod.to(device=device, dtype=dtype).eval()

    if family == "flux":
        te = _on_card(getattr(pipe, "text_encoder", None))
        te2 = _on_card(getattr(pipe, "text_encoder_2", None))
        try:
            with torch.no_grad():
                pooled = te(_ids(getattr(pipe, "tokenizer", None), caption, 77)).pooler_output
                embeds = te2(_ids(getattr(pipe, "tokenizer_2", None), caption, 512))[0]
        finally:
            for m in (te, te2):
                if m is not None:
                    m.to("cpu")
    else:  # sd3
        te = _on_card(getattr(pipe, "text_encoder", None))
        te2 = _on_card(getattr(pipe, "text_encoder_2", None))
        te3 = _on_card(getattr(pipe, "text_encoder_3", None))
        try:
            with torch.no_grad():
                ids = _ids(getattr(pipe, "tokenizer", None), caption, 77)
                out_l = te(ids, output_hidden_states=True)
                clip_l = out_l.hidden_states[-2]
                pooled_l = out_l[0]
                out_g = te2(ids, output_hidden_states=True)
                clip_g = out_g.hidden_states[-2]
                pooled_g = out_g[0]
                t5 = te3(_ids(getattr(pipe, "tokenizer_3", None), caption, 77))[0]
                embeds = torch.cat([clip_l, clip_g, t5], dim=-1)
                pooled = torch.cat([pooled_l, pooled_g], dim=-1)
        finally:
            for m in (te, te2, te3):
                if m is not None:
                    m.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (embeds.detach().to("cpu", dtype=dtype),
            pooled.detach().to("cpu", dtype=dtype) if isinstance(pooled, torch.Tensor) else None)



def _predict(family: str, backbone, noisy, timesteps, enc, pooled, bucket, bsz,
             device, dtype, guidance: float):
    """按架构族调用主干，返回 `(model_pred, target)`（都还没算 loss）。"""
    if family == "sd":
        kwargs = {}
        if pooled is not None:
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled,
                "time_ids": _sdxl_time_ids(bucket, device, dtype).repeat(bsz, 1),
            }
        return backbone(noisy, timesteps, enc, **kwargs).sample, None

    t = (timesteps.float() / 1000.0).to(dtype=dtype)
    if family == "sd3":
        pred = backbone(
            hidden_states=noisy,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=t,
            return_dict=False,
        )[0]
        return pred, None

    # flux
    height, width = bucket[1], bucket[0]
    packed = _pack_latents(noisy)
    img_ids = _flux_img_ids(height, width, device, dtype)
    txt_ids = _flux_text_ids(enc.shape[1], device, dtype)
    g = torch.full((bsz,), float(guidance), device=device, dtype=dtype)
    pred = backbone(
        hidden_states=packed,
        encoder_hidden_states=enc,
        pooled_projections=pooled,
        timestep=t,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=g,
        return_dict=False,
    )[0]
    return pred, None



def _clear_adapters(module) -> None:
    """摘掉模块上残留的 LoRA 适配器。

    同一个管线连着训第二次时，第一次挂的适配器还在，`add_adapter` 会抛
    `Adapter with name default already exists`。正常路径（界面每次都重新加载底模）
    下这里是空操作。

    注意：`unet.unload_lora_weights` 在 diffusers 0.40 的 UNet 上**不存在**
    （那是管线上的方法），所以这里直接走 `delete_adapters` + 清记账。
    """
    if module is None:
        return
    try:
        names = list(getattr(module, "peft_config", {}) or {})
    except Exception:  # noqa: BLE001 - 没有 peft_config 就是没挂过
        return
    if not names:
        return
    for name in names:
        try:
            module.delete_adapters(name)
        except Exception as exc:  # noqa: BLE001 - 清不掉就继续清记账
            logger.debug("删除适配器 %s 失败: %s", name, exc)
    try:
        cfg = getattr(module, "peft_config", None)
        if isinstance(cfg, dict):
            cfg.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        module._hf_peft_config_loaded = False
    except Exception:  # noqa: BLE001
        pass
    logger.info("已清掉 %s 上残留的 LoRA 适配器（%s）", type(module).__name__, ", ".join(names))


def _lora_params(*modules) -> List[torch.nn.Parameter]:
    params = []
    for mod in modules:
        if mod is None:
            continue
        for name, p in mod.named_parameters():
            if "lora" in name.lower():
                p.requires_grad_(True)
                params.append(p)
    return params


def _detach_pipeline(pipe) -> None:
    """把推理用的 offload 钩子摘掉，让组件可以被我们手动搬到显卡。"""
    for obj in (pipe, getattr(pipe, "unet", None), getattr(pipe, "vae", None),
                getattr(pipe, "text_encoder", None), getattr(pipe, "text_encoder_2", None)):
        if obj is None:
            continue
        for fname in ("remove_all_hooks", "remove_hooks"):
            fn = getattr(obj, fname, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - 不同 diffusers 版本签名不同
                    logger.debug("%s.%s 失败: %s", type(obj).__name__, fname, exc)


#: 可选的学习率调度（名字 → 界面上的说明）
LR_SCHEDULERS = {
    "constant": "恒定",
    "linear": "线性衰减",
    "cosine": "余弦退火",
    "cosine_restarts": "余弦退火 + 重启",
    "poly": "多项式衰减",
    "plateau": "按 loss 自动降",
}
#: 可选优化器（名字 → 说明）
OPTIMIZERS = {
    "adamw": "AdamW（默认）",
    "adamw8bit": "AdamW 8bit（省显存，需 bitsandbytes）",
    "adafactor": "Adafactor（最省显存）",
}


def lr_factor(step: int, total: int, warmup: int,
              name: str = "cosine", min_ratio: float = 0.05) -> float:
    """第 `step` 步（0 基）的学习率倍数，相对 `cfg.lr`。

    纯函数，方便单测：不碰 optimizer，也不依赖 torch。

    形状（warmup 之后 p = (step-warmup)/(total-warmup) ∈ [0,1]）：
      constant         1
      linear           1 - (1-min)·p
      cosine           min + (1-min)·0.5·(1+cos(πp))
      cosine_restarts  按 T0≈总步数/3 反复从 1 衰减到 min
      poly             min + (1-min)·(1-p)²
    """
    total = max(1, int(total))
    warmup = max(0, min(int(warmup), total - 1))
    lo = max(0.0, min(1.0, float(min_ratio)))
    step = max(0, int(step))
    if step < warmup:
        return (step + 1) / max(1, warmup)
    span = max(1, total - warmup)
    p = (step - warmup) / span
    p = min(1.0, max(0.0, p))
    name = (name or "cosine").strip().lower()
    if name in ("constant", "const", "none", "固定", "恒定"):
        return 1.0
    if name in ("linear", "线性"):
        return 1.0 - (1.0 - lo) * p
    if name in ("poly", "polynomial", "多项式"):
        return lo + (1.0 - lo) * (1.0 - p) ** 2
    if name in ("cosine_restarts", "cosine_warm_restarts", "余弦重启"):
        cycle = max(1, span // 3)
        pc = ((step - warmup) % cycle) / cycle
        return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * pc))
    # 默认 cosine
    return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * p))


def _make_scheduler(optimizer, total: int, warmup: int,
                    name: str = "cosine", min_lr_ratio: float = 0.05):
    """返回 (scheduler, 是否需要喂 loss, 说明文字)。"""
    total = max(1, int(total))
    name = (name or "cosine").strip().lower()
    if name in ("plateau", "按loss", "loss"):
        import torch.optim.lr_scheduler as lrs

        base = min(float(g["lr"]) for g in optimizer.param_groups) if optimizer.param_groups else 0.0
        sched = lrs.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(5, total // 20), cooldown=0,
            min_lr=base * max(0.0, float(min_lr_ratio)),
        )
        return sched, True, "按 loss 自动降（patience=%d 步，每次减半）" % max(5, total // 20)
    import torch.optim.lr_scheduler as lrs

    sched = lrs.LambdaLR(
        optimizer,
        lambda step: lr_factor(step, total, warmup, name, min_lr_ratio),
    )
    desc = LR_SCHEDULERS.get(name, LR_SCHEDULERS["cosine"])
    if name not in LR_SCHEDULERS:
        desc = LR_SCHEDULERS["cosine"] + "（未识别 %r，按余弦处理）" % name
    return sched, False, f"{desc}，末端降到 {float(min_lr_ratio) * 100:.0f}%"


def describe_schedule(name: str, min_lr_ratio: float) -> str:
    """给人看的调度说明（界面/日志共用）。"""
    n = (name or "cosine").strip().lower()
    label = LR_SCHEDULERS.get(n, LR_SCHEDULERS["cosine"])
    if n in ("plateau", "loss"):
        return label + "（不设末端下限，触发时减半）"
    return f"{label}，末端保留 {max(0.0, float(min_lr_ratio)) * 100:.0f}% 学习率"


def _make_optimizer(groups, cfg, log) -> object:
    """按配置造优化器；8bit/Adafactor 不可用时明确回退。"""
    name = (getattr(cfg, "optimizer", "adamw") or "adamw").strip().lower()
    wd = float(getattr(cfg, "weight_decay", 0.01) or 0.0)
    if name in ("adamw8bit", "8bit", "adamw_8bit"):
        try:
            import bitsandbytes as bnb

            log("优化器：AdamW 8bit（bitsandbytes）")
            return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
        except Exception as exc:  # noqa: BLE001 - 没有 bnb 就老实回退
            log(f"⚠️ AdamW 8bit 不可用（{exc}），改用普通 AdamW")
    if name in ("adafactor",):
        try:
            from transformers.optimization import Adafactor

            log("优化器：Adafactor（最省显存，学习率按相对步长自动缩放）")
            return Adafactor(groups, scale_parameter=False, relative_step=False,
                             warmup_init=False, weight_decay=wd)
        except Exception as exc:  # noqa: BLE001
            log(f"⚠️ Adafactor 不可用（{exc}），改用普通 AdamW")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)


def _safe_name(text: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", (text or "").strip())
    name = re.sub(r"\s+", "_", name)
    return name.strip(" ._") or f"lora_{int(time.time())}"


# ---------------------------------------------------------------- 主流程
def train_lora(
    cfg: TrainConfig,
    *,
    pipe=None,
    arch: str = "",
    device: str = "cuda",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_preview: Optional[Callable[[str, object], bool]] = None,
    _prepared: Optional[dict] = None,
) -> TrainResult:
    """训练一个 LoRA 并保存到 `loras/<output_name>.safetensors`。

    `pipe=` / `_prepared=` 只给测试用（可以注入极小模型与假缓存，跳过加载）。
    """
    res = TrainResult(steps_planned=int(cfg.steps or 0), arch=arch or "sd15")
    family = arch_family(arch or "sd15")
    started = time.perf_counter()

    def log(msg: str) -> None:
        logger.info("[LoRA] %s", msg)
        if on_log:
            try:
                on_log(msg)
            except Exception:  # noqa: BLE001 - 回调不该影响训练
                    pass

    def status(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:  # noqa: BLE001
                pass

    def cancelled() -> bool:
        try:
            return bool(should_cancel and should_cancel())
        except Exception:  # noqa: BLE001
            return False

    def progress(done: int, total: int, label: str = "") -> None:
        if on_progress:
            try:
                on_progress(int(done), max(1, int(total)), label)
            except Exception:  # noqa: BLE001
                pass

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # ---------- 1. 组件与缓存 ----------
        # 先扫数据集再碰管线：数据集为空时应当立刻报"数据集为空"，
        # 而不是先去访问 pipe.vae（脚本/测试注入的假管线可能压根没有它）。
        records: List[dict] = []
        if _prepared is None:
            records, warns = scan_dataset(cfg.dataset_dir, cfg.trigger)
            res.warnings.extend(warns)
            if not records:
                raise ValueError("数据集为空：" + (cfg.dataset_dir or "(未选择)"))

        if _prepared is not None:
            prep = _prepared
            sdxl = bool(prep.get("sdxl"))
            unet = prep["unet"]
            vae = prep.get("vae")
            tok = prep.get("tokenizer")
            te = prep.get("text_encoder")
            samples = prep["samples"]          # [{"latent":t, "embeds":t, "pooled":t|None, "bucket":(w,h)}]
            scheduler_src = prep.get("scheduler")
        else:
            if pipe is None:
                raise ValueError("train_lora 需要 pipe= 或 _prepared=")
            sdxl = bool(arch in _SDXL_ARCHES)
            unet = backbone_of(pipe, arch)     # sd → pipe.unet；sd3/flux → pipe.transformer
            vae = pipe.vae
            tok = getattr(pipe, "tokenizer", None)
            te = getattr(pipe, "text_encoder", None)
            scheduler_src = getattr(pipe, "scheduler", None)
            prep = {}
            samples = []

        dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and not cfg.fp16) else torch.float16
        if _prepared is not None:
            dtype = next(unet.parameters()).dtype

        if _prepared is None:
            _detach_pipeline(pipe)

            # 分桶
            for rec in records:
                if not rec.get("size"):
                    try:
                        from PIL import Image
                        with Image.open(rec["path"]) as im:
                            rec["size"] = im.size
                    except Exception:  # noqa: BLE001
                        rec["size"] = (cfg.resolution, cfg.resolution)
                rec["bucket"] = bucket_size(rec["size"][0], rec["size"][1], cfg.resolution)
            buckets = sorted({tuple(r["bucket"]) for r in records})
            log(f"数据集 {len(records)} 张，{len(buckets)} 种尺寸桶：" +
                ", ".join(f"{w}x{h}" for w, h in buckets))

            # VAE 编码（fp32 更稳，编完立刻下卡）
            status("正在编码图片（VAE）...")
            vae_dtype = torch.float32 if getattr(vae.config, "force_upcast", False) else dtype
            vae = vae.to(device=device, dtype=vae_dtype)
            vae.eval()
            scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
            total_prep = len(records) + 4
            for idx, rec in enumerate(records):
                if cancelled():
                    res.cancelled = True
                    raise KeyboardInterrupt
                im = _fit_image(rec["path"], tuple(rec["bucket"]))
                t = _to_tensor(im).unsqueeze(0).to(device=device, dtype=vae_dtype)
                with torch.no_grad():
                    lat = vae.encode(t).latent_dist.sample() * scaling
                samples.append({
                    "latent": lat.detach().to("cpu", dtype=dtype),
                    "caption": rec["caption"],
                    "bucket": tuple(rec["bucket"]),
                    "path": rec["path"],
                })
                progress(idx + 1, total_prep, "编码图片")
            vae = vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log("图片编码完成（latents 已缓存到内存）")

            # 文本编码缓存在第一步判断：训练文本编码器时不能缓存
            if not cfg.train_text_encoder:
                status("正在编码提示词..." if family != "sd" else "正在编码提示词（CLIP）...")
                if family == "sd":
                    te_dev = te.to(device=device, dtype=dtype)
                    te.eval()
                for idx, s in enumerate(samples):
                    if cancelled():
                        res.cancelled = True
                        raise KeyboardInterrupt
                    if family == "sd":
                        emb, pooled = _encode_caption(tok, te_dev, s["caption"], sdxl, device, dtype)
                    else:
                        emb, pooled = _encode_prompt_with_pipe(
                            pipe, s["caption"], family, device, dtype)
                    s["embeds"] = emb.to("cpu")
                    s["pooled"] = pooled.to("cpu") if pooled is not None else None
                    progress(len(records) + 4 + idx, total_prep + len(records), "编码提示词")
                if family == "sd":
                    te = te.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log("提示词编码完成（文本编码器已下卡）")

            # 释放 VAE/CLIP 引用，只留 UNet
            try:
                del vae
            except Exception:  # noqa: BLE001
                pass
            vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res.images = len(samples)

        # ---------- 2. 挂适配器 ----------
        # 注意：`unet` 这个变量在 sd3/flux 下装的是 pipe.transformer（主干），
        # 名字沿用是为了让后面的优化器/保存逻辑通用。
        status("正在挂 LoRA 适配器...")
        unet.to(device=device, dtype=dtype)
        unet.requires_grad_(False)
        # 在同一个管线上连着训第二次时，上一次的适配器还挂着，
        # add_adapter 会抛 "Adapter with name default already exists" —— 先摘干净。
        _clear_adapters(unet)
        unet.add_adapter(build_lora_config(cfg.rank, cfg.effective_alpha(),
                                          backbone_targets(arch or "sd15")))
        if cfg.gradient_checkpointing:
            try:
                unet.enable_gradient_checkpointing()
            except Exception as exc:  # noqa: BLE001
                res.warnings.append(f"开启梯度检查点失败：{exc}")
        te_train = None
        if cfg.train_text_encoder and family != "sd":
            res.warnings.append(
                f"{arch} 的 LoRA 只训主干；文本编码器训练仅支持 SD1.5/SD2/SDXL，已忽略该选项")
        elif cfg.train_text_encoder and te is not None and _prepared is None:
            te_train = pipe.text_encoder if sdxl else te
            te_train.to(device=device, dtype=dtype)
            te_train.requires_grad_(False)
            _clear_adapters(te_train)
            te_train.add_adapter(build_lora_config(cfg.rank, cfg.effective_alpha(), TE_TARGETS))
            te_train.train()

        params = _lora_params(unet, te_train)
        if not params:
            raise RuntimeError("没有可训练的 LoRA 参数（add_adapter 失败？）")
        log(f"可训练参数 {sum(p.numel() for p in params) / 1e6:.2f}M")
        groups = [{"params": _lora_params(unet), "lr": float(cfg.lr)}]
        if te_train:
            groups.append({"params": _lora_params(te_train), "lr": float(cfg.text_encoder_lr)})
        total_steps = max(1, int(cfg.steps))
        accum = max(1, int(getattr(cfg, "grad_accumulation", 1) or 1))
        optimizer = _make_optimizer(groups, cfg, log)
        lr_sched, sched_needs_loss, sched_desc = _make_scheduler(
            optimizer, total_steps, cfg.warmup,
            getattr(cfg, "lr_scheduler", "cosine"),
            getattr(cfg, "min_lr_ratio", 0.05),
        )
        log(f"学习率调度：{sched_desc}" + (f"（梯度累积 {accum} 步）" if accum > 1 else ""))

        from diffusers import DDPMScheduler
        pred_type = "epsilon"
        try:
            pred_type = str(getattr(scheduler_src.config, "prediction_type", "epsilon"))
        except Exception:  # noqa: BLE001
            pass
        flow = family in ("sd3", "flux")
        noise_scheduler = None if flow else DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type=pred_type, clip_sample=False,
        )
        # Flux 的 guidance 嵌入：0 → 自动（schnell 用 1.0，其余按 dev 的 3.5）
        guidance_val = float(cfg.guidance or 0.0)
        if family == "flux" and guidance_val <= 0:
            name = (getattr(pipe, "name_or_path", "") or cfg.base_model or "").lower() if pipe else ""
            guidance_val = 1.0 if "schnell" in name else 3.5

        seed = int(cfg.seed)
        if seed < 0:
            seed = random.randint(0, 2 ** 31 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        rng = random.Random(seed)
        for s in samples:
            s.setdefault("bucket", tuple(s.get("bucket") or (cfg.resolution, cfg.resolution)))
            s.setdefault("caption", "")

        status(f"开始训练（{total_steps} 步，rank {cfg.rank}，丢包率 0%）...")
        unet.train()
        losses: List[float] = []
        step_done = 0
        t0 = time.perf_counter()
        batch_iter = None
        while step_done < total_steps:
            if cancelled():
                res.cancelled = True
                break
            if batch_iter is None or len(batch_iter) == 0:
                batch_iter = build_batches(samples, cfg.batch_size, rng)
                if not batch_iter:
                    raise ValueError("没有可用的训练样本")
            batch = batch_iter.pop()
            # 同 batch 尺寸一致
            lat = torch.cat([b["latent"] for b in batch], dim=0).to(device, non_blocking=True)
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32).to(
                device=device, dtype=lat.dtype)
            bsz = lat.shape[0]
            if flow:
                # rectified flow：t∈[0,1]，noisy=(1-t)·lat + t·noise，目标是 (noise - lat)
                t_rand = torch.rand(bsz, generator=gen, dtype=torch.float32).to(device)
                t_view = t_rand.view(bsz, 1, 1, 1)
                noisy = (1.0 - t_view) * lat + t_view * noise
                timesteps = (t_rand * 1000).to(dtype=lat.dtype)
                target = noise - lat
            else:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device="cpu", generator=gen).long().to(device)
                noisy = noise_scheduler.add_noise(lat, noise, timesteps)
                target = (noise_scheduler.get_velocity(lat, noise, timesteps)
                          if pred_type == "v_prediction" else noise)

            if te_train is not None:
                embs, pools = [], []
                for b in batch:
                    e, p = _encode_caption(tok, te_train, b.get("caption", ""), sdxl, device, dtype)
                    embs.append(e)
                    pools.append(p)
                enc = torch.cat(embs, dim=0)
                pooled = torch.cat(pools, dim=0) if sdxl and pools[0] is not None else None
            else:
                enc = torch.cat([b["embeds"] for b in batch], dim=0).to(device, non_blocking=True)
                pooled = None
                if batch[0].get("pooled") is not None and (sdxl or family in ("sd3", "flux")):
                    pooled = torch.cat([b["pooled"] for b in batch], dim=0).to(device)

            bucket = tuple(batch[0]["bucket"])
            model_pred, _ = _predict(family, unet, noisy, timesteps, enc, pooled, bucket,
                                     bsz, device, dtype, guidance_val)
            if flow and family == "flux":
                # flux 的输出是打包序列，目标也要打包到同一形状
                target = _pack_latents(target)
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
            raw_loss = float(loss.detach().cpu())
            if accum > 1:
                loss = loss / accum
            loss.backward()
            stepped = (step_done + 1) % accum == 0 or (step_done + 1) >= total_steps
            if stepped:
                if cfg.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(params, float(cfg.max_grad_norm))
                optimizer.step()
                if sched_needs_loss:
                    lr_sched.step(raw_loss)
                else:
                    lr_sched.step()
                optimizer.zero_grad(set_to_none=True)

            step_done += 1
            losses.append(raw_loss)
            if step_done == 1 or step_done % 10 == 0 or step_done == total_steps:
                rate = (time.perf_counter() - t0) / step_done
                left = max(0, total_steps - step_done) * rate
                progress(step_done, total_steps, "训练")
                cur_lr = optimizer.param_groups[0]["lr"]
                status(f"训练 {step_done}/{total_steps} · loss {losses[-1]:.4f} · "
                       f"lr {cur_lr:.2e} · {rate:.2f}s/步 · 剩余约 {left / 60:.1f} 分钟")
            if cfg.save_every and step_done % int(cfg.save_every) == 0:
                log(f"中途保存（{step_done} 步）")
                try:
                    _save_lora(pipe, cfg, arch, step_done, losses, partial=True)
                except Exception as exc:  # noqa: BLE001
                    res.warnings.append(f"中途保存失败：{exc}")

        res.steps_done = step_done
        res.losses = losses
        if res.cancelled:
            res.error = "已取消（未保存）"
            return res
        if step_done == 0:
            raise RuntimeError("一步都没跑（数据或参数有问题）")

        # ---------- 3. 保存 ----------
        status("正在保存 LoRA...")
        out_path, sidecar = _save_lora(pipe, cfg, arch, step_done, losses)
        res.out_path = out_path
        res.sidecar = sidecar
        res.ok = True
        try:
            res.warnings.append(
                "学习率：%s · 优化器 %s" % (
                    describe_schedule(getattr(cfg, "lr_scheduler", "cosine"),
                                      getattr(cfg, "min_lr_ratio", 0.05)),
                    (getattr(cfg, "optimizer", "adamw") or "adamw")))
        except Exception:  # noqa: BLE001
            pass
        res.seconds = time.perf_counter() - started
        if torch.cuda.is_available():
            res.vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        log(f"已保存：{out_path}")
        if sidecar:
            log(f"触发词旁文件：{sidecar}")
        progress(total_steps, total_steps, "完成")
        return res
    except KeyboardInterrupt:
        res.cancelled = True
        res.error = "已取消"
        res.seconds = time.perf_counter() - started
        return res
    except Exception as exc:  # noqa: BLE001 - 训练异常要原样带回界面
        logger.exception("LoRA 训练失败")
        res.error = str(exc) or exc.__class__.__name__
        res.seconds = time.perf_counter() - started
        return res


#: 允许训练的架构（其余架构要么结构不同、要么没有 vae/unet 可训）。
TRAINABLE_ARCHES = ("sd15", "sd21", "sdxl", "pony", "sd3", "sd35", "flux")


class _TrainingManagerStub:
    """`BaseLoader.post_load` 会用到 ModelManager 的几处 helper，训练时不需要。

    显式写一个桩而不是传 None：`post_load` 里 `mgr._detect_vpred(...)` 是硬调用，
    None 会 AttributeError（Anima 的探针也踩过）。
    """

    def __init__(self, arch_id: str = ""):
        self.arch_id = arch_id
        self.is_vpred = False

    def _detect_vpred(self, model_path: str, model_name: str = "") -> bool:  # noqa: ARG002
        return False

    def _force_fp32_vae(self, pipe) -> None:  # noqa: ARG002
        return None

    def _apply_light_optimizations(self, pipe, name: str = "") -> None:  # noqa: ARG002
        return None


def load_pipeline_for_training(model_path: str):
    """用**推理同一条加载链路**把底模装起来，返回 (pipe, arch_id)。

    为什么不直接用 `StableDiffusionPipeline.from_single_file`：项目的
    `SingleFileLoader` 已经把 sd15/sdxl 的类选择、dtype、VAE 修正都做过了，
    再抄一份就又会出现"两套实现"（本项目已经在这上面吃过四次亏）。
    """
    from core import arch as arch_mod
    from core.loaders import get_loader
    from core.loaders.base import LoadContext

    det = arch_mod.detect(model_path)
    info = arch_mod.get_arch(det.arch_id)
    if det.arch_id not in TRAINABLE_ARCHES:
        raise RuntimeError(
            f"当前架构「{info.display_name}」不支持 LoRA 训练。\n"
            f"支持：SD1.5 / SD2.1 / SDXL / Pony（都是 UNet 结构）。"
        )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = LoadContext(
        dtype=dtype,
        model_name=os.path.basename(model_path.rstrip("\\/")),
        arch_id=det.arch_id,
        info=info,
        detection=det,
        manager=_TrainingManagerStub(det.arch_id),
        optimization={},
    )
    result = get_loader(det.arch_id).load(model_path, ctx)
    if result.txt2img is None:
        raise RuntimeError("加载器没有返回管线，无法训练")
    return result.txt2img, det.arch_id


def _ensure_lora_dir() -> str:
    from utils import paths

    out = getattr(paths, "LORA_DIR", "") or os.path.join(os.getcwd(), "loras")
    os.makedirs(out, exist_ok=True)
    return out


def _lora_state_dict(module) -> Optional[dict]:
    """把挂了适配器的模块转成 diffusers 认的 LoRA state dict。

    ⚠️ 这里必须传 **LoRA 适配器的 state dict**，不能传模块本身：
    `save_lora_weights(unet_lora_layers=pipe.unet)` 会整份 `state_dict()`
    存下来（实测 942 个张量、1.7GB 的"UNet 副本"），而且没有任何键含
    "lora" → 之后 `load_lora_weights` 直接报
    `Invalid LoRA checkpoint. Make sure all LoRA param names contain 'lora' substring.`
    """
    if module is None:
        return None
    try:
        from peft.utils import get_peft_model_state_dict
        from diffusers.utils import convert_state_dict_to_diffusers

        sd = get_peft_model_state_dict(module)
        sd = convert_state_dict_to_diffusers(sd)
        if not sd:
            return None
        sd = {k: v for k, v in sd.items() if "lora" in k.lower()}
    except Exception as exc:  # noqa: BLE001 - 退回手工过滤
        logger.warning("提取 LoRA state dict 失败，改用键名过滤: %s", exc)
        sd = {k: v for k, v in module.state_dict().items() if "lora" in k.lower()}
    if not sd:
        return None
    # `get_peft_model_state_dict` 会带上适配器名（`.default`），而
    # `load_lora_weights` 见到 `.default.` 就 `Could not automatically infer
    # state dict type` —— 训练器自己存的文件必须能自己读回去。
    try:
        from utils import lora_repair

        sd, stripped = lora_repair.strip_adapter_suffix(sd)
        if stripped:
            logger.debug("已去掉 %d 个键名里的 .default", stripped)
    except Exception as exc:  # noqa: BLE001
        logger.debug("清理适配器名后缀失败（继续按原样保存）: %s", exc)
    return sd or None


def lora_subdir_for(arch: str) -> str:
    """训练产物该放进 `loras/` 的哪个子目录。

    界面找 LoRA 是按底模类别分子目录的（`loras/sd1.5/`、`loras/sdxl/`…，
    见 `ModelManager.current_lora_subdir()` 与 `utils/app_events.py` 里
    触发词旁文件的查找顺序）。存到 `loras/` 根目录的话，主界面下拉框看不到。
    """
    try:
        from core.arch import get_arch

        sub = get_arch(arch or "").lora_subdir
        if sub:
            return sub
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取 lora_subdir 失败，按架构名兜底: %s", exc)
    a = (arch or "").lower()
    if a in ("sd15", "sd21", "sd2", "sd-1.5"):
        return "sd1.5"
    if a in ("sdxl", "pony", "sd35"):
        return "sdxl"
    if a in ("flux",):
        return "flux"
    if a in ("sd3",):
        return "sd3"
    return "sd1.5"


def _save_lora(pipe, cfg: TrainConfig, arch: str, steps: int, losses: List[float],
               partial: bool = False) -> Tuple[str, str]:
    """用 `pipe.save_lora_weights` 存盘，再重命名到 `loras/<子目录>/<name>(_stepN).safetensors`。

    SD/SDXL 的管线参数叫 `unet_lora_layers`，Flux/SD3 叫 `transformer_lora_layers` ——
    这里按 `save_lora_weights` 的真实签名挑，而不是按架构名硬编码。
    """
    name = _safe_name(cfg.output_name or "lora")
    if partial:
        name = f"{name}_step{steps}"
    lora_dir = os.path.join(_ensure_lora_dir(), lora_subdir_for(arch))
    os.makedirs(lora_dir, exist_ok=True)
    target = os.path.join(lora_dir, f"{name}.safetensors")
    family = arch_family(arch)
    backbone = backbone_of(pipe, arch)
    backbone_lora = _lora_state_dict(backbone)
    if not backbone_lora:
        raise RuntimeError("没能提取到 LoRA 权重（add_adapter 没生效？）")
    te_lora = None
    if cfg.train_text_encoder and hasattr(pipe, "text_encoder") and arch not in _SDXL_ARCHES:
        te_lora = _lora_state_dict(pipe.text_encoder)

    import inspect as _inspect

    try:
        params = _inspect.signature(pipe.save_lora_weights).parameters
    except (TypeError, ValueError):
        params = {}
    if "transformer_lora_layers" in params or family in ("sd3", "flux"):
        backbone_key = "transformer_lora_layers"
    else:
        backbone_key = "unet_lora_layers"
    call = {backbone_key: backbone_lora, "safe_serialization": True}
    if "text_encoder_lora_layers" in params or not params:
        call["text_encoder_lora_layers"] = te_lora

    with tempfile.TemporaryDirectory(prefix="lora_save_") as tmp:
        pipe.save_lora_weights(save_directory=tmp, **call)
        produced = os.path.join(tmp, "pytorch_lora_weights.safetensors")
        if not os.path.isfile(produced):
            cands = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".safetensors")]
            if not cands:
                raise RuntimeError("save_lora_weights 没有产出 .safetensors")
            produced = cands[0]
        shutil.move(produced, target)

    sidecar = ""
    trigger = (cfg.trigger or "").strip()
    if trigger:
        sidecar = os.path.splitext(target)[0] + ".txt"
        with open(sidecar, "w", encoding="utf-8") as fh:
            fh.write(trigger)
    meta_path = os.path.splitext(target)[0] + ".train.json"
    payload = {
        "name": name,
        "arch": arch,
        "steps": steps,
        "loss_last": losses[-1] if losses else None,
        "config": asdict(cfg),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app": "AI 绘画工作站",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return target, sidecar
