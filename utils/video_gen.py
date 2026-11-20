# utils/video_gen.py
"""
🎬 AnimateDiff 视频生成器 v2
四模式: txt2video / img2video / vid2vid / prompt_travel
"""
import os
import copy
import torch
import numpy as np
from datetime import datetime
from PIL import Image
import logging
logging.getLogger("diffusers.loaders.unet").setLevel(logging.ERROR)

from utils.paths import MODEL_DIR

from diffusers import (
    AnimateDiffPipeline,
    AnimateDiffVideoToVideoPipeline,
    MotionAdapter,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    DDIMScheduler,
)

logger = logging.getLogger(__name__)

MOTION_ADAPTER_DIR = os.path.join(MODEL_DIR, "motion_adapter")
MOTION_LORA_DIR    = os.path.join(MODEL_DIR, "motion_lora")

#: AnimateDiff v1-5-3 的 motion module 的**位置编码只按 16 帧训练**。
#: 真机实测（2026-11-17，512²，v1-5-3）：8 帧正常、16 帧基本正常、**20 帧直接崩成纯色块**。
#: 所以 16 帧是"单次生成"的硬上限，要更长必须分段（见 _run_segmented）。
SINGLE_PASS_MAX_FRAMES = 16
#: 兼容旧名：以前把它当"建议上限"，实际是硬上限。
RECOMMENDED_MAX_FRAMES = SINGLE_PASS_MAX_FRAMES
#: 不勾「长视频模式」时允许的最大帧数（超过就夹住并提示）。
#: 17~32 帧会自动分段、照做不误；再往上耗时按帧数线性涨，必须显式勾选。
DEFAULT_MAX_FRAMES = 32
#: AnimateDiff-Lightning 蒸馏 LoRA 的存放目录（用户自行下载后放进来）
LIGHTNING_DIR = os.path.join(MODEL_DIR, "lightning")
#: 分段生成时每段的默认帧数（正好等于 motion module 的训练长度，单次最多 16）
DEFAULT_SEGMENT_FRAMES = 16
#: 续接时喂给模型的历史帧数：1 = 只给一张静止尾帧（实测段内几乎不动），
#: 4 = 把最近 4 帧当真实时间前缀喂进去，接缝更顺（实测 5.55 vs 5.93）
CONTINUATION_FRAMES = 4
#: 续接强度：越大越敢动，越小越贴上一段。真机扫描（384²/12 步/32 帧）：
#:   0.35 → 段内帧差 1.4（静止）、接缝 5.8
#:   0.55 → 段内 3.0（几乎静止）、接缝 7.0
#:   0.75 → 段内 7.6（动作明显）、接缝 14.0（像镜头切了一下）
#: 取 0.70：动作明显，接缝仍落在一次正常帧间变化的量级。
CONTINUATION_STRENGTH = 0.70
#: SD1.5 的 tokenizer 上限。超过就一定会被截断，不允许静默发生。
CLIP_MAX_TOKENS = 77

DEFAULT_NEG = (
    "bad hands, bad fingers, extra fingers, missing fingers, deformed hands, "
    "orange tint, warm color cast, oversaturated, "
    "lowres, worst quality, low quality, jpeg artifacts, blurry, watermark"
)


def _make_step_guard(inner=None):
    """把 NaN/Inf 从 latents 里抹掉，并链上界面进度回调。

    真机实测（2026-11-17，512²/v1-5-3）：fp16 下时间注意力的偶发溢出会让某些帧的
    latent 变成 NaN，解码出来就是**纯黑帧**（10 帧里能黑 4 帧，且每次位置不同）。
    在每步结束时清一次 NaN 比事后补救干净得多。
    """
    def _cb(pipe, step_index, timestep, callback_kwargs):
        try:
            lat = callback_kwargs.get("latents")
            if isinstance(lat, torch.Tensor) and not torch.isfinite(lat).all():
                callback_kwargs["latents"] = torch.nan_to_num(
                    lat, nan=0.0, posinf=0.0, neginf=0.0)
                logger.warning(
                    f"⚠️ 第 {step_index} 步 latents 出现 NaN/Inf，已就地清零"
                    f"（不清会出现纯黑帧）")
        except Exception as e:      # 守卫本身绝不能把生成搞挂
            logger.debug(f"[nan_guard] 检查失败: {e}")
        if inner is not None:
            try:
                out = inner(pipe, step_index, timestep, callback_kwargs)
                if isinstance(out, dict):
                    callback_kwargs = out
            except Exception as e:
                logger.debug(f"[nan_guard] 进度回调失败: {e}")
        return callback_kwargs
    return _cb


def _clean_name(raw: str) -> str:
    """'zoom-in (放大)' → 'zoom-in'"""
    if not raw:
        return ""
    return raw.split("(")[0].strip()


class VideoGenerator:
    def __init__(self, ai_manager):
        self.ai = ai_manager
        self.pipe = None
        self.v2v_pipe = None
        self.adapter = None
        self.current_adapter_name = None
        self.loaded_motion_loras = []
        self._ipa_loaded = False

    # ---------- MotionAdapter ----------
    def _pick_default_adapter(self):
        if not os.path.isdir(MOTION_ADAPTER_DIR):
            raise FileNotFoundError(
                f"❌ 找不到 {MOTION_ADAPTER_DIR}\n"
                f"请先执行:\n"
                f"  huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-3 "
                f"--local-dir models/motion_adapter/v1-5-3"
            )
        dirs = [d for d in os.listdir(MOTION_ADAPTER_DIR)
                if os.path.isdir(os.path.join(MOTION_ADAPTER_DIR, d))]
        if not dirs:
            raise FileNotFoundError(f"❌ {MOTION_ADAPTER_DIR} 下没有子目录")
        for pref in ("v3", "v1-5-3", "v1-5-2"):
            for d in dirs:
                if pref in d:
                    return d
        return dirs[0]

    def _load_adapter(self, name=None):
        if name is None:
            name = self._pick_default_adapter()
        if self.adapter is not None and self.current_adapter_name == name:
            return
        path = os.path.join(MOTION_ADAPTER_DIR, name)
        logger.info(f"🎬 加载 MotionAdapter: {name}")
        self.adapter = MotionAdapter.from_pretrained(path, torch_dtype=torch.float16)
        self.current_adapter_name = name

    # ---------- Pipeline ----------
    def _build_pipe(self, scheduler="dpm++", need_v2v=False):
        base = self.ai.txt2img_pipe
        if base is None:
            raise RuntimeError("❌ 请先加载 SD1.5 底模")
        if hasattr(base, "text_encoder_2"):
            raise RuntimeError("⚠️ AnimateDiff 只支持 SD1.5")

        self._load_adapter()
        # 视频管线必须用**不带 LoRA** 的 UNet：motion module 会把 UNet 再包一层，
        # 挂着 peft hook 会互相干扰。但底模是用户正在用的那份，卸完必须挂回去，
        # 否则"生成过一次视频后，图片生成就不带 LoRA 了"。
        _saved_loras = []
        if hasattr(self.ai, "active_loras"):
            try:
                _saved_loras = self.ai.active_loras()
            except Exception as e:
                logger.debug(f"[video] 读取当前 LoRA 失败: {e}")
        try:
            base.unload_lora_weights()
        except Exception:
            pass

        if self.pipe is None:
            unet = copy.deepcopy(base.unet)
            self.pipe = AnimateDiffPipeline(
                vae=base.vae,
                text_encoder=base.text_encoder,
                tokenizer=base.tokenizer,
                unet=unet,
                motion_adapter=self.adapter,
                scheduler=base.scheduler,
                feature_extractor=None,
                image_encoder=None,
            )
            self._offload(self.pipe)
            logger.info("✅ AnimateDiff txt2v 就绪")

        if need_v2v and self.v2v_pipe is None:
            self.v2v_pipe = AnimateDiffVideoToVideoPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                motion_adapter=self.adapter,
                scheduler=self.pipe.scheduler,
                feature_extractor=None,
                image_encoder=None,
            )
            # 与 self.pipe 共享组件，不能再次安装 offload hook
            self._offload(self.v2v_pipe, already_offloaded=True)
            logger.info("✅ AnimateDiff v2v 就绪")

        self._apply_scheduler(scheduler)

        # 建完（含 deepcopy 的 UNet）再挂回去：deepcopy 发生在 LoRA 卸载之后，
        # 所以视频管线拿到的是干净 UNet，图片管线恢复原状。
        if _saved_loras:
            try:
                self.ai.apply_multiple_loras(_saved_loras)
                logger.info(f"🔁 视频管线已就绪，图片管线的 LoRA 已恢复: {[n for n, _ in _saved_loras]}")
            except Exception as e:
                logger.warning(f"⚠️ 恢复 LoRA 失败（图片生成可能暂时不带 LoRA）: {e}")

    def dispose(self):
        """释放 AnimateDiff 管线与 hook。

        切换底模时 ui 会把 self.video_generator 置 None —— 只丢 Python 引用
        无法回收 deepcopy 的 UNet（~0.9GB），而且共享组件上残留的 offload
        hook 会与新底模冲突。所以置 None 前必须先调本方法。
        """
        for attr in ("pipe", "v2v_pipe"):
            pipe = getattr(self, attr, None)
            if pipe is None:
                continue
            try:
                from accelerate.hooks import remove_hook_from_module
                for sub in ("unet", "vae", "text_encoder", "image_encoder"):
                    mod = getattr(pipe, sub, None)
                    if mod is not None:
                        remove_hook_from_module(mod, recurse=True)
            except Exception as e:
                logger.debug(f"[dispose] 移除 offload hook 失败: {e}")
            setattr(self, attr, None)
        self.adapter = None
        self.current_adapter_name = None
        self.loaded_motion_loras = []
        self._ipa_loaded = False
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("🗑️ AnimateDiff 管线已释放")

    def _has_offload_hooks(self, pipe) -> bool:
        """该管线（或其组件）是否已经挂过 accelerate offload hook。"""
        try:
            if getattr(pipe, "_all_hooks", None):
                return True
        except Exception:
            pass
        for attr in ("unet", "vae", "text_encoder", "image_encoder"):
            mod = getattr(pipe, attr, None)
            if mod is not None and getattr(mod, "_hf_hook", None) is not None:
                return True
        return False

    def _offload(self, pipe, already_offloaded: bool = False):
        """为新管线启用 offload / VAE 优化。

        ⚠️ 关键：`self.pipe` 与 `self.v2v_pipe` **共享** vae / text_encoder /
        scheduler（unet 是 deepcopy 出来的独立副本）。若对两者各调一次
        enable_model_cpu_offload()，accelerate 会在同一批 module 上重复安装
        hook（第二次覆盖第一次），典型症状就是
        "Expected all tensors to be on the same device"。
        所以只在第一条管线上装，第二条（already_offloaded=True）跳过。
        """
        from utils.vram_manager import VRAMManager

        # VAE 优化作用在共享的 pipe.vae 上，第一条管线设置一次即可
        if not already_offloaded:
            VRAMManager._set_vae_option(pipe, "slicing", True)
            VRAMManager._set_vae_option(pipe, "tiling", True)

        if already_offloaded or self._has_offload_hooks(pipe):
            logger.debug("⏭️ 组件已挂 offload hook，跳过重复安装")
            return

        try:
            pipe.enable_model_cpu_offload()
            logger.info("✅ 已启用 model CPU offload")
        except Exception as e:
            logger.warning(f"⚠️ cpu offload 失败，回退全量上卡: {e}")
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_scheduler(self, name="dpm++"):
        pipes = [p for p in (self.pipe, self.v2v_pipe) if p is not None]
        if not pipes:
            return
        cfg = pipes[0].scheduler.config
        name = (name or "dpm++").lower()
        if "dpm" in name:
            sched = DPMSolverMultistepScheduler.from_config(
                cfg, algorithm_type="dpmsolver++",
                use_karras_sigmas=True, beta_schedule="linear")
        elif "lcm" in name:
            sched = LCMScheduler.from_config(cfg, beta_schedule="linear")
        elif "ddim" in name:
            sched = DDIMScheduler.from_config(cfg, beta_schedule="linear")
        else:
            sched = EulerDiscreteScheduler.from_config(cfg, beta_schedule="linear")
        for p in pipes:
            p.scheduler = sched

    # ---------- Motion LoRA ----------
    def _load_motion_loras(self, configs):
        """[{'name':'zoom-in','weight':0.8}, ...]"""
        if self.pipe is None:
            return
        if self.loaded_motion_loras:
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
            self.loaded_motion_loras = []

        if not configs:
            return

        names, weights = [], []
        for cfg in configs:
            name = _clean_name(cfg.get("name", ""))
            if not name:
                continue
            path = os.path.join(MOTION_LORA_DIR, name)
            if not os.path.isdir(path):
                logger.warning(f"⚠️ Motion LoRA 不存在: {path}")
                continue
            aid = name.replace("-", "_")
            try:
                self.pipe.load_lora_weights(path, adapter_name=aid)
                names.append(aid)
                weights.append(float(cfg.get("weight", 0.8)))
                logger.info(f"✅ Motion LoRA: {name} @ {weights[-1]}")
            except Exception as e:
                logger.error(f"❌ 加载 {name} 失败: {e}")

        if names:
            self.pipe.set_adapters(names, adapter_weights=weights)
            if self.v2v_pipe is not None:
                try:
                    self.v2v_pipe.set_adapters(names, adapter_weights=weights)
                except Exception:
                    pass
            self.loaded_motion_loras = names

    # ---------- IP-Adapter ----------
    @staticmethod
    def _ipa_paths():
        """IP-Adapter 权重与编码器的路径。

        ⚠️ 用 project 锚定的绝对路径，不要 os.path.join("models", ...)：
        相对路径依赖 CWD，从别的目录启动时会误判"缺失"。
        """
        from utils import paths
        ipa_dir = os.path.join(paths.MODEL_DIR, "ip_adapter")
        return (ipa_dir,
                os.path.join(ipa_dir, "ip-adapter_sd15.safetensors"),
                os.path.join(ipa_dir, "image_encoder"))

    def ip_adapter_present(self) -> bool:
        _d, weight, encoder = self._ipa_paths()
        return (os.path.isfile(weight)
                and os.path.isfile(os.path.join(encoder, "model.safetensors")))

    def preflight_ip_adapter(self, parent=None) -> bool:
        """**必须在 GUI 线程调用**：图生视频前检查 IP-Adapter。

        缺失时告知体积与来源、由用户决定是否下载（项目定位：模型自行下载），
        而不是在后台线程里静默下载 2.5GB。
        返回 True 表示已就绪（原本就有或用户确认下载成功）。
        """
        if self.ip_adapter_present():
            return True
        from utils.model_downloader import confirm_and_install
        ok1 = confirm_and_install(
            "ip_adapter_sd15",
            "图生视频需要 IP-Adapter 锁定画面风格", parent=parent)
        ok2 = confirm_and_install(
            "ip_adapter_image_encoder",
            "IP-Adapter 的图像编码器（CLIP ViT-H）", parent=parent)
        return bool(ok1 and ok2 and self.ip_adapter_present())

    def _ensure_ip_adapter(self, scale: float = 0.6):
        """加载 IP-Adapter。缺失时**不下载**，由 preflight_ip_adapter 在
        GUI 线程提前处理；这里只做加载与显存安置。"""
        import torch

        ipa_dir, weight_path, encoder_dir = self._ipa_paths()

        if not getattr(self, "_ipa_loaded", False):
            if not self.ip_adapter_present():
                raise RuntimeError(
                    "❌ 图生视频需要 IP-Adapter，但模型缺失。\n"
                    "可自行下载到 models/ip_adapter/ 后重试，或运行：\n"
                    "python -m utils.model_downloader install ip_adapter_sd15")

            logger.info(f"🔌 加载 IP-Adapter: {weight_path}")
            self.pipe.load_ip_adapter(
                ipa_dir,
                subfolder="",
                weight_name="ip-adapter_sd15.safetensors",
                image_encoder_folder="image_encoder",
            )

            # 🔧 强制把 image_encoder 固定在 CUDA,并从 offload hook 剥离
            ie = getattr(self.pipe, "image_encoder", None)
            if ie is not None:
                target_device = "cuda" if torch.cuda.is_available() else "cpu"

                # 剥离 accelerate 的 offload hook,防止被搬回 CPU
                if hasattr(ie, "_hf_hook"):
                    try:
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(ie, recurse=True)
                        logger.info("🔧 已剥离 image_encoder 的 offload hook")
                    except Exception as e:
                        logger.warning(f"⚠️ 剥离 hook 失败: {e}")

                ie.to(device=target_device, dtype=torch.float16)
                logger.info(f"🔧 image_encoder → {target_device} / float16")

            self._ipa_loaded = True

        # 每次调用都更新 scale
        self.pipe.set_ip_adapter_scale(scale)
        logger.info(f"  IP-Adapter scale = {scale}")


    def _drop_ip_adapter(self):
        if self._ipa_loaded:
            try:
                self.pipe.unload_ip_adapter()
            except Exception:
                pass
            self._ipa_loaded = False

    # ---------- Prompt Travel ----------
    def _encode_single(self, text):
        tok = self.pipe.tokenizer
        te = self.pipe.text_encoder
        ids = tok(text or "", padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(te.device)
        with torch.no_grad():
            emb = te(ids)[0]
        return emb  # [1,77,dim]

    def _encode_prompt_travel(self, travel, negative, num_frames):
        pts = sorted([(int(f), p) for f, p in travel if p], key=lambda x: x[0])
        if not pts:
            return None, None
        if pts[0][0] > 0:
            pts.insert(0, (0, pts[0][1]))
        if pts[-1][0] < num_frames - 1:
            pts.append((num_frames - 1, pts[-1][1]))

        keys = [self._encode_single(p) for _, p in pts]
        idxs = [f for f, _ in pts]

        all_emb = []
        for f in range(num_frames):
            emb = keys[-1]
            for i in range(len(idxs) - 1):
                a, b = idxs[i], idxs[i + 1]
                if a <= f <= b:
                    t = (f - a) / max(b - a, 1)
                    emb = (1 - t) * keys[i] + t * keys[i + 1]
                    break
            all_emb.append(emb)
        pos = torch.cat(all_emb, dim=0)
        neg = self._encode_single(negative or DEFAULT_NEG).repeat(num_frames, 1, 1)
        return pos, neg

    # ---------- Utility ----------
    def list_adapters(self):
        if not os.path.isdir(MOTION_ADAPTER_DIR):
            return []
        return sorted([d for d in os.listdir(MOTION_ADAPTER_DIR)
                       if os.path.isdir(os.path.join(MOTION_ADAPTER_DIR, d))])

    def list_motion_loras(self):
        if not os.path.isdir(MOTION_LORA_DIR):
            return []
        return sorted([d for d in os.listdir(MOTION_LORA_DIR)
                       if os.path.isdir(os.path.join(MOTION_LORA_DIR, d))])

    # ---------- AnimateDiff-Lightning（4/8 步蒸馏） ----------
    @staticmethod
    def list_lightning():
        """列出 models/lightning/ 下的加速 LoRA 文件名。"""
        if not os.path.isdir(LIGHTNING_DIR):
            return []
        return sorted(f for f in os.listdir(LIGHTNING_DIR)
                      if f.lower().endswith((".safetensors", ".ckpt", ".pt")))

    @staticmethod
    def lightning_defaults(name):
        """按文件名猜推荐参数，返回 (steps, cfg)。

        ByteDance 的 AnimateDiff-Lightning 分 2/4/8 步三档，文件名里带
        "4step"/"8step"；4 步档推荐 CFG 1.0，8 步档 1.5。
        """
        low = str(name or "").lower()
        steps = 8 if ("8step" in low or "8-step" in low or "8_step" in low) else 4
        if "2step" in low or "2-step" in low:
            steps = 2
        cfg = 1.0 if steps <= 4 else 1.5
        return steps, cfg

    def _lightning_registered(self, pipe) -> bool:
        """LoRA 是否已经挂上（diffusers 会把 adapter 记在 peft_config 里）。"""
        for attr in ("unet", "transformer"):
            mod = getattr(pipe, attr, None)
            cfg = getattr(mod, "peft_config", None)
            if isinstance(cfg, dict) and "lightning" in cfg:
                return True
        return False

    def _apply_lightning(self, pipe, name, log_hint=True):
        """把加速 LoRA 挂到动画管线上（txt2video / v2v 各挂一次）。

        现实里遇到的 LoRA 分两种：干净的（官方 lightning 文件就是 UNet-only）
        和"带文本编码器部分、键名是 peft 原生写法"的（会让 diffusers 在
        TE 那半抛 ValueError，尽管 UNet 已经挂上）。这里两条路都兜住，
        复用 utils/lora_repair.py —— 和图片链路的插件加载同一套逻辑。
        """
        if pipe is None or not name:
            return False
        path = name if os.path.isabs(name) else os.path.join(LIGHTNING_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 找不到加速 LoRA: {path}\n"
                f"   请把 AnimateDiff-Lightning 的 safetensors 放到 {LIGHTNING_DIR}")
        first_err = None
        try:
            pipe.load_lora_weights(path, adapter_name="lightning")
        except Exception as e:
            first_err = e
            if not self._lightning_registered(pipe):
                # 原样加载失败且一点没挂上 —— 修键名再试一次
                try:
                    from safetensors.torch import load_file as _load_file
                    from utils import lora_repair
                    sd = _load_file(path)
                    sd, notes, warnings = lora_repair.repair_lora_state_dict(sd)
                    for w in warnings:
                        logger.warning(w)
                    pipe.load_lora_weights(sd, adapter_name="lightning")
                    logger.info(f"🔧 加速 LoRA 键名已修复后加载: {os.path.basename(name)}"
                                + (f"（{', '.join(notes)}）" if notes else ""))
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ 加速 LoRA 加载失败: {os.path.basename(name)}: {e}"
                        f"（修复尝试也失败: {e2}）")
            else:
                logger.warning(
                    f"🔧 加速 LoRA {os.path.basename(name)}：U-Net 部分已挂上，"
                    f"文本编码器部分失败（{first_err}）。动画加速照常生效。")
        try:
            pipe.set_adapters(["lightning"], adapter_weights=[1.0])
        except Exception as e:
            logger.debug(f"set_adapters(lightning) 失败，用默认权重: {e}")
        if log_hint:
            steps, cfg = self.lightning_defaults(name)
            logger.info(f"⚡ 已挂载加速 LoRA: {os.path.basename(name)}"
                        f"（推荐 {steps} 步 / CFG {cfg}）")
        return True

    # ---------- 真·长视频：分段 + 尾帧续接 ----------
    def _run_segmented(self, prompt, negative, total_frames, common, *,
                       mode="txt2video", input_path=None, strength=0.75,
                       segment_frames=DEFAULT_SEGMENT_FRAMES,
                       continuation_strength=CONTINUATION_STRENGTH, progress_callback=None,
                       continuation_frames=CONTINUATION_FRAMES, segment_callback=None):
        """把整段拆成若干 <=segment_frames 的小段依次生成，后一段用前一段的
        最后一帧当 v2v 首帧参考，最后拼起来。

        为什么这样做：motion module 只按 16 帧训练，一次要 150 帧动作必然崩；
        分段 + 尾帧续接能让动作越长越"顺"，代价是段与段之间可能有一次轻微跳变
        （用 continuation_strength 控制：越大越敢动，越小越贴上一段）。
        """
        seg = max(1, min(int(segment_frames or DEFAULT_SEGMENT_FRAMES),
                         SINGLE_PASS_MAX_FRAMES))
        total = max(1, int(total_frames))
        # 尽量均分：20 帧 + 上限 16 → 10+10（而不是 16+4，尾巴太短会明显漂）
        parts = max(1, -(-total // seg))
        base, extra = divmod(total, parts)
        plan = [base + (1 if i < extra else 0) for i in range(parts)]
        logger.info(f"🧱 分段生成：共 {total} 帧，每段最多 {seg} 帧 "
                    f"→ {len(plan)} 段 {plan}；续接强度 {continuation_strength:.2f}")

        out = []
        first_image = None
        if mode == "img2video" and input_path:
            first_image = self._load_first_frame(input_path, common["width"], common["height"])

        for idx, n in enumerate(plan):
            # 注意：不要在这里额外调 progress_callback —— 它会被塞进
            # callback_on_step_end，diffusers 每步自己会调，额外插一次会让
            # 界面进度条跳变。段级进度只用日志表达。
            logger.info(f"🧱 第 {idx + 1}/{len(plan)} 段开始（{n} 帧）")
            if segment_callback is not None:
                try:
                    segment_callback(idx, len(plan), n)
                except Exception as e:
                    logger.debug(f"[seg] 段回调失败: {e}")

            if idx == 0 and mode == "img2video" and first_image is not None:
                result = self.v2v_pipe(video=[first_image] * n, prompt=prompt,
                                       negative_prompt=negative,
                                       strength=float(strength), **common)
                got = result.frames[0]
            elif idx == 0:
                result = self.pipe(prompt=prompt, negative_prompt=negative,
                                   num_frames=n, **common)
                got = result.frames[0]
            else:
                # 续接：把前一段**最后 k 帧**当作真实时间前缀喂进去，再补 n 帧，
                # 生成 k+n 帧后丢掉前缀。三种做法的真机对比（384²/10 步/48 帧）：
                #   A 只放一张尾帧      → 接缝 5.9（顺），但段内帧差只有 2.8（几乎静止）
                #   B 4 帧循环铺满本段  → 段内 20.6（动了），接缝 30.5（每段像换镜头）
                #   C 4 帧前缀 + 丢弃   → 模型在真实时间轴上续写，接缝与运动都保住
                k = max(0, min(int(continuation_frames or 0), len(out), n))
                if k >= 2:
                    prefix = out[-k:]
                    # 输入长度 = k + n：前 k 帧是历史，后面补 n 帧静止尾帧让模型往外画
                    pad = [prefix[-1]] * n
                    video_in = prefix + pad
                    result = self.v2v_pipe(video=video_in, prompt=prompt,
                                           negative_prompt=negative,
                                           strength=float(continuation_strength),
                                           **common)
                    got = result.frames[0][k:]
                else:
                    tail = out[-1]
                    result = self.v2v_pipe(video=[tail] * n, prompt=prompt,
                                           negative_prompt=negative,
                                           strength=float(continuation_strength),
                                           **common)
                    got = result.frames[0]
            got = list(got)
            if idx > 0 and out:
                got = self._match_segment_color(got, out[-1])
            out.extend(got)
            logger.info(f"   ✅ 第 {idx + 1}/{len(plan)} 段完成（{len(got)} 帧）")

        if len(out) > total:
            out = out[:total]
        return out

    @staticmethod
    def _match_segment_color(frames, ref_frame, max_shift=18.0):
        """把一段的整体亮度/对比度向参考帧拉齐，减少接缝处的"闪一下"。

        只做逐通道的均值 + 标准差匹配，且偏移量夹在 max_shift 内 —— 宁可少改，
        也不要把画面洗白。返回新列表（PIL Image）。
        """
        if not frames or ref_frame is None:
            return frames
        try:
            ref = np.asarray(ref_frame).astype(np.float32)
            ref_m = ref.reshape(-1, ref.shape[-1]).mean(axis=0)
            ref_s = max(1e-3, float(ref.reshape(-1, ref.shape[-1]).std()))
            arr = np.asarray(frames[0]).astype(np.float32)
            cur_m = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
            cur_s = max(1e-3, float(arr.reshape(-1, arr.shape[-1]).std()))
            gain = float(np.clip(ref_s / cur_s, 0.92, 1.08))
            shift = np.clip(ref_m - cur_m, -max_shift, max_shift)
            if float(np.abs(shift).max()) < 2.0 and abs(gain - 1.0) < 0.01:
                return frames
            out = []
            for f in frames:
                a = np.asarray(f).astype(np.float32)
                a = a * gain + shift.reshape(1, 1, -1)
                out.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
            logger.debug(f"[seg] 颜色对齐：shift={np.round(shift, 1).tolist()} gain={gain:.3f}")
            return out
        except Exception as e:
            logger.debug(f"[seg] 颜色对齐失败: {e}")
            return frames

    @staticmethod
    def _repair_black_frames(frames, threshold=1.0):
        """把纯黑帧换成相邻的好帧。返回新的帧列表（并记录修复数量）。

        threshold 是"整帧均值 <= 这个值就算全黑"（0~255 尺度，1.0 已经很宽容，
        正常画面最低也有 30+）。
        """
        if not frames:
            return frames
        means = []
        for f in frames:
            try:
                means.append(float(np.asarray(f).mean()))
            except Exception:
                means.append(255.0)
        bad = [i for i, m in enumerate(means) if m <= threshold]
        if not bad:
            return frames
        good = [i for i, m in enumerate(means) if m > threshold]
        if not good:
            logger.warning("⚠️ 整段视频全是黑帧（latents 全部 NaN），建议换种子或降 CFG 重试")
            return frames
        out = list(frames)
        for i in bad:
            prev = max((g for g in good if g < i), default=None)
            nxt = min((g for g in good if g > i), default=None)
            src_idx = prev if prev is not None else nxt
            src = frames[src_idx]
            try:
                out[i] = src.copy()
            except Exception:
                out[i] = src
        logger.warning(
            f"🧽 修复了 {len(bad)} 帧纯黑帧（fp16 偶发 NaN），已用相邻帧补齐：{bad[:8]}"
            + ("…" if len(bad) > 8 else ""))
        return out

    @staticmethod
    def _load_first_frame(path, w, h):
        return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)

    @staticmethod
    def _load_video_frames(path, n, w, h):
        import cv2
        from utils.video_io import SafeVideoCapture
        # 用 SafeVideoCapture 而不是直接 cv2.VideoCapture：Windows 下 OpenCV
        # 打不开含中文的路径（会静默返回 isOpened()=False），视频用户几乎
        # 一定会用中文目录名。
        cap = SafeVideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"❌ 无法读取视频: {path}")
        targets = np.linspace(0, total - 1, n).astype(int)
        frames, cur, last = [], 0, None
        for t in targets:
            while cur <= t:
                ret, fr = cap.read()
                if not ret:
                    break
                cur += 1
                last = fr
            if last is None:
                break
            rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))
        cap.release()
        while len(frames) < n:
            frames.append(frames[-1] if frames else Image.new("RGB", (w, h)))
        return frames

    # ---------- Main ----------
    def generate(
        self,
        prompt,
        negative="",
        num_frames=16,
        num_steps=25,
        guidance=7.5,
        width=512,
        height=512,
        fps=8,
        scheduler="dpm++",
        motion_loras=None,
        motion_lora=None,
        motion_scale=0.8,
        use_context_window=False,
        prompt_travel=None,
        output_format="mp4",
        output_dir="photo/videos",
        progress_callback=None,
        mode="txt2video",       # txt2video / img2video / vid2vid / prompt_travel
        input_path=None,
        strength=0.75,          # v2v
        ip_adapter_scale=0.7,   # i2v
        seed=-1,
        lightning=None,         # AnimateDiff-Lightning 文件名（models/lightning/ 下）
        segment_frames=DEFAULT_SEGMENT_FRAMES,
        continuation_strength=CONTINUATION_STRENGTH,
        continuation_frames=CONTINUATION_FRAMES,
        segment_callback=None,  # (段序号, 总段数, 本段帧数) → 界面显示"第 2/4 段"
    ):
        mode = (mode or "txt2video").lower()
        # 单次生成上限是硬上限：超过 16 帧 motion module 的位置编码会越界，出片直接崩。
        _seg_target = min(int(segment_frames or DEFAULT_SEGMENT_FRAMES),
                          SINGLE_PASS_MAX_FRAMES)
        num_frames = max(1, int(num_frames))
        _want_frames = num_frames
        if _want_frames > SINGLE_PASS_MAX_FRAMES and not bool(use_context_window) \
                and _want_frames > DEFAULT_MAX_FRAMES:
            # 没勾「长视频模式」又要超过默认上限 → 夹到默认上限，绝不静默产出垃圾
            logger.warning(
                f"⚠️ 未勾选「长视频模式」时最多 {DEFAULT_MAX_FRAMES} 帧，"
                f"本次已把 {_want_frames} 帧限制为 {DEFAULT_MAX_FRAMES} 帧；"
                f"想要更多请勾选「长视频模式」（分段生成，上限 150 帧）。")
            num_frames = DEFAULT_MAX_FRAMES
            _want_frames = DEFAULT_MAX_FRAMES

        if _want_frames > SINGLE_PASS_MAX_FRAMES:
            # 只要超过 16 帧就一定分段：这是硬上限，单次生成必然崩成噪声
            uses_segments = True
            logger.info(
                f"🧱 {_want_frames} 帧超过单次上限 {SINGLE_PASS_MAX_FRAMES} 帧，"
                f"自动改成分段生成（每段 {_seg_target} 帧，尾帧续接）")
        else:
            uses_segments = False
        # 图生视频也用 v2v 管线（首帧约束），否则画面根本不会从那张图开始
        need_v2v = mode in ("vid2vid", "img2video") or uses_segments

        self._build_pipe(scheduler=scheduler, need_v2v=need_v2v)

        # AnimateDiff-Lightning：先挂 LoRA，再把步数/CFG 拉回蒸馏档位
        if lightning:
            self._apply_lightning(self.pipe, lightning)
            if need_v2v and self.v2v_pipe is not None:
                self._apply_lightning(self.v2v_pipe, lightning, log_hint=False)
            l_steps, l_cfg = self.lightning_defaults(lightning)
            if int(num_steps) != l_steps or abs(float(guidance) - l_cfg) > 1e-6:
                logger.info(f"⚡ Lightning 参数覆盖：steps {num_steps}→{l_steps}，"
                            f"CFG {float(guidance):g}→{l_cfg:g}")
            num_steps, guidance = l_steps, l_cfg

        lora_cfgs = []
        if motion_loras:
            lora_cfgs = [c for c in motion_loras if c.get("name")]
        elif motion_lora:
            lora_cfgs = [{"name": motion_lora, "weight": motion_scale}]
        self._load_motion_loras(lora_cfgs)

        # Seed
        if seed is None or seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        if not negative or not negative.strip():
            negative = DEFAULT_NEG

        common = dict(
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=gen,
        )
        # 每步都挂 NaN 守卫（进度回调链在里面），否则 fp16 偶发溢出会产出纯黑帧
        common["callback_on_step_end"] = _make_step_guard(progress_callback)

        logger.info(f"🎬 [{mode}] {num_frames}帧 {width}x{height} steps={num_steps} seed={seed}")
        self._warn_prompt_length(prompt, negative)
        # 分段生成时这条告警没有意义（我们本来就是分段做的）
        if num_frames >= SINGLE_PASS_MAX_FRAMES:
            logger.info(
                f"ℹ️ 本次 {num_frames} 帧已到单次上限（motion module 只按 "
                f"{SINGLE_PASS_MAX_FRAMES} 帧训练）；想更长请勾「长视频模式」分段生成。")

        frames = None

        # 真·长视频：帧数超过单段上限且勾了"长视频"/上下文窗口时，走分段 + 尾帧续接
        if uses_segments:
            if mode == "prompt_travel" and prompt_travel:
                logger.warning("⚠️ 分段生成模式下不支持 Prompt Travel（每段的提示词走同一个），"
                               "需要按帧改提示词请关掉「长视频模式」")
            self._drop_ip_adapter()
            frames = self._run_segmented(
                prompt, negative, num_frames, common,
                mode=mode, input_path=input_path, strength=strength,
                segment_frames=segment_frames,
                continuation_strength=continuation_strength,
                continuation_frames=continuation_frames,
                progress_callback=progress_callback,
                segment_callback=segment_callback,
            )
        elif mode == "prompt_travel" and prompt_travel and len(prompt_travel) >= 2:
            self._drop_ip_adapter()
            logger.info(f"📝 Prompt Travel: {len(prompt_travel)} 段关键帧")
            for f, p in prompt_travel:
                logger.info(f"   [帧 {f}] {p[:60]}")
            frames = self._run_prompt_travel(
                prompt_travel, negative, num_frames, guidance, common
            )

        elif mode == "img2video":
            if not input_path or not os.path.exists(input_path):
                raise FileNotFoundError(f"❌ 图生视频需要首帧图: {input_path}")
            self._drop_ip_adapter()
            first = self._load_first_frame(input_path, width, height)
            # 真正的首帧约束：把这张图重复成 num_frames 帧当 v2v 的参考序列。
            # strength 越大越贴近原图（0.85 基本只动光影和细节，0.55 允许较大改动）。
            strength_i2v = float(strength) if strength is not None else 0.75
            logger.info(f"🖼️ 图生视频：首帧约束 strength={strength_i2v:.2f}")
            result = self.v2v_pipe(
                video=[first] * num_frames,
                prompt=prompt,
                negative_prompt=negative,
                strength=strength_i2v,
                **common,
            )
            frames = result.frames[0]

        elif mode == "vid2vid":
            if not input_path or not os.path.exists(input_path):
                raise FileNotFoundError(f"❌ 视频转绘需要输入视频: {input_path}")
            self._drop_ip_adapter()
            vin = self._load_video_frames(input_path, num_frames, width, height)
            result = self.v2v_pipe(
                video=vin,
                prompt=prompt,
                negative_prompt=negative,
                strength=float(strength),
                **common,
            )
            frames = result.frames[0]

        else:  # txt2video (含 prompt_travel 段数不足的兜底)
            self._drop_ip_adapter()
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_frames=num_frames,
                **common,
            )
            frames = result.frames[0]

        # 兜底修补：万一仍漏出纯黑帧，用相邻好帧补上（并如实告警）
        frames = self._repair_black_frames(frames)

        # 保存：支持 MP4 / GIF / MP4 + GIF（两个都存，不是二选一）
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        want = str(output_format or "MP4").lower()
        want_mp4 = "mp4" in want
        want_gif = "gif" in want
        if not want_mp4 and not want_gif:
            want_mp4 = True
        stem = os.path.join(output_dir, f"video_{mode}_{ts}_seed{seed}")
        out = f"{stem}.mp4" if want_mp4 else f"{stem}.gif"
        if want_mp4:
            self._save(frames, f"{stem}.mp4", fps=fps, fmt="mp4")
            logger.info(f"✅ 已保存: {stem}.mp4")
        if want_gif:
            self._save(frames, f"{stem}.gif", fps=fps, fmt="gif")
            logger.info(f"✅ 已保存: {stem}.gif")
        return out, seed

    def _warn_prompt_length(self, prompt, negative):
        """超过 CLIP 77 token 会被截断 —— 不能沉默，要让用户知道丢了什么。"""
        tok = getattr(self.pipe, "tokenizer", None)
        if tok is None:
            return
        for label, text in (("正面", prompt), ("负面", negative)):
            if not text:
                continue
            try:
                n = len(tok(text, add_special_tokens=True)["input_ids"])
            except Exception:
                continue
            if n > CLIP_MAX_TOKENS:
                logger.warning(
                    f"✂️ {label}提示词 {n} token，超过 SD1.5 的 {CLIP_MAX_TOKENS} 上限，"
                    f"大约后面 {n - CLIP_MAX_TOKENS} 个词不会生效。"
                    f"视频面板暂不支持长提示词分段（可缩短，或删掉冗余修饰词）。")

    def _save(self, frames, path, fps=8, fmt="mp4"):
        if fmt == "gif":
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0, optimize=False)
        else:
            import imageio
            imageio.mimsave(path, [np.array(f) for f in frames],
                            fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)

    def _run_prompt_travel(self, prompt_travel, negative, num_frames, guidance, common):
        """
        Prompt Travel 专用推理:
        通过 monkey-patch UNet.forward 注入 per-frame encoder_hidden_states,
        并绕过 UNetMotionModel 内部的 repeat_interleave 广播。
        """
        pos_pf, neg_pf = self._encode_prompt_travel(prompt_travel, negative, num_frames)
        if pos_pf is None:
            raise RuntimeError("❌ prompt_travel 编码失败")

        device = self.pipe.unet.device
        dtype  = self.pipe.unet.dtype
        pos_pf = pos_pf.to(device=device, dtype=dtype)   # [nf, 77, dim]
        neg_pf = neg_pf.to(device=device, dtype=dtype)   # [nf, 77, dim]

        do_cfg = float(guidance) > 1.0

        # dummy 单帧 embeds:仅用于让 pipeline 通过尺寸检查(batch_size=1)
        dummy_pos = pos_pf[:1]
        dummy_neg = neg_pf[:1]

        # 真正要注入的 full encoder_hidden_states
        if do_cfg:
            full_ehs = torch.cat([neg_pf, pos_pf], dim=0)   # [2*nf, 77, dim]
        else:
            full_ehs = pos_pf                                # [nf, 77, dim]

        unet = self.pipe.unet
        orig_forward = unet.forward
        orig_repeat  = torch.Tensor.repeat_interleave

        nf = num_frames
        expected_batch = full_ehs.shape[0]   # 2*nf or nf

        def noop_repeat(tensor_self, repeats, dim=0, *args, **kwargs):
            # 只拦截 encoder_hidden_states 的那一次广播
            try:
                if (dim == 0
                        and isinstance(repeats, int)
                        and repeats == nf
                        and tensor_self.ndim == 3
                        and tensor_self.shape[0] == expected_batch):
                    return tensor_self  # 已经是 per-frame,不再复制
            except Exception:
                pass
            return orig_repeat(tensor_self, repeats, dim=dim, *args, **kwargs)

        def patched_forward(sample, timestep, encoder_hidden_states, **kwargs):
            # sample: [b, 4, nf, h/8, w/8],b=2(cfg) 或 1
            b = sample.shape[0]
            # 用完整 per-frame embeds 覆盖 pipeline 传入的 dummy
            if full_ehs.shape[0] != b * nf:
                # batch 不匹配时兜底(理论上不会发生)
                return orig_forward(sample, timestep, encoder_hidden_states, **kwargs)

            torch.Tensor.repeat_interleave = noop_repeat
            try:
                return orig_forward(sample, timestep, full_ehs, **kwargs)
            finally:
                torch.Tensor.repeat_interleave = orig_repeat

        unet.forward = patched_forward
        try:
            result = self.pipe(
                prompt_embeds=dummy_pos,
                negative_prompt_embeds=dummy_neg,
                num_frames=num_frames,
                **common,
            )
        finally:
            unet.forward = orig_forward
            torch.Tensor.repeat_interleave = orig_repeat

        return result.frames[0]
