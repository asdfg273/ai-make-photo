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

DEFAULT_NEG = (
    "bad hands, bad fingers, extra fingers, missing fingers, deformed hands, "
    "orange tint, warm color cast, oversaturated, "
    "lowres, worst quality, low quality, jpeg artifacts, blurry, watermark"
)


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
            self._offload(self.v2v_pipe)
            logger.info("✅ AnimateDiff v2v 就绪")

        self._apply_scheduler(scheduler)

    def _offload(self, pipe):
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
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
    def _ensure_ip_adapter(self, scale: float = 0.6):
        """确保 IP-Adapter 已加载,缺失时自动下载"""
        import torch

        ipa_dir = os.path.join("models", "ip_adapter")
        weight_path = os.path.join(ipa_dir, "ip-adapter_sd15.safetensors")
        encoder_dir = os.path.join(ipa_dir, "image_encoder")

        if not getattr(self, "_ipa_loaded", False):
            # 检查缺失 → 自动下载
            if not os.path.isfile(weight_path) or not os.path.isfile(
                os.path.join(encoder_dir, "model.safetensors")
            ):
                logger.info("📥 IP-Adapter 缺失,开始自动下载...")
                from utils.model_downloader import install
                install("ip_adapter_sd15")
                install("ip_adapter_image_encoder")

            if not os.path.isfile(weight_path):
                raise RuntimeError(
                    "❌ IP-Adapter 下载失败,请手动运行: "
                    "python -m utils.model_downloader install ip_adapter_sd15"
                )

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

    @staticmethod
    def _load_first_frame(path, w, h):
        return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)

    @staticmethod
    def _load_video_frames(path, n, w, h):
        import cv2
        cap = cv2.VideoCapture(path)
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
        # 新增
        mode="txt2video",       # txt2video / img2video / vid2vid / prompt_travel
        input_path=None,
        strength=0.75,          # v2v
        ip_adapter_scale=0.7,   # i2v
        seed=-1,
    ):
        mode = (mode or "txt2video").lower()
        need_v2v = (mode == "vid2vid")

        self._build_pipe(scheduler=scheduler, need_v2v=need_v2v)

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
        if progress_callback is not None:
            common["callback_on_step_end"] = progress_callback

        logger.info(f"🎬 [{mode}] {num_frames}帧 {width}x{height} steps={num_steps} seed={seed}")

        frames = None

        if mode == "prompt_travel" and prompt_travel and len(prompt_travel) >= 2:
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
            self._ensure_ip_adapter(scale=ip_adapter_scale)
            first = self._load_first_frame(input_path, width, height)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_frames=num_frames,
                ip_adapter_image=first,
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

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "gif" if str(output_format).lower().endswith("gif") else "mp4"
        out = os.path.join(output_dir, f"video_{mode}_{ts}_seed{seed}.{ext}")
        self._save(frames, out, fps=fps, fmt=ext)
        logger.info(f"✅ 已保存: {out}")
        return out, seed

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