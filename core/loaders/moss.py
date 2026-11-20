# core/loaders/moss.py
"""MOSS-Image（OpenMOSS 5B DiT）加载器。

和 SD 系最大的不同：**官方权重是 diffusers 目录，不是单文件检查点**。
仓库里除了 transformer / vae / text_encoder 等组件，还带一个自定义管线文件
`pipeline_moss_image.py`（官方示例用
`DiffusionPipeline.from_pretrained(repo, custom_pipeline="pipeline_moss_image",
trust_remote_code=True, ...)` 加载）。

本加载器的取舍
--------------
1. **只认本地目录**：不拿 repo id 去联网（Anima 就是踩过这个坑：离线时
   卡在 5 次重试上，最后抛出一句和真实原因无关的 httpx 错误）。
   目录可以在 `models/mossimage/` 或 `models/puton/` 下，也可以在 HF 缓存里
   （用 `local_files_only=True` 查询，绝不发网络请求）。
2. **优先用目录里的管线文件**，而不是 `trust_remote_code=True`：
   后者会在加载时执行仓库里的代码，来源不受控。目录里那个 .py 是用户
   自己下载的，按本地文件路径传给 diffusers，语义一样但不引入远程执行。
3. **显存不交给 `utils.vram_manager`，走本文件自己的路径**：5B bf16 权重
   约 10GB，而 `vram_manager` 的通用策略会调 `enable_model_cpu_offload()`，
   那会把 16GB 的文本编码器也搬上卡 —— 8GB 卡必然 OOM（实测撞过）。所以这里
   按「谁放显存、谁留内存」分三档自己排：
     * 默认：DiT 留 CPU + block 级 group offload，文本编码器留 CPU，VAE 上卡；
     * `moss_quant=int8`：DiT 量化到 4.7GB 后常驻显存（省约 9.5GB 内存）；
     * `moss_quant=int8te`：再把文本编码器量化到 6.9GB，内存峰值从 23.8GB
       降到 11.0GB（换提示词时编码变慢）。
   三个分支各自有显存/内存预检，不够时给的是带数字的中文报错，不是空
   `MemoryError`（见 `_apply_int8_vram` / `_apply_accel_vram`）。

官方采样参数（README）：Flow-matching Euler、50 步、CFG 3.0、shift 5、
`uncond_skip_layers=[22]`。其中 `uncond_skip_layers` 由调用方注入
（`utils/app_generation.py::_adapt_call` 与 `workflow/adapters.py`）。
"""
import logging
import os

from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


_ORIGINAL_ENCODE_PROMPT = None

#: 提示词向量缓存容量（条）。文本编码器在 8GB 卡上只能跑 CPU，一次编码
#: 实测 15 秒；同一段提示词连出多张（批量、抽卡、改种子）时这 15 秒纯浪费。
_PROMPT_CACHE_SIZE = 4


def _encode_prompt_on_encoder_device(self, prompt, max_sequence_length=4096):
    """官方 `encode_prompt` 的设备修正版 + 提示词向量缓存。

    官方实现把 tokenized 输入 `.to(self._execution_device)` 再喂文本编码器；
    可 8GB 卡上文本编码器（16GB）只能待在 CPU，而 `_execution_device` 是
    显卡 → `RuntimeError: Expected all tensors to be on the same device`。
    这里改成：输入留在**文本编码器所在设备**，输出也留在那儿（`__call__`
    里的 `prepare()` 会自己搬到显卡），于是文本编码器永远不必上卡。

    命中缓存时直接返回**CPU 上的**旧结果（`prepare()` 会搬到显卡），
    因此缓存不会占显存。
    """
    import torch

    if self.text_encoder is None or self.tokenizer is None:
        raise ValueError("Text encoder/tokenizer required for string prompts.")
    prompts_tuple = (prompt,) if isinstance(prompt, str) else tuple(prompt)
    cache = getattr(self, "_moss_prompt_cache", None)
    if cache is None:
        cache = {}
        try:
            self._moss_prompt_cache = cache
            self._moss_prompt_cache_hits = 0
        except Exception as e:
            logger.debug("MOSS-Image 提示词缓存无法挂载（忽略）: %s", e)
    key = (prompts_tuple, int(max_sequence_length))
    hit = cache.get(key)
    if hit is not None:
        cache[key] = cache.pop(key)          # LRU：把命中项挪到末尾
        try:
            self._moss_prompt_cache_hits = int(getattr(
                self, "_moss_prompt_cache_hits", 0)) + 1
        except Exception as e:
            logger.debug("MOSS-Image 缓存计数失败（忽略）: %s", e)
        logger.info("♻️ 命中提示词向量缓存（省下一次文本编码，约 15 秒）")
        return hit
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    if not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("prompt must be a string or a nonempty list of strings.")
    try:
        device = next(self.text_encoder.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    rows = []
    for value in prompts:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": value}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                                max_length=max_sequence_length).to(device)
        output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
        rows.append(output.last_hidden_state[0])

    length = max(row.shape[0] for row in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
    for index, row in enumerate(rows):
        embeds[index, :row.shape[0]] = row
        mask[index, :row.shape[0]] = True

    # 存一份 CPU 副本：下次同样提示词直接复用（不占显存）
    cache = getattr(self, "_moss_prompt_cache", None)
    if isinstance(cache, dict):
        try:
            cache[key] = (embeds.detach().to("cpu"), mask.detach().to("cpu"))
            while len(cache) > _PROMPT_CACHE_SIZE:
                cache.pop(next(iter(cache)))
        except Exception as e:
            logger.debug("MOSS-Image 提示词缓存写入失败（忽略）: %s", e)
    return embeds, mask


def _patch_prompt_encoding(pipe) -> bool:
    """把管线实例上的 `encode_prompt` 换成设备自适应的版本（幂等）。"""
    import types

    if getattr(pipe, "_moss_encode_patched", False):
        return True
    try:
        pipe.encode_prompt = types.MethodType(_encode_prompt_on_encoder_device, pipe)
    except Exception as e:
        logger.warning("⚠️ MOSS-Image encode_prompt 设备修正失败（%s），"
                       "文本编码器将被迫和显卡同设备", e)
        return False
    pipe._moss_encode_patched = True
    return True


class MossImageLoader(BaseLoader):
    """MOSS-Image：diffusers 目录 + 自定义管线。"""

    arch_ids = ("mossimage",)

    REPO = "OpenMOSS-Team/MOSS-Image-Base"
    PIPELINE_FILE = "pipeline_moss_image.py"
    # models/<x>/ 下的候选目录（puton 是投递箱，由 model_inbox 归位）
    _SEARCH = ("mossimage", "puton")
    _MARKERS = ("model_index.json", "modular_model_index.json")
    # 这两块必须真有权重，否则管线建不起来
    _REQUIRED_PARTS = ("transformer", "vae")

    # ---------------------------------------------------------------- 目录探测
    @classmethod
    def _looks_like_pipeline_dir(cls, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(os.path.isfile(os.path.join(path, m))
                   for m in cls._MARKERS)

    @classmethod
    def _dir_problem(cls, path: str) -> str:
        """目录能不能直接用。可用返回 ""，否则返回中文原因。

        ⚠️ 必须检查体积：下到一半的目录会留下 model_index.json，
        但组件权重是 0 字节，diffusers 会抛一个和"缺权重"无关的 JSON 错误。
        ⚠️ HF 缓存里的文件是**符号链接**，只能用 os.path.getsize 判断
        （PowerShell 的 Length 对符号链接显示 0）。
        """
        if not os.path.isdir(path):
            return "目录不存在"
        marker = ""
        for name in cls._MARKERS:
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                marker = fp
                break
        if not marker:
            return "缺 " + " / ".join(cls._MARKERS)
        try:
            if os.path.getsize(marker) == 0:
                return f"{os.path.basename(marker)} 是 0 字节（下载未完成）"
        except OSError as e:
            return f"读不到 {os.path.basename(marker)}: {e}"

        bad = []
        for part in cls._REQUIRED_PARTS:
            d = os.path.join(path, part)
            if not os.path.isdir(d):
                bad.append(f"{part}(无目录)")
                continue
            total = 0
            try:
                names = os.listdir(d)
            except OSError as e:
                bad.append(f"{part}(读不到: {e})")
                continue
            for name in names:
                if not name.endswith((".safetensors", ".bin")):
                    continue
                try:
                    total += os.path.getsize(os.path.join(d, name))
                except OSError:
                    continue
            if total == 0:
                bad.append(f"{part}(0 字节)")
        if bad:
            return "组件缺失或为空: " + ", ".join(bad)
        return ""

    def local_dirs(self) -> list:
        """用户手动放置的 MOSS-Image 目录（不联网、不下载）。"""
        from utils import paths

        found = []
        for sub in self._SEARCH:
            root = os.path.join(paths.MODEL_DIR, sub)
            if self._looks_like_pipeline_dir(root):
                found.append(root)
                continue
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                cand = os.path.join(root, name)
                if self._looks_like_pipeline_dir(cand):
                    found.append(cand)
        return found

    def cached_dirs(self) -> list:
        """HF 缓存里已下载的快照目录（可能不完整）。"""
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            logger.debug(f"huggingface_hub 不可用（忽略缓存探测）: {e}")
            return []
        try:
            path = snapshot_download(self.REPO, local_files_only=True)
        except Exception as e:
            logger.debug(f"{self.REPO} 不在本地缓存: {e}")
            return []
        return [path] if self._looks_like_pipeline_dir(path) else []

    def _missing_message(self, problems=None) -> str:
        from utils import paths

        want = os.path.join(paths.MODEL_DIR, "mossimage",
                            self.REPO.split("/")[-1])
        lines = [
            "❌ 找不到可用的 MOSS-Image 模型目录。",
            f"   需要: {self.REPO}（diffusers 目录形式，bf16 约 10GB）",
            "   ⚠️ MOSS-Image 不是单文件检查点：必须整个目录一起放，",
            "      里面有 model_index.json、transformer/、vae/、text_encoder/",
            f"      以及自定义管线文件 {self.PIPELINE_FILE}。",
        ]
        if problems:
            lines.append("")
            lines.append("   已找到但不可用的位置：")
            for p in problems[:4]:
                lines.append(f"     · {p}")
        lines += [
            "",
            "   请自行准备（本程序不提供下载入口，也不会在生成时自动联网）：",
            f"     1) 把整个目录放进: {want}",
            f"     2) 或放进 {os.path.join(paths.MODEL_DIR, 'puton')}\\<目录>"
            "（投递箱会自动归位到 models/mossimage/）",
            "     3) 以前下载过、还在 HuggingFace 缓存里的，程序直接复用；",
            "        缓存是「下到一半」的（下面报了 0 字节）就删掉重新准备。",
            "",
            "   ℹ️ Comfy-Org 的「ComfyUI 单文件格式」本项目暂不支持 ——",
            "      请用上面的 diffusers 目录版本。",
        ]
        return "\n".join(lines)

    def resolve_dir(self) -> tuple:
        """返回 (目录, 来源说明)；找不到就抛 RuntimeError。"""
        problems = []
        candidates = ([(d, "本地目录") for d in self.local_dirs()]
                      + [(d, "HF 缓存") for d in self.cached_dirs()])
        for src, how in candidates:
            why = self._dir_problem(src)
            if not why:
                return src, how
            problems.append(f"{src}（{how}）→ {why}")
            logger.warning(f"⚠️ MOSS-Image 目录不可用: {src} → {why}")
        raise RuntimeError(self._missing_message(problems))

    # ------------------------------------------------------------------ 装配
    @staticmethod
    def _dir_weight_gb(path):
        """目录里权重文件合计体积（GB）。"""
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                if not name.lower().endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total / (1024 ** 3)

    def build(self, model_path, ctx):
        import torch
        from diffusers import DiffusionPipeline

        if model_path and not os.path.isdir(model_path):
            raise RuntimeError(
                "❌ MOSS-Image 只能用 **diffusers 目录** 加载，不支持单文件。\n"
                f"   当前选中的是文件: {model_path}\n"
                "   请把整个模型目录放进 models/mossimage/ 后在列表里选择目录。")

        why = self._dir_problem(model_path) if model_path else ""
        if model_path and not why:
            target, how = model_path, "所选目录"
        else:
            if why:
                logger.warning(f"⚠️ 所选目录不可用（{why}），改为自动查找")
            target, how = self.resolve_dir()

        logger.info(f"⏳ 装配 MOSS-Image（{how}）: {target}")

        # 内存预检：这套权重 25.4GB，实测加载后 RSS 约 18GB。
        # 内存不够时 diffusers 内部只会抛一个 `MemoryError:`（空消息），
        # 被包成"管线装配失败"，用户完全看不出是内存问题 —— 那是最难查的一类。
        weight_gb = self._dir_weight_gb(target)
        hard_gb = weight_gb * 0.55     # 绝对下限：低于这个必然失败
        want_gb = weight_gb * 0.75     # 建议值：低于它有失败风险
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if avail_gb < hard_gb:
                raise RuntimeError(
                    "❌ 内存不够，MOSS-Image 装不进来。\n"
                    f"   这个模型约 {weight_gb:.1f}GB"
                    "（DiT 9.7GB + 文本编码器 16GB），"
                    f"加载时至少要 {hard_gb:.0f}GB 可用内存。\n"
                    f"   当前：可用 {avail_gb:.1f}GB / 总共 {total_gb:.1f}GB。\n"
                    "   请先关掉占内存的程序（浏览器、游戏、其它 AI 工具）再试；\n"
                    "   或者把 Windows 的「虚拟内存/页面文件」调大一些。")
            if avail_gb < want_gb:
                logger.warning(
                    f"⚠️ 可用内存只有 {avail_gb:.1f}GB（建议 {want_gb:.0f}GB），"
                    "加载可能失败；失败请先关掉其它占内存的程序")
            else:
                logger.info(f"🧠 内存预检通过：可用 {avail_gb:.1f}GB / "
                            f"需要约 {want_gb:.1f}GB")
        except ImportError:
            logger.debug("没有 psutil，跳过内存预检")

        kwargs = {"torch_dtype": {"default": torch.bfloat16, "vae": torch.float32}}
        # DiT 量化：**MOSS-Image 用不了**（实测结论，见下面注释）。
        # 它的自定义 transformer 用 `weight.dtype` 推导计算精度
        # （pipeline_moss_image.py 第 120/141/239/240/241/248/256/258 行 +
        #  _TextBlock 第 110 行），把权重换成 int8/float8 之后，这些地方会把
        # **激活**也转成 int8 → `NotImplementedError: "silu_cuda" not
        # implemented for 'Char'`（实测：加载成功、出图必崩）。
        # 8bit 对 Flux / SD3 仍然有效（它们用 diffusers 标准 transformer）。
        quant_mode = str((ctx.optimization or {}).get("dit_quantization", "off")
                         or "off").lower()
        if quant_mode != "off":
            logger.warning(
                "⚠️ MOSS-Image 不支持 DiT 量化（%s），已按原精度 bf16 加载 —— "
                "它的自定义 transformer 用 weight.dtype 推计算精度，量化后激活会被"
                "当成 int8，出图直接报 silu_cuda 未实现。8bit 对 Flux / SD3 仍然有效。",
                quant_mode)
            quant_mode = "off"
        script = os.path.join(target, self.PIPELINE_FILE)
        if os.path.isfile(script):
            # 用目录里自带的自定义管线文件。
            # 注意：diffusers 只要看到目录里有自定义管线代码，就要求
            # trust_remote_code=True，哪怕传的是本地绝对路径 —— 这是它
            # 的安全门槛，不给这个标志会直接报 ValueError（实测）。
            # 这里执行的是**你自己下载到 models/ 里的那个文件**（Apache-2.0，
            # 官方仓库自带），不是从网上现拉的。
            kwargs["custom_pipeline"] = script
            kwargs["trust_remote_code"] = True
            logger.info(f"🧩 使用目录内的自定义管线: {script}")
            logger.info(
                "🔓 说明：MOSS-Image 的采样逻辑写在模型目录的 pipeline_moss_image.py 里，"
                "加载时会执行该文件（diffusers 要求 trust_remote_code=True）")
        else:
            logger.warning(
                f"⚠️ 目录里没有 {self.PIPELINE_FILE}，尝试按 model_index.json "
                "里的类名直接加载（若报 unknown pipeline 就是这个原因）")

        # ---- 8bit 常驻显存（可选，默认关闭）--------------------------------
        # 目的不是提速（实测每步反而慢 18%），而是**把 DiT 从内存里挪走**：
        # bf16 + 分块 offload 要让 9.5GB 权重常驻内存，开个 IDE 就爆；
        # int8 后权重只有 4.7GB 且住在显存里，内存直接省下约 9.5GB。
        accel_on = bool((ctx.optimization or {}).get("moss_accel", False))
        accel_mod = None
        if accel_on and quant_mode == "off":
            try:
                from core import moss_accel
            except Exception as e:
                logger.warning("⚠️ 8bit 常驻显存不可用（%s），按 bf16 运行", e)
                moss_accel = None
            if moss_accel is not None:
                ready, why_bnb = moss_accel.bnb8_available()
                mod = None
                if not ready:
                    logger.warning(
                        "⚠️ 8bit 常驻显存不可用：bitsandbytes 的 int8 内核在这台机器上"
                        "跑不起来（%s）—— 本次按 bf16 + 分块 offload 运行，"
                        "功能与画质不受影响，只是内存里要多放 9.5GB 权重", why_bnb)
                else:
                    mod, why = moss_accel.build_module(target)
                    if mod is None:
                        logger.warning(
                            "⚠️ 8bit 常驻显存已跳过：%s —— 本次按 bf16 + 分块 offload 运行",
                            why)
                if mod is not None:
                    try:
                        from core.dit_optimization import quantization_config

                        kwargs["quantization_config"] = quantization_config("bnb8")
                        accel_mod = mod
                        logger.info("🧮 8bit 常驻显存已启用 —— %s", moss_accel.describe())
                        logger.info(
                            "ℹ️ 这是实验特性：本机（bnb 0.49.2 + torch 2.11 + RTX 5060）"
                            "采样阶段可能报 addmm_cuda not implemented for 'Char'。"
                            "真报错就把 app_config.json 里的 moss_accel 改回 false，"
                            "其余功能不受影响。")
                    except Exception as e:
                        logger.warning("⚠️ 8bit 量化不可用（%s），按 bf16 运行", e)

        try:
            pipe = DiffusionPipeline.from_pretrained(target, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "❌ MOSS-Image 管线装配失败。\n"
                f"   目录: {target}（来源: {how}）\n"
                "   常见原因：\n"
                f"     · 目录里缺 {self.PIPELINE_FILE}（自定义管线文件）\n"
                "     · 组件不完整（transformer / vae / text_encoder）\n"
                "     · 该目录其实是 ComfyUI 单文件格式，不是 diffusers 目录\n"
                f"   原始错误: {type(e).__name__}: {e}") from e

        # 官方示例显式重建一次 scheduler（flow-matching，shift 由配置决定）
        try:
            pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        except Exception as e:
            logger.warning(f"⚠️ MOSS-Image scheduler 重建失败（忽略）: {e}")

        # 8bit 补丁必须在**采样前**绑到活的类上：量化后的权重是 int8，
        # 官方 forward 用 weight.dtype 推计算精度会把激活也变成 int8。
        if accel_mod is not None:
            try:
                from core import moss_accel

                patched = moss_accel.apply(pipe, accel_mod)
            except Exception as e:
                patched = 0
                logger.error("❌ 8bit 补丁应用失败: %s: %s", type(e).__name__, e)
            if patched:
                pipe._moss_accel = True
            else:
                # 权重已经是 int8 了，没打上补丁的话一采样就报 silu_cuda
                raise RuntimeError(
                    "❌ MOSS-Image 的 8bit 补丁没能绑定到模型类上。\n"
                    "   这通常意味着模型目录里的 pipeline_moss_image.py 与当前"
                    "支持的版本不一致。\n"
                    "   请到「设置 → 推理性能」关闭「MOSS-Image 8bit 常驻显存」后重试"
                    "（关掉后按 bf16 运行，能出图，但 DiT 会占 9.5GB 内存）。")

        # 文本编码器在 8GB 卡上只能留 CPU；官方 encode_prompt 会把输入搬到
        # 显卡，导致设备不匹配，这里装上设备自适应版本。
        _patch_prompt_encoding(pipe)

        # ---- int8 权重量化（纯 torch，不依赖 bnb 内核）-------------------
        # 与上面 bnb 那条互斥：那次量化由 diffusers 完成、需要补丁；这里是我们
        # 自己把 nn.Linear 的权重换成 int8，forward 里做 W8A8/反量化，
        # 模型里 `weight.dtype` 依旧读到 bf16，所以不用改模型代码。
        quant_key = str((ctx.optimization or {}).get("moss_quant", "off") or "off").lower()
        if quant_key in ("int8", "w8a8", "8bit", "int8te", "int8-te") and not getattr(pipe, "_moss_accel", False):
            try:
                from core import moss_int8

                ok, why = moss_int8.available()
                if not ok:
                    logger.warning("⚠️ int8 权重量化不可用（%s）—— 按 bf16 运行", why)
                else:
                    stats = moss_int8.quantize_module(pipe.transformer)
                    if stats.get("count"):
                        pipe._moss_int8_stats = stats
                        logger.info("🧮 %s", moss_int8.describe(stats))
                        if quant_key in ("int8te", "int8-te"):
                            te_stats = moss_int8.quantize_module(pipe.text_encoder)
                            if te_stats.get("count"):
                                pipe._moss_int8_te_stats = te_stats
                                logger.info("🧮 文本编码器 %s", moss_int8.describe(te_stats))
                            else:
                                logger.warning("⚠️ 文本编码器没有可量化的层 —— 仍按 bf16 放内存")
                    else:
                        logger.warning("⚠️ int8 权重量化没有命中任何层 —— 按 bf16 运行")
            except Exception as e:
                logger.warning("⚠️ int8 权重量化失败（%s）—— 按 bf16 运行", e)

        logger.info(f"✅ MOSS-Image 管线就绪: {type(pipe).__name__}")
        # 官方只有文生图；img2img / inpaint 如实留空
        return LoadResult(txt2img=pipe)

    def _apply_int8_vram(self, pipe, transformer, vae, te, dit_gb, te_gb, total_gb):
        """int8 权重量化模式下的组件摆放：DiT 整个住显卡，内存里不留副本。

        与 bnb 那条的区别是**不需要任何 int8 内核**（权重是自己量化的，forward
        里用 `torch._int_mm` 或逐层反量化），所以不会撞 `addmm_cuda` 那种报错。
        """
        import gc

        import torch

        int8_gb = dit_gb          # 量化后 element_size 已是 1，直接就是实际体积
        stats = getattr(pipe, "_moss_int8_stats", {}) or {}
        free_gb = 0.0
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                free, _total = torch.cuda.mem_get_info()
                free_gb = free / 1024 ** 3
            except Exception as e:
                logger.debug("读取显存失败（忽略）: %s", e)

        needed = int8_gb + 1.6    # 激活 + VAE 解码的余量
        if free_gb and free_gb < needed:
            raise RuntimeError(
                f"❌ 显存不够放 int8 的 DiT。\n"
                f"   需要约 {needed:.2f}GB（DiT {int8_gb:.2f}GB + 激活余量），"
                f"当前可用 {free_gb:.2f}GB。\n"
                "   关掉别的占显存的程序，或在设置里把「MOSS-Image 量化」改回关闭。")

        if transformer is not None:
            already = 0.0
            try:
                already = sum(p.numel() * p.element_size()
                              for p in transformer.parameters()
                              if p.device.type == "cuda") / 1024 ** 3
            except Exception as e:
                logger.debug("统计已驻留显存失败（忽略）: %s", e)
            if already < int8_gb * 0.9:
                transformer.to("cuda:0")
        if vae is not None:
            try:
                vae.to("cuda:0")
                for fn_name in ("enable_tiling", "enable_slicing"):
                    fn = getattr(vae, fn_name, None)
                    if callable(fn):
                        fn()
            except Exception as e:
                logger.warning("⚠️ VAE 上卡失败（%s），解码会更慢", e)
        if te is not None:
            try:
                te.to("cpu")
            except Exception as e:
                logger.debug("文本编码器移回 CPU 失败（忽略）: %s", e)

        gc.collect()
        # 量化过程会把 25GB 权重文件整份读一遍，这些页会留在进程工作集里
        # （实测 RSS 18.6GB）。DiT 已经搬到显卡了，这些页用不上，主动交还给系统，
        # 否则"省内存"这件事在任务管理器里看不出来。
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.EmptyWorkingSet(ctypes.c_void_p(handle))
        except Exception as e:
            logger.debug("释放工作集失败（忽略）: %s", e)
        try:
            torch.cuda.empty_cache()
            free, _total = torch.cuda.mem_get_info()
            free_gb = free / 1024 ** 3
        except Exception as e:
            logger.debug("读取显存失败（忽略）: %s", e)
        try:
            import psutil

            rss_gb = psutil.Process().memory_info().rss / 1024 ** 3
            logger.info(
                "🧮 int8 常驻显存：DiT %.2fGB 已在显卡上（%s 层），"
                "进程内存 %.2fGB（bf16 + offload 时要多占约 %.1fGB）",
                int8_gb, stats.get("count", "?"), rss_gb, dit_gb / 2)
        except Exception as e:
            logger.debug("读取内存失败（忽略）: %s", e)
        logger.info("   显存余量 %.2fGB", free_gb)
        if 0 < free_gb < 0.9:
            logger.warning(
                "⚠️ 显存只剩 %.2fGB，1024² 解码可能不够；建议先用 768²。", free_gb)

    def _apply_accel_vram(self, pipe, transformer, vae, te, dit_gb, te_gb, total_gb):
        """8bit 常驻显存模式下的组件摆放。

        与分块 offload 相反：DiT **整个**搬上显卡并且**不再留内存副本**，
        于是 bf16 时代那 9.5GB 内存占用消失（实测加载后 RSS 从 18GB 降到
        约 8.5GB）。代价是每步慢约 18%、显存顶到 7.8/7.9GB。
        """
        import gc

        import torch

        # 注意：量化后的参数**已经是 int8**（element_size=1），所以这里的
        # dit_gb 就是 int8 实际体积，不能再乘缩放系数（早期版本乘了 0.51，
        # 结果把"需要 4.7GB"算成 2.4GB，误判成装得下）。
        int8_gb = dit_gb
        free_gb = 0.0
        if torch.cuda.is_available():
            try:
                # 加载时 bnb 会在显存里留一块缓存池，先回收再判断余量，
                # 否则会把"可回收的缓存"当成"被别人占用"，白白拒绝。
                torch.cuda.empty_cache()
                free, _total = torch.cuda.mem_get_info()
                free_gb = free / 1024 ** 3
            except Exception as e:
                logger.debug("读取显存失败（忽略）: %s", e)

        if transformer is not None:
            # bnb 量化在加载阶段就可能已经把权重放到显卡上了，所以只把
            # "还不在显存里的那部分"算成需要新增的空间，否则会误报不够。
            already_gb = 0.0
            try:
                already_gb = sum(
                    p.numel() * p.element_size()
                    for p in transformer.parameters() if p.device.type == "cuda"
                ) / 1024 ** 3
            except Exception as e:
                logger.debug("统计已驻留显存失败（忽略）: %s", e)
            needed = max(0.0, int8_gb - already_gb) + 1.6   # + 1024² 激活余量
            if free_gb and free_gb < needed:
                raise RuntimeError(
                    "❌ 显存不够把 DiT 常驻：还需要约 "
                    f"{needed:.1f}GB（8bit 权重 {int8_gb:.1f}GB 已有 "
                    f"{already_gb:.1f}GB 在卡上 + 激活），当前只有 {free_gb:.1f}GB。\n"
                    "   请先关掉别的显卡占用（浏览器/游戏/其它 AI 工具），或在"
                    "「设置 → 推理性能」里关掉「MOSS-Image 8bit 常驻显存」"
                    "（关掉后走分块 offload，更省显存但要多占约 9.5GB 内存）。")
            try:
                transformer.to("cuda:0")
                logger.info(
                    "📦 8bit DiT 已常驻显存：%.2fGB（bf16 时约 %.2fGB）—— "
                    "内存里不再保留权重副本", int8_gb, int8_gb / 0.51)
            except Exception as e:
                raise RuntimeError(
                    "❌ 把 DiT 搬到显卡失败（8bit 常驻显存模式）："
                    f"{type(e).__name__}: {e}") from e

        if vae is not None:
            try:
                vae.to("cuda:0")
                for fn_name in ("enable_tiling", "enable_slicing"):
                    fn = getattr(vae, fn_name, None)
                    if callable(fn):
                        fn()
                logger.info("🖼️ VAE 已上卡（%.0fMB，tiling/slicing 已开）",
                            self._module_gb(vae) * 1024)
            except Exception as e:
                logger.warning("⚠️ VAE 上卡失败: %s", e)

        if te is not None:
            logger.info("🧠 文本编码器 %.1fGB 留在 CPU（编码完只把向量搬到显卡）", te_gb)

        # 释放内存副本：量化模块 move 之后 CPU 侧只剩空壳，
        # 但 diffusers 可能还挂着一些引用，主动回收一次更稳。
        gc.collect()
        try:
            import psutil

            avail = psutil.virtual_memory().available / 1024 ** 3
            logger.info(
                "🧠 8bit 常驻显存带来的内存收益：DiT 不再占 %.1fGB 内存"
                "（当前可用内存 %.1fGB）", dit_gb, avail)
        except Exception as e:
            logger.debug("读取内存失败（忽略）: %s", e)

        try:
            import torch as _torch

            if _torch.cuda.is_available():
                used = _torch.cuda.memory_allocated() / 1024 ** 3
                free, total = _torch.cuda.mem_get_info()
                logger.info("📊 显存：已分配 %.1fGB，空闲 %.0fMB / 共 %.0fMB",
                            used, free / 1024 ** 2, total / 1024 ** 2)
                if free < 900 * 1024 * 1024:
                    logger.warning(
                        "⚠️ 显存只剩 %.0fMB：8bit 常驻模式在 1024² 上很紧，"
                        "建议先跑 768²；实在不够就关掉这个选项",
                        free / 1024 ** 2)
        except Exception as e:
            logger.debug("显存报告失败（忽略）: %s", e)

    # --------------------------------------------------------------- 显存策略
    @staticmethod
    def _module_gb(module) -> float:
        """一个 nn.Module 的权重体积（GB）。"""
        if module is None:
            return 0.0
        try:
            total = 0
            for p in module.parameters():
                total += p.numel() * p.element_size()
            return total / 1024 ** 3
        except Exception:
            return 0.0

    def apply_vram(self, result, ctx):
        """MOSS-Image 专用显存策略。

        通用 `apply_optimal_strategy` 会调 `enable_model_cpu_offload()`，而那个
        函数会把**所有组件**（含 16GB 文本编码器）轮流搬上显卡 —— 8GB 卡上
        必然 OOM。这里按组件分开处理：

        * DiT（9.69GB）> 8GB 显存 → 只有 block 级 offload 走得通；
        * 文本编码器（16GB）永远留 CPU，只把编码结果搬到卡上（管线自己会做）；
        * VAE（250MB fp32）上卡 —— 解码必须和 latents 同设备，并开 tiling。
        """
        import torch
        from utils.vram_manager import VRAMManager

        pipe = result.txt2img
        if pipe is None:
            return

        transformer = getattr(pipe, "transformer", None)
        vae = getattr(pipe, "vae", None)
        te = getattr(pipe, "text_encoder", None)
        dit_gb = self._module_gb(transformer)
        te_gb = self._module_gb(te)
        _options = getattr(ctx, "optimization", None) or {}

        # 注意：build() 里已经强制把 quant_mode 归零（MOSS-Image 不支持量化，
        # 见那里的注释），所以这里按原始 bf16 体积估算即可 —— 别再拿
        # quantized_weight_ratio 去缩，否则会误判成"装得下"而 OOM。
        dit_eff_gb = dit_gb

        total_gb = 0.0
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        # 显存够装下 DiT + 余量就整体上卡（24GB 以上的卡会走这条）
        full_fits = bool(dit_eff_gb and total_gb and total_gb >= dit_eff_gb + 3.5)

        # ---- 8bit 常驻显存：DiT 整个住显卡，把 9.5GB 内存让给系统 --------
        if getattr(pipe, "_moss_accel", False):
            self._apply_accel_vram(pipe, transformer, vae, te, dit_gb, te_gb, total_gb)
            VRAMManager.print_status()
            return

        # ---- int8 权重量化（纯 torch）：DiT 4.73GB 常驻显存 ----------------
        # 与上面 bnb 那条的区别：这里权重量化是自己做的（int8 + 逐层/激活反量化），
        # 不依赖 bitsandbytes 的 int8 内核，所以不会撞上 weight.dtype 推精度的问题。
        if getattr(pipe, "_moss_int8_stats", None):
            self._apply_int8_vram(pipe, transformer, vae, te, dit_gb, te_gb, total_gb)
            VRAMManager.print_status()
            return

        if full_fits:
            try:
                pipe.to("cuda:0")
                logger.info("📦 MOSS-Image 全量上卡：DiT %.2fGB，显卡 %.1fGB",
                            dit_eff_gb, total_gb)
            except Exception as e:
                logger.warning("⚠️ MOSS-Image 全量上卡失败（%s），改用分块 offload", e)
                full_fits = False

        if not full_fits:
            # 1) 先确保都在 CPU 上（from_pretrained 默认就是，别多搬一次）
            try:
                if any(p.device.type != "cpu" for p in transformer.parameters()):
                    pipe.to("cpu")
            except Exception as e:
                logger.debug("MOSS-Image 全部移回 CPU 失败（忽略）: %s", e)

            # 2) VAE 上卡（解码要和 latents 同设备），并开 tiling/slicing
            if vae is not None:
                try:
                    vae.to("cuda:0")
                    for fn_name in ("enable_tiling", "enable_slicing"):
                        fn = getattr(vae, fn_name, None)
                        if callable(fn):
                            fn()
                    logger.info("🖼️ VAE 已上卡（%.0fMB，tiling/slicing 已开）",
                                self._module_gb(vae) * 1024)
                except Exception as e:
                    logger.warning("⚠️ VAE 上卡失败: %s", e)

            # 3) DiT 分块 offload（每块约 231MB）
            if transformer is not None:
                from core.dit_optimization import apply_group_offload

                try:
                    blocks = int(_options.get("dit_offload_blocks", 1) or 1)
                except Exception:
                    blocks = 1
                blocks = max(1, min(8, blocks))
                try:
                    apply_group_offload(pipe, True, blocks, "cuda:0")
                    logger.info(
                        "🧩 DiT 分块 offload 已启用：%.2fGB / 42 层，每组 %d 块，"
                        "显卡 %.1fGB —— 每步都要把权重从内存搬一次（这是 8GB 卡跑 "
                        "MOSS-Image 的常态：DiT 9.5GB > 整卡 7.9GB，量化也救不了，"
                        "因为它的自定义 transformer 用 weight.dtype 推计算精度）。"
                        "想更快请降分辨率：768² 比 1024² 快一倍多",
                        dit_gb, blocks, total_gb)
                except Exception as e:
                    logger.error("❌ MOSS-Image DiT 分块 offload 失败: %s", e)
                    raise

            # 4) 文本编码器留 CPU：它比整张卡还大
            if te is not None:
                logger.info(
                    "🧠 文本编码器 %.1fGB 留在 CPU（24 线程编码，"
                    "只把 embedding 搬到显卡）—— 上卡必然 OOM", te_gb)

        VRAMManager.print_status()

        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.info("📊 MOSS-Image 显存: 空闲 %.0fMB / 共 %.0fMB",
                            free / 1024 ** 2, total / 1024 ** 2)
                if free < 1000 * 1024 * 1024:
                    logger.warning(
                        "⚠️ 剩余显存很少：MOSS-Image 是 5B DiT，"
                        "建议 1024×1024 起步；再大请先关掉别的显卡占用")
        except Exception as e:
            logger.debug(f"MOSS-Image 显存报告失败（忽略）: {e}")

        # 内存提示：DiT + 文本编码器同时驻留需要约 26GB
        if dit_gb and te_gb:
            need = dit_gb + te_gb + 4.0
            try:
                import psutil
                have = psutil.virtual_memory().total / 1024 ** 3
                if have < need:
                    logger.warning(
                        "⚠️ 内存可能不够：DiT %.1fGB + 文本编码器 %.1fGB "
                        "约需 %.0fGB，本机只有 %.0fGB；不够时系统会换页，会非常慢",
                        dit_gb, te_gb, need, have)
            except Exception as e:
                logger.debug("内存检查跳过: %s", e)
