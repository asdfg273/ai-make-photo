# core/loaders/anima.py
import logging
import os
import re
import torch

from .base import BaseLoader, LoadResult

logger = logging.getLogger(__name__)


class AnimaLoader(BaseLoader):
    """Anima（Cosmos DiT 2B）加载器。

    ⚠️ 权重 key 命名映射（这是本加载器的核心）：
    社区分发的 Anima 微调多为 **ComfyUI / 原始训练格式**，而
    diffusers 的 `CosmosTransformer3DModel` 用另一套命名。实测对照：

        检查点（提交格式）                       基座（diffusers）
        blocks.{i}.self_attn.q_proj            transformer_blocks.{i}.attn1.to_q
        blocks.{i}.self_attn.k_proj            transformer_blocks.{i}.attn1.to_k
        blocks.{i}.self_attn.v_proj            transformer_blocks.{i}.attn1.to_v
        blocks.{i}.self_attn.output_proj       transformer_blocks.{i}.attn1.to_out.0
        blocks.{i}.self_attn.q_norm/k_norm     transformer_blocks.{i}.attn1.norm_q/k
        blocks.{i}.cross_attn.q_proj           transformer_blocks.{i}.attn2.to_q
        blocks.{i}.cross_attn.k_proj           transformer_blocks.{i}.attn2.to_k
        blocks.{i}.cross_attn.v_proj           transformer_blocks.{i}.attn2.to_v
        blocks.{i}.cross_attn.output_proj      transformer_blocks.{i}.attn2.to_out.0
        blocks.{i}.cross_attn.q_norm/k_norm    transformer_blocks.{i}.attn2.norm_q/k
        blocks.{i}.mlp.layer1                  transformer_blocks.{i}.ff.net.0.proj
        blocks.{i}.mlp.layer2                  transformer_blocks.{i}.ff.net.2
        blocks.{i}.adaln_modulation_self_attn  transformer_blocks.{i}.norm1
        blocks.{i}.adaln_modulation_cross_attn transformer_blocks.{i}.norm2
        blocks.{i}.adaln_modulation_mlp        transformer_blocks.{i}.norm3
        x_embedder.proj.1                      patch_embed.proj
        t_embedder.1.*                         time_embed.t_embedder.*
        t_embedding_norm                       time_embed.norm
        final_layer.adaln_modulation.*         norm_out.*
        final_layer.linear                     proj_out
        llm_adapter.blocks.*                   → **属于 text_conditioner**，去掉前缀

    ⚠️ 旧实现只做了一次 `model.diffusion_model.` 前缀剥离就直接
    `load_state_dict(strict=False)`，实测 **0/685 形状匹配** ——
    等于一个权重都没加载，Anima 实际用的是裸基座模型，且只有一条
    warning，用户完全看不出。
    """

    arch_ids = ("anima",)
    BASE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
    PREFIX = "model.diffusion_model."

    # ── 基座组件（Qwen3-0.6B 文本编码器 / LLM adapter / Qwen VAE + 调度器）──
    # 社区分发的 Anima 检查点**只有 transformer + llm_adapter 权重**，
    # 其余组件必须来自基座仓。所以加载前先找一个**本地**基座目录：
    # 本地目录优先，其次 HuggingFace 缓存快照；都没有就明确报错。
    #
    # ⚠️ 曾经直接用 repo id 调 `ModularPipeline.from_pretrained(BASE_REPO)`：
    #    即使本地已有缓存、或用户手动放好了基座，也照样发网络请求；离线时
    #    getaddrinfo 失败 → 重试 5 次 → 抛出与真实原因无关的
    #    "Cannot send a request, as the client has been closed."，
    #    用户完全看不出是"缺基座组件"。
    #    现在解析出**本地路径**再交给 diffusers，全程不碰网络。
    _BASE_SEARCH = ("anima", "puton")   # models/<x>/ 下的候选目录
    # 已发布的模块化基座仓里是 modular_model_index.json；
    # modular_config.json 是"代码即配置"的导出格式（Anima 官方仓里没有这个
    # 文件 —— HF 缓存的 .no_exist 记录里能查到它 404）。两个都当入口标记，
    # 但必须非空。
    _BASE_MARKERS = ("modular_model_index.json", "modular_config.json",
                     "model_index.json")
    # 这几个组件必须真有权重：检查点里只有 transformer + llm_adapter，
    # 文本编码器 / VAE 只能来自基座。
    _REQUIRED_PARTS = ("transformer", "text_encoder", "vae")

    @classmethod
    def _looks_like_base(cls, path: str) -> bool:
        """目录里有没有模块化/普通 diffusers 管线的入口文件。"""
        return any(os.path.isfile(os.path.join(path, m))
                   for m in cls._BASE_MARKERS)

    @classmethod
    def _base_problem(cls, path: str) -> str:
        """检查基座目录能否直接用。可用返回 ""，否则返回中文原因。

        ⚠️ 必须校验体积：HF 缓存里"下到一半"的快照会留下入口文件，
        但组件权重是 0 字节 —— 直接交给 diffusers 会抛 JSON 解析错误，
        用户同样看不出真实原因。
        ⚠️ 缓存里的文件是**符号链接**：必须用 os.path.getsize（会跟到
        真实 blob），PowerShell 的 Length 对符号链接显示 0，不能用来判空。
        """
        if not os.path.isdir(path):
            return "目录不存在"
        marker = ""
        for name in cls._BASE_MARKERS:
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                marker = fp
                break
        if not marker:
            return "缺 " + " / ".join(cls._BASE_MARKERS)
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

    def local_base_dirs(self) -> list[str]:
        """用户手动放置的基座目录（不联网、不下载）。

        约定：把基座目录（`Anima-Base-v1.0-Diffusers` 或改名后的）放进
        `models/anima/` 或 `models/puton/` 即可 —— 目录名不限，只要里面有
        modular_model_index.json / modular_config.json。投递箱（puton）
        由 model_inbox 自动归位到 models/anima/。
        """
        from utils import paths

        found: list[str] = []
        for sub in self._BASE_SEARCH:
            root = os.path.join(paths.MODEL_DIR, sub)
            if not os.path.isdir(root):
                continue
            if self._looks_like_base(root):
                found.append(root)
            try:
                names = sorted(os.listdir(root))
            except OSError as e:
                logger.debug(f"读取 {root} 失败（忽略）: {e}")
                continue
            for name in names:
                cand = os.path.join(root, name)
                try:
                    if os.path.isdir(cand) and self._looks_like_base(cand):
                        found.append(cand)
                except OSError:                     # 权限/长度问题，跳过
                    continue
        return found

    def cached_base_dirs(self) -> list[str]:
        """已下载到 HuggingFace 缓存里的基座快照目录（可能不完整）。"""
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:                      # 没装 hub 也不该崩
            logger.debug(f"huggingface_hub 不可用（忽略缓存探测）: {e}")
            return []
        try:
            # local_files_only=True：只查缓存，**绝不发网络请求**
            path = snapshot_download(self.BASE_REPO, local_files_only=True)
        except Exception as e:
            logger.debug(f"基座 {self.BASE_REPO} 不在本地缓存: {e}")
            return []
        return [path] if path and self._looks_like_base(path) else []

    def _missing_base_message(self, problems: list[str] | None = None) -> str:
        from utils import paths

        want = os.path.join(paths.MODEL_DIR, "anima",
                            self.BASE_REPO.split("/")[-1])
        comps = "Qwen3-0.6B 文本编码器 / LLM adapter / Qwen VAE + 调度器配置"
        lines = [
            "❌ Anima 缺少可用的「基座组件」，所以无法出图。",
            f"   需要: {self.BASE_REPO}（{comps}，约 1.7GB）",
            "   只放一个 .safetensors 检查点是不够的 —— 里面只有 transformer 和",
            "   llm_adapter 权重，文本编码器与 VAE 在基座里。",
        ]
        if problems:
            lines.append("")
            lines.append("   已找到但不可用的位置：")
            for p in problems[:4]:
                lines.append(f"     · {p}")
        lines += [
            "",
            "   请自行准备（本程序不提供下载入口，也不会在生成时自动联网）：",
            "     1) 以前下载过这个仓：留在 HuggingFace 缓存里就行，程序直接复用；",
            "        若缓存是「下到一半」的（下面报了 0 字节），删掉该缓存目录重新准备；",
            "     2) 或者把整个基座目录放进下面任意一个位置（目录名不限，",
            "        里面要有 modular_model_index.json）：",
            f"          {want}",
            f"          {os.path.join(paths.MODEL_DIR, 'puton')}\\<基座目录>",
            "        放好后重新加载模型即可。",
            "",
            "   ⚠️ 离线时加载会在这一步直接停下（不会再联网重试卡住）。",
        ]
        return "\n".join(lines)

    def resolve_base(self) -> tuple[str, str]:
        """返回 (本地基座目录, 来源说明)；找不到就抛 RuntimeError。"""
        problems: list[str] = []
        for src, how in [(d, "本地目录") for d in self.local_base_dirs()] + \
                        [(d, "HF 缓存") for d in self.cached_base_dirs()]:
            why = self._base_problem(src)
            if not why:
                return src, how
            problems.append(f"{src}（{how}）→ {why}")
            logger.warning(f"⚠️ Anima 基座不可用: {src} → {why}")
        raise RuntimeError(self._missing_base_message(problems))

    # 块内子模块重命名（精确匹配后缀；索引型另由 _BLOCK_RENAMES_IDX 处理）
    _BLOCK_RENAMES = {
        "self_attn.q_norm":     "attn1.norm_q",
        "self_attn.k_norm":     "attn1.norm_k",
        "self_attn.q_proj":     "attn1.to_q",
        "self_attn.k_proj":     "attn1.to_k",
        "self_attn.v_proj":     "attn1.to_v",
        "self_attn.output_proj": "attn1.to_out.0",
        "cross_attn.q_norm":    "attn2.norm_q",
        "cross_attn.k_norm":    "attn2.norm_k",
        "cross_attn.q_proj":    "attn2.to_q",
        "cross_attn.k_proj":    "attn2.to_k",
        "cross_attn.v_proj":    "attn2.to_v",
        "cross_attn.output_proj": "attn2.to_out.0",
        "mlp.layer1":           "ff.net.0.proj",
        "mlp.layer2":           "ff.net.2",
    }

    # adaln 等「Sequential 索引 → 命名层」的映射
    #   检查点 adaln_modulation_*.1 / .2  →  基座 norm*.linear_1 / linear_2
    #   检查点 final_layer.adaln_modulation.1 / .2 → 基座 norm_out.linear_1 / linear_2
    # ⚠️ 去掉前缀后的尾巴是 `1.weight`（**没有前导点**），所以不能用
    #    `".1." -> ".linear_1."` 这种 replace；必须按「段边界」匹配。
    _ADALN_IDX_RE = re.compile(r"(?:^|\.)([12])\.")

    @classmethod
    def _map_adaln_tail(cls, tail: str) -> str:
        """`1.weight` -> `linear_1.weight`；`0.1.weight` -> `0.linear_1.weight`。"""
        return cls._ADALN_IDX_RE.sub(
            lambda m: ("" if m.start() == 0 else ".") + f"linear_{m.group(1)}.", tail)

    _ADALN_PREFIXES = (
        ("adaln_modulation_cross_attn.", "norm2."),
        ("adaln_modulation_self_attn.", "norm1."),
        ("adaln_modulation_mlp.", "norm3."),
        ("final_layer.adaln_modulation.", "norm_out."),
    )

    @classmethod
    def _map_transformer_key(cls, k: str):
        """把检查点里的 transformer key 映射到 diffusers 命名。失败返回 None。"""
        if k.startswith("blocks."):
            # blocks.{i}.rest  →  transformer_blocks.{i}.{rest 映射后}
            parts = k.split(".", 2)
            if len(parts) < 3:
                return None
            idx, rest = parts[1], parts[2]
            # 1) adaln 索引映射（先做，避免被下面的精确表漏掉）
            for src, dst in cls._ADALN_PREFIXES:
                if rest.startswith(src):
                    tail = rest[len(src):]
                    return f"transformer_blocks.{idx}.{dst}{cls._map_adaln_tail(tail)}"
            # 2) 精确后缀映射
            for src, dst in cls._BLOCK_RENAMES.items():
                if rest == src or rest.startswith(src + "."):
                    return f"transformer_blocks.{idx}.{dst}{rest[len(src):]}"
            return None

        # 顶层
        if k.startswith("x_embedder.proj.1"):
            return "patch_embed.proj" + k[len("x_embedder.proj.1"):]
        if k.startswith("t_embedder.1."):
            return "time_embed.t_embedder." + k[len("t_embedder.1."):]
        if k == "t_embedding_norm.weight":
            return "time_embed.norm.weight"
        if k.startswith("final_layer.adaln_modulation."):
            tail = k[len("final_layer.adaln_modulation."):]
            return "norm_out." + cls._map_adaln_tail(tail)
        if k == "final_layer.linear.weight":
            return "proj_out.weight"
        return None

    def build(self, model_path, ctx):
        from diffusers import ModularPipeline
        from safetensors.torch import load_file

        # 先解析基座（本地目录 / 缓存快照），绝不拿 repo id 去联网
        base_dir, how = self.resolve_base()
        logger.info(f"⏳ 装配 Anima 组件（{how}）: {base_dir}")
        try:
            pipe = ModularPipeline.from_pretrained(base_dir)
            pipe.load_components(dtype=torch.bfloat16)
        except Exception as e:
            raise RuntimeError(
                "❌ Anima 基座组件加载失败。\n"
                f"   目录: {base_dir}（来源: {how}）\n"
                "   常见原因：目录不完整（少了 text_encoder / vae 等子目录）、\n"
                "   或与当前 diffusers 版本不兼容。\n"
                f"   原始错误: {type(e).__name__}: {e}") from e
        pipe.to("cuda")

        logger.info(f"🔄 替换 Anima 权重: {ctx.model_name}")
        raw = load_file(model_path)
        sd = {k[len(self.PREFIX):]: v for k, v in raw.items()
              if k.startswith(self.PREFIX)}
        if not sd:
            # 兼容不带该前缀的检查点
            sd = dict(raw)
            logger.info(f"ℹ️ 检查点无 {self.PREFIX!r} 前缀，按原键名处理")
        logger.info(f"🔧 剥离前缀后共 {len(sd)} keys")

        # ── 分离三部分：transformer / text_conditioner / 未识别 ──
        tr_map, tc_map, unknown = {}, {}, []
        for k, v in sd.items():
            if k.startswith("llm_adapter."):
                tc_map[k[len("llm_adapter."):]] = v
                continue
            mapped = self._map_transformer_key(k)
            if mapped is None:
                unknown.append(k)
                continue
            tr_map[mapped] = v

        # ── 校验（对齐后才算数）──
        def _report(label, mod, mapping):
            if mod is None:
                logger.warning(f"⚠️ {label} 组件不存在，跳过")
                return 0
            exp = set(mod.state_dict().keys())
            got = set(mapping)
            missing = exp - got
            unexpected = got - exp
            logger.info(f"📊 [{label}] 期望 {len(exp)} / 映射后 {len(got)} | "
                        f"missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                logger.warning(f"⚠️ [{label}] missing 样例: {sorted(missing)[:5]}")
            if unexpected:
                logger.warning(f"⚠️ [{label}] unexpected 样例: {sorted(unexpected)[:5]}")
            mod.load_state_dict(mapping, strict=False)
            return len(exp) - len(missing)

        n_tr = _report("transformer", getattr(pipe, "transformer", None), tr_map)
        n_tc = _report("text_conditioner", getattr(pipe, "text_conditioner", None),
                       tc_map)

        if unknown:
            logger.warning(f"⚠️ {len(unknown)} 个键未识别（未加载）: {unknown[:5]}")

        # 明确结论：加载失败必须让用户看得见
        if n_tr == 0:
            raise RuntimeError(
                "❌ Anima 权重一个都没加载成功（key 命名不匹配）。\n"
                "   该检查点可能不是 Cosmos DiT / Anima 格式，或命名约定不同。\n"
                f"   请检查文件是否与基座 {self.BASE_REPO} 对应。")
        logger.info(f"✅ Anima 权重加载完成：transformer {n_tr} 键"
                    f" + text_conditioner {n_tc} 键")

        # Anima 是 ModularPipeline：同一个对象按**传入的参数**自动选择 block
        # （给 image= 就走 vae_encoder + denoise，即图生图）。所以 txt2img 与
        # img2img 是同一个管线实例，不能像 SD15/SDXL 那样建两个对象。
        # 实测 blocks: text_encoder / vae_encoder / denoise / decode。
        # 注意 img2img 必须显式传 height/width（块尺寸），否则会按默认 1024
        # 出图，等于每块都在跑一张 1024 的整图。
        return LoadResult(txt2img=pipe, img2img=pipe)

    def apply_vram(self, result, ctx):
        pipe = result.txt2img
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            logger.info("✅ Anima: VAE tiling+slicing 已启用")
        except Exception as e:
            logger.warning(f"⚠️ Anima VAE 优化失败: {e}")

        # ModularPipeline 没有 enable_model_cpu_offload / sequential offload
        # （MRO 是 ConfigMixin，不是 DiffusionPipeline），所以显存只能靠
        # 「全 GPU 常驻 + 降分辨率」。这里如实告知余量，别让用户以为
        # 有 offload 兜底（实测 512²: 权重 5.6GB、峰值 5.7GB/8.1GB）。
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.info("📊 Anima 显存: 空闲 %.0fMB / 共 %.0fMB"
                            "（该架构不支持 CPU offload）",
                            free / 1024 ** 2, total / 1024 ** 2)
                if free < 1500 * 1024 * 1024:
                    logger.warning("⚠️ Anima 剩余显存偏少，768×768 以上"
                                   "可能需要降低分辨率或关掉别的显卡占用")
        except Exception as e:
            logger.debug(f"Anima 显存报告失败（忽略）: {e}")
