# core/loaders/anima.py
import logging
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

        logger.info(f"⏳ 装配 Anima 组件: {self.BASE_REPO}")
        pipe = ModularPipeline.from_pretrained(self.BASE_REPO)
        pipe.load_components(dtype=torch.bfloat16)
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

        return LoadResult(txt2img=pipe)

    def apply_vram(self, result, ctx):
        pipe = result.txt2img
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            logger.info("✅ Anima: VAE tiling+slicing 已启用")
        except Exception as e:
            logger.warning(f"⚠️ Anima VAE 优化失败: {e}")
