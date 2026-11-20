# utils/hf_compat.py
"""依赖兼容补丁：老组件 ↔ 新版 HuggingFace 生态。

目前只有一个问题
----------------
`huggingface_hub` 1.0 移除了 `HfFolder`，而 GPT-SoVITS 自带的旧版 gradio
（本机 4.44.1）在 `gradio/oauth.py:13` 仍写：

    from huggingface_hub import HfFolder, whoami

于是 `import GPT_SoVITS.inference_webui` 抛

    ImportError: cannot import name 'HfFolder' from 'huggingface_hub'

报错点在 gradio，但受害者是「GPT-SoVITS 日语」TTS —— 整条链路直接不可用，
而且看着像"TTS 坏了"，排查成本高。

为什么不降级 huggingface_hub：transformers 5.x / diffusers 0.40 都依赖 1.x，
降级会连带打断出图主链路。所以这里只补一个最小实现（只读写 token 文件），
`import gradio` 就能过。

用法
----
需要 import gradio / GPT-SoVITS 的入口，在**导入前**调用一次：

    from utils.hf_compat import ensure_hf_folder
    ensure_hf_folder()
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def ensure_hf_folder() -> bool:
    """补回 `huggingface_hub.HfFolder`。已存在则什么也不做。

    返回 True 表示现在这个属性可用（本来就有的也算），False 表示补不上。
    """
    global _PATCHED

    try:
        from huggingface_hub import HfFolder  # noqa: F401
        _PATCHED = True
        return True
    except ImportError:
        pass

    try:
        import huggingface_hub as _hf
    except ImportError:
        logger.warning("未安装 huggingface_hub，跳过 HfFolder 兼容补丁")
        return False

    class HfFolder:  # noqa: D401  (最小替身，只实现旧版被用到的三个方法)
        """huggingface_hub < 1.0 的 HfFolder 替身。"""

        path = os.path.join(os.path.expanduser("~"), ".cache",
                            "huggingface", "token")

        @classmethod
        def get_token(cls):
            try:
                with open(cls.path, "r", encoding="utf-8") as f:
                    return f.read().strip() or None
            except OSError:
                return None

        @classmethod
        def save_token(cls, token):
            os.makedirs(os.path.dirname(cls.path), exist_ok=True)
            with open(cls.path, "w", encoding="utf-8") as f:
                f.write(str(token))

        @classmethod
        def delete_token(cls):
            try:
                os.remove(cls.path)
            except OSError:
                pass

    _hf.HfFolder = HfFolder
    _PATCHED = True
    logger.info("已为旧版 gradio 补回 huggingface_hub.HfFolder 兼容属性")
    return True
