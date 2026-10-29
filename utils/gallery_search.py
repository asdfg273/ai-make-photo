"""画廊全文匹配与本地 CLIP 相似检索；不自动下载模型。"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile

logger = logging.getLogger(__name__)


def match_all_terms(query: str, filename: str, prompt: str) -> bool:
    terms = re.findall(r"\w+", (query or "").casefold(), flags=re.UNICODE)
    haystack = f"{filename or ''}\n{prompt or ''}".casefold()
    return all(term in haystack for term in terms)


def _normalized(vector):
    values = [float(value) for value in vector]
    length = math.sqrt(sum(value * value for value in values))
    if length <= 1e-12:
        return values
    return [value / length for value in values]


def cosine_rank(query_vector, vectors: dict) -> dict:
    query = _normalized(query_vector)
    scores = {}
    for path, vector in vectors.items():
        other = _normalized(vector)
        size = min(len(query), len(other))
        scores[path] = (sum(query[i] * other[i] for i in range(size))
                        if size and len(query) == len(other) else 0.0)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


class _LocalClipEncoder:
    def __init__(self, model_path: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        explicit = model_path or os.environ.get("AI_STUDIO_CLIP_MODEL", "")
        if explicit and not os.path.isdir(explicit):
            raise RuntimeError(f"本地 CLIP 模型目录不存在: {explicit}")
        if not explicit:
            from utils import paths
            for name in ("clip-vit-base-patch32", "ViT-B-32"):
                candidate = os.path.join(paths.MODEL_DIR, "clip", name)
                if os.path.isdir(candidate):
                    explicit = candidate
                    break
        source = explicit or "openai/clip-vit-base-patch32"
        try:
            self.processor = CLIPProcessor.from_pretrained(source, local_files_only=True)
            self.model = CLIPModel.from_pretrained(source, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "无法从本地加载 CLIP。请把模型放到 models/clip/clip-vit-base-patch32/"
                " 或在设置中填写本地目录；也可以先放入 Hugging Face 缓存。"
                f"原始错误: {exc}") from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self._torch = torch

    def _finish(self, features):
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features[0].detach().cpu().tolist()

    def encode_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return self._finish(features)

    def encode_image(self, path: str):
        from PIL import Image
        with Image.open(path) as source:
            image = source.convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return self._finish(features)


class ClipSearchEngine:
    def __init__(self, model_path: str | None = None, index_path: str | None = None,
                 encoder=None):
        self.model_path = model_path
        if index_path is None:
            from utils import paths
            index_path = os.path.join(paths.DATA_DIR, "gallery_clip_index.json")
        self.index_path = index_path
        self.encoder = encoder

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = _LocalClipEncoder(self.model_path)
        return self.encoder

    def encode_text(self, text: str):
        return self._get_encoder().encode_text(text)

    def encode_image(self, path: str):
        return self._get_encoder().encode_image(path)

    @staticmethod
    def _signature(path: str) -> dict:
        stat = os.stat(path)
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _load_index(self) -> dict:
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_index(self, data: dict):
        directory = os.path.dirname(self.index_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".gallery_clip_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            os.replace(temporary, self.index_path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def image_vectors(self, paths: list[str]) -> dict:
        index = self._load_index()
        vectors = {}
        changed = False
        for original in paths:
            path = os.path.abspath(original)
            try:
                signature = self._signature(path)
                entry = index.get(path) or {}
                if (entry.get("size") != signature["size"]
                        or entry.get("mtime_ns") != signature["mtime_ns"]
                        or not isinstance(entry.get("vector"), list)):
                    vector = self.encode_image(original)
                    entry = {**signature, "vector": vector}
                    index[path] = entry
                    changed = True
                vectors[original] = entry["vector"]
            except (OSError, ValueError) as exc:
                logger.warning("跳过无法索引的图片 %s: %s", original, exc)
        if changed:
            self._save_index(index)
        return vectors

    def search_text(self, text: str, paths: list[str]) -> dict:
        return cosine_rank(self.encode_text(text), self.image_vectors(paths))

    def search_image(self, path: str, paths: list[str]) -> dict:
        return cosine_rank(self.encode_image(path), self.image_vectors(paths))
