"""基于 InsightFace 的角色脸部一致性校验；模型仅在首次调用时加载。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceCheckResult:
    ok: bool
    score: float | None = None
    error: str = ""


class FaceConsistencyChecker:
    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self._app = None

    def _load(self):
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
            providers = ["CPUExecutionProvider"]
            try:
                import onnxruntime
                available = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.insert(0, "CUDAExecutionProvider")
            except Exception:
                pass
            app = FaceAnalysis(name=self.model_name, providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
            return app
        except Exception as exc:
            raise RuntimeError(
                f"InsightFace 不可用或模型未就绪: {exc}") from exc

    @staticmethod
    def _embedding(app, path: Path):
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"无法读取图片 {path}: {exc}") from exc
        # InsightFace/OpenCV 约定 BGR。
        faces = app.get(array[:, :, ::-1])
        if not faces:
            raise ValueError(f"未检测到人脸: {path}")
        face = max(faces, key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        embedding = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"人脸特征无效: {path}")
        return embedding / norm

    def similarity(self, generated_path, reference_paths,
                   threshold: float = 0.55) -> FaceCheckResult:
        """返回生成图对全部参考图的最高身份余弦相似度。"""
        references = [Path(path) for path in reference_paths
                      if path and Path(path).is_file()]
        if not references:
            return FaceCheckResult(False, None, "没有有效的角色参考图")
        try:
            app = self._load()
            generated = self._embedding(app, Path(generated_path))
            scores = []
            for reference in references:
                try:
                    embedding = self._embedding(app, reference)
                    scores.append(float(np.dot(generated, embedding)))
                except Exception as exc:
                    logger.warning("参考图人脸校验跳过 %s: %s", reference, exc)
            if not scores:
                return FaceCheckResult(False, None, "所有参考图均未检测到有效人脸")
            score = max(0.0, min(1.0, max(scores)))
            return FaceCheckResult(score >= threshold, score, "")
        except Exception as exc:
            logger.warning("脸部一致性校验失败: %s", exc)
            return FaceCheckResult(False, None, str(exc))
