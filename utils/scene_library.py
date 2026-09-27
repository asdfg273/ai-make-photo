"""场景变体库 — 绑定角色,支持批量出图"""
from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCENE_FILE = Path("data/scenes.json")


@dataclass
class Scene:
    uuid: str = field(default_factory=lambda: _uuid.uuid4().hex)
    character_uuid: str = ""
    name: str = "新场景"
    pose_prompt: str = ""          # 追加到角色 base_prompt
    negative_extra: str = ""       # 追加到角色 negative_prompt
    seed: int = -1                 # -1 = 每张随机
    count: int = 1                 # 该场景出几张
    steps: Optional[int] = None    # None = 沿用页面当前值
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    enabled: bool = True
    status: str = "pending"        # pending / running / done / failed
    error: str = ""
    output_paths: list[str] = field(default_factory=list)

    @staticmethod
    def _join(base: str, extra: str) -> str:
        base = (base or "").strip().strip(",")
        extra = (extra or "").strip().strip(",")
        if not extra:
            return base
        if not base:
            return extra
        return f"{base}, {extra}"

    def merged_positive(self, base: str) -> str:
        return self._join(base, self.pose_prompt)

    def merged_negative(self, base: str) -> str:
        return self._join(base, self.negative_extra)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class SceneLibrary:
    def __init__(self, path: Path = SCENE_FILE):
        self.path = Path(path)
        self._scenes: list[Scene] = []
        self.load()

    def load(self) -> None:
        self._scenes = []
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"场景库读取失败: {e}")
            return
        items = raw.get("scenes", []) if isinstance(raw, dict) else raw
        for d in items or []:
            try:
                self._scenes.append(Scene.from_dict(d))
            except Exception as e:
                logger.warning(f"跳过损坏场景: {e}")

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "scenes": [s.to_dict() for s in self._scenes]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)          # 原子写,避免中途崩溃损坏
            return True
        except Exception as e:
            logger.error(f"场景库保存失败: {e}")
            return False

    def list_for_character(self, char_uuid: str) -> list[Scene]:
        return [s for s in self._scenes if s.character_uuid == char_uuid]

    def get(self, uuid: str) -> Optional[Scene]:
        return next((s for s in self._scenes if s.uuid == uuid), None)

    def add(self, scene: Scene) -> bool:
        if not scene.character_uuid:
            logger.warning("场景缺少 character_uuid,拒绝入库")
            return False
        if not getattr(scene, "uuid", ""):
            scene.uuid = _uuid.uuid4().hex
        self._scenes.append(scene)
        return self.save()

    def delete(self, uuid: str) -> bool:
        before = len(self._scenes)
        self._scenes = [s for s in self._scenes if s.uuid != uuid]
        return self.save() if len(self._scenes) != before else False

    def move(self, uuid: str, delta: int) -> bool:
        idx = next((i for i, s in enumerate(self._scenes) if s.uuid == uuid), -1)
        if idx < 0:
            return False
        j = max(0, min(len(self._scenes) - 1, idx + delta))
        if j == idx:
            return False
        self._scenes.insert(j, self._scenes.pop(idx))
        return self.save()

    def reset_status(self) -> None:
        for s in self._scenes:
            s.status, s.error = "pending", ""




