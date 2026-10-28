"""批量出图逐张任务清单：原子持久化与崩溃恢复。"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchJobStore:
    """按角色保存最近一次批量队列，每个场景的第 N 张独立记录。"""

    def __init__(self, path):
        self.path = Path(path)

    def _read_sessions(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
            return sessions if isinstance(sessions, dict) else {}
        except Exception as exc:
            logger.error("批量任务清单读取失败 %s: %s", self.path, exc)
            return {}

    def _write_sessions(self, sessions: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "sessions": sessions}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _new_session(character, jobs: list[dict]) -> dict:
        timestamp = _now()
        return {
            "character_uuid": getattr(character, "uuid", ""),
            "character_name": getattr(character, "name", ""),
            "created_at": timestamp,
            "updated_at": timestamp,
            "jobs": jobs,
        }

    def begin(self, character, scenes) -> dict:
        jobs = [
            {
                "scene_uuid": getattr(scene, "uuid", ""),
                "scene_name": getattr(scene, "name", ""),
                "index": index,
                "status": "pending",
                "error": "",
                "output_paths": [],
            }
            for scene in scenes
            if getattr(scene, "enabled", True) and int(getattr(scene, "count", 1)) > 0
            for index in range(int(getattr(scene, "count", 1)))
        ]
        session = self._new_session(character, jobs)
        sessions = self._read_sessions()
        sessions[getattr(character, "uuid", "")] = session
        self._write_sessions(sessions)
        return copy.deepcopy(session)

    def get(self, character_uuid: str) -> dict:
        sessions = self._read_sessions()
        session = copy.deepcopy(sessions.get(character_uuid, {}))
        changed = False
        for job in session.get("jobs", []):
            if job.get("status") == "running":
                job["status"] = "pending"
                changed = True
        if changed:
            session["updated_at"] = _now()
            sessions[character_uuid] = copy.deepcopy(session)
            try:
                self._write_sessions(sessions)
            except Exception as exc:
                logger.warning("批量任务恢复状态保存失败: %s", exc)
        return session

    def unfinished(self, character_uuid: str) -> list[dict]:
        return [job for job in self.get(character_uuid).get("jobs", [])
                if job.get("status") != "done"]

    def mark(self, character_uuid: str, scene_uuid: str, index: int,
             status: str, error: str = "", output_paths=None) -> dict:
        sessions = self._read_sessions()
        session = sessions.get(character_uuid)
        if not isinstance(session, dict):
            return {}
        for job in session.get("jobs", []):
            if (job.get("scene_uuid") == scene_uuid
                    and int(job.get("index", -1)) == int(index)):
                job["status"] = status
                job["error"] = error or ""
                if output_paths is not None:
                    job["output_paths"] = [str(path) for path in output_paths]
                session["updated_at"] = _now()
                self._write_sessions(sessions)
                return copy.deepcopy(job)
        return {}
