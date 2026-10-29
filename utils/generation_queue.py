"""线程安全的 FIFO 生成队列；不包含 Qt 或模型推理依赖。"""
from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class QueueJob:
    id: str
    snapshot: dict
    status: str = "pending"
    error: str = ""

    @property
    def label(self) -> str:
        values = self.snapshot.get("values", {})
        prompt = str(values.get("txt_prompt", "") or "未命名任务").strip()
        return prompt[:48] + ("…" if len(prompt) > 48 else "")


class GenerationQueue:
    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._jobs: list[QueueJob] = []
        self._current: QueueJob | None = None
        self._paused = False

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def current(self) -> QueueJob | None:
        with self._condition:
            return self._current

    def jobs(self) -> list[QueueJob]:
        with self._condition:
            return list(self._jobs)

    def pending_count(self) -> int:
        with self._condition:
            return sum(job.status == "pending" for job in self._jobs)

    def enqueue(self, snapshot: dict) -> str:
        job = QueueJob(uuid.uuid4().hex, copy.deepcopy(snapshot))
        with self._condition:
            self._jobs.append(job)
            self._condition.notify_all()
            return job.id

    def _get(self, job_id: str) -> QueueJob:
        for job in self._jobs:
            if job.id == job_id:
                return job
        raise KeyError(job_id)

    def update(self, job_id: str, values: dict):
        with self._condition:
            job = self._get(job_id)
            if job.status != "pending":
                raise ValueError("只能编辑等待中的任务")
            job.snapshot.setdefault("values", {}).update(copy.deepcopy(values))

    def promote(self, job_id: str):
        with self._condition:
            job = self._get(job_id)
            if job.status != "pending":
                raise ValueError("只能调整等待中的任务")
            self._jobs.remove(job)
            index = next((i for i, item in enumerate(self._jobs)
                          if item.status == "pending"), len(self._jobs))
            self._jobs.insert(index, job)
            self._condition.notify_all()

    def move(self, job_id: str, offset: int):
        with self._condition:
            job = self._get(job_id)
            if job.status != "pending":
                raise ValueError("只能调整等待中的任务")
            old = self._jobs.index(job)
            new = max(0, min(len(self._jobs) - 1, old + int(offset)))
            if self._jobs[new].status != "pending":
                return
            self._jobs.insert(new, self._jobs.pop(old))

    def remove(self, job_id: str):
        with self._condition:
            job = self._get(job_id)
            if job.status != "pending":
                raise ValueError("运行中的任务请使用中断，不能直接从队列删除")
            self._jobs.remove(job)

    def start_next(self) -> QueueJob | None:
        with self._condition:
            if self._paused or self._current is not None:
                return None
            job = next((item for item in self._jobs if item.status == "pending"), None)
            if job is None:
                return None
            job.status = "running"
            self._current = job
            return job

    def finish_current(self, status: str = "done", error: str = ""):
        with self._condition:
            if self._current is None:
                return
            self._current.status = status
            self._current.error = error
            self._current = None
            self._condition.notify_all()

    def set_paused(self, paused: bool):
        with self._condition:
            self._paused = bool(paused)
            self._condition.notify_all()

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def wait_if_paused(self, cancelled=None) -> bool:
        """在生成线程的步边界等待；返回 False 表示已取消。"""
        with self._condition:
            while self._paused and not (cancelled and cancelled()):
                self._condition.wait(timeout=0.25)
            return not (cancelled and cancelled())
