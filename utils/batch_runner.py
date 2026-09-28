"""批量出图驱动器 —— 主线程轮询版。

为什么不用信号:app_generation.py 的生成链路没有完成信号,
完成标志是 generation_task 的 finally 里 self.is_generating = False。

为什么不用工作线程:参数下发必须写 Qt 控件,只能在主线程。
靠 QTimer 300ms 轮询判断单场景结束,全程零跨线程控件操作。

为什么轮询不会卡界面:生成本身在 threading.Thread 里跑,主线程事件循环空闲。
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.scene_library import Scene
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

POLL_MS = 300        # 轮询间隔
MAX_DEAD = 3         # 连续多少个场景压根没启动起来就中止
RENAME_OUTPUTS = False   # 是否把产出文件改名为 角色_场景_序号.png（见文末风险）

_BAD = re.compile(r'[\\/:*?"<>|\s]+')


def _safe_name(s: str, fallback: str = "role") -> str:
    s = _BAD.sub("_", (s or "").strip()).strip("_")
    return s[:48] or fallback


def resolve_output_dir(config=None) -> Path:
    """统一走 utils.paths.resolve_output_dir（全项目唯一实现）。

    ⚠️ 以前这里调用 `_rod()` **不传 config**，于是永远返回默认的 photo/，
    用户在「设置」里改的输出目录对批量生成完全无效：
    `_out_dir` 指向错误目录 → `rglob("*.png")` 快照/新文件检测全部落空，
    批量出图的成功判定与归档都会错，还会白扫一遍整个 photo/ 树。

    没显式传 config 时自行从 AppConfig 读一份（load 很轻，且调用点很少）。
    """
    try:
        from utils.paths import resolve_output_dir as _rod
        if config is None:
            try:
                from core.config_manager import AppConfig
                config = AppConfig.load()
            except Exception:
                config = None      # 读配置失败就退回默认目录，不阻断批量
        return Path(_rod(config))
    except Exception:
        from utils.paths import OUTPUT_DIR as _O
        return Path(_O)


@dataclass
class _Job:
    scene: object
    index: int          # 该场景内第几张（0-based）


class BatchRunner(QObject):
    """串行跑完一个角色的全部启用场景。信号名与旧版保持一致，调用方不用改。"""

    progress   = pyqtSignal(int, int, str)    # done, total, 说明
    scene_done = pyqtSignal(str, str)         # scene_uuid, status
    finished   = pyqtSignal(int, int)         # ok, fail
    log        = pyqtSignal(str)

    def __init__(self, host, library, parent=None):
        """host: 拥有 start_generation / txt_prompt / spin_steps 的宿主对象,即 shell。"""
        super().__init__(parent)
        self.host = host
        self.library = library
        self._queue: list[_Job] = []
        self._cur: Optional[_Job] = None
        self._char = None
        self._total = 0
        self._ok = self._fail = 0
        self._running = False
        self._stop = False
        self._wrapped = False
        self._finished = False
        self._dead = 0
        self._saw_running = False
        self._ts_dispatch = 0.0
        self._out_before: set[Path] = set()
        self._out_dir = resolve_output_dir()
        self._seed_widget = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    # ───────── 对外 ─────────
    @property
    def running(self) -> bool:
        return self._running

    def start(self, char, scenes) -> bool:
        if self._running:
            return False
        if not callable(getattr(self.host, "start_generation", None)):
            self.log.emit("❌ host 上没有 start_generation，批量中止（是不是传了 page 而不是 shell？）")
            return False

        jobs = [_Job(s, i)
                for s in scenes
                if getattr(s, "enabled", True) and int(getattr(s, "count", 1)) > 0
                for i in range(max(1, int(getattr(s, "count", 1))))]
        if not jobs:
            self.log.emit("没有启用的场景")
            return False

        self._queue, self._ok, self._fail = jobs, 0, 0
        self._total = len(jobs)
        self._char = char
        self._stop = self._wrapped = self._finished = False
        self._dead = 0
        self._running = True
        self._set_batch_mode(True)

        for s in scenes:
            s.status, s.error, s.output_paths = "pending", "", []
        self._save_library()

        self.progress.emit(0, self._total, f"共 {self._total} 张，开始…")
        self._timer.start()
        self._dispatch_next()
        return True

    def stop(self):
        if not self._running:
            return
        self._stop = True
        self._queue.clear()
        self.log.emit("⏹ 正在中断当前生成…")
        fn = getattr(self.host, "stop_generation", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                logger.error(f"stop_generation 调用失败: {e}")
        try:
            self.host.cancel_flag = True      # 兜底：主循环会抛 InterruptedError
        except Exception:
            pass

    # ───────── 调度 ─────────
    def _dispatch_next(self):
        if self._stop:
            return self._wrap_up()

        job = self._queue.pop(0) if self._queue else None
        if job is None:
            return self._wrap_up()

        self._cur = job
        scene = job.scene

        try:
            self._push_params(self._char, scene)
        except Exception as e:
            logger.exception("参数下发失败")
            return self._finish_job("failed", f"参数下发失败: {e}")

        scene.status = "running"
        done = self._ok + self._fail
        self.progress.emit(
            done, self._total,
            f"[{scene.name}] 第 {job.index + 1}/{getattr(scene, 'count', 1)} 张 — 生成中…")

        self._ts_dispatch = time.time()
        self._out_before = self._snapshot_outputs()
        self._saw_running = False
        self._apply_seed(scene, job.index)
        seed = int(getattr(scene, "seed", -1))
        self.host._seed_override = None if seed < 0 else seed

        try:
            self.host.start_generation()      # 同步置 is_generating=True，然后开线程
        except Exception as e:
            logger.exception("start_generation 抛出")
            return self._finish_job("failed", str(e))

    def _poll(self):
        # 已进入收尾：等生成线程真正退出后再落地 finished（非阻塞）
        if self._wrapped:
            return self._finalize()

        if self._stop:
            return self._wrap_up()

        host = self.host
        if getattr(host, "is_generating", False):
            self._saw_running = True
            return
        # 线程还活着也算"在跑"，避免提前判定失败
        if not self._gen_thread_stopped():
            self._saw_running = True
            return

        job = self._cur
        if job is None:                        # 异常兜底，不该发生
            return self._wrap_up()

        # 先看盘上有没有新文件 —— 比看标志位可靠
        paths = self._harvest(job, self._ts_dispatch)
        if paths:
            self._dead = 0
            job.scene.output_paths.extend(str(p) for p in paths)
            return self._finish_job("done", "")

        if not self._saw_running:
            self._dead += 1
            logger.warning(
                f"⚠️ 场景「{getattr(job.scene, 'name', '?')}」未进入生成状态"
                f"（连续 {self._dead}/{MAX_DEAD}）")
            if self._dead >= MAX_DEAD:
                self.log.emit("连续多次无法启动生成，批量中止（看日志里 start_generation 的早退原因）")
                self._queue.clear()
                return self._wrap_up()
            return self._finish_job("failed", "未进入生成状态")

        return self._finish_job("failed", "未产出文件")

    def _finish_job(self, status: str, err: str):
        job = self._cur
        self._cur = None
        if job is not None:
            if status == "done":
                self._ok += 1
            else:
                self._fail += 1
            job.scene.status = status
            job.scene.error = err
            self._save_library()
            self.scene_done.emit(getattr(job.scene, "uuid", ""), status)
        if self._stop:
            return self._wrap_up()
        self._dispatch_next()

    def _wrap_up(self):
        """进入收尾流程。

        注意：本方法跑在主线程的 QTimer 回调里，**不能阻塞等待**，
        否则界面会卡住。所以这里只记录"已收尾"，等生成线程真正退出后
        再由 _poll 调 _finalize()。
        """
        if self._wrapped:
            return
        self._wrapped = True
        self._running = False
        self._cur = None
        self._set_batch_mode(False)
        self._finalize()

    def _finalize(self):
        """生成线程确认退出后才做最终清理并发出 finished。

        ⚠️ 关键：不能无条件把 host.cancel_flag 复位成 False。
        stop() 靠 cancel_flag=True 让采样循环抛 InterruptedError；
        若立刻清掉，当前那张图的生成线程会**继续采样到结束**，
        用户以为停了其实还在出图/写盘。
        """
        if getattr(self, "_finished", False):
            return
        if not self._gen_thread_stopped():
            return                      # 还没停，等下一次 _poll 再试
        self._finished = True
        self._timer.stop()
        try:
            self.host.cancel_flag = False
        except Exception:
            pass
        # 清掉本次批量留下的种子覆盖，避免影响之后的手动生成
        try:
            self.host._seed_override = None
        except Exception:
            pass
        self._save_library()
        total = self._ok + self._fail
        self.progress.emit(total, total, f"完成：成功 {self._ok} / 失败 {self._fail}")
        self.finished.emit(self._ok, self._fail)

    def _gen_thread_stopped(self) -> bool:
        """生成线程是否已经退出（非阻塞）。"""
        if getattr(self.host, "is_generating", False):
            return False
        t = getattr(self.host, "_gen_thread", None)
        if t is None:
            return True
        alive = getattr(t, "is_alive", None)
        return not (callable(alive) and alive())

    # ───────── 参数下发 ─────────
    def _push_params(self, char, scene):
        host = self.host
        skipped: list[str] = []

        # 提示词
        pos = (scene.merged_positive(getattr(char, "base_prompt", ""))
               if hasattr(scene, "merged_positive") else getattr(scene, "prompt", ""))
        neg = (scene.merged_negative(getattr(char, "negative_prompt", ""))
               if hasattr(scene, "merged_negative") else getattr(scene, "negative", ""))

        if self._write_text(getattr(host, "txt_prompt", None), pos):
            pass
        else:
            skipped.append("prompt(无 txt_prompt)")

        # ⚠️ 实际别名是 txt_neg（见 _gt_prepare_context），不是 txt_neg_prompt
        neg_box = (getattr(host, "txt_neg", None)
                   or getattr(host, "txt_neg_prompt", None)
                   or getattr(host, "txt_negative", None))
        if neg and not self._write_text(neg_box, neg):
            skipped.append("negative(无 txt_neg)")

        # 分辨率：实际控件是 combo_res（文本 "512x512"），不是 spin_width/height
        w, h = getattr(scene, "width", None), getattr(scene, "height", None)
        if w and h:
            combo = getattr(host, "combo_res", None)
            if combo is not None and hasattr(combo, "findText"):
                idx = combo.findText(f"{int(w)}x{int(h)}")
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    skipped.append(f"res({int(w)}x{int(h)} 不在 combo_res 里)")
            else:
                for attr, val in (("spin_width", w), ("spin_height", h)):
                    ww = getattr(host, attr, None)
                    if ww is not None and hasattr(ww, "setValue"):
                        ww.setValue(int(val))
                    else:
                        skipped.append(f"{attr}(不存在)")

        # 步数
        steps = getattr(scene, "steps", None)
        if steps:
            sp = getattr(host, "spin_steps", None)
            if sp is not None and hasattr(sp, "setValue"):
                sp.setValue(int(steps))
            else:
                skipped.append("steps(无 spin_steps)")

        # 种子控件探测（不确定下面会不会被读，见文末待确认 B）
        self._seed_widget = (getattr(host, "spin_seed", None)
                             or getattr(host, "spin_seed_value", None))
        if self._seed_widget is None:
            skipped.append("seed(未找到种子控件，本批用随机种子)")

        # CFG：scale_cfg 是通过 _sld() 读的滑块，刻度未确认，暂不下发（见待确认 C）
        cfg = getattr(scene, "cfg", None)
        if cfg:
            sl = getattr(host, "scale_cfg", None)
            if sl is not None and hasattr(sl, "setValue"):
                sl.setValue(float(cfg))
            else:
                skipped.append("cfg(无 scale_cfg)")

        if skipped:
            self.log.emit("参数跳过: " + "、".join(skipped))

    @staticmethod
    def _write_text(box, text: str) -> bool:
        if box is None:
            return False
        if hasattr(box, "setPlainText"):
            box.setPlainText(str(text))
            return True
        if hasattr(box, "setText"):
            box.setText(str(text))
            return True
        return False

    def _apply_seed(self, scene: Scene, index: int):
        w = self._seed_widget
        if w is None or not hasattr(w, "setValue"):
            return
        seed = int(getattr(scene, "seed", -1))
        w.setValue(-1 if seed < 0 else seed + index)

    # ───────── 收图 ─────────
    def _snapshot_outputs(self) -> set[Path]:
        try:
            return {p for p in self._out_dir.rglob("*.png")}
        except Exception:
            return set()

    def _harvest(self, job: _Job, since: float) -> list[Path]:
        try:
            cand = [p for p in self._out_dir.rglob("*.png")
                    if p.stat().st_mtime >= since - 1 and p not in self._out_before]
        except Exception as e:
            logger.error(f"扫描输出目录失败: {e}")
            return []
        if not cand:
            return []
        cand.sort(key=lambda p: p.stat().st_mtime)

        if not RENAME_OUTPUTS:
            return cand

        char_name = _safe_name(getattr(self._char, "name", ""), "role")
        scene_name = _safe_name(getattr(job.scene, "name", ""), "scene")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = []
        for k, src in enumerate(cand):
            dst = src.with_name(f"{char_name}_{scene_name}_{job.index + 1}"
                                f"{'_' + str(k + 1) if len(cand) > 1 else ''}_{stamp}{src.suffix}")
            try:
                if dst != src:
                    src.rename(dst)
                out.append(dst)
            except Exception as e:
                logger.warning(f"重命名 {src} 失败: {e}")
                out.append(src)
        return out

    # ───────── 杂项 ─────────
    def _set_batch_mode(self, on: bool):
        try:
            self.host._batch_mode = on
        except Exception:
            pass

    def _save_library(self):
        if self.library is None:
            return
        if not hasattr(self.library, "save"):
            # 传错库（例如 CharacterLibrary）过去会被静默吞掉，
            # 导致场景 status/output_paths 永不落盘、重启即丢
            logger.error(
                "❌ 场景库保存失败：library 是 %s，没有 save() 方法"
                "（应传 SceneLibrary）", type(self.library).__name__)
            return
        try:
            self.library.save()
        except Exception as e:
            logger.warning(f"场景库保存失败: {e}")