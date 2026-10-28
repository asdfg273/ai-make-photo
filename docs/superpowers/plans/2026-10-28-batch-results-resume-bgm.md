# Batch Results, Resume and BGM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为角色批量出图增加可挑选的结果页和逐张断点续跑，并接通视频背景配乐文件选择。

**Architecture:** `BatchJobStore` 独占 `jobs.json` 的读写与恢复归一化；`BatchRunner` 只负责调度并逐张更新清单；`BatchResultView` 只展示清单并发出精修路径；现有场景对话框负责组合这些组件。视频页只增加控件，Mixin 负责文件选择与既有参数透传。

**Tech Stack:** Python 3、PyQt6、JSON 原子文件替换、QTimer 主线程轮询、项目现有离屏测试脚本。

## Global Constraints

- 工作目录固定为 `E:\pythonprojects\ai  make photo`，不重写生成链路，不引入新线程，不改变既有 `BatchRunner` 信号签名。
- 正常清单位于 `data/jobs.json`；测试必须注入临时路径，禁止写用户真实任务清单。
- 已完成任务不得因续跑重新生成；不移动、不重命名原始输出。
- 保留所有既有未提交修改，只精准暂存本计划文件。
- 新增界面文案提供中、英、日三语；测试使用 `QT_QPA_PLATFORM=offscreen`。

---

### Task 1: 逐张任务清单

**Files:**
- Create: `utils/batch_jobs.py`
- Create: `tests/test_batch_workflow.py`

**Interfaces:**
- Consumes: `utils.scene_library.Scene` 的 `uuid/name/count/enabled`。
- Produces: `BatchJobStore(path)`、`begin(character, scenes) -> dict`、`get(character_uuid) -> dict`、`unfinished(character_uuid) -> list[dict]`、`mark(character_uuid, scene_uuid, index, status, error="", output_paths=None) -> dict`。

- [ ] **Step 1: 写持久化失败测试**

```python
def test_job_store_roundtrip_and_recovers_running(tmp_path):
    from utils.batch_jobs import BatchJobStore
    from utils.character_library import Character
    from utils.scene_library import Scene

    store = BatchJobStore(tmp_path / "jobs.json")
    char = Character(name="角色甲")
    scene = Scene(character_uuid=char.uuid, name="雨夜", count=2)
    created = store.begin(char, [scene])
    assert [job["index"] for job in created["jobs"]] == [0, 1]
    store.mark(char.uuid, scene.uuid, 0, "done", output_paths=["a.png"])
    store.mark(char.uuid, scene.uuid, 1, "running")
    loaded = BatchJobStore(tmp_path / "jobs.json")
    assert loaded.get(char.uuid)["jobs"][0]["output_paths"] == ["a.png"]
    assert [j["index"] for j in loaded.unfinished(char.uuid)] == [1]
    assert loaded.get(char.uuid)["jobs"][1]["status"] == "pending"
```

- [ ] **Step 2: 运行并确认失败**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py::test_job_store_roundtrip_and_recovers_running -q`
Expected: FAIL，原因是 `utils.batch_jobs` 尚不存在。

- [ ] **Step 3: 实现原子清单**

```python
class BatchJobStore:
    def __init__(self, path):
        self.path = Path(path)

    def begin(self, character, scenes):
        jobs = [
            {"scene_uuid": scene.uuid, "scene_name": scene.name,
             "index": index, "status": "pending", "error": "",
             "output_paths": []}
            for scene in scenes if scene.enabled and int(scene.count) > 0
            for index in range(int(scene.count))
        ]
        session = self._new_session(character, jobs)
        sessions = self._read_sessions()
        sessions[character.uuid] = session
        self._write_sessions(sessions)
        return session

    def get(self, character_uuid):
        session = copy.deepcopy(self._read_sessions().get(character_uuid, {}))
        changed = False
        for job in session.get("jobs", []):
            if job.get("status") == "running":
                job["status"] = "pending"
                changed = True
        if changed:
            sessions = self._read_sessions()
            sessions[character_uuid] = copy.deepcopy(session)
            self._write_sessions(sessions)
        return session
```

实现时使用 `datetime.now(timezone.utc).isoformat()` 记录时间，以 `.tmp` 文件替换目标文件；`mark` 只修改 UUID 与序号同时匹配的一项，`unfinished` 返回所有 `status != "done"` 的项。

- [ ] **Step 4: 运行并确认通过**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py::test_job_store_roundtrip_and_recovers_running -q`
Expected: PASS。

- [ ] **Step 5: 精准提交**

```bash
git add -- utils/batch_jobs.py tests/test_batch_workflow.py
git commit -m "feat(batch): 增加逐张任务持久化清单"
```

### Task 2: Runner 续跑与状态更新

**Files:**
- Modify: `utils/batch_runner.py`
- Modify: `tests/test_batch_workflow.py`

**Interfaces:**
- Consumes: Task 1 的 `BatchJobStore` 全部接口。
- Produces: `BatchRunner.resume(character, scenes) -> bool`、`BatchRunner.job_store`；`_Job.output_paths: list[str]`。

- [ ] **Step 1: 写续跑失败测试**

使用不启动真实生成的宿主，直接验证恢复后的内部队列与完成计数：

```python
def test_runner_resume_skips_done_and_keeps_outputs(tmp_path):
    from utils.batch_runner import BatchRunner
    from utils.batch_jobs import BatchJobStore

    char = Character(name="角色甲")
    done_scene = Scene(character_uuid=char.uuid, name="已完成", count=1)
    retry_scene = Scene(character_uuid=char.uuid, name="待续跑", count=2)
    store = BatchJobStore(tmp_path / "jobs.json")
    store.begin(char, [done_scene, retry_scene])
    store.mark(char.uuid, done_scene.uuid, 0, "done", output_paths=["done.png"])
    store.mark(char.uuid, retry_scene.uuid, 0, "failed", " interrupted ")
    store.mark(char.uuid, retry_scene.uuid, 1, "running")
    runner = BatchRunner(_Host(), None, jobs_path=tmp_path / "jobs.json")
    runner._activate = lambda *args, **kwargs: None
    assert runner.resume(char, [done_scene, retry_scene])
    assert [(j.scene.uuid, j.index) for j in runner._queue] == [
        (retry_scene.uuid, 0), (retry_scene.uuid, 1)]
    assert runner._ok == 1 and runner._total == 3
    assert retry_scene.output_paths == []
```

测试宿主提供 `start_generation/stop_generation`；测试用 `_activate` 替换只为了避免启动 QTimer，生产代码仍由公共 `start/resume` 进入相同激活路径。

- [ ] **Step 2: 运行并确认失败**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py::test_runner_resume_skips_done_and_keeps_outputs -q`
Expected: FAIL，原因是 `jobs_path` 与 `resume` 尚不存在。

- [ ] **Step 3: 接入 Runner 生命周期**

```python
def resume(self, character, scenes):
    if self.running:
        return False
    session = self.job_store.get(character.uuid)
    by_uuid = {scene.uuid: scene for scene in scenes}
    jobs = []
    for item in session.get("jobs", []):
        scene = by_uuid.get(item.get("scene_uuid"))
        if scene is None:
            continue
        job = _Job(scene, int(item["index"]), list(item.get("output_paths", [])))
        if item.get("status") == "done":
            scene.output_paths.extend(job.output_paths)
        else:
            jobs.append(job)
    if not jobs:
        return False
    self._ok = sum(item.get("status") == "done" for item in session.get("jobs", []))
    self._total = len(session.get("jobs", []))
    return self._activate(character, jobs, reset=False)
```

在 `_dispatch_next` 设置 `scene.status = "running"` 后调用 `mark(..., "running")`；收图同时写入 `job.output_paths` 与 `scene.output_paths`；`_finish_job` 把状态、错误和该张路径写入清单。新任务调用 `begin`，续跑调用 `resume`，二者共用 `_activate` 设置批量模式、启动计时器和派发。

- [ ] **Step 4: 运行清单与 Runner 测试**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py -q`
Expected: 所有测试 PASS。

- [ ] **Step 5: 精准提交**

```bash
git add -- utils/batch_runner.py tests/test_batch_workflow.py
git commit -m "feat(batch): 支持未完成任务断点续跑"
```

### Task 3: 批量结果页与图生图桥接

**Files:**
- Create: `ui/batch_result_view.py`
- Modify: `ui/scene_batch_dialog.py`
- Modify: `ui/i18n_catalog.py`
- Modify: `tests/test_batch_workflow.py`

**Interfaces:**
- Consumes: `BatchJobStore.get()` 返回的会话和 `shell.send_path_to_img2img(path)`。
- Produces: `BatchResultView.set_manifest(session)`、`BatchResultView.send_requested = pyqtSignal(str)`；对话框 `btn_resume` 与 `result_view`。

- [ ] **Step 1: 写结果层级与精修失败测试**

创建临时 PNG 和包含一个角色、一个场景、两张任务的会话，断言树根有两个叶节点；选择第一张后 `btn_send` 可用，点击时信号发出真实路径。另写对话框测试：预置一项 `pending` 后 `btn_resume` 可用，点击调用 fake runner 的 `resume`。

```python
def test_result_view_groups_and_sends_selected_image(tmp_path, qapp):
    from ui.batch_result_view import BatchResultView
    image = tmp_path / "shot.png"
    Image.new("RGB", (16, 12), "white").save(image)
    session = {"character_name": "角色甲", "jobs": [
        {"scene_uuid": "s1", "scene_name": "雨夜", "index": 0,
         "status": "done", "error": "", "output_paths": [str(image)]},
        {"scene_uuid": "s1", "scene_name": "雨夜", "index": 1,
         "status": "pending", "error": "", "output_paths": []},
    ]}
    view = BatchResultView()
    sent = []
    view.send_requested.connect(sent.append)
    view.set_manifest(session)
    assert view.tree.topLevelItem(0).child(0).childCount() == 2
    view.tree.setCurrentItem(view.tree.topLevelItem(0).child(0).child(0))
    assert view.btn_send.isEnabled()
    view.btn_send.click()
    assert sent == [str(image)]
```

- [ ] **Step 2: 运行并确认失败**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py::test_result_view_groups_and_sends_selected_image -q`
Expected: FAIL，原因是结果视图尚不存在。

- [ ] **Step 3: 实现结果视图**

```python
class BatchResultView(QWidget):
    send_requested = pyqtSignal(str)

    def _on_send(self):
        path = self._selected_path()
        if path and Path(path).is_file():
            self.send_requested.emit(path)
```

`set_manifest` 每次清空并重建树；叶节点 `UserRole` 保存路径、`UserRole + 1` 保存状态。选择变化时加载 `QPixmap`，使用 `KeepAspectRatioByExpanding` 之外的 `KeepAspectRatio` 保证完整图片可见；非完成项或文件缺失禁用发送。

- [ ] **Step 4: 组合双页对话框与续跑按钮**

在对话框根布局加入 `QTabWidget`，把现有设置区放入“场景设置”，把 `BatchResultView` 放入“批量结果”。新增：

```python
def _send_result_to_img2img(self, path):
    sender = getattr(self.shell, "send_path_to_img2img", None)
    if not callable(sender):
        self.lbl_status.setText(ui_i18n.tr("当前宿主不支持图生图精修"))
        return
    sender(path)
    if self._runner is None or not self._runner.running:
        self.accept()
```

`_refresh_results` 在初始化、场景完成和整批结束时调用；`_refresh_resume` 根据 `unfinished()` 更新按钮文字和可用状态，`_on_resume` 调用 `runner.resume(character, self._scenes)` 并复用现有信号连接和进度控件。

- [ ] **Step 5: 运行结果与既有角色场景测试**

Run: `venv/Scripts/python.exe -m pytest tests/test_batch_workflow.py tests/test_character_scene.py -q`
Expected: 所有测试 PASS。

- [ ] **Step 6: 精准提交**

```bash
git add -- ui/batch_result_view.py ui/scene_batch_dialog.py ui/i18n_catalog.py tests/test_batch_workflow.py
git commit -m "feat(batch): 增加结果归档与精修入口"
```

### Task 4: 视频背景配乐控件

**Files:**
- Modify: `ui/pages/video_page.py`
- Modify: `ui/video_panel_mixin.py`
- Modify: `ui/i18n_catalog.py`
- Create: `tests/test_video_bgm.py`

**Interfaces:**
- Consumes: 既有 `voice_params["bg_music"]` 与 FFmpeg 混音分支。
- Produces: `chk_bg_music`、`txt_bg_music_path`、`btn_pick_bg_music`、`btn_clear_bg_music`、`_pick_bg_music()`、`_clear_bg_music()`。

- [ ] **Step 1: 写控件和路径透传失败测试**

```python
def test_video_bgm_controls_and_voice_params(tmp_path, qapp):
    from ui.pages.video_page import VideoPage
    from ui.video_panel_mixin import VideoPanelMixin

    class Host(QWidget, VideoPanelMixin):
        pass

    host = Host()
    page = VideoPage()
    page.build(host)
    assert host.txt_bg_music_path.isReadOnly()
    audio = tmp_path / "music.mp3"
    audio.write_bytes(b"test")
    host.chk_bg_music.setChecked(True)
    host.txt_bg_music_path.setText(str(audio))
    assert host._collect_bg_music({"engine": "chattts"}) == {
        "engine": "chattts", "bg_music": str(audio)}
    host._clear_bg_music()
    assert host._collect_bg_music({"engine": "chattts"}) == {"engine": "chattts"}
```

另用 `QFileDialog.getOpenFileName` mock 返回同一音频路径，调用 `_pick_bg_music()` 后断言 `txt_bg_music_path.text() == str(audio)`；整个测试不调用 `on_generate_video`，因此不会加载模型或启动后台生成。

- [ ] **Step 2: 运行并确认失败**

Run: `venv/Scripts/python.exe -m pytest tests/test_video_bgm.py -q`
Expected: FAIL，原因是配乐控件与路径方法尚不存在。

- [ ] **Step 3: 添加控件和文件方法**

```python
def _clear_bg_music(self):
    editor = getattr(self, "txt_bg_music_path", None)
    if editor is not None:
        editor.clear()


def _collect_bg_music(self, voice_params):
    params = dict(voice_params) if voice_params else None
    if not params or not self.chk_bg_music.isChecked():
        return params
    path = self.txt_bg_music_path.text().strip()
    if path and os.path.isfile(path):
        params["bg_music"] = path
    return params
```

`_pick_bg_music` 使用过滤器 `音频文件 (*.mp3 *.wav *.flac *.m4a *.aac *.ogg)`，选择成功后写入只读路径框；`on_generate_video` 用 `voice_params = self._collect_bg_music(voice_params)` 替换失效分支。

- [ ] **Step 4: 运行配乐、视频 UI 契约与多语言测试**

Run: `venv/Scripts/python.exe -m pytest tests/test_video_bgm.py tests/test_ui_contract.py tests/test_ui_i18n.py -q`
Expected: 所有测试 PASS。

- [ ] **Step 5: 精准提交**

```bash
git add -- ui/pages/video_page.py ui/video_panel_mixin.py ui/i18n_catalog.py tests/test_video_bgm.py
git commit -m "feat(video): 接通背景配乐文件选择"
```

### Task 5: 隔离回归与交付

**Files:**
- Verify only: 本计划全部实现文件及既有相关测试。

**Interfaces:**
- Consumes: Tasks 1–4 的最终索引内容。
- Produces: 最新提交哈希和可复核测试输出。

- [ ] **Step 1: 检查本轮差异**

Run: `git diff --check HEAD~4..HEAD`
Expected: exit 0，无空白错误。

- [ ] **Step 2: 导出隔离快照并编译**

Run: 用 `git archive HEAD` 导出到 `tests/artifacts/tmp/staged_batch_results_bgm`，再运行 `python -m py_compile utils/batch_jobs.py utils/batch_runner.py ui/batch_result_view.py ui/scene_batch_dialog.py ui/pages/video_page.py ui/video_panel_mixin.py`。
Expected: exit 0。

- [ ] **Step 3: 运行完整相关回归**

Run: 在快照中依次运行 `tests/test_batch_workflow.py`、`tests/test_video_bgm.py`、`tests/test_character_scene.py`、`tests/test_ui_contract.py`、`tests/test_ui_i18n.py`、`tests/test_v6_features.py`。
Expected: 每个脚本 exit 0；目标测试全部 PASS。

- [ ] **Step 4: 清理并核对工作区**

Run: 删除快照，`git status --short` 与 `git log -5 --oneline`。
Expected: 快照消失；本轮五个提交存在；其他既有未提交修改保留且未被全量暂存。
