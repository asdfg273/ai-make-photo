"""批量结果与断点续跑测试；不启动真实模型生成。"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from utils.character_library import Character
from utils.scene_library import Scene

_APP = QApplication.instance() or QApplication([])


def test_job_store_roundtrip_and_recovers_running():
    from utils.batch_jobs import BatchJobStore

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jobs.json"
        store = BatchJobStore(path)
        char = Character(name="角色甲")
        scene = Scene(character_uuid=char.uuid, name="雨夜", count=2)

        created = store.begin(char, [scene])
        assert [job["index"] for job in created["jobs"]] == [0, 1]
        store.mark(char.uuid, scene.uuid, 0, "done", output_paths=["a.png"])
        store.mark(char.uuid, scene.uuid, 1, "running")

        loaded = BatchJobStore(path)
        assert loaded.get(char.uuid)["jobs"][0]["output_paths"] == ["a.png"]
        assert [j["index"] for j in loaded.unfinished(char.uuid)] == [1]
        assert loaded.get(char.uuid)["jobs"][1]["status"] == "pending"


def test_runner_resume_skips_done_and_keeps_outputs():
    from utils.batch_jobs import BatchJobStore
    from utils.batch_runner import BatchRunner

    class Host:
        def start_generation(self):
            pass

        def stop_generation(self):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jobs.json"
        char = Character(name="角色甲")
        done_scene = Scene(character_uuid=char.uuid, name="已完成", count=1)
        retry_scene = Scene(character_uuid=char.uuid, name="待续跑", count=2)
        store = BatchJobStore(path)
        store.begin(char, [done_scene, retry_scene])
        store.mark(char.uuid, done_scene.uuid, 0, "done", output_paths=["done.png"])
        store.mark(char.uuid, retry_scene.uuid, 0, "failed", "interrupted")
        store.mark(char.uuid, retry_scene.uuid, 1, "running")

        runner = BatchRunner(Host(), None, jobs_path=path)
        activated = []
        runner._activate = lambda character: activated.append(character) or True
        assert runner.resume(char, [done_scene, retry_scene])
        assert [(job.scene.uuid, job.index) for job in runner._queue] == [
            (retry_scene.uuid, 0), (retry_scene.uuid, 1)]
        assert runner._ok == 1 and runner._total == 3
        assert done_scene.output_paths == ["done.png"]
        assert retry_scene.output_paths == []
        assert activated == [char]


def test_runner_dispatch_and_finish_persist_each_job():
    from utils.batch_jobs import BatchJobStore
    from utils.batch_runner import BatchRunner, _Job

    class Host:
        is_generating = False
        cancel_flag = False
        _gen_thread = None

        def start_generation(self):
            pass

        def stop_generation(self):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jobs.json"
        char = Character(name="角色甲")
        scene = Scene(character_uuid=char.uuid, name="雨夜", count=1)
        store = BatchJobStore(path)
        store.begin(char, [scene])
        runner = BatchRunner(Host(), None, jobs_path=path)
        runner._char = char
        runner._out_dir = Path(directory)
        runner._queue = [_Job(scene, 0)]
        runner._total = 1

        runner._dispatch_next()
        # get() 会模拟重开对话框并把 running 恢复为 pending；这里检查原始落盘状态。
        assert store._read_sessions()[char.uuid]["jobs"][0]["status"] == "running"
        runner._cur.output_paths = ["a.png"]
        runner._finish_job("done", "")
        saved = store.get(char.uuid)["jobs"][0]
        assert saved["status"] == "done"
        assert saved["output_paths"] == ["a.png"]


def test_runner_stop_restores_current_job_for_resume():
    from utils.batch_jobs import BatchJobStore
    from utils.batch_runner import BatchRunner, _Job

    class Host:
        is_generating = False
        cancel_flag = False
        _gen_thread = None

        def start_generation(self):
            pass

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jobs.json"
        char = Character(name="角色甲")
        scene = Scene(character_uuid=char.uuid, name="雨夜", count=1)
        store = BatchJobStore(path)
        store.begin(char, [scene])
        store.mark(char.uuid, scene.uuid, 0, "running")
        scene.status = "running"
        runner = BatchRunner(Host(), None, jobs_path=path)
        runner._char = char
        runner._cur = _Job(scene, 0)
        runner._stop = True
        runner._running = True

        runner._wrap_up()
        assert scene.status == "pending"
        assert store._read_sessions()[char.uuid]["jobs"][0]["status"] == "pending"


def test_result_view_groups_and_sends_selected_image():
    from PIL import Image
    from ui.batch_result_view import BatchResultView

    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "shot.png"
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
        try:
            view.set_manifest(session)
            scene_item = view.tree.topLevelItem(0).child(0)
            assert scene_item.childCount() == 2
            assert "1" in scene_item.child(0).text(0)
            view.tree.setCurrentItem(scene_item.child(0))
            assert view.btn_send.isEnabled()
            assert not view.preview.pixmap().isNull()
            view.btn_send.click()
            assert sent == [str(image)]
        finally:
            view.close()
            view.deleteLater()


def test_dialog_offers_resume_for_unfinished_jobs():
    from utils.batch_jobs import BatchJobStore
    from utils.batch_runner import BatchRunner
    from utils.scene_library import SceneLibrary
    from ui.scene_batch_dialog import SceneBatchDialog

    class Host:
        def start_generation(self):
            pass

    with tempfile.TemporaryDirectory() as directory:
        jobs_path = Path(directory) / "jobs.json"
        library = SceneLibrary(Path(directory) / "scenes.json")
        char = Character(name="角色甲")
        scene = Scene(character_uuid=char.uuid, name="雨夜", count=1)
        library.add(scene)
        BatchJobStore(jobs_path).begin(char, [scene])
        runner = BatchRunner(Host(), library, jobs_path=jobs_path)
        runner._activate = lambda character: True
        dialog = SceneBatchDialog(
            shell=Host(), character=char, library=library, runner=runner)
        try:
            assert dialog.btn_resume.isEnabled()
            assert "1" in dialog.btn_resume.text()
            dialog.btn_resume.click()
            assert len(runner._queue) == 1
        finally:
            dialog.close()
            dialog.deleteLater()


def test_dialog_send_result_uses_shell_bridge():
    from PyQt6.QtWidgets import QDialog
    from ui.scene_batch_dialog import SceneBatchDialog

    class Host:
        def __init__(self):
            self.paths = []

        def send_path_to_img2img(self, path):
            self.paths.append(path)

    host = Host()
    dialog = SceneBatchDialog(shell=host)
    try:
        dialog._send_result_to_img2img("chosen.png")
        assert host.paths == ["chosen.png"]
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        dialog.close()
        dialog.deleteLater()


def test_batch_ui_translations_exist():
    from ui.i18n_catalog import TRANSLATIONS

    keys = {
        "角色 / 场景 / 序号", "选择一张图片查看预览", "发送到图生图精修",
        "暂无批量结果", "暂无可预览的已完成图片", "图片尚未生成或文件不存在",
        "无法读取图片预览", "第 {index} 张 — {status}", "继续未完成场景",
        "继续未完成场景 ({count})", "场景设置", "批量结果",
        "没有可续跑的未完成任务", "当前宿主不支持图生图精修", "待生成",
    }
    missing = sorted(key for key in keys if key not in TRANSLATIONS)
    assert not missing, f"缺少批量结果英日翻译: {missing}"
    assert all(len(TRANSLATIONS[key]) == 2 for key in keys)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"BATCH_WORKFLOW_OK: {len(tests)} tests")
