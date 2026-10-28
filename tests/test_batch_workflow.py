"""批量结果与断点续跑测试；不启动真实模型生成。"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.character_library import Character
from utils.scene_library import Scene


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


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"BATCH_WORKFLOW_OK: {len(tests)} tests")
