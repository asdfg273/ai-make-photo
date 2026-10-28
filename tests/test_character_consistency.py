"""角色一致性纯数据测试；真实模型由独立 fake 接口替代。"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication
_APP = QApplication.instance() or QApplication([])

from utils.character_library import Character
from utils.character_references import (
    cached_feature, normalize_references, select_reference, store_feature,
)


def test_reference_migration_and_pose_selection(tmp_path):
    front = tmp_path / "front.png"
    front.write_bytes(b"front")
    side = tmp_path / "side.png"
    side.write_bytes(b"side")
    char = Character(name="甲", ref_image_path=str(front))
    char.reference_images.append({"path": str(side), "pose": "side"})
    refs = normalize_references(char)
    assert refs[0]["pose"] == "front"
    assert select_reference(char, "side profile", 0)["path"] == str(side)
    assert select_reference(char, "walking", 1)["path"] == str(side)


def test_feature_cache_hits_and_invalidates(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"a")
    char = Character(name="甲", ref_image_path=str(image))
    assert cached_feature(char, str(image)) == ""
    store_feature(char, str(image), "silver hair")
    assert cached_feature(char, str(image)) == "silver hair"
    image.write_bytes(b"changed")
    assert cached_feature(char, str(image)) == ""


def test_editor_supports_four_reference_poses(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from ui.pages.characters_page import CharacterEditor

    QApplication.instance() or QApplication([])
    editor = CharacterEditor()
    try:
        char = Character(name="甲")
        editor.load(char)
        paths = []
        for name in ("a.png", "b.png", "c.png"):
            path = tmp_path / name
            path.write_bytes(b"x")
            paths.append(str(path))
        with patch("ui.pages.characters_page.QFileDialog.getOpenFileNames",
                   return_value=(paths, "图像")):
            editor.browse_refs()
        assert editor.ref_list.count() == 3
        assert [ref["pose"] for ref in char.reference_images] == [
            "front", "side", "full"]
    finally:
        editor.close()
        editor.deleteLater()


def test_generation_uses_persistent_feature_before_ai(tmp_path):
    from utils.character_features import feature_for_generation

    image = tmp_path / "ref.png"
    image.write_bytes(b"x")
    char = Character(name="甲", ref_image_path=str(image))
    store_feature(char, str(image), "cached tags")
    host = SimpleNamespace(
        _active_character=char, _gt_extract_character_features=Mock())
    assert feature_for_generation(host, str(image)) == "cached tags"
    host._gt_extract_character_features.assert_not_called()


def test_generation_backfills_missing_feature(tmp_path):
    from utils.character_features import feature_for_generation

    image = tmp_path / "ref.png"
    image.write_bytes(b"x")
    char = Character(name="甲", ref_image_path=str(image))
    library = SimpleNamespace(save_character=Mock(return_value=True))
    host = SimpleNamespace(
        _active_character=char, _character_library=library,
        _gt_extract_character_features=Mock(return_value="fresh tags"))
    assert feature_for_generation(host, str(image)) == "fresh tags"
    assert cached_feature(char, str(image)) == "fresh tags"
    library.save_character.assert_called_once_with(char)


def test_batch_low_similarity_rerolls_same_job_once(tmp_path):
    from utils.batch_runner import BatchRunner, _Job
    from utils.face_consistency import FaceCheckResult
    from utils.scene_library import Scene

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref")
    char = Character(name="甲", ref_image_path=str(ref), face_max_retries=1)
    scene = Scene(character_uuid=char.uuid, name="近景", count=1)
    checker = Mock()
    checker.similarity.side_effect = [
        FaceCheckResult(False, 0.2, ""), FaceCheckResult(True, 0.8, "")]
    host = SimpleNamespace(start_generation=Mock(), stop_generation=Mock())
    runner = BatchRunner(
        host, None, jobs_path=tmp_path / "jobs.json", face_checker=checker)
    runner._char = char
    job = _Job(scene, 0)
    runner._cur = job
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(b"candidate")
    assert runner._check_face(job, [candidate]) == "retry"
    assert runner._cur is job and job.retry_count == 1
    assert runner._check_face(job, [candidate]) == "done"


def test_batch_face_error_does_not_reroll(tmp_path):
    from utils.batch_runner import BatchRunner, _Job
    from utils.face_consistency import FaceCheckResult
    from utils.scene_library import Scene

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref")
    char = Character(name="甲", ref_image_path=str(ref), face_max_retries=3)
    scene = Scene(character_uuid=char.uuid, name="近景", count=1)
    checker = SimpleNamespace(
        similarity=Mock(return_value=FaceCheckResult(False, None, "no face")))
    host = SimpleNamespace(start_generation=Mock(), stop_generation=Mock())
    runner = BatchRunner(
        host, None, jobs_path=tmp_path / "jobs.json", face_checker=checker)
    runner._char = char
    job = _Job(scene, 0)
    runner._cur = job
    assert runner._check_face(job, [tmp_path / "candidate.png"]) == "failed"
    assert job.retry_count == 0


def test_lora_trigger_merge_and_dedup(tmp_path):
    from utils.lora_triggers import merge_triggers

    prompt = merge_triggers("1girl", ["hero[mikasa ackerman]"])
    assert prompt == "mikasa ackerman, 1girl"
    assert merge_triggers(prompt, ["hero[mikasa ackerman]"]) == prompt
    sidecar = tmp_path / "hero.txt"
    sidecar.write_text("masterpiece, hero", encoding="utf-8")
    merged = merge_triggers(
        "1girl", ["hero"], [str(tmp_path / "hero.safetensors")])
    assert merged == "masterpiece, hero, 1girl"


def test_batch_reference_pose_is_pushed_to_host(tmp_path):
    from utils.batch_runner import BatchRunner, _Job
    from utils.scene_library import Scene

    front = tmp_path / "front.png"
    side = tmp_path / "side.png"
    front.write_bytes(b"front")
    side.write_bytes(b"side")
    char = Character(name="甲", ref_image_path=str(front))
    char.reference_images.append({"path": str(side), "pose": "side"})
    scene = Scene(character_uuid=char.uuid, name="侧脸", pose_prompt="side profile")
    host = SimpleNamespace(txt_prompt=None, txt_neg=None, combo_res=None,
                           spin_steps=None, spin_seed=None, scale_cfg=None)
    runner = BatchRunner(
        host, None, jobs_path=tmp_path / "jobs.json",
        face_checker=SimpleNamespace())
    runner._char = char
    runner._cur = _Job(scene, 0)
    runner._push_params(char, scene)
    assert host.ipa_image_path == str(side)
    assert host.ref_image_path == str(side)


def test_multi_reference_export_import_roundtrip(tmp_path):
    from PIL import Image
    from utils.character_library import CharacterLibrary

    source = tmp_path / "source"
    exported = tmp_path / "exported"
    imported = tmp_path / "imported"
    source.mkdir()
    refs = []
    for name, pose in (("front.png", "front"), ("side.png", "side")):
        path = source / name
        Image.new("RGB", (12, 12), "white").save(path)
        refs.append({"path": str(path), "pose": pose})
    library = CharacterLibrary(str(source / "library"))
    char = Character(name="甲", reference_images=refs)
    assert library.save_character(char)
    assert library.export_character(char.uuid, str(exported))
    loaded = CharacterLibrary(str(imported)).import_character(str(exported))
    assert loaded is not None
    assert [ref["pose"] for ref in loaded.reference_images] == ["front", "side"]
    assert all(Path(ref["path"]).is_file() for ref in loaded.reference_images)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    with tempfile.TemporaryDirectory() as directory:
        for name, fn in tests:
            case_dir = Path(directory) / name
            case_dir.mkdir()
            fn(case_dir)
            print(f"PASS {name}")
    print(f"CHARACTER_CONSISTENCY_OK: {len(tests)} tests")
