# Character Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为角色卡增加多参考图、脸部一致性重 roll、持久提示词特征缓存和 LoRA 触发词自动注入。

**Architecture:** `Character` 持有可序列化参考图与校验配置；独立参考图选择器、人脸校验器和 LoRA 工具保持可测试。`BatchRunner` 负责逐张选图及有限重试，shell 只保存当前角色上下文并复用现有生成链。

**Tech Stack:** Python 3、PyQt6、Pillow、NumPy、可选 InsightFace、项目独立测试脚本。

## Global Constraints

- 工作目录固定为 `E:\pythonprojects\ai  make photo`，分支 `refactor/lazy-load`。
- 保留全部既有未提交修改；含既有修改的文件必须从 HEAD 构造本轮 blob 精准暂存。
- 旧 `ref_image_path` 必须继续可用；不序列化图像张量，不引入无限重试。
- 不在测试下载或加载真实人脸、CLIP、Qwen 模型；模型接口全部注入 fake。
- 默认阈值 0.55、最多重试 1 次，允许范围分别为 0–1 和 0–3。

---

### Task 1: 角色参考图数据与特征缓存

**Files:**
- Modify: `utils/character_library.py`
- Create: `utils/character_references.py`
- Create: `tests/test_character_consistency.py`

**Interfaces:**
- Produces: `normalize_references(character)`、`select_reference(character, pose_prompt, index)`、`image_signature(path)`、`cached_feature(character, path)`、`store_feature(character, path, text)`。
- `Character.reference_images` 每项含 `path/pose`；`face_check_enabled/face_similarity_threshold/face_max_retries` 有默认值。

- [ ] **Step 1: 写旧角色迁移、姿势选择和缓存签名失败测试**

```python
def test_reference_migration_and_pose_selection(tmp_path):
    front = tmp_path / "front.png"; front.write_bytes(b"front")
    side = tmp_path / "side.png"; side.write_bytes(b"side")
    char = Character(name="甲", ref_image_path=str(front))
    char.reference_images.append({"path": str(side), "pose": "side"})
    refs = normalize_references(char)
    assert refs[0]["pose"] == "front"
    assert select_reference(char, "side profile", 0)["path"] == str(side)
    assert select_reference(char, "walking", 1)["path"] == str(side)


def test_feature_cache_hits_and_invalidates(tmp_path):
    image = tmp_path / "ref.png"; image.write_bytes(b"a")
    char = Character(name="甲", ref_image_path=str(image))
    assert cached_feature(char, str(image)) == ""
    store_feature(char, str(image), "silver hair")
    assert cached_feature(char, str(image)) == "silver hair"
    image.write_bytes(b"changed")
    assert cached_feature(char, str(image)) == ""
```

- [ ] **Step 2: 运行测试确认缺少模块而失败**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py -q`
Expected: FAIL，`utils.character_references` 不存在。

- [ ] **Step 3: 实现纯数据工具与角色字段**

规范化最多 4 张有效路径，姿势限定 `front/side/full/extra`；首图同步 `ref_image_path`。签名使用绝对路径、`st_mtime_ns` 和 `st_size`。缓存结构为 `{"version": 1, "refs": {pose: signature+text}}`，同一路径按签名查找。`Character.from_dict` 复制输入并忽略缓存空值，保存/加载后调用规范化。

- [ ] **Step 4: 跑测试并提交**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py -q`
Expected: PASS。
Commit: `feat(character): 增加多参考图与特征缓存结构`

### Task 2: 多参考图编辑与应用

**Files:**
- Modify: `ui/pages/characters_page.py`
- Modify: `tests/test_character_consistency.py`

**Interfaces:**
- Consumes: Task 1 的 `normalize_references/select_reference`。
- Produces: 编辑器 `ref_list`、姿势选择、添加/移除；shell `_active_character` 和 `_character_library`。

- [ ] **Step 1: 写 Qt 离屏失败测试**

```python
def test_editor_supports_four_reference_poses(qapp, tmp_path):
    editor = CharacterEditor()
    char = Character(name="甲")
    editor.load(char)
    with patch.object(QFileDialog, "getOpenFileNames", return_value=([str(tmp_path / n) for n in ("a.png", "b.png", "c.png")], "")):
        for name in ("a.png", "b.png", "c.png"):
            (tmp_path / name).write_bytes(b"x")
        editor.browse_refs()
    assert editor.ref_list.count() == 3
    assert [r["pose"] for r in char.reference_images] == ["front", "side", "full"]
    editor.close()
```

- [ ] **Step 2: 运行并确认旧编辑器无多图接口**

Expected: FAIL，`browse_refs/ref_list` 不存在。

- [ ] **Step 3: 实现编辑控件和应用桥接**

保留 `browse_ref` 兼容入口，新增多文件选择、列表、姿势 combo、移除；参考图变化清空缩略图。应用角色时设置 `_active_character`、`_character_library`，首图写入 `ipa_image_path` 并沿用 `set_reference_image`。保存时调用 `preextract_character_features(shell, char)`，提取失败只记日志。

- [ ] **Step 4: 跑角色测试并精准提交**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py tests/test_character_scene.py -q`
Commit: `feat(character): 支持多姿势参考图编辑与应用`

### Task 3: 生成链复用角色持久特征

**Files:**
- Modify: `utils/app_generation.py`
- Create: `utils/character_features.py`
- Modify: `tests/test_character_consistency.py`

**Interfaces:**
- Produces: `preextract_character_features(host, char)`、`feature_for_generation(host, ref_img)`。
- Consumes: shell `_gt_extract_character_features`、`_active_character`、`_character_library`。

- [ ] **Step 1: 写持久缓存命中失败测试**

```python
def test_generation_uses_persistent_feature_before_ai(tmp_path):
    image = tmp_path / "ref.png"; image.write_bytes(b"x")
    char = Character(name="甲", ref_image_path=str(image))
    store_feature(char, str(image), "cached tags")
    host = SimpleNamespace(_active_character=char, _gt_extract_character_features=Mock())
    assert feature_for_generation(host, str(image)) == "cached tags"
    host._gt_extract_character_features.assert_not_called()
```

- [ ] **Step 2: 运行确认失败后实现**

`feature_for_generation` 命中直接返回；未命中调用现有提取器并非空回填、调用角色库 `save_character`。`preextract_character_features` 遍历有效参考图并复用同一函数。`_gt_prepare_context` 只把原调用替换为 `feature_for_generation(self, ref_img)`。

- [ ] **Step 3: 测试并精准提交**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py -q`
Commit: `feat(character): 持久化并复用提示词视觉特征`

### Task 4: 脸部一致性校验与有限重试

**Files:**
- Create: `utils/face_consistency.py`
- Modify: `utils/batch_runner.py`
- Modify: `tests/test_character_consistency.py`

**Interfaces:**
- Produces: `FaceConsistencyChecker.similarity(generated_path, reference_paths) -> FaceCheckResult`；结果含 `ok/score/error`。
- `BatchRunner(..., face_checker=None)` 可注入 fake。

- [ ] **Step 1: 写低分重 roll 与通过失败测试**

```python
def test_batch_low_similarity_rerolls_same_job_once(tmp_path):
    char = Character(name="甲", face_max_retries=1)
    scene = Scene(character_uuid=char.uuid, name="近景", count=1)
    checker = Mock()
    checker.similarity.side_effect = [FaceCheckResult(False, 0.2, ""), FaceCheckResult(True, 0.8, "")]
    runner = BatchRunner(_Host(), None, jobs_path=tmp_path / "jobs.json", face_checker=checker)
    runner._char = char
    job = _Job(scene, 0)
    assert runner._check_face(job, [tmp_path / "candidate.png"]) == "retry"
    assert runner._cur is job and job.retry_count == 1
    assert runner._check_face(job, [tmp_path / "candidate.png"]) == "done"
```

另测 `ok=False,error="no face"` 返回 `failed` 且不重试，高分直接完成。

- [ ] **Step 2: 运行确认失败后实现校验器**

懒导入 InsightFace，提取最大脸 embedding；余弦值映射到 0–1。所有异常转为 `FaceCheckResult(False, None, error)`，不在模块导入时加载模型。

- [ ] **Step 3: 接入批量收图与重试**

`_Job` 增加 `retry_count`。收图先调用 `_check_face`：`done` 走原成功；`retry` 保留 `_cur`、更新清单 `retrying`、换种子并重新 `start_generation`；`failed` 保留候选路径并走失败。校验阈值与开关取当前角色。

- [ ] **Step 4: 跑 fake 测试并精准提交**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py tests/test_batch_workflow.py -q`
Commit: `feat(batch): 增加脸部相似度校验与自动重试`

### Task 5: LoRA 触发词自动注入

**Files:**
- Create: `utils/lora_triggers.py`
- Modify: `utils/app_generation.py`
- Modify: `ui/pages/characters_page.py`
- Modify: `tests/test_character_consistency.py`

**Interfaces:**
- Produces: `trigger_for_display(text, lora_path="") -> str`、`merge_triggers(prompt, displays, lora_paths=None) -> str`。

- [ ] **Step 1: 写内联、旁文件与去重失败测试**

```python
def test_lora_trigger_merge_and_dedup(tmp_path):
    prompt = merge_triggers("1girl", ["hero[mikasa ackerman]"])
    assert prompt == "mikasa ackerman, 1girl"
    assert merge_triggers(prompt, ["hero[mikasa ackerman]"]) == prompt
    sidecar = tmp_path / "hero.txt"; sidecar.write_text("masterpiece, hero", encoding="utf-8")
    assert "masterpiece, hero" in merge_triggers("1girl", ["hero"], [str(tmp_path / "hero.safetensors")])
```

- [ ] **Step 2: 运行失败后实现统一工具**

方括号内联优先；旁文件先查实际 LoRA 同目录，再查既有 `loras/sdxl/sd1.5/sd15/root`。按完整触发词组去重，不修改源文件。

- [ ] **Step 3: 接入应用角色与生成准备**

应用角色成功选择 LoRA 后合并对应展示文本；`_gt_prepare_context` 在解析 `<lora:>` 前，用所有当前槽位和角色路径合并触发词。UI 只在应用角色时更新，普通生成仅更新上下文。

- [ ] **Step 4: 测试并精准提交**

Run: `venv/Scripts/python.exe -m pytest tests/test_character_consistency.py -q`
Commit: `feat(lora): 自动注入角色与槽位触发词`

### Task 6: 隔离回归与交付

**Files:** Verify only。

- [ ] **Step 1: 检查差异与编译**

Run: `git diff --check <base>..HEAD`，编译全部新增/修改 Python 文件。
Expected: exit 0。

- [ ] **Step 2: 导出 HEAD 干净快照并回归**

Run: `tests/test_character_consistency.py`、`tests/test_character_scene.py`、`tests/test_batch_workflow.py`、`tests/test_ui_contract.py`、`tests/test_ui_i18n.py`、`tests/test_v6_features.py`。
Expected: 所有脚本 exit 0；无真实模型下载。

- [ ] **Step 3: 清理并核对**

删除快照和精准暂存脚本，`git diff --cached --quiet`，确认其他工作区修改保留。报告真实测试范围与未执行的真实模型验收。
