# 推理性能与素材管理增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有桌面生成器增加 DiT 低显存推理、步间缓存、可暂停 FIFO 队列、模型预览和画廊语义检索。

**Architecture:** 每个能力由独立纯 Python 模块负责，Qt 只做展示与主线程调度。量化与 CLIP 使用可选本地后端；所有新功能经现有 `GenerationMixin`、单文件加载器和 `GalleryPanel` 的窄接口接入。

**Tech Stack:** Python 3、PyQt6、PyTorch、diffusers 0.40、transformers、Pillow、项目脚本式测试。

## Global Constraints

- 不自动下载 DiT 或 CLIP 大模型；缺少本地后端时明确报错。
- 默认量化与步间缓存关闭，不改变既有 SDXL 输出。
- 角色批量 `_batch_mode` 绕过普通 FIFO；Qt 控件只在主线程读写。
- 不声称未经真实权重验证的显存或加速数字。
- 精准暂存，工作区已有无关修改不得进入本轮提交。

---

### Task 1: DiT 量化与分组卸载

**Files:**
- Create: `core/dit_optimization.py`
- Modify: `core/loaders/base.py`, `core/loaders/single_file.py`, `core/model_manager.py`, `core/config_manager.py`, `core/arch/registry.py`, `utils/app_generation.py`, `ui/settings_dialog.py`
- Test: `tests/test_inference_productivity.py`

**Interfaces:**
- Produces: `quantization_config(mode: str)`, `apply_group_offload(pipe, enabled: bool, blocks: int, device: str) -> bool`, `optimization_signature(options: dict) -> tuple`。
- Consumes: `LoadContext.optimization: dict` 与 `ModelManager.inference_options: dict`。

- [ ] **Step 1: 写失败测试**：断言 FP8 映射只包含 `transformer`，NVFP4 使用 `NVFP4`，卸载成功标记 `_dit_group_offloaded=True`，相同选项签名一致、精度变化签名不同。
- [ ] **Step 2: 运行测试并确认因模块/字段缺失失败**：`venv/Scripts/python.exe tests/test_inference_productivity.py`。
- [ ] **Step 3: 实现配置工厂与加载接线**：`PipelineQuantizationConfig(quant_mapping={"transformer": config})`；`apply_group_offloading(..., offload_type="block_level", num_blocks_per_group=blocks, use_stream=True)`；加载器跳过重复通用 offload。
- [ ] **Step 4: 接设置与模型重载签名**：新增 `dit_quantization`、`dit_group_offload`、`dit_offload_blocks`；模型加载前由宿主注入 `ai.inference_options`。
- [ ] **Step 5: 运行测试并精准提交**：测试 PASS 后仅暂存本任务文件，提交 `feat(inference): 增加 DiT 量化与分组卸载`。

### Task 2: TeaCache 风格步间缓存

**Files:**
- Create: `utils/step_cache.py`
- Modify: `core/config_manager.py`, `utils/app_generation.py`, `ui/settings_dialog.py`
- Test: `tests/test_inference_productivity.py`

**Interfaces:**
- Produces: `configure_step_cache(pipe, arch_id: str, enabled: bool, threshold: float, max_skips: int = 2)`, `reset_step_cache()`, `step_cache_stats() -> dict`。
- Consumes: `pipe.transformer`（Flux）或 `pipe.unet`（SDXL）的 `forward`。

- [ ] **Step 1: 写失败测试**：用小型真实 `torch.nn.Module` 验证首步计算、相似输入复用、连续跳过上限、`reset_step_cache()` 后重新计算和禁用后恢复原 forward。
- [ ] **Step 2: 运行并确认失败**：同一测试命令，预期模块不存在。
- [ ] **Step 3: 实现可撤销 forward 包装**：相对 L1 使用 detach 抽样张量；缓存输出并记录 `computed/reused/skips`；不得复制完整 latent。
- [ ] **Step 4: 接入生成**：模型加载后配置，单张生成前复位，清理时报告统计；设置页增加开关和阈值。
- [ ] **Step 5: 回归并精准提交**：新增测试与既有加载测试 PASS，提交 `feat(inference): 增加可复位步间缓存`。

### Task 3: FIFO 队列与协作暂停

**Files:**
- Create: `utils/generation_queue.py`, `ui/generation_queue_dialog.py`
- Modify: `utils/app_generation.py`, `ui/core_panel.py`, `main.py`
- Test: `tests/test_inference_productivity.py`

**Interfaces:**
- Produces: `GenerationQueue.enqueue(snapshot: dict) -> str`, `update(job_id, values)`, `promote(job_id)`, `remove(job_id)`, `set_paused(bool)`, `wait_if_paused()`。
- Consumes: 原生成入口改名 `_start_generation_immediate()`；普通 `start_generation()` 负责快照与调度。

- [ ] **Step 1: 写失败测试**：覆盖 FIFO、置顶、仅编辑 pending、暂停时工作线程等待、恢复唤醒、取消唤醒和批量直接执行。
- [ ] **Step 2: 运行并确认失败**：预期队列模块不存在。
- [ ] **Step 3: 实现线程安全队列与 QTimer 调度**：运行项完成后再取下一项；暂停用 `threading.Condition`，不阻塞 Qt。
- [ ] **Step 4: 实现管理对话框与主按钮**：待运行项可编辑 prompt/negative/steps/CFG/seed/count，支持置顶和删除；新增“队列”按钮。
- [ ] **Step 5: 回归并精准提交**：队列测试与批量工作流测试 PASS，提交 `feat(generation): 增加可编辑暂停队列`。

### Task 4: 模型示例图预览

**Files:**
- Create: `utils/model_previews.py`
- Modify: `utils/model_scanner.py`, `main.py`, `ui/core_panel.py`
- Test: `tests/test_inference_productivity.py`

**Interfaces:**
- Produces: `find_model_preview(model_path: str) -> str | None`；扫描项新增 `preview: str`。
- Consumes: `combo_model.currentData()["preview"]`。

- [ ] **Step 1: 写失败测试**：临时目录创建模型与 `name.preview.png`/`name.png`，断言优先级、无图返回 `None`、不会读取其他模型图片。
- [ ] **Step 2: 运行并确认失败**。
- [ ] **Step 3: 实现扫描与 Qt 图标/预览标签**：下拉图标 40×40，当前预览最大 112×112，保持宽高比；无图清空。
- [ ] **Step 4: 回归并精准提交**：模型扫描和 UI 契约 PASS，提交 `feat(models): 显示本地示例图预览`。

### Task 5: 画廊全文与 CLIP 相似检索

**Files:**
- Create: `utils/gallery_search.py`
- Modify: `ui/gallery_panel.py`
- Test: `tests/test_inference_productivity.py`

**Interfaces:**
- Produces: `match_all_terms(query: str, filename: str, prompt: str) -> bool`, `cosine_rank(query_vector, vectors: dict) -> dict`, `ClipSearchEngine.search_text(text, paths)`, `search_image(path, paths)`。
- Consumes: 本地 CLIP、`data/gallery_clip_index.json`、现有 `_all_items`。

- [ ] **Step 1: 写失败测试**：覆盖多词 AND、prompt 大小写、余弦降序、文件签名变化重编码、缺模型明确异常；用假编码器替代真实 CLIP。
- [ ] **Step 2: 运行并确认失败**。
- [ ] **Step 3: 实现本地懒加载索引**：`local_files_only=True`，向量 L2 归一化，JSON 原子写；后台线程完成后 Qt 信号更新排序。
- [ ] **Step 4: 接 UI**：新增“语义”按钮和“查找相似图”菜单；语义模式跳过子串过滤并按分数排序，清空搜索恢复普通模式。
- [ ] **Step 5: 回归并精准提交**：新测试与画廊统一测试 PASS，提交 `feat(gallery): 增加全文与相似图检索`。

### Task 6: 文档与隔离验收

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-10-29-inference-productivity-design.md`（仅在实现与约定有差异时）
- Test: `tests/test_inference_productivity.py`

- [ ] **Step 1: 更新使用文档**：说明模型示例图命名、FP8/NVFP4 后端要求、CLIP 本地目录、队列暂停边界与真实验收限制。
- [ ] **Step 2: 运行完整相关回归**：新测试、loaders、batch workflow、gallery unified、UI contract、UI i18n、V6 features，并编译所有改动 Python 文件。
- [ ] **Step 3: 干净快照验证**：`git archive HEAD` 后使用项目 `venv` 重跑上述命令；检查 `git diff --check` 和暂存区。
- [ ] **Step 4: 精准提交并报告**：提交 `docs(inference): 补充性能与检索使用说明`，报告测试证据和未执行的真实模型项目。
