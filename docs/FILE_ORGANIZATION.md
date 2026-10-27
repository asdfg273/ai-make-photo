# 工程文件归类说明

更新日期：2026-10-27

本次整理的目标是减少项目根目录中的散落文档、预览图、调试脚本和测试输出，同时保持正式源码、模型、配置及用户输出目录不变。移动操作仅改变存放位置，不删除原始资料。

## 一、文档与设计资料

| 原位置 | 新位置 | 用途 |
|---|---|---|
| `aiUsemono.md` | `docs/design/xai_inspiration.md` | xAI 风格设计分析；文件名已改为与内容一致 |
| `make_new.txt` | `docs/audits/ui_structure_audit.txt` | UI 结构、接口约束与交接审计 |
| `结构.txt` | `docs/project_structure.txt` | 项目目录结构说明 |
| `操作方法.mp4` | `docs/media/操作方法.mp4` | 本地操作演示视频 |

说明：`docs/project_structure.txt` 和 UI 审计文档中的文件位置说明已同步更新。视频仍按 `.gitignore` 的 `*.mp4` 规则保留为本地文件，不纳入 Git。

## 二、README 预览图

| 原位置 | 新位置 | 引用处理 |
|---|---|---|
| `postprocess_nodes_preview.png` | `docs/images/postprocess_nodes_preview.png` | README 后处理节点效果图链接已更新 |
| `regional_preview.png` | `docs/images/regional_preview.png` | README 区域提示词效果图链接与正文路径已更新 |

README 的“文件结构”章节也已改为展示新的 `docs/` 分类。

## 三、手动调试脚本

| 原位置 | 新位置 | 用途 |
|---|---|---|
| `_scratch_dbg.py` | `tests/manual/debug_ipa_refs.py` | 单独诊断 IP-Adapter 多参考图界面测试 |
| `_scratch_ui_preview.py` | `tests/manual/ui_preview.py` | 离屏渲染主界面及各功能页截图 |

两个脚本均已改为通过 `Path(__file__).resolve().parents[2]` 定位项目根目录，不再依赖脚本位于根目录。

在项目根目录运行：

```bash
venv/Scripts/python.exe tests/manual/debug_ipa_refs.py
venv/Scripts/python.exe tests/manual/ui_preview.py
```

## 四、测试输出与临时产物

| 原位置 | 新位置 |
|---|---|
| `_scratch_out.txt` | `tests/artifacts/logs/scratch_output.txt` |
| `_scratch_logs/` | `tests/artifacts/ui_preview/` |
| `_tmp_rife/` | `tests/artifacts/tmp/rife/` |
| `_tmp_test/` | `tests/artifacts/tmp/test/` |

`tests/artifacts/` 已加入 `.gitignore`，用于保存本地测试日志、截图和临时目录，避免测试产物进入版本控制。`tests/manual/ui_preview.py` 后续生成的新截图也会写入 `tests/artifacts/ui_preview/`。

## 五、保持原位的目录与文件

以下内容具有明确的运行、配置或数据用途，本次没有移动：

- 正式源码：`main.py`、`core/`、`ui/`、`nodes/`、`utils/`、`workflow/`、`photo_turn/`。
- 测试代码：`tests/test_*.py` 及既有测试辅助文件。
- 模型与依赖：`models/`、`models_cache/`、`controlnets/`、`loras/`、`weights/`、`third_party/`、`venv/`。
- 运行配置：`app_config.json`、`config.json`、`weight.json`。
- 用户数据与输出：`data/`、`output/`、`my_output/`、`photo/`、`logs/`。
- 用途或归属需要进一步确认的根目录内容，例如 `z/`、`NVIDIA Corporation/`，未贸然搬动。

## 六、后续归类约定

1. 长期维护的说明放入 `docs/`，设计资料、审计资料、图片和媒体分别使用对应子目录。
2. 自动化测试留在 `tests/test_*.py`；需要人工执行的诊断脚本放入 `tests/manual/`。
3. 测试生成的日志、截图和临时文件统一写入 `tests/artifacts/`，不要放在项目根目录。
4. 移动被引用的文件时，同步检查 README、脚本和代码中的路径。
