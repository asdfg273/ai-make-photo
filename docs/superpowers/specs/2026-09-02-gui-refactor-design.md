# GUI 全面重构设计文档 — AI 绘画工作站 v6.0

- 日期：2026-09-02
- 范围：PyQt6 GUI 层整体重构（界面布局、主题、统一画廊），业务逻辑层零改动
- 伴随变更：版本号 v5.0 → v6.0

## 1. 背景与目标

当前 GUI 由单体 `ui/ui_builder.py`（2626 行，`UIBuilderMixin`）支撑：左侧固定 500px 的 7 个 Tab（基础/动画/图生图/LoRA/ControlNet/高级/X-Y 矩阵），右侧图片/视频两套面板的 `QStackedWidget`。样式为手写 `DARK_STYLE`（纯黑白 xAI 风）+ 散落在多个文件中的几十处硬编码色值，风格不统一，用户不满意。

**需求**：

1. 界面清晰易懂
2. 保留画廊等要素
3. 确保可扩展性
4. 确保功能组件的可使用性（控件命名契约不破）
5. 与现有 GUI 明显不同
6. 引入 QDarkStyleSheet（venv 已安装）
7. 统一图片/动画画廊（画廊页加媒体切换按钮），同时保留现有搜索与过滤功能
8. 版本号升级 v5.0 → v6.0

**已确认的关键决策**：

| 决策点 | 结论 |
|---|---|
| 布局形态 | 工作台三栏式：左侧导航栏 + 中央工作区 + 右侧参数面板 |
| 画廊形态 | 两者都要：底部常驻胶片条 + 独立画廊大页面 |
| 页面组织 | 按工作流分 4 页：文生图 / 图生图 / 动画 / 画廊；LoRA、ControlNet、高级选项为页内可折叠分组（默认折叠）；X-Y 矩阵收进文生图页 |
| 主题 | QDarkStyleSheet 经典深蓝灰 palette + 少量项目自定义覆盖 |
| 重构策略 | 方案 A：外壳重构 + 页面化迁移（新建模块化 UI 包，逐页搬移控件创建代码，控件名不变） |

## 2. 硬约束：控件命名契约

业务逻辑 mixin（`utils/app_generation.py`、`utils/app_events.py` 等）通过 `self.控件名` 直接访问 UI（如 `self.btn_gen`、`self.combo_model`），且 `ui_builder.py` 中存在一批兼容别名。

- 所有控件属性名原样保留，一个不改
- 兼容别名（`btn_gen`、`btn_stop`、`scale_str`、`scale_hires`、`progress_total`、`progress`、`preview_canvas`、`pose_canvas`、`combo_loras`、`scale_loras` 等）集中到一处安装
- 新增 `ui/contracts.py` 列出全部必需控件名，启动时自检
- `self.gallery` 及其信号（`image_selected`、`image_deleted`、`apply_params_signal`、`reuse_params_signal`、`send_to_i2i_signal`、`send_to_face_signal`、`send_to_editor_signal`）与方法（`add_image`、`reload_from_dir`）签名不变

## 3. 架构

```
ui/
├── theme.py               # 主题中枢：加载 qdarkstyle 深蓝灰 palette
│                          #   + 项目自定义 QSS 覆盖层 + 色板常量（组件一律从此取色）
├── shell.py               # 新主窗口外壳：NavRail + QStackedWidget + 右侧参数面板
│                          #   + 底部胶片条 + 状态栏 + 菜单；提供 setup_ui()
├── nav.py                 # NavRail 导航栏：读 PAGES 注册表自动生成按钮
├── widgets/
│   ├── collapsible.py     # CollapsibleSection 可折叠分组
│   └── filmstrip.py       # FilmStrip 底部最近生成缩略图横条
├── pages/
│   ├── base.py            # PageBase：页面元数据(id/标题/图标) + 控件挂载
│   ├── txt2img_page.py    # 文生图页
│   ├── img2img_page.py    # 图生图页
│   ├── video_page.py      # 动画页
│   └── gallery_page.py    # 统一画廊页
├── gallery_panel.py       # 现有画廊核心，增加媒体类型过滤后复用
└── ui_builder.py          # 迁移完成后删除（git 历史可查）
```

**关键机制**：

1. **页面注册表**：`PAGES = [Txt2ImgPage, Img2ImgPage, VideoPage, GalleryPage]`，NavRail 与中央 Stacked 均由此生成。新增功能 = 新建 page 文件 + 注册表加一行。
2. **控件契约保护**：页面构建把控件按现有属性名挂到主窗口；`ui/contracts.py` 启动自检，缺失立即日志报警。
3. **业务层零改动**：`EventMixin` / `GenerationMixin` / `PresetManagerMixin` / `TooltipMixin` / `VideoPanelMixin` 原样保留；新 shell 是 `UIBuilderMixin` 的替代实现，同样提供 `setup_ui()`。
4. **主题单点**：`theme.apply(app)` 一处调用，未来加主题只改此文件。

## 4. 页面布局

```
┌────────┬──────────────────────────────────┬─────────────────────┐
│ 导航栏 │        中央工作区（随页面切换）      │   右侧参数面板        │
│ 64px   │                                  │   （可折叠，320px）    │
│ 文生图  │  创作页：大预览图 + 生成信息         │  当前页的参数控件      │
│ 图生图  │  画廊页：网格/大图浏览              │  （滚动区+折叠分组）   │
│ 动 画  │                                  │                     │
│ 画 廊  │                                  │                     │
├────────┴──────────────────────────────────┴─────────────────────┤
│ 胶片条：最近生成的图片/动画缩略图（高约 110px，可隐藏）               │
├──────────────────────────────────────────────────────────────────┤
│ 状态栏：状态消息 | 进度条 | 显存 | 输出目录按钮                       │
└──────────────────────────────────────────────────────────────────┘
```

- **文生图页**：右侧参数面板从上到下——模型选择、Prompt/负面 Prompt、尺寸/步数/CFG 核心参数；折叠分组 LoRA → ControlNet → 高级选项 → X-Y 矩阵（默认折叠）；「生成/停止」大按钮固定面板底部。中央为预览大图 + 生成进度。
- **图生图页**：同结构，顶部多参考图拖放/选择区（含蒙版）与 i2i 专属参数，折叠分组复用。
- **动画页**：现有动画参数组原样迁移到参数面板；中央为视频预览（现有 `video_right_panel` 播放器能力搬入）。选页即选模式，不再有隐式模式切换。
- **画廊页**：见第 6 节。
- **胶片条**：任何页面可见，最近 ~20 个生成结果混排，图标区分图片/动画；点击跳转画廊页并选中。

## 5. 主题方案

- **基底**：`qdarkstyle` 的 `dark` palette（深蓝灰 + 蓝色强调），`theme.apply(app)` 一处调用。
- **自定义覆盖层**（`theme.py` 内一小段 QSS，只做三件事）：
  1. 品牌微调：导航选中态、生成大按钮强调；圆角统一 6px；间距 8px 网格。
  2. 清场旧样式：移除 `gallery_panel.py`、`extension_market.py`、`disclaimer.py` 等文件中的全部硬编码 `setStyleSheet` 色值，改走 `theme.py` 色板常量。
  3. 组件语义类：Qt 动态属性打标（`accent="true"`、`role="hint"`、`kind="primary"`），QSS 集中定义。
- **字体**：`"Segoe UI", "Microsoft YaHei", sans-serif`，正文 13-14px；日志/参数区用等宽字体。
- **图标**：Qt 内置标准图标 + 简单 Unicode 符号，不引入外部图标库。
- **深色一致性**：splash、免责声明、扩展市场等对话框统一走 `theme.py`。

## 6. 统一画廊

- **GalleryPanel 增加媒体类型维度**：`set_media_filter("image" | "video" | "all")`；按扩展名分类（`.png/.jpg/.webp` vs `.mp4/.gif/.webm`），动画缩略图右上角播放角标。
- **画廊页顶部工具条**：
  `[图片|动画|全部] | 🔍 搜索框 [✖] | ⭐仅收藏 | 废 | 📋元数据 | N 张`
  - 媒体切换三态按钮组为新增
  - 现有功能全部保留：搜索框（300ms 防抖，搜文件名+prompt 关键字）、清空、仅收藏、废弃内容过滤、元数据面板开关、数量统计；收藏持久化（`gallery_favs.json`）、多选、右键菜单、双击大图不动
  - 媒体类型过滤与现有关键词/收藏/废弃过滤叠加生效
- **选中后右侧**：图片 → 大图 + 现有 `MetadataPanel`；动画 → 内嵌播放器（复用动画页播放组件）+ 基本信息。
- **上下文菜单按类型适配**：图片保留全部现有动作（收藏/删除/回填参数/发送图生图/修脸/编辑器）；动画提供播放/保存副本/打开目录/删除。
- **统一数据源**：扫描 `output/`（图片）+ 动画输出目录，合并同一网格，切换按钮仅过滤视图。
- **胶片条联动**：与画廊页共享数据模型，新生成结果一处调用两处更新。
- **契约保留**：`self.gallery` 属性名与全部信号、方法签名不变，调用方零改动。

## 7. 错误处理

1. **启动自检**：shell 构建完成后对照 `contracts.py` 逐项 `hasattr` 检查，缺失弹警告对话框 + 日志记录具体缺失项。
2. **页面构建隔离**：每页构建包在 try/except 中，单页失败不影响其他页，失败页显示错误占位 + 日志堆栈。
3. **主题兜底**：qdarkstyle 加载失败回退 Qt Fusion 深色，界面永远可用。
4. **画廊扫描兜底**：目录不存在/无权限/缩略图解码失败显示占位图，不中断扫描。

## 8. 验证方案

1. **契约测试**：`tests/` 新增 UI 契约测试——`QApplication` offscreen 模式启动 shell，断言必需控件存在、别名齐全、页面可切换。
2. **冒烟脚本**：`scripts/smoke_ui.py` 一键启动 GUI（跳过模型加载），人工走查四页 + 胶片条 + 画廊切换。
3. **逐页迁移验证**：顺序 文生图 → 图生图 → 动画 → 画廊；每页迁移后跑契约测试 + 冒烟；旧 `ui_builder.py` 全程保留作对照，全绿再删。
4. **git 分阶段提交**：主题层 → 外壳骨架 → 逐页迁移 → 统一画廊 → 删除旧文件 → 版本号 6.0，每阶段一个 commit。

## 9. 版本号 v5.0 → v6.0

需修改的位置（仅版本字符串，数字 5.0 的参数值不动）：

- `main.py:3` 文件头注释
- `main.py:723` `app.setApplicationVersion("5.0")`
- `ui/disclaimer.py:38` 欢迎标题
- `ui/splash.py:69` 启动画面副标题
- `ui/ui_builder.py:43` 窗口标题（随旧文件迁移到新 shell）
- `ui/ui_builder.py:1914` 状态栏就绪消息（迁移）
- `ui/ui_builder.py:2419` 关于对话框（迁移）
- `README.md`、`项目说明.txt` 中的版本描述（如存在）

## 10. 非目标（YAGNI）

- 不改动任何生成/推理/模型加载逻辑
- 不引入外部图标库、不引入新的 GUI 框架依赖（qdarkstyle 已安装）
- 不做主题切换功能（本期单主题，结构上预留单点）
- 不重写业务 mixin
