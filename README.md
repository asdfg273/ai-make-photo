# **AI 绘画工作站 v6.0**

---

## 一、项目概述

**项目名称**：AI 绘画工作站 v6.0

**项目定位**：基于 Stable Diffusion 的专业级 AI 图像与视频生成工具，提供一站式 AI 创作体验。

**核心目标**：为用户提供直观、高效的 AI 图像和视频生成工具，支持文生图、图生图、局部重绘、人脸修复、视频生成等多种功能。

---

## 二、功能特性

### 2.1 核心功能

| 功能模块 | 功能描述 | 技术实现 |
|---------|---------|---------|
| **文生图 (Text2Img)** | 通过提示词生成图像 | StableDiffusionPipeline |
| **图生图 (Img2Img)** | 基于参考图生成新图像 | StableDiffusionImg2ImgPipeline |
| **局部重绘 (Inpaint)** | 对图像指定区域进行修复 | StableDiffusionInpaintPipeline |
| **视频生成 (AnimateDiff)** | 文生视频、图生视频、视频转绘 | AnimateDiff + Motion LoRA |
| **ControlNet** | 姿态/边缘/深度控制生成 | ControlNetModel + OpenPose/Canny/Depth |
| **IP-Adapter** | 角色一致性锁定 | IP-Adapter 模型 |
| **修图编辑器** | 图像调整、滤镜、画笔、裁剪 | PyQt6 绘图引擎 |
| **人脸修复 (ADetailer)** | 自动人脸检测与修复 | OpenCV + YOLO |
| **手部修复** | 自动手部检测与修复 | YOLOv8 手部检测 |

### 2.2 视频生成功能

| 生成模式 | 描述 |
|---------|------|
| **文生视频** | 直接用提示词生成视频，无需输入文件 |
| **图生视频** | 选择一张图片作为首帧，AI 延续动画 |
| **视频转绘** | 选择视频文件，AI 改变画风 |
| **提示词旅行** | 在不同帧使用不同提示词，制作剧情视频 |

### 2.3 扩展功能

- **多模型支持**：支持 SD 1.5 和 SDXL 模型，可识别加载 SD3 / Flux（部分功能）
- **多 LoRA 挂载**：支持同时加载多个 LoRA 插件，独立权重控制
- **Motion LoRA**：支持缩放、平移、旋转等运镜特效
- **采样器切换**：支持 Euler A、Euler、DPM++ 2M、DDIM、UniPC 等多种采样器
- **提示词模板**：内置多种风格提示词预设
- **AI 提示词改写**：自动将自然语言转换为专业提示词
- **X/Y 矩阵生成**：批量参数对比测试
- **Hires.fix**：高清修复放大
- **完整出图元数据**：每张图都写入 40+ 项生成参数（含模型 SHA256、LoRA、ControlNet、Hires、ADetailer、依赖版本），并附带 A1111 兼容参数串，见 2.4
- **水印**：支持文字 / 自定义图片水印，位置、透明度、大小可调，见 2.6
- **工作流引擎 + 节点编辑器**：简易模式的操作会自动编译成一张工作流图，
  高级用户可在画布上增删节点、连线、改参数并直接运行；21 个内置节点，
  见 2.5（菜单栏 → 🧩 工作流）
- **图像管理**：自动保存、历史记录、图片预览
- **视频管理**：视频预览、历史画廊、双击播放
- **v6.0 工作台**：统一媒体画廊（图片/视频切换）、底部胶片条单击回填参数、修前/修后对比滑条（文生图+图生图）、显存/内存常驻监控、设置中心（可改快捷键与默认值）、AI 回译对比（翻译→回译校验防幻觉）

### 2.4 出图元数据

每张输出图都会在 PNG 里写入完整元数据，方便复现、归档与排查。
实现集中在 `utils/image_metadata.py`（全项目唯一入口）。

| 类别 | 字段 |
|------|------|
| 提示词 | `prompt`（原始，含中文）、`prompt_final`（实际送模型的最终提示词）、`negative`、`negative_final`、`prompt_mode`、`clip_skip` |
| 模型 | `model`、`model_path`（真实磁盘路径）、`model_hash`（SHA256）、`architecture`、`vpred`、`zero_snr` |
| LoRA | `loras`（名称+权重）、`lora_count`、`lora_tags` |
| 条件控制 | `controlnet_type`、`controlnet_weight`、`pose_transfer_used`、`ip_adapter_variant`、`ip_adapter_weight` |
| 采样 | `sampler`、`scheduler`（调度器类名）、`steps`、`cfg`、`seed`、`width`、`height`、`size`、`batch_index`、`batch_total` |
| Hires.fix | `hires_enabled`、`hires_scale`、`hires_denoise`、`hires_upscaler`、`hires_size` |
| ADetailer | `adetailer_enabled`、`adetailer_face_strength`、`adetailer_hand`、`adetailer_hand_strength`、`adetailer_hand_blend` |
| 版本 | `version_app`、`version_diffusers`、`version_torch`、`version_transformers`、`version_python`、`version_cuda`、`version_gpu` |
| 其它 | `watermark`、`workflow`、`timestamp` |

- **模型哈希**按「路径+大小+修改时间」缓存到 `models_cache/file_hashes.json`，
  同一个底模只算一次（否则每张图都要重读 2~7GB，保存会卡好几秒）。
- 同时写入 A1111 兼容的 `parameters` 文本块，WebUI / Civitai 可直接识别。
- 任何一个字段取不到都会写成 `unknown`，**元数据采集失败绝不影响出图**。

### 2.5 工作流引擎与节点编辑器

简易模式与高级模式**共享同一个执行引擎**：你在界面上点的每一步，
都会被编译成一张工作流图；高级用户看到的、能改的就是这张图。

#### 两种模式的关系

```
界面操作                          自动生成的工作流图
文生图 → 选模型 → 提示词                  LoadCheckpoint
   → 开 LoRA            ──编译──▶        → LoadLoRA
   → 开 ControlNet                       → TextEncode / NegativeTextEncode
   → 开高清修复                           → Sampler → VAEDecode
   → 开 ADetailer                        → HiresFix → ADetailer → SaveImage
```

编译逻辑在 `workflow/builder.py`，它**复用 `_gt_prepare_context()`**——
也就是界面上那套参数解析规则（提示词解析/翻译、LoRA 标签、ControlNet 力度、
尺寸、种子…）只有一份。否则又是"两套规则、迟早不一致"。

#### 入口在哪（菜单栏 → 🧩 工作流）

| 菜单项 | 快捷键 | 作用 |
|--------|--------|------|
| 🧩 打开工作流编辑器… | `Ctrl+W` | 打开空的节点编辑器 |
| ✨ 用当前界面设置生成流程图 | — | 把当前设置编译成图并打开编辑器 |
| ▶️ **用工作流引擎生成** | `Ctrl+Shift+W` | 编译并**立即执行**（走 workflow 引擎） |
| ✅ 检查当前工作流 | — | 只做校验 + 资源清单，不执行 |
| 📂 打开工作流目录 | — | 打开 `workflows/`（保存的 JSON 都在这里） |

编辑器内工具栏：新建 / 打开 / 保存 / 另存为 · 从当前界面生成 / 载入界面设置 ·
校验 / 资源清单 / **运行** (`Ctrl+R`) / **只跑选中节点** (`Ctrl+Shift+R`) /
取消 / 清空状态 / 删除选中 / 适应窗口 · 显存策略下拉。

#### 用代码调用（不经过界面）

```python
from workflow import builder
from workflow.executor import run_workflow

# 1) 把当前界面状态编译成图
wf = builder.build_from_ui(win)          # win = 主窗口（提供 ai / config）

# 2) 跑它（返回 dict：ok / states / outputs / cached / failed_node / error）
res = run_workflow(wf, host=win)
print(res["summary"] if "summary" in res else res["ok"], res["outputs"])

# 3) 只重跑某个节点（其余用缓存）
from workflow.executor import Executor, make_context
ex = Executor(make_context(host=win, ai=win.ai))
ex.run(wf, only=["saveImage_1"])
```

也可以完全脱离界面跑（`host=None`，输出落 `photo/`）：

```python
from workflow.schema import Workflow, Node
from workflow.executor import run_workflow

wf = Workflow(name="无头文生图")
wf.add(Node("model_1", "LoadCheckpoint", {},
            {"path": "meinamix_v12Final.safetensors"}))
wf.add(Node("prompt_1", "TextEncode", {}, {"text": "1girl, cat ears"}))
wf.add(Node("neg_1", "NegativeTextEncode", {}, {"text": "lowres"}))
wf.add(Node("sampler_1", "Sampler", {},
            {"steps": 28, "cfg": 7.0, "seed": 123, "width": 512, "height": 768}))
wf.add(Node("decode_1", "VAEDecode", {}))
wf.add(Node("save_1", "SaveImage", {}, {"filename_prefix": "wf_"}))
wf.connect("model_1", "model", "prompt_1", "model")
wf.connect("model_1", "model", "neg_1", "model")
wf.connect("model_1", "model", "sampler_1", "model")
wf.connect("prompt_1", "conditioning", "sampler_1", "positive")
wf.connect("neg_1", "conditioning", "sampler_1", "negative")
wf.connect("model_1", "model", "decode_1", "model")
wf.connect("sampler_1", "latent", "decode_1", "latent")
wf.connect("decode_1", "image", "save_1", "image")

res = run_workflow(wf)        # preset="auto" 会按显存自动选档
```

#### 目录结构

```
workflow/            执行引擎
├── schema.py            节点协议（端口类型 / NodeSpec / ResourceRequirement）
├── values.py            端口上流动的具体值（ModelRef / Conditioning / Latent…）
├── registry.py          节点注册表（@node 装饰器）
├── graph.py             拓扑排序 + 运行前校验
├── executor.py          执行器
├── memory_policy.py     能力配置驱动的显存策略
├── cache.py             节点级缓存（"只重跑某节点"的基础）
├── resource_manager.py  资源清单（只读本地，不联网）
├── adapters.py          模型族适配器（SD15/SDXL/Anima/SD3/Flux/Video）
├── serializers.py       JSON 保存/加载/版本迁移
└── builder.py           界面状态 → 工作流图

nodes/               节点实现（21 个）
├── model_nodes.py       LoadCheckpoint / LoadVAE / LoadLoRA
├── text_nodes.py        TextEncode / NegativeTextEncode
├── sampling_nodes.py    Sampler / VAEEncode / VAEDecode
├── image_nodes.py       LoadImage / ImageResize / ImageCrop / MaskCreate / Inpaint / SaveImage
├── control_nodes.py     ControlNetPreprocess / ControlNetApply
├── adapter_nodes.py     IPAdapterApply
├── detailer_nodes.py    ADetailer
├── upscale_nodes.py     HiresFix / Upscale
└── video_nodes.py       VideoFrameBatch

ui/workflow_editor/  节点编辑器
├── canvas.py            画布：节点/端口/连线，类型检查，状态着色
├── palette.py           节点库（从注册表自动生成，带搜索）
├── param_panel.py       参数面板（从 ParamSpec 自动生成控件）
└── window.py            编辑器窗口 + 运行线程

workflows/           工作流 JSON（保存/加载）
```

#### 端口类型与连线规则

`MODEL` `VAE` `TEXT_COND` `IMAGE` `LATENT` `MASK` `CONTROL_IMAGE` `LORA`
`SCHEDULER` `NUMBER` `BOOLEAN` `METADATA`（另有 `ANY` 表示不校验的透传口）

- **类型不兼容直接拒绝**，不会自动转换。最常见的一条：
  `LATENT` 和 `IMAGE` 不互通，**必须显式经过 `VAEDecode` / `VAEEncode`**。
- 一个输入口只接一条线（再连会替换旧的）。
- 必填输入既没连线也没默认值 → 校验阶段就报错，不会跑到一半才崩。

#### 显存策略（能力配置，不是显卡名字）

执行器不再"看显卡型号"决定是否 offload，而是**汇总各节点的资源声明**
（`requires` / `estimated_vram_mb` / `supports_cpu_offload` / `supports_tiled`）
再叠加设备能力选档：

| 预设 | 适用 | 行为 |
|------|------|------|
| 节省显存 | 6~8GB | CPU offload + 分块，缓存小 |
| 平衡模式 | 8~16GB | 优先速度与稳定 |
| 高性能 | 16GB+ | bf16、不 offload、缓存更小、batch 4 |
| 自定义 | — | 手动指定 |

> 阈值按**名义容量**取值：标称 8GB 的卡实际报 7.93GiB，所以分界放在 7.0，
> 保证"8GB 卡"落在平衡档而不是节省档。

#### 工作流 JSON 格式

```json
{
  "schema_version": 1,
  "name": "示例流程",
  "nodes": [
    {"id": "model_1",  "type": "LoadCheckpoint", "inputs": {"path": "a.safetensors"}},
    {"id": "prompt_1", "type": "TextEncode",     "inputs": {"text": "portrait"}},
    {"id": "sampler_1","type": "Sampler",        "inputs": {"steps": 28, "cfg": 6.5, "seed": 123}}
  ],
  "links": [
    ["model_1", "model", "sampler_1", "model"],
    ["prompt_1", "conditioning", "sampler_1", "positive"]
  ]
}
```

- 节点参数写在 `"inputs"` 里即可（加载时会按节点声明自动归位到参数），
  这是对外的约定格式；内部会区分"连线"与"参数"。
- 连线也冗余存在 `links` 里，外部工具可直接读。
- 加载**不会**因为一个未知节点类型就整份作废：未知节点保留在图中待修复。
- `schema_version` 用于将来迁移；高于程序支持的版本会告警但尽力加载。
- 画布坐标存在 `metadata.ui_positions`，不污染节点参数。

#### 加一个新节点（这就是"加功能只需加节点"）

在 `nodes/` 下任意模块里写一个函数即可，**界面零改动**：

```python
from workflow.registry import node
from workflow.schema import DataType, ParamSpec, PortSpec, ResourceRequirement

@node("MyNode", category="图像",
      description="一句话说明",
      inputs={"image": PortSpec(DataType.IMAGE, "输入图")},
      outputs={"image": PortSpec(DataType.IMAGE, "输出图")},
      params={"strength": ParamSpec("float", default=0.5, label="强度",
                                    minimum=0.0, maximum=1.0, step=0.05)},
      resources=ResourceRequirement(requires=("cuda",), estimated_vram_mb=800,
                                    supports_cpu_offload=True, supports_tiled=True))
def run_my_node(ctx, node, inputs):
    img = inputs.get("image")          # 取输入
    v = float(node.params.get("strength", 0.5))
    ctx.check_cancel()                 # 长操作里检查取消
    ctx.status("处理中…")               # 用户可见进度
    return {"image": img}              # 键名必须与 outputs 一致
```

- 参数控件（滑块/下拉/勾选/文本框）由 `ParamSpec` **自动生成**，带上下限。
- `resources` 决定显存策略与资源清单，`asset:<key>` 会自动映射到扩展市场的下载入口。
- 重依赖（torch/diffusers）在函数**内部** import，避免拖慢界面启动。

#### 已实现与暂未做

**已实现**：图结构与校验、执行器（缓存 / 取消 / 逐节点状态 / 失败定位 /
只重跑某节点）、能力配置化显存策略、资源清单、模型族适配器、JSON 读写与版本号、
21 个节点、基础节点编辑器、简易模式自动生成图。

**暂未做**（按优先级）：
1. 把 `start_generation()` 的旧生成路径**逐步**迁到节点（目前是并行通路，
   视频/批处理/XY 矩阵仍走原实现）；
2. 区域提示词、节点循环、分块放大、参考图组合；
3. ControlNet 的 lineart / softedge / tile / segmentation 权重映射
   （`ControlNetPreprocess` 已能预处理这几种，但 `prepare_controlnet`
   目前只认 openpose / canny / depth，接 `ControlNetApply` 会明确报错）；
4. 用户自定义节点 / 插件机制（**刻意最后做**：任意 Python 节点会让调试、
   显存管理与错误定位迅速失控）；
5. ComfyUI 工作流格式兼容。

### 2.6 水印

「输出设置 → 💧 水印」中启用：

| 选项 | 说明 |
|------|------|
| 文字 | 任意文本，带描边（深浅背景下都清晰） |
| 图片 | 选择 PNG/JPG（建议带透明通道），会按画面宽度自适应缩放 |
| 位置 | 右下 / 左下 / 右上 / 左上 / 居中 / 平铺 |
| 透明度 | 0.05 ~ 1.0 |
| 大小 | 0.3 ~ 3.0 倍 |

- 同时配置文字与图片时**图片优先**；图片读取失败自动退回文字，两者都无效则不加。
- 水印在**统一保存入口**施加：主出图、放大、大图都会加；
  **XY 参数矩阵是诊断图，故意不加**（加会干扰参数对比）。
- 水印是**烧进位图**的，导出后无法移除 —— 建议保留一份无水印原图。
- 水印失败只记 warning 并保存原图，绝不让出图失败。

---

## 三、技术架构

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      UI 层 (PyQt6)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ 主窗口      │  │ 修图编辑器  │  │ 图像/视频预览画廊  │   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬──────────┘   │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    业务逻辑层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ Generation  │  │   Events    │  │  Shell Mixin      │   │
│  │   Mixin     │  │   Mixin     │  │ (v6 工作台骨架)    │   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬──────────┘   │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│              核心引擎层 (ModelManager + Loaders)            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  core/arch 指纹检测 arch_id → core/loaders 注册表    │   │
│  │  路由到 SD15 / SDXL / Anima / SingleFile Loader      │   │
│  │  - txt2img_pipe, img2img_pipe, inpaint_pipe         │   │
│  │  - controlnet_pipe, pose_detector, ip_adapter       │   │
│  │  - AnimateDiff Pipeline + Motion LoRA               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    资源层                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────────┐   │
│  │ models  │  │ loras   │  │ video   │  │ 配置文件        │   │
│  └─────────┘  └─────────┘  └─────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心技术栈

| 分类 | 技术 | 版本 | 用途 |
|-----|------|------|------|
| **UI 框架** | PyQt6 | ≥6.7 | 主界面、对话框 |
| **界面主题** | QDarkStyleSheet | - | 深色主题（`ui/theme.py` 接入） |
| **视频播放** | QtMultimedia | 6.x | 视频预览播放 |
| **绘图引擎** | PyQt6 | 6.x | 修图编辑器 |
| **图像处理** | PIL/Pillow | 10.x | 图像加载、处理、保存 |
| **深度学习** | PyTorch | 2.11（CUDA 12.8 单独安装） | GPU 加速推理 |
| **扩散模型** | Diffusers | 0.40.x | Stable Diffusion 管线 |
| **文本编码** | Transformers | 5.16.x | CLIP / T5 文本编码器 |
| **视频扩散** | AnimateDiff | - | 视频生成管线 |
| **控制网络** | ControlNet | - | 姿态/深度控制 |
| **提示词编码** | Compel | ≥2.0 | 高级提示词处理（SD1.5/SDXL） |
| **人脸检测** | OpenCV + YOLO | 4.x / 8.x | ADetailer 人脸/手部检测 |

### 3.3 显存优化策略

由 `utils/vram_manager.py` 按 GPU 显存总量自动分级：

| 显存 | 策略 |
|------|------|
| 充足（SD1.5 ≥6GB / SDXL ≥16GB） | 🚀 全速模式：全量驻留 GPU |
| 中等（SD1.5 ≥4GB / SDXL ≥11GB） | ⚡ 标准模式：驻留 + VAE/Attention 切片 |
| 偏低（SD1.5 ≥2.5GB / SDXL ≥5GB） | 💾 节能模式：Model CPU Offload |
| 极低 | 🐢 极限模式：Sequential CPU Offload |

通用优化：**VAE Tiling/Slicing**（分块解码，支持大图）、**Attention Slicing**、**xFormers**（装了才启用）。

### 3.4 日志设施

全项目统一使用 Python `logging`（不再有裸 `print`）：

- 初始化入口：`utils/system_utils.py::setup_logging()`（幂等，含滚动文件 + 控制台双输出）
- 日志文件：`logs/app.log`（RotatingFileHandler，单文件 5MB × 3 备份，UTF-8）
- 各模块约定：`logger = logging.getLogger(__name__)`，按 ❌=error / ⚠️=warning / 调试标记=debug 分级

---

## 四、文件结构

```
ai  make photo/
├── main.py                    # 主入口，PyQt6 主窗口（Mixin 组合）
├── requirements.txt           # 依赖清单（torch 需按注释单独安装 CUDA 版）
├── requirements-lock.txt      # 本机完整冻结环境（pip freeze）
├── app_config.json            # 应用配置（运行时自动生成/更新）
├── aiUsemono.md               # 使用说明
├── 结构.txt                   # 项目结构图
│
├── core/                      # 核心模块
│   ├── arch/                  # 架构识别（signatures 指纹 + registry 能力声明）
│   │   ├── base.py            #   ArchInfo/Caps/build_arch_profile
│   │   ├── registry.py        #   架构注册表（sd15/sd21/sdxl/sd3/flux/anima…）
│   │   └── signatures.py      #   权重 key 指纹检测
│   ├── loaders/               # 模型加载器（每架构一个类，统一 build/apply_vram/derive_pipes/post_load 分层）
│   │   ├── base.py            #   BaseLoader + LoadContext/LoadResult + 共享 derive_from_components
│   │   ├── sd15.py            #   SD15Loader（sd15 + sd21）
│   │   ├── sdxl.py            #   SDXLLoader（含 Pony/NoobAI 等 XL 衍生）
│   │   ├── anima.py           #   AnimaLoader（ModularPipeline + 权重前缀剥离）
│   │   ├── flux.py            #   FluxLoader（未注册，半成品）
│   │   └── single_file.py     #   SingleFileLoader（sd3/flux 兜底）
│   ├── config_manager.py      # 配置管理（AppConfig dataclass + 快捷键表）
│   ├── model_diagnostics.py   # 模型诊断（文件元数据 + 已加载管线运行时报告）
│   ├── model_manager.py       # 模型加载管理（单例 ModelManager，经 loaders 路由）
│   ├── presets.py             # 提示词预设
│   └── translation_service.py # 翻译服务（词典/AI/自动）
│
├── ui/                        # 界面模块（v6.0 三栏工作台）
│   ├── shell.py               # ShellMixin —— 默认界面骨架（导航栏/中央区/参数面板/胶片条/状态栏）
│   ├── nav.py                 # 左侧导航轨（文生图/图生图/动画/画廊）
│   ├── core_panel.py          # 生成核心区（全局单例：模型/prompt/参数/翻译模式/诊断按钮）
│   ├── shared_groups.py       # 共享折叠分组（LoRA/ControlNet/高级/X-Y）
│   ├── contracts.py           # 控件契约 + 自检降级（关键缺失禁用生成入口）
│   ├── param_snapshot.py      # 参数快照/安全设值（_safe_set_*）
│   ├── settings_dialog.py     # 设置对话框（快捷键改键 + 组件默认值）
│   ├── pages/                 # 四个页面（PageBase 派生，单页失败不影响其他页）
│   │   ├── txt2img_page.py    #   文生图（预览区 + 修前/修后对比滑条 + 日志）
│   │   ├── img2img_page.py    #   图生图（参考图/IP-Adapter/Pose Transfer）
│   │   ├── video_page.py      #   动画（AnimateDiff/运镜/配音）
│   │   └── gallery_page.py    #   画廊页（统一图片+动画）
│   ├── components/
│   │   ├── filmstrip.py       # 底部胶片条（单击回填参数/右键菜单/悬浮预览）
│   │   └── collapsible.py     # 可折叠分组容器
│   ├── gallery_panel.py       # 画廊面板（排序过滤/收藏/元数据浮窗/悬浮大图）
│   ├── theme.py               # QDarkStyleSheet 主题接入
│   ├── widgets.py             # 自定义控件（FloatSlider/GpuCanvas/CompareCanvas 等）
│   ├── ui_builder.py          # 旧版 UI（AI_STUDIO_UI=old 回退用，不再演进）
│   ├── design_tokens.py       # 旧版样式令牌（回退用）
│   ├── extension_market.py    # 扩展市场
│   ├── preset_manager.py      # 预设管理（含工具提示 TooltipMixin）
│   ├── disclaimer.py          # 免责声明
│   ├── splash.py              # 启动画面
│   ├── tooltips.py            # 工具提示文案
│   └── video_panel_mixin.py   # 视频面板业务（VideoPanelMixin）
│
├── utils/                     # 工具模块
│   ├── app_events.py          # 事件处理（EventMixin）
│   ├── app_generation.py      # 生成流程（GenerationMixin）
│   ├── app_utils.py           # 应用工具函数（动态提示词解析等）
│   ├── chattts_patch.py       # ChatTTS 兼容补丁
│   ├── extension_manager.py   # 扩展管理
│   ├── gpu_init.py            # GPU 加速初始化
│   ├── image_processor.py     # 图像处理 / ADetailer 流程
│   ├── model_downloader.py    # 模型下载
│   ├── model_scanner.py       # 模型扫描
│   ├── paths.py               # 统一路径管理（所有目录常量）
│   ├── prompt_enhancer.py     # Qwen 智能改写
│   ├── rife_interpolate.py    # RIFE 帧插值
│   ├── sovits_tts.py          # GPT-SoVITS 配音
│   ├── system_utils.py        # 系统工具 / 日志设施
│   ├── tiled_diffusion.py     # 分块扩散
│   ├── tts_engine.py          # ChatTTS 配音
│   ├── video_gen.py           # 视频生成服务（AnimateDiff）
│   └── vram_manager.py        # 显存管理（按容量分级策略）
│
├── photo_turn/                # 修图编辑器模块（自包含）
│   ├── components.py
│   ├── mixin_ai.py
│   ├── mixin_filters.py
│   ├── mixin_history.py
│   ├── mixin_tools.py
│   └── pro_editor_qt.py
│
├── scripts/                   # 维护脚本
│   ├── download_gemma.py      # 模型下载
│   ├── download_ref_voice.py  # 参考音频下载
│   ├── download_sovits.py     # SoVITS 下载
│   └── codemod_unify_logging.py # 日志统一迁移脚本（一次性工具）
│
├── tests/                     # 手动测试脚本（纯脚本非 pytest，venv python 直接跑）
│   ├── test_real_startup.py   #   真实启动路径（setup_ui + 配置还原 + 页面切换）
│   ├── test_ui_contract.py    #   控件契约自检与分级降级
│   ├── test_v6_features.py    #   v6 新功能（对比滑条/资源监控/悬浮预览/回填/回译/诊断）
│   ├── test_settings.py       #   设置对话框（改键重建/默认值读写回环）
│   ├── test_loaders.py        #   加载器注册表路由与派生管线
│   ├── test_gallery_unified.py#   统一画廊（图片+动画/胶片条/过滤）
│   ├── test_theme.py / test_nav.py / test_collapsible.py  # 主题/导航/折叠组
│   ├── diag_real_click.py     #   真实窗口诊断（面板溢出/回填链路实测）
│   └── test_translate.py / test_tts.py / smoke_photo_turn.py 等  # 翻译/TTS/修图冒烟
│
├── models/                    # AI 模型存储
│   ├── sd15/  sdxl/  sd3/  flux/   # 各系列底模
│   ├── adetailer/             # 面部/手部修复模型（YOLO）
│   ├── ip_adapter/            # IP-Adapter 模型
│   ├── motion_adapter/        # AnimateDiff 运动模块
│   ├── motion_lora/           # Motion LoRA（pan-left/tilt-up/zoom-in/zoom-out）
│   └── tts/                   # TTS 模型
│
├── loras/                     # LoRA 权重（sd1.5/ sdxl/）
├── controlnets/               # ControlNet 模型（本地缓存）
├── tools/rife/                # RIFE 帧插值可执行工具
├── third_party/               # 第三方源码（GPT-SoVITS 等）
├── assets/voices/             # 参考音频
├── weights/                   # 预训练权重（人脸检测级联等）
├── data/                      # 运行时数据（词典、收藏等）
│   ├── dictionaries/          # 分类词典
│   └── zh_to_en_dict.json     # 翻译词典
├── models_cache/              # HF/Torch 统一缓存（HF_HOME 等指向此处）
├── photo/                     # 出图输出（videos/ 子目录存放视频）
├── output/                    # 其他输出（如 TTS 音频）
└── logs/                      # 日志（app.log，滚动切割）
```

---

## 五、安装与运行

### 5.1 环境要求

- **操作系统**：Windows 10/11 (推荐)
- **Python 版本**：3.10.x / 3.12.x
- **GPU 要求**：NVIDIA GPU（推荐 8GB+ 显存）（没有 GPU 也可以使用 CPU）
- **CUDA 版本**：11.8+

### 5.2 安装步骤

```bash
# 1. 克隆或下载项目
git clone <repository_url>
cd "ai  make photo"

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装 PyTorch（CUDA 12.8，务必单独安装，不要用 requirements.txt 装）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安装其余依赖
pip install -r requirements.txt

# 5. 下载模型文件
# 最简单的做法：把下载到的模型统一丢进 models/puton/ 即可，
# 程序启动时会自动识别架构并移动到对应目录，并弹窗告知结果。
# 也可以手动放置（见 5.3 表格）：
# 将 SD 模型(.safetensors)放入 models/sd15/ 或 models/sdxl/ 等对应目录
# 将 LoRA 文件放入 loras/sd1.5/ 或 loras/sdxl/ 目录
# 将 Motion LoRA 文件放入 models/motion_lora/ 目录

# 6. 运行应用
python main.py
```

### 5.3 模型准备

> **本项目不预置任何模型，也不在运行流程中静默下载模型。**
> 底模 / LoRA / ControlNet 等请自行下载后放入下表目录。
> 只有「辅助功能」会在你**明确点击并确认**后下载（见 5.4）。

**不用自己判断架构**：把下载到的模型文件直接丢进 `models/puton/`
（投递箱）即可。程序每次启动会读取文件头判断它是 SD1.5 / SDXL /
Flux / SD3 / Anima / LoRA，自动移动到下表中正确的目录，并弹窗告诉你
每个文件去了哪里。

- 只读文件头，**不加载权重、不联网、不上传**；
- 识别不了的文件（或 ControlNet/VAE 这类非底模）**原样留在投递箱**
  并说明原因，绝不乱猜、绝不删除；
- 同名文件**不会覆盖**：大小相同视为重复跳过，否则自动加 `_1` 后缀；
- 正在下载的中间文件（`.part` / `.tmp` / `.crdownload` / 0 字节）会被忽略。

| 文件类型 | 放置位置 | 说明 |
|---------|---------|------|
| **任何底模 / LoRA（推荐）** | **`models/puton/`** | **丢进来即可，启动时自动归位** |
| SD 底模 | `models/sd15/*.safetensors` | SD 1.5 模型 |
| SDXL 底模 | `models/sdxl/*.safetensors` | SDXL 模型 |
| Flux / SD3 底模 | `models/flux/`、`models/sd3/` | 还需先装引擎组件，见 5.5 |
| LoRA 插件 | `loras/sd1.5/*.safetensors` | SD 1.5 专用 |
| LoRA 插件 | `loras/sdxl/*.safetensors` | SDXL 专用 |
| LoRA（判不出底模） | `loras/_unknown_dit/` | 投递箱无法确定架构时的落点 |
| Motion LoRA | `models/motion_lora/*/` | 运镜特效模型 |
| ADetailer 模型 | `models/adetailer/*.pt` | 人脸/手部检测模型 |
| IP-Adapter | `models/ip_adapter/` | 图生视频用（`ip-adapter_sd15.safetensors` + `image_encoder/`） |
| ChatTTS | `models/tts/ChatTTS/` | 配音用 |

> **LoRA 按架构分目录**：`loras/sd1.5/`、`loras/sdxl/`、`loras/flux/`、
> `loras/sd3/`、`loras/anima/`；判不出底模的进 `loras/_unknown_dit/`。
> 投递箱会自动放到正确子目录，不用你判断。
| ControlNet | `controlnets/<type>/` | 需含 `config.json` |

启动时会扫描缺失项并写进日志（`print_scan_report()`），**只扫描不下载**。
命令行自查与手动获取：

```bash
python -m utils.model_downloader scan             # 查看缺什么
python -m utils.model_downloader install <key>    # 手动下载单项（需自行执行）
```

### 5.4 联网行为说明

| 行为 | 何时发生 | 说明 |
|------|---------|------|
| 启动扫描 | 每次启动 | 只读本地目录，**不联网** |
| 扩展市场 | 你点「⬇️ 下载」 | 辅助功能（ControlNet / RIFE / ChatTTS / GPT-SoVITS / Qwen 等） |
| 提示词智能改写 / 识图 | 你点按钮或勾选「生成前自动改写」 | 缺 Qwen 档位时**先弹窗告知体积与来源**，确认后才从 ModelScope 下载；也可自行放入 `models_cache/modelscope` |
| 图生视频 | 切到该模式且缺 IP-Adapter | **先弹窗确认**，不静默下载 |
| 配音（ChatTTS） | 勾选配音且缺模型 | **先弹窗确认**，不静默下载 |
| ControlNet / OpenPose / 深度 | 首次启用且本地无模型 | 仍会联网获取；可自行放入 `controlnets/` 后离线使用 |
| 翻译 | 默认「词典优先 + AI 兜底」模式 | 未命中词典时会请求 Google 翻译（可切「纯词典」完全离线） |

### 5.5 新架构引擎（Flux / SD3）

这些架构**不能只靠一个 checkpoint 跑起来**，还需要配套的文本编码器 / VAE
等「引擎组件」。本项目把它拆成两部分，且**当前尚未接通**（放上模型也会明确
提示，不会静默下载）：

| 部分 | 体积 | 来源 |
|------|------|------|
| 引擎组件（T5-XXL / CLIP-L / VAE / tokenizer） | 约 10GB | **扩展市场 → 新架构引擎** |
| transformer 主权重 | 约 24GB（bf16） | **你自行下载**，放进 `models/flux/` 或 `models/puton/` |

检测到 Flux 模型但组件缺失时，启动会**单独弹窗**告知，并提供「去扩展市场下载」
按钮 —— 不下载引擎就无法使用 Flux 生成。

#### ⚠️ 许可证差异（选之前务必看清）

| 变体 | 许可证 | 能否商用 |
|------|--------|---------|
| **FLUX.1-schnell** | Apache-2.0 | ✅ **可商用**，无附加限制（推荐） |
| FLUX.1 [dev] | FLUX.1 [dev] Non-Commercial License v1.1.1 | ❌ **不可商用** |

FLUX.1 [dev] 的非商用许可明确禁止「营利性活动」以及「直接面向终端用户、
或对终端用户产生影响」的使用；商用需另行向 Black Forest Labs 申请授权。
两者的**生成图片**都可商用（BFL 声明不对 Outputs 主张权利），受限的是
「使用模型」这件事本身。

本项目为「开源 + 打赏」形态：**schnell 无争议；用 dev 权重请自行确认你的
使用方式落在非商用范围内，风险自负。** 详见
[LICENSE-FLUX1-dev](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)
与 [LICENSE-FLUX1-schnell](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell)。


---

## 六、使用说明

### 6.1 主界面布局（v6.0 三栏工作台）

```
┌──────────────────────────────────────────────────────────────┐
│  [文件] [工具] [内存] [关于] [设置]                            │
├────┬──────────────────────────────────────────┬──────────────┤
│ 导 │                                          │ 右侧参数面板  │
│ 航 │           中央工作区                      │ (整体可滚动)  │
│ 轨 │  ┌────────────────────────────────────┐  │ ┌──────────┐ │
│ 🎨 │  │  预览画布（普通/修前修后对比二合一） │  │ │生成核心区 │ │
│ 🖼 │  ├────────────────────────────────────┤  │ │(全局单例) │ │
│ 🎬 │  │  编辑/另存/转图生图/转重绘/⚖️对比   │  │ ├──────────┤ │
│ 🖻 │  ├────────────────────────────────────┤  │ │页面专属区 │ │
│    │  │  生成日志                          │  │ ├──────────┤ │
│    │  └────────────────────────────────────┘  │ │共享折叠组 │ │
│    ├──────────────────────────────────────────┤ │(LoRA/CN/  │
│    │  底部常驻胶片条（单击回填参数/右键菜单）   │ │高级/X-Y)  │ │
├────┴──────────────────────────────────────────┤ ├──────────┤ │
│ 状态栏：状态 | 显存/内存常驻监控 | 扩展计数      │ │生成/停止  │ │
└───────────────────────────────────────────────┴─┴──────────┴─┘
```

- **导航轨**：文生图 / 图生图 / 动画 / 画廊 四页，页面独立构建，单页失败不影响其他页（契约自检，关键控件缺失时禁用生成入口）
- **生成核心区全局唯一**：模型/prompt/尺寸/步数/CFG/采样器/生成按钮只有一份，切页不重造，业务层零感知
- **胶片条**：跟随画廊过滤，单击 = 全量回填该图参数 + 预览（prompt/steps/CFG/seed/尺寸/采样器/模型），右键 = 套用参数/查看元数据/收藏/删除
- **⚖️ 对比滑条**：开 Hires.fix/ADetailer 生成后对比精修前后；图生图加载参考图生成后对比参考图与结果
- **统一画廊**：图片与动画同墙展示（动画带首帧缩略图+时长角标），顶栏搜索/类型过滤/排序，悬浮 0.5s 显示大图
- **🌐 翻译模式右侧「🔁 对比」**：中→英→中回译对照，检查 AI 翻译幻觉（点对比才载入 Qwen，用完即卸不占生图显存）

### 6.2 动画界面操作

1. **选择生成模式**：文生视频、图生视频、视频转绘、提示词旅行
2. **输入提示词**：设置正面和负面提示词
3. **配置参数**：帧数、FPS、分辨率、采样器等
4. **添加运镜特效**：选择 Motion LoRA 并调整强度
5. **点击生成**：视频生成完成后自动播放并加入画廊

### 6.3 快捷键（默认键位，均可在 设置→偏好设置→快捷键 中修改）

| 快捷键 | 功能 |
|-------|------|
| `Ctrl+Enter` | 触发生成 |
| `Esc` | 中断生成 |
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` / `Ctrl+4` | 切换 文生图 / 图生图 / 动画 / 画廊 |
| `Ctrl+E` | 扩展市场 |
| `Ctrl+,` | 偏好设置 |

改键位：菜单「⚙️ 设置 → 偏好设置 → ⌨️ 快捷键」，点输入框按新键位，清空即禁用，确定后立即生效。修图编辑器内另有 `B` 画笔 / `E` 橡皮 / `M` 遮罩 / `G` 网格等（见 7.4）。

### 6.4 视频参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| 帧数 | 8 ~ 80 | 视频总帧数 |
| FPS | 4 ~ 30 | 每秒帧数，建议 8-12 |
| 步数 | 10 ~ 100 | 每帧生成步数 |
| CFG | 1.0 ~ 20.0 | 提示词引导强度 |
| 运镜强度 | 0.0 ~ 2.0 | Motion LoRA 强度 |

---

## 七、核心模块详解

### 7.1 ModelManager（模型管理器）

**职责**：管理 Stable Diffusion 模型的加载、优化、推理流程

**核心方法**：

| 方法 | 功能 |
|------|------|
| `load_model()` | 加载指定 SD 模型 |
| `apply_multiple_loras()` | 挂载多个 LoRA 插件 |
| `prepare_controlnet()` | 配置 ControlNet |
| `encode_prompt()` | 使用 Compel 编码提示词 |
| `switch_sampler()` | 切换采样器 |
| `apply_optimizations()` | 应用显存优化 |

**设计模式**：单例模式（SingletonMeta）

**加载器路由**：`load_model()` 经 `core/loaders/registry.py` 注册表按 `core/arch/` 检测出的 arch_id 分发到专用 Loader——sd15/sd21 → `SD15Loader`，sdxl → `SDXLLoader`，anima → `AnimaLoader`，sd3/flux → `SingleFileLoader` 兜底。

### 7.2 AIDesktopApp（主窗口）

**职责**：整合所有 UI 组件和业务逻辑

**混入类架构**：

| Mixin | 功能 |
|-------|------|
| `ShellMixin`（`ui/shell.py`） | v6.0 默认 UI：三栏工作台、页面路由、契约自检 |
| `UIBuilderMixin`（`ui/ui_builder.py`） | 旧版 UI 回退（环境变量 `AI_STUDIO_UI=old` 启用） |
| `EventMixin` | 事件处理和信号 |
| `GenerationMixin` | 图像和视频生成逻辑 |
| `ParamSnapshotMixin` | 参数快照（胶片条回填 / 元数据复用） |
| `PresetManagerMixin` / `TooltipMixin` | 参数预设与提示 |
| `VideoPanelMixin` | 视频面板扩展逻辑 |

### 7.3 VideoGenerator（视频生成器）

**职责**：处理视频生成流程

**核心方法**：

| 方法 | 功能 |
|------|------|
| `generate()` | 生成视频（支持多种模式） |
| `apply_motion_lora()` | 应用运镜特效 |
| `apply_frame_interpolation()` | 帧插值提升流畅度 |

### 7.4 ProImageEditor（修图编辑器）

**职责**：提供专业级图像编辑功能

**工具功能**：

- **画布**：OpenGL 加速，滚轮缩放（光标锚点）、中键平移、`Ctrl+0` 适配窗口、三分构图网格
- **画布光标**：绘制类工具显示画笔光标圈（直径随缩放同步，白色外圈 + 笔刷色内圈）；状态栏实时显示光标坐标、hex 色值、图像尺寸与缩放百分比
- **绘图工具**：画笔（大小 `[`/`]` 微调 + 不透明度 + 硬度柔边）、橡皮擦、AI 遮罩画笔；`Shift`+单击从上一笔末端拉直线（PS 同款）
- **遮罩操作**：清除 / 反转 / 羽化（可调半径）/ 扩边 / 收缩（配合局部重绘工作流）；红色叠加预览可临时开关
- **取色**：吸管工具 + 调色板
- **对比**：按住「👁 对比原图」快速查看编辑前后差异
- **选择工具**：裁剪（拖拽实时选区预览，支持比例约束：自由 / 1:1 / 4:3 / 3:4 / 16:9 / 9:16）
- **文字工具**：添加、编辑文字（中文字体优先回退链，按文字亮度自动黑/白描边，描边宽度 0–8）
- **变换工具**：水平/垂直翻转、±90°/任意角度旋转、等比缩放到长边（512–2048）
- **调整工具**：亮度、对比度、饱和度、曝光、色相、锐化、色温
- **滤镜工具**：18 种预设滤镜（含素描、卡通、晕影、像素化等）；「👁 预览」先看效果不入历史（▶ 应用提交 / `Esc` 取消），选「无」还原到叠加前
- **快捷键**：`B` 画笔 / `E` 橡皮 / `M` 遮罩 / `G` 构图网格（输入框聚焦时不触发）
- **导出**：编辑器内「另存为」直接导出；「保存并返回」自动归档到 `photo/` 画廊（`pro_edited_HHMMSS.png`）
- **历史**：撤销/重做 15 步，状态栏显示剩余步数；有未保存改动时关闭编辑器会弹确认（保存并关闭 / 直接关闭 / 取消）

---

## 八、配置与扩展

### 8.1 配置文件 (`app_config.json`)

配置由 `core/config_manager.py` 的 `AppConfig` dataclass 定义，关闭应用时自动保存，加载时高容错（忽略废弃字段、补全缺失字段）。主要字段：

| 字段 | 说明 |
|------|------|
| `default_steps` / `default_cfg` / `default_sampler` | 默认采样参数 |
| `default_width` / `default_height` / `default_batch` | 默认尺寸与批数 |
| `default_strength` / `default_lora_weight` | 默认图生图强度 / LoRA 权重 |
| `device_preference` | 设备偏好（自动 / CUDA:x / CPU） |
| `use_adetailer` / `adetailer_strength` | 人脸修复开关与强度 |
| `use_ad_hand` / `ad_hand_strength` / `ad_hand_blend` | 手部修复参数 |
| `use_hires` / `hires_denoise` | 高清修复开关与重绘幅度 |
| `output_format` / `output_dir` | 输出格式与目录 |
| `last_prompt` / `last_neg` | 上次使用的提示词 |
| `recent_models` / `recent_prompts` | 历史记录 |
| `shortcuts` | 快捷键键位表（设置界面→偏好设置→快捷键 中编辑） |
| `default_trans_mode` | 默认翻译模式（0=关 / 1=词典 / 2=AI） |

### 8.2 扩展开发

**添加新滤镜**：在 `photo_turn/mixin_filters.py` 中添加新的滤镜方法

**添加新工具**：在 `photo_turn/mixin_tools.py` 中添加新的工具类

**添加新模型支持**：在 `core/loaders/` 参考 `anima.py` / `sd15.py` 新建 Loader 类并在 `registry.py` 注册（arch 检测规则在 `core/arch/signatures.py` 补充）

**添加新页面**：在 `ui/pages/` 新建页类并在 `ui/nav.py` 登记导航项；页面专属控件放页面内，生成核心控件（模型/prompt/尺寸等）保持全局唯一

**添加新运镜特效**：在 `models/motion_lora/` 目录添加对应模型

---

## 九、已知限制与说明

### 9.1 模型识别

| 限制 | 说明 |
|------|------|
| 模型类型靠文件名/体积推断 | SDXL 以文件名关键词（xl/pony 等）或体积 4.2~8GB 判定，冷僻命名可能误判 |
| SD3 / Flux 仅部分支持 | 可识别并加载文生图/图生图，局部重绘回退到主 Pipeline，功能不完整 |

### 9.2 网络依赖

| 限制 | 说明 |
|------|------|
| 默认使用 hf-mirror 镜像 | `main.py` 中 `HF_ENDPOINT=https://hf-mirror.com`，海外环境可手动移除 |
| ControlNet / OpenPose 首次需联网 | 也可手动下载到 `controlnets/` 对应目录后离线加载 |

### 9.3 运行时

| 限制 | 说明 |
|------|------|
| 视频无法播放 | QtMultimedia 缺少解码支持时，确保系统可用 FFmpeg |
| 后台生成线程 | 生成在后台线程执行，关闭应用前请先等待或中断当前任务 |

---

## 十、注意事项

1. **首次运行**：首次加载模型会自动下载必要的依赖文件，可能需要较长时间
2. **显存要求**：建议使用 8GB+ 显存的 GPU，否则可能无法加载大型模型
3. **模型文件**：模型文件较大（通常 2GB-8GB），请确保有足够的磁盘空间
4. **网络环境**：首次运行需要联网下载部分依赖，建议在网络稳定的环境下运行
5. **模型格式**：推荐使用 `.safetensors` 格式的模型文件，安全性更高
6. **视频生成**：视频生成比图像生成耗时更长，请耐心等待

---

**版本**：v6.0  
**更新日期**：2026-09-20

---

## 十一、许可与合规（重要）

本项目以 **AGPL-3.0** 发布，完整许可见 `LICENSE`。

之所以是 AGPL 而不是 MIT：项目依赖/集成了两个 **AGPL-3.0** 组件——
`ultralytics`（ADetailer 的 YOLOv8 检测）与 ChatTTS（中文配音）。
这两个是强 copyleft，与本项目结合后整体须以 AGPL-3.0 发布。

> **接受捐赠（打赏）通常不构成商业分发**，因此"开源 + 打赏"与 AGPL 兼容。
> 但一旦提供付费功能 / 企业版 / 云端服务，AGPL 第 13 条要求你向网络用户
> 提供完整源码。

**如果你希望改用宽松许可（MIT / Apache-2.0）以便闭源或商用**，
需要先移除 AGPL 组件，替换方案见 `THIRD_PARTY_NOTICES.md` 第四节，简要版：

| 替换对象 | 建议方案 | 许可 |
|---|---|---|
| `ultralytics` YOLOv8 检测 | OpenCV 自带 `cv2.FaceDetectorYN`（YuNet） | Apache-2.0 |
| ChatTTS | GPT-SoVITS（项目已内置）或系统 TTS | MIT |
| NLLB-200（扩展市场条目） | 移除（CC-BY-NC-4.0 **禁止商用**） | — |
| PyQt6 | PySide6（LGPL）或购买 PyQt6 商业许可 | LGPL / 商业 |

**各依赖与模型权重的完整许可清单**（含底模的 CreativeML Open RAIL-M
使用限制、RIFE/Real-ESRGAN 的 MIT/BSD、FFmpeg 的 LGPL 等）：
见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

> ⚠️ 本项目**不预置任何模型权重**（`.gitignore` 已排除 `models/*`、`tools/`），
> 由用户自行下载。这显著降低了"再分发 AGPL 权重与受限底模"的风险，
> 请继续保持这一做法。

