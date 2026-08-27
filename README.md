# **AI 绘画工作站 v5.0**

---

## 一、项目概述

**项目名称**：AI 绘画工作站 v5.0

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
- **图像管理**：自动保存、历史记录、图片预览
- **视频管理**：视频预览、历史画廊、双击播放

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
│  │ Generation  │  │   Events    │  │     UI Builder    │   │
│  │   Mixin     │  │   Mixin     │  │    Mixin          │   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬──────────┘   │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    核心引擎层 (ModelManager)                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  StableDiffusion Pipeline + ControlNet + LoRA       │   │
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
| **视频播放** | QtMultimedia | 6.x | 视频预览播放 |
| **绘图引擎** | PyQt6 | 6.x | 修图编辑器 |
| **图像处理** | PIL/Pillow | 10.x | 图像加载、处理、保存 |
| **深度学习** | PyTorch | 2.x（CUDA 12.8 单独安装） | GPU 加速推理 |
| **扩散模型** | Diffusers | 0.32.2 | Stable Diffusion 管线 |
| **文本编码** | Transformers | 4.49.0 | CLIP / T5 文本编码器 |
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
│   ├── config_manager.py      # 配置管理（AppConfig dataclass）
│   ├── model_manager.py       # 模型加载管理（单例 ModelManager）
│   ├── presets.py             # 提示词预设
│   └── translation_service.py # 翻译服务（词典/AI/自动）
│
├── ui/                        # 界面模块
│   ├── design_tokens.py       # 样式令牌（DARK_STYLE 等）
│   ├── disclaimer.py          # 免责声明
│   ├── extension_market.py    # 扩展市场
│   ├── gallery_panel.py       # 画廊面板
│   ├── preset_manager.py      # 预设管理面板
│   ├── splash.py              # 启动画面
│   ├── tooltips.py            # 工具提示
│   ├── ui_builder.py          # UI 构建（UIBuilderMixin）
│   ├── video_panel_mixin.py   # 视频面板（VideoPanelMixin）
│   └── widgets.py             # 自定义控件（FloatSlider 等）
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
├── tests/                     # 手动测试脚本（非 pytest）
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
# 将 SD 模型(.safetensors)放入 models/sd15/ 或 models/sdxl/ 等对应目录
# 将 LoRA 文件放入 loras/sd1.5/ 或 loras/sdxl/ 目录
# 将 Motion LoRA 文件放入 models/motion_lora/ 目录

# 6. 运行应用
python main.py
```

### 5.3 模型准备

| 文件类型 | 放置位置 | 说明 |
|---------|---------|------|
| SD 底模 | `models/sd15/*.safetensors` | SD 1.5 模型 |
| SDXL 底模 | `models/sdxl/*.safetensors` | SDXL 模型 |
| LoRA 插件 | `loras/sd1.5/*.safetensors` | SD 1.5 专用 |
| LoRA 插件 | `loras/sdxl/*.safetensors` | SDXL 专用 |
| Motion LoRA | `models/motion_lora/*/` | 运镜特效模型 |
| ADetailer 模型 | `models/adetailer/*.pt` | 人脸/手部检测模型 |

---

## 六、使用说明

### 6.1 主界面布局

```
┌──────────────────────────────────────────────────────────────┐
│  [文件] [工具] [关于] [内存]  │  状态栏                        │
├──────────────────────────────────────────────────────────────┤
│ 左侧控制面板                    │          画布/视频预览区        │
│ ┌─────────────────────────┐    │                              │
│ │  🎨 基础                 │    │                              │
│ │  🌀 动画                 │    │        图像/视频预览          │
│ │  🖼 图生图               │    │                              │
│ │  🧩 LoRA                │    │                              │
│ │  🕹 ControlNet          │    │                              │
│ │  ⚙️ 高级                 │    │                              │
│ │  📊 X/Y 矩阵            │    │                              │
│ └─────────────────────────┘    │                              │
│                                │                              │
│  生成按钮                      └─────────────────────────────┘
└──────────────────────────────────────────────────────────────┘
```

### 6.2 动画界面操作

1. **选择生成模式**：文生视频、图生视频、视频转绘、提示词旅行
2. **输入提示词**：设置正面和负面提示词
3. **配置参数**：帧数、FPS、分辨率、采样器等
4. **添加运镜特效**：选择 Motion LoRA 并调整强度
5. **点击生成**：视频生成完成后自动播放并加入画廊

### 6.3 快捷键

| 快捷键 | 功能 |
|-------|------|
| `Ctrl+S` | 保存图像 |
| `Ctrl+Z` | 撤销操作 |
| `Ctrl+Y` | 重做操作 |
| `ESC` | 取消当前模式 |

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

### 7.2 AIDesktopApp（主窗口）

**职责**：整合所有 UI 组件和业务逻辑

**混入类架构**：

| Mixin | 功能 |
|-------|------|
| `UIBuilderMixin` | UI 构建和布局（含动画界面） |
| `EventMixin` | 事件处理和信号 |
| `GenerationMixin` | 图像和视频生成逻辑 |

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
- **绘图工具**：画笔（大小 `[`/`]` 微调 + 不透明度 + 硬度柔边）、橡皮擦、AI 遮罩画笔
- **遮罩操作**：清除 / 反转 / 羽化（可调半径）/ 扩边 / 收缩（配合局部重绘工作流）
- **取色**：吸管工具 + 调色板
- **对比**：按住「👁 对比原图」快速查看编辑前后差异
- **选择工具**：裁剪（拖拽实时选区预览）
- **文字工具**：添加、编辑文字（中文字体优先回退链）
- **变换工具**：水平/垂直翻转、±90°/任意角度旋转、等比缩放到长边（512–2048）
- **调整工具**：亮度、对比度、饱和度、曝光、色相、锐化、色温
- **滤镜工具**：18 种预设滤镜（含素描、卡通、晕影、像素化等），选「无」可还原到叠加前
- **导出**：编辑器内「另存为」直接导出
- **历史**：撤销/重做 15 步，状态栏显示剩余步数

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

### 8.2 扩展开发

**添加新滤镜**：在 `photo_turn/mixin_filters.py` 中添加新的滤镜方法

**添加新工具**：在 `photo_turn/mixin_tools.py` 中添加新的工具类

**添加新模型支持**：在 `core/model_manager.py` 中扩展模型加载逻辑

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

**版本**：v5.0  
**更新日期**：2026-08-24
