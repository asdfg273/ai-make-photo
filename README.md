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

- **多模型支持**：支持 SD 1.5 和 SDXL 模型
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
| **UI 框架** | PyQt6 | 6.x | 主界面、对话框 |
| **视频播放** | QtMultimedia | 6.x | 视频预览播放 |
| **绘图引擎** | PyQt6 | 6.x | 修图编辑器 |
| **图像处理** | PIL/Pillow | 10.x | 图像加载、处理、保存 |
| **深度学习** | PyTorch | 2.x | GPU 加速推理 |
| **扩散模型** | Diffusers | 0.x | Stable Diffusion 管线 |
| **视频扩散** | AnimateDiff | - | 视频生成管线 |
| **控制网络** | ControlNet | - | 姿态/深度控制 |
| **提示词编码** | Compel | - | 高级提示词处理 |
| **人脸检测** | OpenCV + YOLO | 4.x / 8.x | ADetailer 人脸/手部检测 |

### 3.3 显存优化策略

- **CPU Offload**：模型权重动态卸载到 CPU，节省显存
- **VAE Tiling/Slicing**：分块解码，支持大图生成
- **Attention Slicing**：注意力计算分片处理
- **xFormers**：内存高效注意力机制

---

## 四、文件结构

```
ai  make photo/
├── main.py                    # 主入口，PyQt6 主窗口
├── requirements.txt           # 依赖清单
├── app_config.json            # 应用配置
├── aiUsemono.md               # 使用说明
├── 结构.txt                   # 项目结构图
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── config_manager.py      # 配置管理
│   ├── downup.py              # 图像放大/缩小
│   ├── model_manager.py       # 模型加载管理
│   ├── presets.py             # 预设管理
│   └── translation_service.py # 翻译服务
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── app_events.py          # 事件处理
│   ├── app_generation.py      # 生成流程
│   ├── app_utils.py           # 应用工具函数
│   ├── gallery_panel.py       # 画廊面板
│   ├── image_processor.py     # 图像处理
│   ├── model_scanner.py       # 模型扫描
│   ├── paths.py               # 路径管理
│   ├── preset_manager.py      # 预设管理
│   ├── prompt_enhancer.py     # Qwen 智能改写
│   ├── rife_interpolate.py    # RIFE 帧插值
│   ├── system_utils.py        # 系统工具
│   ├── tiled_diffusion.py     # 分块扩散
│   ├── tooltips.py            # 工具提示
│   ├── ui_builder.py          # UI 构建（含动画界面）
│   ├── video_gen.py           # 视频生成服务
│   └── vram_manager.py        # 显存管理
├── photo_turn/                # 修图编辑器模块
│   ├── __init__.py
│   ├── components.py
│   ├── mixin_ai.py
│   ├── mixin_filters.py
│   ├── mixin_history.py
│   ├── mixin_tools.py
│   └── pro_editor_qt.py
├── models/                    # AI 模型存储
│   ├── adetailer/             # 面部/手部修复模型
│   ├── loras/                 # LoRA 权重
│   │   ├── sd1.5/
│   │   └── sdxl/
│   ├── motion_lora/           # Motion LoRA（运镜特效）
│   │   ├── pan-left/
│   │   ├── tilt-up/
│   │   ├── zoom-in/
│   │   └── zoom-out/
│   └── sd15/                  # SD1.5 底模及相关
├── tools/                     # 外部可执行工具
│   └── rife/                  # RIFE 帧插值工具
├── data/                      # 运行时数据
│   ├── dictionaries/          # 分类词典
│   ├── app_config.json        # 应用配置
│   └── zh_to_en_dict.json     # 翻译词典
├── logs/                      # 日志文件
├── video/                     # 生成视频输出
├── weights/                   # 预训练权重
├── logo/                      # 应用图标和资源
└── venv/                      # Python 虚拟环境
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

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型文件
# 将 SD 模型(.safetensors 或 .ckpt)放入 models/sd15/ 目录
# 将 LoRA 文件放入 models/loras/sd1.5/ 或 models/loras/sdxl/ 目录
# 将 Motion LoRA 文件放入 models/motion_lora/ 目录

# 5. 运行应用
python main.py
```

### 5.3 模型准备

| 文件类型 | 放置位置 | 说明 |
|---------|---------|------|
| SD 底模 | `models/sd15/*.safetensors` | SD 1.5 模型 |
| LoRA 插件 | `models/loras/sd1.5/*.safetensors` | SD 1.5 专用 |
| LoRA 插件 | `models/loras/sdxl/*.safetensors` | SDXL 专用 |
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

- **绘图工具**：画笔、橡皮擦、遮罩画笔
- **选择工具**：裁剪
- **文字工具**：添加、编辑文字
- **变换工具**：水平/垂直翻转、旋转
- **调整工具**：亮度、对比度、饱和度、锐化、色温
- **滤镜工具**：11 种预设滤镜

---

## 八、配置与扩展

### 8.1 配置文件 (`app_config.json`)

```json
{
  "default_model": "majicmixrealisticV6_v10.safetensors",
  "default_sampler": "Euler a",
  "output_format": "png",
  "auto_save": true,
  "show_preview": true
}
```

### 8.2 扩展开发

**添加新滤镜**：在 `photo_turn/mixin_filters.py` 中添加新的滤镜方法

**添加新工具**：在 `photo_turn/mixin_tools.py` 中添加新的工具类

**添加新模型支持**：在 `core/model_manager.py` 中扩展模型加载逻辑

**添加新运镜特效**：在 `models/motion_lora/` 目录添加对应模型

---

## 九、已知问题与解决方案

### 9.1 文件缺失问题

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| `pro_editor_tk.py` 不存在 | Tkinter 版修图编辑器文件缺失 | 使用 Qt 版 `pro_editor_qt.py` 替代 |

### 9.2 依赖兼容性问题

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| `ScrolledFrame` 导入失败 | ttkbootstrap 版本兼容性问题 | 使用标准 Tkinter Canvas + Scrollbar 实现 |

### 9.3 UI 控件联动问题

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| 控件初始化顺序问题 | 某些控件在初始化时可能不存在 | 使用 `hasattr()` 检查或调整初始化顺序 |

### 9.4 视频播放问题

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| 视频无法播放 | QtMultimedia 缺少解码支持 | 确保安装了 FFmpeg 相关依赖 |

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
**更新日期**：2026-07-08
