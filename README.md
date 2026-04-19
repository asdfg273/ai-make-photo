🎨 AI Make Photo Pro (本地纯净版 AI 绘画工作站)
Python
PyTorch
Diffusers
License

AI Make Photo Pro 是一款基于 Diffusers 和 Tkinter (ttkbootstrap) 深度定制的本地轻量化 Stable Diffusion GUI。
它摒弃了 WebUI 繁杂的依赖与环境配置，专为中低显存设备和追求纯净原生体验的创作者打造。拥有极速启动、多功能闭环编辑、SD1.5/SDXL 智能双模隔离等独家黑科技。

🌟 核心特色 (Core Features)
🚀 一键极速启动：无需繁琐的 webui-user.bat 等待，原生 Python 架构，告别浏览器显存抢占，启动即刻画图。
🧠 SD1.5 / SDXL 智能隔离系统：独创的底模与 LoRA 联动机制。切换主模型时，自动过滤并仅显示兼容版本的 LoRA，彻底告别 Size Mismatch 爆显存报错！
♾️ 无限提示词突破：内置 Compel 引擎，打破传统的 77 Token 限制，支持带括号权重的长文本长句直接生成。
🎨 Pro 级内置修图与涂鸦引擎：无需打开 PS！在历史图库中一键调出原生编辑器，支持：画笔涂鸦遮罩 (防锯齿平滑)、一键裁剪、文字实时缩放拖拽叠加，改完即刻回传至主界面进行局部重绘！
🪄 ADetailer 脸部崩坏拯救：内置 OpenCV 人脸特征级联网络（支持写实与二次元动漫），自动进行二度重绘，拒绝“远景邪神脸”。
📖 动态分镜与自动漫画网格：支持输入动态组合词（如 [苹果, 香蕉]在桌子上），引擎将自动拆分并生成多张分镜，最后自动拼接成漫画网格大图。
🔤 原生全中文支持：内置 Deep-Translator 与 Jieba 分词，直接输入中文提示词即可自动翻译并送入底层编码器。
📂 必备目录结构 (Directory Structure)
为了让智能系统生效，请务必在项目根目录下按照以下结构存放您的模型（首次运行会自动创建，也可手动创建）：

text
复制代码
收起
ai_make_photo/
├── models/             # 存放所有基础大模型 (Base Models)
│   ├── ChilloutMix.safetensors
│   └── SDXL_Base.safetensors
├── loras/              # 存放所有 LoRA 插件 (⚠️ 必须严格分类)
│   ├── sd1.5/          # SD 1.5 专用的 LoRA 放这里
│   └── sdxl/           # SDXL 专用的 LoRA 放这里
├── photo/              # 生成的图像将自动保存在这里
├── photo_turn/         # 内置编辑器核心代码
└── main.py             # 主程序入口
复制
(💡 小贴士：您可以在 models 或 loras 目录下新建与模型同名的 .txt 文件，里面写上触发词或心得，界面下拉框会自动读取并显示该备忘录！)

⚙️ 安装与运行 (Installation)
1. 克隆或下载本项目

bash
复制代码
收起
git clone https://github.com/您的用户名/ai_make_photo.git
cd ai_make_photo
复制
2. 创建虚拟环境并激活 (推荐)

bash
复制代码
收起
python -m venv venv
# Windows 激活方式:
venv\Scripts\activate
复制
3. 安装核心依赖

bash
复制代码
收起
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 请根据您的CUDA版本选择
pip install diffusers transformers accelerate compel
pip install ttkbootstrap pillow opencv-python deep-translator jieba controlnet_aux
复制
4. 启动引擎

bash
复制代码
收起
python main.py
复制
🕹️ 操作指引 (Quick Start)
1️⃣ 基础文生图 (TXT2IMG)
在顶部下拉框选择您的主模型。
在“正向提示词”框输入您想要的画面（支持全中文，如：一个赛博朋克风格的少女，高质量，杰作）。
需要挂载梯子，要用谷歌翻译。
设定步数（推荐 20-30）和分辨率。
点击底部 [开始生成]，即可。
2️⃣ 挂载 LoRA 插件
确保已将 .safetensors 的 LoRA 文件放入了对应的 loras/sd1.5 或 loras/sdxl 文件夹。
当您在主界面选择带 XL 名字的模型时，LoRA 下拉框会自动只显示 XL 专用的插件；反之亦然。
选择 LoRA，调整旁边的滑动条控制权重（推荐 0.5 - 0.8），点击生成即可叠加画风。
3️⃣ 局部重绘与内置编辑器 (Inpainting & Edit)
遇到一张背景完美但人物闭眼的图？点击左侧栏的 [📂 打开历史图库并编辑]。
选中需要修改的图片，在弹出的 Pro 级编辑器中选择 [画笔 (遮罩模式)]。
将画笔调粗，把人物脸部涂白（或者涂黑），然后点击 [✅ 保存并发送至主界面]。
此时遮罩和原图已自动载入主界面的“参考图”中。
修改提示词为“睁开的眼睛”，重绘幅度拉到 0.6，点击生成，即可无缝修复！
4️⃣ ADetailer 与 ControlNet
勾选 ADetailer：生成全身照时强烈建议勾选，进度条走完后系统会在后台自动识别脸部并进行高清修复。
开启姿态控制：勾选“启用骨骼约束”，放入一张参考动作图，引擎会自动提取骨架（首次提取会自动下载权重，请保持网络畅通），生成的角色将完美还原该动作。
