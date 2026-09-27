# 第三方组件与许可声明（THIRD_PARTY_NOTICES）

> 本文件用于**商业许可合规自查**，不是法律意见。涉及商用/闭源分发前，
> 请务必让法务确认下表标 ⚠️ 的条目。
>
> 生成时间：2026-09；依据环境实测版本（`venv`）+ 各上游仓库 LICENSE 原文核对。

---

## 一、一句话结论（给决策用）

| 结论 | 内容 |
|---|---|
| 🔴 **最大约束** | `ultralytics`（AGPL-3.0）与 `ChatTTS`（AGPL-3.0）是**强 copyleft**。只要你的发行物构成"衍生作品"，整个项目就必须以 AGPL-3.0 发布并附完整源码。**这与"闭源商用"不兼容**；与"开源 + 打赏"兼容（前提是项目本身也以 AGPL-3.0 发布）。 |
| 🟠 次要约束 | 底模权重（SD1.5 / SDXL 系）多为 **CreativeML Open RAIL-M**，带**使用限制条款**（禁止违法/伤害性用途等），不是传统开源许可；商用需遵守其 Use-based Restrictions。 |
| ✅ 无约束 | 项目自研代码、以及 MIT/BSD/Apache-2.0 的多数依赖（torch、diffusers、transformers、PyQt6 商业许可除外见下）。 |
| ⚠️ **当前缺口** | 仓库**没有 LICENSE 文件**，也没有任何第三方声明。开源发布前必须补齐，否则"开源"在法律上不成立（默认保留全部权利）。 |

---

## 二、Python 依赖许可（实测版本）

| 包 | 版本 | 许可 | 商用影响 |
|---|---|---|---|
| torch / torchvision / torchaudio | 2.11.0+cu128 | BSD-3-Clause | ✅ 宽松 |
| diffusers | 0.40.0 | Apache-2.0 | ✅ |
| transformers | 5.16.1 | Apache-2.0 | ✅ |
| accelerate | 1.11.0 | Apache-2.0 | ✅ |
| peft | 0.19.1 | Apache-2.0 | ✅ |
| safetensors | 0.8.0 | Apache-2.0 | ✅ |
| compel | 2.0.3 | MIT | ✅ |
| huggingface-hub | 1.24.0 | Apache-2.0 | ✅ |
| modelscope | 1.37.1 | Apache-2.0（未在元数据声明，见上游仓库） | ✅ |
| numpy | 2.2.6 | BSD-3-Clause | ✅ |
| pillow | 10.4.0 | HPND（MIT 类） | ✅ |
| opencv-contrib-python | 4.13.0 | Apache-2.0 | ✅ |
| scikit-image | 0.25.2 | BSD-3-Clause | ✅ |
| soundfile | 0.14.0 | BSD-3-Clause | ✅ |
| imageio / imageio-ffmpeg | 2.37.3 / 0.6.0 | BSD-2-Clause | ✅ |
| av (PyAV) | 17.0.1 | BSD-3-Clause | ✅ |
| deep-translator | 1.11.4 | MIT | ✅ |
| jieba | 0.42.1 | MIT | ✅ |
| onnxruntime-gpu | 1.23.2 | MIT | ✅ |
| qwen-vl-utils | 0.0.14 | Apache-2.0 | ✅ |
| controlnet_aux | 0.0.10 | Apache-2.0 | ✅ |
| QDarkStyle | 3.2.3 | MIT | ✅ |
| PyQt6 | 6.11.0 | **GPL-3.0 / 商业双许可**（Riverbank） | ⚠️ **见第三节** |
| **ultralytics** | 8.4.46 | **AGPL-3.0** | 🔴 **见第四节** |

---

## 三、⚠️ PyQt6：GPL-3.0 与商业许可的取舍

PyQt6 由 Riverbank Computing 以 **GPL v3 + 商业许可** 双轨发布（注意：不是 LGPL，
Qt 本身才是 LGPL，PyQt 的 Python 绑定是 GPL）。

- 若你的项目以 **GPL/AGPL 兼容许可**发布 → 可以免费使用 PyQt6。
- 若你想**闭源商用** → 必须向 Riverbank 购买 PyQt 商业许可，
  或改用 **PySide6（LGPL-3.0，Qt 官方绑定）**。
  PySide6 的 API 与 PyQt6 高度相似，迁移成本主要在信号/槽与少量枚举命名。

> 结论：既然已决定 AGPL/开源路线，PyQt6 免费可用；
> **但"开源 + 打赏"如果将来想加闭源增值模块，PyQt6 会成为障碍。**

---

## 四、🔴 ultralytics（AGPL-3.0）：最大的合规风险点

**用在哪**：`utils/image_processor.py` 的 ADetailer 人脸/手部/全身检测（YOLOv8）。

**为什么是风险**：

1. AGPL-3.0 的传染性**强于 GPL**：不仅分发要开源，**通过网络提供服务**也要向用户提供源码。
2. 你的程序 `import ultralytics` 并调用其推理，通常被认定为"结合作品"→
   整个项目需以 AGPL-3.0 发布。
3. `models/adetailer/face_yolov8n.pt`、`hand_yolov8n.pt` 是 Ultralytics 训练的
   YOLOv8 权重，**同样受 AGPL-3.0 约束**（当前 `.gitignore` 已排除，不随仓库分发，
   由扩展市场从 HuggingFace 下载 —— 这一点降低了分发风险，但没有消除"程序调用
   AGPL 库"的问题）。

**三个可选方案**（按推荐度排序）：

| 方案 | 做法 | 代价 |
|---|---|---|
| ① 接受 AGPL | 整个项目以 AGPL-3.0 发布，附完整源码 | 与"开源+打赏"完全兼容；**不能闭源** |
| ② 换检测器 | 把 YOLO 导出为 ONNX，推理改用 `onnxruntime`（MIT），移除 `ultralytics` 依赖 | 需重写 `_yolo_detect()` 的前后处理（约 100 行）；权重许可仍需单独确认 |
| ③ 购买商业许可 | 向 Ultralytics 购买 Enterprise License | 需付费，按年/按规模 |

> 其它同类替代（全 Apache/MIT）：`insightface` 的 SCRFD（人脸检测，MIT 代码 +
> 非商用权重需注意）、OpenCV DNN 的 YuNet（**Apache-2.0，且轻量**）—— YuNet
> 是最省事的替换路径，`cv2.FaceDetectorYN` 直接可用，无需额外依赖。

---

## 五、🟠 模型权重的许可（不随仓库分发，但用户会用到）

| 模型 | 来源 | 许可 | 说明 |
|---|---|---|---|
| SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML Open RAIL-M | 带 Use-based Restrictions：不得用于违法、伤害、误导等用途；需随附许可副本 |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | CreativeML Open RAIL++-M | 同上 |
| Pony / Illustrious / NoobAI 等衍生 | 各作者 | 多数继承 RAIL-M 或自定条款 | **逐一下载页确认**，部分禁止商用 |
| AnimateDiff motion adapter / motion lora | `guoyww/*` | Apache-2.0（仓库） | ✅ |
| IP-Adapter | `h94/IP-Adapter` | Apache-2.0 | ✅ |
| ControlNet（SD1.5） | `lllyasviel/sd-controlnet-*` | Apache-2.0 | ✅ |
| ControlNet（SDXL） | `diffusers/controlnet-*-sdxl-1.0`、`thibaud/controlnet-openpose-sdxl-1.0` | Apache-2.0 / 见仓库 | ✅ |
| ADetailer 权重 | `Bingsu/adetailer` | **AGPL-3.0**（YOLOv8 衍生） | 🔴 同第四节 |
| OpenPose Annotators | `lllyasviel/Annotators` | 见仓库（含 CMU 等条款） | ⚠️ 商用前确认 |
| Depth 预估 | `Intel/dpt-hybrid-midas` | Apache-2.0 | ✅ |
| ChatTTS | `2Noise/ChatTTS` | **AGPL-3.0** | 🔴 同第四节 |
| GPT-SoVITS | `lj1995/GPT-SoVITS` | **MIT**（已核对 `third_party/GPT-SoVITS/LICENSE`） | ✅ 宽松 |
| NLLB-200 | `facebook/nllb-200-distilled-600M` | CC-BY-NC-4.0 | 🔴 **禁止商用**！当前注册在扩展市场里（`nllb_600m`），商用版本应移除 |
| Qwen2-VL / Qwen2.5-VL / Qwen3.5 | ModelScope | 多为 Apache-2.0 / Qwen 自有条款 | ⚠️ 逐档确认（Qwen 部分档位有额外条款） |

---

## 六、随包工具（当前 `tools/` 已被 .gitignore 排除，不随仓库分发）

| 工具 | 许可 | 说明 |
|---|---|---|
| RIFE `rife-ncnn-vulkan.exe` | **MIT**（已核对上游 LICENSE） | ✅ |
| Real-ESRGAN `realesrgan-ncnn-vulkan.exe` + models | **BSD-3-Clause**（已核对上游 LICENSE） | ✅ |
| ncnn / Vulkan 运行库 | BSD-3-Clause / Apache-2.0 | ✅ |
| FFmpeg（经 `imageio-ffmpeg` 分发） | **LGPL-2.1+ 或 GPL**（取决于编译选项） | ⚠️ 若分发的是 GPL 版二进制，需遵守 GPL；`imageio-ffmpeg` 提供的是 LGPL 构建 |
| `third_party/GPT-SoVITS` | MIT（见内部 LICENSE）；其 `BigVGAN` 子目录另有 **CC-BY-4.0 / MIT 混合** 许可（见 `incl_licenses/`） | ⚠️ 逐个子目录确认 |
| `third_party/GPT-SoVITS/TEMP/ja/open_jtalk_dic` | **Modified BSD**（见 `COPYING`） | ✅ 需附声明 |

---

## 七、建议的合规动作清单

1. **补 `LICENSE`**：选定主许可。若继续用 ultralytics/ChatTTS →
   必须是 **AGPL-3.0**；若想保留未来闭源可能 → 先执行第四节方案②/③，
   并把主许可定为 **Apache-2.0 或 MIT**。
2. **补 `THIRD_PARTY_NOTICES.md`**（本文件）并在 README 链接它。
3. **移除或标注不可商用组件**：`nllb_600m`（CC-BY-NC-4.0）在商用版本中应删除；
   ADetailer/ChatTTS 需在 UI 或文档中显著标注许可与来源。
4. **底模许可随附**：首次使用某底模时提示其许可（RAIL-M 要求随附许可副本），
   可在 `ui/disclaimer.py` 里加一条；二进制分发时把许可文本放进 `licenses/`。
5. **不要预置权重到仓库**（当前 `.gitignore` 已正确排除 `models/*`、`tools/`）——
   继续保持。这能显著降低"再分发 AGPL 权重"的风险。
6. **打赏不等于商用**：接受捐赠通常不构成"商业分发"，但一旦提供付费功能/
   企业版/云服务，AGPL 第 13 条（网络交互必须提供源码）即被触发。

---

## 八、附：AI 生成内容的版权提示（与你无关但用户会问）

多数司法辖区（含中国、美国）目前认为**纯 AI 生成内容难以获得著作权保护**，
用户拿生成图商用前应自行评估。建议在 `ui/disclaimer.py` 的免责声明里加一句，
减少后续纠纷。
