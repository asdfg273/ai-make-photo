# main.py

import os
import threading
import warnings

# 忽略烦人的底层库警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ttkbootstrap as tb
from ttkbootstrap.constants import * 

# === 本地核心模块导入 ===
from translation_service import TranslationService
from config_manager import AppConfig

# === 核心挂载组件 (Mixins) ===
from utils.ui_builder import UIMixin
from utils.app_events import EventMixin
from utils.app_generation import GenerationMixin
from utils.system_utils import log_system_info, logger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "photo")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class AIDesktopApp(tb.Window, UIMixin, EventMixin, GenerationMixin):
    def __init__(self):
        log_system_info()
        logger.info("🚀 AI 绘画工作站启动...")
        super().__init__(themename="darkly", title="AI 绘画工作站 (Pro)", size=(1400, 850))
        if not os.path.exists(OUTPUT_DIR): 
            os.makedirs(OUTPUT_DIR)
        
        self.translator = TranslationService()
        self.ai = None  
        self.is_generating = False
        self.cancel_flag = False
        self.ref_image_path = None
        self.mask_image_path = None
        self.pose_image_path = None
        self.current_generated_path = None
        
        # 1. 加载配置并构建UI
        self.config = AppConfig.load()
        self.setup_ui()    
        self.apply_config_to_ui() # (此函数若在 ui_builder.py，正常调用即可)
        
        # 2. 拦截窗口关闭事件，在退出前保存配置！
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 3. 初始UI状态
        if hasattr(self, "btn_gen"):
            self.btn_gen.config(state=DISABLED, text="🚀 AI 引擎预热中...")
        self.lbl_status.config(text="⏳ 正在后台加载大模型与底层环境，请稍候...", foreground="yellow")
        
        # 4. 异步启动底层 AI 引擎
        threading.Thread(target=self.async_init_ai, daemon=True).start()

    # --- 生命周期：引擎异步预热 ---
    def async_init_ai(self):
        print("👉 [系统预热] 后台开始导入重型库(PyTorch/Diffusers)...")
        global torch  
        import torch
        from model_manager import ModelManager 
        
        print("👉 [系统预热] 依赖导入完成，正在加载大模型...")
        self.ai = ModelManager()  
        self.after(0, self.on_ai_loaded)

    # --- 生命周期：引擎就绪回调 ---
    def on_ai_loaded(self):
        self.refresh_models()
        self.refresh_lora_by_model()
        
        if hasattr(self, "btn_gen"):
            self.btn_gen.config(state=NORMAL, text="🚀 开始生成")
        self.lbl_status.config(text="✅ 系统就绪。等待生成指令...", foreground="#00FF00")
        print("👉 [系统预热] 引擎就绪！")
        
        available_devices = ["自动 (Auto)"]
        if torch.cuda.is_available():
            available_devices.append("CUDA (Nvidia GPU)")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append("MPS (Apple Silicon Mac)")
        available_devices.append("CPU (纯内存-极慢)")
        
        self.combo_device.config(state="readonly", values=available_devices)
        try:
            self.combo_device.set(self.config.device_preference)
        except Exception:
            self.combo_device.current(0)

    # --- 生命周期：程序退出保存 ---
    def on_closing(self):
        print("💾 正在保存配置并退出...")
        try:
            self.config.default_steps = int(self.spin_steps.get())
            self.config.default_strength = float(self.scale_str.get())
            lora_list = []
            for i in range(3):
                lname = self.combo_loras[i].get()
                lweight = self.scale_loras[i].get()
                lora_list.append((lname, float(lweight)))
            self.config.default_loras = lora_list
            self.config.device_preference = self.combo_device.get()
            self.config.save()
        except:
            pass
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = AIDesktopApp()
    app.mainloop()