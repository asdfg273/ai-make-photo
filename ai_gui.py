import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import datetime
import json
import gc # 用于清理内存
import jieba
from deep_translator import GoogleTranslator
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# ==========================================
# 核心设置 (文件夹路径)
# ==========================================
MODELS_DIR = "models" # 模型存放文件夹
OUTPUT_DIR = "photo"  # 图片输出文件夹

# ==========================================
# 第一部分：翻译与词典系统 (完全保留)
# ==========================================
DICT_FILE = "zh_to_en_dict.json"
default_dict = {
    "高质量": "best quality", "大师作": "masterpiece", "细节": "highly detailed",
    "女孩": "1girl", "男孩": "1boy", "单人": "solo",
    "低画质": "low quality", "最差画质": "worst quality", "坏手": "bad hands"
}

dictionary = {}

def load_dictionary():
    global dictionary
    if os.path.exists(DICT_FILE):
        try:
            with open(DICT_FILE, 'r', encoding='utf-8') as f: dictionary = json.load(f)
        except: dictionary = {}
    for k, v in default_dict.items():
        if k not in dictionary: dictionary[k] = v
    for word in dictionary.keys(): jieba.add_word(word)

def save_dictionary():
    try:
        with open(DICT_FILE, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=4)
    except: pass

def translate_to_english(text):
    if not text.strip(): return ""
    has_new_word = False
    text = text.replace('，', ',').replace('。', ',').replace('\n', ',')
    segments = [seg.strip() for seg in text.split(',') if seg.strip()]
    final_parts = []
    for seg in segments:
        if all(ord(c) < 128 for c in seg): 
            final_parts.append(seg)
            continue
        if seg in dictionary: 
            final_parts.append(dictionary[seg])
            continue
        words = jieba.lcut(seg)
        seg_trans_words = []
        for word in words:
            word = word.strip()
            if not word or word in ["在", "的", "了", "着", "地", "个", "是", "不要"]: continue 
            if word in dictionary:
                seg_trans_words.append(dictionary[word])
            else:
                try:
                    res = GoogleTranslator(source='zh-CN', target='en').translate(word)
                    if res and res.strip():
                        seg_trans_words.append(res)
                        dictionary[word] = res
                        jieba.add_word(word) 
                        has_new_word = True
                    else: seg_trans_words.append(word)
                except: seg_trans_words.append(word)
        if seg_trans_words: final_parts.append(" ".join(seg_trans_words))
    if has_new_word: save_dictionary()
    return ", ".join(final_parts)


# ==========================================
# 第二部分：AI 模型后台管线 (支持动态切换模型)
# ==========================================
txt2img_pipe = None
img2img_pipe = None
current_loaded_model = None # 记录当前内存里是什么模型

def load_models(model_filename):
    """根据选择的模型动态加载。如果换了模型，自动清理旧内存"""
    global txt2img_pipe, img2img_pipe, current_loaded_model
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # 如果当前内存里的模型和要画的图不是同一个，或者还没加载过
    if current_loaded_model != model_filename:
        print(f"准备加载新模型: {model_filename} ...")
        
        # 释放旧模型的内存 (防止连续换模型导致内存爆炸)
        txt2img_pipe = None
        img2img_pipe = None
        gc.collect() 
        
        # 加载新模型
        txt2img_pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32, 
            use_safetensors=True,
            safety_checker=None,       
            requires_safety_checker=False
        )
        txt2img_pipe.safety_checker = None
        txt2img_pipe.to("cpu")
        txt2img_pipe.enable_attention_slicing()
        
        # 瞬间生成图生图管线
        img2img_pipe = StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)
        img2img_pipe.safety_checker = None
        
        current_loaded_model = model_filename


# ==========================================
# 第三部分：满血版 GUI 界面 (含模型管理与建议)
# ==========================================
class AIDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("私人 AI 绘画工作站 (多模型终极版)")
        self.root.geometry("720x920") # 加长界面
        self.ref_image_path = None
        
        load_dictionary()

        # 确保基础文件夹存在
        for folder in [OUTPUT_DIR, MODELS_DIR]:
            if not os.path.exists(folder): os.makedirs(folder)

        # ========================================
        # 1. 模型选择区域
        # ========================================
        frame_model = tk.LabelFrame(root, text="📦 模型库 (请将 .safetensors 放入 models 文件夹)", font=("微软雅黑", 10, "bold"), fg="#D35400")
        frame_model.pack(fill="x", padx=10, pady=5, ipadx=5, ipady=5)
        
        self.combo_model = ttk.Combobox(frame_model, state="readonly", width=45)
        self.combo_model.grid(row=0, column=0, padx=10, pady=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_select)

        self.lbl_model_size = tk.Label(frame_model, text="大小: 未知", fg="gray")
        self.lbl_model_size.grid(row=0, column=1, padx=10, pady=5)

        tk.Button(frame_model, text="🔄 刷新模型列表", command=self.refresh_models).grid(row=0, column=2, padx=10)
        
        self.refresh_models() # 初始化扫描模型

        # ========================================
        # 2. 提示词区域
        # ========================================
        tk.Label(root, text="正面提示词 (你想画什么):", font=("微软雅黑", 10, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.prompt_text = scrolledtext.ScrolledText(root, height=4, width=90)
        self.prompt_text.pack(padx=10, pady=2)
        self.prompt_text.insert(tk.END, "大师作, 杰作, 最高画质, 单人, 一个美丽的女孩, 完美的人体比例, 极其精致的面部, 唯美光影")

        tk.Label(root, text="反面提示词 (你不想出现什么，防崩坏专用):", font=("微软雅黑", 10, "bold")).pack(anchor="w", padx=10)
        self.neg_prompt_text = scrolledtext.ScrolledText(root, height=3, width=90)
        self.neg_prompt_text.pack(padx=10, pady=2)
        self.neg_prompt_text.insert(tk.END, "低画质, 最差画质, 畸形, 扭曲的脸, 糟糕的人体结构, 错误的比例, 缺失的肢体, 多余的肢体, 融合的手指, 血肉模糊")

        # ========================================
        # 3. 基础参数区域
        # ========================================
        frame_settings = tk.Frame(root)
        frame_settings.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_settings, text="分辨率:").grid(row=0, column=0, padx=5, pady=5)
        self.combo_res = ttk.Combobox(frame_settings, values=["512x512", "512x768 (竖屏)", "768x512 (横屏)"], state="readonly", width=15)
        self.combo_res.current(0)
        self.combo_res.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame_settings, text="生成数量:").grid(row=0, column=2, padx=5, pady=5)
        self.spin_count = ttk.Spinbox(frame_settings, from_=1, to=20, width=5)
        self.spin_count.set(1)
        self.spin_count.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(frame_settings, text="迭代步数:").grid(row=0, column=4, padx=5, pady=5)
        self.spin_steps = ttk.Spinbox(frame_settings, from_=10, to=100, width=5)
        self.spin_steps.set(30)
        self.spin_steps.grid(row=0, column=5, padx=5, pady=5)

        # ========================================
        # 4. 参考图 (图生图) 区域
        # ========================================
        frame_img2img = tk.LabelFrame(root, text="参考图设置 (图生图 / 可选)", font=("微软雅黑", 10, "bold"))
        frame_img2img.pack(fill="x", padx=10, pady=5, ipadx=5, ipady=5)

        self.btn_select_img = tk.Button(frame_img2img, text="🖼️ 选择参考图", command=self.select_reference_image)
        self.btn_select_img.grid(row=0, column=0, padx=10, pady=5)
        self.btn_clear_img = tk.Button(frame_img2img, text="❌ 清除参考图", command=self.clear_reference_image, state=tk.DISABLED)
        self.btn_clear_img.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_img_path = tk.Label(frame_img2img, text="当前未选择参考图 (默认使用纯文字生图)", fg="gray")
        self.lbl_img_path.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        tk.Label(frame_img2img, text="重绘幅度 (0.1微调 ~ 1.0巨变):").grid(row=1, column=0, columnspan=2, pady=5, sticky="e")
        self.scale_strength = tk.Scale(frame_img2img, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=150)
        self.scale_strength.set(0.6)
        self.scale_strength.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # ========================================
        # 5. 【炼丹秘籍】建议区域
        # ========================================
        frame_tips = tk.LabelFrame(root, text="💡 炼丹秘籍与防崩坏建议", font=("微软雅黑", 9, "bold"), fg="#2980B9")
        frame_tips.pack(fill="x", padx=10, pady=5, ipadx=5, ipady=5)
        tips_text = (
            "1. 避免血肉模糊/衣物融合：如果你使用图生图功能想大改画面结构（如脱衣/换装），\n"
            "   【重绘幅度】必须拉到 0.75 以上！否则AI不敢摧毁原图轮廓，会导致极度扭曲。\n"
            "2. 想要瑟瑟 (NSFW)：模型必须选对。强烈推荐下载 Anything V5 (二次元) 或 \n"
            "   ChilloutMix (真人) 放入 models 文件夹。并且要在提示词里明确写出结构词。\n"
            "3. 纯文字生图出奇迹：图生图限制太大，建议先多用【纯文字生图】测试模型效果！"
        )
        tk.Label(frame_tips, text=tips_text, justify="left", fg="#34495E").pack(anchor="w")

        # ========================================
        # 6. 按钮与操作区域
        # ========================================
        frame_actions = tk.Frame(root)
        frame_actions.pack(pady=10)

        self.btn_generate = tk.Button(frame_actions, text="🚀 一键开始生成", font=("微软雅黑", 14, "bold"), bg="#4CAF50", fg="white", command=self.start_generation_thread)
        self.btn_generate.grid(row=0, column=0, padx=20, ipadx=20, ipady=10)

        self.btn_open_folder = tk.Button(frame_actions, text="📂 打开输出文件夹", font=("微软雅黑", 12), command=self.open_output_folder)
        self.btn_open_folder.grid(row=0, column=1, padx=20, ipady=5)

        # ========================================
        # 7. 进度条与状态区域
        # ========================================
        self.status_var = tk.StringVar()
        self.status_var.set("状态：准备就绪。")
        tk.Label(root, textvariable=self.status_var, fg="blue", font=("微软雅黑", 10)).pack(pady=2)

        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=650, mode="determinate")
        self.progress_bar.pack(pady=5)
        self.progress_text = tk.StringVar()
        self.progress_text.set("进度: 0 / 0 步")
        tk.Label(root, textvariable=self.progress_text, font=("微软雅黑", 9)).pack()


    # --- 模型文件扫描管理 ---
    def refresh_models(self):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')]
        if not models:
            self.combo_model['values'] = ["找不到模型！请把模型放入 models 文件夹"]
            self.combo_model.current(0)
            self.lbl_model_size.config(text="大小: 0 GB", fg="red")
        else:
            self.combo_model['values'] = models
            self.combo_model.current(0)
            self.on_model_select(None) # 更新大小显示

    def on_model_select(self, event):
        selected = self.combo_model.get()
        if selected and not selected.startswith("找不到"):
            filepath = os.path.join(MODELS_DIR, selected)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            size_gb = size_mb / 1024
            
            if size_gb >= 1.0:
                self.lbl_model_size.config(text=f"大小: {size_gb:.2f} GB", fg="green")
            else:
                self.lbl_model_size.config(text=f"大小: {size_mb:.0f} MB", fg="green")


    # --- 界面交互函数 ---
    def select_reference_image(self):
        file_path = filedialog.askopenfilename(title="选择参考图", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.ref_image_path = file_path
            self.lbl_img_path.config(text=f"已选择: {os.path.basename(file_path)}", fg="green")
            self.btn_clear_img.config(state=tk.NORMAL)

    def clear_reference_image(self):
        self.ref_image_path = None
        self.lbl_img_path.config(text="当前未选择参考图 (默认使用纯文字生图)", fg="gray")
        self.btn_clear_img.config(state=tk.DISABLED)

    def open_output_folder(self):
        try: os.startfile(os.path.abspath(OUTPUT_DIR))
        except Exception as e: messagebox.showerror("错误", f"无法打开: {e}")

    # --- 后台画图核心函数 ---
    def start_generation_thread(self):
        selected_model = self.combo_model.get()
        if selected_model.startswith("找不到") or not selected_model:
            messagebox.showwarning("警告", "请先下载一个 .safetensors 模型放入 models 文件夹！")
            return

        self.btn_generate.config(state=tk.DISABLED, text="正在后台努力画图中...")
        self.btn_select_img.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        threading.Thread(target=self.generation_task, daemon=True).start()

    def update_progress_gui(self, step, total_steps):
        self.progress_bar["maximum"] = total_steps
        self.progress_bar["value"] = step
        self.progress_text.set(f"当前图片渲染进度: {step} / {total_steps} 步")

    def generation_task(self):
        try:
            # 1. 抓取参数
            raw_prompt = self.prompt_text.get("1.0", tk.END).strip()
            raw_neg = self.neg_prompt_text.get("1.0", tk.END).strip()
            res_choice = self.combo_res.get()
            width, height = 512, 512
            if "512x768" in res_choice: width, height = 512, 768
            if "768x512" in res_choice: width, height = 768, 512
            img_count = int(self.spin_count.get())
            total_steps = int(self.spin_steps.get())
            strength = float(self.scale_strength.get())
            selected_model = self.combo_model.get()

            # 2. 翻译与加载
            self.status_var.set("状态：正在翻译咒语并查阅词典...")
            en_prompt = translate_to_english(raw_prompt)
            en_neg = translate_to_english(raw_neg)

            self.status_var.set(f"状态：正在挂载模型 [{selected_model}]，换模型时会较慢，请稍等...")
            load_models(selected_model) # 动态加载选择的模型

            def pipe_callback(step, timestep, latents):
                self.root.after(0, self.update_progress_gui, step + 1, total_steps)

            # 3. 循环画图
            for i in range(img_count):
                self.status_var.set(f"状态：CPU火力全开，正在渲染 第 {i+1} / {img_count} 张图片...")
                self.root.after(0, self.update_progress_gui, 0, total_steps) 
                
                with torch.no_grad():
                    if self.ref_image_path is None:
                        image = txt2img_pipe(
                            prompt=en_prompt, negative_prompt=en_neg,
                            num_inference_steps=total_steps, guidance_scale=7.0,
                            width=width, height=height,
                            callback=pipe_callback, callback_steps=1
                        ).images[0]
                    else:
                        init_img = Image.open(self.ref_image_path).convert("RGB").resize((width, height))
                        image = img2img_pipe(
                            prompt=en_prompt, negative_prompt=en_neg,
                            image=init_img, strength=strength,
                            num_inference_steps=total_steps, guidance_scale=7.0,
                            callback=pipe_callback, callback_steps=1
                        ).images[0]

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"out_{timestamp}_{i+1}.png"
                save_path = os.path.join(OUTPUT_DIR, filename)
                image.save(save_path)
                print(f"✅ 图片已保存: {save_path}")

            self.status_var.set(f"状态：🎉 完美搞定！")
            self.root.after(0, self.update_progress_gui, total_steps, total_steps)
            messagebox.showinfo("生成完毕", f"全部 {img_count} 张图片已存入 '{OUTPUT_DIR}' 文件夹！")

        except Exception as e:
            self.status_var.set("状态：发生错误，停止工作。")
            messagebox.showerror("报错了", f"错误信息：\n{str(e)}")
            
        finally:
            self.btn_generate.config(state=tk.NORMAL, text="🚀 一键开始生成")
            self.btn_select_img.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIDrawerApp(root)
    root.mainloop()