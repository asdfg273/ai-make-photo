# ui/disclaimer.py
"""
📜 免责声明与用户协议
- 首次启动强制弹出全局免责声明
- 首次使用声音克隆功能弹出专项声明
- 状态持久化到 config.json
"""

import os
import json
import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTextBrowser, QCheckBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt
import logging

logger = logging.getLogger(__name__)

# Determine project root robustly (works for source and frozen builds)
_IS_FROZEN = getattr(sys, 'frozen', False)
if _IS_FROZEN:
    _PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 免责声明版本号 - 修改文案后升级此号可强制用户重新同意
DISCLAIMER_VERSION = "1.0.0"
VOICE_CLONE_VERSION = "1.0.0"

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.json")


# ==================== 文案 ====================
GLOBAL_DISCLAIMER_HTML = """
<div style="color:#dadbdf; font-size:13px; line-height:1.7;">
<h2 style="color:#4a9eff; margin-top:0;">欢迎使用 AI 绘画工作站 v5.0</h2>
<p>本软件是基于开源模型的 AI 内容创作工具,在使用前请了解:</p>

<h3 style="color:#ffb84a;">【关于本软件】</h3>
<ul>
<li>本软件为个人开源项目</li>
<li>完全离线运行,不上传任何用户数据</li>
<li>所有模型均由用户自行下载,本软件不预置任何受版权保护的内容</li>
</ul>

<h3 style="color:#ffb84a;">【关于生成内容】</h3>
<p>AI 生成内容的版权归属请遵守当地法律法规。<br>
<b style="color:#ff7a7a;">请勿使用本软件生成以下内容:</b></p>
<ul>
<li>侵犯他人肖像权、著作权、名誉权的图像/音频/视频</li>
<li>涉及未成年人的不当内容</li>
<li>用于诈骗、造谣、冒充他人身份的内容</li>
<li>违反当地法律法规的其他内容</li>
</ul>

<h3 style="color:#ffb84a;">【关于声音克隆】</h3>
<p>启用日语配音等声音克隆功能时会有额外提示。<br>
声音克隆功能仅供个人学习和授权范围内使用。<br>
<b style="color:#ff7a7a;">严禁克隆真实人物声音用于未经同意的用途。</b></p>

<h3 style="color:#ffb84a;">【责任声明】</h3>
<p>使用本软件即表示您理解并同意:<br>
您对使用本软件生成的所有内容负全责,<br>
开发者不对用户的使用行为及产生的后果承担责任。</p>
</div>
"""

VOICE_CLONE_HTML = """
<div style="color:#dadbdf; font-size:13px; line-height:1.7;">
<h2 style="color:#4a9eff; margin-top:0;">🎙️ 声音克隆功能说明</h2>

<p>您即将使用基于 <b>GPT-SoVITS</b> 的零样本声音克隆功能。</p>

<h3 style="color:#4aff88;">请确保您上传的参考音频:</h3>
<ul>
<li>✅ 是您本人的录音,或</li>
<li>✅ 已获得原声者的明确授权,或</li>
<li>✅ 来自公共领域 / CC0 授权的音色库</li>
</ul>

<h3 style="color:#ff7a7a;">禁止用于:</h3>
<ul>
<li>❌ 未经授权克隆他人(尤其名人、艺人)声音</li>
<li>❌ 冒充身份、诈骗、诽谤等违法用途</li>
</ul>

<p style="color:#ffb84a;">
首次使用需下载 GPT-SoVITS 引擎 (~50MB) 和预训练模型 (~2GB),<br>
请保持网络畅通。
</p>
</div>
"""


# ==================== 配置读写 ====================
def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"⚠️ 保存 config.json 失败: {e}")


# ==================== 通用弹窗 ====================
class DisclaimerDialog(QDialog):
    """
    通用免责声明对话框
    :param title: 窗口标题
    :param html_content: HTML 富文本内容
    :param agree_text: 复选框文本
    :param confirm_text: 确认按钮文本
    :param cancel_text: 取消按钮文本
    :param min_read_time: 强制阅读秒数(0 表示不限制)
    """
    def __init__(self, title, html_content,
                 agree_text="我已阅读并同意上述条款",
                 confirm_text="同意并继续",
                 cancel_text="退出",
                 min_read_time=0,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 520)
        from ui.theme import PALETTE
        self.setStyleSheet(f"""
            QDialog {{ background:{PALETTE['bg']}; }}
            QCheckBox {{ color:{PALETTE['fg']}; font-size:13px; padding:6px; }}
            QCheckBox::indicator {{ width:18px; height:18px; }}
            QPushButton {{
                background:{PALETTE['bg_soft']}; color:{PALETTE['fg']}; border:none;
                padding:10px 24px; border-radius:6px; font-size:13px;
                font-weight:bold; min-width:120px;
            }}
            QPushButton:hover {{ background:#2d3d4d; }}
            QPushButton#btnConfirm {{
                background:{PALETTE['accent']}; color:#fff;
            }}
            QPushButton#btnConfirm:hover {{ background:{PALETTE['accent_hi']}; }}
            QPushButton#btnConfirm:disabled {{
                background:{PALETTE['bg_soft']}; color:{PALETTE['fg_mute']};
            }}
            QTextBrowser {{
                background:{PALETTE['bg_soft']}; border:1px solid #2d3d4d;
                border-radius:6px; padding:12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # 内容区
        browser = QTextBrowser()
        browser.setHtml(html_content)
        browser.setOpenExternalLinks(True)
        layout.addWidget(browser, 1)

        # 同意复选框
        self.chk_agree = QCheckBox(agree_text)
        layout.addWidget(self.chk_agree)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_cancel = QPushButton(cancel_text)
        self.btn_confirm = QPushButton(confirm_text)
        self.btn_confirm.setObjectName("btnConfirm")
        self.btn_confirm.setEnabled(False)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_confirm)
        layout.addLayout(btn_row)

        # 信号
        self.chk_agree.toggled.connect(self.btn_confirm.setEnabled)
        self.btn_confirm.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # 禁用关闭按钮的 X(强制选择)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)


# ==================== 对外接口 ====================
def check_global_disclaimer(parent=None) -> bool:
    """
    检查全局免责声明状态
    :return: True 表示用户同意,False 表示用户拒绝(应退出程序)
    """
    cfg = _load_config()
    agreed_version = cfg.get("disclaimer_agreed_version", "")

    if agreed_version == DISCLAIMER_VERSION:
        return True  # 已同意当前版本,跳过

    dlg = DisclaimerDialog(
        title="用户协议与免责声明",
        html_content=GLOBAL_DISCLAIMER_HTML,
        parent=parent,
    )

    if dlg.exec() == QDialog.DialogCode.Accepted:
        cfg["disclaimer_agreed_version"] = DISCLAIMER_VERSION
        _save_config(cfg)
        return True

    return False


def check_voice_clone_consent(parent=None) -> bool:
    """
    检查声音克隆功能授权状态
    :return: True 表示用户同意,False 表示用户拒绝
    """
    cfg = _load_config()
    agreed_version = cfg.get("voice_clone_agreed_version", "")

    if agreed_version == VOICE_CLONE_VERSION:
        return True

    dlg = DisclaimerDialog(
        title="声音克隆功能授权",
        html_content=VOICE_CLONE_HTML,
        agree_text="我已阅读并同意仅在合法授权范围内使用",
        confirm_text="确认下载并使用",
        cancel_text="取消",
        parent=parent,
    )

    if dlg.exec() == QDialog.DialogCode.Accepted:
        cfg["voice_clone_agreed_version"] = VOICE_CLONE_VERSION
        _save_config(cfg)
        return True

    return False