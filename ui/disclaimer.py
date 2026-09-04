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
<h2 style="color:#4a9eff; margin-top:0;">AI 绘画工作站 v6.0 用户协议与免责声明</h2>
<p>欢迎使用本软件，在使用前请仔细阅读全部条款，首次启动本软件即视为您已完全理解并同意本声明所有内容。若您不同意任何条款，请立即停止使用本软件。</p>

<h3 style="color:#ffb84a;">一、软件性质与开源协议</h3>
<ul>
<li>本软件为个人开源项目，遵循 [填写你选择的协议，如Apache License 2.0] 开源协议，代码公开于 [填写GitHub仓库地址]，任何人可在遵守协议要求的前提下使用、修改、分发本软件代码。</li>
<li>本软件不预置任何模型、LoRA、插件等第三方资源，用户自行下载的模型、扩展资源的版权归属、使用限制由用户自行确认，开发者不对第三方资源的合法性负责。</li>
<li>未经开发者书面授权，任何人不得使用本软件的名称、Logo、商标进行商业推广或背书。</li>
</ul>

<h3 style="color:#ffb84a;">二、AI 生成内容标识合规要求</h3>
<p>根据 2025 年 9 月 1 日起施行的《人工智能生成合成内容标识办法》及相关配套标准，若您将本软件生成的图片、音频、视频等内容公开传播（包括但不限于发布到社交平台、电商平台、新闻媒体等场景），必须履行以下义务：</p>
<ul>
<li><b>显式标识：</b>在生成内容的显著位置添加“本内容由 AI 生成”等清晰可感知的标识（图片可在角落添加半透明水印，视频需在片头 3 秒内显示标识且全程保留角标）。</li>
<li><b>隐式标识：</b>保留本软件自动在文件元数据中添加的生成信息（包含生成工具名称、生成时间等内容），不得恶意删除、篡改、隐匿相关标识。</li>
</ul>
<p style="color:#ff7a7a; background:#2d3d4d; padding:6px 12px; border-radius:4px;">
⚠️ 若未履行标识义务，可能面临平台内容下架、账号限流、封禁等处罚；情节严重的，个人最高可处 5000 元罚款，单位最高可处 10 万元罚款；故意去除、篡改标识的，最高可处 50 万元罚款，构成犯罪的还将依法追究刑事责任。
</p>

<h3 style="color:#ffb84a;">三、数据隐私与第三方组件说明</h3>
<ul>
<li>本软件完全离线运行，不收集、不上传、不存储任何用户数据，包括提示词、生成内容、硬件信息、操作记录等。</li>
<li>若用户自行安装第三方插件、调用外部 API，产生的数据隐私、合规责任由用户自行承担，开发者不对第三方组件的行为负责。</li>
</ul>

<h3 style="color:#ffb84a;">四、禁止使用场景</h3>
<p><b style="color:#ff7a7a;">严禁使用本软件生成以下内容，否则由此产生的所有法律责任由用户自行承担：</b></p>
<ul>
<li>侵犯他人肖像权、著作权、名誉权的图像/音频/视频；</li>
<li>涉及未成年人的不当内容，包括未成年人软色情、暴力、诱导性内容；</li>
<li>用于诈骗、造谣、冒充他人身份、伪造证据的内容；</li>
<li>虚假新闻、虚假信息、误导性内容，扰乱公共秩序；</li>
<li>涉及国家主权、领土完整、民族尊严、社会主义核心价值观的不当内容；</li>
<li>赌博、毒品、邪教、暴力恐怖相关的违法违规内容；</li>
<li>批量生成内容用于垃圾信息传播、恶意营销、网络水军等行为；</li>
<li>未经他人同意克隆真实人物声音、换脸，用于冒充他人身份的内容。</li>
</ul>

<h3 style="color:#ffb84a;">五、未成年人使用规则</h3>
<p>本软件仅限年满 18 周岁的成年人使用。若未成年人确需使用，必须在监护人的全程指导和监督下进行，监护人需对未成年人的使用行为及生成内容承担全部责任。开发者有权对疑似未成年人的使用行为采取限制措施。</p>

<h3 style="color:#ffb84a;">六、生成内容原创性与侵权免责</h3>
<ul>
<li>AI 生成内容具有随机性，可能存在与其他现有作品构成实质性相似的情况，开发者不对生成内容的原创性、非侵权性做任何担保。</li>
<li>用户将生成内容用于商业用途前，需自行进行侵权比对核查，因生成内容侵权产生的法律责任由用户自行承担。</li>
<li>AI 生成内容的版权归属请遵守当地法律法规，若将生成内容用于公开传播或商业目的，请自行评估版权风险，开发者不提供任何商业化授权担保。</li>
</ul>

<h3 style="color:#ffb84a;">七、责任声明与管辖法律</h3>
<ul>
<li>使用本软件即表示您理解并同意：您对使用本软件生成的所有内容负全责，开发者不对用户的使用行为及产生的任何直接或间接后果承担责任。</li>
<li>本软件按“现状”提供，不保证生成内容的准确性、合法性、商业适销性及非侵权性。</li>
<li>除法律明确规定的开发者故意或重大过失导致的责任外，开发者不对用户使用本软件产生的任何损失承担责任。</li>
<li>本声明适用中华人民共和国法律，因本软件使用产生的任何争议，双方应优先协商解决，协商不成的，提交开发者所在地有管辖权的人民法院诉讼解决。</li>
</ul>

<h3 style="color:#ffb84a;">八、协议更新</h3>
<p>若本声明条款发生变更，开发者将在软件启动时弹窗提示用户重新确认，用户继续使用本软件即视为同意更新后的条款。</p>
</div>
"""

VOICE_CLONE_HTML = """
<div style="color:#dadbdf; font-size:13px; line-height:1.7;">
<h2 style="color:#4a9eff; margin-top:0;">🎙️ 声音克隆功能合规使用声明</h2>
<p>您即将使用的是基于开源项目 <b>GPT-SoVITS</b>（遵循 MIT 协议）的零样本/少样本声音克隆功能，该功能属于《互联网信息服务深度合成管理规定》中定义的深度合成服务，使用前请务必确认您已充分了解相关法律义务。</p>

<h3 style="color:#4aff88;">✅ 合规授权要求</h3>
<p>您上传用于克隆的参考音频必须满足以下任一条件，否则禁止使用本功能：</p>
<ul>
<li>音频为您本人录制的原声；</li>
<li>您已获得声音权属人<b>单独、明确的书面授权</b>（口头约定、社交平台沟通记录等非正式授权不具备法律效力；录音制品的著作权授权不等于声音 AI 化使用的授权）；</li>
<li>音频来自公共领域、CC0 协议或明确允许商用、修改的开源音色库，且您遵守对应的协议要求。</li>
</ul>
<p style="color:#ffb84a; background:#2d3d4d; padding:6px 12px; border-radius:4px;">
⚠️ <b>特别说明：</b>未经监护人书面同意，禁止采集、克隆、使用未成年人的声音；自然人声音受《民法典》第 1023 条保护，参照肖像权相关规定，未经授权克隆他人声音无论是否商用均构成侵权，司法判例中侵权方需承担最高 25 万元赔偿及赔礼道歉的民事责任，情节严重的还将被追究刑事责任。
</p>

<h3 style="color:#ff7a7a;">❌ 禁止使用场景</h3>
<ul>
<li>未经授权克隆他人声音，尤其是公众人物、艺人、配音从业者等具有明确身份识别度的声音；</li>
<li>将克隆声音用于电信诈骗、造谣诽谤、冒充他人身份、伪造证据、虚假宣传等违法用途；</li>
<li>将克隆声音用于未经授权的直播带货、商业广告、有声书、短视频配音等商业场景；</li>
<li>批量生成克隆语音用于垃圾信息传播、恶意营销等扰乱网络秩序的行为；</li>
<li>故意隐匿 AI 合成属性、删除溯源标识，误导公众认为声音为真人原声。</li>
</ul>

<h3 style="color:#4aff88;">📌 内容标识义务</h3>
<p>若您将本功能生成的合成音频公开传播（包括但不限于发布到短视频平台、播客平台、电商平台、社交媒体等），必须按照《人工智能生成合成内容标识办法》要求：</p>
<ul>
<li>在音频开头或结尾添加“本音频由 AI 合成”的语音提示，或在发布页面显著位置标注 AI 生成标识；</li>
<li>保留本软件在音频元数据中自动添加的生成信息，不得恶意删除、篡改。</li>
</ul>

<h3 style="color:#4aff88;">🔒 数据安全说明</h3>
<p>本功能完全在本地运行，您上传的参考音频、生成的模型文件、合成音频均存储在本地设备，本软件不会收集、上传、存储任何音频数据，所有数据处理均在您的设备本地完成。</p>

<h3 style="color:#ffb84a;">⚖️ 责任划分</h3>
<p>您使用本功能即视为承诺所有操作符合法律法规要求，对上传音频的合法性、合成音频的使用场景负全部责任，因声音克隆产生的任何侵权、违法纠纷均由您自行承担，开发者不承担任何直接或间接责任。</p>

<p style="color:#8a9aa8; margin-top:12px;">
💡 首次使用需下载 GPT-SoVITS 引擎（~50MB）和预训练模型（~2GB），请保持网络畅通，下载完成后即可完全离线使用。
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