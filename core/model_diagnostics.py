"""
模型诊断工具 —— 不加载权重，只读元数据和统计信息。
用途：排查出图异常（全黑、色偏、噪点、NaN）的根因。
"""
from __future__ import annotations

import json
import os
import struct
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
#  数据结构
# ============================================================
@dataclass
class TensorStat:
    name: str
    shape: list
    dtype: str


@dataclass
class DiagReport:
    """一份模型的完整诊断报告"""
    file_name: str
    file_size_gb: float
    # 架构
    arch: str = "unknown"
    arch_reason: str = ""
    # 层统计
    total_tensors: int = 0
    total_params: int = 0
    # v-prediction
    vpred_detected: bool = False
    vpred_evidence: list = field(default_factory=list)
    # VAE
    has_vae: bool = False
    vae_tensor_count: int = 0
    vae_dtype: str = ""
    # 异常
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    # 兼容性建议
    suggestions: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"📋 模型诊断报告: {self.file_name}",
            "=" * 60,
            f"  文件大小:    {self.file_size_gb:.2f} GB",
            f"  架构:        {self.arch} ({self.arch_reason})",
            f"  张量总数:    {self.total_tensors}",
            f"  参数总量:    {self.total_params:,}",
            "",
            f"  V-Prediction: {'✅ 是' if self.vpred_detected else '❌ 否'}",
        ]
        for ev in self.vpred_evidence:
            lines.append(f"    → {ev}")
        lines += [
            "",
            f"  VAE 内嵌:    {'✅ 有' if self.has_vae else '❌ 无'}",
            f"  VAE 张量数:  {self.vae_tensor_count}",
            f"  VAE 精度:    {self.vae_dtype or 'N/A'}",
        ]
        if self.warnings:
            lines += ["", "  ⚠️ 警告:"]
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.errors:
            lines += ["", "  ❌ 错误:"]
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.suggestions:
            lines += ["", "  💡 建议:"]
            for s in self.suggestions:
                lines.append(f"    - {s}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
#  元数据读取（不加载权重）
# ============================================================
def _read_safetensors_metadata(path: str) -> tuple[dict, dict]:
    """
    读取 safetensors 文件的 header。
    返回 (tensor_info_dict, global_metadata_dict)。
    tensor_info_dict: { "layer.name": {"dtype": "F16", "shape": [x,y], "data_offsets": [a,b]} }
    global_metadata_dict: { "__metadata__": {...} } 里的内容
    """
    with open(path, "rb") as f:
        # safetensors 格式：前 8 字节是 header 长度（uint64 LE）
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > 100_000_000:  # >100MB header 不正常
            raise ValueError(f"Header 异常大: {header_len} bytes")
        header_bytes = f.read(header_len)

    header = json.loads(header_bytes)
    metadata = header.pop("__metadata__", {})
    return header, metadata


def _dtype_to_bytes(dtype_str: str) -> int:
    """safetensors dtype 字符串 → 每元素字节数"""
    mapping = {
        "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "BOOL": 1,
    }
    return mapping.get(dtype_str.upper(), 2)


def _count_params(shape: list) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


# ============================================================
#  架构检测（纯 key 分析）
# ============================================================
_ARCH_SIGNATURES = {
    "sdxl": [
        "conditioner.embedders.1.model.transformer.",     # OpenCLIP-G
        "input_blocks.7.1.transformer_blocks.4.",          # UNet 深层
    ],
    "sd15": [
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.",
    ],
    "sd3": [
        "model.diffusion_model.joint_blocks.",
    ],
    "flux": [
        "model.diffusion_model.double_blocks.",
    ],
}


def _detect_arch(keys: list[str]) -> tuple[str, str]:
    key_set = set(k.split(".data_offsets")[0] for k in keys)
    # 取前缀集合用于模糊匹配
    for arch_id, sigs in _ARCH_SIGNATURES.items():
        matched = []
        for sig in sigs:
            if any(k.startswith(sig) or sig in k for k in key_set):
                matched.append(sig)
        if len(matched) == len(sigs):
            return arch_id, f"匹配签名: {matched[0][:40]}..."

    # 回退：数 key 数量
    n = len(key_set)
    if n > 2000:
        return "sdxl", f"key 数量={n}，疑似 SDXL"
    elif n > 500:
        return "sd15", f"key 数量={n}，疑似 SD1.5"
    return "unknown", f"key 数量={n}，无法判断"


# ============================================================
#  V-Prediction 检测（多重证据）
# ============================================================
def _detect_vpred(path: str, file_name: str,
                  metadata: dict, keys: list[str]) -> tuple[bool, list[str]]:
    """
    多重判据检测 v-prediction：
    1. safetensors 全局元数据
    2. 文件名关键词
    3. scheduler config（如果同目录有的话）
    返回 (is_vpred, evidence_list)
    """
    evidence = []

    # —— 判据 1：元数据字段 ——
    meta_str = json.dumps(metadata).lower()
    if "v_prediction" in meta_str or "v-prediction" in meta_str:
        evidence.append("元数据包含 'v_prediction'")
    if "vprediction" in meta_str:
        evidence.append("元数据包含 'vprediction'")

    # 部分模型在 metadata 里写 prediction_type
    pred_type = metadata.get("prediction_type", "").lower()
    if pred_type == "v_prediction":
        evidence.append(f"metadata.prediction_type = '{pred_type}'")

    # modelspec 格式
    modelspec = metadata.get("modelspec.prediction_type", "").lower()
    if "v" in modelspec:
        evidence.append(f"modelspec.prediction_type = '{modelspec}'")

    # —— 判据 2：文件名（不区分大小写） ——
    fn_lower = file_name.lower()
    vpred_keywords = ["vpred", "v_pred", "v-pred", "vprediction"]
    for kw in vpred_keywords:
        if kw in fn_lower:
            evidence.append(f"文件名包含 '{kw}'")
            break

    # —— 判据 3：同目录 scheduler config ——
    parent = os.path.dirname(path)
    sched_path = os.path.join(parent, "scheduler", "scheduler_config.json")
    if os.path.exists(sched_path):
        try:
            with open(sched_path) as f:
                sc = json.load(f)
            if sc.get("prediction_type") == "v_prediction":
                evidence.append("scheduler_config.json: v_prediction")
        except Exception:
            pass

    is_vpred = len(evidence) > 0
    return is_vpred, evidence


# ============================================================
#  VAE 分析
# ============================================================
def _analyze_vae(tensors: dict) -> tuple[bool, int, str, list[str]]:
    """分析 VAE 层，返回 (has_vae, count, dtype, warnings)"""
    vae_keys = [k for k in tensors if k.startswith(("vae.", "first_stage_model."))]
    warnings = []

    if not vae_keys:
        return False, 0, "", ["模型不包含 VAE 权重，将使用外部 VAE"]

    # 统计精度
    dtypes = set()
    for k in vae_keys:
        info = tensors[k]
        dtypes.add(info.get("dtype", "unknown"))

    dtype_str = "/".join(sorted(dtypes))

    if "F32" in dtypes and len(dtypes) > 1:
        warnings.append(f"VAE 混合精度: {dtype_str}，可能导致色偏")
    if "F32" not in dtypes and "BF16" not in dtypes and "F16" in dtypes:
        # 纯 F16 VAE 在 SDXL 上容易出 NaN
        warnings.append("VAE 为纯 FP16，SDXL 上建议用 FP32 VAE 或开启 fp32_vae 补丁")

    return True, len(vae_keys), dtype_str, warnings


# ============================================================
#  权重健康检查（抽样，不全量加载）
# ============================================================
def _check_tensor_health(path: str, tensors: dict,
                         sample_count: int = 20) -> list[str]:
    """
    抽样读取部分张量的实际数值，检查 NaN / Inf / 全零。
    只读 sample_count 个最大的张量（UNet 核心层）。
    """
    warnings = []

    try:
        from safetensors import safe_open
    except ImportError:
        warnings.append("safetensors 库不支持 safe_open，跳过健康检查")
        return warnings

    # 按参数量排序，取最大的几个
    sorted_keys = sorted(
        tensors.keys(),
        key=lambda k: _count_params(tensors[k].get("shape", [0])),
        reverse=True,
    )[:sample_count]

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted_keys:
                try:
                    tensor = f.get_tensor(key)
                    if tensor.isnan().any():
                        warnings.append(f"NaN 检出: {key} (shape={list(tensor.shape)})")
                    if tensor.isinf().any():
                        warnings.append(f"Inf 检出: {key} (shape={list(tensor.shape)})")
                    if tensor.abs().max().item() == 0:
                        warnings.append(f"全零张量: {key} (shape={list(tensor.shape)})")
                    # 异常大的值（可能是权重爆炸）
                    max_val = tensor.abs().max().item()
                    if max_val > 1000:
                        warnings.append(
                            f"异常大值: {key} max={max_val:.1f} "
                            f"(shape={list(tensor.shape)})"
                        )
                except Exception as e:
                    warnings.append(f"读取失败: {key} → {e}")
    except Exception as e:
        warnings.append(f"safe_open 失败: {e}")

    return warnings


# ============================================================
#  主入口
# ============================================================
def diagnose_model(model_path: str,
                   deep_scan: bool = True) -> DiagReport:
    """
    对单个 safetensors 文件做完整诊断。
    deep_scan=True 时会实际读取部分权重检查 NaN/Inf（耗时 5-10 秒）。
    """
    path = Path(model_path)
    report = DiagReport(
        file_name=path.name,
        file_size_gb=path.stat().st_size / (1024 ** 3),
    )

    # 1. 读 header
    try:
        tensors, metadata = _read_safetensors_metadata(str(path))
    except Exception as e:
        report.errors.append(f"无法读取 safetensors header: {e}")
        report.suggestions.append("文件可能损坏，建议重新下载")
        return report

    # 2. 基础统计
    report.total_tensors = len(tensors)
    total_params = 0
    for k, v in tensors.items():
        total_params += _count_params(v.get("shape", [0]))
    report.total_params = total_params

    # 3. 架构检测
    keys = list(tensors.keys())
    report.arch, report.arch_reason = _detect_arch(keys)

    # 4. V-Prediction 检测
    report.vpred_detected, report.vpred_evidence = _detect_vpred(
        str(path), path.name, metadata, keys
    )

    # 5. VAE 分析
    has_vae, vae_count, vae_dtype, vae_warns = _analyze_vae(tensors)
    report.has_vae = has_vae
    report.vae_tensor_count = vae_count
    report.vae_dtype = vae_dtype
    report.warnings.extend(vae_warns)

    # 6. 健康检查
    if deep_scan:
        health_warns = _check_tensor_health(str(path), tensors)
        report.warnings.extend(health_warns)

    # 7. 生成建议
    _generate_suggestions(report)

    return report


def _generate_suggestions(report: DiagReport):
    """根据诊断结果生成操作建议"""

    # v-pred 模型必须配置 scheduler
    if report.vpred_detected:
        report.suggestions.append(
            "此模型使用 V-Prediction，必须设置 "
            "scheduler.prediction_type='v_prediction' + "
            "rescale_betas_zero_snr=True，否则出图全黑或噪点"
        )

    # 无 VAE
    if not report.has_vae:
        if report.arch == "sdxl":
            report.suggestions.append(
                "模型不含 VAE，将使用 SDXL 默认 VAE。"
                "如果色彩异常，尝试指定 madebyollin/sdxl-vae-fp16-fix"
            )

    # FP16 VAE on SDXL
    if report.arch == "sdxl" and "F16" in report.vae_dtype:
        report.suggestions.append(
            "SDXL + FP16 VAE 容易出现 NaN 黑图，"
            "建议开启 force_fp32_vae 或使用 sdxl-vae-fp16-fix"
        )

    # NaN 检出
    nan_count = sum(1 for w in report.warnings if "NaN" in w)
    if nan_count > 0:
        report.suggestions.append(
            f"检出 {nan_count} 个含 NaN 的张量，模型可能已损坏，建议重新下载"
        )

    # 参数量异常
    if report.arch == "sdxl" and report.total_params < 2_000_000_000:
        report.suggestions.append(
            f"SDXL 通常有 ~2.6B 参数，当前仅 {report.total_params:,}，"
            "可能是裁剪版或不完整模型"
        )


# ============================================================
#  运行时诊断（已加载管线）—— UI 弹窗共用入口
# ============================================================
def runtime_report_lines(mgr) -> list:
    """从 ModelManager 当前加载的管线收集诊断行（不读文件，纯运行时状态）。"""
    pipe = mgr.txt2img_pipe
    vae   = getattr(pipe, 'vae', None)
    sched = getattr(pipe, 'scheduler', None)
    unet  = getattr(pipe, 'unet', None)

    lines = []
    lines.append(f"模型名称:     {getattr(mgr, 'current_model_name', '未知')}")
    lines.append(f"架构 arch_id:  {getattr(mgr, 'current_arch_id', '未知')}")

    arch_profile = getattr(mgr, 'arch_profile', None)
    if arch_profile is not None:
        lines.append(f"架构显示名:   {getattr(arch_profile, 'display_name', '—')}")
        caps = getattr(arch_profile, 'caps', None)
        if caps:
            lines.append(f"heavy_vram:   {getattr(caps, 'heavy_vram', '—')}")
            lines.append(f"needs_fp32_vae:{getattr(caps, 'needs_fp32_vae', '—')}")

    lines.append("")
    lines.append(f"V-Prediction: {'✅ 是' if getattr(mgr, 'is_vpred', False) else '❌ 否'}")

    if sched:
        pred_type = getattr(sched.config, 'prediction_type', '未知')
        zero_snr  = getattr(sched.config, 'rescale_betas_zero_snr', False)
        lines.append(f"Scheduler:    {type(sched).__name__}")
        lines.append(f"  prediction_type:         {pred_type}")
        lines.append(f"  rescale_betas_zero_snr:  {zero_snr}")

    lines.append("")
    if unet:
        lines.append(f"UNet dtype:   {getattr(unet, 'dtype', '未知')}")
        lines.append(f"UNet device:  {getattr(unet, 'device', '未知')}")

    if vae:
        fp32_patch = getattr(vae, '_fp32_patched', False)
        lines.append(f"VAE dtype:    {getattr(vae, 'dtype', '未知')}")
        lines.append(f"VAE tiling:   {getattr(vae, 'use_tiling', '未知')}  "
                     f"slicing: {getattr(vae, 'use_slicing', '未知')}")
        lines.append(f"VAE fp32补丁: {'✅' if fp32_patch else '❌'}")

    lines.append("")
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            resvd = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(f"显存占用:     {alloc:.2f} GB allocated / {resvd:.2f} GB reserved")
            lines.append(f"显存总量:     {total:.2f} GB")
    except Exception:
        pass
    return lines


def show_runtime_diagnostics(parent, mgr) -> bool:
    """弹出运行时诊断对话框。无可诊断对象时提示并返回 False。"""
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextEdit,
                                 QDialogButtonBox, QMessageBox)
    from PyQt6.QtGui import QFont

    if mgr is None or getattr(mgr, 'txt2img_pipe', None) is None:
        QMessageBox.information(parent, "模型诊断", "当前没有已加载的模型。")
        return False

    dlg = QDialog(parent)
    dlg.setWindowTitle(f"模型诊断 — {getattr(mgr, 'current_model_name', '')}")
    dlg.resize(480, 360)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    txt = QTextEdit()
    txt.setReadOnly(True)
    txt.setFont(QFont("Consolas, monospace", 10))
    txt.setStyleSheet(
        "background:#0d0f12; color:#dadbdf; border:1px solid #212327; border-radius:6px;"
    )
    txt.setPlainText("\n".join(runtime_report_lines(mgr)))
    layout.addWidget(txt)

    btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    btns.rejected.connect(dlg.reject)
    btns.accepted.connect(dlg.accept)
    layout.addWidget(btns)

    dlg.show()          # 非模态，方便对照主窗口
    parent._model_diag_dlg = dlg   # 防 GC
    return True


# ============================================================
#  CLI 入口（可直接 python -m core.model_diagnostics path/to/model）
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m core.model_diagnostics <model.safetensors>")
        sys.exit(1)

    target = sys.argv[1]
    print(f"正在诊断 {target} ...")
    r = diagnose_model(target, deep_scan="--no-deep" not in sys.argv)
    print(r.summary())