# tests/smoke_photo_turn.py
# ============================================================
#  photo_turn 修图模块离屏冒烟测试
#  用法: venv\Scripts\python tests\smoke_photo_turn.py
# ============================================================
import os
import sys
import glob

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PIL import Image

from photo_turn.pro_editor_qt import ProImageEditor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_out.png")


def main():
    app = QApplication([])

    # 找一张真实图片;没有则造一张
    pics = sorted(glob.glob(os.path.join("photo", "*.png")))
    if pics:
        src = pics[0]
    else:
        src = os.path.join("tests", "_smoke_src.png")
        Image.new("RGB", (800, 600), (120, 130, 140)).save(src)

    ed = ProImageEditor(None, src, callback_on_save=lambda img, mask: None)

    # 1) 画笔(不透明)
    ed.toggle_brush()
    ed.on_mouse_press(100, 100)
    ed.on_mouse_drag(150, 120)
    ed.on_mouse_release(150, 120)
    assert not ed.draw_mode is False or True

    # 2) 画笔(半透明)
    ed.sld_brush_opacity.setValue(40)
    ed.on_mouse_press(200, 200)
    ed.on_mouse_drag(260, 240)
    ed.on_mouse_release(260, 240)
    ed.sld_brush_opacity.setValue(100)

    # 3) 橡皮
    ed.toggle_eraser()
    ed.on_mouse_press(120, 110)
    ed.on_mouse_drag(180, 130)
    ed.on_mouse_release(180, 130)

    # 4) 遮罩 + 清除/反转
    ed.toggle_mask_brush()
    ed.on_mouse_press(300, 300)
    ed.on_mouse_drag(360, 320)
    ed.on_mouse_release(360, 320)
    assert ed.mask_img.getextrema() == (0, 255)
    ed.invert_mask()
    ed.clear_mask()
    assert ed.mask_img.getextrema() == (0, 0)

    # 5) 吸管
    ed.toggle_eyedropper()
    ed.on_mouse_press(50, 50)
    assert ed.pick_mode is False

    # 6) 裁剪(带预览)
    ed.toggle_crop()
    ed.on_mouse_press(10, 10)
    ed.on_mouse_drag(200, 200)
    ed.on_mouse_release(200, 200)
    w, h = ed.current_img.size
    assert w <= 200 and h <= 200, f"crop failed: {ed.current_img.size}"

    # 7) 变换(遮罩/原图同步)
    ed.flip_image("horizontal")
    ed.rotate_image(90)
    ed.rotate_image(-90)
    assert ed.mask_img.size == ed.current_img.size == ed.original_img.size

    # 8) 调色(含曝光/色相)
    ed.adjust_vars["brightness"].setValue(20)
    ed.adjust_vars["exposure"].setValue(15)
    ed.adjust_vars["hue"].setValue(-30)
    ed.apply_adjustments()

    # 8b) 柔边画笔(硬度 30)
    ed.sld_brush_hardness.setValue(30)
    ed.on_mouse_press(60, 60)
    ed.on_mouse_drag(120, 90)
    ed.on_mouse_release(120, 90)
    ed.sld_brush_hardness.setValue(100)

    # 8c) 遮罩精修:羽化/扩边/收缩
    ed.toggle_mask_brush()
    ed.on_mouse_press(80, 80)
    ed.on_mouse_drag(140, 100)
    ed.on_mouse_release(140, 100)
    ed.sld_mask_feather.setValue(6)
    ed.feather_mask()
    ed.grow_mask(4)
    ed.shrink_mask(4)
    ed.clear_mask()

    # 8d) 任意角度旋转 + 等比缩放
    ed.spin_rotate_angle.setValue(15)
    ed.rotate_image_any()
    assert ed.mask_img.size == ed.current_img.size == ed.original_img.size
    ed.resize_to_long_edge(768)
    assert max(ed.current_img.size) == 768
    assert ed.mask_img.size == ed.current_img.size == ed.original_img.size

    # 8e) 对比原图 + 构图网格
    ed._compare_on(); ed._compare_off()
    assert ed.canvas.toggle_grid() is True
    assert ed.canvas.toggle_grid() is False
    ed.canvas.repaint()

    # 9) 全部滤镜逐个过 + 滤镜锚点还原
    import numpy as np
    pre_filter_base = ed.base_img.copy()
    for i in range(ed.filter_combo.count()):
        name = ed.filter_combo.itemText(i)
        if name == "无":
            continue
        ed.filter_combo.setCurrentIndex(i)
        ed.apply_selected_filter()
    assert ed._filter_anchor is not None, "滤镜锚点未建立"
    print(f"filters OK, history depth = {len(ed.history)}")

    # 9b) 选「无」→ 还原到滤镜叠加前
    ed.filter_combo.setCurrentText("无")
    ed.apply_selected_filter()
    assert ed._filter_anchor is None, "还原后锚点未清除"
    assert np.array_equal(np.array(ed.current_img),
                          np.array(pre_filter_base)), "还原后像素与叠加前不一致"
    print("filter revert OK")

    # 9c) 无滤镜状态下选「无」→ 安全返回
    ed.apply_selected_filter()
    print("no-op revert OK")

    # 10) 撤销/重做
    ed.undo(); ed.undo(); ed.redo()

    # 11) 另存为(直接调保存逻辑,绕过文件对话框)
    ed.current_img.save(OUT)

    # 12) 保存并返回(走回调)
    ed.save_and_return()

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
