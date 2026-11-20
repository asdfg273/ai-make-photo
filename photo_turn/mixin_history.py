from ui.theme import PALETTE


class HistoryMixin:

    def _slider_snapshot(self) -> dict:
        """把 adjust_vars 里所有滑块的当前值打包成字典"""
        return {k: w.value() for k, w in self.adjust_vars.items()}

    def _restore_sliders(self, snapshot: dict | None):
        """还原滑块值 + 同步标签显示"""
        if snapshot is None:
            return
        for k, w in self.adjust_vars.items():
            if k in snapshot:
                w.blockSignals(True)
                w.setValue(snapshot[k])
                w.blockSignals(False)
                # 手动同步标签（命名规则：lbl_{key}）
                lbl = getattr(self, f"lbl_{k}", None)
                if lbl is not None:
                    lbl.setText(str(snapshot[k]))

    def push_history(self, img=None, mask=None):
        if img is None:
            img = self.current_img
        if mask is None:
            mask = self.mask_img

        # 同时快照滑块状态与 original_img：
        # - 滑块：让撤销能还原调色值
        # - original_img：crop/flip/rotate/resize 都会改它，撤销时必须一起还原，
        #   否则它与 current_img/mask_img 尺寸失同步（橡皮 paste / 遮罩叠加会崩）
        self.history.append((
            img.copy(),
            mask.copy(),
            self._slider_snapshot(),
            self.original_img.copy() if getattr(self, "original_img", None) else None,
        ))
        self.future.clear()
        if len(self.history) > 1:
            self._dirty = True

        if len(self.history) > 15:
            self.history.pop(0)

    def _hist_status(self, action: str, ok_color: str | None = None):
        depth = len(self.history) - 1
        redoes = len(self.future)
        self.lbl_status.setText(
            f"{action}（可撤销 {depth} 步，可重做 {redoes} 步）")
        self.lbl_status.setStyleSheet(
            f"color:{ok_color or PALETTE['success']}; font-size:12px;")

    def undo(self):
        if len(self.history) > 1:
            self._filter_anchor = None
            current_state = self.history.pop()
            self.future.append(current_state)

            prev = self.history[-1]
            if prev is None:
                return
            prev_img, prev_mask = prev[0], prev[1]
            prev_sliders = prev[2] if len(prev) > 2 else None
            prev_original = prev[3] if len(prev) > 3 else None

            self.current_img = prev_img.copy()
            self.mask_img    = prev_mask.copy()
            self.base_img    = self.current_img.copy()
            if prev_original is not None:
                self.original_img = prev_original.copy()
            self._restore_sliders(prev_sliders)
            self.update_canvas(self.current_img)
            self._hist_status("已撤销")
        else:
            self.lbl_status.setText("已经是最早的状态")
            self.lbl_status.setStyleSheet(
                f"color:{PALETTE['warning']}; font-size:12px;")

    def redo(self):
        if self.future:
            self._filter_anchor = None
            next_state = self.future.pop()
            self.history.append(next_state)

            next_img, next_mask = next_state[0], next_state[1]
            next_sliders = next_state[2] if len(next_state) > 2 else None
            next_original = next_state[3] if len(next_state) > 3 else None

            self.current_img = next_img.copy()
            self.mask_img    = next_mask.copy()
            self.base_img    = self.current_img.copy()
            if next_original is not None:
                self.original_img = next_original.copy()
            self._restore_sliders(next_sliders)
            self.update_canvas(self.current_img)
            self._hist_status("已重做")
        else:
            self.lbl_status.setText("没有可重做的操作")
            self.lbl_status.setStyleSheet(
                f"color:{PALETTE['warning']}; font-size:12px;")
