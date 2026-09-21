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

        # 同时快照滑块状态，让撤销能还原调色值
        self.history.append((img.copy(), mask.copy(), self._slider_snapshot()))
        self.future.clear()
        if len(self.history) > 1:
            self._dirty = True

        if len(self.history) > 15:
            self.history.pop(0)

    def _hist_status(self, action: str, ok_color: str = "#a6e3a1"):
        depth  = len(self.history) - 1
        redoes = len(self.future)
        self.lbl_status.setText(
            f"✅ {action}（可撤销 {depth} 步 / 可重做 {redoes} 步）")
        self.lbl_status.setStyleSheet(f"color: {ok_color}; font-size:13px;")

    def undo(self):
        if len(self.history) > 1:
            self._filter_anchor = None
            current_state = self.history.pop()
            self.future.append(current_state)

            prev = self.history[-1]
            prev_img, prev_mask = prev[0], prev[1]
            prev_sliders = prev[2] if len(prev) > 2 else None

            self.current_img = prev_img.copy()
            self.mask_img    = prev_mask.copy()
            self.base_img    = self.current_img.copy()
            self._restore_sliders(prev_sliders)
            self.update_canvas(self.current_img)
            self._hist_status("已撤销")
        else:
            self.lbl_status.setText("⚠️ 已经是最早的状态了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")

    def redo(self):
        if self.future:
            self._filter_anchor = None
            next_state = self.future.pop()
            self.history.append(next_state)

            next_img, next_mask = next_state[0], next_state[1]
            next_sliders = next_state[2] if len(next_state) > 2 else None

            self.current_img = next_img.copy()
            self.mask_img    = next_mask.copy()
            self.base_img    = self.current_img.copy()
            self._restore_sliders(next_sliders)
            self.update_canvas(self.current_img)
            self._hist_status("已重做")
        else:
            self.lbl_status.setText("⚠️ 没有可重做的操作了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")