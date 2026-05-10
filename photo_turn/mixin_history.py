class HistoryMixin:

    def push_history(self, img=None, mask=None):
        if img is None:
            img = self.current_img
        if mask is None:
            mask = self.mask_img

        self.history.append((img.copy(), mask.copy()))
        self.future.clear()

        # 最多保留 15 步（与原版一致）
        if len(self.history) > 15:
            self.history.pop(0)

    def undo(self):
        if len(self.history) > 1:
            current_state = self.history.pop()
            self.future.append(current_state)

            prev_img, prev_mask = self.history[-1]
            self.current_img = prev_img.copy()
            self.mask_img    = prev_mask.copy()
            self.base_img    = self.current_img.copy()
            self.update_canvas(self.current_img)

            # ✅ 替代 lbl_status.config(text=..., foreground=...)
            self.lbl_status.setText("✅ 已撤销")
            self.lbl_status.setStyleSheet("color: #a6e3a1; font-size:13px;")
        else:
            self.lbl_status.setText("⚠️ 已经是最早的状态了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")

    def redo(self):
        if self.future:
            next_state = self.future.pop()
            self.history.append(next_state)

            self.current_img = next_state[0].copy()
            self.mask_img    = next_state[1].copy()
            self.base_img    = self.current_img.copy()
            self.update_canvas(self.current_img)

            # ✅ 替代 lbl_status.config(text=..., foreground=...)
            self.lbl_status.setText("✅ 已重做")
            self.lbl_status.setStyleSheet("color: #a6e3a1; font-size:13px;")
        else:
            self.lbl_status.setText("⚠️ 没有可重做的操作了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")