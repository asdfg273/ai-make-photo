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

    def _hist_status(self, action: str, ok_color: str = "#a6e3a1"):
        """带栈深度的状态提示"""
        depth  = len(self.history) - 1          # 可撤销步数
        redoes = len(self.future)               # 可重做步数
        self.lbl_status.setText(
            f"✅ {action}（可撤销 {depth} 步 / 可重做 {redoes} 步）")
        self.lbl_status.setStyleSheet(f"color: {ok_color}; font-size:13px;")

    def undo(self):
        if len(self.history) > 1:
            self._filter_anchor = None   # 历史回滚后,滤镜锚点失效
            current_state = self.history.pop()
            self.future.append(current_state)

            prev_img, prev_mask = self.history[-1]
            self.current_img = prev_img.copy()
            self.mask_img    = prev_mask.copy()
            self.base_img    = self.current_img.copy()
            self.update_canvas(self.current_img)
            self._hist_status("已撤销")
        else:
            self.lbl_status.setText("⚠️ 已经是最早的状态了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")

    def redo(self):
        if self.future:
            self._filter_anchor = None   # 历史回滚后,滤镜锚点失效
            next_state = self.future.pop()
            self.history.append(next_state)

            self.current_img = next_state[0].copy()
            self.mask_img    = next_state[1].copy()
            self.base_img    = self.current_img.copy()
            self.update_canvas(self.current_img)
            self._hist_status("已重做")
        else:
            self.lbl_status.setText("⚠️ 没有可重做的操作了")
            self.lbl_status.setStyleSheet("color: #f9e2af; font-size:13px;")
