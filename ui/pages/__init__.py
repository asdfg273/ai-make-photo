# ui/pages/__init__.py
# 页面注册表：新增页面 = 在此追加一行
from ui.pages.txt2img_page import Txt2ImgPage
from ui.pages.img2img_page import Img2ImgPage
from ui.pages.video_page import VideoPage

PAGES = [Txt2ImgPage, Img2ImgPage, VideoPage]
