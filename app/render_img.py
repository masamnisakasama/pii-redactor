from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

# 画像を処理して文字を描写
def draw_replace(img: Image.Image, bbox: Tuple[int,int,int,int], new_text: str,
                 mode: str = "readable", font_path: str | None = None) -> None:
    x1,y1,x2,y2 = bbox; w,h = x2-x1, y2-y1
    d = ImageDraw.Draw(img)
    if mode in ("readable","keep-font"):
        d.rectangle([x1,y1,x2,y2], fill=(255,255,255))
    size = max(10, int(h*0.82))
    try:
        font = ImageFont.truetype(font_path or "DejaVuSans.ttf", size)
    except:
        font = ImageFont.load_default()
    while d.textlength(new_text, font=font) > w and size > 8:
        size -= 1
        try: font = ImageFont.truetype(font_path or "DejaVuSans.ttf", size)
        except: font = ImageFont.load_default()
    ty = y1 + (h - size)//2
    d.text((x1, ty), new_text, fill=(35,35,35), font=font)
