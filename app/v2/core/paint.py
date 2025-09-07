import os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .image_io import pil_from_bgr

_DEF_STYLE = os.getenv("DEFAULT_STYLE", "box")  # box | pixelate | readable

def _clip(img, x,y,w,h):
    H,W = img.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    return x,y,w,h

def draw_box(img_bgr, box):
    x,y,w,h = _clip(img_bgr, *box)
    cv2.rectangle(img_bgr, (x,y), (x+w, y+h), (0,0,0), thickness=-1)
    return img_bgr

def draw_pixelate(img_bgr, box, grid=12):
    x,y,w,h = _clip(img_bgr, *box)
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size==0: return img_bgr
    small = cv2.resize(roi, (max(1,w//grid), max(1,h//grid)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    img_bgr[y:y+h, x:x+w] = pix
    return img_bgr

def draw_readable(img_bgr, box, text="REDACTED"):
    # 背景を白で塗り、その上にテキスト（フォントは環境依存させずデフォルト）
    x,y,w,h = _clip(img_bgr, *box)
    img_bgr[y:y+h, x:x+w] = (255,255,255)
    im = pil_from_bgr(img_bgr)
    dr = ImageDraw.Draw(im)
    # フォント指定（同梱していないのでNone=デフォルト、はみ出し防止で縮小）
    font = ImageFont.load_default()
    tw, th = dr.textbbox((0,0), text, font=font)[2:]
    scale = min(w/max(1,tw), h/max(1,th)) * 0.9
    # load_defaultはサイズ固定なので、代わりに横詰めで折り返さず中央寄せ
    ox = x + (w - tw)//2
    oy = y + (h - th)//2
    dr.text((ox,oy), text, fill=(0,0,0), font=font)
    return np.array(im)[:, :, ::-1].copy()

def apply_masks(img_bgr, boxes, style=None):
    if style is None: style = _DEF_STYLE
    for b in boxes:
        if style == "pixelate": img_bgr = draw_pixelate(img_bgr, b)
        elif style == "readable": img_bgr = draw_readable(img_bgr, b)
        else: img_bgr = draw_box(img_bgr, b)  # default black box
    return img_bgr
