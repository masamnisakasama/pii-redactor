# app/face_redactor.py
# API使うならクラウドで、各々がダウンロードして使えるようにするなら事実上APIは使えない　
# 反省を生かしコストと処理能力を見積もって決定するつもり
import io
from typing import List, Tuple, Literal
import numpy as np
import cv2
from PIL import Image, ImageOps

Method = Literal["pixelate", "blur", "box"]

def _detect_faces_haar(bgr: np.ndarray,
                       scale_factor: float = 1.2,
                       min_neighbors: int = 5) -> List[Tuple[int,int,int,int]]:
    """Haarで顔矩形 (x,y,w,h) を返す（CPUで軽量）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cc.empty():
        raise RuntimeError("Haar分類器が読み込めませんでした")
    faces = cc.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def _pixelate_roi(img: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 12):
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    img[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, k: int = 51):
    k = int(k) | 1
    img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (k, k), 0)

def _box_roi(img: np.ndarray, x: int, y: int, w: int, h: int):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

def redact_faces_image_bytes(file_bytes: bytes,
                             method: Method = "pixelate",
                             strength: int = 16,
                             expand: float = 0.12,
                             out_format: str = "PNG", 
                             jpeg_quality: int = 90   
                             ) -> bytes:
    """
    画像バイトを受け取り、顔をモザイク/ぼかし/黒塗りしてPNGバイトを返す。
    - method: "pixelate" | "blur" | "box"
    - expand: 顔矩形を上下左右に拡張（見切れ防止のため）
    """
    # EXIFの回転を正しく適用
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    faces = _detect_faces_haar(bgr)
    if not faces:
        # 顔が無い場合もPNGで返す
        out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bio = io.BytesIO(); out.save(bio, format="PNG"); return bio.getvalue()

    for (x, y, w, h) in faces:
        padw, padh = int(w * expand), int(h * expand)
        x0 = max(0, x - padw)
        y0 = max(0, y - padh)
        w0 = min(W - x0, w + 2 * padw)
        h0 = min(H - y0, h + 2 * padh)

        if method == "pixelate":
            _pixelate_roi(bgr, x0, y0, w0, h0, blocks=max(6, int(strength)))
        elif method == "blur":
            _blur_roi(bgr, x0, y0, w0, h0, k=max(7, int(strength)))
        else:
            _box_roi(bgr, x0, y0, w0, h0)
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    print(f"[face_redactor] faces={len(faces)} method={method}")
    if out_format.upper() in ("JPEG", "JPG"):
        out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        out.save(bio, format="PNG")
        return bio.getvalue()
