import io, numpy as np
from PIL import Image, ImageOps

def load_image_from_bytes(b: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(b))
    # EXIFの回転を正す
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.array(im)[:, :, ::-1].copy()  # RGB->BGR
    return arr

def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1].copy())  # BGR->RGB
