import io, fitz  # PyMuPDF だけ使う
from PIL import Image
from typing import Dict, List
from .render_img import draw_replace

def pdf_to_images(pdf_bytes: bytes, dpi: int = 180) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pm = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        images.append(img)
    doc.close()
    return images

def process_pdf_raster(pdf_bytes: bytes, hits_by_page: Dict[int, list[Dict]], style: str = "readable") -> bytes:
    pages = pdf_to_images(pdf_bytes)
    out_doc = fitz.open()
    for i, img in enumerate(pages):
        # 置換適用
        for hit in hits_by_page.get(i, []):
            draw_replace(img, hit["bbox"], hit["new"], mode=style)
        # 画像をそのままPDFページへ
        buf = io.BytesIO()
        img.save(buf, format="PNG"); buf.seek(0)
        page = out_doc.new_page(width=img.width, height=img.height)
        page.insert_image(page.rect, stream=buf.getvalue())
    pdf_out = out_doc.tobytes()
    out_doc.close()
    return pdf_out
