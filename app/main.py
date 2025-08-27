from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List, Dict
from PIL import Image
import io, httpx, hashlib
import fitz  # PyMuPDF使うのに必要
from .settings import Settings
from . import tf_infer, detectors, alias, render_img, render_pdf
from fastapi.responses import Response
from .face_redactor import redact_faces_image_bytes


app = FastAPI(title="Secure PII Redactor")
settings = Settings()

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

async def _load_bytes(file: UploadFile | None, image_url: Optional[str]) -> bytes:
    if file:
        return await file.read()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(image_url)  # 署名URL　VPCエンドポイント使いたいので
        r.raise_for_status()
        return r.content

def _iter_line_hits(lines, policies: set[str]):
    """行文字列に LINE_PATTERNS を当て、(bbox, kind, orig_text) をyield"""
    for line in lines:
        tokens = line["tokens"]
        pieces, offsets, cur = [], [], 0
        for t in tokens:
            s = t["text"]
            pieces.append(s)
            offsets.append((cur, cur + len(s)))
            cur += len(s) + 1
        line_text = " ".join(pieces)

        for kind, pat in detectors.LINE_PATTERNS:
            for m in pat.finditer(line_text):
                if kind not in policies:
                    continue
                a, b = m.span()
                used = [i for i, (sa, sb) in enumerate(offsets) if not (b <= sa or sb <= a)]
                if not used:
                    continue
                x1 = y1 = 10**9
                x2 = y2 = -10**9
                orig = []
                for i in used:
                    bx1, by1, bx2, by2 = tokens[i]["bbox"]
                    x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)
                    orig.append(tokens[i]["text"])
                yield (x1, y1, x2, y2), kind, " ".join(orig)

@app.post("/redact/preview")
async def preview(
    policy: str = Form("email,name,phone,id,amount,address"),
    consistency_key: str = Form("default"),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    if not file and not image_url:
        return JSONResponse({"items": []})
    blob = await _load_bytes(file, image_url)
    policies = set([p.strip() for p in policy.split(",") if p.strip()])

    items: List[Dict] = []
    is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

    if is_pdf:
        # PDF：PyMuPDFで行単位に変換
        doc = fitz.open(stream=blob, filetype="pdf")
        for page_idx, page in enumerate(doc):
            boxes = tf_infer.pdf_text_boxes(page)
            texts = [b["text"] for b in boxes]
            ner_hits = await tf_infer.ner_classify_texts(texts, settings.ner_endpoint)
            for b in boxes:
                rx = detectors.classify_by_regex(b["text"])
                merged = detectors.merge_with_ner(rx, [h for h in ner_hits if h["text"] in b["text"]])
                for m in merged:
                    if m["type"] not in policies:
                        continue
                    items.append({
                        "page": page_idx, "bbox": b["bbox"], "text": m["text"],
                        "type": m["type"], "confidence": m["conf"], "reason": m["reason"]
                    })
    else:
        # 画像：OCR → 行単位に変換
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        lines = tf_infer.ocr_lines(img)
        for bbox, kind, orig in _iter_line_hits(lines, policies):
            items.append({
                "bbox": bbox, "text": orig, "type": kind,
                "confidence": 0.90, "reason": "line-regex"
            })

    return JSONResponse({"items": items})

@app.post("/redact/replace")
async def replace(
    policy: str = Form("email,name,phone,id,amount,address"),
    style: str = Form("readable"),  # keep-font | readable
    consistency_key: str = Form("default"),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    if not file and not image_url:
        return JSONResponse({"error": "no input"}, status_code=400)

    blob = await _load_bytes(file, image_url)
    policies = set([p.strip() for p in policy.split(",") if p.strip()])
    is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

    if is_pdf:
        # PDF: 画像化 → 行単位 → PDF再構成 
        pages = render_pdf.pdf_to_images(blob)
        hits_by_page: Dict[int, list[Dict]] = {}
        for i, pimg in enumerate(pages):
            lines = tf_infer.ocr_lines(pimg)
            for bbox, kind, orig in _iter_line_hits(lines, policies):
                if kind == "amount":
                    new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
                    new_val = f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
                else:
                    mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
                    new_val = alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)
                hits_by_page.setdefault(i, []).append({"bbox": bbox, "new": new_val})

        pdf_bytes = render_pdf.process_pdf_raster(blob, hits_by_page, style=style)
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")

    # 行単位での認識 → 置換描画
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    lines = tf_infer.ocr_lines(img)
    for bbox, kind, orig in _iter_line_hits(lines, policies):
        if kind == "amount":
            new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
            new_val = f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
        else:
            mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
            new_val = alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)
        render_img.draw_replace(img, bbox, new_val, mode=style, pad=6)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/redact/face_image")
async def redact_face_image(
    file: UploadFile = File(...),
    method: str = Query("pixelate", pattern="^(pixelate|blur|box)$"),
    strength: int = Query(16, ge=1, le=200),
    expand: float = Query(0.12, ge=0.0, le=0.5),
    out_format: str = Query("PNG", pattern="^(PNG|png|JPG|JPEG|jpg|jpeg)$"),
):
    data = await file.read()
    out_bytes = redact_faces_image_bytes(
        data, method=method, strength=strength, expand=expand, out_format=out_format
    )
    mt = "image/jpeg" if out_format.lower() in ("jpg","jpeg") else "image/png"
    return Response(content=out_bytes, media_type=mt)
