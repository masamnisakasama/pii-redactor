# --- 先頭付近の import に追加 ---
import base64, os, httpx
from fastapi import HTTPException

def _get_google_key(x_api_key: str | None) -> str:
    key = x_api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(status_code=401, detail="GOOGLE_API_KEY missing")
    return key

async def _vision_annotate(img_bytes: bytes, features: list[dict], key: str) -> dict:
    url = f"https://vision.googleapis.com/v1/images:annotate?key={key}"
    body = {
        "requests": [{
            "image": {"content": base64.b64encode(img_bytes).decode()},
            "features": features
        }]
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(url, json=body)
        r.raise_for_status()
        return r.json()["responses"][0]

# --- /v1/ocr を置換 ---
@app.post("/v1/ocr")
async def ocr(request: Request, file: UploadFile | None = File(default=None),
              x_api_key: str | None = Header(default=None)):
    img = await _read_image_bytes(request, file)
    if not img:
        return JSONResponse({"error": "no image"}, status_code=422)
    key = _get_google_key(x_api_key)
    resp = await _vision_annotate(img, [{"type": "DOCUMENT_TEXT_DETECTION"}], key)

    fulltext = resp.get("fullTextAnnotation", {}).get("text", "")
    words = []
    pages = resp.get("fullTextAnnotation", {}).get("pages", [])
    for p in pages:
        for b in p.get("blocks", []):
            for para in b.get("paragraphs", []):
                for w in para.get("words", []):
                    text = "".join([s.get("text","") for s in w.get("symbols",[])])
                    verts = w.get("boundingBox", {}).get("vertices", [])
                    bbox = []
                    for v in verts:
                        bbox += [v.get("x",0), v.get("y",0)]
                    if text.strip():
                        words.append({"text": text, "confidence": 0.9, "bbox": bbox})
    # fallback: textAnnotations（wordっぽいもの）
    if not words:
        for a in resp.get("textAnnotations", [])[1:]:
            verts = a.get("boundingPoly", {}).get("vertices", [])
            bbox = []
            for v in verts:
                bbox += [v.get("x",0), v.get("y",0)]
            t = a.get("description","").strip()
            if t:
                words.append({"text": t, "confidence": 0.8, "bbox": bbox})
    return JSONResponse({"words": words, "fulltext": fulltext})

# --- /v1/face-detect を置換 ---
@app.post("/v1/face-detect")
async def face_detect(request: Request, file: UploadFile | None = File(default=None),
                      x_api_key: str | None = Header(default=None)):
    img = await _read_image_bytes(request, file)
    if not img:
        return JSONResponse({"error": "no image"}, status_code=422)
    key = _get_google_key(x_api_key)
    resp = await _vision_annotate(img, [{"type": "FACE_DETECTION"}], key)

    faces = []
    for fa in resp.get("faceAnnotations", []):
        verts = (fa.get("fdBoundingPoly") or fa.get("boundingPoly") or {}).get("vertices", [])
        bbox = []
        for v in verts:
            bbox += [v.get("x",0), v.get("y",0)]
        faces.append({"bbox": bbox, "confidence": fa.get("detectionConfidence", 0.9)})
    return JSONResponse({"faces": faces})
