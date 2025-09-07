import os, io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from ..core.image_io import load_image_from_bytes, pil_from_bgr
from ..core.redact import redact_image_bgr

app = FastAPI(title="pii-redactor v2 (minimum)")

@app.get("/v2/health")
def health():
    return {"status":"ok", "mode":"offline-min", "style_default": os.getenv("DEFAULT_STYLE","box")}

@app.post("/v2/detect")
async def detect(file: UploadFile = File(...), policy: str = Form("email,phone,id,face")):
    b = await file.read()
    img = load_image_from_bytes(b)
    # redact() の箱出しだけ使う
    _, boxes = redact_image_bgr(img, policy, style="box")
    return {"count": len(boxes), "boxes": boxes}

@app.post("/v2/redact")
async def redact(
    file: UploadFile = File(...),
    policy: str = Form("email,phone,id,face"),
    style: str = Form("box")  # box|pixelate|readable
):
    b = await file.read()
    img = load_image_from_bytes(b)
    out, boxes = redact_image_bgr(img, policy, style=style)
    im = pil_from_bgr(out)
    bio = io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    headers={"X-PII-Boxes": str(len(boxes))}
    return StreamingResponse(bio, media_type="image/png", headers=headers)
