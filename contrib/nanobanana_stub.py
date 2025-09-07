from fastapi import FastAPI, Request, UploadFile, File

app = FastAPI(title="Nanobanana Stub", version="0.2")

@app.get("/health")
def health():
    return {"status": "ok", "stub": True}

# 共通: どんな形でも画像を受け取って読み捨てる
async def _drain_any(request: Request, **files):
    # files には file/image/content など UploadFile or None が入る
    for f in files.values():
        if isinstance(f, UploadFile) and f is not None:
            await f.read()
            return True
    # マルチパートで来ないケースは生ボディを読む
    try:
        await request.body()
        return True
    except Exception:
        return False

@app.post("/v1/ocr")
async def ocr(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    content: UploadFile | None = File(None),
):
    await _drain_any(request, file=file, image=image, content=content)
    return {
        "words": [{"text": "stub", "confidence": 0.9, "bbox": [0,0,10,0,10,10,0,10]}],
        "fulltext": "stub-ok"
    }

@app.post("/v1/ner")
async def ner(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    content: UploadFile | None = File(None),
):
    await _drain_any(request, file=file, image=image, content=content)
    return {
        "entities": [],
        "fulltext": "stub-ok"
    }

@app.post("/v1/face-detect")
async def face_detect(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    content: UploadFile | None = File(None),
):
    await _drain_any(request, file=file, image=image, content=content)
    return {
        "faces": [{"bbox": [10,10,110,10,110,110,10,110], "confidence": 0.95}]
    }
