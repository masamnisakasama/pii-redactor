from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Dict, Any

app = FastAPI(title="Nanobanana Stub")

# 最低限のダミー応答。コード側が 200 を受け取れば警告は出ません。
# 形は汎用的にしてあります（足りなければ後で増やせます）。

@app.post("/v1/ocr")
async def ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 簡易OCRライン（1行だけ返す）
    return {"lines": [{"bbox": [0, 0, 100, 20], "text": "stub-ocr", "confidence": 0.99}]}

@app.post("/v1/ner")
async def ner(text: str = Form(None)) -> Dict[str, Any]:
    # エンティティ無し
    return {"entities": []}

@app.post("/v1/face-detect")
async def face_detect(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 顔検出無し
    return {"faces": []}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
