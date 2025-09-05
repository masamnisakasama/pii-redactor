from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import StreamingResponse
import httpx

router = APIRouter()

def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")

def _passthrough_headers(src: httpx.Response) -> dict:
    allowed = {}
    for k, v in src.headers.items():
        lk = k.lower()
        if lk in {"content-length","transfer-encoding","content-encoding","connection"}:
            continue
        if lk.startswith("x-") or lk in {"content-type","content-disposition"}:
            allowed[k] = v
    allowed.setdefault("Cache-Control", "no-store")
    return allowed

async def _forward_multipart(request: Request, path: str, file: UploadFile):
    base = _base(request)
    url = f"{base}{path}"
    form = await request.form()
    data = {k: v for k, v in form.items() if k != "file"}
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data=data, files=files, timeout=None)
        resp.raise_for_status()
        return resp

@router.post("/preview")
async def preview(request: Request, file: UploadFile):
    resp = await _forward_multipart(request, "/legacy/redact/preview", file)
    return resp.json()

@router.post("/replace")
async def replace(request: Request, file: UploadFile):
    resp = await _forward_multipart(request, "/legacy/redact/replace", file)
    headers = _passthrough_headers(resp)
    return StreamingResponse(resp.aiter_bytes(),
                             media_type=resp.headers.get("content-type","application/octet-stream"),
                             headers=headers)

@router.post("/face_image")
async def face_image(request: Request, file: UploadFile):
    resp = await _forward_multipart(request, "/legacy/redact/face_image", file)
    headers = _passthrough_headers(resp)
    return StreamingResponse(resp.aiter_bytes(),
                             media_type=resp.headers.get("content-type","application/octet-stream"),
                             headers=headers)
