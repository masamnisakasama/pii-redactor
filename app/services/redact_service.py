from typing import Dict, Any
from fastapi import UploadFile
import httpx

# 旧アプリ（/legacyとしてマウント済み）のASGIアプリ本体
from app.legacy_app import app as legacy_app

def _passthrough_headers(src: httpx.Response) -> dict:
    allowed = {}
    for k, v in src.headers.items():
        lk = k.lower()
        if lk in {"content-length","transfer-encoding","content-encoding","connection"}:
            continue
        if lk.startswith("x-") or lk in {"content-type","content-disposition","cache-control"}:
            allowed[k] = v
    allowed.setdefault("Cache-Control", "no-store")
    return allowed

async def _post_multipart(path: str, file: UploadFile, data: Dict[str, Any]) -> httpx.Response:
    # インプロセスASGI呼び出し（HTTPポート不要）
    transport = httpx.ASGITransport(app=legacy_app)
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}
    async with httpx.AsyncClient(transport=transport, base_url="http://legacy.local") as client:
        resp = await client.post(path, data=data, files=files, timeout=None)
        resp.raise_for_status()
        return resp

async def preview(file: UploadFile, **form: Any) -> Dict[str, Any]:
    resp = await _post_multipart("/redact/preview", file, form)
    return resp.json()

async def replace(file: UploadFile, **form: Any) -> httpx.Response:
    resp = await _post_multipart("/redact/replace", file, form)
    return resp

async def face_image(file: UploadFile, **form: Any) -> httpx.Response:
    resp = await _post_multipart("/redact/face_image", file, form)
    return resp

def headers_for(resp: httpx.Response) -> Dict[str, str]:
    return _passthrough_headers(resp)
