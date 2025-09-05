from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import StreamingResponse
from app.services import redact_service as svc

router = APIRouter()

@router.post("/preview")
async def preview(request: Request, file: UploadFile):
    form = await request.form()
    data = {k: v for k, v in form.items() if k != "file"}
    result = await svc.preview(file=file, **data)
    return result

@router.post("/replace")
async def replace(request: Request, file: UploadFile):
    form = await request.form()
    data = {k: v for k, v in form.items() if k != "file"}
    resp = await svc.replace(file=file, **data)
    headers = svc.headers_for(resp)
    return StreamingResponse(resp.aiter_bytes(),
                             media_type=resp.headers.get("content-type","application/octet-stream"),
                             headers=headers)

@router.post("/face_image")
async def face_image(request: Request, file: UploadFile):
    form = await request.form()
    data = {k: v for k, v in form.items() if k != "file"}
    resp = await svc.face_image(file=file, **data)
    headers = svc.headers_for(resp)
    return StreamingResponse(resp.aiter_bytes(),
                             media_type=resp.headers.get("content-type","application/octet-stream"),
                             headers=headers)
