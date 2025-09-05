from fastapi import APIRouter, Form, Request
import httpx

router = APIRouter()

@router.get("/status")
async def status(request: Request):
    base = str(request.base_url).rstrip("/")
    # legacy の health を読んで現在レベルを取得
    url = f"{base}/legacy/health"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=5.0)
        r.raise_for_status()
        data = r.json()
    level = data.get("current_security_level") or data.get("current_level") or data.get("level") or "enhanced"
    return {"level": level}

@router.post("/level")
async def change_level(request: Request, level: str = Form(...)):
    base = str(request.base_url).rstrip("/")
    # legacy の /security/level に委譲（ここが真のソースオブトゥルース）
    url = f"{base}/legacy/security/level"
    async with httpx.AsyncClient() as client:
        r = await client.post(url, data={"level": level}, timeout=5.0)
        r.raise_for_status()
        try:
            data = r.json()
            new_level = data.get("level") or data.get("current_security_level") or level
        except Exception:
            new_level = level
    return {"level": new_level}
