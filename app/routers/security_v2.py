# app/routers/security_v2.py
from importlib import import_module
from fastapi import APIRouter, Request, Form, Body
from fastapi.responses import JSONResponse

try:
    from app.security_manager import SecurityToggleManager, SecurityLevel
except Exception:
    SecurityToggleManager = None
    SecurityLevel = None

router = APIRouter()

# 既存 main のグローバルにも対応（壊さないためのfallback）
def _get_security_manager(request: Request):
    sm = getattr(request.app.state, "security", None)
    if sm:
        return sm
    try:
        m = import_module("app.main")
        # よくある変数名の総当たり
        for name in ("security_manager", "security_toggle_manager", "manager", "toggle_manager"):
            obj = getattr(m, name, None)
            if SecurityToggleManager and isinstance(obj, SecurityToggleManager):
                return obj
        # 型で探索
        if SecurityToggleManager:
            for name in dir(m):
                obj = getattr(m, name)
                if isinstance(obj, SecurityToggleManager):
                    return obj
    except Exception:
        pass
    return None

@router.get("/status")
async def get_security_status_v2(request: Request):
    sm = _get_security_manager(request)
    if not sm:
        return JSONResponse({"error": "Security manager not initialized"}, status_code=500)
    return JSONResponse(sm.get_security_info())

@router.post("/level")
async def set_security_level_v2(request: Request):
    """
    Content-Type を見て level を取り出す超堅牢版:
      - JSON:   {"level":"enhanced"}   (Content-Type: application/json)
      - FORM:   level=enhanced         (multipart/form-data or application/x-www-form-urlencoded)
      - ついでに ?level=enhanced も拾う（最後の手段）
    """
    sm = _get_security_manager(request)
    if not sm:
        return JSONResponse({"error": "Security manager not initialized"}, status_code=500)

    level = None
    ct = (request.headers.get("content-type") or "").lower()

    try:
        if "application/json" in ct:
            payload = await request.json()
            if isinstance(payload, dict):
                level = payload.get("level")
    except Exception:
        pass

    if not level and ("multipart/form-data" in ct or "application/x-www-form-urlencoded" in ct):
        try:
            form = await request.form()
            level = form.get("level")
        except Exception:
            pass

    # 最後の保険: クエリ文字列 ?level=...
    if not level:
        level = request.query_params.get("level")

    if not level:
        return JSONResponse({"detail": "level is required"}, status_code=422)

    # SecurityLevel が使える環境なら Enum 経由で厳密化
    if SecurityLevel:
        try:
            new_level = SecurityLevel(level)
        except ValueError:
            available = [lvl.value for lvl in sm.get_available_levels()]
            return JSONResponse({"detail": f"Invalid level. Available: {available}"}, status_code=400)
        ok = sm.set_security_level(new_level)
    else:
        # 念のためのフォールバック（Enumが無い異常環境向け）
        ok = sm.set_security_level(level) if hasattr(sm, "set_security_level") else False

    if not ok:
        return JSONResponse({"detail": "Failed to set security level"}, status_code=400)

    return JSONResponse({
        "message": f"Security level set to {level}",
        "security_info": sm.get_security_info()
    })