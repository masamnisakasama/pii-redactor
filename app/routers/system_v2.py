# app/routers/system_v2.py
import asyncio
from importlib import import_module
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

try:
    from app.security_manager import SecurityToggleManager, SecurityLevel
except Exception:
    SecurityToggleManager = None
    SecurityLevel = None

try:
    from app.settings import Settings
except Exception:
    Settings = None  # なくても動くように

router = APIRouter()

def _get_security_manager(request: Request):
    # 1) 通常ルート: app.state.security
    sm = getattr(request.app.state, "security", None)
    if sm:
        return sm
    # 2) 既存 main のグローバルを探す
    try:
        m = import_module("app.main")
        # よくある名前で試す
        for name in ("security_manager", "security_toggle_manager", "manager", "toggle_manager"):
            obj = getattr(m, name, None)
            if SecurityToggleManager and isinstance(obj, SecurityToggleManager):
                return obj
        # 型で総当たり
        if SecurityToggleManager:
            for name in dir(m):
                obj = getattr(m, name)
                if isinstance(obj, SecurityToggleManager):
                    return obj
    except Exception:
        pass
    return None

def _get_settings(request: Request):
    s = getattr(request.app.state, "settings", None)
    if s:
        return s
    # 既存 main に settings があれば使う
    try:
        m = import_module("app.main")
        s = getattr(m, "settings", None)
        if Settings and isinstance(s, Settings):
            return s
    except Exception:
        pass
    # 最後の手段: 新規作成（型がなければ最低限のダミー）
    try:
        return Settings() if Settings else type("S", (), {
            "allowed_extensions_list": ["pdf","png","jpg","jpeg"],
            "max_file_size_mb": 20
        })()
    except Exception:
        return type("S", (), {
            "allowed_extensions_list": ["pdf","png","jpg","jpeg"],
            "max_file_size_mb": 20
        })()

def _desc(level_str: str) -> str:
    return {
        "maximum":  "最高セキュリティ：完全オフライン処理（OpenCV + Tesseract）",
        "high":     "高セキュリティ：オンプレミスAI + 限定的外部API",
        "standard": "標準セキュリティ：バランス型（一部API使用）",
        "enhanced": "AI機能優先：高精度処理（Nanobanan API等使用）",
    }.get(level_str, "Unknown security level")

@router.get("/health")
async def health_v2(request: Request):
    sm = _get_security_manager(request)
    if sm is None:
        return JSONResponse({"status": "error", "message": "Security manager not initialized"})
    level = getattr(sm, "current_level", None)
    level_val = getattr(level, "value", str(level))
    return JSONResponse({
        "status": "healthy",
        "security_manager": "initialized",
        "current_security_level": level_val,
        "available_processors": len(getattr(sm, "processors", {})),
        "timestamp": int(asyncio.get_event_loop().time()),
        "v2": True
    })

@router.get("/capabilities")
async def capabilities_v2(request: Request):
    sm = _get_security_manager(request)
    settings = _get_settings(request)
    if sm is None or settings is None:
        return JSONResponse({"error": "not initialized"}, status_code=500)

    levels = {}
    # SecurityLevel Enum があるならそれを利用、なければ maximum/enhanced だけでも返す
    if SecurityLevel:
        try:
            avail = set(getattr(sm, "get_available_levels", lambda: [])())
        except Exception:
            avail = set()
        for lvl in SecurityLevel:
            levels[lvl.value] = {
                "name": lvl.value.title(),
                "description": _desc(lvl.value),
                "available": (lvl in avail) if avail else True  # 情報なければTrue寄せ
            }
    else:
        for lvl in ("maximum", "enhanced"):
            levels[lvl] = {
                "name": lvl.title(),
                "description": _desc(lvl),
                "available": True
            }

    current = getattr(sm, "current_level", None)
    current_val = getattr(current, "value", str(current))

    return JSONResponse({
        "security_levels": levels,
        "current_level": current_val,
        "supported_formats": getattr(settings, "allowed_extensions_list", ["pdf","png","jpg","jpeg"]),
        "max_file_size_mb": getattr(settings, "max_file_size_mb", 20),
        "features": {
            "ocr": True, "ner": True, "face_detection": True,
            "pdf_processing": True, "batch_processing": False
        },
        "v2": True
    })
