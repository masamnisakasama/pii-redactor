from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings
from app.api import health, security, redact, detect

settings = Settings()
app = FastAPI(title="PII Redactor", version="0.1.0")

# CORS
origins = []
if getattr(settings, "frontend_origins", None):
    origins = [o.strip() for o in settings.frontend_origins.split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
async def _init_security():
    try:
        from app.services.security_service import get_manager
        mgr = get_manager()
        app.state.security_level = mgr.security_level
    except Exception:
        pass

# ルーター登録だけ
app.include_router(health.router)
app.include_router(security.router, prefix="/security")
app.include_router(redact.router,   prefix="/redact")
app.include_router(detect.router,   prefix="/detect")
