from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.settings import Settings
from app.legacy_app import app as legacy_app  # 旧900行版を /legacy に載せる

settings = Settings()
app = FastAPI(title="PII Redactor", version="0.1.0")

# CORS
origins = []
if getattr(settings, "frontend_origins", None):
    origins = [o.strip() for o in settings.frontend_origins.split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware,
        allow_origins=origins, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"])

# 旧アプリを /legacy にマウント
app.mount("/legacy", legacy_app)

# 親起動時にレガシー側の startup も起動
@app.on_event("startup")
async def _startup():
    try:
        await legacy_app.router.startup()
    except Exception:
        pass

@app.on_event("shutdown")
async def _shutdown():
    try:
        await legacy_app.router.shutdown()
    except Exception:
        pass

# ---- ここで新ルーターを明示的に登録（tryで握りつぶさない）----
from app.api import health, security
app.include_router(health.router)                          # /health, /capabilities
app.include_router(security.router, prefix="/security")    # /security/*
