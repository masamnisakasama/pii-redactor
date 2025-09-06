# app/factory.py
# 主にルータ
# Settings → CORS → SecurityToggleManager → ルータ登録の順を崩さない　ステップバイステップでリファクタリング

from fastapi import FastAPI

def create_app() -> FastAPI:
    from .main import app as legacy_app
    # ここで“追加”だけする（既存はそのまま）
    try:
        from .routers.system_v2 import router as system_v2_router
        legacy_app.include_router(system_v2_router, prefix="/_v2", tags=["SystemV2"])
    except Exception as e:
        # 失敗しても既存は壊さない
        import logging
        logging.getLogger("app.factory").warning("system_v2 attach skipped: %s", e)
    
    try:
        from .routers.security_v2 import router as security_v2_router
        legacy_app.include_router(security_v2_router, prefix="/_v2/security", tags=["SecurityV2"])
    except Exception as e:
        import logging
        logging.getLogger("app.factory").warning("security_v2 attach skipped: %s", e)

    try:
        from .routers.detect_v2 import router as detect_v2_router
        legacy_app.include_router(detect_v2_router, prefix="/_v2/detect", tags=["DetectV2"])
    except Exception as e:
        import logging
        logging.getLogger("app.factory").warning("detect_v2 attach skipped: %s", e)
    return legacy_app
