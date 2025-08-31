# app/simple_main.py
# main.pyの前に簡易版でcurl通せるか十分に確認　確認したのちmain.pyで主機能け検査

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from PIL import Image
import io
import logging
import re

# settings.pyのみをインポート
try:
    from .settings import Settings
    settings = Settings()
except ImportError:
    # 相対インポートが失敗した場合
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from app.settings import Settings
    settings = Settings()

# ロギング設定　シンプルな機能のみ実装
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Redactor - Simple Test Version",
    description="起動テスト用の簡略版",
    version="1.0.0-test"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin.split(',') if settings.cors_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return JSONResponse({
        "message": "PII Redactor API is running!",
        "version": "1.0.0-test",
        "status": "healthy",
        "endpoints": [
            "/health",
            "/settings", 
            "/test/upload",
            "/test/dependencies",
            "/test/basic_ocr",
            "/test/face_detection",
            "/test/pii_detection"
        ]
    })

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return JSONResponse({
        "status": "healthy",
        "settings": {
            "max_file_size_mb": settings.max_file_size_mb,
            "max_image_size": settings.max_image_size,
            "allowed_extensions": settings.allowed_extensions_list,
            "default_security_level": settings.default_security_level
        }
    })

@app.get("/settings")
async def get_settings():
    """現在の設定を取得"""
    return JSONResponse({
        "cors_origin": settings.cors_origin,
        "max_file_size_mb": settings.max_file_size_mb,
        "max_image_size": settings.max_image_size,
        "pdf_dpi": settings.pdf_dpi,
        "allowed_file_extensions": settings.allowed_file_extensions,
        "default_security_level": settings.default_security_level,
        "ocr_confidence_threshold": settings.ocr_confidence_threshold,
        "ner_confidence_threshold": settings.ner_confidence_threshold,
        "is_nanobanan_configured": settings.is_nanobanan_configured,
        "nanobanan_endpoint": settings.nanobanan_endpoint if settings.nanobanan_api_key else "not_configured"
    })

def safe_get_version(module_name: str, module_obj) -> str:
    """安全にモジュールのバージョンを取得"""
    try:
        # hasattr
        if hasattr(module_obj, '__version__'):
            return module_obj.__version__
        elif hasattr(module_obj, 'version'):
            return str(module_obj.version)
        elif hasattr(module_obj, 'VERSION'):
            return str(module_obj.VERSION)
        else:
            return "installed (version unknown)"
    except Exception:
        return "installed (version error)"

@app.get("/test/dependencies")
async def test_dependencies():
    """依存関係チェック（修正版）"""
    deps = {}
    
    try:
        import fastapi
        deps["fastapi"] = safe_get_version("fastapi", fastapi)
    except ImportError:
        deps["fastapi"] = "not_installed"
    
    try:
        import uvicorn
        deps["uvicorn"] = safe_get_version("uvicorn", uvicorn)
    except ImportError:
        deps["uvicorn"] = "not_installed"
    
    try:
        from PIL import Image
        import PIL
        deps["pillow"] = safe_get_version("PIL", PIL)
    except ImportError:
        deps["pillow"] = "not_installed"
    
    # OCR依存関係
    try:
        import pytesseract
        deps["pytesseract"] = "installed"
        # Tesseract実行可能かテスト
        try:
            version = pytesseract.get_tesseract_version()
            deps["tesseract_binary"] = f"available (v{version})"
        except Exception as e:
            deps["tesseract_binary"] = f"not_available ({str(e)[:50]})"
    except ImportError:
        deps["pytesseract"] = "not_installed"
        deps["tesseract_binary"] = "not_available"
    
    # OpenCV依存関係
    try:
        import cv2
        deps["opencv"] = safe_get_version("cv2", cv2)
    except ImportError:
        deps["opencv"] = "not_installed"
    
    # PDF依存関係
    try:
        import fitz
        if hasattr(fitz, 'version') and isinstance(fitz.version, tuple):
            deps["pymupdf"] = fitz.version[0]
        else:
            deps["pymupdf"] = "installed"
    except ImportError:
        deps["pymupdf"] = "not_installed"
    
    # NumPy
    try:
        import numpy
        deps["numpy"] = safe_get_version("numpy", numpy)
    except ImportError:
        deps["numpy"] = "not_installed"
    
    # Faker (alias生成用)
    try:
        from faker import Faker
        deps["faker"] = "installed"
    except ImportError:
        deps["faker"] = "not_installed"
    
    missing = [k for k, v in deps.items() if v == "not_installed"]
    
    # インストール推奨事項
    install_commands = []
    if "opencv" in missing:
        install_commands.append("pip install opencv-python-headless")
    if "pytesseract" in missing:
        install_commands.append("pip install pytesseract")
        install_commands.append("brew install tesseract  # macOS")
    if "pymupdf" in missing:
        install_commands.append("pip install pymupdf")
    if "numpy" in missing:
        install_commands.append("pip install numpy")
    if "faker" in missing:
        install_commands.append("pip install faker")
    
    return JSONResponse({
        "dependencies": deps,
        "missing": missing,
        "ready_for_pii_processing": len(missing) == 0,
        "install_commands": install_commands,
        "all_in_one_command": "pip install opencv-python-headless pytesseract pymupdf numpy faker"
    })

@app.post("/test/upload")
async def test_upload(
    file: UploadFile = File(...),
    description: str = Form("test upload")
):
    """ファイルアップロードテスト"""
    try:
        # ファイル検証
        if not file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が必要です")
        
        # 拡張子チェック
        ext = file.filename.lower().split('.')[-1]
        if ext not in settings.allowed_extensions_list:
            raise HTTPException(
                status_code=400, 
                detail=f"許可されていない拡張子: {ext}. 許可: {settings.allowed_extensions_list}"
            )
        
        # ファイルサイズチェック
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        
        if size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"ファイルサイズが大きすぎます: {size_mb:.1f}MB > {settings.max_file_size_mb}MB"
            )
        
        # 画像の場合はPILでテスト読み込み
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            try:
                img = Image.open(io.BytesIO(content))
                img_info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"画像読み込みエラー: {str(e)}")
        else:
            img_info = None
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_mb": round(size_mb, 2),
            "description": description,
            "image_info": img_info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/basic_ocr")
async def test_basic_ocr(file: UploadFile = File(...)):
    """基本OCRテスト（Tesseract使用）"""
    try:
        # ファイル検証
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        
        # TesseractでOCRテスト
        try:
            import pytesseract
            # シンプルなOCR実行
            text = pytesseract.image_to_string(img, lang='eng')
            
            return JSONResponse({
                "status": "success",
                "filename": file.filename,
                "image_size": img.size,
                "extracted_text": text.strip(),
                "text_length": len(text.strip()),
                "ocr_engine": "tesseract"
            })
            
        except ImportError:
            return JSONResponse({
                "status": "error",
                "message": "Tesseract not installed",
                "install_steps": [
                    "pip install pytesseract",
                    "brew install tesseract  # macOS",
                    "apt-get install tesseract-ocr  # Ubuntu"
                ]
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": f"OCR failed: {str(e)}",
                "hint": "Make sure tesseract is installed: brew install tesseract"
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/face_detection")
async def test_face_detection(file: UploadFile = File(...)):
    """基本顔検出テスト（OpenCV使用）"""
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert('RGB')
        
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(img)
            # BGR変換
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
            # Haar Cascade顔検出
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            face_count = len(faces)
            face_locations = [
                {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                for (x, y, w, h) in faces
            ]
            
            return JSONResponse({
                "status": "success",
                "filename": file.filename,
                "image_size": img.size,
                "faces_detected": face_count,
                "face_locations": face_locations,
                "detection_method": "opencv_haar"
            })
            
        except ImportError as e:
            missing = []
            try:
                import cv2
            except ImportError:
                missing.append("opencv-python-headless")
            try:
                import numpy
            except ImportError:
                missing.append("numpy")
                
            return JSONResponse({
                "status": "error",
                "message": f"Missing dependencies: {missing}",
                "install_command": f"pip install {' '.join(missing)}"
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/pii_detection")
async def test_pii_detection(text: str = Form(...)):
    """基本PII検出テスト（正規表現ベース）"""
    results = {
        "input_text": text,
        "detected_pii": []
    }
    
    # 基本的なPII検出パターンは維持
    patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'(?:\+81-\d{1,4}-\d{1,4}-\d{3,4})|(?:0\d{1,4}-\d{1,4}-\d{3,4})',
        "id": r'\b(?:ACC|USR|ORD)-\d{4,6}\b',
        "amount": r'(?:¥|JPY)\s?\d{1,3}(?:,\d{3})+'
    }
    
    for pii_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        for match in matches:
            results["detected_pii"].append({
                "type": pii_type,
                "text": match,
                "method": "regex"
            })
    
    return JSONResponse(results)

@app.get("/test/security_levels")
async def test_security_levels():
    """セキュリティレベルテスト"""
    return JSONResponse({
        "current_level": settings.default_security_level,
        "security_features": settings.security_features,
        "nanobanan_configured": settings.is_nanobanan_configured
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)