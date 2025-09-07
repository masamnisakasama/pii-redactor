# app/routers/detect_v2.py
import io, asyncio, logging
from importlib import import_module
from typing import Optional, Dict, Set

from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# 既存モジュール（存在しない環境でも落ちないようガード）
try:
    from app.security_manager import SecurityToggleManager, SecurityLevel
except Exception:
    SecurityToggleManager = None
    SecurityLevel = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from app import detectors
except Exception:
    detectors = None


router = APIRouter()

def _detect_faces_any(img):
    if detectors is None:
        return []
    if hasattr(detectors, "detect_faces_cv2"):
        try:
            return _detect_faces_any(img)
        except Exception:
            pass
    if hasattr(detectors, "detect_faces"):
        try:
            return detectors.detect_faces(img)
        except Exception:
            pass
    return []



def _get_security_manager(request: Request):
    """V2共通：app.state → app.main グローバル の順で取得"""
    sm = getattr(request.app.state, "security", None)
    if sm:
        return sm
    try:
        m = import_module("app.main")
        if SecurityToggleManager:
            # よくある名前 or 型走査
            for name in ("security_manager", "security_toggle_manager", "manager", "toggle_manager"):
                obj = getattr(m, name, None)
                if isinstance(obj, SecurityToggleManager):
                    return obj
            for name in dir(m):
                obj = getattr(m, name)
                if isinstance(obj, SecurityToggleManager):
                    return obj
    except Exception:
        pass
    return None


def _init_counts(policies: Set[str]) -> Dict[str, int]:
    keys = {"email", "phone", "address", "id", "face"}
    return {k: 0 for k in keys if (k in policies or k == "face")}


async def _ocr_with_processor_or_tesseract(sm, img, timeout_s: float) -> str:
    """現在のプロセッサでOCR→空ならpytesseractにフォールバック（タイムアウト付き）"""
    text = ""
    # 1) processor OCR（あれば）
    try:
        proc = getattr(sm, "get_current_processor", lambda: None)()
        if proc and hasattr(proc, "ocr_process"):
            items = await asyncio.wait_for(proc.ocr_process(img), timeout=timeout_s)
            text = "\n".join([it.get("text", "") if isinstance(it, dict) else str(it) for it in items]) or ""
    except Exception as e:
        logger.info("detect_v2: processor OCR fallback: %s", e)

    # 2) pytesseract
    if not text:
        try:
            import os
            from starlette.concurrency import run_in_threadpool
            import pytesseract
            lang = os.getenv("TESSERACT_LANG", "jpn+eng")
            text = await asyncio.wait_for(
                run_in_threadpool(pytesseract.image_to_string, img, lang=lang),
                timeout=timeout_s
            )
            text = (text or "").strip()
        except Exception as e:
            logger.info("detect_v2: pytesseract OCR failed: %s", e)
            text = ""
    return text


@router.post("/summary")
async def detect_summary_v2(
    request: Request,
    file: UploadFile = File(...),
    policy: str = Form("email,phone,address,id,face"),
    consistency_key: str = Form("ci"),  # 互換のため残す（ここでは未使用）
    ocr_timeout_s: float = Form(5.0),
):
    """
    画像 or PDF を解析して PII の件数を返す（既存 /detect/summary と互換）
    """
    sm = _get_security_manager(request)
    if not sm:
        return JSONResponse({"error": "Security manager not initialized"}, status_code=500)
    if detectors is None or Image is None:
        return JSONResponse({"error": "dependencies missing"}, status_code=500)

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty file"}, status_code=400)

    policies = {p.strip() for p in policy.split(",") if p.strip()}
    counts = _init_counts(policies)

    fname = (file.filename or "").lower()
    is_pdf = fname.endswith(".pdf")

    try:
        if is_pdf:
            if fitz is None:
                return JSONResponse({"error": "PDF engine not available"}, status_code=500)
            pdf = fitz.open(stream=raw, filetype="pdf")
            for page in pdf:
                # 顔
                if "face" in policies:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 解像度を上げて検出安定化
                        if Image:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            faces = _detect_faces_any(img) if hasattr(detectors, "detect_faces_cv2") else []
                            counts["face"] += len(faces)
                    except Exception:
                        pass


            # テキスト：まず get_text()、空ならOCRフォールバック
            if policies.intersection({"email", "phone", "address", "id"}):
                try:
                    t = page.get_text() or ""
                    if not t:
                        # 画像ベースのPDFならOCRする（現在のプロセッサ優先→pytesseract）
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        if Image:
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            t = await _ocr_with_processor_or_tesseract(sm, img, timeout_s=float(ocr_timeout_s))
                    if t and hasattr(detectors, "classify_by_regex"):
                        for h in detectors.classify_by_regex(normalizers.normalize_for_pii(t)):
                            k = h.get("type")
                            if k in counts and k in policies:
                                counts[k] += 1
                except Exception:
                    pass
            pdf.close()

        else:
            # 画像
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if "face" in policies and hasattr(detectors, "detect_faces_cv2"):
                faces = _detect_faces_any(img)
                counts["face"] += len(faces)
            if policies.intersection({"email", "phone", "address", "id"}):
                text = await _ocr_with_processor_or_tesseract(sm, img, timeout_s=float(ocr_timeout_s))
                if text and hasattr(detectors, "classify_by_regex"):
                    for h in detectors.classify_by_regex(text):
                        k = h.get("type")
                        if k in counts and k in policies:
                            counts[k] += 1
    except Exception as e:
        logger.exception("detect_v2 summary failed: %s", e)

    return JSONResponse({"counts": counts, "pii_found": any(counts.values())})


@router.post("/summary_fast")
async def detect_summary_fast_v2(
    file: UploadFile = File(...),
    policy: str = Form("email,phone,address,id,face"),
    ocr_timeout_s: float = Form(4.0),
):
    """
    軽量版：顔はHaar、テキストはpytesseract（タイムアウト）だけで件数を数える
    """
    if detectors is None or Image is None:
        return JSONResponse({"error": "dependencies missing"}, status_code=500)

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty file"}, status_code=400)

    policies = {p.strip() for p in policy.split(",") if p.strip()}
    counts = _init_counts(policies)

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    # 顔
    if "face" in policies and hasattr(detectors, "detect_faces_cv2"):
        faces = _detect_faces_any(img)
        counts["face"] += len(faces)

    # テキスト
    if policies.intersection({"email", "phone", "address", "id"}) and hasattr(detectors, "classify_by_regex"):
        text = ""
        try:
            import os
            from starlette.concurrency import run_in_threadpool
            import pytesseract
            lang = os.getenv("TESSERACT_LANG", "jpn+eng")
            text = await asyncio.wait_for(
                run_in_threadpool(pytesseract.image_to_string, img, lang=lang),
                timeout=float(ocr_timeout_s),
            )
            text = (text or "").strip()
        except Exception:
            text = ""
        if text:
            for h in detectors.classify_by_regex(text):
                k = h.get("type")
                if k in counts and k in policies:
                    counts[k] += 1

    return JSONResponse({"counts": counts, "pii_found": any(v > 0 for v in counts.values())})
