# app/main.py
# セキュリティトグル機能統合版メインAPI　フロントと組み合わせて使う　今の所バグありなのでまずはシンプル版でcurl通す
# 現状最もマシな出力：style=readable + replace_scope=token + font_path=PC内にあるフォント

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Set, Tuple
from PIL import Image
import io, hashlib, asyncio, re
import fitz  # PyMuPDF
import os
import logging

from . import detectors, alias, render_img, render_pdf
from .security_manager import SecurityToggleManager, SecurityLevel
from .face_redactor import redact_faces_image_bytes_enhanced
from .settings import Settings  
from .text_detect import detect_text_boxes_east # 文字検知の後、redactもmain.pyで
# ---------------------------------------------------
# アプリ/設定
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Redactor with Security Toggle",
    description="セキュリティレベルを選択可能な PII 編集システム",
    version="2.1.0",
)
settings = Settings()

# CORS
origins = [o.strip() for o in os.getenv("FRONTEND_ORIGINS","").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 明示したドメインだけにしてセキュリティー確保
    allow_credentials=True,       # Cookie等を使うならTrueにするがまだ未定
    allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
    allow_headers=["*"],
)
# セキュリティトグル（グローバル）
security_manager: Optional[SecurityToggleManager] = None


# ---------------------------------------------------
# 起動時
# ---------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global security_manager
    logger.info("Initializing Security Toggle Manager...")
    security_manager = SecurityToggleManager(settings)

    default_level = (
        SecurityLevel(settings.default_security_level)
        if hasattr(settings, "default_security_level")
        else SecurityLevel.MAXIMUM
    )
    security_manager.set_security_level(default_level)
    logger.info(f"Security Toggle Manager initialized with level: {default_level}")


# ---------------------------------------------------
# ヘルパ
# ---------------------------------------------------
def _validate_file(file: UploadFile) -> bool:
    if not file or not file.filename:
        return False
    allowed_ext = settings.allowed_extensions_list
    ext = file.filename.lower().split(".")[-1]
    return ext in allowed_ext


async def _load_file_bytes(file: UploadFile, image_url: Optional[str]) -> bytes:
    if file:
        if not _validate_file(file):
            raise HTTPException(status_code=400, detail="Invalid file type")
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        return content

    if image_url:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(image_url)
            r.raise_for_status()
            return r.content

    raise HTTPException(status_code=400, detail="No file or URL provided")


def _inline_replace_line(text: str, policies: set, consistency_key: str) -> str:
    """
    行テキスト内の email/phone/amount/id を “文字列として” 置換して返す。
    1パス置換なので、line 置換時に安定した見た目になる。
    detectors の正規表現（RE_EMAIL など）がある場合はそれを利用。
    """
    patterns: List[Tuple[str, re.Pattern]] = []
    if "email" in policies and hasattr(detectors, "RE_EMAIL"):
        patterns.append(("email", detectors.RE_EMAIL))
    if "phone" in policies and hasattr(detectors, "RE_PHONE"):
        patterns.append(("phone", detectors.RE_PHONE))
    if "amount" in policies and hasattr(detectors, "RE_AMOUNT"):
        patterns.append(("amount", detectors.RE_AMOUNT))
    if "id" in policies and hasattr(detectors, "RE_ID"):
        patterns.append(("id", detectors.RE_ID))

    if not patterns:
        return text

    union = re.compile("|".join(f"({p.pattern})" for _, p in patterns))

    def repl(m: re.Match) -> str:
        idx = next(i for i, g in enumerate(m.groups(), start=1) if g is not None)
        kind, _ = patterns[idx - 1]
        orig = m.group(0)
        if kind == "amount":
            new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
            return f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
        mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
        return alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)

    return union.sub(repl, text)


def _generate_alias(kind: str, orig: str, consistency_key: str) -> str:
    """個別トークン置換用エイリアス生成（互換維持）"""
    if kind == "amount":
        new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
        return f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
    mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
    return alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """行 bbox 重複チェック用の IoU（0〜1）"""
    ax1, ay1, ax2, ay2 = map(int, a)
    bx1, by1, bx2, by2 = map(int, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter or 1
    return inter / union


# ---------------------------------------------------
# セキュリティ操作
# ---------------------------------------------------
@app.get("/security/status")
async def get_security_status():
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    return JSONResponse(security_manager.get_security_info())


@app.post("/security/level")
async def set_security_level(level: str = Form(...)):
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    try:
        security_level = SecurityLevel(level)
    except ValueError:
        available = [l.value for l in security_manager.get_available_levels()]
        raise HTTPException(status_code=400, detail=f"Invalid security level. Available: {available}")
    if not security_manager.set_security_level(security_level):
        raise HTTPException(status_code=400, detail="Failed to set security level")
    return JSONResponse({"message": f"Security level set to {level}", "security_info": security_manager.get_security_info()})


# ---------------------------------------------------
# プレビュー　
# 描画せずボックス境界の情報を返す感じ
# ---------------------------------------------------
@app.post("/redact/preview")
async def preview_with_security_toggle(
    policy: str = Form("email,name,phone,id,amount,address"),
    consistency_key: str = Form("default"),
    security_level: Optional[str] = Form(None),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
    font_path: Optional[str] = Form(None),
):
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    if not file and not image_url:
        return JSONResponse({"items": []})

    # 一時レベル
    original_level = security_manager.current_level
    if security_level:
        try:
            security_manager.set_security_level(SecurityLevel(security_level))
        except ValueError:
            available = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(status_code=400, detail=f"Invalid security level. Available: {available}")

    try:
        blob = await _load_file_bytes(file, image_url)
        policies = set([p.strip() for p in policy.split(",") if p.strip()])
        items: List[Dict] = []
        is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

        if is_pdf:
            doc = fitz.open(stream=blob, filetype="pdf")
            for page_idx, page in enumerate(doc):
                text_blocks = page.get_text("dict")
                for block in text_blocks.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        for span in line.get("spans", []):
                            t = span.get("text", "").strip()
                            if not t:
                                continue
                            line_text += t + " "
                            bbox = span.get("bbox")
                            if bbox:
                                if line_bbox is None:
                                    line_bbox = list(bbox)
                                else:
                                    line_bbox[0] = min(line_bbox[0], bbox[0])
                                    line_bbox[1] = min(line_bbox[1], bbox[1])
                                    line_bbox[2] = max(line_bbox[2], bbox[2])
                                    line_bbox[3] = max(line_bbox[3], bbox[3])
                        if not (line_text.strip() and line_bbox):
                            continue

                        regex_hits = detectors.classify_by_regex(line_text)
                        if security_manager.current_level != SecurityLevel.MAXIMUM:
                            proc = security_manager.get_current_processor()
                            ai_ner = await proc.ner_process([line_text])
                            regex_hits.extend(ai_ner)

                        for hit in regex_hits:
                            if hit.get("type") in policies:
                                sub_bbox = render_img.subbbox_from_match(
                                    line_text=line_text,
                                    match_text=hit.get("text", ""),
                                    line_bbox=line_bbox,
                                    font_path=font_path,
                                )
                                items.append({
                                    "page": page_idx,
                                    "bbox": sub_bbox,
                                    "text": hit.get("text", ""),
                                    "type": hit.get("type", "unknown"),
                                    "confidence": hit.get("conf", 0.85),
                                    "reason": hit.get("reason", "pdf_text_extraction"),
                                    "security_level": security_manager.current_level.value,
                                })
            doc.close()

        else:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            if max(img.size) > settings.max_image_size:
                img.thumbnail((settings.max_image_size, settings.max_image_size), Image.Resampling.LANCZOS)

            ai_results = await security_manager.process_document(img, policies)
            for ocr_result in ai_results.get("ocr_results", []):
                text = ocr_result["text"]
                for hit in detectors.classify_by_regex(text):
                    if hit["type"] in policies:
                        sub_bbox = render_img.subbbox_from_match(
                            line_text=text, match_text=hit["text"], line_bbox=ocr_result["bbox"], font_path=font_path
                        )
                        items.append({
                            "bbox": sub_bbox,
                            "text": hit["text"],
                            "type": hit["type"],
                            "confidence": ocr_result.get("confidence", 0.85),
                            "reason": f"{ocr_result.get('method','ocr')}_regex",
                            "security_level": security_manager.current_level.value,
                        })

            for ner_result in ai_results.get("ner_results", []):
                if ner_result["type"] in policies:
                    items.append({
                        "bbox": (0, 0, img.width // 4, 30),
                        "text": ner_result["text"],
                        "type": ner_result["type"],
                        "confidence": ner_result["conf"],
                        "reason": ner_result["reason"],
                        "security_level": security_manager.current_level.value,
                    })

            if "face" in policies:
                for face_result in ai_results.get("face_results", []):
                    items.append({
                        "bbox": face_result["bbox"],
                        "text": "顔",
                        "type": "face",
                        "confidence": face_result["confidence"],
                        "reason": face_result["method"],
                        "security_level": security_manager.current_level.value,
                    })

        return JSONResponse({
            "items": items,
            "total_count": len(items),
            "security_level": security_manager.current_level.value,
            "processor_used": type(security_manager.get_current_processor()).__name__,
            "ai_features_enabled": security_manager.current_level != SecurityLevel.MAXIMUM,
        })

    finally:
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)


# ---------------------------------------------------
# 実際に描写して置換するエンドポイント
# file / image_url：入力を行う
# policy：email,name,phone,id,amount,address,face...：検出対象を列挙
# style：readable / box / pixelate / blur：描画モード決定
# replace_scope：token（既定） or line：トークン or Line
# font_path：readable の精度向上用に設置　ないと崩れるので意外と大事
# consistency_key：同一キーデータの別名一貫性維持用。
# security_level：一時的なレベル上書き。
# ---------------------------------------------------
@app.post("/redact/replace")
async def replace_with_security_toggle(
    policy: str = Form("email,name,phone,id,amount,address"),
    style: str = Form("readable"),
    consistency_key: str = Form("default"),
    security_level: Optional[str] = Form(None),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
    font_path: Optional[str] = Form(None),
    replace_scope: str = Form("token"),  # "token" or "line"
):
    """
    置換（描画あり）
    - token: マッチ部分だけ sub-bbox で描画
    - line : その行のテキストを一括置換して、行 bbox に 1 回だけ描画（IoUで重複描画ガード）
    """
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    if not file and not image_url:
        return JSONResponse({"error": "no input"}, status_code=400)

    allowed_styles = {"readable", "box", "pixelate", "blur"}
    if style not in allowed_styles:
        style = "readable"
    if replace_scope not in {"token", "line"}:
        replace_scope = "token"

    # 一時レベル
    original_level = security_manager.current_level
    if security_level:
        try:
            security_manager.set_security_level(SecurityLevel(security_level))
        except ValueError:
            available = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(status_code=400, detail=f"Invalid security level. Available: {available}")

    try:
        blob = await _load_file_bytes(file, image_url)
        policies = set(p.strip() for p in policy.split(",") if p.strip())
        is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

        # =========================
        # PDF（ラスタ処理→再合成）
        # =========================
        if is_pdf:
            pages = render_pdf.pdf_to_images(blob, dpi=settings.pdf_dpi)
            hits_by_page: Dict[int, List[Dict]] = {}
            seen_lines_per_page: Dict[int, List[Tuple[int, int, int, int]]] = {}
            IOU_TH = 0.6  # 行の重複検出しきい値

            for page_idx, pimg in enumerate(pages):
                ai_results = await security_manager.process_document(pimg, policies)

                for ocr_result in ai_results.get("ocr_results", []):
                    line_text = ocr_result.get("text", "")
                    line_bbox = ocr_result.get("bbox")
                    if not line_text or not line_bbox:
                        continue

                    regex_hits = [h for h in detectors.classify_by_regex(line_text) if h.get("type") in policies]
                    if not regex_hits:
                        continue

                    if replace_scope == "line":
                        lb = tuple(map(int, line_bbox))
                        # 既に近い行を描いていればスキップ（ゴースト防止）
                        prev = seen_lines_per_page.get(page_idx, [])
                        if any(_iou(lb, b) > IOU_TH for b in prev):
                            continue
                        seen_lines_per_page.setdefault(page_idx, []).append(lb)

                        new_line = _inline_replace_line(line_text, policies, consistency_key)
                        hits_by_page.setdefault(page_idx, []).append({"bbox": lb, "new": new_line})
                    else:
                        # token: 部分 bbox を推定して個別に描画
                        for h in regex_hits:
                            alias_txt = _generate_alias(h["type"], h["text"], consistency_key)
                            sub_bbox = render_img.subbbox_from_match(
                                line_text=line_text, match_text=h["text"], line_bbox=line_bbox, font_path=font_path
                            )
                            hits_by_page.setdefault(page_idx, []).append({"bbox": sub_bbox, "new": alias_txt})

            pdf_bytes = render_pdf.process_pdf_raster(blob, hits_by_page, style=style)
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={"X-Security-Level": security_manager.current_level.value},
            )

        # =========================
        # 画像（直接描画）
        # =========================
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        if max(img.size) > settings.max_image_size:
            img.thumbnail((settings.max_image_size, settings.max_image_size), Image.Resampling.LANCZOS)

        ai_results = await security_manager.process_document(img, policies)
        seen_line_boxes_img: List[Tuple[int, int, int, int]] = []
        IOU_TH_IMG = 0.6

        for ocr_result in ai_results.get("ocr_results", []):
            line_text = ocr_result.get("text", "")
            line_bbox = ocr_result.get("bbox")
            if not line_text or not line_bbox:
                continue

            regex_hits = [h for h in detectors.classify_by_regex(line_text) if h.get("type") in policies]
            if not regex_hits:
                continue

            if replace_scope == "line":
                lb = tuple(map(int, line_bbox))
                if any(_iou(lb, b) > IOU_TH_IMG for b in seen_line_boxes_img):
                    continue
                seen_line_boxes_img.append(lb)

                new_line = _inline_replace_line(line_text, policies, consistency_key)
                render_img.draw_replace(img, lb, new_line, mode=style, font_path=font_path)
            else:
                for h in regex_hits:
                    alias_txt = _generate_alias(h["type"], h["text"], consistency_key)
                    sub_bbox = render_img.subbbox_from_match(
                        line_text=line_text, match_text=h["text"], line_bbox=line_bbox, font_path=font_path
                    )
                    render_img.draw_replace(img, sub_bbox, alias_txt, mode=style, font_path=font_path)

        # 顔（必要なら）
        if "face" in policies:
            face_results = ai_results.get("face_results", [])
            if face_results:
                import numpy as np
                arr = np.array(img)
                for fr in face_results:
                    x1, y1, x2, y2 = map(int, fr.get("bbox", (0, 0, 0, 0)))
                    if x2 > x1 and y2 > y1:
                        region = arr[y1:y2, x1:x2]
                        if region.size > 0:
                            small = Image.fromarray(region).resize((16, 16), Image.Resampling.NEAREST)
                            arr[y1:y2, x1:x2] = np.array(small.resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST))
                img = Image.fromarray(arr)

        # === EAST text redaction (optional; OFF by default) =================
        if settings.use_east_text:
            try:
                import numpy as np, cv2, os
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                boxes = detect_text_boxes_east(
                    bgr,
                    model_path=(settings.east_pb or os.getenv("EAST_PB") or "models/text/frozen_east_text_detection.pb"),
                    conf_thr=settings.east_conf_threshold,
                    nms_thr=settings.east_nms_threshold,
                    max_side=settings.east_max_side,
                    min_size=settings.east_min_size,
                )
                # ここでは安全側に倒して、検出テキスト領域を角丸ボックスで塗る
                for (x, y, w, h) in boxes:
                    render_img.draw_replace(img, (x, y, x + w, y + h), "█", mode="box", font_path=None)
                logger.info(f"[east] masked text boxes: {len(boxes)}")
            except Exception as e:
                logger.warning(f"EAST text redaction skipped: {e}")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png", headers={"X-Security-Level": security_manager.current_level.value})

    finally:
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)


# ---------------------------------------------------
# 顔画像の編集
# 顔だけ処理（pixelate/blur/box/smart_blur選択可能）
# ---------------------------------------------------
@app.post("/redact/face_image")
async def redact_face_with_security_toggle(
    file: UploadFile = File(...),
    method: str = Query("pixelate", pattern="^(pixelate|pixelate_strict|blur|box|smart_blur|replace_face)$"),
    strength: int = Query(16, ge=1, le=200),
    expand: float = Query(0.12, ge=0.0, le=0.5),
    out_format: str = Query("PNG", pattern="^(PNG|png|JPG|JPEG|jpg|jpeg)$"),
    security_level: Optional[str] = Query(None),
    persona: Optional[str] = Query(None),
    persona_form: Optional[str] = Form(None), # 日本語対応のためフォームでもうける形
):
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    if not _validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file type")

    original_level = security_manager.current_level
    if security_level:
        try:
            security_manager.set_security_level(SecurityLevel(security_level))
        except ValueError:
            available = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(status_code=400, detail=f"Invalid security level. Available: {available}")

    try:
        data = await file.read()
        if len(data) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        use_advanced = security_manager.current_level == SecurityLevel.ENHANCED
        nanobanan_config = None
        if use_advanced and hasattr(settings, "nanobanan_api_key"):
            nanobanan_config = {"api_key": settings.nanobanan_api_key, "endpoint": settings.nanobanan_endpoint}

        # フォーム優先でマージする感じ（フォームがあればフォーム値、なければクエリ値）
        persona_val = persona_form if persona_form is not None else persona

        out_bytes = await redact_faces_image_bytes_enhanced(
            data,
            method=method,
            strength=strength,
            expand=expand,
            out_format=out_format,
            use_multiple_detectors=use_advanced,
            nanobanan_api_key=(nanobanan_config.get("api_key") if nanobanan_config else None),
            nanobanan_endpoint=(nanobanan_config.get("endpoint") if nanobanan_config else None),
            persona=persona_val,
        )

        mt = "image/jpeg" if out_format.lower() in ("jpg", "jpeg") else "image/png"
        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type=mt,
            headers={"X-Security-Level": security_manager.current_level.value, "X-Detection-Method": "advanced" if use_advanced else "basic"},
        )
    finally:
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)


# ---------------------------------------------------
# その他
# /capabilitiesはセキュリティレベル設定（真ん中二つはとりあえず今はなしで）
# ---------------------------------------------------
@app.get("/capabilities")
async def get_capabilities():
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")

    def _desc(level: SecurityLevel) -> str:
        d = {
            SecurityLevel.MAXIMUM: "最高セキュリティ：完全オフライン処理（OpenCV + Tesseract）",
            SecurityLevel.HIGH: "高セキュリティ：オンプレミスAI + 限定的外部API",
            SecurityLevel.STANDARD: "標準セキュリティ：バランス型（一部API使用）",
            SecurityLevel.ENHANCED: "AI機能優先：高精度処理（Nanobanan API等使用）",
        }
        return d.get(level, "Unknown security level")

    capabilities = {
        "security_levels": {
            level.value: {"name": level.value.title(), "description": _desc(level), "available": level in security_manager.get_available_levels()}
            for level in SecurityLevel
        },
        "current_level": security_manager.current_level.value,
        "supported_formats": settings.allowed_extensions_list,
        "max_file_size_mb": settings.max_file_size_mb,
        "features": {"ocr": True, "ner": True, "face_detection": True, "pdf_processing": True, "batch_processing": False},
    }
    return JSONResponse(capabilities)


@app.get("/health")
async def health_check():
    if not security_manager:
        return JSONResponse({"status": "error", "message": "Security manager not initialized"})
    return JSONResponse({
        "status": "healthy",
        "security_manager": "initialized",
        "current_security_level": security_manager.current_level.value,
        "available_processors": len(security_manager.processors),
        "timestamp": int(asyncio.get_event_loop().time()),
    })
