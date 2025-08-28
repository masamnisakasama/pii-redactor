# app/main.py
# セキュリティトグル機能統合版メインAPI
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Union
from PIL import Image
import io, hashlib, asyncio
import fitz  # PyMuPDF
from .settings import Settings
from . import detectors, alias, render_img, render_pdf
from .security_manager import SecurityToggleManager, SecurityLevel
from .face_redactor import redact_faces_image_bytes_enhanced
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Redactor with Security Toggle",
    description="セキュリティレベルを選択可能なPII編集システム",
    version="2.1.0"
)

settings = Settings()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin.split(',') if settings.cors_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# グローバルなセキュリティマネージャー
security_manager = None

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    global security_manager
    
    logger.info("Initializing Security Toggle Manager...")
    security_manager = SecurityToggleManager(settings)
    
    # デフォルトのセキュリティレベルを設定
    default_level = SecurityLevel(settings.default_security_level) if hasattr(settings, 'default_security_level') else SecurityLevel.MAXIMUM
    security_manager.set_security_level(default_level)
    
    logger.info(f"Security Toggle Manager initialized with level: {default_level}")

def _validate_file(file: UploadFile) -> bool:
    """ファイル検証"""
    if not file or not file.filename:
        return False
    
    allowed_ext = settings.allowed_extensions_list
    ext = file.filename.lower().split('.')[-1]
    return ext in allowed_ext

async def _load_file_bytes(file: UploadFile, image_url: Optional[str]) -> bytes:
    """ファイル読み込み"""
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

@app.get("/security/status")
async def get_security_status():
    """現在のセキュリティ状況を取得"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    return JSONResponse(security_manager.get_security_info())

@app.post("/security/level")
async def set_security_level(level: str = Form(...)):
    """セキュリティレベルを変更"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    try:
        security_level = SecurityLevel(level)
    except ValueError:
        available_levels = [l.value for l in security_manager.get_available_levels()]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid security level. Available: {available_levels}"
        )
    
    success = security_manager.set_security_level(security_level)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set security level")
    
    return JSONResponse({
        "message": f"Security level set to {level}",
        "security_info": security_manager.get_security_info()
    })

@app.post("/redact/preview")
async def preview_with_security_toggle(
    policy: str = Form("email,name,phone,id,amount,address"),
    consistency_key: str = Form("default"),
    security_level: Optional[str] = Form(None),  # 一時的なレベル変更
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    """セキュリティレベル対応プレビュー機能"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    if not file and not image_url:
        return JSONResponse({"items": []})
    
    # 一時的なセキュリティレベル変更
    original_level = security_manager.current_level
    if security_level:
        try:
            temp_level = SecurityLevel(security_level)
            security_manager.set_security_level(temp_level)
        except ValueError:
            available_levels = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid security level. Available: {available_levels}"
            )
    
    try:
        blob = await _load_file_bytes(file, image_url)
        policies = set([p.strip() for p in policy.split(",") if p.strip()])
        
        items: List[Dict] = []
        is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))
        
        if is_pdf:
            # PDF処理
            doc = fitz.open(stream=blob, filetype="pdf")
            
            for page_idx, page in enumerate(doc):
                # テキスト抽出（PyMuPDF - オフライン）
                text_blocks = page.get_text("dict")
                
                # 各ブロックを処理
                for block in text_blocks.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                bbox = span.get("bbox")
                                if bbox:
                                    if line_bbox is None:
                                        line_bbox = list(bbox)
                                    else:
                                        line_bbox[0] = min(line_bbox[0], bbox[0])
                                        line_bbox[1] = min(line_bbox[1], bbox[1])
                                        line_bbox[2] = max(line_bbox[2], bbox[2])
                                        line_bbox[3] = max(line_bbox[3], bbox[3])
                        
                        if line_text.strip() and line_bbox:
                            # 正規表現による基本検出
                            regex_hits = detectors.classify_by_regex(line_text)
                            
                            # AI NER処理（セキュリティレベルに応じて）
                            if security_manager.current_level != SecurityLevel.MAXIMUM:
                                ai_results = await security_manager.process_document(
                                    None, policies, text=line_text  # テキストのみ渡す
                                )
                                regex_hits.extend(ai_results.get('ner_results', []))
                            
                            for hit in regex_hits:
                                if hit.get("type") in policies:
                                    items.append({
                                        "page": page_idx,
                                        "bbox": line_bbox,
                                        "text": hit.get("text", ""),
                                        "type": hit.get("type", "unknown"),
                                        "confidence": hit.get("conf", hit.get("confidence", 0.8)),
                                        "reason": hit.get("reason", "pdf_text_extraction"),
                                        "security_level": security_manager.current_level.value
                                    })
            
            doc.close()
        
        else:
            # 画像処理（AI強化版）
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            
            # 画像サイズ調整
            max_size = settings.max_image_size
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # セキュリティマネージャーを使用してAI処理
            ai_results = await security_manager.process_document(img, policies)
            
            # OCR結果の処理
            for ocr_result in ai_results.get('ocr_results', []):
                # 正規表現でタイプ判定
                text = ocr_result['text']
                regex_hits = detectors.classify_by_regex(text)
                
                for hit in regex_hits:
                    if hit["type"] in policies:
                        items.append({
                            "bbox": ocr_result["bbox"],
                            "text": hit["text"],
                            "type": hit["type"],
                            "confidence": min(ocr_result["confidence"], hit["conf"]),
                            "reason": f"{ocr_result['method']}_regex",
                            "security_level": security_manager.current_level.value
                        })
            
            # NER結果の追加
            for ner_result in ai_results.get('ner_results', []):
                if ner_result["type"] in policies:
                    items.append({
                        "bbox": (0, 0, img.width//4, 30),  # NERは大まかな位置
                        "text": ner_result["text"],
                        "type": ner_result["type"],
                        "confidence": ner_result["conf"],
                        "reason": ner_result["reason"],
                        "security_level": security_manager.current_level.value
                    })
            
            # 顔検出結果の追加（face がポリシーに含まれる場合）
            if "face" in policies:
                for face_result in ai_results.get('face_results', []):
                    items.append({
                        "bbox": face_result["bbox"],
                        "text": "顔",
                        "type": "face",
                        "confidence": face_result["confidence"],
                        "reason": face_result["method"],
                        "security_level": security_manager.current_level.value
                    })
        
        return JSONResponse({
            "items": items,
            "total_count": len(items),
            "security_level": security_manager.current_level.value,
            "processor_used": type(security_manager.get_current_processor()).__name__,
            "ai_features_enabled": security_manager.current_level != SecurityLevel.MAXIMUM
        })
    
    finally:
        # セキュリティレベルを元に戻す
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)

@app.post("/redact/replace")
async def replace_with_security_toggle(
    policy: str = Form("email,name,phone,id,amount,address"),
    style: str = Form("readable"),
    consistency_key: str = Form("default"),
    security_level: Optional[str] = Form(None),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    """セキュリティレベル対応置換機能"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    if not file and not image_url:
        return JSONResponse({"error": "no input"}, status_code=400)
    
    # 一時的なセキュリティレベル変更
    original_level = security_manager.current_level
    if security_level:
        try:
            temp_level = SecurityLevel(security_level)
            security_manager.set_security_level(temp_level)
        except ValueError:
            available_levels = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid security level. Available: {available_levels}"
            )
    
    try:
        blob = await _load_file_bytes(file, image_url)
        policies = set([p.strip() for p in policy.split(",") if p.strip()])
        is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))
        
        if is_pdf:
            # PDF処理
            pages = render_pdf.pdf_to_images(blob, dpi=settings.pdf_dpi)
            hits_by_page: Dict[int, list[Dict]] = {}
            
            for i, pimg in enumerate(pages):
                # AI処理
                ai_results = await security_manager.process_document(pimg, policies)
                
                # OCR結果から置換対象を抽出
                for ocr_result in ai_results.get('ocr_results', []):
                    text = ocr_result['text']
                    regex_hits = detectors.classify_by_regex(text)
                    
                    for hit in regex_hits:
                        if hit["type"] in policies:
                            new_val = _generate_alias(hit["type"], hit["text"], consistency_key)
                            hits_by_page.setdefault(i, []).append({
                                "bbox": ocr_result["bbox"], 
                                "new": new_val
                            })
                
                # NER結果の追加処理（位置推定が必要）
                for ner_result in ai_results.get('ner_results', []):
                    if ner_result["type"] in policies:
                        # NERの場合は大まかな位置を推定（実装は簡略化）
                        estimated_bbox = (50, 50 + i * 30, 200, 80 + i * 30)
                        new_val = _generate_alias(ner_result["type"], ner_result["text"], consistency_key)
                        hits_by_page.setdefault(i, []).append({
                            "bbox": estimated_bbox,
                            "new": new_val
                        })
            
            pdf_bytes = render_pdf.process_pdf_raster(blob, hits_by_page, style=style)
            return StreamingResponse(
                io.BytesIO(pdf_bytes), 
                media_type="application/pdf",
                headers={"X-Security-Level": security_manager.current_level.value}
            )
        
        else:
            # 画像処理
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            
            # 画像サイズ調整
            if max(img.size) > settings.max_image_size:
                img.thumbnail((settings.max_image_size, settings.max_image_size), Image.Resampling.LANCZOS)
            
            # AI処理
            ai_results = await security_manager.process_document(img, policies)
            
            # OCR結果の置換
            for ocr_result in ai_results.get('ocr_results', []):
                text = ocr_result['text']
                regex_hits = detectors.classify_by_regex(text)
                
                for hit in regex_hits:
                    if hit["type"] in policies:
                        new_val = _generate_alias(hit["type"], hit["text"], consistency_key)
                        render_img.draw_replace(img, ocr_result["bbox"], new_val, mode=style, pad=6)
            
            # 顔の処理（顔認識結果がある場合）
            face_results = ai_results.get('face_results', [])
            if face_results and "face" in policies:
                # 顔にモザイク処理を適用
                import numpy as np
                img_array = np.array(img)
                
                for face_result in face_results:
                    bbox = face_result["bbox"]
                    x1, y1, x2, y2 = bbox
                    
                    # 簡単なモザイク処理
                    face_region = img_array[y1:y2, x1:x2]
                    if face_region.size > 0:
                        small = Image.fromarray(face_region).resize((16, 16), Image.Resampling.NEAREST)
                        pixelated = small.resize((x2-x1, y2-y1), Image.Resampling.NEAREST)
                        img_array[y1:y2, x1:x2] = np.array(pixelated)
                
                img = Image.fromarray(img_array)
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(
                buf, 
                media_type="image/png",
                headers={"X-Security-Level": security_manager.current_level.value}
            )
    
    finally:
        # セキュリティレベルを元に戻す
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)

@app.post("/redact/face_image")
async def redact_face_with_security_toggle(
    file: UploadFile = File(...),
    method: str = Query("pixelate", pattern="^(pixelate|blur|box|smart_blur)$"),
    strength: int = Query(16, ge=1, le=200),
    expand: float = Query(0.12, ge=0.0, le=0.5),
    out_format: str = Query("PNG", pattern="^(PNG|png|JPG|JPEG|jpg|jpeg)$"),
    security_level: Optional[str] = Query(None),
):
    """セキュリティレベル対応顔編集機能"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    if not _validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # 一時的なセキュリティレベル変更
    original_level = security_manager.current_level
    if security_level:
        try:
            temp_level = SecurityLevel(security_level)
            security_manager.set_security_level(temp_level)
        except ValueError:
            available_levels = [l.value for l in security_manager.get_available_levels()]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid security level. Available: {available_levels}"
            )
    
    try:
        data = await file.read()
        if len(data) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        # セキュリティレベルに応じて顔認識方法を選択
        use_advanced = security_manager.current_level == SecurityLevel.ENHANCED
        nanobanan_config = None
        
        if use_advanced and hasattr(settings, 'nanobanan_api_key'):
            nanobanan_config = {
                'api_key': settings.nanobanan_api_key,
                'endpoint': settings.nanobanan_endpoint
            }
        
        out_bytes = await redact_faces_image_bytes_enhanced(
            data, 
            method=method, 
            strength=strength, 
            expand=expand, 
            out_format=out_format,
            use_multiple_detectors=use_advanced,
            nanobanan_api_key=nanobanan_config.get('api_key') if nanobanan_config else None,
            nanobanan_endpoint=nanobanan_config.get('endpoint') if nanobanan_config else None
        )
        
        mt = "image/jpeg" if out_format.lower() in ("jpg","jpeg") else "image/png"
        return StreamingResponse(
            io.BytesIO(out_bytes), 
            media_type=mt,
            headers={
                "X-Security-Level": security_manager.current_level.value,
                "X-Detection-Method": "advanced" if use_advanced else "basic"
            }
        )
    
    finally:
        # セキュリティレベルを元に戻す
        if security_level and original_level != security_manager.current_level:
            security_manager.set_security_level(original_level)

def _generate_alias(kind: str, orig: str, consistency_key: str) -> str:
    """エイリアス生成のヘルパー関数"""
    if kind == "amount":
        new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
        return f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
    else:
        mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
        return alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)

@app.get("/capabilities")
async def get_capabilities():
    """システムの機能一覧を取得"""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    capabilities = {
        "security_levels": {
            level.value: {
                "name": level.value.title(),
                "description": _get_security_level_description(level),
                "available": level in security_manager.get_available_levels()
            }
            for level in SecurityLevel
        },
        "current_level": security_manager.current_level.value,
        "supported_formats": settings.allowed_extensions_list,
        "max_file_size_mb": settings.max_file_size_mb,
        "features": {
            "ocr": True,
            "ner": True,
            "face_detection": True,
            "pdf_processing": True,
            "batch_processing": False  # 将来実装予定
        }
    }
    
    return JSONResponse(capabilities)

def _get_security_level_description(level: SecurityLevel) -> str:
    """セキュリティレベルの説明を取得"""
    descriptions = {
        SecurityLevel.MAXIMUM: "最高セキュリティ：完全オフライン処理（OpenCV + Tesseract）",
        SecurityLevel.HIGH: "高セキュリティ：オンプレミスAI + 限定的外部API",
        SecurityLevel.STANDARD: "標準セキュリティ：バランス型（一部API使用）",
        SecurityLevel.ENHANCED: "AI機能優先：高精度処理（Nanobanan API等使用）"
    }
    return descriptions.get(level, "Unknown security level")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    if not security_manager:
        return JSONResponse({"status": "error", "message": "Security manager not initialized"})
    
    return JSONResponse({
        "status": "healthy",
        "security_manager": "initialized",
        "current_security_level": security_manager.current_level.value,
        "available_processors": len(security_manager.processors),
        "timestamp": int(asyncio.get_event_loop().time())
    })
"""
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List, Dict
from PIL import Image
import io, httpx, hashlib
import fitz  # PyMuPDF使うのに必要
from .settings import Settings
from . import tf_infer, detectors, alias, render_img, render_pdf
from fastapi.responses import Response
from .face_redactor import redact_faces_image_bytes


app = FastAPI(title="Secure PII Redactor")
settings = Settings()

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

async def _load_bytes(file: UploadFile | None, image_url: Optional[str]) -> bytes:
    if file:
        return await file.read()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(image_url)  # 署名URL　VPCエンドポイント使いたいので
        r.raise_for_status()
        return r.content

# 行文字列に LINE_PATTERNS を当て、(bbox, kind, orig_text) をyield
def _iter_line_hits(lines, policies: set[str]):
    for line in lines:
        tokens = line["tokens"]
        pieces, offsets, cur = [], [], 0
        for t in tokens:
            s = t["text"]
            pieces.append(s)
            offsets.append((cur, cur + len(s)))
            cur += len(s) + 1
        line_text = " ".join(pieces)

        for kind, pat in detectors.LINE_PATTERNS:
            for m in pat.finditer(line_text):
                if kind not in policies:
                    continue
                a, b = m.span()
                used = [i for i, (sa, sb) in enumerate(offsets) if not (b <= sa or sb <= a)]
                if not used:
                    continue
                x1 = y1 = 10**9
                x2 = y2 = -10**9
                orig = []
                for i in used:
                    bx1, by1, bx2, by2 = tokens[i]["bbox"]
                    x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2)
                    orig.append(tokens[i]["text"])
                yield (x1, y1, x2, y2), kind, " ".join(orig)

@app.post("/redact/preview")
async def preview(
    policy: str = Form("email,name,phone,id,amount,address"),
    consistency_key: str = Form("default"),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    if not file and not image_url:
        return JSONResponse({"items": []})
    blob = await _load_bytes(file, image_url)
    policies = set([p.strip() for p in policy.split(",") if p.strip()])

    items: List[Dict] = []
    is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

    if is_pdf:
        # PDF：PyMuPDFで行単位に変換
        doc = fitz.open(stream=blob, filetype="pdf")
        for page_idx, page in enumerate(doc):
            boxes = tf_infer.pdf_text_boxes(page)
            texts = [b["text"] for b in boxes]
            ner_hits = await tf_infer.ner_classify_texts(texts, settings.ner_endpoint)
            for b in boxes:
                rx = detectors.classify_by_regex(b["text"])
                merged = detectors.merge_with_ner(rx, [h for h in ner_hits if h["text"] in b["text"]])
                for m in merged:
                    if m["type"] not in policies:
                        continue
                    items.append({
                        "page": page_idx, "bbox": b["bbox"], "text": m["text"],
                        "type": m["type"], "confidence": m["conf"], "reason": m["reason"]
                    })
    else:
        # 画像：OCR → 行単位に変換
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        lines = tf_infer.ocr_lines(img)
        for bbox, kind, orig in _iter_line_hits(lines, policies):
            items.append({
                "bbox": bbox, "text": orig, "type": kind,
                "confidence": 0.90, "reason": "line-regex"
            })

    return JSONResponse({"items": items})

@app.post("/redact/replace")
async def replace(
    policy: str = Form("email,name,phone,id,amount,address"),
    style: str = Form("readable"),  # keep-font | readable
    consistency_key: str = Form("default"),
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    if not file and not image_url:
        return JSONResponse({"error": "no input"}, status_code=400)

    blob = await _load_bytes(file, image_url)
    policies = set([p.strip() for p in policy.split(",") if p.strip()])
    is_pdf = (file and file.filename.lower().endswith(".pdf")) or (image_url and image_url.lower().endswith(".pdf"))

    if is_pdf:
        # PDF: 画像化 → 行単位 → PDF再構成 
        pages = render_pdf.pdf_to_images(blob)
        hits_by_page: Dict[int, list[Dict]] = {}
        for i, pimg in enumerate(pages):
            lines = tf_infer.ocr_lines(pimg)
            for bbox, kind, orig in _iter_line_hits(lines, policies):
                if kind == "amount":
                    new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
                    new_val = f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
                else:
                    mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
                    new_val = alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)
                hits_by_page.setdefault(i, []).append({"bbox": bbox, "new": new_val})

        pdf_bytes = render_pdf.process_pdf_raster(blob, hits_by_page, style=style)
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")

    # 行単位での認識 → 置換描画
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    lines = tf_infer.ocr_lines(img)
    for bbox, kind, orig in _iter_line_hits(lines, policies):
        if kind == "amount":
            new_digits = alias.alias_value("amount", orig, settings.tenant_hmac_key_b64, consistency_key)
            new_val = f"JPY {new_digits.lstrip('¥').strip()}" if "JPY" in orig else new_digits
        else:
            mapk = {"email": "email", "phone": "phone", "id": "id"}.get(kind, kind)
            new_val = alias.alias_value(mapk, orig, settings.tenant_hmac_key_b64, consistency_key)
        render_img.draw_replace(img, bbox, new_val, mode=style, pad=6)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/redact/face_image")
async def redact_face_image(
    file: UploadFile = File(...),
    method: str = Query("pixelate", pattern="^(pixelate|blur|box)$"),
    strength: int = Query(16, ge=1, le=200),
    expand: float = Query(0.12, ge=0.0, le=0.5),
    out_format: str = Query("PNG", pattern="^(PNG|png|JPG|JPEG|jpg|jpeg)$"),
):
    data = await file.read()
    out_bytes = redact_faces_image_bytes(
        data, method=method, strength=strength, expand=expand, out_format=out_format
    )
    mt = "image/jpeg" if out_format.lower() in ("jpg","jpeg") else "image/png"
    return Response(content=out_bytes, media_type=mt)
"""