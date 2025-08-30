# app/face_redactor.py
# 精度向上版：複数の検出手法を組み合わせ、より高精度な顔認識を実現
# 追加: Nano-banana(Gemini)で顔を「別人」に置換する method=replace_face を実装
# 既存のモザイク/ブラー/黒塗りはそのまま維持（後方互換）

import io
from typing import List, Tuple, Literal, Optional
import numpy as np
import cv2
from PIL import Image, ImageOps
import httpx
import base64

# ==== 追加インポート（Gemini / Nano-banana 用） ============================
# google-genai は外部依存。未導入でも既存機能が動くように try/except で保護。
try:
    from google import genai
    _GENAI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False

# ===== 既存: メソッド種別 =================================================
Method = Literal["pixelate", "blur", "box", "smart_blur", "replace_face", "pixelate_strict"]

class FaceDetectionError(Exception):
    """顔検出関連のエラー"""
    pass

# ====== 既存: Haar 検出 ====================================================
def _detect_faces_haar(bgr: np.ndarray,
                       scale_factor: float = 1.1,
                       min_neighbors: int = 5,
                       min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int,int,int,int]]:
    """改善されたHaar Cascade検出器"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_files = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml"
    ]
    all_faces = []
    for cascade_file in cascade_files:
        try:
            cc = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
            if cc.empty():
                continue
            faces = cc.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            all_faces.extend(faces)
        except Exception:
            continue

    if not all_faces:
        return []
    return _merge_overlapping_faces(all_faces)

# ====== 既存: DNN 検出（任意/重い）=========================================
def _detect_faces_dnn(bgr: np.ndarray,
                      confidence_threshold: float = 0.5) -> List[Tuple[int,int,int,int]]:
    """OpenCV DNN（より高精度だが重い）"""
    try:
        net = cv2.dnn.readNetFromTensorflow(
            'opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt'
        )
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), [104, 117, 123])
        net.setInput(blob)
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces
    except Exception as e:
        print(f"DNN face detection failed: {e}")
        return []

# ====== 既存: Nanobanan（Gemini）による顔検出 ===============================
async def _detect_faces_nanobanan(bgr: np.ndarray,
                                  api_key: str,
                                  endpoint: str) -> List[Tuple[int,int,int,int]]:
    """NanobananaのAPIを使用した高精度顔検出（/v1/face-detect を想定）"""
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "image": img_b64,
            "detection_type": "face",
            "confidence_threshold": 0.3,
            "return_landmarks": False
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{endpoint}/v1/face-detect", json=payload, headers=headers)
            r.raise_for_status()
            result = r.json()

            faces = []
            for detection in result.get("faces", []):
                bbox = detection.get("bbox", [])
                confidence = float(detection.get("confidence", 0.0))
                if len(bbox) == 4 and confidence > 0.3:
                    x, y, w, h = bbox
                    faces.append((int(x), int(y), int(w), int(h)))
            return faces
    except Exception as e:
        print(f"Nanobanan face detection failed: {e}")
        return []

# ====== 既存: BBox マージ ====================================================
def _merge_overlapping_faces(faces: List[Tuple[int,int,int,int]],
                             iou_threshold: float = 0.3) -> List[Tuple[int,int,int,int]]:
    """重複する顔矩形を統合"""
    if not faces:
        return []

    def calculate_iou(box1, box2):
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0.0

    faces_with_area = [(f, f[2] * f[3]) for f in faces]
    faces_with_area.sort(key=lambda x: x[1], reverse=True)

    merged = []
    for face, _ in faces_with_area:
        should_merge = False
        for i, existing in enumerate(merged):
            if calculate_iou(face, existing) > iou_threshold:
                if face[2] * face[3] > existing[2] * existing[3]:
                    merged[i] = face
                should_merge = True
                break
        if not should_merge:
            merged.append(face)
    return merged

# ====== 既存: 修正系（ピクセル化/ブラー/黒塗り）=============================
def _pixelate_roi_advanced(img: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 12):
    """改良されたピクセル化（エッジを保持）"""
    roi = img[y:y+h, x:x+w].copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)

    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    edge_blocks = max(blocks // 2, 6)
    small_edge = cv2.resize(roi, (max(1, w // edge_blocks), max(1, h // edge_blocks)), interpolation=cv2.INTER_LINEAR)
    edge_pixelated = cv2.resize(small_edge, (w, h), interpolation=cv2.INTER_NEAREST)

    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255.0
    final = pixelated * (1 - edges_3ch) + edge_pixelated * edges_3ch
    img[y:y+h, x:x+w] = final.astype(np.uint8)

def _smart_blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, strength: int = 25):
    """スマートブラー（顔の特徴に応じて強度を調整）"""
    roi = img[y:y+h, x:x+w].copy()
    blur1 = cv2.GaussianBlur(roi, (strength//2 | 1, strength//2 | 1), 0)
    blur2 = cv2.GaussianBlur(roi, (strength | 1, strength | 1), 0)
    blur3 = cv2.GaussianBlur(roi, (strength*2 | 1, strength*2 | 1), 0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    h_roi, w_roi = roi.shape[:2]
    center_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.ellipse(center_mask, (w_roi//2, h_roi//2), (w_roi//3, h_roi//3), 0, 0, 360, 255, -1)

    result = roi.copy()
    result = np.where(center_mask[..., None] > 128, blur3, result)
    result = np.where((edges > 50)[..., None], blur2, result)
    result = np.where((edges <= 50)[..., None], blur1, result)
    img[y:y+h, x:x+w] = result

def _blur_roi_advanced(img: np.ndarray, x: int, y: int, w: int, h: int, k: int = 51):
    """改良されたブラー処理"""
    k = int(k) | 1
    roi = img[y:y+h, x:x+w]
    blurred1 = cv2.GaussianBlur(roi, (k, k), 0)
    blurred2 = cv2.medianBlur(roi, min(k, 255))
    img[y:y+h, x:x+w] = cv2.addWeighted(blurred1, 0.7, blurred2, 0.3, 0)

def _box_roi_rounded(img: np.ndarray, x: int, y: int, w: int, h: int,
                     color: Tuple[int,int,int] = (32, 32, 32), corner_radius: int = 10):
    """角丸の黒塗り"""
    overlay = img[y:y+h, x:x+w].copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (corner_radius, 0), (w - corner_radius, h), 255, -1)
    cv2.rectangle(mask, (0, corner_radius), (w, h - corner_radius), 255, -1)
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (w - corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (corner_radius, h - corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (w - corner_radius, h - corner_radius), corner_radius, 255, -1)
    overlay[mask > 0] = color
    img[y:y+h, x:x+w] = overlay

# ====== 追加: 顔置換（Nano-banana/Gemini） ================================

# Poissonブレンディングを使うか（失敗時は自動でフェザーにフォールバック）
_USE_POISSON = True

def _ellipse_mask(w: int, h: int, feather_ratio: float = 0.08) -> np.ndarray:
    """楕円のフェザーマスク（0-255）"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2, h//2), (int(w*0.48), int(h*0.48)), 0, 0, 360, 255, -1)
    sigma = max(3, int(min(w, h) * feather_ratio))
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return mask

def _feather_blend(dst_roi_bgr: np.ndarray, src_roi_bgr: np.ndarray) -> np.ndarray:
    """楕円フェザー合成"""
    h, w = dst_roi_bgr.shape[:2]
    src = src_roi_bgr.astype(np.float32)
    dst = dst_roi_bgr.astype(np.float32)
    mask = _ellipse_mask(w, h).astype(np.float32) / 255.0
    m3 = cv2.merge([mask, mask, mask])
    out = (src * m3 + dst * (1 - m3)).astype(np.uint8)
    return out

def _poisson_clone(dst_roi_bgr: np.ndarray, src_roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Poissonブレンディング（cv2.seamlessClone）。失敗なら None を返す。"""
    try:
        h, w = dst_roi_bgr.shape[:2]
        src = src_roi_bgr
        dst = dst_roi_bgr
        mask = _ellipse_mask(w, h)
        center = (w//2, h//2)
        blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        return blended
    except Exception as e:
        print(f"seamlessClone failed: {e}")
        return None

_genai_client_cache: Optional[genai.Client] = None  # type: ignore

def _get_genai_client(explicit_api_key: Optional[str] = None) -> Optional["genai.Client"]:
    """
    google-genai のクライアントを返す。
    - explicit_api_key があればそれを優先
    - なければ環境変数 GOOGLE_API_KEY を利用（ライブラリ側で自動）
    - ライブラリ未導入の場合は None
    """
    global _genai_client_cache
    if not _GENAI_AVAILABLE:
        return None
    if _genai_client_cache is None:
        try:
            _genai_client_cache = genai.Client(api_key=explicit_api_key) if explicit_api_key else genai.Client()
        except Exception as e:
            print(f"genai.Client init failed: {e}")
            return None
    return _genai_client_cache

def _prepare_face_prompt(persona: Optional[str] = None) -> str:
    """顔置換プロンプトを生成"""
    persona_hint = f" Target look: {persona}." if persona else ""
    return (
        "Replace ONLY the person's face with a completely different person. "
        "Keep lighting, skin tone, head pose, hair edges, and background consistent."
        + persona_hint
    )

def _resize_to(w: int, h: int, pil_img: Image.Image) -> Image.Image:
    """ROIに自然に合うよう高品質リサイズ"""
    return pil_img.resize((w, h), Image.LANCZOS)

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _pixelate_roi_strict(img: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 24):
    """完全モザイク（輪郭を残さない）"""
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pix

def _gemini_edit_face(roi_bgr: np.ndarray,
                      api_key: Optional[str],
                      persona: Optional[str] = None) -> Optional[np.ndarray]:
    """
    ROI（顔部分）を Nano-banana/Gemini で別人の顔に編集し、BGRで返す。
    失敗時は None を返す（候補ゼロや異常応答も安全に処理）。
    """
    client = _get_genai_client(api_key)
    if client is None:
        return None
    try:
        pil_roi = _bgr_to_pil(roi_bgr)
        prompt = _prepare_face_prompt(persona)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt, pil_roi],
        )
        new_img = None
        if getattr(resp, "candidates", None):
            for cand in resp.candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts:
                    for part in parts:
                        inline = getattr(part, "inline_data", None)
                        data = getattr(inline, "data", None) if inline is not None else None
                        if data:
                            new_img = Image.open(io.BytesIO(data))
                            break
                if new_img is not None:
                    break
        if new_img is None:
            return None
        new_img = _resize_to(roi_bgr.shape[1], roi_bgr.shape[0], new_img)
        return _pil_to_bgr(new_img)
    except Exception as e:
        print(f"Gemini edit failed: {e}")
        return None

# ====== メイン：顔修正（既存 + 置換を追加）===============================
async def redact_faces_image_bytes_enhanced(
    file_bytes: bytes,
    method: Method = "pixelate",
    strength: int = 16,
    expand: float = 0.15,
    out_format: str = "PNG",
    jpeg_quality: int = 90,
    use_multiple_detectors: bool = True,
    nanobanan_api_key: Optional[str] = None,
    nanobanan_endpoint: Optional[str] = None,
    confidence_threshold: float = 0.3,
    # 追加: 置換用ペルソナ（任意、後方互換のため末尾に配置）
    persona: Optional[str] = None,
) -> bytes:
    """
    AI強化版顔認識・修正機能

    Args:
        file_bytes: 入力画像のバイト
        method: 修正方法（pixelate/blur/box/smart_blur/replace_face）
        strength: 修正強度
        expand: 顔矩形の拡張率
        use_multiple_detectors: 複数の検出器を使用するか
        nanobanan_api_key: Nanobanana/Gemini の API キー
        nanobanan_endpoint: Nanobanana のエンドポイント（検出時に使用）
        confidence_threshold: DNN 検出の信頼度閾値
        persona: 「20代男性の笑顔」など任意の外見ヒント
    """
    # EXIFの回転を正しく適用
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    all_faces: List[Tuple[int,int,int,int]] = []

    # 検出器を合成
    if use_multiple_detectors:
        haar_faces = _detect_faces_haar(bgr, scale_factor=1.1, min_neighbors=3)
        all_faces.extend(haar_faces)
        try:
            dnn_faces = _detect_faces_dnn(bgr, confidence_threshold=confidence_threshold)
            all_faces.extend(dnn_faces)
        except Exception as e:
            print(f"DNN detection skipped: {e}")
        if nanobanan_api_key and nanobanan_endpoint:
            try:
                api_faces = await _detect_faces_nanobanan(bgr, nanobanan_api_key, nanobanan_endpoint)
                all_faces.extend(api_faces)
            except Exception as e:
                print(f"Nanobanan detection failed: {e}")
    else:
        all_faces = _detect_faces_haar(bgr)

    faces = _merge_overlapping_faces(all_faces, iou_threshold=0.3)

    if not faces:
        out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bio = io.BytesIO()
        if out_format.upper() in ("JPEG", "JPG"):
            out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
        else:
            out.save(bio, format="PNG")
        return bio.getvalue()

    # 各顔に対して処理
    for (x, y, w, h) in faces:
        padw, padh = int(w * expand), int(h * expand)
        x0 = max(0, x - padw)
        y0 = max(0, y - padh)
        w0 = min(W - x0, w + 2 * padw)
        h0 = min(H - y0, h + 2 * padh)

        if method == "pixelate":
            _pixelate_roi_advanced(bgr, x0, y0, w0, h0, blocks=max(6, int(strength)))
        # Haar輪郭残るので輪郭ごと丸ごと消してpixalate
        elif method == "pixelate_strict":
            _pixelate_roi_strict(bgr, x0, y0, w0, h0, blocks=max(16, int(strength)))
        elif method == "blur":
            _blur_roi_advanced(bgr, x0, y0, w0, h0, k=max(7, int(strength)))
        elif method == "smart_blur":
            _smart_blur_roi(bgr, x0, y0, w0, h0, strength=max(7, int(strength)))
        elif method == "replace_face":
            # === 顔置換（Gemini） ===
            roi = bgr[y0:y0+h0, x0:x0+w0].copy()
            edited = _gemini_edit_face(roi, api_key=nanobanan_api_key, persona=persona)

            if edited is None:
                # 置換失敗 → フォールバック（自然さ重視のスマートブラー）
                _smart_blur_roi(bgr, x0, y0, w0, h0, strength=max(9, int(strength)))
            else:
                edited = cv2.resize(edited, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
                blended = None
                if _USE_POISSON:
                    blended = _poisson_clone(roi, edited)
                if blended is None:
                    blended = _feather_blend(roi, edited)
                bgr[y0:y0+h0, x0:x0+w0] = blended
        else:
            _box_roi_rounded(bgr, x0, y0, w0, h0, corner_radius=min(10, w0//10))

    # 出力
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    print(f"[enhanced_face_redactor] faces={len(faces)} method={method} detectors={'multi' if use_multiple_detectors else 'haar'}")
    if out_format.upper() in ("JPEG", "JPG"):
        out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        out.save(bio, format="PNG")
    return bio.getvalue()

# ====== 既存: 同期ラッパ（後方互換）========================================
def redact_faces_image_bytes(
    file_bytes: bytes,
    method: Method = "pixelate",
    strength: int = 16,
    expand: float = 0.12,
    out_format: str = "PNG",
    jpeg_quality: int = 90
) -> bytes:
    """既存APIとの互換性のためのラッパー関数（シングルスレッド）"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            redact_faces_image_bytes_enhanced(
                file_bytes, method, strength, expand, out_format, jpeg_quality,
                use_multiple_detectors=False  # デフォルトは既存動作（Haarのみ）
            )
        )
    finally:
        loop.close()
