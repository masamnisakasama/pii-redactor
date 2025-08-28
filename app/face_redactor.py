
# app/face_redactor.py
# 精度向上版：複数の検出手法を組み合わせ、より高精度な顔認識を実現
import io
from typing import List, Tuple, Literal, Optional
import numpy as np
import cv2
from PIL import Image, ImageOps
import asyncio
import httpx
import base64

Method = Literal["pixelate", "blur", "box", "smart_blur"]

class FaceDetectionError(Exception):
    """顔検出関連のエラー"""
    pass

def _detect_faces_haar(bgr: np.ndarray,
                       scale_factor: float = 1.1,
                       min_neighbors: int = 5,
                       min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int,int,int,int]]:
    """改善されたHaar Cascade検出器"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # 複数の分類器を試行
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
    
    # 重複除去（IoU > 0.3の矩形を統合）
    if not all_faces:
        return []
    
    return _merge_overlapping_faces(all_faces)

def _detect_faces_dnn(bgr: np.ndarray, 
                      confidence_threshold: float = 0.5) -> List[Tuple[int,int,int,int]]:
    """OpenCV DNN（より高精度だが重い）"""
    try:
        # OpenCVのDNNモデル（事前にダウンロードが必要）
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
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
        
    except Exception as e:
        print(f"DNN face detection failed: {e}")
        return []

async def _detect_faces_nanobanan(bgr: np.ndarray,
                                  api_key: str,
                                  endpoint: str) -> List[Tuple[int,int,int,int]]:
    """NanobananaのAPIを使用した高精度顔検出"""
    try:
        # BGRをRGBに変換してPIL Imageに
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Base64エンコード
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
        
        # 重複領域の計算
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # 信頼度で並べ替え（面積の大きいものを優先）
    faces_with_area = [(f, f[2] * f[3]) for f in faces]
    faces_with_area.sort(key=lambda x: x[1], reverse=True)
    
    merged = []
    for face, _ in faces_with_area:
        should_merge = False
        for i, existing in enumerate(merged):
            if calculate_iou(face, existing) > iou_threshold:
                # より大きい矩形で置き換え
                if face[2] * face[3] > existing[2] * existing[3]:
                    merged[i] = face
                should_merge = True
                break
        
        if not should_merge:
            merged.append(face)
    
    return merged

def _pixelate_roi_advanced(img: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 12):
    """改良されたピクセル化（エッジを保持）"""
    roi = img[y:y+h, x:x+w].copy()
    
    # エッジ検出
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # 通常のピクセル化
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # エッジ部分は少し細かく
    edge_blocks = max(blocks // 2, 6)
    small_edge = cv2.resize(roi, (max(1, w // edge_blocks), max(1, h // edge_blocks)), interpolation=cv2.INTER_LINEAR)
    edge_pixelated = cv2.resize(small_edge, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # エッジマスクで合成
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255.0
    final = pixelated * (1 - edges_3ch) + edge_pixelated * edges_3ch
    
    img[y:y+h, x:x+w] = final.astype(np.uint8)

def _smart_blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, strength: int = 25):
    """スマートブラー（顔の特徴に応じて強度を調整）"""
    roi = img[y:y+h, x:x+w].copy()
    
    # 複数段階のブラー
    blur1 = cv2.GaussianBlur(roi, (strength//2|1, strength//2|1), 0)
    blur2 = cv2.GaussianBlur(roi, (strength|1, strength|1), 0)
    blur3 = cv2.GaussianBlur(roi, (strength*2|1, strength*2|1), 0)
    
    # エッジ検出で重要部分を特定
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # 顔の中央部分（目、鼻、口）により強いブラーを適用
    h_roi, w_roi = roi.shape[:2]
    center_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.ellipse(center_mask, (w_roi//2, h_roi//2), (w_roi//3, h_roi//3), 0, 0, 360, 255, -1)
    
    # 3段階のブラーを適用
    result = roi.copy()
    result = np.where(center_mask[..., None] > 128, blur3, result)
    result = np.where((edges > 50)[..., None], blur2, result)
    result = np.where((edges <= 50)[..., None], blur1, result)
    
    img[y:y+h, x:x+w] = result

def _blur_roi_advanced(img: np.ndarray, x: int, y: int, w: int, h: int, k: int = 51):
    """改良されたブラー処理"""
    k = int(k) | 1
    roi = img[y:y+h, x:x+w]
    
    # ガウシアンブラーとモーションブラーの組み合わせ
    blurred1 = cv2.GaussianBlur(roi, (k, k), 0)
    blurred2 = cv2.medianBlur(roi, min(k, 255))
    
    # 2つのブラーを重み付き合成
    img[y:y+h, x:x+w] = cv2.addWeighted(blurred1, 0.7, blurred2, 0.3, 0)

def _box_roi_rounded(img: np.ndarray, x: int, y: int, w: int, h: int, 
                     color: Tuple[int,int,int] = (32, 32, 32), corner_radius: int = 10):
    """角丸の黒塗り"""
    overlay = img[y:y+h, x:x+w].copy()
    
    # 角丸矩形マスクの作成
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 角丸矩形を描画
    cv2.rectangle(mask, (corner_radius, 0), (w-corner_radius, h), 255, -1)
    cv2.rectangle(mask, (0, corner_radius), (w, h-corner_radius), 255, -1)
    
    # 角の円を描画
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (w-corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (corner_radius, h-corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (w-corner_radius, h-corner_radius), corner_radius, 255, -1)
    
    # 指定色で塗りつぶし
    overlay[mask > 0] = color
    img[y:y+h, x:x+w] = overlay

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
    confidence_threshold: float = 0.3
) -> bytes:
    """
    AI強化版顔認識・修正機能
    
    Args:
        file_bytes: 入力画像のバイト
        method: 修正方法
        strength: 修正強度
        expand: 顔矩形の拡張率
        use_multiple_detectors: 複数の検出器を使用するか
        nanobanan_api_key: NanobananaのAPIキー
        nanobanan_endpoint: Nanobananaのエンドポイント
        confidence_threshold: 信頼度の閾値
    """
    # EXIFの回転を正しく適用
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    all_faces = []

    # 複数の検出器を使用
    if use_multiple_detectors:
        # Haar Cascade検出
        haar_faces = _detect_faces_haar(bgr, scale_factor=1.1, min_neighbors=3)
        all_faces.extend(haar_faces)
        
        # DNN検出（利用可能な場合）
        try:
            dnn_faces = _detect_faces_dnn(bgr, confidence_threshold=confidence_threshold)
            all_faces.extend(dnn_faces)
        except Exception as e:
            print(f"DNN detection skipped: {e}")
        
        # Nanobanan API検出（設定されている場合）
        if nanobanan_api_key and nanobanan_endpoint:
            try:
                api_faces = await _detect_faces_nanobanan(bgr, nanobanan_api_key, nanobanan_endpoint)
                all_faces.extend(api_faces)
            except Exception as e:
                print(f"Nanobanan detection failed: {e}")
    else:
        # 基本のHaar Cascadeのみ使用
        all_faces = _detect_faces_haar(bgr)

    # 重複除去
    faces = _merge_overlapping_faces(all_faces, iou_threshold=0.3)
    
    if not faces:
        # 顔が無い場合もそのまま返す
        out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bio = io.BytesIO()
        if out_format.upper() in ("JPEG", "JPG"):
            out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
        else:
            out.save(bio, format="PNG")
        return bio.getvalue()

    # 各顔にモザイク/ぼかし/黒塗りを適用
    for (x, y, w, h) in faces:
        # 拡張された矩形を計算
        padw, padh = int(w * expand), int(h * expand)
        x0 = max(0, x - padw)
        y0 = max(0, y - padh)
        w0 = min(W - x0, w + 2 * padw)
        h0 = min(H - y0, h + 2 * padh)

        # 修正方法に応じて処理
        if method == "pixelate":
            _pixelate_roi_advanced(bgr, x0, y0, w0, h0, blocks=max(6, int(strength)))
        elif method == "blur":
            _blur_roi_advanced(bgr, x0, y0, w0, h0, k=max(7, int(strength)))
        elif method == "smart_blur":
            _smart_blur_roi(bgr, x0, y0, w0, h0, strength=max(7, int(strength)))
        else:  # box
            _box_roi_rounded(bgr, x0, y0, w0, h0, corner_radius=min(10, w0//10))

    # 結果をPIL Imageに変換
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    
    print(f"[enhanced_face_redactor] faces={len(faces)} method={method} detectors={'multi' if use_multiple_detectors else 'haar'}")
    
    if out_format.upper() in ("JPEG", "JPG"):
        out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        out.save(bio, format="PNG")
    
    return bio.getvalue()

# 既存の関数との互換性維持
def redact_faces_image_bytes(
    file_bytes: bytes,
    method: Method = "pixelate",
    strength: int = 16,
    expand: float = 0.12,
    out_format: str = "PNG", 
    jpeg_quality: int = 90   
) -> bytes:
    """既存APIとの互換性のためのラッパー関数"""
    # 非同期関数を同期的に実行
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            redact_faces_image_bytes_enhanced(
                file_bytes, method, strength, expand, out_format, jpeg_quality,
                use_multiple_detectors=False  # デフォルトは既存動作
            )
        )
    finally:
        loop.close()
"""
# app/face_redactor.py
# API使うならクラウドで、各々がダウンロードして使えるようにするなら事実上APIは使えない　
# 反省を生かしコストと処理能力を見積もって決定するつもり
import io
from typing import List, Tuple, Literal
import numpy as np
import cv2
from PIL import Image, ImageOps

Method = Literal["pixelate", "blur", "box"]

# Haarで顔矩形 (x,y,w,h) を返す（CPUで軽量）。
def _detect_faces_haar(bgr: np.ndarray,
                       scale_factor: float = 1.2,
                       min_neighbors: int = 5) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cc.empty():
        raise RuntimeError("Haar分類器が読み込めませんでした")
    faces = cc.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def _pixelate_roi(img: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 12):
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    img[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _blur_roi(img: np.ndarray, x: int, y: int, w: int, h: int, k: int = 51):
    k = int(k) | 1
    img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (k, k), 0)

def _box_roi(img: np.ndarray, x: int, y: int, w: int, h: int):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

def redact_faces_image_bytes(file_bytes: bytes,
                             method: Method = "pixelate",
                             strength: int = 16,
                             expand: float = 0.12,
                             out_format: str = "PNG", 
                             jpeg_quality: int = 90   
                             ) -> bytes:
    
    # 画像バイトを受け取り、顔をモザイク/ぼかし/黒塗りしてPNGバイトを返す。
    # method: "pixelate" | "blur" | "box"
    # expand: 顔矩形を上下左右に拡張（見切れ防止のため）
    
    # EXIFの回転を正しく適用
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    faces = _detect_faces_haar(bgr)
    if not faces:
        # 顔が無い場合もPNGで返す
        out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        bio = io.BytesIO(); out.save(bio, format="PNG"); return bio.getvalue()

    for (x, y, w, h) in faces:
        padw, padh = int(w * expand), int(h * expand)
        x0 = max(0, x - padw)
        y0 = max(0, y - padh)
        w0 = min(W - x0, w + 2 * padw)
        h0 = min(H - y0, h + 2 * padh)

        if method == "pixelate":
            _pixelate_roi(bgr, x0, y0, w0, h0, blocks=max(6, int(strength)))
        elif method == "blur":
            _blur_roi(bgr, x0, y0, w0, h0, k=max(7, int(strength)))
        else:
            _box_roi(bgr, x0, y0, w0, h0)
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    print(f"[face_redactor] faces={len(faces)} method={method}")
    if out_format.upper() in ("JPEG", "JPG"):
        out.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        out.save(bio, format="PNG")
        return bio.getvalue()
"""