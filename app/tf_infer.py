
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pytesseract, httpx, json, base64, io
from pytesseract import Output


# OCR（nanobananaのOCR APIの方が良さそう　問題はセキュリティ）
async def ocr_words_nanobanan(img: Image.Image, api_key: str, endpoint: str) -> List[Dict]:
    """NanobananaのOCR APIを使用してワード単位で文字認識"""
    # 画像をbase64エンコード
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    payload = {
        "image": img_b64,
        "languages": ["ja", "en"],
        "output_format": "word_level"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(f"{endpoint}/v1/ocr", json=payload, headers=headers)
            r.raise_for_status()
            result = r.json()
            
            out = []
            for item in result.get("words", []):
                bbox = item.get("bbox", [0,0,0,0])  # [x1,y1,x2,y2] format expected
                text = item.get("text", "").strip()
                confidence = float(item.get("confidence", 0.0))
                
                if text and confidence > 0.6:  # 60%以上の信頼度
                    out.append({
                        "bbox": tuple(bbox), 
                        "text": text,
                        "confidence": confidence
                    })
            return out
        except Exception as e:
            print(f"Nanobanan OCR API error: {e}")
            # フォールバック：Tesseractを使用
            return ocr_words_tesseract(img)

def ocr_words_tesseract(img: Image.Image) -> List[Dict]:
    """Tesseractによるフォールバック処理"""
    data = pytesseract.image_to_data(img, lang="jpn+eng", output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        conf = data.get("conf", ["-1"]*n)[i]
        try: 
            conf = float(conf) / 100.0  # 0-1の範囲に正規化
        except: 
            conf = -1.0
        if conf < 0.6:  # 60%未満は除外
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"bbox": (x,y,x+w,y+h), "text": txt, "confidence": conf})
    return out

# 高精度NER（固有表現認識）
async def ner_classify_nanobanan(texts: List[str], api_key: str, endpoint: str) -> List[Dict]:
    """NanobananaのNER APIで高精度な固有表現認識"""
    if not texts:
        return []
    
    payload = {
        "texts": texts,
        "entities": ["PERSON", "ADDRESS", "ORGANIZATION", "PHONE", "EMAIL", "ID_NUMBER"],
        "language": "ja"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(f"{endpoint}/v1/ner", json=payload, headers=headers)
            r.raise_for_status()
            result = r.json()
            
            hits = []
            for prediction in result.get("predictions", []):
                text = prediction.get("text", "")
                entity_type = prediction.get("entity", "")
                confidence = float(prediction.get("confidence", 0.0))
                
                # エンティティタイプをアプリケーション用にマッピング
                type_mapping = {
                    "PERSON": "name",
                    "ADDRESS": "address", 
                    "ORGANIZATION": "org",
                    "PHONE": "phone",
                    "EMAIL": "email",
                    "ID_NUMBER": "id"
                }
                
                app_type = type_mapping.get(entity_type)
                if app_type and confidence >= 0.85:  # 高信頼度のもののみ
                    hits.append({
                        "type": app_type,
                        "text": text,
                        "conf": confidence,
                        "reason": f"ner:nanobanan:{entity_type.lower()}"
                    })
            
            return hits
            
        except Exception as e:
            print(f"Nanobanan NER API error: {e}")
            return []  # エラー時は空リストを返す

# PDFデータ抽出（既存のまま）
def pdf_text_boxes(page) -> List[Dict]:
    """fitz.Pageからテキストボックスを抽出"""
    out = []
    for b in page.get_text("blocks"):
        x1,y1,x2,y2, text, *_ = b + (None,)
        if not text: continue
        out.append({"bbox": (int(x1),int(y1),int(x2),int(y2)), "text": text})
    return out

# 改善されたライン認識
async def ocr_lines_enhanced(img: Image.Image, use_nanobanan: bool = True, 
                            api_key: str = None, endpoint: str = None) -> List[Dict]:
    """
    改善されたライン認識：nanobananaとTesseractのハイブリッド
    """
    if use_nanobanan and api_key and endpoint:
        words = await ocr_words_nanobanan(img, api_key, endpoint)
    else:
        words = ocr_words_tesseract(img)
    
    # ワードを行にグルーピング（Y座標ベース + 重複除去）
    lines = {}
    for word in words:
        x1, y1, x2, y2 = word["bbox"]
        center_y = (y1 + y2) / 2
        
        # 近い行を探す（±5ピクセル）
        line_key = None
        for existing_y in lines.keys():
            if abs(center_y - existing_y) <= 5:
                line_key = existing_y
                break
        
        if line_key is None:
            line_key = center_y
            lines[line_key] = {
                "tokens": [word],
                "bbox": word["bbox"]
            }
        else:
            lines[line_key]["tokens"].append(word)
            # バウンディングボックス更新
            ex1, ey1, ex2, ey2 = lines[line_key]["bbox"]
            lines[line_key]["bbox"] = (
                min(ex1, x1), min(ey1, y1), 
                max(ex2, x2), max(ey2, y2)
            )
    
    # 各行のトークンを左から右へソート
    result = []
    for line_data in lines.values():
        line_data["tokens"].sort(key=lambda t: t["bbox"][0])
        result.append(line_data)
    
    # 行を上から下へソート
    result.sort(key=lambda line: line["bbox"][1])
    return result

# 旧関数との互換性維持
def ocr_lines(img) -> List[Dict]:
    """既存コードとの互換性のためのラッパー"""
    return ocr_lines_tesseract_original(img)

def ocr_lines_tesseract_original(img) -> List[Dict]:
    """元のTesseractベース処理（フォールバック用）"""
    data = pytesseract.image_to_data(img, lang="jpn+eng", output_type=Output.DICT)
    lines = {}  # key -> {"tokens":[{"text","bbox","conf"}], "bbox":(x1,y1,x2,y2)}
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        conf = data.get("conf", ["-1"]*n)[i]
        try: 
            conf = float(conf)
        except: 
            conf = -1.0
        if conf < 60:  # 信頼度60%未満は除外
            continue
        x,y,w,h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        entry = {"text": txt, "bbox": (x,y,x+w,y+h), "conf": conf, "left": x}
        if key not in lines:
            lines[key] = {"tokens":[entry], "bbox": (x,y,x+w,y+h)}
        else:
            lines[key]["tokens"].append(entry)
            x1,y1,x2,y2 = lines[key]["bbox"]
            nx1,ny1,nx2,ny2 = x,y,x+w,y+h
            lines[key]["bbox"] = (min(x1,nx1), min(y1,ny1), max(x2,nx2), max(y2,ny2))
    
    # 左から順に並べる
    out = []
    for k,v in lines.items():
        v["tokens"].sort(key=lambda t: t["left"])
        out.append(v)
    return out
