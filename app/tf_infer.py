from typing import List, Dict, Tuple, Optional
from PIL import Image
import pytesseract, httpx, json


# OCR（最初はTesseract使うつもりだが、次期にTFLiteに変更する予定） 
def ocr_words(img: Image.Image) -> List[Dict]:
    data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"bbox": (x,y,x+w,y+h), "text": txt})
    return out

# PDFデータ抽出 
def pdf_text_boxes(page) -> List[Dict]:
    # page: fitz.Page
    out = []
    for b in page.get_text("blocks"):
        x1,y1,x2,y2, text, *_ = b + (None,)
        if not text: continue
        out.append({"bbox": (int(x1),int(y1),int(x2),int(y2)), "text": text})
    return out

# TF-Servingの予定　endpoint未設定なら空で返す
async def ner_classify_texts(texts: List[str], endpoint: Optional[str]) -> List[Dict]:
    if not endpoint or not texts: return []
    # 例: {"instances": [{"text": "山田太郎"}...]}
    payload = {"instances": [{"text": t} for t in texts]}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(endpoint, json=payload)
        r.raise_for_status()
        preds = r.json().get("predictions", [])
    hits = []
    for t, p in zip(texts, preds):
        # p から {"label":"PERSON","score":0.92} などを受け取り整形
        lbl = p.get("label"); sc = float(p.get("score", 0))
        if lbl == "PERSON" and sc >= 0.85: hits.append({"type":"name","text":t,"conf":sc,"reason":"ner:person"})
        if lbl == "ADDRESS" and sc >= 0.85: hits.append({"type":"address","text":t,"conf":sc,"reason":"ner:address"})
    return hits
#処理のフローはこんな感じ
#画像  ──► ocr_lines(img) ──► lines
#                         └─► _iter_line_hits(lines, policies) ──► (bbox, kind, orig)
#                                                           └─► alias生成 → draw_replace()
#将来のTFLite等の導入を見据えてレイヤーごとに分割（拡張性重視）

def ocr_lines(img) -> List[Dict]:
    """Tesseractのword出力を (block,par,line) ごとにグルーピング"""
    data = pytesseract.image_to_data(img, lang="jpn+eng", output_type=Output.DICT)
    lines = {}  # key -> {"tokens":[{"text","bbox","conf"}], "bbox":(x1,y1,x2,y2)}
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        conf = data.get("conf", ["-1"]*n)[i]
        try: conf = float(conf)
        except: conf = -1.0
        if conf < 60:  # 信頼度６０%未満はもう弾いてしまう(安全性重視、手動で調整機能を加える)
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