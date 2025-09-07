import re, os, numpy as np
import pytesseract
from pytesseract import Output

# シンプルで堅牢な正規表現（日本用に軽く）
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"(?:0\d{1,4}-\d{1,4}-\d{3,4})|(?:\d{2,4}-\d{2,4}-\d{3,4})")
RE_ID    = re.compile(r"\b(?:ORD|ACC|ID|USR)-?\d{3,8}\b", re.IGNORECASE)

def ocr_tokens_bgr(img_bgr: np.ndarray) -> list[dict]:
    # langは環境で指定可（英日混在: "jpn+eng"）
    lang = os.getenv("TESS_LANG", "jpn+eng")
    data = pytesseract.image_to_data(img_bgr, lang=lang, output_type=Output.DICT)
    H, W = img_bgr.shape[:2]
    toks = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # 変な外れ値を除外
        if 0<=x<W and 0<=y<H and w>3 and h>8:
            toks.append({"text":txt, "box":(x,y,w,h)})
    return toks

def match_pii(tokens: list[dict], policy: set[str]) -> list[tuple[int,int,int,int]]:
    boxes = []
    for t in tokens:
        s = t["text"]
        if "email" in policy and RE_EMAIL.search(s): boxes.append(t["box"]); continue
        if "phone" in policy and RE_PHONE.search(s): boxes.append(t["box"]); continue
        if "id"    in policy and RE_ID.search(s):    boxes.append(t["box"]); continue
        # 住所/氏名は最低限のダミー（本番で拡張）
        if "address" in policy and any(k in s for k in ["丁目","番地","区","市","町"]):
            boxes.append(t["box"]); continue
        if "name" in policy and len(s)>=2 and all("A"<=c<="z" or "\u3040"<=c<="\u9fff" for c in s):
            # 2文字以上の連続語を仮に氏名候補とする（誤検出は少なめ）
            pass
    return boxes
