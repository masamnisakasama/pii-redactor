# detectors.py
import re
from typing import List, Dict
import cv2
import numpy as np
from functools import lru_cache
try:
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# 部分だけdetectされないよう行単位での表示を導入
LINE_PATTERNS = [
    ("amount_full", re.compile(r"(?:JPY|¥)\s?\d{1,3}(?:,\d{3})+")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?:\+81-\d{1,4}-\d{1,4}-\d{3,4})|(?:0\d{1,4}-\d{1,4}-\d{3,4})")),
    ("id",    re.compile(r"\b(?:ACC|USR|ORD)-\d{4,6}\b")),
]

RE_EMAIL  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE  = re.compile(r"(?:0\d{1,4}-\d{1,4}-\d{3,4})|(?:\+81-\d{1,4}-\d{1,4}-\d{3,4})")
RE_AMOUNT = re.compile(r"(?:¥|JPY)?\s?\d{1,3}(?:,\d{3})+")
RE_ID     = re.compile(r"\b(?:ACC|USR|ORD)-\d{4,6}\b")

# 名前と住所以外は正規表現が活きる
def classify_by_regex(text: str) -> List[Dict]:
    out: List[Dict] = []
    for m in RE_EMAIL.finditer(text):  out.append({"type":"email","text":m.group(),"conf":0.99,"reason":"regex:email"})
    for m in RE_PHONE.finditer(text):  out.append({"type":"phone","text":m.group(),"conf":0.95,"reason":"regex:phone"})
    for m in RE_AMOUNT.finditer(text): out.append({"type":"amount","text":m.group(),"conf":0.90,"reason":"regex:amount"})
    for m in RE_ID.finditer(text):     out.append({"type":"id","text":m.group(),"conf":0.88,"reason":"regex:id"})
    return out

def merge_with_ner(regex_hits: List[Dict], ner_hits: List[Dict]) -> List[Dict]:
    # NERから {text,type='name'|'address'|'org',conf} などを受け取り統合
    return regex_hits + ner_hits


def _post_filter(rects, img_w, img_h, min_px=56, max_ratio=0.7, nms_thresh=0.30):
    """小さすぎ・デカすぎ・縦横比が変な箱を捨て、NMSで重複排除"""
    filtered = []
    for (x, y, w, h) in rects:
        if w < min_px or h < min_px:
            continue
        if w > img_w * max_ratio or h > img_h * max_ratio:
            continue
        ar = w / float(h)
        if ar < 0.6 or ar > 1.6:
            continue
        filtered.append([x, y, w, h])
    if not filtered:
        return []

    scores = [1.0] * len(filtered)
    indices = cv2.dnn.NMSBoxes(filtered, scores, score_threshold=0.0, nms_threshold=nms_thresh)
    keep = indices.flatten().tolist() if len(indices) else []
    return [tuple(filtered[i]) for i in keep]


# --- HF NER 本体 -----------------------------------------------------------

@lru_cache(maxsize=4)
def _get_hf_ner(model_name: str, aggregation: str):
    """HF の NER パイプライン（CPU）をキャッシュして返す。"""
    if not _HF_AVAILABLE:
        return None
    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy=aggregation,  # 'simple'|'average'|'max'
        device=-1,  # CPU 固定（安全）
    )

# HFラベル→PII種別の簡易マップ（必要に応じて拡張）
_HF2PII = {
    "PER": "name",
    "ORG": "org",
    "LOC": "address",
    "MISC": "misc",
}

def hf_ner(text: str, model_name: str, aggregation: str) -> List[Dict]:
    """HF NER を {type,text,conf} に正規化して返す。未導入なら空配列。"""
    pipe = _get_hf_ner(model_name, aggregation)
    if pipe is None:
        return []
    out: List[Dict] = []
    try:
        for ent in pipe(text):
            label = (ent.get("entity_group") or ent.get("entity") or "").upper()
            pii_type = _HF2PII.get(label, "misc")
            word = ent.get("word") or ent.get("text") or ""
            out.append({"type": pii_type, "text": word, "conf": float(ent.get("score", 0.0))})
    except Exception as e:
        print(f"HF NER failed: {e}")
    return out

def detect_text_pii(
    text: str,
    ner_engine: str = "regex",
    model_name: str = "dslim/bert-base-NER",
    aggregation: str = "simple",
) -> List[Dict]:
    """
    テキストPII検出の統合入口。regex / hf / hybrid を切替。
    既存の classify_by_regex(...) はそのまま活かす。
    """
    regex_hits = classify_by_regex(text)
    if ner_engine == "regex":
        return regex_hits
    ner_hits = hf_ner(text, model_name, aggregation) if ner_engine in ("hf", "hybrid") else []
    if ner_engine == "hf":
        return ner_hits
    # hybrid
    return regex_hits + ner_hits

# detectors.py 末尾に（再掲）
def detect_text_regions_east(image_bgr):
    if os.getenv("USE_TEXT_EAST", "false").lower() != "true":
        return []
    try:
        from .text_detect import detect_text_boxes_east
        return detect_text_boxes_east(image_bgr)
    except Exception as e:
        print(f"EAST disabled: {e}")
        return []
