import os, numpy as np
from .face import detect_faces_bgr
from .ocr_regex import ocr_tokens_bgr, match_pii
from .paint import apply_masks

def redact_image_bgr(img_bgr: np.ndarray, policy_csv: str, style: str|None=None):
    policy = set([p.strip() for p in (policy_csv or "").split(",") if p.strip()])
    # 顔ポリシー
    boxes = []
    if "face" in policy:
        boxes += detect_faces_bgr(img_bgr)
    # テキストPII
    toks = ocr_tokens_bgr(img_bgr)
    boxes += match_pii(toks, policy)
    out = apply_masks(img_bgr.copy(), boxes, style=style)
    return out, boxes
