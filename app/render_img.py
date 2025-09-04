# app/render_img.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, List
from typing import Optional # 型ヒント導入で使いやすく
import os
import cv2
import pytesseract
from .text_detect import detect_text_boxes_east
from .detectors import detect_text_pii
from .face_redactor import _box_roi_rounded  # 角丸ボックスを再利用


def _auto_pad(bbox, base: int = 8) -> int:
    x1, y1, x2, y2 = map(int, bbox)
    h = max(1, y2 - y1)
    # 高さに応じて余白を増やす（行が薄いときに有効）
    return max(base, int(h * 0.35))

# Optionalが必要な場所1
def draw_replace(img, bbox, new_text, mode="readable", pad: Optional[int]=None, font_path=None):
    """
    bbox 部分を置換して描画。
    mode: "readable"（白塗り+テキスト）, "box"（黒塗り）, "pixelate", "blur"
    pad : None のとき高さに応じて自動決定
    """
    x1, y1, x2, y2 = map(int, bbox)
    if pad is None:
        pad = _auto_pad(bbox)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width,  x2 + pad)
    y2 = min(img.height, y2 + pad)

    if mode == "pixelate":
        region = img.crop((x1, y1, x2, y2)).resize((16, 16), Image.Resampling.NEAREST)
        img.paste(region.resize((x2-x1, y2-y1), Image.Resampling.NEAREST), (x1, y1))
        return

    if mode == "blur":
        region = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(region, (x1, y1))
        return

    draw = ImageDraw.Draw(img)

    if mode == "box":
        draw.rectangle([x1, y1, x2, y2], fill="black")
        return

    # readable: 白塗りしてテキスト
    draw.rectangle([x1, y1, x2, y2], fill="white")

    # フォント決定（最初は枠高の 0.9、そこからフィットまで縮小）
    target_h = max(16, int((y2 - y1) * 0.9))
    try:
        font = ImageFont.truetype(font_path or "/System/Library/Fonts/Hiragino Sans W3.ttc", target_h)
    except Exception:
        font = ImageFont.load_default()

     # textlength は複数行を測れずエラーになったため、改行はスペースに正規化
    text = str(new_text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # 幅フィットのバイナリサーチ的縮小（最大20回）

    for _ in range(20):
        w = draw.textlength(text, font=font)
        if w <= (x2 - x1) - 6:
            break
        target_h = max(12, int(target_h * 0.9))
        try:
            font = ImageFont.truetype(font_path or "/System/Library/Fonts/Hiragino Sans W3.ttc", target_h)
        except Exception:
            font = ImageFont.load_default()

    # 垂直中央寄せ
    bbox_f = font.getbbox(text)
    th = bbox_f[3] - bbox_f[1]
    ty = y1 + max(0, ((y2 - y1) - th) // 2)
    draw.text((x1 + 3, ty), text, fill=(0, 0, 0), font=font)

# Optionalが必要な場所2
# 狭すぎ/薄すぎ/測長不可は行bboxにFB
def subbbox_from_match(line_text: str, match_text: str, line_bbox, font_path=None, pad: Optional[int]=None):
    """
    行テキスト中の match_text の描画位置を line_bbox 比で推定。
    推定が不安定（非常に狭い/見つからない）なら line_bbox 全体を返す。
    """
    x1, y1, x2, y2 = map(int, line_bbox)
    w = max(1, x2 - x1)

    try:
        font = ImageFont.truetype(font_path or "/System/Library/Fonts/Hiragino Sans W3.ttc", max(14, (y2 - y1) - 2))
    except Exception:
        font = ImageFont.load_default()

    
    # 正規化：改行→スペース、さらに空白連結を1つに圧縮（textlength対策）
    lt = " ".join(((line_text or "")
                   .replace("\r\n", " ").replace("\n", " ").replace("\r", " ")).split())
    mt = " ".join(((match_text or "")
                   .replace("\r\n", " ").replace("\n", " ").replace("\r", " ")).split())
 


    start = lt.find(mt)
    if start < 0:
        # 見つからない → 行全体
        return (x1, y1, x2, y2)

    # 計測
    dummy = Image.new("RGB", (10, 10), "white")
    d = ImageDraw.Draw(dummy)
    left_px_total = d.textlength(lt[:start], font=font)
    match_px      = max(1.0, d.textlength(mt, font=font))
    total_px      = max(1.0, d.textlength(lt, font=font))

    rel_left = min(1.0, left_px_total / total_px)
    rel_w    = min(1.0, match_px / total_px)

    sx1 = int(x1 + w * rel_left)
    sx2 = int(x1 + w * (rel_left + rel_w))

    # 狭すぎる/高さが薄すぎる場合は行全体へフォールバック
    too_narrow = (sx2 - sx1) < max(12, int((x2 - x1) * 0.05))
    too_thin   = (y2 - y1) < 12
    if too_narrow or too_thin:
        return (x1, y1, x2, y2)

    if pad is None:
        pad = _auto_pad(line_bbox, base=6)
    sy1 = max(0, y1 - pad)
    sy2 = y2 + pad
    return (sx1 - pad, sy1, sx2 + pad, sy2)

def redact_text_with_east(
    bgr, *,
    east_pb: str | None = None,
    conf: float = None,
    nms: float = None,
    max_side: int = None,
    min_size: int = None,
    ocr_lang: str | None = None,
    ner_engine: str | None = None,
    hf_model: str | None = None,
) -> Tuple[any, List[Tuple[int,int,int,int]]]:
    """
    EAST→OCR→PII→塗り の最小実装。bgrを直接編集して返す。
    戻り値: (bgr, マスクしたbox一覧)
    """
    # 環境変数で上書きできるように（なければデフォルト）
    east_pb   = east_pb   or os.getenv("EAST_PB", "models/text/frozen_east_text_detection.pb")
    conf      = conf      if conf      is not None else float(os.getenv("EAST_CONF", "0.50"))
    nms       = nms       if nms       is not None else float(os.getenv("EAST_NMS",  "0.40"))
    max_side  = max_side  if max_side  is not None else int(os.getenv("EAST_MAX_SIDE", "1280"))
    min_size  = min_size  if min_size  is not None else int(os.getenv("EAST_MIN_SIZE", "8"))
    ocr_lang  = ocr_lang  or os.getenv("OCR_LANG", "eng+jpn")
    ner_engine= ner_engine or os.getenv("TEXT_NER", "regex")  # 'regex'|'hf'|'hybrid'
    hf_model  = hf_model  or os.getenv("HF_NER_MODEL", "dslim/bert-base-NER")

    boxes = detect_text_boxes_east(
        bgr,
        model_path=east_pb,
        conf_thr=conf,
        nms_thr=nms,
        max_side=max_side,
        min_size=min_size,
    )
    if not boxes:
        return bgr, []

    masked: List[Tuple[int,int,int,int]] = []
    for (x,y,w,h) in boxes:
        roi = bgr[y:y+h, x:x+w]

        # 軽めの二値化でOCRを安定（必要に応じて調整）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(thr, lang=ocr_lang, config="--psm 6")
        text = (text or "").strip()
        if not text:
            continue

        # 既存のPII判定をそのまま使う
        hits = detect_text_pii(text, ner_engine=ner_engine, model_name=hf_model, aggregation="simple")
        if not hits:
            continue

        # 丸角黒塗り（既存と同じ見た目）
        _box_roi_rounded(bgr, x, y, w, h, corner_radius=max(6, min(w,h)//10))
        masked.append((x,y,w,h))

    return bgr, masked