# app/render_img.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple
from typing import Optional # 型ヒント導入で使いやすく
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

    text = str(new_text)
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

    # 正規化（前後スペース詰め）
    lt = (line_text or "").strip()
    mt = (match_text or "").strip()

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
