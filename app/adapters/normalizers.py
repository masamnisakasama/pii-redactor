import re, unicodedata
_DASH_MAP = dict.fromkeys(map(ord, "-‐–—−ー―"), ord('-'))
_WS_MAP   = dict.fromkeys(map(ord, "\u200b\u00a0"), ord(' '))

def normalize_for_pii(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DASH_MAP).translate(_WS_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    return s
