import re
from typing import List, Dict

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
