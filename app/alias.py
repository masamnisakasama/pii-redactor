import hmac, hashlib, base64, random
from faker import Faker

def _seed(key_b64: str, ns: str, kind: str, orig: str) -> int:
    key = base64.b64decode(key_b64)
    dig = hmac.new(key, f"{ns}|{kind}|{orig}".encode(), hashlib.sha256).digest()
    return int.from_bytes(dig[:8], "big")

#　上から順にメール、電話番号、金額、ID、名前、住所をエイリアスにする
def alias_value(kind: str, orig: str, key_b64: str, ns: str) -> str:
    s = _seed(key_b64, ns, kind, orig)
    fk = Faker("ja_JP"); fk.seed_instance(s); rnd = random.Random(s)
    if kind == "email":  return f"{fk.user_name()}@{rnd.choice(['example.dev','example.com','invalid.test'])}"
    if kind == "phone":  return f"0{rnd.randint(1,9)}-{rnd.randint(1000,9999)}-{rnd.randint(1000,9999)}"
    if kind == "amount": return f"¥{rnd.randint(1,9)},{rnd.randint(0,999):03},{rnd.randint(0,999):03}"
    if kind == "id":     return f"{rnd.choice(['USR','ORD','ACC'])}-{rnd.randint(10000,99999)}"
    if kind == "name":   return fk.name()
    if kind == "address":return fk.address().split("\n")[0]
    return fk.word()
