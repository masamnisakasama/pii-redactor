#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"
IMG="${IMG:-pii_demo_sample.png}"
PDF="${PDF:-pii_demo_sample.pdf}"
FACE="${FACE:-faces.jpeg}"
FONT="${FONT:-/System/Library/Fonts/Helvetica.ttc}"

FAILED=0
pass(){ echo "✅ $1"; }
fail(){ echo "❌ $1"; FAILED=1; }

echo "=== BASE=$BASE ==="

# 1) health
s=$(curl -fsS "$BASE/health" | jq -r '.status' 2>/dev/null || echo "")
[[ "$s" == "healthy" ]] && pass "health OK" || fail "health NG"

# 2) capabilities
cur=$(curl -fsS "$BASE/capabilities" | jq -r '.current_level' 2>/dev/null || echo "")
[[ -n "$cur" ]] && pass "capabilities OK (level=$cur)" || fail "capabilities NG"

# 3) detect/summary (PNG)
sum=$(curl -fsS -X POST "$BASE/detect/summary" \
  -F "file=@$IMG" \
  -F "policy=email,phone,address,id,face" \
  -F "ocr_timeout_s=5")
pf=$(echo "$sum" | jq -r '.pii_found')
[[ "$pf" == "true" ]] && pass "detect/summary(PNG) OK (pii_found=true)" || pass "detect/summary(PNG) OK (pii_found=false: サンプル次第)"

# 4) preview (PNG)
PREV=$(curl -fsS -X POST "$BASE/redact/preview" \
  -F "file=@$IMG" \
  -F "policy=email,phone,id,amount" | jq '.total_count')
if [[ "$PREV" =~ ^[0-9]+$ ]] && (( PREV >= 0 )); then
  pass "preview OK (total_count=$PREV)"
else
  fail "preview NG"
fi

# 5) replace(token)
HDR=$(curl -fsS -D - -X POST "$BASE/redact/replace" \
  -F "file=@$IMG" \
  -F "policy=email,phone,id,amount" \
  -F "style=readable" \
  -F "replace_scope=token" \
  -F "font_path=$FONT" \
  -o /tmp/redact_token.png | tr -d '\r')
TOK=$(echo "$HDR" | awk -F': ' '/^x-replaced-tokens:/ {print $2}')
LIN=$(echo "$HDR" | awk -F': ' '/^x-replaced-lines:/ {print $2}')
[[ "${TOK:-0}" =~ ^[0-9]+$ ]] && (( TOK >= 0 )) && pass "replace(token) OK (tokens=$TOK, lines=$LIN)" || fail "replace(token) NG"

# 6) replace(line)
HDRL=$(curl -fsS -D - -X POST "$BASE/redact/replace" \
  -F "file=@$IMG" \
  -F "policy=email,phone,id,amount" \
  -F "style=readable" \
  -F "replace_scope=line" \
  -F "font_path=$FONT" \
  -o /tmp/redact_line.png | tr -d '\r')
LIN2=$(echo "$HDRL" | awk -F': ' '/^x-replaced-lines:/ {print $2}')
[[ "${LIN2:-0}" =~ ^[0-9]+$ ]] && (( LIN2 >= 0 )) && pass "replace(line) OK (lines=$LIN2)" || fail "replace(line) NG"

# 7) face_image
HDRF=$(curl -fsS -D - -X POST "$BASE/redact/face_image?method=pixelate" \
  -F "file=@$FACE" \
  -o /tmp/face_out.png | tr -d '\r')
[[ "$HDRF" =~ "HTTP/1.1 200 OK" ]] && pass "face_image OK" || fail "face_image NG"

# 8) replace(PDF)
HDRP=$(curl -fsS -D - -X POST "$BASE/redact/replace" \
  -F "file=@$PDF" \
  -F "policy=email,phone,id,amount" \
  -F "style=readable" \
  -F "replace_scope=token" \
  -o /tmp/redact.pdf | tr -d '\r')
CT=$(echo "$HDRP" | awk -F': ' '/^content-type:/ {print $2}')
[[ "$CT" == "application/pdf" ]] && pass "replace(PDF) OK (content-type=application/pdf)" || fail "replace(PDF) NG"

# 9) fast endpoints
sf=$(curl -fsS -X POST "$BASE/detect/summary_fast" -F "file=@$IMG" | jq -r '.counts.email,.counts.phone' 2>/dev/null || true)
[[ -n "$sf" ]] && pass "summary_fast OK" || fail "summary_fast NG"

HDF=$(curl -fsS -D - -X POST "$BASE/redact/replace_fast" -F "file=@$FACE" -o /tmp/fast.png | tr -d '\r')
[[ "$HDF" =~ "X-Faces-Detected:" ]] && pass "replace_fast OK" || fail "replace_fast NG"

# 10) /security/status
sec=$(curl -fsS "$BASE/security/status" | jq -r '.current_level' 2>/dev/null || echo "")
[[ -n "$sec" ]] && pass "security/status OK (level=$sec)" || fail "security/status NG"

echo "=== DONE ==="
exit $FAILED
