#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"

say() { printf "\n=== %s ===\n" "$*"; }

say "1) /health"
curl -sf "$BASE/health" >/dev/null

say "2) /capabilities (enhanced available?)"
cap=$(curl -sf "$BASE/capabilities")
echo "$cap" | grep -E '"enhanced".*("available":\s*true)' >/dev/null || {
  echo "[WARN] enhanced.available=false かも。env を確認してください。"
  echo "$cap" | jq
}

say "3) switch -> maximum -> enhanced"
curl -sf -X POST "$BASE/security/level" -H 'Content-Type: application/json' -d '{"level":"maximum"}' >/dev/null
curl -sf -X POST "$BASE/security/level" -H 'Content-Type: application/json' -d '{"level":"enhanced"}'  >/dev/null
curl -sf "$BASE/security/status" | jq '.current_level'

say "4) redact/replace (png)"
curl -sf -X POST "$BASE/redact/replace" \
  -F "file=@./pii_demo_sample.png" \
  -F "policy=email,phone,address,id,face" \
  -F "style=readable" -F "consistency_key=demo" -o out_replace.png
[ -s out_replace.png ] || { echo "NG: out_replace.png 空"; exit 1; }

say "5) redact/replace (pdf)"
curl -sf -X POST "$BASE/redact/replace" \
  -F "file=@./pii_demo_sample.pdf" \
  -F "policy=email,phone,address,id,face" \
  -F "style=readable" -F "consistency_key=demo" -o out_replace.pdf
[ -s out_replace.pdf ] || { echo "NG: out_replace.pdf 空"; exit 1; }

say "6) face_image pixelate"
curl -sf -X POST "$BASE/redact/face_image" \
  -F "file=@./faces.jpeg" -F "method=pixelate" -o out_face_pixelate.png
[ -s out_face_pixelate.png ] || { echo "NG: out_face_pixelate.png 空"; exit 1; }

say "7) face_image replace_face (may take a bit)"
curl -sf -X POST "$BASE/redact/face_image" \
  -F "file=@./faces.jpeg" -F "method=replace_face" -o out_face_replace.png || {
    echo "[WARN] replace_face 失敗。GOOGLE_API_KEY / Enhanced / 外部API設定を確認してください。"
  }

echo "OK: smoke passed"
