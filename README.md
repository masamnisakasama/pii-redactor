# pii-redactor
書面や動画において、名前、住所、顔などの個人情報を認識してエイリアスに変えるアプリです
## 目的<br>
個人的な目的<br><br>
diaogbotで出てきた反省点を元にPDCAを回そうと思い作りました<br>
色々考えた結果、セキュリティーレベルを切り替えできるシステムにし、より魅力的なサービスを目指します。<br>
簡素な方がユーザビリティが上がると考え、セキュリティは下の2段階に分けます。

## 1)完全オフライン（OpenCV + Tesseract or DNN）（ダウンロードで完全オフラインにして使える形式にすることを考えています）<br>
狙い：ネット遮断でも動く“最強に安全”な処理<br>
ネットワーク：外部通信なし（API不使用）<br>
OCR：Tesseract<br>
NER（個人情報抽出）：正規表現＋ルール（日本語住所・氏名パターン等）<br>
顔検出：OpenCV Haar<br>
強み：データ持ち出しゼロ／コンプラ最優先<br>
弱み：精度はAPIや学習済みモデルに劣ることがある、処理がやや重い<br>
## 代表環境変数例(あくまで予定)：<br>
- default_security_level=maximum
- disable_network_access=true
- use_nanobanana_ocr=false
- use_nanobanana_ner=false



## 2)AI機能最優先（外部ネットワーク使用、Nanobanana API）<br>
狙い：精度・利便性最優先。外部の高性能OCR/NER/顔検出をフル活用<br>
ネットワーク：許可（API前提）<br>
OCR：外部API（単語レベル/行レベルの高精度の予定）<br>
NER：外部API（BERT系等の高精度推論）、閾値高めで誤検出抑制<br>
顔検出：外部API／高精度アルゴリズム、失敗時OpenCV<br>
強み：最高の精度・多言語/非定型に強い・運用が軽い<br>
弱み：データ外送のリスクとAPIコスト／レイテンシ依存<br>
## 代表環境変数例(あくまで予定)：<br>
- default_security_level=enhanced
- nanobanana_api_key=
- nanobanana_endpoint=
- use_nanobanana_ocr=true
- use_nanobanana_ner=true

## API
GET /health <br>
稼働状況・現在のセキュリティレベルなどを返します。<br><br>
GET /capabilities<br>
サポート形式、最大サイズ、利用可能セキュリティレベルを返します。<br><br>
GET /security/status<br>
POST /security/level<br>
一時的にセキュリティレベルを変更できます。<br><br>
curl -s -X POST -F "level=maximum" http://127.0.0.1:8000/security/level | jq<br>
POST /redact/preview<br>
対象領域（bbox）とタイプを描画せず返します。<br>
policy：email,name,phone,id,amount,address,face からカンマ区切り<br>
font_path：style=readable系で描画合わせに使うフォント（トーンが崩れる場合に指定）<br><br>
POST /redact/replace<br>
実際に描画して置換（画像はPNG、PDFは再合成PDFで返却）<br>
style：readable | box | pixelate | blur<br>
replace_scope：token | line<br>
token：一致トークン単位で描画<br>
line：行全体を置換テキストで1回だけ描画（ゴースト抑制あり）<br>
face_mode：blur | pixelate | pixelate_strict | smart_blur | replace_face | keep<br>
consent_faces が granted でない場合は replace_face を smart_blur に強制ダウングレード<br><br>
POST /redact/face_image<br>
顔だけを指定モードで処理（画像入力限定）<br>
curl -s -X POST http://127.0.0.1:8000/redact/face_image \<br>
  -F "file=@faces.jpeg" -F "method=pixelate" -o out_face.png -D -<br><br>
POST /detect/summary<br>
置換はせず、ファイル内の PII 検出数を返却。<br>
（CIの公開前スキャン等で使う想定）<br>
policy：email,phone,address,id,face,...<br>
ocr_timeout_s：OCRが固まる環境でのタイムアウト用（実装が入っているブランチで有効）<br>

