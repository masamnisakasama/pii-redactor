from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import model_validator
from typing import Literal
class Settings(BaseSettings):
    # 基本設定
    cors_origin: str = "*"
    tenant_hmac_key_b64: str = "REPLACE_ME_BASE64"
    confidence_threshold: float = 0.80
    
    # セキュリティレベル設定
    default_security_level: str = "maximum"  # "maximum" | "high" | "standard" | "enhanced"
    disable_network_access: bool = False     # 最高セキュリティモード
    require_auth: bool = False               # 認証を必須にするか
    security_token: Optional[str] = None     # セキュリティトークン
    
    # Nanobanan API設定（AI機能用）
    nanobanan_api_key: Optional[str] = None
    nanobanan_endpoint: Optional[str] = None  # https://api.nanobanan.ai
    use_nanobanan_ocr: bool = False
    use_nanobanan_ner: bool = False
    
    # レガシーTensorFlow Serving設定（後方互換性）
    ner_endpoint: Optional[str] = None
    
    # OCR設定
    ocr_confidence_threshold: float = 0.6
    tesseract_lang: str = "jpn+eng"
    tesseract_custom_config: str = r'--oem 3 --psm 6'
    
    # NER設定
    ner_confidence_threshold: float = 0.85
    
    # 顔認識設定
    face_detection_method: str = "haar"  # "haar" | "dnn" | "mediapipe"
    face_cascade_scale_factor: float = 1.1
    face_cascade_min_neighbors: int = 5
    face_detection_confidence: float = 0.3
    
    # パフォーマンス設定
    max_image_size: int = 2048
    pdf_dpi: int = 180
    max_file_size_mb: int = 50
    allowed_file_extensions: str = "pdf,png,jpg,jpeg,gif,bmp,tiff"
    
    # モデルキャッシュ設定（protected namespaceを避けるためリネーム）
    ai_model_cache_dir: str = "./models"
    download_models_on_startup: bool = True
    
    # ログ設定
    log_level: str = "INFO"
    security_log_file: str = "security.log"
    enable_audit_logging: bool = True
    
    # セキュリティ機能トグル
    allow_security_level_change: bool = True  # 実行時のセキュリティレベル変更を許可
    allow_temporary_level_change: bool = True # リクエスト単位での一時的レベル変更
    
    # 追加フィールド（.envから読み込み用）
    secret_key: Optional[str] = None
    api_rate_limit: int = 100
    
    # HF/NER関連
    ner_engine: Literal['regex', 'hf', 'hybrid'] = 'hf'
    ner_model_name: str = 'dslim/bert-base-NER'
    ner_aggregation: Literal['simple', 'average', 'max'] = 'simple'
    
    # DNN 検出の閾値（顔検出で使用）
    dnn_confidence_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = False  # 大文字小文字を区別しない
        extra = "ignore"        # 余分なフィールドを無視
        # protected_namespaceを設定してmodel_警告を回避
        protected_namespaces = ('settings_',)
    
    @property
    def allowed_extensions_list(self) -> list[str]:
        """許可された拡張子のリストを返す"""
        return [ext.strip().lower() for ext in self.allowed_file_extensions.split(",")]
    
    @property
    def is_nanobanan_configured(self) -> bool:
        """Nanobanan APIが設定されているかチェック"""
        return bool(self.nanobanan_api_key and self.nanobanan_endpoint)
    
    @property
    def security_features(self) -> dict:
        """利用可能なセキュリティ機能の情報"""
        return {
            "levels": {
                "maximum": {
                    "available": True,
                    "description": "完全オフライン処理",
                    "features": ["opencv", "tesseract", "rule_based_ner"]
                },
                "enhanced": {
                    "available": self.is_nanobanan_configured,
                    "description": "AI機能フル活用",
                    "features": ["nanobanan_ocr", "nanobanan_ner", "nanobanan_face", "fallback_offline"]
                }
            },
            "network_restrictions": not self.disable_network_access,
            "authentication_required": self.require_auth
        }
    
