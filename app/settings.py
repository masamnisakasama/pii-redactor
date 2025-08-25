from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    cors_origin: str = "*"  # 本番はセキュリティ的に限定した方がよし
    tenant_hmac_key_b64: str = "REPLACE_ME_BASE64"  # KMS注入するつもり
    ner_endpoint: str | None = None  # NER:TFの固有表現認識　使う時の例→http://tfserving:8500/v1/models/ja_ner:predict
    confidence_threshold: float = 0.80

    class Config:
        env_file = ".env"
        case_sensitive = True
