# app/security_manager.py
# 完全オフラインOpenCV + Tesseract、精度低）かオンライン（Nanobanan APIを使う、精度高）か選べるように

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio

class SecurityLevel(Enum):
    """セキュリティレベルの定義"""
    MAXIMUM = "maximum"      # 完全オフライン（OpenCV + Tesseract）
    HIGH = "high"           # オンプレミスAI + 限定的外部API
    STANDARD = "standard"   # バランス型（一部API使用）
    ENHANCED = "enhanced"   # AI機能最優先（Nanobanan API）

@dataclass
class SecurityConfig:
    """セキュリティ設定"""
    level: SecurityLevel
    allow_external_api: bool
    allow_cloud_ocr: bool
    allow_cloud_ner: bool
    allow_cloud_face_detection: bool
    data_encryption_required: bool
    audit_logging: bool
    network_restrictions: List[str]
    
    def __post_init__(self):
        """設定の整合性チェック"""
        if self.level == SecurityLevel.MAXIMUM:
            self.allow_external_api = False
            self.allow_cloud_ocr = False
            self.allow_cloud_ner = False
            self.allow_cloud_face_detection = False
            self.data_encryption_required = True
            self.audit_logging = True

class ProcessorInterface(ABC):
    """AI処理インターフェース"""
    
    @abstractmethod
    async def ocr_process(self, image, **kwargs) -> List[Dict]:
        pass
    
    @abstractmethod
    async def ner_process(self, texts: List[str], **kwargs) -> List[Dict]:
        pass
    
    @abstractmethod
    async def face_detect(self, image, **kwargs) -> List[Dict]:
        pass
    
    @property
    @abstractmethod
    def security_level(self) -> SecurityLevel:
        pass

class MaximumSecurityProcessor(ProcessorInterface):
    """最高セキュリティ：完全オフライン処理"""
    
    def __init__(self):
        self.security_logger = logging.getLogger('MaxSecurityProcessor')
        self._initialize_offline_models()
    
    def _initialize_offline_models(self):
        """オフラインモデルの初期化"""
        try:
            import cv2
            import pytesseract
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.security_logger.info("Offline models initialized")
        except Exception as e:
            self.security_logger.error(f"Failed to initialize offline models: {e}")
    
    async def ocr_process(self, image, **kwargs) -> List[Dict]:
        """Tesseractのみを使用したOCR"""
        import pytesseract
        from PIL import Image
        import numpy as np
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            # OCR設定（高精度モード）
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(
                image, 
                lang='jpn+eng',
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text or int(data['conf'][i]) < 60:
                    continue
                
                bbox = (
                    data['left'][i], data['top'][i],
                    data['left'][i] + data['width'][i],
                    data['top'][i] + data['height'][i]
                )
                
                results.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': float(data['conf'][i]) / 100.0,
                    'method': 'offline_tesseract',
                    'security_level': 'maximum'
                })
            
            self.security_logger.info(f"Offline OCR processed {len(results)} elements")
            return results
            
        except Exception as e:
            self.security_logger.error(f"Offline OCR error: {e}")
            return []
    
    async def ner_process(self, texts: List[str], **kwargs) -> List[Dict]:
        """ルールベース + 辞書ベースNER"""
        import re
        results = []
        
        # 日本語人名パターン（改良版）
        patterns = {
            'name': [
                r'[一-龠]{2,4}\s*[一-龠]{1,3}(?:さん|様|氏|先生|君|ちゃん)?',
                r'[ぁ-ゔ]{3,8}(?:さん|様|氏|先生|君|ちゃん)?',
                r'[ァ-ヶ]{3,8}(?:さん|様|氏|先生|君|ちゃん)?'
            ],
            'address': [
                r'[一-龠]{1,10}[都道府県][一-龠]{1,15}[市区町村][一-龠0-9\-\s]{0,30}',
                r'〒\s*\d{3}-?\d{4}',
                r'[0-9]{1,5}-[0-9]{1,5}-[0-9]{1,5}\s*[一-龠]{1,10}'
            ]
        }
        
        for text in texts:
            for entity_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        results.append({
                            'text': match.group(),
                            'type': entity_type,
                            'conf': 0.75,  # ルールベースは固定信頼度
                            'reason': f'offline_rule_{entity_type}',
                            'security_level': 'maximum'
                        })
        
        self.security_logger.info(f"Offline NER found {len(results)} entities")
        return results
    
    async def face_detect(self, image, **kwargs) -> List[Dict]:
        """OpenCV Haar Cascadesによる顔検出"""
        import cv2
        import numpy as np
        from PIL import Image
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.8,  # Haarは信頼度を返さないので固定値
                    'method': 'offline_haar',
                    'security_level': 'maximum'
                })
            
            self.security_logger.info(f"Offline face detection found {len(results)} faces")
            return results
            
        except Exception as e:
            self.security_logger.error(f"Offline face detection error: {e}")
            return []
    
    @property
    def security_level(self) -> SecurityLevel:
        return SecurityLevel.MAXIMUM

class EnhancedAIProcessor(ProcessorInterface):
    """AI機能強化：外部API使用"""
    
    def __init__(self, nanobanan_config: Dict[str, Any]):
        self.nanobanan_config = nanobanan_config
        self.security_logger = logging.getLogger('EnhancedAIProcessor')
    
    async def ocr_process(self, image, **kwargs) -> List[Dict]:
        """Nanobanan OCR API + Tesseractフォールバック"""
        import httpx
        import base64
        import io
        from PIL import Image
        
        if self.nanobanan_config.get('api_key'):
            try:
                # Nanobanan API使用
                buf = io.BytesIO()
                if hasattr(image, 'save'):
                    image.save(buf, format='PNG')
                else:
                    Image.fromarray(image).save(buf, format='PNG')
                
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                
                payload = {
                    "image": img_b64,
                    "languages": ["ja", "en"],
                    "enhance_quality": True
                }
                
                headers = {
                    "Authorization": f"Bearer {self.nanobanan_config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.nanobanan_config['endpoint']}/v1/ocr",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()
                
                results = []
                for item in result.get("words", []):
                    bbox = item.get("bbox", [0,0,0,0])
                    text = item.get("text", "").strip()
                    confidence = float(item.get("confidence", 0.0))
                    
                    if text and confidence > 0.7:
                        results.append({
                            'text': text,
                            'bbox': tuple(bbox),
                            'confidence': confidence,
                            'method': 'nanobanan_ocr',
                            'security_level': 'enhanced'
                        })
                
                self.security_logger.info(f"Nanobanan OCR processed {len(results)} elements")
                return results
                
            except Exception as e:
                self.security_logger.warning(f"Nanobanan OCR failed, falling back: {e}")
        
        # フォールバック：Tesseract
        fallback_processor = MaximumSecurityProcessor()
        return await fallback_processor.ocr_process(image, **kwargs)
    
    async def ner_process(self, texts: List[str], **kwargs) -> List[Dict]:
        """Nanobanan NER API + ルールベースフォールバック"""
        if self.nanobanan_config.get('api_key') and texts:
            try:
                import httpx
                
                payload = {
                    "texts": texts,
                    "entities": ["PERSON", "ADDRESS", "ORGANIZATION", "PHONE", "EMAIL"],
                    "language": "ja"
                }
                
                headers = {
                    "Authorization": f"Bearer {self.nanobanan_config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.nanobanan_config['endpoint']}/v1/ner",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()
                
                results = []
                for prediction in result.get("predictions", []):
                    text = prediction.get("text", "")
                    entity_type = prediction.get("entity", "")
                    confidence = float(prediction.get("confidence", 0.0))
                    
                    type_mapping = {
                        "PERSON": "name",
                        "ADDRESS": "address",
                        "ORGANIZATION": "org",
                        "PHONE": "phone",
                        "EMAIL": "email"
                    }
                    
                    app_type = type_mapping.get(entity_type)
                    if app_type and confidence >= 0.85:
                        results.append({
                            'text': text,
                            'type': app_type,
                            'conf': confidence,
                            'reason': f'nanobanan_ner_{entity_type.lower()}',
                            'security_level': 'enhanced'
                        })
                
                self.security_logger.info(f"Nanobanan NER found {len(results)} entities")
                return results
                
            except Exception as e:
                self.security_logger.warning(f"Nanobanan NER failed, falling back: {e}")
        
        # フォールバック：ルールベース
        fallback_processor = MaximumSecurityProcessor()
        return await fallback_processor.ner_process(texts, **kwargs)
    
    async def face_detect(self, image, **kwargs) -> List[Dict]:
        """Nanobanan Face API + OpenCVフォールバック"""
        if self.nanobanan_config.get('api_key'):
            try:
                import httpx
                import base64
                import io
                from PIL import Image
                
                # 画像をBase64エンコード
                buf = io.BytesIO()
                if hasattr(image, 'save'):
                    image.save(buf, format='PNG')
                else:
                    Image.fromarray(image).save(buf, format='PNG')
                
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                
                payload = {
                    "image": img_b64,
                    "detection_threshold": 0.3,
                    "return_landmarks": False
                }
                
                headers = {
                    "Authorization": f"Bearer {self.nanobanan_config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.nanobanan_config['endpoint']}/v1/face-detect",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()
                
                results = []
                for detection in result.get("faces", []):
                    bbox = detection.get("bbox", [])
                    confidence = float(detection.get("confidence", 0.0))
                    
                    if len(bbox) == 4 and confidence > 0.3:
                        x, y, w, h = bbox
                        results.append({
                            'bbox': (int(x), int(y), int(x+w), int(y+h)),
                            'confidence': confidence,
                            'method': 'nanobanan_face',
                            'security_level': 'enhanced'
                        })
                
                self.security_logger.info(f"Nanobanan face detection found {len(results)} faces")
                return results
                
            except Exception as e:
                self.security_logger.warning(f"Nanobanan face detection failed, falling back: {e}")
        
        # フォールバック：OpenCV
        fallback_processor = MaximumSecurityProcessor()
        return await fallback_processor.face_detect(image, **kwargs)
    
    @property
    def security_level(self) -> SecurityLevel:
        return SecurityLevel.ENHANCED

class SecurityToggleManager:
    """セキュリティレベル切り替え管理"""
    
    def __init__(self, settings):
        self.settings = settings
        self.current_level = SecurityLevel.MAXIMUM  # デフォルトは最高セキュリティ
        self.processors: Dict[SecurityLevel, ProcessorInterface] = {}
        self.security_logger = logging.getLogger('SecurityToggleManager')
        self._initialize_processors()
    
    def _initialize_processors(self):
        """各セキュリティレベルのプロセッサを初期化"""
        # 最高セキュリティ（必須）
        self.processors[SecurityLevel.MAXIMUM] = MaximumSecurityProcessor()
        
        # AI強化版（設定がある場合のみ）
        if self.settings.nanobanan_api_key:
            nanobanan_config = {
                'api_key': self.settings.nanobanan_api_key,
                'endpoint': self.settings.nanobanan_endpoint
            }
            self.processors[SecurityLevel.ENHANCED] = EnhancedAIProcessor(nanobanan_config)
        
        self.security_logger.info(f"Initialized processors for levels: {list(self.processors.keys())}")
    
    def set_security_level(self, level: SecurityLevel) -> bool:
        """セキュリティレベルを変更"""
        if level not in self.processors:
            self.security_logger.error(f"Security level {level} not available")
            return False
        
        old_level = self.current_level
        self.current_level = level
        
        self.security_logger.info(f"Security level changed: {old_level} -> {level}")
        return True
    
    def get_current_processor(self) -> ProcessorInterface:
        """現在のセキュリティレベルのプロセッサを取得"""
        return self.processors[self.current_level]
    
    def get_available_levels(self) -> List[SecurityLevel]:
        """利用可能なセキュリティレベルを取得"""
        return list(self.processors.keys())
    
    def get_security_info(self) -> Dict[str, Any]:
        """現在のセキュリティ状況を取得"""
        processor = self.get_current_processor()
        return {
            'current_level': self.current_level.value,
            'available_levels': [level.value for level in self.get_available_levels()],
            'processor_type': type(processor).__name__,
            'external_api_enabled': self.current_level != SecurityLevel.MAXIMUM,
            'capabilities': {
                'ocr': True,
                'ner': True,
                'face_detection': True,
                'high_accuracy_ocr': self.current_level == SecurityLevel.ENHANCED,
                'advanced_ner': self.current_level == SecurityLevel.ENHANCED,
                'ml_face_detection': self.current_level == SecurityLevel.ENHANCED
            }
        }
    
    async def process_document(self, image, policies: set, **kwargs) -> Dict[str, Any]:
        """現在のセキュリティレベルで文書処理"""
        processor = self.get_current_processor()
        
        results = {
            'security_level': self.current_level.value,
            'processor': type(processor).__name__,
            'ocr_results': [],
            'ner_results': [],
            'face_results': []
        }
        
        try:
            # OCR処理
            if any(p in policies for p in ['email', 'phone', 'amount', 'id']):
                ocr_results = await processor.ocr_process(image, **kwargs)
                results['ocr_results'] = ocr_results
            
            # NER処理
            if any(p in policies for p in ['name', 'address', 'org']):
                texts = [r['text'] for r in results['ocr_results']]
                if texts:
                    ner_results = await processor.ner_process(texts, **kwargs)
                    results['ner_results'] = ner_results
            
            # 顔検出処理
            if 'face' in policies:
                face_results = await processor.face_detect(image, **kwargs)
                results['face_results'] = face_results
            
            self.security_logger.info(
                f"Document processed at {self.current_level.value} level: "
                f"OCR={len(results['ocr_results'])}, "
                f"NER={len(results['ner_results'])}, "
                f"Faces={len(results['face_results'])}"
            )
            
        except Exception as e:
            self.security_logger.error(f"Processing error at {self.current_level.value} level: {e}")
            results['error'] = str(e)
        
        return results