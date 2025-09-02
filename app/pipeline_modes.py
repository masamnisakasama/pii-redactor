# pipeline_modes.py
import os
_MODE_MAP = {
    "keep": "keep",
    "blur": "blur",
    "mosaic": "pixelate",
    "mosaic_strict": "pixelate_strict",
    "pixelate": "pixelate",
    "replace_face": "replace_face",
    "swap": "replace_face",
    "smart_blur": "smart_blur",
    "block": "keep",
}
def resolve_face_mode(form_face_mode: str | None = None) -> str:
    if form_face_mode:
        key = form_face_mode.strip().lower()
        return _MODE_MAP.get(key, "blur")
    key = os.getenv("PII_IMAGES_MODE", "blur").strip().lower()
    return _MODE_MAP.get(key, "blur")
def enforce_face_consent(face_method: str, consent_faces: str) -> str:
    if face_method == "replace_face" and str(consent_faces).lower() != "granted":
        return "smart_blur"
    return face_method
