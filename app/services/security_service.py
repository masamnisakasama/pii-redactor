from app.settings import Settings
from app.security_manager import SecurityToggleManager, SecurityLevel

# 旧900行版アプリ（/legacy にマウントされている）
try:
    from app.legacy_app import app as legacy_app
except Exception:
    legacy_app = None

_mgr: SecurityToggleManager | None = None

def _try_legacy_manager() -> SecurityToggleManager | None:
    """legacy_app.state に既に初期化済みのマネージャがあればそれを使う"""
    if not legacy_app:
        return None
    for key in ("security_toggle_manager", "security_manager", "manager"):
        m = getattr(legacy_app.state, key, None)
        if m is not None:
            return m
    return None

def get_manager() -> SecurityToggleManager:
    global _mgr
    if _mgr is not None:
        return _mgr
    # 1) まず legacy 側のインスタンスを共有
    m = _try_legacy_manager()
    if m is not None:
        _mgr = m
        return _mgr
    # 2) 無ければ新規に生成（将来レガシーを外したらこちらが動く）
    _mgr = SecurityToggleManager(Settings())
    return _mgr

def _level_to_str(lv) -> str:
    try:
        return lv.value
    except Exception:
        return str(lv)

def get_level_str() -> str:
    m = get_manager()
    # メソッド優先
    for meth in ("get_security_level", "get_current_level"):
        if hasattr(m, meth):
            return _level_to_str(getattr(m, meth)())
    # プロパティ候補
    for attr in ("security_level", "current_level", "level"):
        if hasattr(m, attr):
            return _level_to_str(getattr(m, attr))
    return "enhanced"

def set_level(level_str: str) -> str:
    m = get_manager()
    try:
        target = SecurityLevel(level_str)
    except Exception:
        target = level_str
    for meth in ("set_security_level", "set_level"):
        if hasattr(m, meth):
            getattr(m, meth)(target)
            return get_level_str()
    for attr in ("security_level", "current_level", "level"):
        if hasattr(m, attr):
            setattr(m, attr, target)
            return get_level_str()
    return get_level_str()
