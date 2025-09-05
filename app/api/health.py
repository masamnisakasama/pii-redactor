from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def health():
    return {"status": "healthy"}

@router.get("/capabilities")
def capabilities():
    return {"current_level": "enhanced"}
