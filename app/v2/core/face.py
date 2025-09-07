import os, cv2, numpy as np

_CASCADE = None

def _cascade_path() -> str:
    # 1) 同梱モデル 2) OpenCV内蔵 3) 明示パス(OPENCV_HAAR_PATH)
    p1 = os.path.join(os.path.dirname(__file__), "../../models/haarcascade_frontalface_default.xml")
    p1 = os.path.abspath(p1)
    if os.path.exists(p1): return p1
    p2 = getattr(cv2.data, "haarcascades", None)
    if p2: 
        p2f = os.path.join(p2, "haarcascade_frontalface_default.xml")
        if os.path.exists(p2f): return p2f
    p3 = os.getenv("OPENCV_HAAR_PATH")
    if p3 and os.path.exists(p3): return p3
    raise FileNotFoundError("haar cascade xml not found")

def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(_cascade_path())
        if _CASCADE.empty():
            raise RuntimeError("Failed to load Haar cascade")
    return _CASCADE

def detect_faces_bgr(img_bgr: np.ndarray) -> list[tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 明度のばらつきに強く
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape[:2]
    min_size = int(os.getenv("FACE_MIN_SIZE_PX", "30"))
    scale = 1.0
    # 小さい顔対策：短辺<240なら等倍検出が弱いのでアップサンプリング
    if min(h, w) < 240:
        scale = 240.0 / min(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    scaleFactor = float(os.getenv("FACE_CASCADE_SCALE_FACTOR", "1.1"))
    minNeighbors = int(os.getenv("FACE_CASCADE_MIN_NEIGHBORS", "4"))

    rects = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    # 元スケールに戻してint化
    boxes = []
    for (x,y,w2,h2) in rects:
        if scale != 1.0:
            x = int(x/scale); y=int(y/scale); w2=int(w2/scale); h2=int(h2/scale)
        # 極端な縦横比は弾く（過度に厳しくしない）
        if h2>=min_size and w2>=min_size and 0.5 <= (w2/max(h2,1)) <= 2.0:
            boxes.append((int(x),int(y),int(w2),int(h2)))
    return boxes
