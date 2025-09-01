# app/text_detect.py
# EAST text detector (CPU / offline). Returns list[(x,y,w,h)] in original image coords.

# 主な中身：
# _get_east_net(model_path) : EAST(テキスト認識ライブラリ)pb を読み込む 出力をDNNに繋いでいる
# _east_input_size(w,h,max_side) : 32の倍数に丸める前処理
# _decode_east(scores, geometry, conf_thr) : OpenCV サンプル
# detect_text_boxes_east　読み込んだテキストをボックス化　→ 元画像座標系の (x,y,w,h) リストを返す
# draw_boxes_debug(bgr, boxes) … デバッグ用矩形描画（実際には未だ未使用だが今後使うかも）

from __future__ import annotations
import os, math
from typing import List, Tuple
import cv2
import numpy as np

# Simple in-proc cache so we don't reload the .pb every request
_EAST_NET_CACHE: dict[str, "cv2.dnn_Net"] = {}

def _get_east_net(model_path: str) -> "cv2.dnn_Net":
    net = _EAST_NET_CACHE.get(model_path)
    if net is None:
        net = cv2.dnn.readNet(model_path)
        # CPU only; if backend/target unsupported, OpenCV falls back internally
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass
        _EAST_NET_CACHE[model_path] = net
    return net

def _east_input_size(w: int, h: int, max_side: int = 1280) -> Tuple[int, int, float, float]:
    """Resize so both sides are multiples of 32, not exceeding max_side.
    Returns (new_w, new_h, sx, sy), where sx = new_w / w, sy = new_h / h.
    """
    scale = min(max_side / float(max(w, h)), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    nw = max(32, (nw // 32) * 32)
    nh = max(32, (nh // 32) * 32)
    return nw, nh, nw / float(w), nh / float(h)

def _decode_east(scores: np.ndarray,
                 geometry: np.ndarray,
                 conf_thr: float = 0.5) -> Tuple[List[List[int]], List[float]]:
    """Decode EAST output (OpenCV sample style) -> axis-aligned boxes + confidences."""
    numRows, numCols = scores.shape[2], scores.shape[3]
    boxes: List[List[int]] = []
    confs: List[float] = []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(numCols):
            score = float(scoresData[x])
            if score < conf_thr:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = float(angles[x])
            c, s = math.cos(angle), math.sin(angle)
            h = float(x0[x] + x2[x])
            w = float(x1[x] + x3[x])
            endX = int(offsetX + (c * x1[x]) + (s * x2[x]))
            endY = int(offsetY - (s * x1[x]) + (c * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            boxes.append([startX, startY, int(w), int(h)])
            confs.append(score)
    return boxes, confs

def detect_text_boxes_east(bgr: np.ndarray,
                           model_path: str | None = None,
                           conf_thr: float = 0.5,
                           nms_thr: float = 0.4,
                           max_side: int = 1280,
                           min_size: int = 6) -> List[Tuple[int, int, int, int]]:
    """Detect text regions with EAST; returns list of (x,y,w,h) on the original image."""
    model_path = model_path or os.getenv("EAST_PB", "models/text/frozen_east_text_detection.pb")
    if not model_path or not os.path.exists(model_path):
        return []

    H, W = bgr.shape[:2]
    inpW, inpH, sx, sy = _east_input_size(W, H, max_side=max_side)

    blob = cv2.dnn.blobFromImage(
        bgr, scalefactor=1.0, size=(inpW, inpH),
        mean=(123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net = _get_east_net(model_path)
    net.setInput(blob)

    # Try canonical output names; fallback to unconnected layer names
    try:
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                        "feature_fusion/concat_3"])
    except Exception:
        outs = net.forward(net.getUnconnectedOutLayersNames())
        if len(outs) < 2:
            return []
        # Identify by channels: scores has C=1, geometry has C=5
        a, b = outs[0], outs[1]
        if a.shape[1] == 1:
            scores, geometry = a, b
        else:
            scores, geometry = b, a

    boxes, confs = _decode_east(scores, geometry, conf_thr=conf_thr)
    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thr, nms_thr)
    if len(indices) == 0:
        return []

    out: List[Tuple[int, int, int, int]] = []
    for i in indices.flatten().tolist():
        x, y, w, h = boxes[i]
        # Map back to original coordinates
        x = int(x / sx); y = int(y / sy)
        w = int(w / sx); h = int(h / sy)
        # Clamp and discard tiny/invalid
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(0, min(w, W - x)); h = max(0, min(h, H - y))
        if w >= min_size and h >= min_size:
            out.append((x, y, w, h))
    return out

# Optional small debug helper (not used in production)
def draw_boxes_debug(bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    vis = bgr.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
    return vis
