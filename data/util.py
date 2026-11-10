"""
Image and Bounding Box Utilities for Low-Light Object Detection Datasets.
Supports:
  - Letterbox resizing (aspect-ratio preserving resize + stride-32 padding)
  - Coordinate transforms (xywhn2xyxy, xyxy2xywhn, clip_coords)
  - YOLOv5 geometric & photometric augmentations (random_perspective, augment_hsv, random_gamma)
  - Multi-format label parsers (YOLO normalized, ExDark raw, DarkFace pixel format)
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
EXDARK_CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(EXDARK_CLASSES)}

DARKFACE_CLASSES = ["Face"]


def is_image_file(filename: str) -> bool:
    """Check if a filename has a supported image extension."""
    return filename.lower().endswith(IMG_EXTENSIONS)


def img2label_paths(img_paths: List[str]) -> List[str]:
    """
    Map image file paths to corresponding YOLO label (.txt) file paths.
    Handles standard `images/` -> `labels/` and `image/` -> `label/` conventions.
    """
    label_paths = []
    for p in img_paths:
        p_norm = p.replace("\\", "/")
        if "/images/" in p_norm:
            lp = p_norm.replace("/images/", "/labels/")
        elif "/image/" in p_norm:
            lp = p_norm.replace("/image/", "/label/")
        elif "/Image/" in p_norm:
            lp = p_norm.replace("/Image/", "/Label/")
        else:
            lp = p_norm
        lp = str(Path(lp).with_suffix(".txt"))
        # ExDark raw labels sometimes have .jpg.txt or .png.txt
        if not os.path.exists(lp) and os.path.exists(lp[:-4] + Path(p).suffix + ".txt"):
            lp = lp[:-4] + Path(p).suffix + ".txt"
        label_paths.append(lp)
    return label_paths


def xywhn2xyxy(
    x: np.ndarray,
    w: float = 640.0,
    h: float = 640.0,
    padw: float = 0.0,
    padh: float = 0.0,
) -> np.ndarray:
    """
    Convert normalized [x_center, y_center, width, height] in [0, 1]
    to pixel [x1, y1, x2, y2] coordinates with optional letterbox padding.
    """
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2.0) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2.0) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2.0) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2.0) + padh  # bottom right y
    return y


def xyxy2xywhn(
    x: np.ndarray,
    w: float = 640.0,
    h: float = 640.0,
    clip: bool = False,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Convert pixel [x1, y1, x2, y2] coordinates to normalized
    [x_center, y_center, width, height] in [0, 1].
    """
    if clip:
        clip_coords(x, (int(h - eps), int(w - eps)))
    y = np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2.0) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2.0) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def clip_coords(boxes: np.ndarray, shape: Tuple[int, int]) -> None:
    """Clip bounding boxes [x1, y1, x2, y2] in-place to image shape (height, width)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints (Official YOLOv5 Letterbox).

    Returns:
        im (np.ndarray): Resized and padded image of shape (H_new, W_new, 3).
        ratio (Tuple[float, float]): (width_ratio, height_ratio) scaling factors.
        (dw, dh) (Tuple[float, float]): Left/right and top/bottom padding in pixels.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2.0,
    ar_thr: float = 100.0,
    area_thr: float = 0.1,
    eps: float = 1e-16,
) -> np.ndarray:
    """
    Compute candidate boxes after geometric augmentation:
    box1(4,n) is before augmentation, box2(4,n) is after augmentation.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )


def random_perspective(
    im: np.ndarray,
    targets: np.ndarray = (),
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    border: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random affine / perspective transformation to image and pixel bounding boxes
    [cls, x1, y1, x2, y2]. Official YOLOv5 implementation.
    """
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(
                im, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            im = cv2.warpAffine(
                im, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
            n, 8
        )  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # Filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def augment_hsv(
    im: np.ndarray,
    hgain: float = 0.015,
    sgain: float = 0.7,
    vgain: float = 0.4,
) -> None:
    """HSV color-space augmentation in-place on BGR uint8 image."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def random_gamma(
    im: np.ndarray,
    start_gamma: int = 60,
    end_gamma: int = 120,
) -> np.ndarray:
    """
    Apply random gamma illumination curve to simulate varying low-light exposure levels.
    Matches the `--gamma`, `--start_gamma`, `--end_gamma` options from IGMamba-EdgeLCA-LLIE.
    """
    gamma = random.randint(start_gamma, end_gamma) / 100.0
    if abs(gamma - 1.0) < 1e-3:
        return im
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(im, table)


def parse_label_file(
    label_path: str,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    class_to_id: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Universal label file parser supporting:
      1. Standard YOLO normalized format: `<class_id> <xc> <yc> <w> <h>` in [0, 1]
      2. ExDark raw format: `% bbGt version=3 \n <class_name> <left> <top> <w> <h> ...` (pixels)
      3. DarkFace format: line 1 = count N, lines 2.. = `<xmin> <ymin> <xmax> <ymax>` (pixels)

    Returns:
        labels (np.ndarray): Array of shape (N, 5) with rows `[class_id, xc, yc, w, h]`
                             normalized in [0, 1].
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return np.zeros((0, 5), dtype=np.float32)

    if class_to_id is None:
        class_to_id = EXDARK_CLASS_TO_ID

    # Case 1: ExDark raw format (starts with % or first token is a class name)
    non_comment = [ln for ln in lines if not ln.startswith("%")]
    if not non_comment:
        return np.zeros((0, 5), dtype=np.float32)

    first_tokens = non_comment[0].split()

    # Check if DarkFace format: line 1 has 1 integer (count N), and line 2 has 4 numbers
    if len(first_tokens) == 1 and first_tokens[0].isdigit():
        if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = []
        for ln in non_comment[1:]:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                x1, y1, x2, y2 = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
            except ValueError:
                continue
            x1 = max(0.0, min(x1, float(img_w)))
            y1 = max(0.0, min(y1, float(img_h)))
            x2 = max(0.0, min(x2, float(img_w)))
            y2 = max(0.0, min(y2, float(img_h)))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            xc = (x1 + bw / 2.0) / img_w
            yc = (y1 + bh / 2.0) / img_h
            boxes.append([0.0, xc, yc, bw / img_w, bh / img_h])
        return (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 5), dtype=np.float32)
        )

    # Check if ExDark raw format: first token is a non-numeric class name
    is_numeric = True
    try:
        float(first_tokens[0])
    except ValueError:
        is_numeric = False

    if not is_numeric:
        if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = []
        for ln in non_comment:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cname = parts[0]
            if cname not in class_to_id:
                continue
            cid = float(class_to_id[cname])
            try:
                x, y, w, h = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except ValueError:
                continue
            x1 = max(0.0, min(x, float(img_w)))
            y1 = max(0.0, min(y, float(img_h)))
            x2 = max(0.0, min(x + w, float(img_w)))
            y2 = max(0.0, min(y + h, float(img_h)))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            xc = (x1 + bw / 2.0) / img_w
            yc = (y1 + bh / 2.0) / img_h
            boxes.append([cid, xc, yc, bw / img_w, bh / img_h])
        return (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 5), dtype=np.float32)
        )

    # Case 3: Standard YOLO normalized format `<class_id> <xc> <yc> <w> <h>`
    boxes = []
    for ln in non_comment:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid, xc, yc, bw, bh = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        except ValueError:
            continue
        if bw <= 0 or bh <= 0:
            continue
        boxes.append([cid, xc, yc, bw, bh])

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)

    labels = np.array(boxes, dtype=np.float32)
    labels[:, 1:5] = np.clip(labels[:, 1:5], 0.0, 1.0)
    return np.unique(labels, axis=0)
