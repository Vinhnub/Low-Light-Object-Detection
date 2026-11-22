"""
tools/visualize_yolov11.py

Visualize intermediate feature maps, spatial attention heatmaps, Backbone/FPN/PAN
representations, anchor-free decoupled Detect head maps (Class Confidence & DFL Box
Certainty), and low-light edge enhancement internals for both:
  1. YOLOv11 Base (24-layer standard architecture on branch `main`)
  2. Enhanced YOLOv11 (27-layer `EdgeExtractor` + `EdgeSAB` architecture on branch `develop-yolov11`)
  3. Side-by-Side Comparison Mode (Base vs Enhanced on identical inputs)

Usage Examples:
  # 1. Visualize YOLOv11 Base (works on both `main` and `develop-yolov11`)
  python tools/visualize_yolov11.py --weights model_results/yolov11s/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65

  # 2. Visualize Enhanced YOLOv11 (`develop-yolov11`) including EdgeExtractor & EdgeSAB internals
  python tools/visualize_yolov11.py --weights model_results/yolov11s-edge-plus-sab-ffn/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65 --sublayers

  # 3. Side-by-side comparison of YOLOv11 Base vs Enhanced YOLOv11
  python tools/visualize_yolov11.py ^
      --base_weights model_results/yolov11s/best_mAP50.pth ^
      --enhanced_weights model_results/yolov11s-edge-plus-sab-ffn/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.util import (  # noqa: E402
    DARKFACE_CLASSES,
    EXDARK_CLASSES,
    img2label_paths,
    is_image_file,
    letterbox,
    parse_label_file,
    xywhn2xyxy,
)
from net import (  # noqa: E402
    Yolov11,
    get_model_family,
    infer_model_from_checkpoint,
    load_yolo_checkpoint,
    non_max_suppression,
    normalize_model_name,
)
from net.Yolov11.utils import (  # noqa: E402
    _register_ultralytics_unpickle_aliases,
    de_parallel,
    scale_boxes,
)
from net.Yolov11.yolo import YOLOV11_SCALES, DetectionModel  # noqa: E402


# ==================================================================================================
# 1. Layer Semantic Metadata & Architecture Config Builders
# ==================================================================================================

YOLOV11_BASE_LAYER_META: Dict[int, Tuple[str, str]] = {
    0: ("Backbone", "L00_Stem_Conv3x3_P1_2"),
    1: ("Backbone", "L01_Conv3x3_P2_4"),
    2: ("Backbone", "L02_C3k2_P2_4"),
    3: ("Backbone", "L03_Conv3x3_P3_8"),
    4: ("Backbone", "L04_C3k2_P3_8_SkipP3"),
    5: ("Backbone", "L05_Conv3x3_P4_16"),
    6: ("Backbone", "L06_C3k2_P4_16_SkipP4"),
    7: ("Backbone", "L07_Conv3x3_P5_32"),
    8: ("Backbone", "L08_C3k2_P5_32"),
    9: ("Backbone", "L09_SPPF_P5_32"),
    10: ("Backbone_Attn", "L10_C2PSA_P5_32_SkipP5"),
    11: ("FPN_Neck", "L11_FPN_Upsample_P5_P4"),
    12: ("FPN_Neck", "L12_FPN_Concat_P4"),
    13: ("FPN_Neck", "L13_FPN_C3k2_P4"),
    14: ("FPN_Neck", "L14_FPN_Upsample_P4_P3"),
    15: ("FPN_Neck", "L15_FPN_Concat_P3"),
    16: ("PAN_HeadIn", "L16_NeckOut_N3_P3_8"),
    17: ("PAN_Neck", "L17_PAN_Down_N3"),
    18: ("PAN_Neck", "L18_PAN_Concat_P4"),
    19: ("PAN_HeadIn", "L19_NeckOut_N4_P4_16"),
    20: ("PAN_Neck", "L20_PAN_Down_N4"),
    21: ("PAN_Neck", "L21_PAN_Concat_P5"),
    22: ("PAN_HeadIn", "L22_NeckOut_N5_P5_32"),
    23: ("Head", "L23_Detect_P3_P4_P5"),
}

YOLOV11_ENHANCED_SKIP_P3_LAYER_META: Dict[int, Tuple[str, str]] = {
    **YOLOV11_BASE_LAYER_META,
    15: ("EdgeSAB_HeadIn", "L15_EdgeSAB_Skip_B3_P3_8"),
    16: ("FPN_Neck", "L16_FPN_Concat_P3"),
    17: ("PAN_HeadIn", "L17_NeckOut_N3_P3_8"),
    18: ("PAN_Neck", "L18_PAN_Down_N3"),
    19: ("PAN_Neck", "L19_PAN_Concat_P4"),
    20: ("PAN_HeadIn", "L20_NeckOut_N4_P4_16"),
    21: ("PAN_Neck", "L21_PAN_Down_N4"),
    22: ("PAN_Neck", "L22_PAN_Concat_P5"),
    23: ("PAN_HeadIn", "L23_NeckOut_N5_P5_32"),
    24: ("Head", "L24_Detect_P3_P4_P5"),
}

YOLOV11_ENHANCED_P3_INLINE_LAYER_META: Dict[int, Tuple[str, str]] = {
    **YOLOV11_BASE_LAYER_META,
    16: ("FPN_Neck", "L16_FPN_C3k2_N3_P3_8"),
    17: ("EdgeSAB_HeadIn", "L17_EdgeSAB_Inline_N3_P3_8"),
    18: ("PAN_Neck", "L18_PAN_Down_N3"),
    19: ("PAN_Neck", "L19_PAN_Concat_P4"),
    20: ("PAN_HeadIn", "L20_NeckOut_N4_P4_16"),
    21: ("PAN_Neck", "L21_PAN_Down_N4"),
    22: ("PAN_Neck", "L22_PAN_Concat_P5"),
    23: ("PAN_HeadIn", "L23_NeckOut_N5_P5_32"),
    24: ("Head", "L24_Detect_P3_P4_P5"),
}

YOLOV11_ENHANCED_P3_LAYER_META: Dict[int, Tuple[str, str]] = {
    **YOLOV11_BASE_LAYER_META,
    16: ("PAN_Neck", "L16_PAN_C3k2_N3_P3_8"),
    19: ("PAN_HeadIn", "L19_NeckOut_N4_P4_16"),
    22: ("PAN_HeadIn", "L22_NeckOut_N5_P5_32"),
    23: ("EdgeSAB_HeadIn", "L23_EdgeSAB_N3_P3_8"),
    24: ("Head", "L24_Detect_P3_P4_P5"),
}

YOLOV11_ENHANCED_LAYER_META: Dict[int, Tuple[str, str]] = {
    **YOLOV11_BASE_LAYER_META,
    16: ("PAN_Neck", "L16_PAN_C3k2_N3_P3_8"),
    19: ("PAN_Neck", "L19_PAN_C3k2_N4_P4_16"),
    22: ("PAN_Neck", "L22_PAN_C3k2_N5_P5_32"),
    23: ("EdgeSAB_HeadIn", "L23_EdgeSAB_N3_P3_8"),
    24: ("EdgeSAB_HeadIn", "L24_EdgeSAB_N4_P4_16"),
    25: ("EdgeSAB_HeadIn", "L25_EdgeSAB_N5_P5_32"),
    26: ("Head", "L26_Detect_P3_P4_P5"),
}

STAGE_ACCENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "EdgeExtractor": (60, 220, 255),
    "Backbone": (50, 170, 255),
    "Backbone_Attn": (40, 215, 255),
    "FPN_Neck": (210, 110, 255),
    "PAN_Neck": (230, 210, 80),
    "PAN_HeadIn": (140, 230, 80),
    "EdgeSAB_HeadIn": (255, 80, 220),
    "Head": (90, 110, 255),
}

COLORMAP_TABLE: Dict[str, int] = {
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
}

BOX_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 255), (255, 128, 0), (50, 255, 50), (0, 165, 255),
    (255, 0, 255), (255, 255, 0), (0, 75, 255), (180, 105, 255),
    (0, 215, 255), (226, 43, 138), (128, 255, 0), (0, 128, 255),
]


def _parse_variant(model_name: str) -> str:
    variant = model_name.lower().replace("yolov11", "").replace("yolo11", "")
    if variant not in YOLOV11_SCALES:
        raise ValueError(
            f"Unsupported YOLOv11 variant '{variant}' from '{model_name}'. "
            f"Choose from {list(YOLOV11_SCALES.keys())}."
        )
    return variant


def _yolov11_backbone_cfg() -> List[List[Any]]:
    return [
        [-1, 1, "Conv", [64, 3, 2]],               # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],              # 1-P2/4
        [-1, 2, "C3k2", [256, False, 0.25]],       # 2
        [-1, 1, "Conv", [256, 3, 2]],              # 3-P3/8
        [-1, 2, "C3k2", [512, False, 0.25]],       # 4
        [-1, 1, "Conv", [512, 3, 2]],              # 5-P4/16
        [-1, 2, "C3k2", [512, True]],              # 6
        [-1, 1, "Conv", [1024, 3, 2]],             # 7-P5/32
        [-1, 2, "C3k2", [1024, True]],             # 8
        [-1, 1, "SPPF", [1024, 5]],                # 9
        [-1, 2, "C2PSA", [1024]],                  # 10
    ]


def _yolov11_fpn_pan_cfg() -> List[List[Any]]:
    return [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 11
        [[-1, 6], 1, "Concat", [1]],                   # 12 cat backbone P4
        [-1, 2, "C3k2", [512, False]],                 # 13
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 14
        [[-1, 4], 1, "Concat", [1]],                   # 15 cat backbone P3
        [-1, 2, "C3k2", [256, False]],                 # 16 (P3/8-small)
        [-1, 1, "Conv", [256, 3, 2]],                  # 17
        [[-1, 13], 1, "Concat", [1]],                  # 18 cat head P4
        [-1, 2, "C3k2", [512, False]],                 # 19 (P4/16-medium)
        [-1, 1, "Conv", [512, 3, 2]],                  # 20
        [[-1, 10], 1, "Concat", [1]],                  # 21 cat head P5
        [-1, 2, "C3k2", [1024, True]],                 # 22 (P5/32-large)
    ]


def get_yolov11_base_cfg_dict(variant: str = "s", nc: int = 80) -> Dict[str, Any]:
    """Return canonical 24-layer YOLOv11 Base config (matching branch `main`)."""
    v = _parse_variant(variant)
    return {
        "nc": nc,
        "scale": v,
        "scales": {k: list(vals) for k, vals in YOLOV11_SCALES.items()},
        "backbone": _yolov11_backbone_cfg(),
        "head": _yolov11_fpn_pan_cfg() + [
            [[16, 19, 22], 1, "Detect", ["nc"]],       # 23 Detect(P3, P4, P5)
        ],
    }


def get_yolov11_enhanced_skip_p3_cfg_dict(variant: str = "s", nc: int = 80) -> Dict[str, Any]:
    """Return 25-layer Enhanced YOLOv11 config with lateral Skip EdgeSAB on Backbone P3/8 (L4) before Concat."""
    v = _parse_variant(variant)
    return {
        "nc": nc,
        "scale": v,
        "scales": {k: list(vals) for k, vals in YOLOV11_SCALES.items()},
        "backbone": _yolov11_backbone_cfg(),
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 11
            [[-1, 6], 1, "Concat", [1]],                   # 12 cat backbone P4
            [-1, 2, "C3k2", [512, False]],                 # 13
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 14
            [4, 1, "EdgeSAB", [4, False, 2]],              # 15 (Backbone P3/8 Skip EdgeSAB before Concat)
            [[14, -1], 1, "Concat", [1]],                  # 16 cat upsample P4 & enriched backbone P3
            [-1, 2, "C3k2", [256, False]],                 # 17 (FPN P3/8-small -> Detect P3 & PAN)
            [-1, 1, "Conv", [256, 3, 2]],                  # 18 (PAN Down from P3)
            [[-1, 13], 1, "Concat", [1]],                  # 19 cat head P4
            [-1, 2, "C3k2", [512, False]],                 # 20 (P4/16-medium)
            [-1, 1, "Conv", [512, 3, 2]],                  # 21 (PAN Down from P4)
            [[-1, 10], 1, "Concat", [1]],                  # 22 cat head P5
            [-1, 2, "C3k2", [1024, True]],                 # 23 (P5/32-large)
            [[17, 20, 23], 1, "Detect", ["nc"]],           # 24 Detect(P3, P4, P5)
        ],
    }


def get_yolov11_enhanced_p3_inline_cfg_dict(variant: str = "s", nc: int = 80) -> Dict[str, Any]:
    """Return 25-layer Enhanced YOLOv11 config with inline EdgeSAB on P3/8 feeding Detect P3 & PAN."""
    v = _parse_variant(variant)
    return {
        "nc": nc,
        "scale": v,
        "scales": {k: list(vals) for k, vals in YOLOV11_SCALES.items()},
        "backbone": _yolov11_backbone_cfg(),
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 11
            [[-1, 6], 1, "Concat", [1]],                   # 12 cat backbone P4
            [-1, 2, "C3k2", [512, False]],                 # 13
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 14
            [[-1, 4], 1, "Concat", [1]],                   # 15 cat backbone P3
            [-1, 2, "C3k2", [256, False]],                 # 16 (FPN P3/8-small)
            [-1, 1, "EdgeSAB", [4, False, 2]],             # 17 (Inline EdgeSAB P3/8)
            [-1, 1, "Conv", [256, 3, 2]],                  # 18 (PAN Down from EdgeSAB P3)
            [[-1, 13], 1, "Concat", [1]],                  # 19 cat head P4
            [-1, 2, "C3k2", [512, False]],                 # 20 (P4/16-medium)
            [-1, 1, "Conv", [512, 3, 2]],                  # 21 (PAN Down from P4)
            [[-1, 10], 1, "Concat", [1]],                  # 22 cat head P5
            [-1, 2, "C3k2", [1024, True]],                 # 23 (P5/32-large)
            [[17, 20, 23], 1, "Detect", ["nc"]],           # 24 Detect(P3, P4, P5)
        ],
    }


def get_yolov11_enhanced_p3_cfg_dict(variant: str = "s", nc: int = 80) -> Dict[str, Any]:
    """Return 25-layer Enhanced YOLOv11 config with pre-head EdgeSAB on P3/8 only (`develop-yolov11-p3-edgesab`)."""
    v = _parse_variant(variant)
    return {
        "nc": nc,
        "scale": v,
        "scales": {k: list(vals) for k, vals in YOLOV11_SCALES.items()},
        "backbone": _yolov11_backbone_cfg(),
        "head": _yolov11_fpn_pan_cfg() + [
            [16, 1, "EdgeSAB", [4, False, 2]],         # 23 (P3/8 EdgeSAB self-attn)
            [[23, 19, 22], 1, "Detect", ["nc"]],       # 24 Detect(P3, P4, P5)
        ],
    }


def get_yolov11_enhanced_cfg_dict(variant: str = "s", nc: int = 80) -> Dict[str, Any]:
    """Return 27-layer Enhanced YOLOv11 config with pre-head EdgeSAB (branch `develop-yolov11`)."""
    v = _parse_variant(variant)
    return {
        "nc": nc,
        "scale": v,
        "scales": {k: list(vals) for k, vals in YOLOV11_SCALES.items()},
        "backbone": _yolov11_backbone_cfg(),
        "head": _yolov11_fpn_pan_cfg() + [
            [16, 1, "EdgeSAB", [4, False, 2]],         # 23 (P3/8 EdgeSAB self-attn)
            [19, 1, "EdgeSAB", [4, False, 2]],         # 24 (P4/16 EdgeSAB self-attn)
            [22, 1, "EdgeSAB", [4, False, 2]],         # 25 (P5/32 EdgeSAB self-attn)
            [[23, 24, 25], 1, "Detect", ["nc"]],       # 26 Detect(P3, P4, P5)
        ],
    }


# ==================================================================================================
# 2. CLI Argument Parsing
# ==================================================================================================

def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"yes", "true", "t", "y", "1"}:
        return True
    if s in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLOv11 Base & Enhanced YOLOv11 (EdgeExtractor + EdgeSAB) layers and Detect heads."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to single YOLOv11 checkpoint (.pth or .pt) for single-model mode.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "base", "enhanced"],
        help="Architecture type for --weights ('auto', 'base', or 'enhanced').",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default="",
        help="Path to YOLOv11 Base checkpoint for side-by-side comparison mode.",
    )
    parser.add_argument(
        "--enhanced_weights",
        type=str,
        default="",
        help="Path to Enhanced YOLOv11 checkpoint for side-by-side comparison mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional YOLOv11 variant override (yolov11n/s/m/l/x) for --weights.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="Optional YOLOv11 variant override for --base_weights.",
    )
    parser.add_argument(
        "--enhanced_model",
        type=str,
        default="",
        help="Optional YOLOv11 variant override for --enhanced_weights.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional YAML config path override (single-model mode).",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        help="Number of classes override. Auto-inferred from checkpoint if omitted.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Class names ('exdark', 'darkface', comma-separated list, or .yaml path).",
    )
    parser.add_argument(
        "--modular",
        type=str2bool,
        default=None,
        help="Use modular YOLOv11 (backbone/neck/head). Auto-detected from checkpoint keys if omitted.",
    )
    parser.add_argument(
        "--compare_only",
        action="store_true",
        help="In comparison mode, skip saving per-layer channel sheets inside base/layers and enhanced/layers.",
    )
    parser.add_argument(
        "--no_edge_raw",
        action="store_true",
        help="Skip saving individual PNG files under edge/ (composite 05/06 sheets and edge_stats.json are still saved).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Input image file or directory. Defaults to dataset/ExDark/images/test (or val) if present.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization",
        help="Root directory to save feature visualizations (default: visualization).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference input resolution for letterbox (default: 640).",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Max number of images to visualize when --source is a directory (default: 1, -1 for all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly sample images when --source is a directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="Confidence threshold for predicted bounding boxes overlay (default: 0.25).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=15,
        help="Number of top active channels to display per layer grid (default: 15).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layer indices for detailed channel sheets (e.g. '0,4,10,16,23' or 'all' or 'none').",
    )
    parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Also capture internal sub-modules inside C3k2, SPPF, C2PSA, EdgeSAB, and Detect.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=list(COLORMAP_TABLE.keys()),
        help="OpenCV colormap for heatmaps (default: turbo).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Heatmap overlay opacity in [0.0, 1.0] (default: 0.48).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.5,
        help="Upper percentile clipping before heatmap normalization (100.0 = raw min-max, default: 99.5).",
    )
    parser.add_argument(
        "--vis_gamma",
        type=float,
        default=1.0,
        help="Display-only gamma (<1.0 brightens background underlay without altering model input; default: 1.0).",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=420,
        help="Display panel width in pixels for summary tiles (default: 420).",
    )
    args = parser.parse_args()

    has_pair = bool(args.base_weights.strip() or args.enhanced_weights.strip())
    if has_pair and not (args.base_weights.strip() and args.enhanced_weights.strip()):
        parser.error("Comparison mode requires BOTH --base_weights and --enhanced_weights.")
    if not args.weights.strip() and not has_pair:
        parser.error("Provide --weights for single-model mode, or both --base_weights and --enhanced_weights for comparison mode.")
    return args


# ==================================================================================================
# 3. Model Loading & Class Resolution
# ==================================================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_class_names(classes_arg: str, nc: int) -> List[str]:
    if classes_arg:
        low = classes_arg.strip().lower()
        if low == "exdark":
            return list(EXDARK_CLASSES)
        if low == "darkface":
            return list(DARKFACE_CLASSES)
        p = Path(classes_arg)
        if p.is_file():
            try:
                import yaml

                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data:
                    names = data["names"]
                    if isinstance(names, dict):
                        return [str(names[k]) for k in sorted(names.keys())]
                    if isinstance(names, list):
                        return [str(x) for x in names]
            except Exception as exc:
                print(f"[Warn] Could not parse classes YAML '{classes_arg}': {exc}")
        return [c.strip() for c in classes_arg.split(",") if c.strip()]

    if nc == len(EXDARK_CLASSES):
        return list(EXDARK_CLASSES)
    if nc == len(DARKFACE_CLASSES):
        return list(DARKFACE_CLASSES)
    return [f"cls_{i}" for i in range(nc)]


def _extract_raw_state_dict(ckpt_meta: Any) -> Dict[str, Any]:
    if not isinstance(ckpt_meta, dict):
        return {}
    if isinstance(ckpt_meta.get("state_dict"), dict):
        return ckpt_meta["state_dict"]
    if isinstance(ckpt_meta.get("model_state_dict"), dict):
        return ckpt_meta["model_state_dict"]
    if ckpt_meta.get("ema") is not None:
        ema_obj = ckpt_meta["ema"]
        return ema_obj.state_dict() if hasattr(ema_obj, "state_dict") else dict(ema_obj)
    if ckpt_meta.get("model") is not None:
        m_obj = ckpt_meta["model"]
        return m_obj.state_dict() if hasattr(m_obj, "state_dict") else dict(m_obj)
    return {}


def _infer_nc_from_state_dict(raw_sd: Dict[str, Any]) -> Optional[int]:
    for key, val in raw_sd.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean.endswith(("cv3.0.2.bias", "cv3.0.2.weight")) and isinstance(val, torch.Tensor):
            return int(val.shape[0])
    return None


def _detect_ckpt_properties(raw_sd: Dict[str, Any]) -> Tuple[bool, str, str]:
    clean_keys = [k[7:] if k.startswith("module.") else k for k in raw_sd.keys()]
    clean_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
    is_modular = any(k.startswith("backbone.") for k in clean_keys)
    is_27 = any(k.startswith(("model.26.", "neck.attn_p4.")) for k in clean_keys)
    has_det_24 = any(k.startswith(("model.24.cv", "model.24.dfl")) for k in clean_keys)
    is_25_skip_p3 = (
        has_det_24
        and any(k.startswith(("model.15.q.", "model.15.gamma", "model.15.edge_gate.")) for k in clean_keys)
        and not is_27
    )
    is_25_inline = (
        has_det_24
        and any(k.startswith(("model.17.q.", "model.17.gamma", "model.17.edge_gate.")) for k in clean_keys)
        and not is_27
    )
    is_25_prehead = (
        has_det_24
        and any(k.startswith(("model.23.q.", "model.23.gamma", "model.23.edge_gate.")) for k in clean_keys)
        and not is_27
    )
    is_mod_p3 = (
        any(k.startswith("neck.attn_p3.") for k in clean_keys)
        and not any(k.startswith("neck.attn_p4.") for k in clean_keys)
        and not is_27
    )
    is_mod_skip_p3 = False
    if is_mod_p3:
        attn_g = clean_sd.get("neck.attn_p3.gamma")
        p3_out_w = clean_sd.get("neck.c3k2_p3.cv2.conv.weight")
        if isinstance(attn_g, torch.Tensor) and isinstance(p3_out_w, torch.Tensor):
            is_mod_skip_p3 = int(attn_g.shape[1]) != int(p3_out_w.shape[0])
    is_enhanced = is_27 or is_25_skip_p3 or is_25_inline or is_25_prehead or is_mod_p3 or any(
        k.startswith(("edge_extractor.", "model.15.q.", "model.17.q.", "model.23.q.", "model.23.edge_gate.", "model.23.gamma"))
        for k in clean_keys
    )
    if is_27:
        enh_mode = "full"
    elif is_25_skip_p3 or is_mod_skip_p3:
        enh_mode = "p3_skip"
    elif is_25_inline or is_mod_p3:
        enh_mode = "p3_inline"
    elif is_25_prehead:
        enh_mode = "p3"
    elif is_enhanced:
        enh_mode = "full"
    else:
        enh_mode = "none"
    return is_modular, ("enhanced" if is_enhanced else "base"), enh_mode


def _load_modular_base_into_seq24(model: nn.Module, raw_sd: Dict[str, Any], weights_path: str) -> None:
    """Map modular 24-layer YOLOv11 (`backbone.*/neck.*/head.*`) weights into 24-layer DetectionModel."""
    mod_to_seq24 = {
        "backbone.stem.": "model.0.",
        "backbone.stage1.0.": "model.1.",
        "backbone.stage1.1.": "model.2.",
        "backbone.stage2.0.": "model.3.",
        "backbone.stage2.1.": "model.4.",
        "backbone.stage3.0.": "model.5.",
        "backbone.stage3.1.": "model.6.",
        "backbone.stage4.0.": "model.7.",
        "backbone.stage4.1.": "model.8.",
        "backbone.stage4.2.": "model.9.",
        "backbone.stage4.3.": "model.10.",
        "neck.c3k2_p4.": "model.13.",
        "neck.c3k2_p3.": "model.16.",
        "neck.down_p3.": "model.17.",
        "neck.c3k2_n4.": "model.19.",
        "neck.down_p4.": "model.20.",
        "neck.c3k2_n5.": "model.22.",
        "head.": "model.23.",
    }
    target = de_parallel(model)
    model_dict = target.state_dict()
    mapped = {}
    for k, v in raw_sd.items():
        name = k[7:] if k.startswith("module.") else k
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            v = v.float()
        for src_p, dst_p in mod_to_seq24.items():
            if name.startswith(src_p):
                cand = dst_p + name[len(src_p) :]
                if cand in model_dict and model_dict[cand].shape == v.shape:
                    mapped[cand] = v
                    break
    target.load_state_dict(mapped, strict=False)
    print(f"===> Loaded {len(mapped)}/{len(model_dict)} modular base tensors into 24-layer DetectionModel from '{weights_path}'")


def load_yolov11_model(
    weights_path: str,
    arch_override: str = "auto",
    model_override: str = "",
    cfg_override: str = "",
    nc_override: Optional[int] = None,
    modular_override: Optional[bool] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, str, int, Dict[str, Any]]:
    """Build exact YOLOv11 Base (24-layer) or Enhanced YOLOv11 (27-layer) architecture and load weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    _register_ultralytics_unpickle_aliases()
    ckpt_meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    raw_sd = _extract_raw_state_dict(ckpt_meta)
    auto_modular, auto_arch, auto_enh_mode = _detect_ckpt_properties(raw_sd)

    arch = auto_arch if arch_override == "auto" else arch_override
    is_modular = auto_modular if modular_override is None else bool(modular_override)

    model_name = (
        normalize_model_name(model_override.strip())
        if model_override.strip()
        else (infer_model_from_checkpoint(ckpt_meta, fallback_path=weights_path) or "yolov11s")
    )
    if get_model_family(model_name) != "yolov11":
        raise ValueError(
            f"tools/visualize_yolov11.py only supports YOLOv11 checkpoints, got '{model_name}'."
        )
    variant = _parse_variant(model_name)

    nc = nc_override
    if nc is None and isinstance(ckpt_meta, dict):
        if ckpt_meta.get("num_classes") is not None:
            nc = int(ckpt_meta["num_classes"])
        elif hasattr(ckpt_meta.get("model"), "nc"):
            nc = int(ckpt_meta["model"].nc)
        elif isinstance(getattr(ckpt_meta.get("model"), "yaml", None), dict):
            nc = ckpt_meta["model"].yaml.get("nc")
    if nc is None:
        nc = _infer_nc_from_state_dict(raw_sd)
    nc = int(nc or 12)

    has_enhanced_code = hasattr(Yolov11, "SAB") or hasattr(Yolov11, "EdgeSAB")
    cfg_path = cfg_override.strip() if (cfg_override and os.path.exists(cfg_override.strip())) else ""

    enhanced_mode = "none"
    if arch == "enhanced":
        if not has_enhanced_code:
            raise RuntimeError(
                "Enhanced YOLOv11 (EdgeExtractor + EdgeSAB) is not available on the current branch. "
                "Please switch to an enhanced YOLOv11 branch (e.g. 'develop-yolov11', 'develop-yolov11-p3-edgesab', "
                "or 'develop-yolov11-skip-p3-edgesab') to visualize Enhanced YOLOv11."
            )
        branch_head = Yolov11.get_yolov11_cfg_dict(variant, nc).get("head", [])
        branch_is_p3_skip = len(branch_head) == 14 and len(branch_head) > 4 and branch_head[4][2] in {"EdgeSAB", "SAB"}
        branch_is_p3_inline = len(branch_head) == 14 and len(branch_head) > 6 and branch_head[6][2] in {"EdgeSAB", "SAB"}
        branch_is_p3_prehead = len(branch_head) == 14 and not branch_is_p3_skip and not branch_is_p3_inline
        branch_mode = (
            "p3_skip"
            if branch_is_p3_skip
            else ("p3_inline" if branch_is_p3_inline else ("p3" if branch_is_p3_prehead else "full"))
        )

        if auto_enh_mode in {"p3_skip", "p3_inline", "p3", "full"}:
            enhanced_mode = auto_enh_mode
        else:
            enhanced_mode = branch_mode

        if cfg_path:
            model = DetectionModel(cfg=cfg_path, ch=3, nc=nc)
            sab_idxs = [i for i, m in enumerate(model.model) if type(m).__name__ in {"EdgeSAB", "SAB"}]
            if sab_idxs == [15]:
                enhanced_mode = "p3_skip"
            elif sab_idxs == [17]:
                enhanced_mode = "p3_inline"
            elif len(sab_idxs) == 1:
                enhanced_mode = "p3"
            elif len(sab_idxs) >= 2:
                enhanced_mode = "full"
            else:
                enhanced_mode = "none"
        elif is_modular and (enhanced_mode == branch_mode):
            model = Yolov11.YOLOv11(num_classes=nc, variant=variant, in_channels=3)
        elif enhanced_mode == "p3_skip":
            model = DetectionModel(cfg=get_yolov11_enhanced_skip_p3_cfg_dict(variant, nc), ch=3, nc=nc)
        elif enhanced_mode == "p3_inline":
            model = DetectionModel(cfg=get_yolov11_enhanced_p3_inline_cfg_dict(variant, nc), ch=3, nc=nc)
        elif enhanced_mode == "p3":
            model = DetectionModel(cfg=get_yolov11_enhanced_p3_cfg_dict(variant, nc), ch=3, nc=nc)
        else:
            model = DetectionModel(cfg=get_yolov11_enhanced_cfg_dict(variant, nc), ch=3, nc=nc)
        load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)
    else:
        if cfg_path:
            model = DetectionModel(cfg=cfg_path, ch=3, nc=nc)
            if hasattr(model, "edge_extractor") and not any(
                type(m).__name__ in {"EdgeSAB", "SAB"} for m in model.model
            ):
                model.edge_extractor = None
            load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)
        elif is_modular and not has_enhanced_code:
            model = Yolov11.YOLOv11(num_classes=nc, variant=variant, in_channels=3)
            load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)
        else:
            model = DetectionModel(cfg=get_yolov11_base_cfg_dict(variant, nc), ch=3, nc=nc)
            if hasattr(model, "edge_extractor"):
                # Disable EdgeExtractor on 24-layer Base so forward and state_dict match `main` 100%
                model.edge_extractor = None
            if auto_modular:
                _load_modular_base_into_seq24(model, raw_sd, weights_path)
            else:
                load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)

    ckpt_names: List[str] = []
    if isinstance(ckpt_meta, dict) and hasattr(ckpt_meta.get("model"), "names"):
        raw_names = ckpt_meta["model"].names
        if isinstance(raw_names, dict):
            ckpt_names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
        elif isinstance(raw_names, (list, tuple)):
            ckpt_names = [str(x) for x in raw_names]

    num_layers = (
        len(model.model)
        if hasattr(model, "model")
        else (25 if enhanced_mode in {"p3_skip", "p3_inline", "p3"} else (27 if arch == "enhanced" else 24))
    )
    meta_info = {
        "epoch": ckpt_meta.get("epoch", "?") if isinstance(ckpt_meta, dict) else "?",
        "arch": arch,
        "enhanced_mode": enhanced_mode,
        "is_modular": is_modular,
        "num_layers": num_layers,
        "ckpt_names": ckpt_names,
    }
    model = model.to(device).eval()
    return model, model_name, nc, meta_info


def resolve_model_class_names(args_classes: str, meta_info: Dict[str, Any], nc: int) -> List[str]:
    if not args_classes and meta_info.get("ckpt_names") and len(meta_info["ckpt_names"]) == nc:
        if not all(str(n).isdigit() for n in meta_info["ckpt_names"]):
            return list(meta_info["ckpt_names"])
    return resolve_class_names(args_classes, nc)


# ==================================================================================================
# 4. Input Resolution, Preprocessing & Ground Truth Loading
# ==================================================================================================

def collect_input_images(source_arg: str, max_num: int, shuffle: bool) -> List[str]:
    if source_arg:
        p = Path(source_arg)
        if p.is_file():
            return [str(p)]
        if p.is_dir():
            imgs = sorted(str(f) for f in p.iterdir() if f.is_file() and is_image_file(f.name))
        else:
            raise FileNotFoundError(f"Source path not found: {source_arg}")
    else:
        candidates = [
            PROJECT_ROOT / "dataset" / "ExDark" / "images" / "test",
            PROJECT_ROOT / "dataset" / "ExDark" / "images" / "val",
            PROJECT_ROOT / "dataset" / "DarkFace" / "image",
        ]
        imgs = []
        for cand in candidates:
            if cand.is_dir():
                imgs = sorted(str(f) for f in cand.iterdir() if f.is_file() and is_image_file(f.name))
                if imgs:
                    print(f"===> Auto-selected source directory: {cand}")
                    break
        if not imgs:
            raise FileNotFoundError(
                "No --source specified and no default dataset images found in dataset/ExDark or dataset/DarkFace."
            )

    if not imgs:
        raise FileNotFoundError(f"No supported images found in '{source_arg}'.")

    if shuffle:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(imgs))
        imgs = [imgs[i] for i in idx]

    if max_num > 0:
        imgs = imgs[:max_num]
    return imgs


def preprocess_image(
    img_bgr: np.ndarray, img_size: int = 640, stride: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Apply letterbox preprocessing and return (tensor, unpad_wh, crop_box, ratio_pad)."""
    h0, w0 = img_bgr.shape[:2]
    im_lb, ratio, (dw, dh) = letterbox(
        img_bgr,
        new_shape=(img_size, img_size),
        auto=False,
        scaleup=True,
        stride=stride,
    )
    new_unpad_w = int(round(w0 * ratio[0]))
    new_unpad_h = int(round(h0 * ratio[1]))
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    crop_box = (top, left, top + new_unpad_h, left + new_unpad_w)

    rgb = im_lb[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0), (new_unpad_w, new_unpad_h), crop_box, (ratio, (dw, dh))


def load_optional_gt(
    img_path: str,
    orig_hw: Tuple[int, int],
    class_names: List[str],
) -> np.ndarray:
    """Attempt to locate and parse matching ground truth labels `[x1, y1, x2, y2, 1.0, cls_id]`."""
    lbl_path = img2label_paths([img_path])[0]
    if not os.path.exists(lbl_path):
        return np.zeros((0, 6), dtype=np.float32)
    h0, w0 = orig_hw
    class_to_id = {c: i for i, c in enumerate(class_names)}
    parsed = parse_label_file(lbl_path, img_w=w0, img_h=h0, class_to_id=class_to_id)
    if len(parsed) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    boxes = xywhn2xyxy(parsed[:, 1:5], w=float(w0), h=float(h0))
    confs = np.ones((len(parsed), 1), dtype=np.float32)
    cls_ids = parsed[:, 0:1].astype(np.float32)
    return np.hstack((boxes, confs, cls_ids)).astype(np.float32)


# ==================================================================================================
# 5. Forward Hook Feature Recorder & Detect / Edge Introspection
# ==================================================================================================

class YOLOv11FeatureRecorder:
    """Registers PyTorch forward hooks across YOLOv11 Base or Enhanced layers & sub-modules."""

    def __init__(self, model: nn.Module, capture_sublayers: bool = False):
        self.model = model
        self.capture_sublayers = capture_sublayers
        self.handles: List[Any] = []
        self.records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.edgesab_modules: "OrderedDict[str, Tuple[int, nn.Module]]" = OrderedDict()
        self.edgesab_inputs: Dict[str, torch.Tensor] = {}
        self.edge_map: Optional[torch.Tensor] = None
        self._register()

    def _make_hook(self, key: str, stage: str, layer_idx: int, mod_type: str, is_sub: bool):
        def hook_fn(module, _inputs, output):
            if mod_type == "EdgeExtractor" and isinstance(output, torch.Tensor):
                self.edge_map = output.detach().float().cpu()
                self.records[key] = {
                    "key": key,
                    "stage": stage,
                    "layer_idx": layer_idx,
                    "mod_type": mod_type,
                    "is_sublayer": is_sub,
                    "tensor": self.edge_map,
                }
                return

            if isinstance(output, torch.Tensor):
                self.records[key] = {
                    "key": key,
                    "stage": stage,
                    "layer_idx": layer_idx,
                    "mod_type": mod_type,
                    "is_sublayer": is_sub,
                    "tensor": output.detach().float().cpu(),
                }
            elif "Detect" in key and isinstance(output, (list, tuple)):
                preds = output[1] if len(output) >= 2 else None
                if isinstance(preds, dict) and "one2many" in preds:
                    preds = preds["one2many"]
                if isinstance(preds, dict) and "boxes" in preds and "scores" in preds and "feats" in preds:
                    self.records[key] = {
                        "key": key,
                        "stage": stage,
                        "layer_idx": layer_idx,
                        "mod_type": mod_type,
                        "is_sublayer": False,
                        "detect_preds": {
                            "boxes": preds["boxes"].detach().float().cpu(),
                            "scores": preds["scores"].detach().float().cpu(),
                            "feat_shapes": [tuple(f.shape[2:]) for f in preds["feats"]],
                            "reg_max": int(getattr(module, "reg_max", 16)),
                            "nc": int(getattr(module, "nc", preds["scores"].shape[1])),
                        },
                    }
        return hook_fn

    def _make_edgesab_pre_hook(self, key: str):
        def pre_hook(_module, inputs):
            if inputs and len(inputs) > 0:
                x0 = inputs[0]
                if isinstance(x0, (list, tuple)) and x0:
                    x0 = x0[0]
                if isinstance(x0, torch.Tensor):
                    self.edgesab_inputs[key] = x0.detach().float().cpu()
        return pre_hook

    def _register(self) -> None:
        if getattr(self.model, "edge_extractor", None) is not None:
            ee = self.model.edge_extractor
            self.handles.append(
                ee.register_forward_hook(
                    self._make_hook("EdgeExtractor_PriorityMap", "EdgeExtractor", -1, "EdgeExtractor", False)
                )
            )

        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            sab_idxs = [i for i, m in enumerate(self.model.model) if type(m).__name__ in {"EdgeSAB", "SAB"}]
            if sab_idxs == [15]:
                meta_table = YOLOV11_ENHANCED_SKIP_P3_LAYER_META
            elif sab_idxs == [17]:
                meta_table = YOLOV11_ENHANCED_P3_INLINE_LAYER_META
            elif len(sab_idxs) == 1:
                meta_table = YOLOV11_ENHANCED_P3_LAYER_META
            elif len(sab_idxs) >= 2:
                meta_table = YOLOV11_ENHANCED_LAYER_META
            else:
                meta_table = YOLOV11_BASE_LAYER_META
            for idx, mod in enumerate(self.model.model):
                mod_type = type(mod).__name__
                stage, label = meta_table.get(idx, ("Layer", f"L{idx:02d}_{mod_type}"))
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(label, stage, idx, mod_type, False))
                )
                if mod_type in {"EdgeSAB", "SAB"}:
                    self.edgesab_modules[label] = (idx, mod)
                    self.handles.append(mod.register_forward_pre_hook(self._make_edgesab_pre_hook(label)))

                if self.capture_sublayers and mod_type in {
                    "C3k2", "C2f", "SPPF", "C2PSA", "EdgeSAB", "SAB", "Detect"
                }:
                    for sub_name, sub_mod in mod.named_children():
                        if mod_type == "Detect":
                            if sub_name not in {"cv2", "cv3"}:
                                continue
                            if isinstance(sub_mod, nn.ModuleList):
                                for s_idx, scale_sub in enumerate(sub_mod):
                                    sub_key = f"{label}.{sub_name}_P{s_idx + 3}"
                                    self.handles.append(
                                        scale_sub.register_forward_hook(
                                            self._make_hook(sub_key, f"{stage}_Sub", idx, type(scale_sub).__name__, True)
                                        )
                                    )
                            continue
                        sub_key = f"{label}.{sub_name}"
                        self.handles.append(
                            sub_mod.register_forward_hook(
                                self._make_hook(sub_key, f"{stage}_Sub", idx, type(sub_mod).__name__, True)
                            )
                        )
        else:
            # Modular YOLOv11 (`backbone`, `neck`, `head`)
            modular_points: List[Tuple[str, str, int, nn.Module]] = []
            if hasattr(self.model, "backbone"):
                bb = self.model.backbone
                modular_points.extend([
                    ("Backbone", "L00_Stem_P1_2", 0, bb.stem),
                    ("Backbone", "L02_Stage1_P2_4", 2, bb.stage1),
                    ("Backbone", "L04_Stage2_P3_8", 4, bb.stage2),
                    ("Backbone", "L06_Stage3_P4_16", 6, bb.stage3),
                    ("Backbone", "L09_SPPF_P5_32", 9, bb.stage4[2] if len(bb.stage4) >= 3 else bb.stage4),
                    ("Backbone_Attn", "L10_C2PSA_P5_32", 10, bb.stage4),
                ])
            has_p3_attn = False
            has_p4_attn = False
            has_p5_attn = False
            if hasattr(self.model, "neck"):
                nk = self.model.neck
                has_p3_attn = hasattr(nk, "attn_p3")
                has_p4_attn = hasattr(nk, "attn_p4")
                has_p5_attn = hasattr(nk, "attn_p5")
                is_p3_single_mod = has_p3_attn and not has_p4_attn and not has_p5_attn
                is_p3_skip_mod = (
                    is_p3_single_mod
                    and hasattr(nk.attn_p3, "dim")
                    and hasattr(nk, "out_channels")
                    and int(nk.attn_p3.dim) != int(nk.out_channels[0])
                )
                if is_p3_skip_mod:
                    modular_points.extend([
                        ("FPN_Neck", "L13_FPN_C3k2_P4", 13, nk.c3k2_p4),
                        ("EdgeSAB_HeadIn", "L15_EdgeSAB_Skip_B3_P3_8", 15, nk.attn_p3),
                        ("PAN_HeadIn", "L17_NeckOut_N3_P3_8", 17, nk.c3k2_p3),
                        ("PAN_Neck", "L18_PAN_Down_N3", 18, nk.down_p3),
                        ("PAN_HeadIn", "L20_NeckOut_N4_P4_16", 20, nk.c3k2_n4),
                        ("PAN_Neck", "L21_PAN_Down_N4", 21, nk.down_p4),
                        ("PAN_HeadIn", "L23_NeckOut_N5_P5_32", 23, nk.c3k2_n5),
                    ])
                elif is_p3_single_mod:
                    modular_points.extend([
                        ("FPN_Neck", "L13_FPN_C3k2_P4", 13, nk.c3k2_p4),
                        ("FPN_Neck", "L16_FPN_C3k2_N3_P3_8", 16, nk.c3k2_p3),
                        ("EdgeSAB_HeadIn", "L17_EdgeSAB_Inline_N3_P3_8", 17, nk.attn_p3),
                        ("PAN_Neck", "L18_PAN_Down_N3", 18, nk.down_p3),
                        ("PAN_HeadIn", "L20_NeckOut_N4_P4_16", 20, nk.c3k2_n4),
                        ("PAN_Neck", "L21_PAN_Down_N4", 21, nk.down_p4),
                        ("PAN_HeadIn", "L23_NeckOut_N5_P5_32", 23, nk.c3k2_n5),
                    ])
                else:
                    modular_points.extend([
                        ("FPN_Neck", "L13_FPN_C3k2_P4", 13, nk.c3k2_p4),
                        ("PAN_Neck" if has_p3_attn else "PAN_HeadIn", "L16_PAN_C3k2_N3_P3_8", 16, nk.c3k2_p3),
                        ("PAN_Neck", "L17_PAN_Down_N3", 17, nk.down_p3),
                        ("PAN_Neck" if has_p4_attn else "PAN_HeadIn", "L19_PAN_C3k2_N4_P4_16", 19, nk.c3k2_n4),
                        ("PAN_Neck", "L20_PAN_Down_N4", 20, nk.down_p4),
                        ("PAN_Neck" if has_p5_attn else "PAN_HeadIn", "L22_PAN_C3k2_N5_P5_32", 22, nk.c3k2_n5),
                    ])
                    if has_p3_attn:
                        modular_points.append(("EdgeSAB_HeadIn", "L23_EdgeSAB_N3_P3_8", 23, nk.attn_p3))
                    if has_p4_attn:
                        modular_points.append(("EdgeSAB_HeadIn", "L24_EdgeSAB_N4_P4_16", 24, nk.attn_p4))
                    if has_p5_attn:
                        modular_points.append(("EdgeSAB_HeadIn", "L25_EdgeSAB_N5_P5_32", 25, nk.attn_p5))
            if hasattr(self.model, "head"):
                det_idx = 26 if (has_p4_attn or has_p5_attn) else (24 if has_p3_attn else 23)
                modular_points.append(("Head", f"L{det_idx:02d}_Detect_P3_P4_P5", det_idx, self.model.head))

            for stage, label, idx, mod in modular_points:
                mod_type = type(mod).__name__
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(label, stage, idx, mod_type, False))
                )
                if mod_type in {"EdgeSAB", "SAB"}:
                    self.edgesab_modules[label] = (idx, mod)
                    self.handles.append(mod.register_forward_pre_hook(self._make_edgesab_pre_hook(label)))

    def clear(self) -> None:
        self.records.clear()
        self.edgesab_inputs.clear()
        self.edge_map = None

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compute_yolov11_detect_heatmaps(
    detect_preds: Dict[str, Any],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Decode anchor-free decoupled YOLOv11 Detect predictions (`boxes`, `scores`, `feat_shapes`)
    into per-scale spatial maps for [P3/8, P4/16, P5/32]:
      - `cls_max`: Peak sigmoid class probability across all classes in [0, 1]
      - `top1_*` / `top2_*`: Top-2 highest-peaking class spatial probability maps
      - `box_certainty`: DFL distribution certainty `1 - normalized_entropy(softmax(dfl))` in [0, 1]
      - `joint_map`: `cls_max * box_certainty`
    """
    boxes_all = detect_preds["boxes"][0]    # (4 * reg_max, sum(H*W))
    scores_all = detect_preds["scores"][0]  # (nc, sum(H*W))
    feat_shapes: List[Tuple[int, int]] = detect_preds["feat_shapes"]
    reg_max = int(detect_preds.get("reg_max", 16))
    nc = int(detect_preds.get("nc", scores_all.shape[0]))

    scale_names = ["P3/8 (Small)", "P4/16 (Medium)", "P5/32 (Large)"]
    results: List[Dict[str, Any]] = []
    offset = 0
    log_r = math.log(max(reg_max, 2))

    for idx, (hf, wf) in enumerate(feat_shapes):
        n_cells = int(hf * wf)
        sc_slice = scores_all[:, offset : offset + n_cells].view(nc, hf, wf)
        bx_slice = boxes_all[:, offset : offset + n_cells]
        offset += n_cells

        cls_probs = sc_slice.sigmoid()  # (nc, H, W)
        cls_max = cls_probs.max(dim=0).values.numpy()

        # Rank classes by peak spatial probability at this scale
        cls_peaks = cls_probs.view(nc, -1).max(dim=1).values
        order = torch.argsort(cls_peaks, descending=True).tolist()
        t1_id = int(order[0])
        t2_id = int(order[1]) if len(order) > 1 else t1_id
        t1_name = class_names[t1_id] if t1_id < len(class_names) else f"cls_{t1_id}"
        t2_name = class_names[t2_id] if t2_id < len(class_names) else f"cls_{t2_id}"

        if reg_max > 1 and bx_slice.shape[0] == 4 * reg_max:
            dfl_logits = bx_slice.view(4, reg_max, hf, wf)
            dfl_probs = F.softmax(dfl_logits, dim=1)
            entropy = -(dfl_probs * torch.log(dfl_probs.clamp(min=1e-8))).sum(dim=1) / log_r
            box_certainty = (1.0 - entropy.mean(dim=0)).clamp(0.0, 1.0).numpy()
        else:
            raw_mag = torch.sqrt((bx_slice.view(-1, hf, wf) ** 2).mean(dim=0) + 1e-12)
            denom = float(raw_mag.max().item()) + 1e-8
            box_certainty = (raw_mag / denom).clamp(0.0, 1.0).numpy()

        joint_map = cls_max * box_certainty
        results.append({
            "name": scale_names[idx] if idx < len(scale_names) else f"Scale_{idx}",
            "hw": (hf, wf),
            "cls_max": cls_max,
            "top1_id": t1_id,
            "top1_name": t1_name,
            "top1_map": cls_probs[t1_id].numpy(),
            "top2_id": t2_id,
            "top2_name": t2_name,
            "top2_map": cls_probs[t2_id].numpy(),
            "box_certainty": box_certainty,
            "joint_map": joint_map,
        })
    return results


def trace_edge_extractor(
    edge_extractor: nn.Module,
    input_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """Replay `EdgeExtractor.forward` step-by-step and return all intermediate maps & learned params."""
    with torch.no_grad():
        x = input_tensor.detach().float().to(next(edge_extractor.buffers(), input_tensor).device)
        b, c, h, w = x.shape

        if c > 1:
            if getattr(edge_extractor, "rgb_to_gray", False) and c == 3:
                target = torch.sum(x * edge_extractor.rgb_weight.float(), dim=1, keepdim=True)
            else:
                target = x.mean(dim=1, keepdim=True)
        else:
            target = x
        luminance = target.clone()

        if getattr(edge_extractor, "use_blur", False):
            pad = int(edge_extractor.blur_kernel_size) // 2
            t_pad = F.pad(target, (pad, pad, pad, pad), mode="replicate")
            target = F.conv2d(t_pad, edge_extractor.gaussian_kernel.float())
        blurred = target.clone()

        weber_enabled = bool(getattr(edge_extractor, "weber_log", False))
        weber_alpha_val = None
        if weber_enabled and hasattr(edge_extractor, "weber_alpha"):
            eps = 1e-3
            log_min = math.log(eps)
            t_fp32 = target.float()
            target_log = (torch.log(t_fp32.clamp(min=eps)) - log_min) / (-log_min)
            alpha = edge_extractor.weber_alpha.float().clamp(0.0, 1.0)
            weber_alpha_val = float(alpha.item())
            target = alpha * target_log + (1.0 - alpha) * t_fp32
        weber_target = target.clone()

        t_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(t_pad, edge_extractor.sobel_x.float())
        grad_y = F.conv2d(t_pad, edge_extractor.sobel_y.float())
        grad_lap = F.conv2d(t_pad, edge_extractor.laplacian.float())

        grad_stack = torch.cat([grad_x, grad_y, grad_lap], dim=1)
        fusion_w = edge_extractor.channel_fusion.weight.float()
        fused_mag = torch.abs(F.conv2d(grad_stack, fusion_w))

        g_flat = fused_mag.view(b, -1)
        g_min = g_flat.min(dim=1, keepdim=True)[0]
        g_max = g_flat.max(dim=1, keepdim=True)[0]
        grad_norm = ((g_flat - g_min) / (g_max - g_min + 1e-6)).view(b, 1, h, w)

        scale_val = float(edge_extractor.scale.float().item())
        offset_val = float(edge_extractor.offset.float().item())
        priority = torch.sigmoid(scale_val * grad_norm + offset_val)

    prio_np = priority[0, 0].cpu().numpy()
    return {
        "maps": {
            "luminance": luminance[0, 0].cpu().numpy(),
            "blurred": blurred[0, 0].cpu().numpy(),
            "weber_target": weber_target[0, 0].cpu().numpy(),
            "sobel_x": torch.abs(grad_x[0, 0]).cpu().numpy(),
            "sobel_y": torch.abs(grad_y[0, 0]).cpu().numpy(),
            "laplacian": torch.abs(grad_lap[0, 0]).cpu().numpy(),
            "fused_mag": fused_mag[0, 0].cpu().numpy(),
            "grad_norm": grad_norm[0, 0].cpu().numpy(),
            "edge_priority": prio_np,
        },
        "priority_tensor": priority.cpu(),
        "params": {
            "weber_log_enabled": weber_enabled,
            "weber_alpha": weber_alpha_val,
            "scale": round(scale_val, 6),
            "offset": round(offset_val, 6),
            "channel_fusion_weights": [round(float(v), 6) for v in fusion_w.view(-1).cpu().tolist()],
            "edge_priority_min": round(float(prio_np.min()), 6),
            "edge_priority_mean": round(float(prio_np.mean()), 6),
            "edge_priority_max": round(float(prio_np.max()), 6),
            "edge_priority_p90": round(float(np.percentile(prio_np, 90)), 6),
        },
    }


def trace_edgesab_module(
    mod: nn.Module,
    x_in: torch.Tensor,
    edge_map: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Replay `EdgeSAB.forward` step-by-step and return spatial maps, attention heads, and metrics."""
    dev = mod.q.weight.device
    with torch.no_grad():
        x = x_in.detach().float().to(dev)
        b, c, h, w = x.shape

        q_raw = F.conv2d(
            F.conv2d(x, mod.q.weight.float(), mod.q.bias.float() if mod.q.bias is not None else None),
            mod.q_dwconv.weight.float(),
            mod.q_dwconv.bias.float() if mod.q_dwconv.bias is not None else None,
            padding=mod.q_dwconv.padding,
            groups=mod.q_dwconv.groups,
        )
        kv_raw = F.conv2d(
            F.conv2d(x, mod.kv.weight.float(), mod.kv.bias.float() if mod.kv.bias is not None else None),
            mod.kv_dwconv.weight.float(),
            mod.kv_dwconv.bias.float() if mod.kv_dwconv.bias is not None else None,
            padding=mod.kv_dwconv.padding,
            groups=mod.kv_dwconv.groups,
        )
        k_raw, v_raw = kv_raw.chunk(2, dim=1)

        edge_scaled: Optional[torch.Tensor] = None
        edge_weight: Optional[torch.Tensor] = None
        if edge_map is not None and hasattr(mod, "edge_scale"):
            em = edge_map.detach().float().to(dev)
            if em.shape[-2:] != (h, w):
                edge_bil = F.interpolate(em, size=(h, w), mode="bilinear", align_corners=False)
                edge_max = F.adaptive_max_pool2d(em, output_size=(h, w))
                edge_scaled = 0.5 * edge_bil + 0.5 * edge_max
            else:
                edge_scaled = em
            edge_weight = 1.0 + mod.edge_scale.float() * edge_scaled
            q = q_raw * edge_weight
            k = k_raw * edge_weight
        else:
            q = q_raw
            k = k_raw

        num_heads = int(mod.num_heads)
        head_dim = int(mod.head_dim)
        inner_dim = int(mod.inner_dim)

        qh = q.reshape(b, num_heads, head_dim, h * w)
        kh = k.reshape(b, num_heads, head_dim, h * w)
        vh = v_raw.reshape(b, num_heads, head_dim, h * w)

        q_norm = F.normalize(qh, dim=-1, eps=1e-6)
        k_norm = F.normalize(kh, dim=-1, eps=1e-6)
        attn = F.softmax((q_norm @ k_norm.transpose(-2, -1)) * mod.temperature.float(), dim=-1)

        out = (attn @ vh).reshape(b, inner_dim, h, w)
        # Apply FFN depthwise gate + projection in FP32 safely
        ffn_conv = mod.ffn_gate[0]
        gate_act = F.silu(
            F.conv2d(
                out,
                ffn_conv.weight.float(),
                ffn_conv.bias.float() if ffn_conv.bias is not None else None,
                stride=ffn_conv.stride,
                padding=ffn_conv.padding,
                groups=ffn_conv.groups,
            )
        )
        out = out * gate_act
        out_pre_gate = F.conv2d(
            out,
            mod.project_out.weight.float(),
            mod.project_out.bias.float() if mod.project_out.bias is not None else None,
        )

        if edge_scaled is not None and hasattr(mod, "edge_gate"):
            eg_mod = mod.edge_gate
            eg_feat = F.conv2d(
                edge_scaled,
                eg_mod.local_conv.weight.float(),
                eg_mod.local_conv.bias.float() if eg_mod.local_conv.bias is not None else None,
                padding=eg_mod.local_conv.padding,
            )
            eg_wide = F.conv2d(
                eg_feat,
                eg_mod.wide_dwconv.weight.float(),
                eg_mod.wide_dwconv.bias.float() if eg_mod.wide_dwconv.bias is not None else None,
                padding=eg_mod.wide_dwconv.padding,
                dilation=eg_mod.wide_dwconv.dilation,
                groups=eg_mod.wide_dwconv.groups,
            )
            edge_gate_t = torch.sigmoid(eg_feat + eg_wide)
            out_gated = out_pre_gate * edge_gate_t
        else:
            edge_gate_t = torch.ones_like(out_pre_gate)
            out_gated = out_pre_gate

        residual_delta = mod.gamma.float() * out_gated
        y_out = x + residual_delta

        # Quantitative diagnostics
        x_flat = x.view(b, -1)
        y_flat = y_out.view(b, -1)
        cos_sim = float(F.cosine_similarity(x_flat, y_flat, dim=1).mean().item())
        res_ratio = float(
            torch.sqrt((residual_delta ** 2).mean() + 1e-12).item()
            / (torch.sqrt((x ** 2).mean() + 1e-12).item() + 1e-8)
        )
        gamma_f = mod.gamma.detach().float().cpu()
        eg_mean_map = edge_gate_t[0].mean(dim=0).cpu().numpy()

    def _l2_2d(t: torch.Tensor) -> np.ndarray:
        return torch.sqrt((t[0] ** 2).mean(dim=0) + 1e-12).cpu().numpy()

    return {
        "maps": {
            "x_in": _l2_2d(x),
            "edge_scaled": edge_scaled[0, 0].cpu().numpy() if edge_scaled is not None else np.zeros((h, w), dtype=np.float32),
            "edge_weight": edge_weight[0, 0].cpu().numpy() if edge_weight is not None else np.ones((h, w), dtype=np.float32),
            "q_edge_mod": _l2_2d(q),
            "k_edge_mod": _l2_2d(k),
            "out_pre_gate": _l2_2d(out_pre_gate),
            "edge_gate": eg_mean_map,
            "out_gated": _l2_2d(out_gated),
            "residual_delta": _l2_2d(residual_delta),
            "output": _l2_2d(y_out),
        },
        "attn_heads": attn[0].cpu().numpy(),  # (num_heads, head_dim, head_dim)
        "y_out_tensor": y_out.cpu(),
        "stats": {
            "dim": int(mod.dim),
            "inner_dim": inner_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hw": [h, w],
            "edge_scale": round(float(mod.edge_scale.detach().float().item()), 6) if hasattr(mod, "edge_scale") else 0.0,
            "gamma_mean": round(float(gamma_f.mean().item()), 6),
            "gamma_abs_mean": round(float(gamma_f.abs().mean().item()), 6),
            "gamma_abs_max": round(float(gamma_f.abs().max().item()), 6),
            "temperature": [round(float(v), 5) for v in mod.temperature.detach().float().view(-1).cpu().tolist()],
            "edge_gate_mean": round(float(eg_mean_map.mean()), 6),
            "edge_gate_std": round(float(eg_mean_map.std()), 6),
            "residual_l2_ratio": round(res_ratio, 6),
            "cos_sim_in_out": round(cos_sim, 6),
        },
    }


# ==================================================================================================
# 6. Heatmap, Diff Map & Tile Rendering Utilities
# ==================================================================================================

def apply_display_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction strictly for visual readability (never passed to model)."""
    if abs(gamma - 1.0) < 1e-4:
        return img_bgr
    inv = np.clip(img_bgr.astype(np.float32) / 255.0, 1e-6, 1.0) ** gamma
    return np.clip(inv * 255.0, 0, 255).astype(np.uint8)


def normalize_map_to_uint8(
    arr2d: np.ndarray,
    clip_percentile: float = 99.5,
    fixed_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    arr = np.nan_to_num(arr2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        vmin = float(arr.min())
        vmax = float(np.percentile(arr, clip_percentile)) if clip_percentile < 100.0 else float(arr.max())
        if vmax <= vmin + 1e-8:
            vmax = float(arr.max())
    if vmax <= vmin + 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def feature_to_spatial_maps(feat: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute aggregated spatial maps from a `(1, C, H, W)` activation tensor."""
    f = feat[0]
    return {
        "l2": torch.sqrt((f ** 2).mean(dim=0) + 1e-12).numpy(),
        "mean": f.mean(dim=0).numpy(),
        "max": f.max(dim=0).values.numpy(),
    }


def compute_tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    arr = t.numpy()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "l2": float(np.sqrt(np.mean(arr ** 2))),
        "sparsity": float(np.mean(np.abs(arr) < 1e-2) * 100.0),
    }


def unpad_and_resize_map(
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
) -> np.ndarray:
    """Upsample feature map to letterbox canvas, strip padding, and resize to display size."""
    up = cv2.resize(map2d.astype(np.float32), (lb_size, lb_size), interpolation=cv2.INTER_CUBIC)
    top, left, bottom_y, right_x = crop_box
    cropped = up[top:bottom_y, left:right_x]
    if cropped.size == 0:
        cropped = up
    return cv2.resize(cropped, target_wh, interpolation=cv2.INTER_LINEAR)


def render_heatmap_overlay(
    base_bgr: np.ndarray,
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float = 0.48,
    clip_percentile: float = 99.5,
    fixed_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(blended_overlay_bgr, pure_colored_heatmap_bgr)` matching `base_bgr` size."""
    h, w = base_bgr.shape[:2]
    aligned = unpad_and_resize_map(map2d, lb_size, crop_box, (w, h))
    u8 = normalize_map_to_uint8(aligned, clip_percentile=clip_percentile, fixed_range=fixed_range)
    heat = cv2.applyColorMap(u8, colormap_id)
    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, heat, alpha, 0.0)
    return overlay, heat


def _diverging_lut_bgr() -> np.ndarray:
    """Build 256x3 BGR lookup table: Blue (-Max) -> Dark Neutral (0) -> Bright Red/Amber (+Max)."""
    anchors = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    # BGR values
    b = np.array([255, 220, 28, 45, 45], dtype=np.float32)
    g = np.array([170, 95, 30, 70, 145], dtype=np.float32)
    r = np.array([35, 35, 32, 235, 255], dtype=np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.stack([np.interp(x, anchors, b), np.interp(x, anchors, g), np.interp(x, anchors, r)], axis=-1)
    return np.clip(lut, 0, 255).astype(np.uint8)


DIVERGING_LUT_BGR = _diverging_lut_bgr()


def render_signed_diff_map(
    base_bgr: np.ndarray,
    diff2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    blend_alpha: float = 0.72,
    abs_percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Render signed delta `(Enhanced - Base)` using Blue (-) / Neutral (0) / Red (+) diverging colormap.
    Returns `(blended_bgr, pure_diff_bgr, vmax_used)`.
    """
    h, w = base_bgr.shape[:2]
    aligned = unpad_and_resize_map(diff2d, lb_size, crop_box, (w, h))
    abs_vals = np.abs(aligned)
    vmax = float(np.percentile(abs_vals, abs_percentile))
    if vmax < 1e-6:
        vmax = float(abs_vals.max())
    if vmax < 1e-8:
        vmax = 1.0
    norm = np.clip(aligned / vmax, -1.0, 1.0)
    idx = np.clip(((norm + 1.0) * 0.5 * 255.0).round(), 0, 255).astype(np.uint8)
    pure_bgr = DIVERGING_LUT_BGR[idx]
    blended = cv2.addWeighted(base_bgr, 1.0 - blend_alpha, pure_bgr, blend_alpha, 0.0)
    return blended, pure_bgr, vmax


def render_attn_heads_panel(
    attn_heads: np.ndarray,
    target_wh: Tuple[int, int],
    colormap_id: int,
) -> np.ndarray:
    """Render `(num_heads, D, D)` channel attention matrices as a crisp grid matching `target_wh`."""
    tw, th = target_wh
    n_heads = attn_heads.shape[0]
    cols = 2 if n_heads > 1 else 1
    rows = (n_heads + cols - 1) // cols
    pad = 6
    cell_w = max(48, (tw - (cols + 1) * pad) // cols)
    cell_h = max(48, (th - (rows + 1) * pad) // rows)

    panel = np.full((th, tw, 3), 22, dtype=np.uint8)
    for idx in range(n_heads):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        mat = attn_heads[idx]
        u8 = normalize_map_to_uint8(mat, clip_percentile=99.5)
        heat = cv2.applyColorMap(u8, colormap_id)
        heat_rs = cv2.resize(heat, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        panel[y0 : y0 + cell_h, x0 : x0 + cell_w] = heat_rs
        cv2.rectangle(panel, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (130, 140, 160), 1)
        cv2.putText(
            panel, f"H{idx}", (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (245, 248, 255), 1, cv2.LINE_AA,
        )
    return panel


def draw_detections_on_img(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    show_conf: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    canvas = img_bgr.copy()
    if boxes is None or len(boxes) == 0:
        return canvas
    h, w = canvas.shape[:2]
    font_scale = max(0.38, min(0.55, min(h, w) / 900.0))
    for row in boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
        conf = float(row[4]) if len(row) >= 6 else 1.0
        cls_id = int(row[5]) if len(row) >= 6 else int(row[4])
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cname = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        label = f"{cname} {conf:.2f}" if show_conf else cname
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        by2 = min(h, max(y1, th + baseline + 4))
        by1 = max(0, by2 - (th + baseline + 4))
        bx2 = min(w, x1 + tw + 6)
        cv2.rectangle(canvas, (x1, by1), (bx2, by2), color, -1)
        bright = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
        txt_col = (0, 0, 0) if bright > 140 else (255, 255, 255)
        cv2.putText(
            canvas, label, (x1 + 3, by2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA,
        )
    return canvas


def resize_to_panel(img_bgr: np.ndarray, panel_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel_h = max(64, int(round(h * (panel_w / float(w)))))
    return cv2.resize(
        img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA if panel_w < w else cv2.INTER_LINEAR
    )


def make_labeled_tile(
    img_bgr: np.ndarray,
    title: str,
    subtitle: str = "",
    header_color: Tuple[int, int, int] = (230, 235, 245),
    accent_bgr: Tuple[int, int, int] = (70, 140, 255),
) -> np.ndarray:
    """Wrap an image tile with a dark-themed two-line header bar."""
    h, w = img_bgr.shape[:2]
    header_h = 42 if subtitle else 28
    tile = np.full((h + header_h, w, 3), 24, dtype=np.uint8)
    tile[header_h:, :] = img_bgr
    cv2.rectangle(tile, (0, 0), (w - 1, header_h - 1), (32, 34, 40), -1)
    cv2.line(tile, (0, 0), (w - 1, 0), accent_bgr, 2)
    cv2.line(tile, (0, header_h - 1), (w - 1, header_h - 1), (65, 68, 78), 1)

    t_scale = min(0.50, max(0.32, w / 860.0))
    cv2.putText(tile, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, t_scale, header_color, 1, cv2.LINE_AA)
    if subtitle:
        s_scale = min(0.40, max(0.28, w / 1080.0))
        cv2.putText(tile, subtitle, (8, 35), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (165, 175, 190), 1, cv2.LINE_AA)
    return tile


def assemble_grid(
    tiles: Sequence[np.ndarray],
    ncols: int,
    banner_title: str = "",
    banner_sub: str = "",
    pad: int = 10,
    bg_color: Tuple[int, int, int] = (16, 17, 20),
    colormap_id: Optional[int] = None,
    cbar_mode: str = "standard",
) -> np.ndarray:
    """Arrange image tiles into a crisp grid with banner and optional standard/diverging colorbar."""
    if not tiles:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    nrows = (len(tiles) + ncols - 1) // ncols

    banner_h = 56 if banner_title else 0
    show_std = colormap_id is not None and cbar_mode in {"standard", "both"}
    show_div = cbar_mode in {"diverging", "both"}
    cbar_w = (58 if show_std else 0) + (58 if show_div else 0)

    canvas_h = banner_h + pad + nrows * (th + pad)
    canvas_w = pad + ncols * (tw + pad) + cbar_w
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    if banner_title:
        cv2.rectangle(canvas, (0, 0), (canvas_w - 1, banner_h - 1), (26, 28, 34), -1)
        cv2.line(canvas, (0, banner_h - 1), (canvas_w - 1, banner_h - 1), (80, 150, 255), 2)
        cv2.putText(canvas, banner_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 248, 255), 2, cv2.LINE_AA)
        if banner_sub:
            cv2.putText(canvas, banner_sub, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 200), 1, cv2.LINE_AA)

    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, ncols)
        y0 = banner_h + pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        h_t, w_t = tile.shape[:2]
        canvas[y0 : y0 + h_t, x0 : x0 + w_t] = tile

    bar_h = max(120, canvas_h - banner_h - 2 * pad - 48)
    bar_y = banner_h + pad + 24
    cur_x = canvas_w - cbar_w + 14

    if show_std and colormap_id is not None:
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        grad_bar = cv2.applyColorMap(np.repeat(grad, 18, axis=1), colormap_id)
        canvas[bar_y : bar_y + bar_h, cur_x : cur_x + 18] = grad_bar
        cv2.rectangle(canvas, (cur_x, bar_y), (cur_x + 18, bar_y + bar_h), (120, 125, 135), 1)
        cv2.putText(canvas, "High", (cur_x - 4, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Low", (cur_x - 2, bar_y + bar_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 185, 195), 1, cv2.LINE_AA)
        cur_x += 54

    if show_div:
        grad_idx = np.linspace(255, 0, bar_h, dtype=np.float32).round().astype(np.uint8)
        div_col = DIVERGING_LUT_BGR[grad_idx][:, None, :]
        div_bar = np.repeat(div_col, 18, axis=1)
        canvas[bar_y : bar_y + bar_h, cur_x : cur_x + 18] = div_bar
        cv2.rectangle(canvas, (cur_x, bar_y), (cur_x + 18, bar_y + bar_h), (120, 125, 135), 1)
        cv2.putText(canvas, "+Enh", (cur_x - 6, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (cur_x + 22, bar_y + bar_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)
        cv2.putText(canvas, "-Base", (cur_x - 8, bar_y + bar_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 190, 120), 1, cv2.LINE_AA)

    return canvas


# ==================================================================================================
# 7. Per-Model Visualizers (Base & Enhanced YOLOv11)
# ==================================================================================================

def _get_edgesab_indices(records: "OrderedDict[str, Dict[str, Any]]") -> set:
    return {
        r["layer_idx"]
        for r in records.values()
        if not r["is_sublayer"] and r.get("mod_type") in {"EdgeSAB", "SAB"} and r["layer_idx"] >= 0
    }


def save_network_overview(
    save_path: str,
    display_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    gt_bgr: Optional[np.ndarray],
    records: "OrderedDict[str, Dict[str, Any]]",
    detect_info: List[Dict[str, Any]],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    model_name: str,
    meta_info: Dict[str, Any],
    img_name: str,
    n_preds: int,
) -> None:
    """Create `00_network_overview.png` summarizing Backbone, FPN/PAN, optional Edge/EdgeSAB, and Detect."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = args.panel_width
    base_tile = resize_to_panel(display_bgr, pw)
    arch = meta_info["arch"]
    epoch = meta_info["epoch"]
    sab_indices = _get_edgesab_indices(records)
    is_p3_skip = arch == "enhanced" and sab_indices == {15}
    is_p3_inline = arch == "enhanced" and sab_indices == {17}
    is_p3_only = arch == "enhanced" and sab_indices == {23}
    arch_tag = (
        "Enhanced YOLOv11 (Edge+SkipEdgeSAB-P3)"
        if is_p3_skip
        else (
            "Enhanced YOLOv11 (Edge+InlineEdgeSAB-P3)"
            if is_p3_inline
            else (
                "Enhanced YOLOv11 (Edge+EdgeSAB-P3)"
                if is_p3_only
                else ("Enhanced YOLOv11 (Edge+EdgeSAB)" if arch == "enhanced" else "YOLOv11 Base")
            )
        )
    )

    tiles: List[np.ndarray] = [
        make_labeled_tile(
            base_tile.copy(),
            "Input Image (Display)",
            f"Orig {display_bgr.shape[1]}x{display_bgr.shape[0]} | vis_gamma={args.vis_gamma:.2f}",
            accent_bgr=(180, 180, 180),
        ),
        make_labeled_tile(
            resize_to_panel(pred_bgr, pw),
            f"{arch_tag} Preds ({n_preds})",
            f"{model_name} (ep={epoch}) | conf>={args.conf_thres:.2f}",
            accent_bgr=(50, 230, 100),
        ),
    ]
    if gt_bgr is not None:
        tiles.append(
            make_labeled_tile(
                resize_to_panel(gt_bgr, pw),
                "Ground Truth",
                "Dataset annotation",
                accent_bgr=(80, 220, 255),
            )
        )

    if arch == "enhanced":
        if is_p3_skip:
            key_indices = {-1, 0, 4, 6, 10, 13, 15, 17, 20, 23} if gt_bgr is not None else {-1, 0, 2, 4, 6, 10, 13, 15, 17, 20, 23}
        elif is_p3_inline:
            key_indices = {-1, 0, 2, 4, 6, 9, 10, 13, 17, 20, 23} if gt_bgr is not None else {-1, 0, 2, 4, 6, 8, 9, 10, 13, 17, 20, 23}
        elif is_p3_only:
            key_indices = {-1, 0, 2, 4, 6, 9, 10, 13, 19, 22, 23} if gt_bgr is not None else {-1, 0, 2, 4, 6, 8, 9, 10, 13, 19, 22, 23}
        else:
            # Pick 10 (with GT) or 11 (no GT) feature layers + EdgeExtractor + 3 Detect maps = 16 tiles (4x4)
            key_indices = {-1, 0, 2, 4, 6, 9, 10, 13, 23, 24, 25} if gt_bgr is not None else {-1, 0, 2, 4, 6, 8, 9, 10, 13, 23, 24, 25}
    else:
        # Base 24-layer: pick 10 (with GT) or 11 (no GT) feature layers + 3 Detect maps = 16 tiles (4x4)
        key_indices = {0, 2, 4, 6, 9, 10, 13, 16, 19, 22} if gt_bgr is not None else {0, 2, 4, 6, 8, 9, 10, 13, 16, 19, 22}

    for rec in records.values():
        if rec["is_sublayer"] or "tensor" not in rec:
            continue
        if rec["layer_idx"] not in key_indices:
            continue
        t = rec["tensor"]
        c, h_f, w_f = t.shape[1], t.shape[2], t.shape[3]
        if rec["layer_idx"] == -1:
            prio = t[0, 0].numpy()
            ov, _ = render_heatmap_overlay(
                base_tile, prio, imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            subtitle = f"[1x{h_f}x{w_f}] min={prio.min():.2f} mean={prio.mean():.2f} max={prio.max():.2f}"
        else:
            maps = feature_to_spatial_maps(t)
            stats = compute_tensor_stats(t)
            ov, _ = render_heatmap_overlay(
                base_tile, maps["l2"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=args.clip_percentile,
            )
            stride = max(1, imgsz // max(1, h_f))
            subtitle = f"[{c}x{h_f}x{w_f}] s={stride} | L2={stats['l2']:.3f} max={stats['max']:.2f}"

        tiles.append(
            make_labeled_tile(
                ov,
                f"{rec['key']} ({rec['mod_type']})",
                subtitle,
                accent_bgr=STAGE_ACCENT_COLORS.get(rec["stage"], (120, 170, 255)),
            )
        )

    for d in detect_info:
        ov, _ = render_heatmap_overlay(
            base_tile, d["cls_max"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
        )
        tiles.append(
            make_labeled_tile(
                ov,
                f"Detect ClsProb: {d['name']}",
                f"max_p={d['cls_max'].max():.3f} | top={d['top1_name']} | DFL={d['box_certainty'].max():.2f}",
                accent_bgr=(90, 110, 255),
            )
        )

    grid = assemble_grid(
        tiles,
        ncols=4,
        banner_title=f"{arch_tag} Feature Flow Overview - {img_name}",
        banner_sub=f"Model: {model_name} (ep={epoch}, arch={arch}) | Colormap: {args.colormap} | L2 Energy & Cls Prob Maps",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_backbone_vs_neck_scales(
    save_path: str,
    display_bgr: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    detect_info: List[Dict[str, Any]],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    arch: str,
    img_name: str,
) -> None:
    """Compare Backbone -> Neck -> (optional EdgeSAB) -> Detect Class Conf & DFL Certainty at P3, P4, P5."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = args.panel_width
    base_tile = resize_to_panel(display_bgr, pw)
    by_idx = {r["layer_idx"]: r for r in records.values() if not r["is_sublayer"] and "tensor" in r}

    is_enh = arch == "enhanced"
    sab_indices = _get_edgesab_indices(records)
    is_p3_skip = is_enh and sab_indices == {15}
    is_p3_inline = is_enh and sab_indices == {17}
    if is_p3_skip:
        scale_rows = [
            ("P3/8 (Small)", 4, 17, 15, 0),
            ("P4/16 (Medium)", 6, 20, None, 1),
            ("P5/32 (Large)", 10, 23, None, 2),
        ]
    elif is_p3_inline:
        scale_rows = [
            ("P3/8 (Small)", 4, 16, 17, 0),
            ("P4/16 (Medium)", 6, 20, None, 1),
            ("P5/32 (Large)", 10, 23, None, 2),
        ]
    else:
        scale_rows = [
            ("P3/8 (Small)", 4, 16, 23 if 23 in sab_indices else None, 0),
            ("P4/16 (Medium)", 6, 19, 24 if 24 in sab_indices else None, 1),
            ("P5/32 (Large)", 10, 22, 25 if 25 in sab_indices else None, 2),
        ]

    tiles: List[np.ndarray] = []
    for scale_label, bb_idx, pan_idx, sab_idx, det_idx in scale_rows:
        stages: List[Tuple[str, Optional[int], Tuple[int, int, int]]] = [("Backbone Skip", bb_idx, (50, 170, 255))]
        if is_p3_skip:
            stages.append(("EdgeSAB Skip", sab_idx, (255, 80, 220)))
            stages.append(("Neck Fused Out", pan_idx, (140, 230, 80)))
        elif is_enh:
            stages.append(("PAN Pre-Attn" if sab_idx is not None else "Neck Fused Out", pan_idx, (230, 210, 80) if sab_idx is not None else (140, 230, 80)))
            stages.append(("EdgeSAB Refined", sab_idx, (255, 80, 220)))
        else:
            stages.append(("Neck Fused Out", pan_idx, (140, 230, 80)))

        for role_title, rec_idx, accent in stages:
            rec = by_idx.get(rec_idx) if rec_idx is not None else None
            if rec is not None:
                t = rec["tensor"]
                maps = feature_to_spatial_maps(t)
                stats = compute_tensor_stats(t)
                ov, _ = render_heatmap_overlay(
                    base_tile, maps["l2"], imgsz, crop_box, cmap_id,
                    alpha=args.alpha, clip_percentile=args.clip_percentile,
                )
                tiles.append(
                    make_labeled_tile(
                        ov,
                        f"{scale_label} - {role_title} (L{rec_idx:02d})",
                        f"[{t.shape[1]}x{t.shape[2]}x{t.shape[3]}] L2={stats['l2']:.3f} max={stats['max']:.2f}",
                        accent_bgr=accent,
                    )
                )
            else:
                direct_note = (
                    "Direct Skip -> Concat (Enriched via P3)"
                    if is_p3_skip and role_title == "EdgeSAB Skip"
                    else ("Direct PAN -> Detect" if is_enh and role_title == "EdgeSAB Refined" else "N/A")
                )
                tiles.append(make_labeled_tile(base_tile.copy(), f"{scale_label} - No EdgeSAB", direct_note, accent_bgr=(120, 125, 140)))

        if det_idx < len(detect_info):
            d = detect_info[det_idx]
            cls_ov, _ = render_heatmap_overlay(
                base_tile, d["cls_max"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            dfl_ov, _ = render_heatmap_overlay(
                base_tile, d["box_certainty"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            tiles.append(
                make_labeled_tile(
                    cls_ov,
                    f"{scale_label} - Detect Cls Conf (cv3)",
                    f"max_cls={d['cls_max'].max():.3f} | top1={d['top1_name']}",
                    accent_bgr=(90, 110, 255),
                )
            )
            tiles.append(
                make_labeled_tile(
                    dfl_ov,
                    f"{scale_label} - DFL Box Certainty (cv2)",
                    f"max={d['box_certainty'].max():.3f} | mean={d['box_certainty'].mean():.3f}",
                    accent_bgr=(255, 110, 180),
                )
            )

    ncols = 5 if is_enh else 4
    grid = assemble_grid(
        tiles,
        ncols=ncols,
        banner_title=f"Multi-Scale Progression ({arch.upper()}): Backbone -> Neck -> Detect ({img_name})",
        banner_sub=(
            "Rows: P3/8, P4/16, P5/32 | Cols: Backbone -> Skip EdgeSAB -> Neck Out -> Detect Cls -> DFL Certainty"
            if is_p3_skip
            else (
                "Rows: P3/8, P4/16, P5/32 | Cols: Backbone -> PAN -> EdgeSAB -> Detect Cls -> DFL Certainty"
                if is_enh
                else "Rows: P3/8, P4/16, P5/32 | Cols: Backbone -> PAN Neck -> Detect Cls -> DFL Certainty"
            )
        ),
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_all_layers_progression(
    save_path: str,
    display_bgr: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    arch: str,
    img_name: str,
) -> None:
    """Render every feature layer chronologically (L00..L22 for Base; EdgeMap + L00..L25 for Enhanced)."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(320, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)

    tiles: List[np.ndarray] = []
    for rec in records.values():
        if rec["is_sublayer"] or "tensor" not in rec:
            continue
        t = rec["tensor"]
        c, h_f, w_f = t.shape[1], t.shape[2], t.shape[3]
        if rec["layer_idx"] == -1:
            prio = t[0, 0].numpy()
            ov, _ = render_heatmap_overlay(
                base_tile, prio, imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            sub = f"EdgeMap [1x{h_f}x{w_f}] mean={prio.mean():.3f} max={prio.max():.2f}"
        else:
            maps = feature_to_spatial_maps(t)
            stats = compute_tensor_stats(t)
            ov, _ = render_heatmap_overlay(
                base_tile, maps["l2"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=args.clip_percentile,
            )
            sub = f"{rec['mod_type']} [{c}x{h_f}x{w_f}] L2={stats['l2']:.3f} sp={stats['sparsity']:.0f}%"

        tiles.append(
            make_labeled_tile(
                ov,
                f"{rec['key']}",
                sub,
                accent_bgr=STAGE_ACCENT_COLORS.get(rec["stage"], (150, 170, 255)),
            )
        )

    max_idx = max(
        (r["layer_idx"] for r in records.values() if not r["is_sublayer"] and "tensor" in r and r["layer_idx"] >= 0),
        default=(23 if arch == "enhanced" else 22),
    )
    grid = assemble_grid(
        tiles,
        ncols=6,
        banner_title=f"Full YOLOv11 ({arch.upper()}) Layer Progression (L00 -> L{max_idx:02d}) - {img_name}",
        banner_sub="Orange: Backbone | Gold: C2PSA | Pink: FPN | Cyan: PAN | Green: HeadIn | Magenta: EdgeSAB",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_detect_head_details(
    save_path: str,
    display_bgr: np.ndarray,
    detect_info: List[Dict[str, Any]],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Visualize decoupled Detect head maps (Max Class, Top-1 Class, Top-2 Class, DFL Certainty, Joint)."""
    if not detect_info:
        return
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(280, min(360, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)

    tiles: List[np.ndarray] = []
    for d in detect_info:
        items = [
            (d["cls_max"], f"{d['name']} - Max Cls Prob", f"max={d['cls_max'].max():.3f} | mean={d['cls_max'].mean():.3f}", (90, 110, 255)),
            (d["top1_map"], f"Top-1 Class: {d['top1_name']}", f"peak={d['top1_map'].max():.3f}", (80, 220, 255)),
            (d["top2_map"], f"Top-2 Class: {d['top2_name']}", f"peak={d['top2_map'].max():.3f}", (200, 160, 255)),
            (d["box_certainty"], "DFL Box Certainty (1-H)", f"max={d['box_certainty'].max():.3f} | mean={d['box_certainty'].mean():.3f}", (80, 230, 140)),
            (d["joint_map"], "Joint (Cls * DFL Cert)", f"max={d['joint_map'].max():.3f} | mean={d['joint_map'].mean():.3f}", (255, 80, 180)),
        ]
        for mp, title, sub, acc in items:
            ov, _ = render_heatmap_overlay(
                base_tile, mp, imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            tiles.append(make_labeled_tile(ov, title, sub, accent_bgr=acc))

    grid = assemble_grid(
        tiles,
        ncols=5,
        banner_title=f"YOLOv11 Decoupled Detect Head Breakdown - {img_name}",
        banner_sub="Rows: P3/8, P4/16, P5/32 | Cols: Max Class, Top-1 Class, Top-2 Class, DFL Certainty, Joint (Range [0, 1])",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_activation_stats_chart(
    save_path: str,
    records: "OrderedDict[str, Dict[str, Any]]",
    img_name: str,
    model_name: str,
    arch: str,
) -> None:
    """Plot per-layer L2 Energy, Mean, Max, and Near-Zero Sparsity (%) across all captured layers."""
    layer_recs = [
        r for r in records.values()
        if not r["is_sublayer"] and "tensor" in r and r["layer_idx"] >= 0
    ]
    if not layer_recs:
        return

    labels = [f"L{r['layer_idx']:02d}\n{r['mod_type']}" for r in layer_recs]
    stats_list = [compute_tensor_stats(r["tensor"]) for r in layer_recs]
    l2_vals = [s["l2"] for s in stats_list]
    mean_vals = [s["mean"] for s in stats_list]
    max_vals = [s["max"] for s in stats_list]
    sparsity_vals = [s["sparsity"] for s in stats_list]

    stage_plot_colors = {
        "Backbone": "#f5a623",
        "Backbone_Attn": "#ffd54f",
        "FPN_Neck": "#d96bff",
        "PAN_Neck": "#4fc3f7",
        "PAN_HeadIn": "#4cd98c",
        "EdgeSAB_HeadIn": "#ff5cc8",
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(13, len(labels) * 0.65), 8), sharex=True)
        fig.patch.set_facecolor("#121318")
        for ax in (ax1, ax2):
            ax.set_facecolor("#191b22")
            ax.tick_params(colors="#d8dce6", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3a3e4d")
            ax.grid(True, linestyle="--", alpha=0.25, color="#8a92a8")

        ax1.plot(x, l2_vals, marker="o", linewidth=2.0, color="#4cd98c", label="L2 Energy")
        ax1.plot(x, mean_vals, marker="s", linewidth=1.6, color="#56b4ff", label="Mean Activation")
        ax1_r = ax1.twinx()
        ax1_r.plot(x, max_vals, marker="^", linewidth=1.4, linestyle=":", color="#ffb84d", label="Max Activation")
        ax1_r.tick_params(colors="#ffb84d", labelsize=8)
        ax1_r.set_ylabel("Max Activation", color="#ffb84d", fontsize=9)
        ax1.set_ylabel("L2 / Mean Activation", color="#d8dce6", fontsize=9)
        ax1.set_title(
            f"Layer-wise Activation Magnitude & Sparsity ({model_name} [{arch}] - {img_name})",
            color="#f0f4ff", fontsize=11, pad=10,
        )

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", facecolor="#222530", edgecolor="#404556", labelcolor="#e6eaf2", fontsize=8)

        bar_colors = [stage_plot_colors.get(r["stage"], "#90a4ae") for r in layer_recs]
        ax2.bar(x, sparsity_vals, color=bar_colors, alpha=0.88, width=0.65)
        ax2.set_ylabel("Near-Zero Ratio (%)", color="#d8dce6", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as exc:
        print(f"[Warn] Could not render matplotlib stats plot ({exc}); skipping.")


def save_edge_extractor_pipeline(
    save_path: str,
    edge_dir: str,
    display_bgr: np.ndarray,
    edge_trace: Dict[str, Any],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Create `05_edge_extractor_pipeline.png` and optionally dump raw stage PNGs under `edge/`."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(260, min(360, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)
    maps = edge_trace["maps"]
    params = edge_trace["params"]
    th, tw = base_tile.shape[:2]

    def _gray_tile_img(arr2d: np.ndarray, clip_pct: float = 99.5, fixed: Optional[Tuple[float, float]] = None) -> np.ndarray:
        aligned = unpad_and_resize_map(arr2d, imgsz, crop_box, (tw, th))
        u8 = normalize_map_to_uint8(aligned, clip_percentile=clip_pct, fixed_range=fixed)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    def _heat_tile_img(arr2d: np.ndarray, clip_pct: float = 99.5, fixed: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        return render_heatmap_overlay(
            base_tile, arr2d, imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=clip_pct, fixed_range=fixed,
        )

    lum_gray = _gray_tile_img(maps["luminance"], clip_pct=100.0, fixed=(0.0, 1.0))
    blur_gray = _gray_tile_img(maps["blurred"], clip_pct=100.0, fixed=(0.0, 1.0))
    weber_gray = _gray_tile_img(maps["weber_target"], clip_pct=100.0, fixed=(0.0, 1.0))
    _, sx_heat = _heat_tile_img(maps["sobel_x"], clip_pct=99.5)
    _, sy_heat = _heat_tile_img(maps["sobel_y"], clip_pct=99.5)
    _, lap_heat = _heat_tile_img(maps["laplacian"], clip_pct=99.5)
    _, fused_heat = _heat_tile_img(maps["fused_mag"], clip_pct=99.5)
    prio_ov, prio_heat = _heat_tile_img(maps["edge_priority"], clip_pct=100.0, fixed=(0.0, 1.0))
    prio_gray = _gray_tile_img(maps["edge_priority"], clip_pct=100.0, fixed=(0.0, 1.0))

    if not args.no_edge_raw:
        os.makedirs(edge_dir, exist_ok=True)
        raw_exports = {
            "01_luminance_gray.png": lum_gray,
            "02_gaussian_blur_gray.png": blur_gray,
            "03_weber_log_gray.png": weber_gray,
            "04_sobel_x_heat.png": sx_heat,
            "05_sobel_y_heat.png": sy_heat,
            "06_laplacian_heat.png": lap_heat,
            "07_fused_magnitude_heat.png": fused_heat,
            "08_edge_priority_gray.png": prio_gray,
            "09_edge_priority_heat.png": prio_heat,
            "10_edge_overlay.png": prio_ov,
        }
        for fname, img in raw_exports.items():
            cv2.imwrite(os.path.join(edge_dir, fname), img)

    w_alpha_str = f"{params['weber_alpha']:.3f}" if params["weber_alpha"] is not None else "off"
    tiles = [
        make_labeled_tile(base_tile, "1. Input Image", f"vis_gamma={args.vis_gamma:.2f}", accent_bgr=(180, 180, 180)),
        make_labeled_tile(lum_gray, "2. Luminance (Y)", "0.299R + 0.587G + 0.114B", accent_bgr=(160, 200, 240)),
        make_labeled_tile(blur_gray, "3. Gaussian Blur 3x3", "Low-light ISO noise denoising", accent_bgr=(120, 200, 240)),
        make_labeled_tile(weber_gray, "4. Weber-Log Luminance", f"alpha={w_alpha_str} | dark boost", accent_bgr=(80, 220, 255)),
        make_labeled_tile(sx_heat, "5. |Sobel X| Gradient", "Vertical edge responses", accent_bgr=(255, 180, 60)),
        make_labeled_tile(sy_heat, "6. |Sobel Y| Gradient", "Horizontal edge responses", accent_bgr=(255, 150, 60)),
        make_labeled_tile(lap_heat, "7. |Laplacian| Gradient", "Fine corners & contours", accent_bgr=(255, 110, 140)),
        make_labeled_tile(fused_heat, "8. Fused Gradient Mag", f"w={[round(v, 2) for v in params['channel_fusion_weights']]}", accent_bgr=(210, 110, 255)),
        make_labeled_tile(prio_heat, "9. Edge Priority Map", f"sig({params['scale']:.2f}*norm + {params['offset']:.2f})", accent_bgr=(80, 230, 140)),
        make_labeled_tile(prio_ov, "10. Edge Priority Overlay", f"mean={params['edge_priority_mean']:.3f} p90={params['edge_priority_p90']:.3f}", accent_bgr=(60, 220, 255)),
    ]
    grid = assemble_grid(
        tiles,
        ncols=5,
        banner_title=f"EdgeExtractor Structural Edge Priority Pipeline - {img_name}",
        banner_sub=f"weber_alpha={w_alpha_str} | scale={params['scale']:.3f} | offset={params['offset']:.3f} | fusion_w={params['channel_fusion_weights']}",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_edgesab_attention_details(
    save_path: str,
    edge_dir: str,
    display_bgr: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    edgesab_traces: "OrderedDict[str, Dict[str, Any]]",
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Create `06_edgesab_attention_details.png` and optionally dump raw EdgeSAB maps under `edge/`."""
    if not edgesab_traces:
        return
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(330, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)
    th, tw = base_tile.shape[:2]

    by_idx = {r["layer_idx"]: r for r in records.values() if not r["is_sublayer"] and "tensor" in r}
    pre_idx_map = {15: 4, 17: 16, 23: 16, 24: 19, 25: 22}
    scale_names = {15: "P3/8 Skip (L15)", 17: "P3/8 Inline (L17)", 23: "P3/8 (L23)", 24: "P4/16 (L24)", 25: "P5/32 (L25)"}

    tiles: List[np.ndarray] = []
    for label, tr in edgesab_traces.items():
        l_idx = tr["layer_idx"]
        maps = tr["maps"]
        st = tr["stats"]
        pre_idx = pre_idx_map.get(l_idx)
        pre_rec = by_idx.get(pre_idx) if pre_idx is not None else None
        pre_map = feature_to_spatial_maps(pre_rec["tensor"])["l2"] if pre_rec is not None else maps["x_in"]

        # Shared min/max range between Pre-EdgeSAB and Post-EdgeSAB makes residual gain directly visible
        shared_vmax = max(float(np.percentile(pre_map, args.clip_percentile)), float(np.percentile(maps["output"], args.clip_percentile)), 1e-6)
        pre_ov, _ = render_heatmap_overlay(base_tile, pre_map, imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, shared_vmax))
        es_ov, es_heat = render_heatmap_overlay(base_tile, maps["edge_scaled"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        eg_ov, eg_heat = render_heatmap_overlay(base_tile, maps["edge_gate"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        attn_panel = render_attn_heads_panel(tr["attn_heads"], (tw, th), cmap_id)
        res_ov, _ = render_heatmap_overlay(base_tile, maps["residual_delta"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)
        post_ov, post_heat = render_heatmap_overlay(base_tile, maps["output"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, shared_vmax))

        sname = scale_names.get(l_idx, label)
        tiles.extend([
            make_labeled_tile(pre_ov, f"{sname} Pre-EdgeSAB (L{pre_idx if pre_idx is not None else l_idx:02d})", f"L2_in={np.sqrt((pre_map**2).mean()):.3f}", accent_bgr=(230, 210, 80)),
            make_labeled_tile(es_ov, f"{sname} Scaled Edge", f"edge_scale={st['edge_scale']:.3f} | 0.5*Bil+Max", accent_bgr=(60, 220, 255)),
            make_labeled_tile(eg_ov, f"{sname} MultiScaleEdgeGate", f"mean={st['edge_gate_mean']:.3f} std={st['edge_gate_std']:.3f}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(attn_panel, f"{sname} Channel Attn ({st['num_heads']}H)", f"temp={[round(v, 2) for v in st['temperature']]}", accent_bgr=(210, 110, 255)),
            make_labeled_tile(res_ov, f"{sname} Residual |gamma*out|", f"|gamma|_max={st['gamma_abs_max']:.3f} | ratio={st['residual_l2_ratio']:.3f}", accent_bgr=(255, 160, 60)),
            make_labeled_tile(post_ov, f"{sname} Post-EdgeSAB (L{l_idx:02d})", f"cos(in,out)={st['cos_sim_in_out']:.4f}", accent_bgr=(255, 80, 220)),
        ])

        if not args.no_edge_raw:
            sub_dir = os.path.join(edge_dir, label)
            os.makedirs(sub_dir, exist_ok=True)
            _, xin_heat = render_heatmap_overlay(base_tile, maps["x_in"], imgsz, crop_box, cmap_id, alpha=1.0, clip_percentile=args.clip_percentile)
            _, ew_heat = render_heatmap_overlay(base_tile, maps["edge_weight"], imgsz, crop_box, cmap_id, alpha=1.0, clip_percentile=args.clip_percentile)
            _, q_heat = render_heatmap_overlay(base_tile, maps["q_edge_mod"], imgsz, crop_box, cmap_id, alpha=1.0, clip_percentile=args.clip_percentile)
            _, k_heat = render_heatmap_overlay(base_tile, maps["k_edge_mod"], imgsz, crop_box, cmap_id, alpha=1.0, clip_percentile=args.clip_percentile)
            pre_g_ov, _ = render_heatmap_overlay(base_tile, maps["out_pre_gate"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)
            gated_ov, _ = render_heatmap_overlay(base_tile, maps["out_gated"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)
            raw_files = {
                "x_in.png": xin_heat,
                "edge_scaled.png": es_heat,
                "edge_weight.png": ew_heat,
                "q_edge_mod.png": q_heat,
                "k_edge_mod.png": k_heat,
                "attn_matrix_heads.png": attn_panel,
                "out_pre_gate_overlay.png": pre_g_ov,
                "edge_gate.png": eg_heat,
                "edge_gate_overlay.png": eg_ov,
                "out_gated_overlay.png": gated_ov,
                "residual_delta_overlay.png": res_ov,
                "output.png": post_heat,
                "output_overlay.png": post_ov,
            }
            for fname, img in raw_files.items():
                cv2.imwrite(os.path.join(sub_dir, fname), img)

    row_desc = ", ".join(scale_names.get(tr["layer_idx"], lbl) for lbl, tr in edgesab_traces.items())
    grid = assemble_grid(
        tiles,
        ncols=6,
        banner_title=f"EdgeSAB Edge-Guided Channel Self-Attention Internals - {img_name}",
        banner_sub=f"Rows: {row_desc} | Cols: Pre-Attn -> Scaled Edge -> EdgeGate -> Attn Heads -> Residual -> Output",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_per_layer_channel_sheets(
    layers_dir: str,
    display_bgr: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
) -> int:
    """Save detailed per-layer channel grids + spatial Mean/Max/L2 maps into `layers/`."""
    layer_filter = args.layers.strip().lower()
    if layer_filter == "none":
        return 0

    selected_indices: Optional[set] = None
    if layer_filter != "all":
        selected_indices = {int(tok.strip()) for tok in layer_filter.split(",") if tok.strip().isdigit()}

    os.makedirs(layers_dir, exist_ok=True)
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(320, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)

    saved = 0
    for rec in records.values():
        if "tensor" not in rec or rec["layer_idx"] < 0:
            continue
        if selected_indices is not None and rec["layer_idx"] not in selected_indices:
            continue

        t = rec["tensor"]
        if t.ndim != 4:
            continue
        f = t[0]
        c, h_f, w_f = f.shape
        maps = feature_to_spatial_maps(t)
        stats = compute_tensor_stats(t)

        l2_ov, l2_heat = render_heatmap_overlay(base_tile, maps["l2"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)
        mean_ov, _ = render_heatmap_overlay(base_tile, maps["mean"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)
        max_ov, _ = render_heatmap_overlay(base_tile, maps["max"], imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=args.clip_percentile)

        tiles: List[np.ndarray] = [
            make_labeled_tile(base_tile, "Input Display", f"Layer: {rec['key']}", accent_bgr=(180, 180, 180)),
            make_labeled_tile(l2_ov, "L2 Energy Overlay", f"L2={stats['l2']:.3f} std={stats['std']:.3f}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(l2_heat, "L2 Energy Map (Raw)", f"shape=[{c}, {h_f}, {w_f}]", accent_bgr=(80, 210, 230)),
            make_labeled_tile(max_ov, "Channel Max Overlay", f"max={stats['max']:.3f} min={stats['min']:.3f}", accent_bgr=(255, 160, 60)),
            make_labeled_tile(mean_ov, "Channel Mean Overlay", f"mean={stats['mean']:.3f} sp={stats['sparsity']:.1f}%", accent_bgr=(240, 110, 210)),
        ]

        k_ch = min(max(0, args.channels), c)
        if k_ch > 0:
            ch_energy = torch.sqrt((f.view(c, -1) ** 2).mean(dim=1))
            top_indices = torch.topk(ch_energy, k=k_ch).indices.tolist()
            for ch_idx in top_indices:
                ch_map = f[ch_idx].numpy()
                _, ch_heat = render_heatmap_overlay(
                    base_tile, ch_map, imgsz, crop_box, cmap_id,
                    alpha=1.0, clip_percentile=args.clip_percentile,
                )
                tiles.append(
                    make_labeled_tile(
                        ch_heat,
                        f"Ch #{ch_idx:03d}",
                        f"mean={ch_map.mean():.2f} max={ch_map.max():.2f}",
                        accent_bgr=(110, 140, 200),
                    )
                )

        ncols = 5 if len(tiles) >= 10 else min(5, len(tiles))
        safe_name = rec["key"].replace("/", "_").replace(" ", "_")
        grid = assemble_grid(
            tiles,
            ncols=ncols,
            banner_title=f"Layer Detail: {rec['key']} ({rec['mod_type']})",
            banner_sub=f"Stage: {rec['stage']} | Output Shape: (1, {c}, {h_f}, {w_f}) | Top {k_ch} channels by L2 energy",
            colormap_id=cmap_id,
        )
        cv2.imwrite(os.path.join(layers_dir, f"{safe_name}.png"), grid)
        saved += 1

    return saved


# ==================================================================================================
# 8. Side-by-Side Comparison Visualizers (Base vs Enhanced YOLOv11)
# ==================================================================================================

def save_compare_predictions_and_edge(
    save_path: str,
    display_bgr: np.ndarray,
    gt_bgr: Optional[np.ndarray],
    base_res: Dict[str, Any],
    enh_res: Dict[str, Any],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Create `compare/00_compare_predictions_and_edge.png`."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = args.panel_width
    base_tile = resize_to_panel(display_bgr, pw)

    tiles = [
        make_labeled_tile(base_tile, "Input Image (Display)", f"Orig {display_bgr.shape[1]}x{display_bgr.shape[0]} | vis_gamma={args.vis_gamma:.2f}", accent_bgr=(180, 180, 180)),
    ]
    if gt_bgr is not None:
        tiles.append(make_labeled_tile(resize_to_panel(gt_bgr, pw), "Ground Truth", "Dataset annotation", accent_bgr=(80, 220, 255)))

    enh_mode = enh_res["meta"].get("enhanced_mode", "full")
    enh_suffix = (
        "-Skip-P3"
        if enh_mode == "p3_skip"
        else ("-P3-Inline" if enh_mode == "p3_inline" else ("-P3" if enh_mode == "p3" else ""))
    )
    tiles.extend([
        make_labeled_tile(
            resize_to_panel(base_res["pred_bgr"], pw),
            f"YOLOv11 Base Preds ({base_res['n_preds']})",
            f"{base_res['model_name']} (ep={base_res['meta']['epoch']}, 24L)",
            accent_bgr=(50, 170, 255),
        ),
        make_labeled_tile(
            resize_to_panel(enh_res["pred_bgr"], pw),
            f"Enhanced YOLOv11 Preds ({enh_res['n_preds']})",
            f"{enh_res['model_name']} (ep={enh_res['meta']['epoch']}, {enh_res['meta']['num_layers']}L+EdgeSAB{enh_suffix})",
            accent_bgr=(80, 230, 140),
        ),
    ])

    if enh_res.get("edge_trace") is not None:
        prio = enh_res["edge_trace"]["maps"]["edge_priority"]
        ep_params = enh_res["edge_trace"]["params"]
        prio_ov, _ = render_heatmap_overlay(base_tile, prio, imgsz, crop_box, cmap_id, alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0))
        tiles.append(
            make_labeled_tile(
                prio_ov,
                "EdgeExtractor Priority",
                f"mean={ep_params['edge_priority_mean']:.3f} p90={ep_params['edge_priority_p90']:.3f}",
                accent_bgr=(60, 220, 255),
            )
        )

    grid = assemble_grid(
        tiles,
        ncols=len(tiles),
        banner_title=f"Base vs Enhanced YOLOv11 Detections & Edge Priority - {img_name}",
        banner_sub=f"conf_thres={args.conf_thres:.2f} | iou_thres={args.iou_thres:.2f} | Compare detection recall & low-light edge focus",
        colormap_id=cmap_id,
    )
    cv2.imwrite(save_path, grid)


def save_compare_multiscale_features(
    save_path: str,
    display_bgr: np.ndarray,
    base_res: Dict[str, Any],
    enh_res: Dict[str, Any],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Create `compare/01_compare_multiscale_features.png` across P3/8, P4/16, P5/32."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(280, min(360, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)

    base_by_idx = {r["layer_idx"]: r for r in base_res["records"].values() if not r["is_sublayer"] and "tensor" in r}
    enh_by_idx = {r["layer_idx"]: r for r in enh_res["records"].values() if not r["is_sublayer"] and "tensor" in r}
    enh_edge_by_idx = {tr["layer_idx"]: tr for tr in enh_res["edgesab_traces"].values()}

    enh_mode = enh_res["meta"].get("enhanced_mode", "full")
    if enh_mode == "p3_skip":
        # (scale_label, base_pan_idx, enh_mid_idx, enh_final_idx, sab_trace_idx_or_None)
        scale_rows = [
            ("P3/8 (Small)", 16, 15, 17, 15),
            ("P4/16 (Medium)", 19, 6, 20, None),
            ("P5/32 (Large)", 22, 10, 23, None),
        ]
    elif enh_mode == "p3_inline":
        scale_rows = [
            ("P3/8 (Small)", 16, 16, 17, 17),
            ("P4/16 (Medium)", 19, 20, 20, None),
            ("P5/32 (Large)", 22, 23, 23, None),
        ]
    else:
        scale_rows = [
            ("P3/8 (Small)", 16, 16, 23 if 23 in enh_by_idx else 16, 23 if 23 in enh_by_idx else None),
            ("P4/16 (Medium)", 19, 19, 24 if 24 in enh_by_idx else 19, 24 if 24 in enh_by_idx else None),
            ("P5/32 (Large)", 22, 22, 25 if 25 in enh_by_idx else 22, 25 if 25 in enh_by_idx else None),
        ]

    tiles: List[np.ndarray] = []
    for s_label, base_pan_idx, enh_mid_idx, enh_final_idx, sab_idx in scale_rows:
        b_map = feature_to_spatial_maps(base_by_idx[base_pan_idx]["tensor"])["l2"]
        e_mid_t = enh_by_idx[enh_mid_idx]["tensor"]
        e_mid_map = feature_to_spatial_maps(e_mid_t)["l2"]
        has_sab = sab_idx is not None and sab_idx in enh_by_idx
        e_final_t = enh_by_idx[enh_final_idx]["tensor"]
        e_post_map = feature_to_spatial_maps(e_final_t)["l2"]
        diff_map = e_post_map - b_map

        # Normalize Base, Mid, and Final Enhanced on identical scale per row for fair visual comparison
        vmax = max(
            float(np.percentile(b_map, args.clip_percentile)),
            float(np.percentile(e_mid_map, args.clip_percentile)),
            float(np.percentile(e_post_map, args.clip_percentile)),
            1e-6,
        )
        b_ov, _ = render_heatmap_overlay(base_tile, b_map, imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, vmax))
        ep_ov, _ = render_heatmap_overlay(base_tile, e_mid_map, imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, vmax))
        eo_ov, _ = render_heatmap_overlay(base_tile, e_post_map, imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, vmax))
        diff_ov, _, diff_vmax = render_signed_diff_map(base_tile, diff_map, imgsz, crop_box)

        tr = enh_edge_by_idx.get(sab_idx) if sab_idx is not None else None
        if tr is not None:
            e_scaled = tr["maps"]["edge_scaled"]
        elif enh_res.get("edge_trace") is not None:
            em = enh_res["edge_trace"]["priority_tensor"]
            hf, wf = e_final_t.shape[2], e_final_t.shape[3]
            e_bil = F.interpolate(em, size=(hf, wf), mode="bilinear", align_corners=False)
            e_max = F.adaptive_max_pool2d(em, output_size=(hf, wf))
            e_scaled = (0.5 * e_bil + 0.5 * e_max)[0, 0].numpy()
        else:
            e_scaled = np.zeros_like(b_map)
        es_ov, _ = render_heatmap_overlay(base_tile, e_scaled, imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))

        b_l2 = float(np.sqrt((b_map ** 2).mean()))
        e_l2 = float(np.sqrt((e_post_map ** 2).mean()))
        if enh_mode == "p3_skip":
            mid_title = f"{s_label} Enh Skip EdgeSAB (L{enh_mid_idx:02d})" if has_sab else f"{s_label} Enh Backbone (L{enh_mid_idx:02d})"
            mid_accent = (255, 80, 220) if has_sab else (50, 170, 255)
            post_title = f"{s_label} Enh Neck Out (L{enh_final_idx:02d})"
            post_sub = f"L2={e_l2:.3f} | Pre-Concat EdgeSAB" if has_sab else f"L2={e_l2:.3f} | Enriched via Skip P3"
            post_accent = (80, 230, 140)
        else:
            mid_title = f"{s_label} Enh Pre-Attn (L{enh_mid_idx:02d})" if has_sab else f"{s_label} Enh PAN (L{enh_mid_idx:02d})"
            mid_accent = (230, 210, 80)
            post_title = f"{s_label} Enh EdgeSAB (L{enh_final_idx:02d})" if has_sab else f"{s_label} Enh PAN Out (L{enh_final_idx:02d})"
            post_sub = f"L2={e_l2:.3f}" if has_sab else (f"L2={e_l2:.3f} | Enriched via P3" if enh_mode == "p3_inline" else f"L2={e_l2:.3f} | No EdgeSAB")
            post_accent = (255, 80, 220) if has_sab else (80, 230, 140)
        tiles.extend([
            make_labeled_tile(b_ov, f"{s_label} Base Neck (L{base_pan_idx:02d})", f"L2={b_l2:.3f} | shared scale", accent_bgr=(50, 170, 255)),
            make_labeled_tile(ep_ov, mid_title, f"L2={np.sqrt((e_mid_map**2).mean()):.3f}", accent_bgr=mid_accent),
            make_labeled_tile(eo_ov, post_title, post_sub, accent_bgr=post_accent),
            make_labeled_tile(diff_ov, f"{s_label} Delta (Enh - Base)", f"mean={diff_map.mean():+.3f} | +-={diff_vmax:.2f}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(es_ov, f"{s_label} Scaled Edge Map", f"mean={e_scaled.mean():.3f} | max={e_scaled.max():.2f}", accent_bgr=(60, 220, 255)),
        ])

    grid = assemble_grid(
        tiles,
        ncols=5,
        banner_title=f"Multi-Scale Feature Comparison: Base vs Enhanced YOLOv11 - {img_name}",
        banner_sub="Rows: P3/8, P4/16, P5/32 | Col 1-3 share identical intensity range per row | Col 4: Red(+)=Enh boosted, Blue(-)=Suppressed",
        colormap_id=cmap_id,
        cbar_mode="both",
    )
    cv2.imwrite(save_path, grid)


def save_compare_detect_heads(
    save_path: str,
    display_bgr: np.ndarray,
    base_res: Dict[str, Any],
    enh_res: Dict[str, Any],
    imgsz: int,
    crop_box: Tuple[int, int, int, int],
    args: argparse.Namespace,
    img_name: str,
) -> None:
    """Create `compare/02_compare_detect_heads.png` comparing Class Confidence & DFL Certainty."""
    b_det = base_res["detect_info"]
    e_det = enh_res["detect_info"]
    if not b_det or not e_det:
        return

    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(330, args.panel_width))
    base_tile = resize_to_panel(display_bgr, pw)

    tiles: List[np.ndarray] = []
    for bd, ed in zip(b_det, e_det):
        sname = bd["name"]
        cls_diff = ed["cls_max"] - bd["cls_max"]
        dfl_diff = ed["box_certainty"] - bd["box_certainty"]

        b_cls_ov, _ = render_heatmap_overlay(base_tile, bd["cls_max"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        e_cls_ov, _ = render_heatmap_overlay(base_tile, ed["cls_max"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        c_diff_ov, _, c_vmax = render_signed_diff_map(base_tile, cls_diff, imgsz, crop_box)

        b_dfl_ov, _ = render_heatmap_overlay(base_tile, bd["box_certainty"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        e_dfl_ov, _ = render_heatmap_overlay(base_tile, ed["box_certainty"], imgsz, crop_box, cmap_id, alpha=args.alpha, fixed_range=(0.0, 1.0))
        d_diff_ov, _, d_vmax = render_signed_diff_map(base_tile, dfl_diff, imgsz, crop_box)

        tiles.extend([
            make_labeled_tile(b_cls_ov, f"{sname} Base Cls", f"max={bd['cls_max'].max():.3f} top={bd['top1_name']}", accent_bgr=(50, 170, 255)),
            make_labeled_tile(e_cls_ov, f"{sname} Enh Cls", f"max={ed['cls_max'].max():.3f} top={ed['top1_name']}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(c_diff_ov, f"{sname} Cls Delta", f"max+={cls_diff.max():+.3f} min-={cls_diff.min():+.3f} (+-={c_vmax:.2f})", accent_bgr=(255, 80, 220)),
            make_labeled_tile(b_dfl_ov, f"{sname} Base DFL", f"max={bd['box_certainty'].max():.3f} mean={bd['box_certainty'].mean():.3f}", accent_bgr=(50, 170, 255)),
            make_labeled_tile(e_dfl_ov, f"{sname} Enh DFL", f"max={ed['box_certainty'].max():.3f} mean={ed['box_certainty'].mean():.3f}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(d_diff_ov, f"{sname} DFL Delta", f"mean={dfl_diff.mean():+.3f} (+-={d_vmax:.2f})", accent_bgr=(255, 160, 60)),
        ])

    grid = assemble_grid(
        tiles,
        ncols=6,
        banner_title=f"Detect Head Comparison (Class Confidence & DFL Box Certainty) - {img_name}",
        banner_sub="Rows: P3/8, P4/16, P5/32 | Cols 1-3: Max Class Conf | Cols 4-6: DFL Box Certainty | Red(+)=Enhanced higher, Blue(-)=Base higher",
        colormap_id=cmap_id,
        cbar_mode="both",
    )
    cv2.imwrite(save_path, grid)


def save_compare_activation_stats(
    save_path: str,
    base_res: Dict[str, Any],
    enh_res: Dict[str, Any],
    img_name: str,
) -> None:
    """Create `compare/03_compare_activation_stats.png` with Base vs Enhanced layer curves & head bars."""
    b_recs = [r for r in base_res["records"].values() if not r["is_sublayer"] and "tensor" in r and r["layer_idx"] >= 0]
    e_recs = [r for r in enh_res["records"].values() if not r["is_sublayer"] and "tensor" in r and r["layer_idx"] >= 0]
    if not b_recs or not e_recs:
        return

    b_stats = {r["layer_idx"]: compute_tensor_stats(r["tensor"]) for r in b_recs}
    e_stats = {r["layer_idx"]: compute_tensor_stats(r["tensor"]) for r in e_recs}
    all_idx = sorted(set(b_stats.keys()) | set(e_stats.keys()))
    e_type_by_idx = {r["layer_idx"]: r["mod_type"] for r in e_recs}
    x_labels = [f"L{i:02d}\n{e_type_by_idx.get(i, '')}" for i in all_idx]

    b_l2 = [b_stats[i]["l2"] if i in b_stats else np.nan for i in all_idx]
    e_l2 = [e_stats[i]["l2"] if i in e_stats else np.nan for i in all_idx]
    b_sp = [b_stats[i]["sparsity"] if i in b_stats else np.nan for i in all_idx]
    e_sp = [e_stats[i]["sparsity"] if i in e_stats else np.nan for i in all_idx]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(all_idx))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(14, len(all_idx) * 0.62), 11))
        fig.patch.set_facecolor("#121318")
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#191b22")
            ax.tick_params(colors="#d8dce6", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#3a3e4d")
            ax.grid(True, linestyle="--", alpha=0.25, color="#8a92a8")

        ax1.plot(x, b_l2, marker="o", linewidth=2.0, color="#56b4ff", label="YOLOv11 Base L2 Energy")
        ax1.plot(x, e_l2, marker="s", linewidth=2.0, color="#ff5cc8", label="Enhanced YOLOv11 L2 Energy")
        sab_idx_set = {r["layer_idx"] for r in e_recs if r.get("mod_type") in {"EdgeSAB", "SAB"}}
        sab_positions = [pos for pos, idx_val in enumerate(all_idx) if idx_val in sab_idx_set]
        if sab_positions:
            s_lo, s_hi = min(sab_positions), max(sab_positions)
            l_lo, l_hi = all_idx[s_lo], all_idx[s_hi]
            sab_prefix = (
                "Skip-P3 EdgeSAB"
                if sab_idx_set == {15}
                else ("Inline EdgeSAB" if sab_idx_set == {17} else "Pre-Head EdgeSAB")
            )
            sab_lbl = f"{sab_prefix} (L{l_lo}" + (f"-L{l_hi})" if l_hi > l_lo else ")")
            ax1.axvspan(s_lo - 0.5, s_hi + 0.5, color="#ff5cc8", alpha=0.08, label=sab_lbl)
        ax1.set_ylabel("L2 Energy", color="#d8dce6", fontsize=9)
        ax1.set_title(f"Base vs Enhanced YOLOv11 Layer & Head Quantitative Comparison ({img_name})", color="#f0f4ff", fontsize=11, pad=10)
        ax1.legend(loc="upper left", facecolor="#222530", edgecolor="#404556", labelcolor="#e6eaf2", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7.5)

        w = 0.36
        ax2.bar(x - w / 2, [0.0 if np.isnan(v) else v for v in b_sp], width=w, color="#56b4ff", alpha=0.85, label="Base Sparsity (%)")
        ax2.bar(x + w / 2, [0.0 if np.isnan(v) else v for v in e_sp], width=w, color="#4cd98c", alpha=0.85, label="Enhanced Sparsity (%)")
        ax2.set_ylabel("Near-Zero Ratio (%)", color="#d8dce6", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left", facecolor="#222530", edgecolor="#404556", labelcolor="#e6eaf2", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7.5)

        # Panel 3: Detect head peak Class Prob & Mean DFL Certainty across P3/P4/P5
        b_det = base_res["detect_info"]
        e_det = enh_res["detect_info"]
        if b_det and e_det:
            cats = []
            b_vals, e_vals = [], []
            for bd, ed in zip(b_det, e_det):
                short = bd["name"].split()[0]
                cats.extend([f"{short}\nPeak Cls", f"{short}\nPeak DFL", f"{short}\nMean DFL"])
                b_vals.extend([float(bd["cls_max"].max()), float(bd["box_certainty"].max()), float(bd["box_certainty"].mean())])
                e_vals.extend([float(ed["cls_max"].max()), float(ed["box_certainty"].max()), float(ed["box_certainty"].mean())])
            xh = np.arange(len(cats))
            ax3.bar(xh - w / 2, b_vals, width=w, color="#56b4ff", alpha=0.88, label="Base Detect")
            ax3.bar(xh + w / 2, e_vals, width=w, color="#ff5cc8", alpha=0.88, label="Enhanced Detect")
            ax3.set_ylabel("Score [0, 1]", color="#d8dce6", fontsize=9)
            ax3.set_ylim(0, 1.08)
            ax3.set_xticks(xh)
            ax3.set_xticklabels(cats, fontsize=8)
            ax3.legend(loc="upper right", facecolor="#222530", edgecolor="#404556", labelcolor="#e6eaf2", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as exc:
        print(f"[Warn] Could not render comparison matplotlib stats plot ({exc}); skipping.")


# ==================================================================================================
# 9. Single-Model Inference & Visualization Pipeline
# ==================================================================================================

def run_single_model_on_image(
    model: nn.Module,
    recorder: YOLOv11FeatureRecorder,
    model_name: str,
    nc: int,
    meta_info: Dict[str, Any],
    class_names: List[str],
    img_path: str,
    img_bgr: np.ndarray,
    display_bgr: np.ndarray,
    gt_bgr: Optional[np.ndarray],
    tensor: torch.Tensor,
    crop_box: Tuple[int, int, int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
    run_dir: Path,
    args: argparse.Namespace,
    save_layers: bool = True,
) -> Dict[str, Any]:
    """Run forward pass + hooks + tracing and export all visualizations for one model on one image."""
    h0, w0 = img_bgr.shape[:2]
    arch = meta_info["arch"]
    run_dir.mkdir(parents=True, exist_ok=True)

    recorder.clear()
    with torch.no_grad():
        out = model(tensor)
        pred_out = out[0] if isinstance(out, tuple) else out

    dets = non_max_suppression(
        pred_out.float(),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=300,
        nc=nc,
    )[0]

    pred_np = np.zeros((0, 6), dtype=np.float32)
    if len(dets):
        det_c = dets.clone()
        det_c[:, :4] = scale_boxes(tensor.shape[2:], det_c[:, :4], (h0, w0), ratio_pad=ratio_pad)
        pred_np = det_c.cpu().numpy()

    pred_bgr = draw_detections_on_img(display_bgr, pred_np, class_names, show_conf=True)

    detect_rec = next((r for r in recorder.records.values() if "detect_preds" in r), None)
    detect_info = (
        compute_yolov11_detect_heatmaps(detect_rec["detect_preds"], class_names)
        if detect_rec is not None
        else []
    )

    records_copy = OrderedDict(recorder.records)
    img_name = Path(img_path).name

    save_network_overview(
        str(run_dir / "00_network_overview.png"),
        display_bgr, pred_bgr, gt_bgr, records_copy, detect_info,
        args.imgsz, crop_box, args, model_name, meta_info, img_name, len(pred_np),
    )
    save_backbone_vs_neck_scales(
        str(run_dir / "01_backbone_vs_neck_scales.png"),
        display_bgr, records_copy, detect_info,
        args.imgsz, crop_box, args, arch, img_name,
    )
    save_all_layers_progression(
        str(run_dir / "02_all_layers_progression.png"),
        display_bgr, records_copy,
        args.imgsz, crop_box, args, arch, img_name,
    )
    save_detect_head_details(
        str(run_dir / "03_detect_head_details.png"),
        display_bgr, detect_info,
        args.imgsz, crop_box, args, img_name,
    )
    save_activation_stats_chart(
        str(run_dir / "04_layer_activation_stats.png"),
        records_copy, img_name, model_name, arch,
    )

    edge_trace: Optional[Dict[str, Any]] = None
    edgesab_traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    if arch == "enhanced" and (getattr(model, "edge_extractor", None) is not None or recorder.edgesab_modules):
        edge_dir = str(run_dir / "edge")
        os.makedirs(edge_dir, exist_ok=True)
        edge_map_for_sab: Optional[torch.Tensor] = None
        if getattr(model, "edge_extractor", None) is not None:
            edge_trace = trace_edge_extractor(model.edge_extractor, tensor)

            if recorder.edge_map is not None:
                ee_diff = float((edge_trace["priority_tensor"] - recorder.edge_map).abs().max().item())
                if ee_diff > 1e-3:
                    print(f"[Warn] EdgeExtractor trace max diff = {ee_diff:.6f}")

            save_edge_extractor_pipeline(
                str(run_dir / "05_edge_extractor_pipeline.png"),
                edge_dir, display_bgr, edge_trace, args.imgsz, crop_box, args, img_name,
            )
            edge_map_for_sab = recorder.edge_map if recorder.edge_map is not None else edge_trace["priority_tensor"]

        for sab_label, (sab_idx, sab_mod) in recorder.edgesab_modules.items():
            x_in = recorder.edgesab_inputs.get(sab_label)
            if x_in is None:
                continue
            tr = trace_edgesab_module(sab_mod, x_in, edge_map_for_sab)
            tr["layer_idx"] = sab_idx
            hooked_out = records_copy.get(sab_label, {}).get("tensor")
            if hooked_out is not None:
                sab_diff = float((tr["y_out_tensor"] - hooked_out).abs().max().item())
                if sab_diff > 1e-3:
                    print(f"[Warn] {sab_label} trace max diff = {sab_diff:.6f}")
            edgesab_traces[sab_label] = tr

        if edgesab_traces:
            save_edgesab_attention_details(
                str(run_dir / "06_edgesab_attention_details.png"),
                edge_dir, display_bgr, records_copy, edgesab_traces,
                args.imgsz, crop_box, args, img_name,
            )

        stats_payload = {
            "image": img_name,
            "model": model_name,
            "arch": arch,
            "edge_extractor": edge_trace["params"] if edge_trace is not None else None,
            "edge_sab": {lbl: tr["stats"] for lbl, tr in edgesab_traces.items()},
        }
        with open(os.path.join(edge_dir, "edge_stats.json"), "w", encoding="utf-8") as jf:
            json.dump(stats_payload, jf, indent=2)

    n_layers_saved = 0
    if save_layers:
        n_layers_saved = save_per_layer_channel_sheets(
            str(run_dir / "layers"),
            display_bgr, records_copy, args.imgsz, crop_box, args,
        )

    return {
        "model_name": model_name,
        "meta": meta_info,
        "pred_np": pred_np,
        "n_preds": len(pred_np),
        "pred_bgr": pred_bgr,
        "records": records_copy,
        "detect_info": detect_info,
        "edge_trace": edge_trace,
        "edgesab_traces": edgesab_traces,
        "n_layers_saved": n_layers_saved,
    }


# ==================================================================================================
# 10. Main Entrypoint
# ==================================================================================================

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    img_paths = collect_input_images(args.source, max_num=args.num, shuffle=args.shuffle)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    is_compare = bool(args.base_weights.strip() and args.enhanced_weights.strip())

    print("=" * 76)
    print("YOLOv11 FEATURE & ARCHITECTURE VISUALIZATION TOOL")
    print("=" * 76)

    if not is_compare:
        model, model_name, nc, meta_info = load_yolov11_model(
            weights_path=args.weights,
            arch_override=args.arch,
            model_override=args.model,
            cfg_override=args.cfg,
            nc_override=args.nc,
            modular_override=args.modular,
            device=device,
        )
        class_names = resolve_model_class_names(args.classes, meta_info, nc)
        ckpt_stem = Path(args.weights).stem
        stride = int(max(model.stride).item()) if hasattr(model, "stride") else 32

        print(
            f"Mode       : Single ({meta_info['arch'].upper()}, {meta_info['num_layers']} layers, "
            f"modular={meta_info['is_modular']})"
        )
        print(f"Weights    : {args.weights} ({model_name}, ep={meta_info['epoch']}, nc={nc})")
        print(f"Device     : {device}")
        print(f"Classes    : {len(class_names)} {class_names[:6]}{'...' if len(class_names) > 6 else ''}")
        print(f"Images     : {len(img_paths)} file(s)")
        print(f"Output Dir : {out_root.resolve()}")
        print("-" * 76)

        recorder = YOLOv11FeatureRecorder(model, capture_sublayers=args.sublayers)
        try:
            for idx, img_path in enumerate(img_paths, 1):
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"[Warn] Could not read image: {img_path}")
                    continue

                h0, w0 = img_bgr.shape[:2]
                tensor, _unpad, crop_box, ratio_pad = preprocess_image(img_bgr, img_size=args.imgsz, stride=stride)
                tensor = tensor.to(device)
                display_bgr = apply_display_gamma(img_bgr, args.vis_gamma)
                gt_np = load_optional_gt(img_path, (h0, w0), class_names)
                gt_bgr = draw_detections_on_img(display_bgr, gt_np, class_names, show_conf=False) if len(gt_np) else None

                img_stem = Path(img_path).stem
                run_dir = out_root / f"{model_name}_{meta_info['arch']}_{ckpt_stem}_{img_stem}"
                res = run_single_model_on_image(
                    model, recorder, model_name, nc, meta_info, class_names,
                    img_path, img_bgr, display_bgr, gt_bgr, tensor, crop_box, ratio_pad,
                    run_dir, args, save_layers=True,
                )
                n_sheets = 7 if meta_info["arch"] == "enhanced" else 5
                print(
                    f"[{idx}/{len(img_paths)}] {Path(img_path).name} -> {res['n_preds']} dets | "
                    f"{n_sheets} summary sheets + {res['n_layers_saved']} layer detail sheets saved to '{run_dir}'"
                )
        finally:
            recorder.close()
    else:
        base_model, base_name, base_nc, base_meta = load_yolov11_model(
            weights_path=args.base_weights,
            arch_override="base",
            model_override=args.base_model,
            nc_override=args.nc,
            modular_override=args.modular,
            device=device,
        )
        enh_model, enh_name, enh_nc, enh_meta = load_yolov11_model(
            weights_path=args.enhanced_weights,
            arch_override="enhanced",
            model_override=args.enhanced_model,
            nc_override=args.nc,
            modular_override=args.modular,
            device=device,
        )
        base_classes = resolve_model_class_names(args.classes, base_meta, base_nc)
        enh_classes = resolve_model_class_names(args.classes, enh_meta, enh_nc)
        stride = int(max(base_model.stride).item()) if hasattr(base_model, "stride") else 32

        print("Mode       : Comparison (YOLOv11 Base vs Enhanced YOLOv11)")
        print(f"Base       : {args.base_weights} ({base_name}, ep={base_meta['epoch']}, {base_meta['num_layers']}L, nc={base_nc})")
        enh_mode = enh_meta.get("enhanced_mode", "full")
        enh_suffix = (
            "-Skip-P3"
            if enh_mode == "p3_skip"
            else ("-P3-Inline" if enh_mode == "p3_inline" else ("-P3" if enh_mode == "p3" else ""))
        )
        enh_tag = f"{enh_meta['num_layers']}L+EdgeSAB{enh_suffix}"
        print(f"Enhanced   : {args.enhanced_weights} ({enh_name}, ep={enh_meta['epoch']}, {enh_tag}, nc={enh_nc})")
        print(f"Device     : {device}")
        print(f"Images     : {len(img_paths)} file(s)")
        print(f"Output Dir : {out_root.resolve()}")
        print("-" * 76)

        base_rec = YOLOv11FeatureRecorder(base_model, capture_sublayers=args.sublayers)
        enh_rec = YOLOv11FeatureRecorder(enh_model, capture_sublayers=args.sublayers)
        try:
            for idx, img_path in enumerate(img_paths, 1):
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"[Warn] Could not read image: {img_path}")
                    continue

                h0, w0 = img_bgr.shape[:2]
                tensor, _unpad, crop_box, ratio_pad = preprocess_image(img_bgr, img_size=args.imgsz, stride=stride)
                tensor = tensor.to(device)
                display_bgr = apply_display_gamma(img_bgr, args.vis_gamma)
                gt_np = load_optional_gt(img_path, (h0, w0), enh_classes)
                gt_bgr = draw_detections_on_img(display_bgr, gt_np, enh_classes, show_conf=False) if len(gt_np) else None

                img_stem = Path(img_path).stem
                cmp_root = out_root / f"compare_{base_name}_vs_{enh_name}_{img_stem}"
                base_dir = cmp_root / "base"
                enh_dir = cmp_root / "enhanced"
                cmp_dir = cmp_root / "compare"
                cmp_dir.mkdir(parents=True, exist_ok=True)

                save_layers = not args.compare_only
                base_res = run_single_model_on_image(
                    base_model, base_rec, base_name, base_nc, base_meta, base_classes,
                    img_path, img_bgr, display_bgr, gt_bgr, tensor, crop_box, ratio_pad,
                    base_dir, args, save_layers=save_layers,
                )
                enh_res = run_single_model_on_image(
                    enh_model, enh_rec, enh_name, enh_nc, enh_meta, enh_classes,
                    img_path, img_bgr, display_bgr, gt_bgr, tensor, crop_box, ratio_pad,
                    enh_dir, args, save_layers=save_layers,
                )

                img_name = Path(img_path).name
                save_compare_predictions_and_edge(
                    str(cmp_dir / "00_compare_predictions_and_edge.png"),
                    display_bgr, gt_bgr, base_res, enh_res, args.imgsz, crop_box, args, img_name,
                )
                save_compare_multiscale_features(
                    str(cmp_dir / "01_compare_multiscale_features.png"),
                    display_bgr, base_res, enh_res, args.imgsz, crop_box, args, img_name,
                )
                save_compare_detect_heads(
                    str(cmp_dir / "02_compare_detect_heads.png"),
                    display_bgr, base_res, enh_res, args.imgsz, crop_box, args, img_name,
                )
                save_compare_activation_stats(
                    str(cmp_dir / "03_compare_activation_stats.png"),
                    base_res, enh_res, img_name,
                )

                print(
                    f"[{idx}/{len(img_paths)}] {img_name} -> Base {base_res['n_preds']} dets vs "
                    f"Enhanced {enh_res['n_preds']} dets | saved to '{cmp_root}'"
                )
        finally:
            base_rec.close()
            enh_rec.close()

    print("=" * 76)
    print(f"[SUCCESS] All YOLOv11 visualizations saved under: {out_root.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
