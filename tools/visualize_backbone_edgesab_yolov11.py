"""
tools/visualize_backbone_edgesab_yolov11.py

Visualize YOLOv11 Backbone features (160 -> 80 -> 40 -> 20) together with full
Low-Light EdgeExtractor + P2 EdgeSAB (160x160) + Skip-P3 SAB (80x80) internals.

Designed for the 25-layer `develop-yolov11-p2-edgesab` architecture (Backbone P2 160x160 EdgeSAB,
standard FPN+PAN Neck without Skip-P3 SAB) and backward-compatible with 24L Base, 25L Skip-P3,
26L P2-EdgeSAB+Skip-P3-SAB, and modular YOLOv11 checkpoints:
  - EdgeExtractor (640 -> 160): Luminance, Weber-Log compression, Sobel/Laplacian fusion,
                                Priority Edge Map, and MultiScaleEdgeGate at 160x160
  - Stage 1 (P2/4,  160x160): L01_Conv -> L02_C3k2 (Pre-EdgeSAB) -> L03_EdgeSAB (Post-EdgeSAB)
  - Stage 2 (P3/8,   80x80 ): L04_Conv -> L05_C3k2 (Skip-P3 Source)
  - Stage 3 (P4/16,  40x40 ): L06_Conv -> L07_C3k2 (Skip-P4 Source)
  - Stage 4 (P5/32,  20x20 ): L08_Conv -> L09_C3k2 -> L10_SPPF -> L11_C2PSA (Skip-P5 Source)

Outputs saved under `<output>/<run_name>/<image_stem>/`:
  - `00_backbone_pyramid_160_to_20.png`    - Key pyramid stages (L02 Pre -> L03 EdgeSAB 160 -> 80 -> 40 -> 20)
  - `01_edge_extractor_and_p2_edgesab.png` - EdgeExtractor pipeline (Weber-Log, Gradients, Priority Map) +
                                             EdgeSAB internals (Q/K Edge Weight, MultiScaleEdgeGate, |L03-L02| Delta)
  - `02_all_backbone_layers_160_to_20.png` - Chronological progression across L01..L11
  - `03_stage4_c2psa_20x20.png`            - Stage 4 SPPF->C2PSA (20x20) refinement (+ Skip-P3 SAB if present)
  - `04_backbone_activation_stats.png`     - Quantitative L2 energy, Max, Sparsity + learned Edge/EdgeSAB scalars
  - `05_backbone_comparison_160_to_20.png` - Side-by-side comparison across semantic stages (when `--compare_weights` set)
  - `edge/`                                - Standalone borderless edge & gate maps for papers/slides
  - `layers/<layer_name>/`                 - Standalone maps + Top-K individual channel sheets

Usage Examples:
  # 1. Visualize trained P2-EdgeSAB + Skip-P3-SAB model on an ExDark image
  python tools/visualize_backbone_edgesab_yolov11.py ^
      --weights weights/training/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65

  # 2. Visualize with pretrained COCO weights + initialized EdgeExtractor/EdgeSAB
  python tools/visualize_backbone_edgesab_yolov11.py ^
      --weights auto --model yolov11s --source dataset/ExDark/images/test/2015_01138.jpg

  # 3. Compare two checkpoints across semantic scales (160 -> 80 -> 40 -> 20)
  python tools/visualize_backbone_edgesab_yolov11.py ^
      --weights model_results/yolov11s/best_mAP50.pth ^
      --compare_weights weights/training/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_01138.jpg --vis_gamma 0.65
"""

import argparse
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
    build_yolo_model,
    get_model_family,
    infer_model_from_checkpoint,
    load_yolo_checkpoint,
    non_max_suppression,
)
from net.Yolov11.common import EdgeExtractor, EdgeSAB, SAB  # noqa: E402
from net.Yolov11.yolo import DetectionModel, YOLOV11_SCALES  # noqa: E402
from net.Yolov11.utils import (  # noqa: E402
    _register_ultralytics_unpickle_aliases,
    de_parallel,
    scale_boxes,
)


# ==================================================================================================
# 1. Layer Semantic Metadata & Color Palettes
# ==================================================================================================

STAGE_ACCENT_BGR: Dict[str, Tuple[int, int, int]] = {
    "edge": (60, 220, 255),     # Bright Cyan-Gold for EdgeExtractor / EdgeGate
    "p1": (200, 200, 90),       # P1/2  (320)
    "p2": (255, 195, 55),       # P2/4  (160)
    "p2_edge": (255, 110, 225), # P2/4  (160) EdgeSAB
    "p3": (95, 225, 110),       # P3/8  (80)
    "p3_sab": (130, 255, 190),  # P3/8  (80) Skip-P3 SAB
    "p4": (55, 175, 255),       # P4/16 (40)
    "p5": (120, 95, 255),       # P5/32 (20)
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
        description="Visualize YOLOv11 Backbone (160->20), EdgeExtractor, P2 EdgeSAB, and Skip-P3 SAB."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="auto",
        help="Path to YOLOv11 checkpoint (.pth/.pt), or 'auto' to use/download official COCO weights.",
    )
    parser.add_argument(
        "--compare_weights",
        type=str,
        default="",
        help="Optional second checkpoint (.pth/.pt) for side-by-side 160->20 semantic comparison.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="YOLOv11 variant override (yolov11n/s/m/l/x). Auto-inferred from checkpoint if omitted.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional YAML config path override.",
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
        help="Use modular YOLOv11 (backbone/neck/head). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "p2_edgesab", "p2_edgesab_skip_p3", "skip_p3_sab", "skip_p3_edgesab", "base"],
        help="Architecture variant ('auto' matches each checkpoint exactly, or 25L p2_edgesab for 'auto'/'none' weights).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Input image file or directory. Auto-selects from dataset/ExDark or dataset/DarkFace if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization/backbone_edgesab_yolov11",
        help="Root output directory (default: visualization/backbone_edgesab_yolov11).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input letterbox resolution (default: 640 -> scales 160, 80, 40, 20).",
    )
    parser.add_argument(
        "--include_stem",
        action="store_true",
        help="Also include L00 Stem (P1/2, 320x320) in layer progression.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Max images when --source is a directory (default: 1, -1 for all).",
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
        help="Confidence threshold for detection preview tile (default: 0.25).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold for detection preview tile (default: 0.45).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=16,
        help="Number of top active channels per layer sheet (default: 16, 0 to skip sheets).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=list(COLORMAP_TABLE.keys()),
        help="OpenCV colormap for activation heatmaps (default: turbo).",
    )
    parser.add_argument(
        "--edge_colormap",
        type=str,
        default="inferno",
        choices=list(COLORMAP_TABLE.keys()),
        help="OpenCV colormap for EdgeExtractor & EdgeGate maps (default: inferno).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Heatmap overlay opacity in [0.0, 1.0] (default: 0.50).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.5,
        help="Upper percentile clipping before heatmap normalization (default: 99.5).",
    )
    parser.add_argument(
        "--vis_gamma",
        type=float,
        default=0.65,
        help="Display-only gamma (<1.0 brightens dark background underlay without altering model input; default: 0.65).",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=360,
        help="Display panel width in pixels for summary tiles (default: 360).",
    )
    return parser.parse_args()


# ==================================================================================================
# 2. Model Loading & Dynamic Backbone + EdgeExtractor + EdgeSAB + SAB Recorder
# ==================================================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_class_names(
    classes_arg: str, nc: int, ckpt_names: Optional[List[str]] = None
) -> List[str]:
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

    if ckpt_names and len(ckpt_names) == nc:
        return list(ckpt_names)
    if nc == len(EXDARK_CLASSES):
        return list(EXDARK_CLASSES)
    if nc == len(DARKFACE_CLASSES):
        return list(DARKFACE_CLASSES)
    return [f"cls_{i}" for i in range(nc)]


def _extract_raw_sd_keys(ckpt_meta: Any) -> List[str]:
    if not isinstance(ckpt_meta, dict):
        return []
    raw_sd = None
    if "state_dict" in ckpt_meta and isinstance(ckpt_meta["state_dict"], dict):
        raw_sd = ckpt_meta["state_dict"]
    elif "ema" in ckpt_meta and ckpt_meta["ema"] is not None:
        ema_obj = ckpt_meta["ema"]
        raw_sd = ema_obj.state_dict() if hasattr(ema_obj, "state_dict") else ema_obj
    elif "model" in ckpt_meta and ckpt_meta["model"] is not None:
        m_obj = ckpt_meta["model"]
        raw_sd = m_obj.state_dict() if hasattr(m_obj, "state_dict") else m_obj
    elif "model_state_dict" in ckpt_meta and isinstance(ckpt_meta["model_state_dict"], dict):
        raw_sd = ckpt_meta["model_state_dict"]
    else:
        raw_sd = ckpt_meta
    if not isinstance(raw_sd, dict):
        return []
    return [k[7:] if k.startswith("module.") else k for k in raw_sd.keys()]


def detect_is_modular_ckpt(ckpt_meta: Any) -> bool:
    keys = _extract_raw_sd_keys(ckpt_meta)
    return any(k.startswith("backbone.") for k in keys[:30])


def detect_ckpt_arch(ckpt_meta: Any) -> Optional[str]:
    keys = _extract_raw_sd_keys(ckpt_meta)
    if not keys:
        return None
    if any(k.startswith("backbone.") for k in keys[:30]):
        return "modular"

    def _has_detect(idx: int) -> bool:
        p = f"model.{idx}."
        return any(k.startswith((f"{p}cv", f"{p}dfl")) for k in keys)

    def _has_sab(idx: int) -> bool:
        p = f"model.{idx}."
        return any(k.startswith((f"{p}q.", f"{p}gamma", f"{p}edge_gate.")) for k in keys)

    if _has_detect(25) and _has_sab(3):
        return "p2_edgesab_skip_p3"
    if _has_detect(24) and _has_sab(3):
        return "p2_edgesab"
    if _has_detect(24) and _has_sab(15):
        if any(k.startswith("model.15.edge_gate.") for k in keys):
            return "skip_p3_edgesab"
        return "skip_p3_sab"
    if _has_detect(23):
        return "base"
    return None


def _build_legacy_cfg_dict(model_name: str, nc: int, arch: str) -> Dict[str, Any]:
    variant = model_name.lower().replace("yolov11", "").replace("yolo11", "")
    scales = {k: list(v) for k, v in YOLOV11_SCALES.items()}
    if arch == "p2_edgesab_skip_p3":
        backbone_p2 = [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "EdgeSAB", [4, False, 2]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ]
        head_26 = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 7], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [5, 1, "SAB", [4, False, 2]],
            [[15, -1], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 11], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[18, 21, 24], 1, "Detect", ["nc"]],
        ]
        return {"nc": nc, "scale": variant, "scales": scales, "backbone": backbone_p2, "head": head_26}

    backbone_base = [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 2, "C3k2", [256, False, 0.25]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 2, "C3k2", [512, False, 0.25]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
        [-1, 2, "C2PSA", [1024]],
    ]
    if arch == "base":
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[16, 19, 22], 1, "Detect", ["nc"]],
        ]
    else:
        attn_cls = "EdgeSAB" if arch == "skip_p3_edgesab" else "SAB"
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [4, 1, attn_cls, [4, False, 2]],
            [[14, -1], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[17, 20, 23], 1, "Detect", ["nc"]],
        ]
    return {"nc": nc, "scale": variant, "scales": scales, "backbone": backbone_base, "head": head}


def load_yolov11_model(
    weights_path: str,
    model_override: str = "",
    cfg_override: str = "",
    nc_override: Optional[int] = None,
    modular_override: Optional[bool] = None,
    arch_override: str = "auto",
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, str, int, Dict[str, Any]]:
    w_str = (weights_path or "auto").strip()
    is_auto_or_none = w_str.lower() in {"auto", "none", "false", "no"}

    ckpt_meta: Any = None
    if not is_auto_or_none:
        if not os.path.exists(w_str):
            raise FileNotFoundError(f"Weights file not found: {w_str}")
        _register_ultralytics_unpickle_aliases()
        ckpt_meta = torch.load(w_str, map_location="cpu", weights_only=False)

    if model_override.strip():
        model_name = model_override.strip().lower()
    elif ckpt_meta is not None:
        inferred = infer_model_from_checkpoint(ckpt_meta, fallback_path=w_str)
        model_name = inferred if (inferred and get_model_family(inferred) == "yolov11") else "yolov11s"
    else:
        model_name = "yolov11s"

    if get_model_family(model_name) != "yolov11":
        raise ValueError(f"Expected a YOLOv11 variant, got '{model_name}'.")

    nc = nc_override
    if nc is None and isinstance(ckpt_meta, dict):
        if ckpt_meta.get("num_classes") is not None:
            nc = int(ckpt_meta["num_classes"])
        elif hasattr(ckpt_meta.get("model"), "nc"):
            nc = int(ckpt_meta["model"].nc)
        elif isinstance(getattr(ckpt_meta.get("model"), "yaml", None), dict):
            nc = ckpt_meta["model"].yaml.get("nc")
    if nc is None:
        nc = 80 if w_str.lower() == "auto" else 12
    nc = int(nc)

    is_modular = (
        detect_is_modular_ckpt(ckpt_meta) if modular_override is None else bool(modular_override)
    )
    cfg_path = cfg_override if (cfg_override and os.path.exists(cfg_override)) else None

    resolved_arch = arch_override
    if resolved_arch == "auto":
        detected = detect_ckpt_arch(ckpt_meta) if ckpt_meta is not None else None
        resolved_arch = (
            detected
            if detected in {"base", "skip_p3_sab", "skip_p3_edgesab", "p2_edgesab", "p2_edgesab_skip_p3"}
            else "p2_edgesab"
        )

    if cfg_path is not None or is_modular or resolved_arch == "p2_edgesab":
        model = build_yolo_model(
            model_name=model_name,
            num_classes=nc,
            in_channels=3,
            cfg=cfg_path,
            modular=is_modular,
        )
    else:
        cfg_dict = _build_legacy_cfg_dict(model_name, nc, resolved_arch)
        model = DetectionModel(cfg=cfg_dict, ch=3, nc=nc)

    if w_str.lower() not in {"none", "false", "no"}:
        load_yolo_checkpoint(model, w_str, model_name=model_name, verbose=True)

    ckpt_names: List[str] = []
    if isinstance(ckpt_meta, dict) and hasattr(ckpt_meta.get("model"), "names"):
        raw_names = ckpt_meta["model"].names
        if isinstance(raw_names, dict):
            ckpt_names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
        elif isinstance(raw_names, (list, tuple)):
            ckpt_names = [str(x) for x in raw_names]

    if not is_auto_or_none:
        p_obj = Path(w_str)
        parent_name = p_obj.parent.name
        if parent_name and parent_name.lower() not in {".", "weights", "pretrained", "training", "train"}:
            w_label = f"{parent_name}_{p_obj.stem}"
        else:
            w_label = p_obj.stem
    else:
        w_label = f"{model_name}_{w_str.lower()}"

    meta_info = {
        "epoch": ckpt_meta.get("epoch", "pretrained" if w_str.lower() == "auto" else "?")
        if isinstance(ckpt_meta, dict)
        else ("pretrained" if w_str.lower() == "auto" else "?"),
        "is_modular": is_modular,
        "ckpt_names": ckpt_names,
        "weights_label": w_label,
    }
    model = model.to(device).eval()
    return model, model_name, nc, meta_info


@torch.no_grad()
def inspect_edge_extractor_internals(
    edge_extractor: EdgeExtractor, x_img: torch.Tensor
) -> Dict[str, Any]:
    """
    Reproduce `EdgeExtractor.forward` step-by-step to capture every intermediate map:
      - `gray`: Luminance after optional Gaussian blur `(H, W)`
      - `weber`: Luminance after Weber-Fechner log compression `(H, W)`
      - `grad_mag`: Fused Sobel X/Y + Laplacian magnitude normalized to `[0, 1]` `(H, W)`
      - `edge_priority`: Final sigmoid priority map `[0, 1]` `(H, W)`
      - Learned scalars: `weber_alpha`, `scale`, `offset`
    """
    b, c, h, w = x_img.shape
    if c > 1 and edge_extractor.rgb_to_gray and c == 3:
        target = torch.sum(x_img * edge_extractor.rgb_weight.to(dtype=x_img.dtype), dim=1, keepdim=True)
    elif c > 1:
        target = x_img.mean(dim=1, keepdim=True)
    else:
        target = x_img

    if edge_extractor.use_blur:
        pad = edge_extractor.blur_kernel_size // 2
        t_pad = F.pad(target, (pad, pad, pad, pad), mode="replicate")
        target = F.conv2d(t_pad, edge_extractor.gaussian_kernel.to(dtype=target.dtype))
    gray_np = target[0, 0].float().cpu().numpy()

    weber_alpha_val = 0.0
    if getattr(edge_extractor, "weber_log", True):
        eps = 1e-3
        log_min = math.log(eps)
        t_fp32 = target.float()
        t_log = (torch.log(t_fp32.clamp(min=eps)) - log_min) / (-log_min)
        alpha = edge_extractor.weber_alpha.float().clamp(0.0, 1.0)
        weber_alpha_val = float(alpha.item())
        target = (alpha * t_log + (1.0 - alpha) * t_fp32).to(dtype=x_img.dtype)
    weber_np = target[0, 0].float().cpu().numpy()

    t_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(t_pad, edge_extractor.sobel_x.to(dtype=target.dtype))
    gy = F.conv2d(t_pad, edge_extractor.sobel_y.to(dtype=target.dtype))
    gl = F.conv2d(t_pad, edge_extractor.laplacian.to(dtype=target.dtype))
    g_stack = torch.cat([gx, gy, gl], dim=1)
    g_mag = torch.abs(edge_extractor.channel_fusion(g_stack))

    g_flat = g_mag.float().view(b, -1)
    g_min = g_flat.min(dim=1, keepdim=True)[0]
    g_max = g_flat.max(dim=1, keepdim=True)[0]
    g_norm = ((g_flat - g_min) / (g_max - g_min + 1e-6)).view(b, 1, h, w)

    scale_val = float(edge_extractor.scale.float().item())
    offset_val = float(edge_extractor.offset.float().item())
    edge_priority = torch.sigmoid(scale_val * g_norm + offset_val)

    return {
        "gray": gray_np,
        "weber": weber_np,
        "grad_norm": g_norm[0, 0].cpu().numpy(),
        "edge_priority": edge_priority[0, 0].cpu().numpy(),
        "edge_tensor": edge_priority.detach().to(device=x_img.device, dtype=x_img.dtype),
        "weber_alpha": weber_alpha_val,
        "scale": scale_val,
        "offset": offset_val,
    }


@torch.no_grad()
def inspect_edgesab_internals(
    edgesab_mod: EdgeSAB, x_in: torch.Tensor, edge_tensor: torch.Tensor
) -> Dict[str, Any]:
    """Extract `edge_scaled (160x160)`, Q/K edge modulation weight, `MultiScaleEdgeGate`, and residual delta."""
    h, w = x_in.shape[-2:]
    if edge_tensor.shape[-2:] != (h, w):
        e_bil = F.interpolate(edge_tensor, size=(h, w), mode="bilinear", align_corners=False)
        e_max = F.adaptive_max_pool2d(edge_tensor, output_size=(h, w))
        edge_scaled = 0.5 * e_bil + 0.5 * e_max
    else:
        edge_scaled = edge_tensor

    edge_scale_val = float(edgesab_mod.edge_scale.float().item())
    qk_weight = 1.0 + edge_scale_val * edge_scaled.float()
    gate_map = edgesab_mod.edge_gate(edge_scaled.to(dtype=x_in.dtype)).float()
    gamma_vals = edgesab_mod.gamma.detach().float().view(-1)

    return {
        "edge_scaled": edge_scaled[0, 0].float().cpu().numpy(),
        "qk_weight": qk_weight[0, 0].cpu().numpy(),
        "edge_gate_mean": gate_map[0].mean(dim=0).cpu().numpy(),
        "edge_gate_max": gate_map[0].max(dim=0).values.cpu().numpy(),
        "edge_scale": edge_scale_val,
        "gamma_mean": float(gamma_vals.mean().item()),
        "gamma_abs_mean": float(gamma_vals.abs().mean().item()),
        "gamma_max": float(gamma_vals.max().item()),
    }


class YOLOv11BackboneEdgeRecorder:
    """
    Dynamically inspects and hooks all YOLOv11 Backbone layers (including L03 EdgeSAB if present),
    EdgeExtractor, and Lateral Skip-P3 SAB in the Neck, supporting both `DetectionModel` and modular `YOLOv11`.
    """

    def __init__(self, model: nn.Module, include_stem: bool = False):
        self.model = de_parallel(model)
        self.include_stem = include_stem
        self.handles: List[Any] = []
        self.records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.edge_extractor_mod: Optional[EdgeExtractor] = getattr(self.model, "edge_extractor", None)
        self.p2_edgesab_mod: Optional[EdgeSAB] = None
        self.p2_edgesab_input: Optional[torch.Tensor] = None
        self.skip_p3_sab_mod: Optional[nn.Module] = None
        self.stage_slots: Dict[str, str] = {}  # semantic slot -> record key
        self._register()

    def _make_hook(
        self,
        key: str,
        layer_idx: int,
        stage_group: str,
        stride: int,
        accent_key: str,
        is_stage_out: bool,
        capture_input: bool = False,
    ):
        def hook_fn(module, inputs, output):
            if capture_input and inputs and isinstance(inputs[0], torch.Tensor):
                self.p2_edgesab_input = inputs[0].detach().clone()
            if isinstance(output, torch.Tensor):
                self.records[key] = {
                    "key": key,
                    "layer_idx": layer_idx,
                    "stage_group": stage_group,
                    "stride": stride,
                    "accent_key": accent_key,
                    "is_stage_out": is_stage_out,
                    "mod_type": type(module).__name__,
                    "tensor": output.detach().float().cpu(),
                }

        return hook_fn

    def _register(self) -> None:
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            has_p2_edgesab = len(layers) > 3 and isinstance(layers[3], EdgeSAB)

            if has_p2_edgesab:
                # 26-layer P2-EdgeSAB + Skip-P3-SAB architecture
                spec = [
                    (0, "L00_Stem_Conv3x3_P1_2", "P1/2 (Stem)", 2, "p1", True, "p1"),
                    (1, "L01_Conv3x3_P2_4", "P2/4 (Stage1)", 4, "p2", False, ""),
                    (2, "L02_C3k2_P2_4_PreEdgeSAB", "P2/4 (Stage1)", 4, "p2", True, "p2_pre"),
                    (3, "L03_EdgeSAB_P2_4_160", "P2/4 (EdgeSAB)", 4, "p2_edge", True, "p2"),
                    (4, "L04_Conv3x3_P3_8", "P3/8 (Stage2)", 8, "p3", False, ""),
                    (5, "L05_C3k2_P3_8_SkipP3", "P3/8 (Stage2)", 8, "p3", True, "p3"),
                    (6, "L06_Conv3x3_P4_16", "P4/16 (Stage3)", 16, "p4", False, ""),
                    (7, "L07_C3k2_P4_16_SkipP4", "P4/16 (Stage3)", 16, "p4", True, "p4"),
                    (8, "L08_Conv3x3_P5_32", "P5/32 (Stage4)", 32, "p5", False, ""),
                    (9, "L09_C3k2_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_c3k2"),
                    (10, "L10_SPPF_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_sppf"),
                    (11, "L11_C2PSA_P5_32_SkipP5", "P5/32 (Stage4)", 32, "p5", True, "p5"),
                ]
                self.p2_edgesab_mod = layers[3]
                if len(layers) > 16 and isinstance(layers[16], (SAB, EdgeSAB)):
                    spec.append((16, "L16_SAB_SkipP3_80", "Neck (Skip-P3 SAB)", 8, "p3_sab", True, "p3_sab"))
                    self.skip_p3_sab_mod = layers[16]
            else:
                # 24L Base or 25L Skip-P3 architecture
                spec = [
                    (0, "L00_Stem_Conv3x3_P1_2", "P1/2 (Stem)", 2, "p1", True, "p1"),
                    (1, "L01_Conv3x3_P2_4", "P2/4 (Stage1)", 4, "p2", False, ""),
                    (2, "L02_C3k2_P2_4", "P2/4 (Stage1)", 4, "p2", True, "p2"),
                    (3, "L03_Conv3x3_P3_8", "P3/8 (Stage2)", 8, "p3", False, ""),
                    (4, "L04_C3k2_P3_8_SkipP3", "P3/8 (Stage2)", 8, "p3", True, "p3"),
                    (5, "L05_Conv3x3_P4_16", "P4/16 (Stage3)", 16, "p4", False, ""),
                    (6, "L06_C3k2_P4_16_SkipP4", "P4/16 (Stage3)", 16, "p4", True, "p4"),
                    (7, "L07_Conv3x3_P5_32", "P5/32 (Stage4)", 32, "p5", False, ""),
                    (8, "L08_C3k2_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_c3k2"),
                    (9, "L09_SPPF_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_sppf"),
                    (10, "L10_C2PSA_P5_32_SkipP5", "P5/32 (Stage4)", 32, "p5", True, "p5"),
                ]
                if len(layers) > 15 and isinstance(layers[15], (SAB, EdgeSAB)):
                    spec.append((15, "L15_SAB_SkipP3_80", "Neck (Skip-P3 SAB)", 8, "p3_sab", True, "p3_sab"))
                    self.skip_p3_sab_mod = layers[15]

            for idx, key, group, stride, acc, is_out, slot in spec:
                if idx == 0 and not self.include_stem:
                    continue
                if idx < len(layers):
                    cap_in = (layers[idx] is self.p2_edgesab_mod)
                    self.handles.append(
                        layers[idx].register_forward_hook(
                            self._make_hook(key, idx, group, stride, acc, is_out, capture_input=cap_in)
                        )
                    )
                    if slot:
                        self.stage_slots[slot] = key

        elif hasattr(self.model, "backbone"):
            bb = self.model.backbone
            spec_mod: List[Tuple[int, str, str, int, str, bool, str, nn.Module]] = [
                (0, "L00_Stem_Conv3x3_P1_2", "P1/2 (Stem)", 2, "p1", True, "p1", bb.stem),
                (1, "L01_Conv3x3_P2_4", "P2/4 (Stage1)", 4, "p2", False, "", bb.stage1[0]),
                (2, "L02_C3k2_P2_4_PreEdgeSAB", "P2/4 (Stage1)", 4, "p2", True, "p2_pre" if hasattr(bb, "attn_p2") else "p2", bb.stage1[1]),
            ]
            if hasattr(bb, "attn_p2"):
                self.p2_edgesab_mod = bb.attn_p2
                spec_mod.append((3, "L03_EdgeSAB_P2_4_160", "P2/4 (EdgeSAB)", 4, "p2_edge", True, "p2", bb.attn_p2))
            off = 1 if hasattr(bb, "attn_p2") else 0
            spec_mod.extend([
                (3 + off, f"L{3+off:02d}_Conv3x3_P3_8", "P3/8 (Stage2)", 8, "p3", False, "", bb.stage2[0]),
                (4 + off, f"L{4+off:02d}_C3k2_P3_8_SkipP3", "P3/8 (Stage2)", 8, "p3", True, "p3", bb.stage2[1]),
                (5 + off, f"L{5+off:02d}_Conv3x3_P4_16", "P4/16 (Stage3)", 16, "p4", False, "", bb.stage3[0]),
                (6 + off, f"L{6+off:02d}_C3k2_P4_16_SkipP4", "P4/16 (Stage3)", 16, "p4", True, "p4", bb.stage3[1]),
                (7 + off, f"L{7+off:02d}_Conv3x3_P5_32", "P5/32 (Stage4)", 32, "p5", False, "", bb.stage4[0]),
                (8 + off, f"L{8+off:02d}_C3k2_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_c3k2", bb.stage4[1]),
                (9 + off, f"L{9+off:02d}_SPPF_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_sppf", bb.stage4[2]),
                (10 + off, f"L{10+off:02d}_C2PSA_P5_32_SkipP5", "P5/32 (Stage4)", 32, "p5", True, "p5", bb.stage4[3]),
            ])
            if hasattr(self.model, "neck") and hasattr(self.model.neck, "attn_p3"):
                self.skip_p3_sab_mod = self.model.neck.attn_p3
                spec_mod.append((15 + off, f"L{15+off:02d}_SAB_SkipP3_80", "Neck (Skip-P3 SAB)", 8, "p3_sab", True, "p3_sab", self.model.neck.attn_p3))

            for idx, key, group, stride, acc, is_out, slot, mod in spec_mod:
                if idx == 0 and not self.include_stem:
                    continue
                cap_in = (mod is self.p2_edgesab_mod)
                self.handles.append(
                    mod.register_forward_hook(
                        self._make_hook(key, idx, group, stride, acc, is_out, capture_input=cap_in)
                    )
                )
                if slot:
                    self.stage_slots[slot] = key
        else:
            raise AttributeError("Unsupported YOLOv11 model structure.")

    def clear(self) -> None:
        self.records.clear()
        self.p2_edgesab_input = None

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==================================================================================================
# 3. Preprocessing, Projections & Rendering Helpers
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
            PROJECT_ROOT / "dataset" / "DarkFace" / "images" / "test",
            PROJECT_ROOT / "dataset" / "DarkFace" / "images" / "val",
        ]
        imgs = []
        for cand in candidates:
            if cand.is_dir():
                imgs = sorted(str(f) for f in cand.iterdir() if f.is_file() and is_image_file(f.name))
                if imgs:
                    print(f"===> Auto-selected source directory: {cand}")
                    break
        if not imgs:
            raise FileNotFoundError("No --source specified and no default dataset images found.")

    if not imgs:
        raise FileNotFoundError(f"No supported images found in '{source_arg}'.")
    if shuffle:
        rng = np.random.default_rng(42)
        imgs = [imgs[i] for i in rng.permutation(len(imgs))]
    if max_num > 0:
        imgs = imgs[:max_num]
    return imgs


def preprocess_image(
    img_bgr: np.ndarray, img_size: int = 640, stride: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    h0, w0 = img_bgr.shape[:2]
    im_lb, ratio, (dw, dh) = letterbox(
        img_bgr, new_shape=(img_size, img_size), auto=False, scaleup=True, stride=stride
    )
    new_unpad_w = int(round(w0 * ratio[0]))
    new_unpad_h = int(round(h0 * ratio[1]))
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    crop_box = (top, left, top + new_unpad_h, left + new_unpad_w)

    rgb = im_lb[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(tensor).unsqueeze(0), crop_box, (ratio, (dw, dh))


def load_optional_gt(img_path: str, orig_hw: Tuple[int, int], class_names: List[str]) -> np.ndarray:
    lbl_path = img2label_paths([img_path])[0]
    if not os.path.exists(lbl_path):
        return np.zeros((0, 6), dtype=np.float32)
    h0, w0 = orig_hw
    class_to_id = {c: i for i, c in enumerate(class_names)}
    parsed = parse_label_file(lbl_path, img_w=w0, img_h=h0, class_to_id=class_to_id)
    if len(parsed) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    boxes = xywhn2xyxy(parsed[:, 1:5], w=float(w0), h=float(h0))
    return np.hstack((boxes, np.ones((len(parsed), 1), dtype=np.float32), parsed[:, 0:1].astype(np.float32)))


def apply_display_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-4:
        return img_bgr.copy()
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
    return (np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)


def compute_spatial_projections(feat: torch.Tensor) -> Dict[str, Any]:
    f = feat[0].float()
    c, h, w = f.shape
    l2_map = torch.sqrt((f ** 2).mean(dim=0) + 1e-12).numpy()
    mean_map = f.mean(dim=0).numpy()
    max_map = f.max(dim=0).values.numpy()

    flat = f.reshape(c, h * w).T
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    if min(h * w, c) >= 3 and float(flat_centered.std()) > 1e-8:
        try:
            u, s, _ = torch.pca_lowrank(flat_centered, q=min(6, c, h * w), center=False, niter=2)
            proj = (u[:, :3] * s[:3]).numpy()
        except Exception:
            proj = flat_centered[:, :3].numpy()
        for k in range(3):
            if float(np.percentile(proj[:, k], 95) + np.percentile(proj[:, k], 5)) < 0.0:
                proj[:, k] = -proj[:, k]
            p_lo, p_hi = np.percentile(proj[:, k], 1.0), np.percentile(proj[:, k], 99.0)
            proj[:, k] = np.clip((proj[:, k] - p_lo) / (p_hi - p_lo), 0.0, 1.0) if p_hi > p_lo + 1e-8 else 0.0
        pca_bgr = (proj.reshape(h, w, 3) * 255.0).astype(np.uint8)[:, :, ::-1].copy()
    else:
        pca_bgr = cv2.cvtColor(normalize_map_to_uint8(l2_map), cv2.COLOR_GRAY2BGR)

    return {
        "shape": (c, h, w),
        "l2": l2_map,
        "mean": mean_map,
        "max": max_map,
        "pca_bgr": pca_bgr,
        "l2_mean": float(l2_map.mean()),
        "max_val": float(f.max().item()),
        "sparsity": float((f.abs() < 1e-2).float().mean().item()) * 100.0,
    }


def unpad_and_resize_2d(
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    nearest: bool = False,
) -> np.ndarray:
    up = cv2.resize(
        map2d.astype(np.float32),
        (lb_size, lb_size),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC,
    )
    top, left, bottom_y, right_x = crop_box
    cropped = up[top:bottom_y, left:right_x]
    if cropped.size == 0:
        cropped = up
    return cv2.resize(
        cropped, target_wh, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    )


def unpad_and_resize_bgr(
    img_grid_bgr: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    nearest: bool = True,
) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    up = cv2.resize(img_grid_bgr, (lb_size, lb_size), interpolation=interp)
    top, left, bottom_y, right_x = crop_box
    cropped = up[top:bottom_y, left:right_x]
    if cropped.size == 0:
        cropped = up
    return cv2.resize(cropped, target_wh, interpolation=interp)


def render_views_for_map(
    base_bgr: np.ndarray,
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float = 0.50,
    clip_percentile: float = 99.5,
    fixed_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = base_bgr.shape[:2]
    smooth_aligned = unpad_and_resize_2d(map2d, lb_size, crop_box, (w, h), nearest=False)
    u8_smooth = normalize_map_to_uint8(smooth_aligned, clip_percentile=clip_percentile, fixed_range=fixed_range)
    heat_smooth = cv2.applyColorMap(u8_smooth, colormap_id)
    overlay_smooth = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_smooth, alpha, 0.0)

    u8_raw = normalize_map_to_uint8(map2d, clip_percentile=clip_percentile, fixed_range=fixed_range)
    heat_native = unpad_and_resize_bgr(cv2.applyColorMap(u8_raw, colormap_id), lb_size, crop_box, (w, h), nearest=True)
    return overlay_smooth, heat_smooth, heat_native


def render_grayscale_view(
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    fixed_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> np.ndarray:
    aligned = unpad_and_resize_2d(map2d, lb_size, crop_box, target_wh, nearest=False)
    u8 = normalize_map_to_uint8(aligned, clip_percentile=100.0, fixed_range=fixed_range)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def resize_to_panel(img_bgr: np.ndarray, panel_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel_h = max(64, int(round(h * (panel_w / float(w)))))
    return cv2.resize(img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA if panel_w < w else cv2.INTER_LINEAR)


def draw_detections_on_img(
    img_bgr: np.ndarray, boxes: np.ndarray, class_names: List[str], show_conf: bool = True, thickness: int = 2
) -> np.ndarray:
    canvas = img_bgr.copy()
    if boxes is None or len(boxes) == 0:
        return canvas
    h, w = canvas.shape[:2]
    font_scale = max(0.36, min(0.54, min(h, w) / 900.0))
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
        cv2.putText(
            canvas, label, (x1 + 3, by2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0) if bright > 140 else (255, 255, 255),
            1, cv2.LINE_AA
        )
    return canvas


def make_labeled_tile(
    img_bgr: np.ndarray, title: str, subtitle: str = "", accent_bgr: Tuple[int, int, int] = (70, 150, 255)
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    header_h = 44 if subtitle else 28
    tile = np.full((h + header_h, w, 3), 24, dtype=np.uint8)
    tile[header_h:, :] = img_bgr
    cv2.rectangle(tile, (0, 0), (w - 1, header_h - 1), (30, 32, 38), -1)
    cv2.line(tile, (0, 0), (w - 1, 0), accent_bgr, 3)
    cv2.line(tile, (0, header_h - 1), (w - 1, header_h - 1), (62, 66, 76), 1)
    t_scale = min(0.46, max(0.30, w / 840.0))
    cv2.putText(tile, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (240, 244, 252), 1, cv2.LINE_AA)
    if subtitle:
        s_scale = min(0.37, max(0.26, w / 1020.0))
        cv2.putText(tile, subtitle, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (168, 178, 195), 1, cv2.LINE_AA)
    return tile


def assemble_grid(
    tiles: Sequence[np.ndarray],
    ncols: int,
    banner_title: str = "",
    banner_sub: str = "",
    pad: int = 10,
    bg_color: Tuple[int, int, int] = (16, 17, 20),
    colormap_id: Optional[int] = None,
) -> np.ndarray:
    if not tiles:
        return np.zeros((100, 300, 3), dtype=np.uint8)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    nrows = (len(tiles) + ncols - 1) // ncols
    banner_h = 58 if banner_title else 0
    cbar_w = 56 if colormap_id is not None else 0
    canvas_h = banner_h + pad + nrows * (th + pad)
    canvas_w = pad + ncols * (tw + pad) + cbar_w
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    if banner_title:
        cv2.rectangle(canvas, (0, 0), (canvas_w - 1, banner_h - 1), (25, 27, 33), -1)
        cv2.line(canvas, (0, banner_h - 1), (canvas_w - 1, banner_h - 1), (80, 155, 255), 2)
        b_scale = min(0.64, max(0.44, (canvas_w - 40) / max(1, len(banner_title) * 14.5)))
        cv2.putText(canvas, banner_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, b_scale, (245, 248, 255), 2, cv2.LINE_AA)
        if banner_sub:
            s_scale = min(0.41, max(0.32, (canvas_w - 40) / max(1, len(banner_sub) * 9.5)))
            cv2.putText(canvas, banner_sub, (14, 47), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (175, 185, 202), 1, cv2.LINE_AA)

    for idx, tile in enumerate(tiles):
        r, c = idx // ncols, idx % ncols
        y0 = banner_h + pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    if colormap_id is not None:
        bar_h = max(120, canvas_h - banner_h - 2 * pad - 44)
        bar_x = canvas_w - cbar_w + 12
        bar_y = banner_h + pad + 22
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        canvas[bar_y : bar_y + bar_h, bar_x : bar_x + 18] = cv2.applyColorMap(np.repeat(grad, 18, axis=1), colormap_id)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 18, bar_y + bar_h), (125, 130, 140), 1)
        cv2.putText(canvas, "High", (bar_x - 4, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (232, 236, 244), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Low", (bar_x - 2, bar_y + bar_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA)

    return canvas


# ==================================================================================================
# 4. Composite Figures (Pyramid + EdgeExtractor/EdgeSAB + Skip-P3 SAB + Stats + Comparison)
# ==================================================================================================

def build_pyramid_hero_figure(
    raw_panel: np.ndarray,
    ref_panel: np.ndarray,
    det_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    stage_slots: Dict[str, str],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
    img_name: str,
) -> np.ndarray:
    ph, pw = ref_panel.shape[:2]
    slot_order = [s for s in ("p2_pre", "p2", "p3", "p4", "p5") if s in stage_slots and stage_slots[s] in records]

    row1 = [make_labeled_tile(raw_panel, "Input (Raw Low-Light)", img_name, (130, 140, 155))]
    row2 = [make_labeled_tile(ref_panel, "Display Underlay", "Gamma-adjusted reference", (170, 185, 205))]
    row3 = [make_labeled_tile(det_panel, "Model Predictions / GT", "Reference detections", (80, 220, 120))]

    for slot in slot_order:
        key = stage_slots[slot]
        rec = records[key]
        proj = projections[key]
        c, h_f, w_f = proj["shape"]
        stride = rec["stride"]
        accent = STAGE_ACCENT_BGR.get(rec["accent_key"], (80, 160, 255))

        ov_bgr, _, native_bgr = render_views_for_map(
            ref_panel, proj["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
        )
        pca_aligned = unpad_and_resize_bgr(proj["pca_bgr"], lb_size, crop_box, (pw, ph), nearest=True)

        tag = f"L{rec['layer_idx']:02d} {rec['mod_type']} | {h_f}x{w_f}"
        sub = f"C={c} | L2={proj['l2_mean']:.3f} | Max={proj['max_val']:.2f}"
        row1.append(make_labeled_tile(ov_bgr, f"{tag} [Overlay]", sub, accent))
        row2.append(make_labeled_tile(native_bgr, f"{h_f}x{w_f} Native Grid [L2]", f"Cell {stride}x{stride}px | Sparsity {proj['sparsity']:.1f}%", accent))
        row3.append(make_labeled_tile(pca_aligned, f"{h_f}x{w_f} PCA-RGB [Top-3 PC]", f"Subspace ({c} -> 3 RGB)", accent))

    return assemble_grid(
        row1 + row2 + row3,
        ncols=1 + len(slot_order),
        banner_title=f"YOLOv11 P2-EdgeSAB Backbone Pyramid (160 -> 80 -> 40 -> 20) — {model_title}",
        banner_sub="Row 1: Aligned L2 Overlay | Row 2: Discrete Native-Grid Resolution | Row 3: Top-3 PCA Semantic RGB",
        colormap_id=colormap_id,
    )


def build_edge_and_p2_edgesab_figure(
    raw_panel: np.ndarray,
    ref_panel: np.ndarray,
    edge_info: Dict[str, Any],
    edgesab_info: Optional[Dict[str, Any]],
    rec_pre: Optional[Dict[str, Any]],
    proj_pre: Optional[Dict[str, Any]],
    rec_post: Optional[Dict[str, Any]],
    proj_post: Optional[Dict[str, Any]],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    edge_colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
    edge_out_dir: Path,
) -> np.ndarray:
    """
    Create `01_edge_extractor_and_p2_edgesab.png` and export standalone borderless edge maps into `edge/`.
    """
    edge_out_dir.mkdir(parents=True, exist_ok=True)
    ph, pw = ref_panel.shape[:2]
    acc_e = STAGE_ACCENT_BGR["edge"]
    acc_p2 = STAGE_ACCENT_BGR["p2_edge"]

    # Row 1: EdgeExtractor pipeline (640 -> 160)
    weber_gray = render_grayscale_view(edge_info["weber"], lb_size, crop_box, (pw, ph), fixed_range=(0.0, 1.0))
    _, grad_heat, _ = render_views_for_map(
        ref_panel, edge_info["grad_norm"], lb_size, crop_box, edge_colormap_id, alpha=alpha, clip_percentile=99.5
    )
    edge_ov, edge_heat, _ = render_views_for_map(
        ref_panel, edge_info["edge_priority"], lb_size, crop_box, edge_colormap_id,
        alpha=0.58, clip_percentile=100.0, fixed_range=(0.0, 1.0)
    )

    cv2.imwrite(str(edge_out_dir / "edge_weber_log.png"), weber_gray)
    cv2.imwrite(str(edge_out_dir / "edge_gradients_640.png"), grad_heat)
    cv2.imwrite(str(edge_out_dir / "edge_priority_640.png"), edge_heat)
    cv2.imwrite(str(edge_out_dir / "edge_priority_overlay.png"), edge_ov)

    row1 = [
        make_labeled_tile(raw_panel, "1. Raw Input (640x640)", "Original low-light image", (130, 140, 155)),
        make_labeled_tile(weber_gray, "2. Weber-Log Luminance", f"weber_alpha={edge_info['weber_alpha']:.3f} (dark lift + glare comp.)", acc_e),
        make_labeled_tile(grad_heat, "3. Fused Sobel+Laplacian", "Multi-directional gradient norm [0,1]", acc_e),
        make_labeled_tile(
            edge_heat,
            "4. Edge Priority Map (640)",
            f"sigmoid({edge_info['scale']:.2f}*g + {edge_info['offset']:.2f}) | mean={edge_info['edge_priority'].mean():.3f}",
            acc_e,
        ),
        make_labeled_tile(edge_ov, "5. Edge Priority Overlay", "Aligned structural edges on dark scene", acc_e),
    ]

    if edgesab_info is None or rec_pre is None or rec_post is None or proj_pre is None or proj_post is None:
        return assemble_grid(
            row1,
            ncols=5,
            banner_title=f"Low-Light EdgeExtractor Pipeline — {model_title}",
            banner_sub="Luminance -> Gaussian Blur -> Weber-Log Compression -> Sobel/Laplacian Fusion -> Priority Edge Map",
            colormap_id=edge_colormap_id,
        )

    # Row 2: P2 EdgeSAB (160x160) internal modulation + Pre vs Post + Residual Delta
    _, edge160_heat, edge160_nat = render_views_for_map(
        ref_panel, edgesab_info["edge_scaled"], lb_size, crop_box, edge_colormap_id,
        alpha=0.55, clip_percentile=100.0, fixed_range=(0.0, 1.0)
    )
    _, gate_heat, _ = render_views_for_map(
        ref_panel, edgesab_info["edge_gate_mean"], lb_size, crop_box, edge_colormap_id,
        alpha=0.55, clip_percentile=99.5
    )
    ov_pre, _, _ = render_views_for_map(
        ref_panel, proj_pre["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )
    ov_post, _, _ = render_views_for_map(
        ref_panel, proj_post["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )

    delta_l2 = torch.sqrt(((rec_post["tensor"][0] - rec_pre["tensor"][0]) ** 2).mean(dim=0) + 1e-12).numpy()
    ov_delta, heat_delta, _ = render_views_for_map(
        ref_panel, delta_l2, lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct
    )

    cv2.imwrite(str(edge_out_dir / "edge_scaled_160.png"), edge160_nat)
    cv2.imwrite(str(edge_out_dir / "edge_gate_160.png"), gate_heat)
    cv2.imwrite(str(edge_out_dir / "p2_edgesab_delta_overlay.png"), ov_delta)
    cv2.imwrite(str(edge_out_dir / "p2_edgesab_delta_heatmap.png"), heat_delta)

    row2 = [
        make_labeled_tile(
            edge160_nat,
            "6. Edge Scaled (160x160)",
            f"0.5*Bilinear + 0.5*MaxPool | edge_scale={edgesab_info['edge_scale']:.2f}",
            acc_e,
        ),
        make_labeled_tile(
            gate_heat,
            "7. MultiScaleEdgeGate (160)",
            f"3x3 Local + Dilated Wide | mean={edgesab_info['edge_gate_mean'].mean():.3f}",
            acc_p2,
        ),
        make_labeled_tile(
            ov_pre,
            f"8. Pre-EdgeSAB ({rec_pre['mod_type']} L{rec_pre['layer_idx']:02d})",
            f"160x160 | L2={proj_pre['l2_mean']:.4f} | Max={proj_pre['max_val']:.2f}",
            STAGE_ACCENT_BGR["p2"],
        ),
        make_labeled_tile(
            ov_delta,
            "9. EdgeSAB Residual |L03 - L02|",
            f"gamma_mean={edgesab_info['gamma_mean']:.3f} | Mean Delta={float(delta_l2.mean()):.4f}",
            (60, 220, 255),
        ),
        make_labeled_tile(
            ov_post,
            f"10. Post-EdgeSAB (L{rec_post['layer_idx']:02d} Out)",
            f"160x160 | L2={proj_post['l2_mean']:.4f} | Max={proj_post['max_val']:.2f}",
            acc_p2,
        ),
    ]

    return assemble_grid(
        row1 + row2,
        ncols=5,
        banner_title=f"EdgeExtractor & P2 EdgeSAB (160x160) Deep Diagnostics — {model_title}",
        banner_sub="Row 1: EdgeExtractor (Weber-Log -> Gradients -> Priority Map) | Row 2: EdgeSAB (160x160 Gate, Pre/Post & Residual Delta)",
        colormap_id=edge_colormap_id,
    )


def build_all_layers_progression_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
) -> np.ndarray:
    overlay_tiles: List[np.ndarray] = []
    native_tiles: List[np.ndarray] = []

    for key, rec in records.items():
        proj = projections[key]
        c, h_f, w_f = proj["shape"]
        accent = STAGE_ACCENT_BGR.get(rec["accent_key"], (80, 160, 255))
        ov_bgr, _, nat_bgr = render_views_for_map(
            ref_panel, proj["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
        )
        title = f"L{rec['layer_idx']:02d} {rec['mod_type']} ({h_f}x{w_f})"
        sub = f"{rec['stage_group']} | C={c} | L2={proj['l2_mean']:.3f}"
        overlay_tiles.append(make_labeled_tile(ov_bgr, title, sub, accent))
        native_tiles.append(make_labeled_tile(nat_bgr, f"L{rec['layer_idx']:02d} Native {h_f}x{w_f}", sub, accent))

    ncols = min(6, max(1, len(overlay_tiles)))
    rem = len(overlay_tiles) % ncols
    if rem != 0:
        blank_tile = np.full_like(overlay_tiles[0], (18, 20, 24))
        overlay_tiles.extend([blank_tile] * (ncols - rem))
        native_tiles.extend([blank_tile] * (ncols - rem))
    has_skip_sab = any(r.get("accent_key") == "p3_sab" for r in records.values())
    prog_title = (
        f"Full Backbone + EdgeSAB + Skip-P3 SAB Progression — {model_title}"
        if has_skip_sab
        else f"Full Backbone + P2 EdgeSAB Progression (160 -> 80 -> 40 -> 20) — {model_title}"
    )
    return assemble_grid(
        overlay_tiles + native_tiles,
        ncols=ncols,
        banner_title=prog_title,
        banner_sub="Top block: Spatial L2 Overlay | Bottom block: Discrete Native-Resolution Heatmaps (160 -> 80 -> 40 -> 20)",
        colormap_id=colormap_id,
    )


def build_skip_p3_and_stage4_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    stage_slots: Dict[str, str],
    skip_p3_sab_mod: Optional[nn.Module],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
) -> Optional[np.ndarray]:
    rows: List[np.ndarray] = []

    # Row A: Stage 4 (20x20) refinement: Conv -> C3k2 -> SPPF -> C2PSA + |C2PSA - SPPF|
    s4_keys = [k for k, r in records.items() if r["stride"] == 32]
    s4_tiles: List[np.ndarray] = []
    for key in s4_keys[:4]:
        rec, proj = records[key], projections[key]
        c, h_f, w_f = proj["shape"]
        ov, _, _ = render_views_for_map(ref_panel, proj["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        s4_tiles.append(make_labeled_tile(ov, f"L{rec['layer_idx']:02d} {rec['mod_type']} ({h_f}x{w_f})", f"C={c} | L2={proj['l2_mean']:.3f} | Max={proj['max_val']:.2f}", STAGE_ACCENT_BGR["p5"]))

    sppf_k, c2psa_k = stage_slots.get("s4_sppf", ""), stage_slots.get("p5", "")
    if sppf_k in records and c2psa_k in records:
        t_sppf, t_psa = records[sppf_k]["tensor"][0], records[c2psa_k]["tensor"][0]
        if t_sppf.shape == t_psa.shape:
            d20 = torch.sqrt(((t_psa - t_sppf) ** 2).mean(dim=0) + 1e-12).numpy()
            ov_d20, _, _ = render_views_for_map(ref_panel, d20, lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
            s4_tiles.append(make_labeled_tile(ov_d20, "C2PSA Delta |C2PSA - SPPF|", f"20x20 | Mean Delta L2={float(d20.mean()):.4f}", (60, 220, 255)))

    # Row B: Skip-P3 SAB (80x80) refinement: P3 Pre-SAB -> P3 Post-SAB -> |Delta| + Native Grids
    p3_k, p3_sab_k = stage_slots.get("p3", ""), stage_slots.get("p3_sab", "")
    p3_tiles: List[np.ndarray] = []
    if p3_k in records and p3_sab_k in records:
        rec_p3, proj_p3 = records[p3_k], projections[p3_k]
        rec_sab, proj_sab = records[p3_sab_k], projections[p3_sab_k]
        ov_p3, _, nat_p3 = render_views_for_map(ref_panel, proj_p3["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        ov_sab, _, nat_sab = render_views_for_map(ref_panel, proj_sab["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        d80 = torch.sqrt(((rec_sab["tensor"][0] - rec_p3["tensor"][0]) ** 2).mean(dim=0) + 1e-12).numpy()
        ov_d80, _, _ = render_views_for_map(ref_panel, d80, lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct)
        g_sab = float(skip_p3_sab_mod.gamma.detach().float().mean().item()) if (skip_p3_sab_mod is not None and hasattr(skip_p3_sab_mod, "gamma")) else 0.0

        p3_tiles = [
            make_labeled_tile(ov_p3, f"Pre-SAB L{rec_p3['layer_idx']:02d} (80x80)", f"L2={proj_p3['l2_mean']:.4f} | Max={proj_p3['max_val']:.2f}", STAGE_ACCENT_BGR["p3"]),
            make_labeled_tile(ov_sab, f"Post-SAB L{rec_sab['layer_idx']:02d} (80x80)", f"L2={proj_sab['l2_mean']:.4f} | gamma={g_sab:.3f}", STAGE_ACCENT_BGR["p3_sab"]),
            make_labeled_tile(ov_d80, "Skip-P3 SAB |Post - Pre|", f"80x80 | Mean Delta L2={float(d80.mean()):.4f}", (60, 220, 255)),
            make_labeled_tile(nat_p3, "Pre-SAB Native 80x80", f"Sparsity={proj_p3['sparsity']:.1f}%", STAGE_ACCENT_BGR["p3"]),
            make_labeled_tile(nat_sab, "Post-SAB Native 80x80", f"Sparsity={proj_sab['sparsity']:.1f}%", STAGE_ACCENT_BGR["p3_sab"]),
        ]

    all_tiles = s4_tiles + p3_tiles
    if not all_tiles:
        return None
    ncols = max(len(s4_tiles), len(p3_tiles), 1)
    if p3_tiles:
        b_title = f"Attention Refinement: Stage 4 C2PSA (20x20) & Neck Skip-P3 SAB (80x80) — {model_title}"
        b_sub = "Row 1: Stage 4 (Conv -> C3k2 -> SPPF -> C2PSA + Delta) | Row 2: Lateral Skip-P3 SAB (Pre, Post, Delta & Native 80x80)"
    else:
        b_title = f"Stage 4 (20x20) Refinement: Conv -> C3k2 -> SPPF -> C2PSA & Delta — {model_title}"
        b_sub = "20x20 Spatial L2 Overlays across Stage 4 + C2PSA Attention Residual |C2PSA - SPPF|"
    return assemble_grid(
        all_tiles,
        ncols=ncols,
        banner_title=b_title,
        banner_sub=b_sub,
        colormap_id=colormap_id,
    )


def build_stats_chart(
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    edge_info: Optional[Dict[str, Any]],
    edgesab_info: Optional[Dict[str, Any]],
    skip_p3_sab_mod: Optional[nn.Module],
    model_title: str,
) -> np.ndarray:
    rows_data: List[Tuple[str, str, float, float, float, Tuple[int, int, int]]] = []
    if edge_info is not None:
        ep = edge_info["edge_priority"]
        rows_data.append((
            "EdgeExtractor (640x640)",
            f"(1, {ep.shape[0]}, {ep.shape[1]})",
            float(np.sqrt((ep ** 2).mean() + 1e-12)),
            float((np.abs(ep) < 1e-2).mean() * 100.0),
            float(ep.max()),
            STAGE_ACCENT_BGR["edge"],
        ))
    for key, rec in records.items():
        proj = projections[key]
        c, h_f, w_f = proj["shape"]
        rows_data.append((
            f"L{rec['layer_idx']:02d} {rec['mod_type']} ({h_f}x{w_f})",
            f"({c}, {h_f}, {w_f})",
            proj["l2_mean"],
            proj["sparsity"],
            proj["max_val"],
            STAGE_ACCENT_BGR.get(rec["accent_key"], (90, 170, 255)),
        ))

    n = len(rows_data)
    row_h = 40
    top_h = 96
    bot_h = 46
    width = 1220
    height = top_h + n * row_h + bot_h
    canvas = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width - 1, 62), (28, 30, 36), -1)
    cv2.line(canvas, (0, 62), (width - 1, 62), (80, 155, 255), 2)
    title_txt = f"YOLOv11 Backbone + EdgeSAB Quantitative Diagnostics — {model_title}"
    t_scale = min(0.58, max(0.42, (width - 40) / max(1, len(title_txt) * 13.0)))
    cv2.putText(canvas, title_txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (245, 248, 255), 2, cv2.LINE_AA)

    param_parts = []
    if edge_info is not None:
        param_parts.append(f"EdgeExtractor(weber_alpha={edge_info['weber_alpha']:.2f}, scale={edge_info['scale']:.2f}, offset={edge_info['offset']:.2f})")
    if edgesab_info is not None:
        param_parts.append(f"P2-EdgeSAB(gamma={edgesab_info['gamma_mean']:.3f}, edge_scale={edgesab_info['edge_scale']:.2f})")
    if skip_p3_sab_mod is not None and hasattr(skip_p3_sab_mod, "gamma"):
        g_sp3 = float(skip_p3_sab_mod.gamma.detach().float().mean().item())
        if hasattr(skip_p3_sab_mod, "edge_scale"):
            es_sp3 = float(skip_p3_sab_mod.edge_scale.detach().float().item())
            param_parts.append(f"SkipP3-EdgeSAB(gamma={g_sp3:.3f}, edge_scale={es_sp3:.2f})")
        else:
            param_parts.append(f"SkipP3-SAB(gamma={g_sp3:.3f})")
    if param_parts:
        cv2.putText(canvas, " | ".join(param_parts), (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 210, 240), 1, cv2.LINE_AA)

    # Column headers cleanly aligned with bars
    cv2.putText(canvas, "Layer / Module", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Shape (C,H,W)", (295, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, "L2 Energy (RMS)", (450, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Sparsity (|x|<0.01)", (850, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Max", (1115, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)

    max_l2 = max((r[2] for r in rows_data), default=1.0)
    max_l2 = max(max_l2, 1e-6)

    for r_i, (name_txt, shape_txt, l2_val, sp_val, max_val, accent) in enumerate(rows_data):
        y_top = top_h + r_i * row_h
        y_mid = y_top + 24
        if r_i % 2 == 0:
            cv2.rectangle(canvas, (10, y_top), (width - 10, y_top + row_h - 2), (24, 27, 33), -1)
        cv2.rectangle(canvas, (10, y_top), (15, y_top + row_h - 2), accent, -1)

        cv2.putText(canvas, name_txt, (24, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 248), 1, cv2.LINE_AA)
        cv2.putText(canvas, shape_txt, (295, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (180, 190, 208), 1, cv2.LINE_AA)

        bar_x0, bar_max_w = 450, 300
        l2_w = int(round((l2_val / max_l2) * bar_max_w))
        cv2.rectangle(canvas, (bar_x0, y_top + 9), (bar_x0 + bar_max_w, y_top + 27), (40, 44, 52), -1)
        if l2_w > 0:
            cv2.rectangle(canvas, (bar_x0, y_top + 9), (bar_x0 + l2_w, y_top + 27), accent, -1)
        cv2.putText(canvas, f"{l2_val:.4f}", (bar_x0 + bar_max_w + 10, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (230, 235, 245), 1, cv2.LINE_AA)

        sp_x0, sp_max_w = 850, 165
        sp_w = int(round((sp_val / 100.0) * sp_max_w))
        cv2.rectangle(canvas, (sp_x0, y_top + 9), (sp_x0 + sp_max_w, y_top + 27), (40, 44, 52), -1)
        if sp_w > 0:
            cv2.rectangle(canvas, (sp_x0, y_top + 9), (sp_x0 + sp_w, y_top + 27), (110, 195, 130), -1)
        cv2.putText(canvas, f"{sp_val:.1f}%", (sp_x0 + sp_max_w + 10, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (205, 215, 228), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{max_val:.2f}", (1115, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (235, 215, 150), 1, cv2.LINE_AA)

    return canvas


def save_per_layer_assets(
    layer_dir: Path,
    ref_panel: np.ndarray,
    rec: Dict[str, Any],
    proj: Dict[str, Any],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    top_k_channels: int,
) -> None:
    layer_dir.mkdir(parents=True, exist_ok=True)
    ph, pw = ref_panel.shape[:2]
    c, h_f, w_f = proj["shape"]
    accent = STAGE_ACCENT_BGR.get(rec["accent_key"], (80, 160, 255))

    ov_l2, heat_l2, native_l2 = render_views_for_map(
        ref_panel, proj["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )
    pca_aligned = unpad_and_resize_bgr(proj["pca_bgr"], lb_size, crop_box, (pw, ph), nearest=True)

    cv2.imwrite(str(layer_dir / "l2_overlay.png"), ov_l2)
    cv2.imwrite(str(layer_dir / "l2_heatmap.png"), heat_l2)
    cv2.imwrite(str(layer_dir / "native_grid.png"), native_l2)
    cv2.imwrite(str(layer_dir / "pca_rgb.png"), pca_aligned)

    if top_k_channels <= 0:
        return

    _, _, native_max = render_views_for_map(
        ref_panel, proj["max"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )
    summary_tiles = [
        make_labeled_tile(ov_l2, "L2 Energy Overlay", f"Shape ({c}, {h_f}, {w_f})", accent),
        make_labeled_tile(native_l2, f"L2 Native Grid ({h_f}x{w_f})", f"Mean L2={proj['l2_mean']:.4f}", accent),
        make_labeled_tile(native_max, f"Max Response ({h_f}x{w_f})", f"Peak={proj['max_val']:.3f}", accent),
        make_labeled_tile(pca_aligned, f"PCA-RGB ({h_f}x{w_f})", "Top-3 Principal Components", accent),
    ]

    feat = rec["tensor"][0]
    ch_energy = torch.sqrt((feat ** 2).mean(dim=(1, 2)) + 1e-12)
    k = min(int(top_k_channels), c)
    top_vals, top_idxs = torch.topk(ch_energy, k=k, largest=True)

    ch_tiles: List[np.ndarray] = []
    for rank, (ch_i, ch_e) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), start=1):
        ch_map = feat[ch_i].numpy()
        _, _, ch_nat = render_views_for_map(ref_panel, ch_map, lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        ch_tiles.append(
            make_labeled_tile(
                ch_nat, f"Top#{rank} Ch {ch_i}/{c} ({h_f}x{w_f})",
                f"RMS={ch_e:.4f} | min={ch_map.min():.2f} max={ch_map.max():.2f}", (95, 110, 135)
            )
        )

    sheet = assemble_grid(
        summary_tiles + ch_tiles,
        ncols=4,
        banner_title=f"{rec['key']} — Resolution {h_f}x{w_f} (Stride {rec['stride']}, Channels {c})",
        banner_sub=f"Row 1: Summary Maps | Remaining rows: Top-{k} Highest-Energy Individual Channels",
        colormap_id=colormap_id,
    )
    cv2.imwrite(str(layer_dir / "channel_sheet.png"), sheet)


def build_semantic_comparison_figure(
    ref_panel: np.ndarray,
    records_a: "OrderedDict[str, Dict[str, Any]]",
    proj_a: Dict[str, Dict[str, Any]],
    slots_a: Dict[str, str],
    label_a: str,
    records_b: "OrderedDict[str, Dict[str, Any]]",
    proj_b: Dict[str, Dict[str, Any]],
    slots_b: Dict[str, str],
    label_b: str,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
) -> np.ndarray:
    """Compare two architectures across semantic stages P2(160) -> P3(80) -> P4(40) -> P5(20) even if layer indices shift."""
    common_slots = [s for s in ("p2", "p3", "p4", "p5") if s in slots_a and s in slots_b]
    row_a, row_b, row_diff = [], [], []

    for slot in common_slots:
        ka, kb = slots_a[slot], slots_b[slot]
        ra, rb = records_a[ka], records_b[kb]
        pa, pb = proj_a[ka], proj_b[kb]
        _, h_f, w_f = pa["shape"]
        accent = STAGE_ACCENT_BGR.get(ra["accent_key"], (80, 160, 255))

        vmax = max(float(np.percentile(pa["l2"], clip_pct)), float(np.percentile(pb["l2"], clip_pct)), 1e-6)
        vmin = min(float(pa["l2"].min()), float(pb["l2"].min()))

        ov_a, _, _ = render_views_for_map(ref_panel, pa["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct, fixed_range=(vmin, vmax))
        ov_b, _, _ = render_views_for_map(ref_panel, pb["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct, fixed_range=(vmin, vmax))
        diff_2d = np.abs(pb["l2"] - pa["l2"])
        _, _, diff_nat = render_views_for_map(ref_panel, diff_2d, lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)

        row_a.append(make_labeled_tile(ov_a, f"[A] {slot.upper()} L{ra['layer_idx']:02d} ({h_f}x{w_f})", f"{label_a} | L2={pa['l2_mean']:.4f}", accent))
        row_b.append(make_labeled_tile(ov_b, f"[B] {slot.upper()} L{rb['layer_idx']:02d} ({h_f}x{w_f})", f"{label_b} | L2={pb['l2_mean']:.4f}", (90, 230, 130)))
        row_diff.append(make_labeled_tile(diff_nat, f"|B - A| {slot.upper()} ({h_f}x{w_f})", f"Mean |Delta L2|={float(diff_2d.mean()):.4f}", (80, 210, 255)))

    return assemble_grid(
        row_a + row_b + row_diff,
        ncols=max(1, len(common_slots)),
        banner_title=f"Semantic Backbone Comparison (160 -> 80 -> 40 -> 20): [A] {label_a} vs [B] {label_b}",
        banner_sub="Row 1: Model A (shared scale) | Row 2: Model B (shared scale) | Row 3: Absolute L2 Difference |B - A|",
        colormap_id=colormap_id,
    )


# ==================================================================================================
# 5. Main Pipeline
# ==================================================================================================

def run_inference_and_record(
    model: nn.Module,
    recorder: YOLOv11BackboneEdgeRecorder,
    inp_tensor: torch.Tensor,
    orig_hw: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
    conf_thres: float,
    iou_thres: float,
) -> Tuple[
    "OrderedDict[str, Dict[str, Any]]",
    Dict[str, Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    np.ndarray,
]:
    recorder.clear()
    with torch.no_grad():
        out = model(inp_tensor)
        pred_raw = out[0] if isinstance(out, tuple) else out

    edge_info: Optional[Dict[str, Any]] = None
    edgesab_info: Optional[Dict[str, Any]] = None
    if recorder.edge_extractor_mod is not None:
        edge_info = inspect_edge_extractor_internals(recorder.edge_extractor_mod, inp_tensor)
        if recorder.p2_edgesab_mod is not None and recorder.p2_edgesab_input is not None:
            edgesab_info = inspect_edgesab_internals(
                recorder.p2_edgesab_mod, recorder.p2_edgesab_input, edge_info["edge_tensor"]
            )

    dets_np = np.zeros((0, 6), dtype=np.float32)
    if isinstance(pred_raw, torch.Tensor):
        dets = non_max_suppression(
            pred_raw.float(),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=300,
            end2end=bool(getattr(model, "end2end", False)),
        )[0]
        if len(dets):
            dets = dets.clone()
            dets[:, :4] = scale_boxes(inp_tensor.shape[2:], dets[:, :4], orig_hw, ratio_pad=ratio_pad)
            dets_np = dets.cpu().numpy()

    records_copy: "OrderedDict[str, Dict[str, Any]]" = OrderedDict(
        (k, dict(v)) for k, v in recorder.records.items()
    )
    projections = {k: compute_spatial_projections(v["tensor"]) for k, v in records_copy.items()}
    return records_copy, projections, edge_info, edgesab_info, dets_np


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    colormap_id = COLORMAP_TABLE[args.colormap]
    edge_colormap_id = COLORMAP_TABLE[args.edge_colormap]

    model_a, model_name_a, nc_a, meta_a = load_yolov11_model(
        weights_path=args.weights,
        model_override=args.model,
        cfg_override=args.cfg,
        nc_override=args.nc,
        modular_override=args.modular,
        arch_override=args.arch,
        device=device,
    )
    class_names = resolve_class_names(args.classes, nc_a, meta_a.get("ckpt_names"))
    recorder_a = YOLOv11BackboneEdgeRecorder(model_a, include_stem=args.include_stem)

    model_b, recorder_b, meta_b = None, None, {}
    if args.compare_weights.strip():
        model_b, _, _, meta_b = load_yolov11_model(
            weights_path=args.compare_weights,
            model_override=args.model,
            cfg_override=args.cfg,
            nc_override=args.nc,
            modular_override=args.modular,
            arch_override="auto",
            device=device,
        )
        recorder_b = YOLOv11BackboneEdgeRecorder(model_b, include_stem=args.include_stem)

    image_paths = collect_input_images(args.source, args.num, args.shuffle)
    run_tag = (
        f"{model_name_a}_{meta_a['weights_label']}"
        if not args.compare_weights.strip()
        else f"compare_{meta_a['weights_label']}_vs_{meta_b['weights_label']}"
    )
    out_root = Path(args.output) / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"===> Visualizing YOLOv11 Backbone + EdgeExtractor + P2-EdgeSAB on {len(image_paths)} image(s) -> '{out_root}'")

    try:
        for img_idx, img_path in enumerate(image_paths, start=1):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[Warn] Skipping unreadable image: {img_path}")
                continue

            stem = Path(img_path).stem
            img_out_dir = out_root / stem
            img_out_dir.mkdir(parents=True, exist_ok=True)

            orig_hw = img_bgr.shape[:2]
            inp_tensor, crop_box, ratio_pad = preprocess_image(img_bgr, img_size=args.imgsz)
            inp_tensor = inp_tensor.to(device)

            records_a, proj_a, edge_info_a, edgesab_info_a, dets_a = run_inference_and_record(
                model_a, recorder_a, inp_tensor, orig_hw, ratio_pad, args.conf_thres, args.iou_thres
            )

            gt_boxes = load_optional_gt(img_path, orig_hw, class_names)
            ref_full = apply_display_gamma(img_bgr, args.vis_gamma)
            det_full = (
                draw_detections_on_img(ref_full, dets_a, class_names, show_conf=True)
                if len(dets_a) > 0
                else draw_detections_on_img(ref_full, gt_boxes, class_names, show_conf=False)
            )
            raw_panel = resize_to_panel(img_bgr, args.panel_width)
            ref_panel = resize_to_panel(ref_full, args.panel_width)
            det_panel = resize_to_panel(det_full, args.panel_width)
            model_title = f"{model_name_a} ({meta_a['weights_label']}, epoch={meta_a['epoch']})"

            # 1. Hero Pyramid (Pre-EdgeSAB 160 -> Post-EdgeSAB 160 -> 80 -> 40 -> 20)
            fig0 = build_pyramid_hero_figure(
                raw_panel, ref_panel, det_panel, records_a, proj_a, recorder_a.stage_slots,
                args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile,
                model_title, Path(img_path).name,
            )
            cv2.imwrite(str(img_out_dir / "00_backbone_pyramid_160_to_20.png"), fig0)

            # 2. Dedicated EdgeExtractor + P2 EdgeSAB (160x160) Deep Diagnostics
            if edge_info_a is not None:
                pre_k = recorder_a.stage_slots.get("p2_pre", "")
                post_k = recorder_a.stage_slots.get("p2", "")
                fig1 = build_edge_and_p2_edgesab_figure(
                    raw_panel, ref_panel, edge_info_a, edgesab_info_a,
                    records_a.get(pre_k), proj_a.get(pre_k),
                    records_a.get(post_k), proj_a.get(post_k),
                    args.imgsz, crop_box, colormap_id, edge_colormap_id,
                    args.alpha, args.clip_percentile, model_title, img_out_dir / "edge",
                )
                cv2.imwrite(str(img_out_dir / "01_edge_extractor_and_p2_edgesab.png"), fig1)

            # 3. Full Layer Progression (L01..L11 + L16 Skip-P3 SAB)
            fig2 = build_all_layers_progression_figure(
                ref_panel, records_a, proj_a, args.imgsz, crop_box,
                colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            cv2.imwrite(str(img_out_dir / "02_all_backbone_layers_160_to_20.png"), fig2)

            # 4. Stage 4 (20x20) Refinement (+ Neck Skip-P3 SAB if present)
            fig3 = build_skip_p3_and_stage4_figure(
                ref_panel, records_a, proj_a, recorder_a.stage_slots, recorder_a.skip_p3_sab_mod,
                args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            if fig3 is not None:
                fig3_name = (
                    "03_skip_p3_sab_and_stage4_20x20.png"
                    if recorder_a.skip_p3_sab_mod is not None
                    else "03_stage4_c2psa_20x20.png"
                )
                cv2.imwrite(str(img_out_dir / fig3_name), fig3)

            # 5. Quantitative Activation & Learned Edge/SAB Parameter Chart
            fig4 = build_stats_chart(
                records_a, proj_a, edge_info_a, edgesab_info_a, recorder_a.skip_p3_sab_mod, model_title
            )
            cv2.imwrite(str(img_out_dir / "04_backbone_activation_stats.png"), fig4)

            # 6. Per-Layer Standalone Assets & Top-K Channel Sheets
            layers_root = img_out_dir / "layers"
            for key, rec in records_a.items():
                _, h_f, w_f = proj_a[key]["shape"]
                save_per_layer_assets(
                    layers_root / f"{key}_{h_f}x{w_f}", ref_panel, rec, proj_a[key],
                    args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile, args.channels,
                )

            # 7. Optional Side-by-Side Comparison
            if model_b is not None and recorder_b is not None:
                records_b, proj_b, _, _, _ = run_inference_and_record(
                    model_b, recorder_b, inp_tensor, orig_hw, ratio_pad, args.conf_thres, args.iou_thres
                )
                fig5 = build_semantic_comparison_figure(
                    ref_panel,
                    records_a, proj_a, recorder_a.stage_slots, meta_a["weights_label"],
                    records_b, proj_b, recorder_b.stage_slots, meta_b["weights_label"],
                    args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile,
                )
                cv2.imwrite(str(img_out_dir / "05_backbone_comparison_160_to_20.png"), fig5)

            print(f"  [{img_idx}/{len(image_paths)}] Saved {stem} -> {img_out_dir}")
    finally:
        recorder_a.close()
        if recorder_b is not None:
            recorder_b.close()

    print(f"===> Done! Visualizations saved to: {out_root}")


if __name__ == "__main__":
    main()
