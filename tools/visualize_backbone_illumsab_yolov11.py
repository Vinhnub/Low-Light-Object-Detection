"""
tools/visualize_backbone_illumsab_yolov11.py

Visualize YOLOv11 Backbone multi-scale features (160 -> 80 -> 40 -> 20) together with
full Backbone P2 (160x160) Multi-Prior CWA Illumination-Guided SAB (IllumSAB - PA1)
internal diagnostics.

Tailored for the 25-layer `develop-yolov11-p2-illumsab` architecture (Backbone P2 160x160
`IllumSAB` with Multinex 4-channel analytic luminance stack S_L, Pre-Attention centered CWA
Q/K re-weighting, Post-Attention Retinex-Delta FiLM Gate, gamma_init=0.1):
  - Optional Stem  (P1/2,  320x320): L00_Stem_Conv3x3_P1_2 (via `--include_stem`)
  - Stage 1        (P2/4,  160x160): L01_Conv3x3_P2_4 -> L02_C3k2_P2_4 (Pre-IllumSAB) -> L03_IllumSAB_P2_4 (Post-IllumSAB)
  - Stage 2        (P3/8,   80x80 ): L04_Conv3x3_P3_8 -> L05_C3k2_P3_8 (Skip-P3)
  - Stage 3        (P4/16,  40x40 ): L06_Conv3x3_P4_16 -> L07_C3k2_P4_16 (Skip-P4)
  - Stage 4        (P5/32,  20x20 ): L08_Conv3x3_P5_32 -> L09_C3k2_P5_32 -> L10_SPPF_P5_32 -> L11_C2PSA_P5_32 (Skip-P5)

Also auto-detects other YOLOv11 ablation checkpoints (`base`, `p2_sab`, `p2_edgesab`,
`skip_p3_sab`, `skip_p3_edgesab`, `p3_inline_edgesab`, `p3_prehead_edgesab`, `full_edgesab`,
modular `YOLOv11`) so `--compare_weights` seamlessly compares PA1 against any baseline across
semantic scales (160 -> 80 -> 40 -> 20).

Outputs saved under `<output>/<run_name>/<image_stem>/`:
  - `00_backbone_pyramid_160_to_20.png`                   - Hero pyramid (Pre-IllumSAB 160 -> Post-IllumSAB 160 -> 80 -> 40 -> 20)
  - `01_multinex_illumination_priors_and_cwa_160x160.png` - 4-ch S_L stack (Y_Rec.709, Y_vmax, Y_lightness, Y_L2),
                                                            Y_vmax - Y_lightness gap, Pre-Attn CWA A_qk & W_qk,
                                                            Post-Attn Retinex FiLM scale & ||shift||
  - `02_p2_illumsab_internals_160x160.png`                - Full IllumSAB pipeline: Pre-L02 -> Raw Q -> Illum Q' -> V ->
                                                            4-Head Channel Attn -> Pre-FiLM -> Post-FiLM -> |gamma*out| ->
                                                            Signed Shift -> Post-L03
  - `03_all_backbone_layers_160_to_20.png`                - Chronological progression across L01..L11 (Overlays + Native Grids)
  - `04_attention_stages_p2_and_p5.png`                   - Dual Backbone Attention: P2 IllumSAB (160x160) & P5 C2PSA (20x20)
  - `05_backbone_activation_stats.png`                    - Quantitative L2/Sparsity table + learned gamma chart +
                                                            learned 0.9*tanh(illum_scale) chart + Dark/Mid/Bright regime table
  - `06_backbone_comparison_160_to_20.png`                - Side-by-side semantic comparison & signed/abs delta (`--compare_weights`)
  - `illum/`                                              - Standalone Multinex S_L, CWA, W_qk, FiLM scale/shift maps
  - `illumsab/`                                           - Standalone IllumSAB Q/Q'/V/Attn/Residual/Gamma/Scale charts
  - `layers/<layer_name>/`                                - Standalone maps + Top-K individual channel sheets

Usage Examples:
  # 1. Visualize trained YOLOv11s-P2-IllumSAB (PA1) checkpoint on an ExDark image
  python tools/visualize_backbone_illumsab_yolov11.py ^
      --weights model_results/yolov11s-p2-illumsab/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65

  # 2. Compare Pure P2-SAB vs P2-IllumSAB (PA1) across semantic stages (160 -> 80 -> 40 -> 20)
  python tools/visualize_backbone_illumsab_yolov11.py ^
      --weights model_results/yolov11s-p2-illumsab/best_mAP50.pth ^
      --compare_weights model_results/yolov11s-p2-sab-advanced-training/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65

  # 3. Compare Base YOLOv11s vs P2-IllumSAB (PA1) on 5 random test images
  python tools/visualize_backbone_illumsab_yolov11.py ^
      --weights model_results/yolov11s-p2-illumsab/best_mAP50.pth ^
      --compare_weights model_results/yolov11s/best_mAP50.pth ^
      --source dataset/ExDark/images/test --num 5 --shuffle --vis_gamma 0.65
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
from net.Yolov11.common import (  # noqa: E402
    EdgeExtractor,
    EdgeSAB,
    IllumSAB,
    MultiPriorIllumExtractor,
    SAB,
)
from net.Yolov11.utils import (  # noqa: E402
    _register_ultralytics_unpickle_aliases,
    de_parallel,
    scale_boxes,
)
from net.Yolov11.yolo import DetectionModel, YOLOV11_SCALES  # noqa: E402


# ==================================================================================================
# 1. Visual Styling & CLI Arguments
# ==================================================================================================

STAGE_ACCENT_BGR: Dict[str, Tuple[int, int, int]] = {
    "illum": (40, 210, 255),      # Warm Gold for Multinex S_L illumination priors
    "cwa": (90, 245, 220),        # Mint-Cyan for CWA & Q/K modulation
    "film": (120, 170, 255),      # Coral-Orange for Retinex FiLM scale/shift
    "sab": (60, 220, 255),        # Cyan-Gold for attention residuals
    "edge": (60, 220, 255),       # Cyan-Gold fallback for EdgeExtractor
    "p1": (200, 200, 90),         # P1/2  (320)
    "p2": (255, 195, 55),         # P2/4  (160) Conv / C3k2
    "p2_sab": (255, 110, 225),    # P2/4  (160) IllumSAB / SAB / EdgeSAB
    "p3": (95, 225, 110),         # P3/8  (80)
    "p3_sab": (130, 255, 190),    # P3/8  (80) Legacy Skip-P3 SAB
    "p4": (55, 175, 255),         # P4/16 (40)
    "p5": (120, 95, 255),         # P5/32 (20)
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

SUPPORTED_ARCH_CHOICES = [
    "auto",
    "p2_illumsab",
    "p2_illumsab_v1",
    "p2_sab",
    "p2_edgesab",
    "p2_edgesab_skip_p3",
    "base",
    "skip_p3_sab",
    "skip_p3_edgesab",
    "p3_inline_sab",
    "p3_inline_edgesab",
    "p3_prehead_edgesab",
    "p3_skip_inline_edgesab",
    "full_edgesab",
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
        description=(
            "Visualize YOLOv11 Backbone (160 -> 80 -> 40 -> 20) and Backbone P2 (160x160) "
            "Multi-Prior CWA Illumination-Guided SAB (IllumSAB - PA1) internals."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="auto",
        help="Path to YOLOv11 checkpoint (.pth/.pt), or 'auto' to load official COCO pretrained weights.",
    )
    parser.add_argument(
        "--compare_weights",
        type=str,
        default="",
        help="Optional second checkpoint (.pth/.pt) for semantic side-by-side 160->20 comparison.",
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
        choices=SUPPORTED_ARCH_CHOICES,
        help="Architecture variant ('auto' matches checkpoint exactly; defaults to 25L p2_illumsab for 'auto'/'none').",
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
        default="visualization/backbone_illumsab_yolov11",
        help="Root output directory (default: visualization/backbone_illumsab_yolov11).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input letterbox resolution (default: 640 -> backbone scales 160, 80, 40, 20).",
    )
    parser.add_argument(
        "--include_stem",
        action="store_true",
        help="Also include L00 Stem (P1/2, 320x320) in full layer progression and stats.",
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
        help="OpenCV colormap for spatial activation heatmaps (default: turbo).",
    )
    parser.add_argument(
        "--attn_colormap",
        type=str,
        default="inferno",
        choices=list(COLORMAP_TABLE.keys()),
        help="OpenCV colormap for channel attention matrices & CWA/Illum maps (default: inferno).",
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
# 2. Checkpoint Auto-Detection, Model Loading & Backbone + IllumSAB Recorder
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


def _extract_raw_sd(ckpt_meta: Any) -> Dict[str, Any]:
    if not isinstance(ckpt_meta, dict):
        return {}
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
        return {}
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_sd.items()
    }


def _infer_variant_from_sd(sd: Dict[str, Any]) -> Optional[str]:
    stem_w = sd.get("model.0.conv.weight")
    if stem_w is None:
        stem_w = sd.get("backbone.stem.conv.weight")
    if not isinstance(stem_w, torch.Tensor) or stem_w.ndim != 4:
        return None
    c0 = int(stem_w.shape[0])
    if c0 == 16:
        return "yolov11n"
    if c0 == 32:
        return "yolov11s"
    if c0 == 96:
        return "yolov11x"
    if c0 == 64:
        has_deep_c3k2 = any(
            k.startswith(("model.2.m.1.", "backbone.stage1.1.m.1."))
            for k in sd.keys()
        )
        return "yolov11l" if has_deep_c3k2 else "yolov11m"
    return None


def _infer_nc_from_sd(sd: Dict[str, Any]) -> Optional[int]:
    for k, v in sd.items():
        if k.endswith(".cv3.0.2.bias") and isinstance(v, torch.Tensor) and v.ndim == 1:
            return int(v.shape[0])
    return None


def detect_is_modular_ckpt(sd: Dict[str, Any]) -> bool:
    return any(k.startswith("backbone.") for k in list(sd.keys())[:30])


def detect_ckpt_arch(sd: Dict[str, Any]) -> Optional[str]:
    keys = list(sd.keys())
    if not keys:
        return None
    if any(k.startswith("backbone.") for k in keys[:30]):
        return "modular"

    def _has_detect(idx: int) -> bool:
        p = f"model.{idx}."
        return any(k.startswith((f"{p}cv", f"{p}dfl")) for k in keys)

    def _has_sab(idx: int) -> bool:
        p = f"model.{idx}."
        return any(
            k.startswith((f"{p}q.", f"{p}gamma", f"{p}edge_gate.", f"{p}illum_gate."))
            for k in keys
        )

    def _has_edge_gate(idx: int) -> bool:
        return any(k.startswith(f"model.{idx}.edge_gate.") for k in keys)

    def _has_illum_gate(idx: int) -> bool:
        p = f"model.{idx}."
        return any(
            k.startswith((f"{p}illum_gate.", f"{p}illum_qk_cwa.", f"{p}illum_scale"))
            for k in keys
        )

    def _has_illum_v2_bn(idx: int) -> bool:
        p = f"model.{idx}."
        return any(
            k.startswith((f"{p}illum_qk_cwa.bn.", f"{p}illum_gate.proj_bn.", f"{p}illum_gate.refine_bn."))
            for k in keys
        )

    if _has_detect(26) and _has_sab(23):
        return "full_edgesab"
    if _has_detect(25) and _has_sab(3):
        return "p2_edgesab_skip_p3"
    if _has_detect(25) and _has_sab(15) and _has_sab(18):
        return "p3_skip_inline_edgesab"
    if _has_detect(24):
        if _has_sab(3):
            if _has_illum_gate(3):
                return "p2_illumsab" if _has_illum_v2_bn(3) else "p2_illumsab_v1"
            if _has_edge_gate(3):
                return "p2_edgesab"
            return "p2_sab"
        if _has_sab(15):
            return "skip_p3_edgesab" if _has_edge_gate(15) else "skip_p3_sab"
        if _has_sab(17):
            return "p3_inline_edgesab" if _has_edge_gate(17) else "p3_inline_sab"
        if _has_sab(23):
            return "p3_prehead_edgesab"
    if _has_detect(23):
        return "base"
    return None


def _build_arch_cfg_dict(model_name: str, nc: int, arch: str) -> Dict[str, Any]:
    """Construct exact architecture cfg dict for any YOLOv11 ablation branch."""
    variant = model_name.lower().replace("yolov11", "").replace("yolo11", "")
    scales = {k: list(v) for k, v in YOLOV11_SCALES.items()}

    if arch in {"p2_illumsab", "p2_illumsab_v1", "p2_sab", "p2_edgesab", "p2_edgesab_skip_p3"}:
        if arch == "p2_illumsab":
            p2_cls = "IllumSAB"
            p2_args = [4, False, 2]
        elif arch == "p2_illumsab_v1":
            p2_cls = "IllumSAB"
            # Legacy PA1-v1: no BN, no darkness-weighted shift, illum_scale_init=0.5
            p2_args = [4, False, 2, None, 0.1, 4, 0.5, False, False]
        elif arch == "p2_sab":
            p2_cls = "SAB"
            p2_args = [4, False, 2]
        else:
            p2_cls = "EdgeSAB"
            p2_args = [4, False, 2]

        backbone_p2 = [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, p2_cls, p2_args],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 2, "C2PSA", [1024]],
        ]
        if arch in {"p2_illumsab", "p2_illumsab_v1", "p2_sab", "p2_edgesab"}:
            head_25 = [
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 7], 1, "Concat", [1]],
                [-1, 2, "C3k2", [512, False]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 5], 1, "Concat", [1]],
                [-1, 2, "C3k2", [256, False]],
                [-1, 1, "Conv", [256, 3, 2]],
                [[-1, 14], 1, "Concat", [1]],
                [-1, 2, "C3k2", [512, False]],
                [-1, 1, "Conv", [512, 3, 2]],
                [[-1, 11], 1, "Concat", [1]],
                [-1, 2, "C3k2", [1024, True]],
                [[17, 20, 23], 1, "Detect", ["nc"]],
            ]
            return {"nc": nc, "scale": variant, "scales": scales, "backbone": backbone_p2, "head": head_25}

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
    base_neck = [
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
    ]

    if arch == "base":
        head = base_neck + [[[16, 19, 22], 1, "Detect", ["nc"]]]
    elif arch in {"skip_p3_sab", "skip_p3_edgesab"}:
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
    elif arch in {"p3_inline_sab", "p3_inline_edgesab"}:
        attn_cls = "EdgeSAB" if arch == "p3_inline_edgesab" else "SAB"
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, attn_cls, [4, False, 2]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[17, 20, 23], 1, "Detect", ["nc"]],
        ]
    elif arch == "p3_skip_inline_edgesab":
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [4, 1, "EdgeSAB", [4, False, 2]],
            [[14, -1], 1, "Concat", [1]],
            [-1, 2, "C3k2", [256, False]],
            [-1, 1, "EdgeSAB", [4, False, 2]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "C3k2", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 2, "C3k2", [1024, True]],
            [[18, 21, 24], 1, "Detect", ["nc"]],
        ]
    elif arch == "p3_prehead_edgesab":
        head = base_neck + [
            [16, 1, "EdgeSAB", [4, False, 2]],
            [[23, 19, 22], 1, "Detect", ["nc"]],
        ]
    elif arch == "full_edgesab":
        head = base_neck + [
            [16, 1, "EdgeSAB", [4, False, 2]],
            [19, 1, "EdgeSAB", [4, False, 2]],
            [22, 1, "EdgeSAB", [4, False, 2]],
            [[23, 24, 25], 1, "Detect", ["nc"]],
        ]
    else:
        raise ValueError(f"Unsupported architecture key: '{arch}'")

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
    raw_sd: Dict[str, Any] = {}
    if not is_auto_or_none:
        if not os.path.exists(w_str):
            raise FileNotFoundError(f"Weights file not found: {w_str}")
        _register_ultralytics_unpickle_aliases()
        ckpt_meta = torch.load(w_str, map_location="cpu", weights_only=False)
        raw_sd = _extract_raw_sd(ckpt_meta)

    if model_override.strip():
        model_name = model_override.strip().lower()
    elif ckpt_meta is not None:
        inferred = infer_model_from_checkpoint(ckpt_meta, fallback_path=w_str)
        if inferred and get_model_family(inferred) == "yolov11":
            model_name = inferred
        else:
            model_name = _infer_variant_from_sd(raw_sd) or "yolov11s"
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
    if nc is None and raw_sd:
        nc = _infer_nc_from_sd(raw_sd)
    if nc is None:
        nc = 80 if w_str.lower() == "auto" else 12
    nc = int(nc)

    is_modular = (
        detect_is_modular_ckpt(raw_sd) if modular_override is None else bool(modular_override)
    )
    cfg_path = cfg_override if (cfg_override and os.path.exists(cfg_override)) else None

    resolved_arch = arch_override
    if resolved_arch == "auto":
        detected = detect_ckpt_arch(raw_sd) if raw_sd else None
        resolved_arch = (
            detected
            if detected in set(SUPPORTED_ARCH_CHOICES) - {"auto"}
            else "p2_illumsab"
        )

    if cfg_path is not None or is_modular or resolved_arch == "p2_illumsab":
        model = build_yolo_model(
            model_name=model_name,
            num_classes=nc,
            in_channels=3,
            cfg=cfg_path,
            modular=is_modular,
        )
    else:
        cfg_dict = _build_arch_cfg_dict(model_name, nc, resolved_arch)
        model = DetectionModel(cfg=cfg_dict, ch=3, nc=nc)
        if resolved_arch == "p2_illumsab_v1" and getattr(model, "illum_extractor", None) is not None:
            model.illum_extractor.log_mu = 0.0

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
        "arch": resolved_arch,
        "ckpt_names": ckpt_names,
        "weights_label": w_label,
    }
    model = model.to(device).eval()
    return model, model_name, nc, meta_info


def _build_valid_spatial_mask(
    feat_hw: Tuple[int, int],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Boolean mask `(H_f, W_f)` marking non-letterbox active image pixels."""
    h_f, w_f = feat_hw
    top, left, bot, right = crop_box
    sy = h_f / float(lb_size)
    sx = w_f / float(lb_size)
    r0 = max(0, min(h_f - 1, int(round(top * sy))))
    r1 = max(r0 + 1, min(h_f, int(round(bot * sy))))
    c0 = max(0, min(w_f - 1, int(round(left * sx))))
    c1 = max(c0 + 1, min(w_f, int(round(right * sx))))
    mask = np.zeros((h_f, w_f), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


@torch.no_grad()
def inspect_p2_attn_internals(
    p2_mod: nn.Module,
    x_in: torch.Tensor,
    illum_tensor: Optional[torch.Tensor] = None,
    linear_illum_tensor: Optional[torch.Tensor] = None,
    log_mu: float = 20.0,
    edge_tensor: Optional[torch.Tensor] = None,
    lb_size: int = 640,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Reproduce `IllumSAB.forward` (PA1-v2 or legacy PA1-v1, plus SAB / EdgeSAB fallback) step-by-step:
      - Multinex 4-channel Luminance Stack S_L (mu-law log-expanded for v2, linear for v1)
      - Darkness mask `1 - Y_Rec709_log` used by Darkness-Weighted FiLM Shift (PA1-v2)
      - Pre-Attention CWA maps A_qk & centered modulation weight W_qk in (0.1, 1.9)
      - Learned per-channel `0.9 * tanh(illum_scale)` vector
      - Raw Q vs Illumination-modulated Q' (and relative shift %)
      - 4-Head Transposed Channel-Channel Attention matrices (temp, diag, entropy)
      - Pre-FiLM vs Post-FiLM outputs, Retinex FiLM `scale` and `||shift||_2`
      - Residual `|gamma * out|`, signed energy shift `L2(post) - L2(pre)`
      - Active-region physical Dark (<0.15) / Mid (0.15..0.50) / Bright (>0.50) diagnostics
        always stratified on physical linear Y_Rec.709 for apple-to-apple v1 vs v2 comparisons.
    """
    b, _, h, w = x_in.shape
    is_illumsab = isinstance(p2_mod, IllumSAB)
    is_edgesab = isinstance(p2_mod, EdgeSAB)
    use_bn_v2 = bool(getattr(p2_mod, "use_bn", False)) if is_illumsab else False
    dark_shift_v2 = bool(getattr(p2_mod, "darkness_weighted_shift", False)) if is_illumsab else False
    is_v2 = is_illumsab and (use_bn_v2 or dark_shift_v2 or log_mu > 0.0)
    mod_name = ("IllumSAB-v2" if is_v2 else "IllumSAB-v1") if is_illumsab else ("EdgeSAB" if is_edgesab else "SAB")

    def _l2_map(t: torch.Tensor) -> np.ndarray:
        f = t[0].float()
        return torch.sqrt((f ** 2).mean(dim=0) + 1e-12).cpu().numpy()

    q_raw = p2_mod.q_dwconv(p2_mod.q(x_in))
    kv_sp = p2_mod.kv_dwconv(p2_mod.kv(x_in))
    k_raw, v_sp = kv_sp.chunk(2, dim=1)

    q_sp = q_raw
    k_sp = k_raw

    # Illumination extraction & Pre-Attention CWA modulation (PA1-v2 / v1)
    illum_scaled: Optional[torch.Tensor] = None
    illum_priors: Optional[Dict[str, np.ndarray]] = None
    linear_y709: Optional[np.ndarray] = None
    dark_mask_map: Optional[np.ndarray] = None
    qk_cwa_mean: Optional[np.ndarray] = None
    qk_weight_mean: Optional[np.ndarray] = None
    qk_weight_std: Optional[np.ndarray] = None
    illum_scale_eff: Optional[np.ndarray] = None
    q_illum_shift_pct: float = 0.0

    if illum_tensor is not None:
        if illum_tensor.shape[-2:] != (h, w):
            illum_scaled = F.interpolate(
                illum_tensor, size=(h, w), mode="bilinear", align_corners=False
            )
        else:
            illum_scaled = illum_tensor
        s_np = illum_scaled[0].float().cpu().numpy()  # (4, H, W)
        illum_priors = {
            "y_rec709": s_np[0],
            "y_vmax": s_np[1],
            "y_lightness": s_np[2],
            "y_l2": s_np[3],
            "prior_gap": np.clip(s_np[1] - s_np[2], 0.0, 1.0),
        }
        dark_mask_map = np.clip(1.0 - s_np[0], 0.0, 1.0)

    if linear_illum_tensor is not None:
        lin_t = (
            F.interpolate(linear_illum_tensor, size=(h, w), mode="bilinear", align_corners=False)
            if linear_illum_tensor.shape[-2:] != (h, w)
            else linear_illum_tensor
        )
        linear_y709 = lin_t[0, 0].float().cpu().numpy()
    elif illum_priors is not None:
        linear_y709 = illum_priors["y_rec709"]

    if is_illumsab and illum_scaled is not None:
        il_t = illum_scaled.to(dtype=x_in.dtype)
        qk_attn = p2_mod.illum_qk_cwa(il_t)  # (1, inner_dim, H, W)
        eff_scale_t = 0.9 * torch.tanh(p2_mod.illum_scale.float())
        qk_weight = 1.0 + eff_scale_t.to(dtype=q_raw.dtype) * (2.0 * qk_attn - 1.0)
        q_sp = q_raw * qk_weight
        k_sp = k_raw * qk_weight

        qk_cwa_mean = qk_attn[0].float().mean(dim=0).cpu().numpy()
        qk_weight_mean = qk_weight[0].float().mean(dim=0).cpu().numpy()
        qk_weight_std = qk_weight[0].float().std(dim=0).cpu().numpy()
        illum_scale_eff = eff_scale_t.view(-1).cpu().numpy()
        q_raw_norm = float(torch.linalg.vector_norm(q_raw[0].float()).item())
        q_diff_norm = float(torch.linalg.vector_norm((q_sp - q_raw)[0].float()).item())
        q_illum_shift_pct = 100.0 * q_diff_norm / (q_raw_norm + 1e-8)

    # EdgeSAB fallback support
    edge_scaled: Optional[torch.Tensor] = None
    edge_scale_val: Optional[float] = None
    edge_gate_mean: Optional[np.ndarray] = None
    if is_edgesab and edge_tensor is not None:
        if edge_tensor.shape[-2:] != (h, w):
            e_bil = F.interpolate(edge_tensor, size=(h, w), mode="bilinear", align_corners=False)
            e_max = F.adaptive_max_pool2d(edge_tensor, output_size=(h, w))
            edge_scaled = 0.5 * e_bil + 0.5 * e_max
        else:
            edge_scaled = edge_tensor
        edge_scale_val = float(p2_mod.edge_scale.float().item())
        ew = (1.0 + p2_mod.edge_scale * edge_scaled).to(q_sp.dtype)
        q_sp = q_sp * ew
        k_sp = k_sp * ew

    # Transposed Channel Self-Attention
    q_h = q_sp.reshape(b, p2_mod.num_heads, p2_mod.head_dim, h * w)
    k_h = k_sp.reshape(b, p2_mod.num_heads, p2_mod.head_dim, h * w)
    v_h = v_sp.reshape(b, p2_mod.num_heads, p2_mod.head_dim, h * w)

    q_norm = F.normalize(q_h.float(), dim=-1, eps=1e-6)
    k_norm = F.normalize(k_h.float(), dim=-1, eps=1e-6)
    attn = F.softmax((q_norm @ k_norm.transpose(-2, -1)) * p2_mod.temperature.float(), dim=-1)

    attn_out = (attn @ v_h.float()).to(x_in.dtype).reshape(b, p2_mod.inner_dim, h, w)
    ffn_gate = p2_mod.ffn_gate(attn_out)
    proj_pre_gate = p2_mod.project_out(attn_out * ffn_gate)
    proj_post_gate = proj_pre_gate

    # Post-Attention Retinex-Delta FiLM Gate (PA1-v2 / v1)
    film_scale_mean: Optional[np.ndarray] = None
    film_scale_std: Optional[np.ndarray] = None
    film_shift_l2: Optional[np.ndarray] = None
    film_shift_mean: Optional[np.ndarray] = None
    film_delta_pct: float = 0.0

    if is_illumsab and illum_scaled is not None:
        il_t = illum_scaled.to(dtype=x_in.dtype)
        f_scale, f_shift = p2_mod.illum_gate(il_t)
        proj_post_gate = proj_pre_gate * f_scale + f_shift

        film_scale_mean = f_scale[0].float().mean(dim=0).cpu().numpy()
        film_scale_std = f_scale[0].float().std(dim=0).cpu().numpy()
        film_shift_l2 = _l2_map(f_shift)
        film_shift_mean = f_shift[0].float().mean(dim=0).cpu().numpy()
        pre_n = float(torch.linalg.vector_norm(proj_pre_gate[0].float()).item())
        dif_n = float(torch.linalg.vector_norm((proj_post_gate - proj_pre_gate)[0].float()).item())
        film_delta_pct = 100.0 * dif_n / (pre_n + 1e-8)
    elif is_edgesab and edge_scaled is not None:
        gate_t = p2_mod.edge_gate(edge_scaled.to(x_in.dtype))
        edge_gate_mean = gate_t[0].float().mean(dim=0).cpu().numpy()
        proj_post_gate = proj_pre_gate * gate_t

    residual = p2_mod.gamma * proj_post_gate
    y_out = x_in + residual

    pre_l2 = _l2_map(x_in)
    post_l2 = _l2_map(y_out)
    residual_l2 = _l2_map(residual)
    signed_l2_shift = post_l2 - pre_l2

    x_norm = float(torch.linalg.vector_norm(x_in[0].float()).item())
    res_norm = float(torch.linalg.vector_norm(residual[0].float()).item())
    rel_residual_pct = 100.0 * res_norm / (x_norm + 1e-8)

    attn_np = attn[0].float().cpu().numpy()
    max_ent = math.log(max(p2_mod.head_dim, 2))
    head_entropies: List[float] = []
    head_diag_mass: List[float] = []
    for hi in range(attn_np.shape[0]):
        p = np.clip(attn_np[hi], 1e-12, 1.0)
        head_entropies.append(float(-np.sum(p * np.log(p), axis=-1).mean() / max_ent))
        head_diag_mass.append(float(np.diag(p).mean()))

    gamma_vec = p2_mod.gamma.detach().float().view(-1).cpu().numpy()
    temps = [float(t) for t in p2_mod.temperature.detach().float().view(-1).cpu().tolist()]

    # Active-region (non-letterbox) physical Dark / Mid / Bright regime diagnostics (linear Y_Rec.709)
    valid_mask = (
        _build_valid_spatial_mask((h, w), lb_size, crop_box)
        if crop_box is not None
        else np.ones((h, w), dtype=bool)
    )
    regime_stats: List[Dict[str, Any]] = []
    scene_luma_stats: Dict[str, float] = {}
    if linear_y709 is not None:
        y_act = linear_y709[valid_mask]
        scene_luma_stats = {
            "mean": float(y_act.mean()),
            "p10": float(np.percentile(y_act, 10)),
            "p90": float(np.percentile(y_act, 90)),
            "dark_pct": float((y_act < 0.15).mean() * 100.0),
            "mid_pct": float(((y_act >= 0.15) & (y_act <= 0.50)).mean() * 100.0),
            "bright_pct": float((y_act > 0.50).mean() * 100.0),
        }
        regimes = [
            ("Dark (<0.15)", valid_mask & (linear_y709 < 0.15)),
            ("Mid (0.15-0.50)", valid_mask & (linear_y709 >= 0.15) & (linear_y709 <= 0.50)),
            ("Bright (>0.50)", valid_mask & (linear_y709 > 0.50)),
            ("All Active", valid_mask),
        ]
        n_valid = max(1, int(valid_mask.sum()))
        for r_name, r_mask in regimes:
            cnt = int(r_mask.sum())
            if cnt == 0:
                regime_stats.append({
                    "name": r_name,
                    "pct": 0.0,
                    "count": 0,
                    "w_qk": float("nan"),
                    "film_scale": float("nan"),
                    "film_shift": float("nan"),
                    "pre_l2": float("nan"),
                    "post_l2": float("nan"),
                    "delta_l2": float("nan"),
                    "boost_pct": float("nan"),
                })
            else:
                regime_stats.append({
                    "name": r_name,
                    "pct": 100.0 * cnt / n_valid,
                    "count": cnt,
                    "w_qk": float(qk_weight_mean[r_mask].mean()) if qk_weight_mean is not None else float("nan"),
                    "film_scale": float(film_scale_mean[r_mask].mean()) if film_scale_mean is not None else float("nan"),
                    "film_shift": float(film_shift_l2[r_mask].mean()) if film_shift_l2 is not None else float("nan"),
                    "pre_l2": float(pre_l2[r_mask].mean()),
                    "post_l2": float(post_l2[r_mask].mean()),
                    "delta_l2": float(signed_l2_shift[r_mask].mean()),
                    "boost_pct": float((signed_l2_shift[r_mask] > 0).mean() * 100.0),
                })

    return {
        "mod_name": mod_name,
        "is_illumsab": is_illumsab,
        "is_v2": is_v2,
        "use_bn": use_bn_v2,
        "darkness_weighted_shift": dark_shift_v2,
        "log_mu": float(log_mu),
        "illum_scale_init_eff": 0.9 * math.tanh(0.3 if is_v2 else 0.5),
        "is_edgesab": is_edgesab,
        "dim": int(p2_mod.dim),
        "inner_dim": int(p2_mod.inner_dim),
        "num_heads": int(p2_mod.num_heads),
        "head_dim": int(p2_mod.head_dim),
        "illum_priors": illum_priors,
        "linear_y709": linear_y709,
        "dark_mask_map": dark_mask_map,
        "scene_luma_stats": scene_luma_stats,
        "qk_cwa_mean": qk_cwa_mean,
        "qk_weight_mean": qk_weight_mean,
        "qk_weight_std": qk_weight_std,
        "illum_scale_eff": illum_scale_eff,
        "q_raw_l2": _l2_map(q_raw),
        "q_mod_l2": _l2_map(q_sp),
        "k_raw_l2": _l2_map(k_raw),
        "k_mod_l2": _l2_map(k_sp),
        "v_l2": _l2_map(v_sp),
        "q_illum_shift_pct": q_illum_shift_pct,
        "attn_out_l2": _l2_map(attn_out),
        "ffn_gate_l2": _l2_map(ffn_gate),
        "proj_pre_gate_l2": _l2_map(proj_pre_gate),
        "proj_post_gate_l2": _l2_map(proj_post_gate),
        "film_scale_mean": film_scale_mean,
        "film_scale_std": film_scale_std,
        "film_shift_l2": film_shift_l2,
        "film_shift_mean": film_shift_mean,
        "film_delta_pct": film_delta_pct,
        "residual_l2": residual_l2,
        "signed_l2_shift": signed_l2_shift,
        "rel_residual_pct": rel_residual_pct,
        "attn_matrices": attn_np,
        "head_entropies": head_entropies,
        "head_diag_mass": head_diag_mass,
        "temperatures": temps,
        "gamma_vec": gamma_vec,
        "gamma_mean": float(gamma_vec.mean()),
        "gamma_abs_mean": float(np.abs(gamma_vec).mean()),
        "gamma_min": float(gamma_vec.min()),
        "gamma_max": float(gamma_vec.max()),
        "gamma_std": float(gamma_vec.std()),
        "edge_scale": edge_scale_val,
        "edge_gate_mean": edge_gate_mean,
        "regime_stats": regime_stats,
    }


class YOLOv11BackboneIllumSABRecorder:
    """
    Dynamically hooks YOLOv11 Backbone layers (L00..L11 for P2-IllumSAB / P2-SAB / P2-EdgeSAB,
    or L00..L10 for Base / Skip-P3), capturing P2 input and Illumination/Edge extractors.
    """

    def __init__(self, model: nn.Module, include_stem: bool = False):
        self.model = de_parallel(model)
        self.include_stem = include_stem
        self.handles: List[Any] = []
        self.records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.edge_extractor_mod: Optional[EdgeExtractor] = getattr(self.model, "edge_extractor", None)
        self.illum_extractor_mod: Optional[MultiPriorIllumExtractor] = getattr(
            self.model, "illum_extractor", None
        )
        if self.illum_extractor_mod is None and hasattr(self.model, "backbone"):
            self.illum_extractor_mod = getattr(self.model.backbone, "illum_extractor", None)
        self.p2_attn_mod: Optional[nn.Module] = None
        self.p2_attn_input: Optional[torch.Tensor] = None
        self.skip_p3_sab_mod: Optional[nn.Module] = None
        self.stage_slots: Dict[str, str] = {}
        self.arch_display_tag: str = "Base"
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
                self.p2_attn_input = inputs[0].detach().clone()
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
        p2_types = (IllumSAB, SAB, EdgeSAB)
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            has_p2_attn = len(layers) > 3 and isinstance(layers[3], p2_types)

            if has_p2_attn:
                if isinstance(layers[3], IllumSAB):
                    p2_tag = "IllumSAB"
                elif isinstance(layers[3], EdgeSAB):
                    p2_tag = "EdgeSAB"
                else:
                    p2_tag = "SAB"

                spec = [
                    (0, "L00_Stem_Conv3x3_P1_2", "P1/2 (Stem)", 2, "p1", True, "p1"),
                    (1, "L01_Conv3x3_P2_4", "P2/4 (Stage1)", 4, "p2", False, ""),
                    (2, f"L02_C3k2_P2_4_Pre{p2_tag}", "P2/4 (Stage1)", 4, "p2", True, "p2_pre"),
                    (3, f"L03_{p2_tag}_P2_4_160", f"P2/4 ({p2_tag})", 4, "p2_sab", True, "p2"),
                    (4, "L04_Conv3x3_P3_8", "P3/8 (Stage2)", 8, "p3", False, ""),
                    (5, "L05_C3k2_P3_8_SkipP3", "P3/8 (Stage2)", 8, "p3", True, "p3"),
                    (6, "L06_Conv3x3_P4_16", "P4/16 (Stage3)", 16, "p4", False, ""),
                    (7, "L07_C3k2_P4_16_SkipP4", "P4/16 (Stage3)", 16, "p4", True, "p4"),
                    (8, "L08_Conv3x3_P5_32", "P5/32 (Stage4)", 32, "p5", False, ""),
                    (9, "L09_C3k2_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_c3k2"),
                    (10, "L10_SPPF_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_sppf"),
                    (11, "L11_C2PSA_P5_32_SkipP5", "P5/32 (Stage4)", 32, "p5", True, "p5"),
                ]
                self.p2_attn_mod = layers[3]
                self.arch_display_tag = f"P2-{p2_tag}"
                if len(layers) > 16 and isinstance(layers[16], p2_types):
                    sp3_tag = type(layers[16]).__name__
                    spec.append((16, f"L16_{sp3_tag}_SkipP3_80", f"Neck (Skip-P3 {sp3_tag})", 8, "p3_sab", True, "p3_sab"))
                    self.skip_p3_sab_mod = layers[16]
                    self.arch_display_tag = f"P2-{p2_tag} + Skip-P3 {sp3_tag}"
            else:
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
                self.arch_display_tag = "Base"
                if len(layers) > 15 and isinstance(layers[15], p2_types):
                    sp3_tag = type(layers[15]).__name__
                    spec.append((15, f"L15_{sp3_tag}_SkipP3_80", f"Neck (Skip-P3 {sp3_tag})", 8, "p3_sab", True, "p3_sab"))
                    self.skip_p3_sab_mod = layers[15]
                    self.arch_display_tag = f"Skip-P3 {sp3_tag}"

            for idx, key, group, stride, acc, is_out, slot in spec:
                if idx == 0 and not self.include_stem:
                    continue
                if idx < len(layers):
                    cap_in = (layers[idx] is self.p2_attn_mod)
                    self.handles.append(
                        layers[idx].register_forward_hook(
                            self._make_hook(key, idx, group, stride, acc, is_out, capture_input=cap_in)
                        )
                    )
                    if slot:
                        self.stage_slots[slot] = key

        elif hasattr(self.model, "backbone"):
            bb = self.model.backbone
            has_p2_attn = hasattr(bb, "attn_p2") and isinstance(bb.attn_p2, p2_types)
            p2_tag = type(bb.attn_p2).__name__ if has_p2_attn else ""
            spec_mod: List[Tuple[int, str, str, int, str, bool, str, nn.Module]] = [
                (0, "L00_Stem_Conv3x3_P1_2", "P1/2 (Stem)", 2, "p1", True, "p1", bb.stem),
                (1, "L01_Conv3x3_P2_4", "P2/4 (Stage1)", 4, "p2", False, "", bb.stage1[0]),
                (
                    2,
                    f"L02_C3k2_P2_4_Pre{p2_tag}" if has_p2_attn else "L02_C3k2_P2_4",
                    "P2/4 (Stage1)",
                    4,
                    "p2",
                    True,
                    "p2_pre" if has_p2_attn else "p2",
                    bb.stage1[1],
                ),
            ]
            if has_p2_attn:
                self.p2_attn_mod = bb.attn_p2
                self.arch_display_tag = f"P2-{p2_tag}"
                spec_mod.append(
                    (3, f"L03_{p2_tag}_P2_4_160", f"P2/4 ({p2_tag})", 4, "p2_sab", True, "p2", bb.attn_p2)
                )
            off = 1 if has_p2_attn else 0
            spec_mod.extend([
                (3 + off, f"L{3 + off:02d}_Conv3x3_P3_8", "P3/8 (Stage2)", 8, "p3", False, "", bb.stage2[0]),
                (4 + off, f"L{4 + off:02d}_C3k2_P3_8_SkipP3", "P3/8 (Stage2)", 8, "p3", True, "p3", bb.stage2[1]),
                (5 + off, f"L{5 + off:02d}_Conv3x3_P4_16", "P4/16 (Stage3)", 16, "p4", False, "", bb.stage3[0]),
                (6 + off, f"L{6 + off:02d}_C3k2_P4_16_SkipP4", "P4/16 (Stage3)", 16, "p4", True, "p4", bb.stage3[1]),
                (7 + off, f"L{7 + off:02d}_Conv3x3_P5_32", "P5/32 (Stage4)", 32, "p5", False, "", bb.stage4[0]),
                (8 + off, f"L{8 + off:02d}_C3k2_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_c3k2", bb.stage4[1]),
                (9 + off, f"L{9 + off:02d}_SPPF_P5_32", "P5/32 (Stage4)", 32, "p5", False, "s4_sppf", bb.stage4[2]),
                (10 + off, f"L{10 + off:02d}_C2PSA_P5_32_SkipP5", "P5/32 (Stage4)", 32, "p5", True, "p5", bb.stage4[3]),
            ])
            if hasattr(self.model, "neck") and hasattr(self.model.neck, "attn_p3"):
                self.skip_p3_sab_mod = self.model.neck.attn_p3
                sp3_tag = type(self.skip_p3_sab_mod).__name__
                spec_mod.append(
                    (15 + off, f"L{15 + off:02d}_{sp3_tag}_SkipP3_80", f"Neck (Skip-P3 {sp3_tag})", 8, "p3_sab", True, "p3_sab", self.skip_p3_sab_mod)
                )

            for idx, key, group, stride, acc, is_out, slot, mod in spec_mod:
                if idx == 0 and not self.include_stem:
                    continue
                cap_in = (mod is self.p2_attn_mod)
                self.handles.append(
                    mod.register_forward_hook(
                        self._make_hook(key, idx, group, stride, acc, is_out, capture_input=cap_in)
                    )
                )
                if slot:
                    self.stage_slots[slot] = key
        else:
            raise AttributeError("Unsupported YOLOv11 model structure (missing `.model` and `.backbone`).")

    def clear(self) -> None:
        self.records.clear()
        self.p2_attn_input = None

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==================================================================================================
# 3. Preprocessing, Feature Projections & Rendering Helpers
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


def render_signed_diverging_map(
    base_bgr: np.ndarray,
    diff2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    clip_percentile: float = 99.0,
    blend_alpha: float = 0.78,
    min_limit: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a diverging signed map centered at 0:
      - Negative (suppressed / <1.0): Deep Blue -> Bright Cyan
      - Zero (unchanged / ==1.0): Dark Neutral Slate
      - Positive (boosted / >1.0): Deep Crimson -> Bright Orange-Gold
    """
    h, w = base_bgr.shape[:2]
    aligned = unpad_and_resize_2d(diff2d, lb_size, crop_box, (w, h), nearest=False)
    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
    limit = max(float(np.percentile(np.abs(aligned), clip_percentile)), min_limit)
    norm = np.clip(aligned / limit, -1.0, 1.0)

    pos = np.clip(norm, 0.0, 1.0)
    neg = np.clip(-norm, 0.0, 1.0)

    base_val = 20.0
    b = base_val + neg * 235.0 + np.clip(pos - 0.65, 0.0, 0.35) / 0.35 * 90.0
    g = base_val + (neg ** 1.25) * 185.0 + (pos ** 1.35) * 205.0
    r = base_val + (pos ** 0.75) * 235.0 + np.clip(neg - 0.70, 0.0, 0.30) / 0.30 * 55.0

    pure_bgr = np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(base_bgr, 1.0 - blend_alpha, pure_bgr, blend_alpha, 0.0)
    return blended, pure_bgr


def render_channel_attn_panel(
    attn_matrices: np.ndarray,
    temperatures: List[float],
    head_entropies: List[float],
    head_diag_mass: List[float],
    target_hw: Tuple[int, int],
    colormap_id: int,
) -> np.ndarray:
    ph, pw = target_hw
    canvas = np.full((ph, pw, 3), (18, 20, 25), dtype=np.uint8)
    num_heads = attn_matrices.shape[0]
    ncols = 2 if num_heads > 1 else 1
    nrows = math.ceil(num_heads / ncols)

    pad = 6
    hdr_h = 18
    cell_w = max(24, (pw - pad * (ncols + 1)) // ncols)
    cell_h = max(24, (ph - pad * (nrows + 1)) // nrows)
    mat_h = max(16, cell_h - hdr_h)
    side = min(mat_h, max(16, cell_w))

    vmin = float(attn_matrices.min())
    vmax = float(np.percentile(attn_matrices, 99.5))
    if vmax <= vmin + 1e-8:
        vmax = float(attn_matrices.max())

    for hi in range(num_heads):
        r, c = hi // ncols, hi % ncols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (28, 31, 38), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (58, 64, 76), 1)

        t_val = temperatures[hi] if hi < len(temperatures) else 1.0
        ent_val = head_entropies[hi] if hi < len(head_entropies) else 0.0
        diag_val = head_diag_mass[hi] if hi < len(head_diag_mass) else 0.0
        lbl = f"H{hi} t={t_val:.2f} d={diag_val:.2f} E={ent_val:.2f}"
        f_scale = min(0.32, max(0.24, cell_w / 520.0))
        cv2.putText(
            canvas, lbl, (x0 + 4, y0 + 13),
            cv2.FONT_HERSHEY_SIMPLEX, f_scale, (220, 228, 240), 1, cv2.LINE_AA
        )

        u8 = normalize_map_to_uint8(attn_matrices[hi], fixed_range=(vmin, vmax))
        resized = cv2.resize(cv2.applyColorMap(u8, colormap_id), (side, side), interpolation=cv2.INTER_NEAREST)
        mx = x0 + (cell_w - side) // 2
        my = y0 + hdr_h + max(0, (mat_h - side) // 2)
        canvas[my : my + side, mx : mx + side] = resized
        cv2.rectangle(canvas, (mx, my), (mx + side - 1, my + side - 1), (95, 102, 118), 1)

    return canvas


def render_standalone_channel_attn_figure(
    info: Dict[str, Any],
    colormap_id: int,
    model_title: str,
) -> np.ndarray:
    attn_mats = info["attn_matrices"]
    temps = info["temperatures"]
    ents = info["head_entropies"]
    diags = info["head_diag_mass"]
    num_heads, dh, _ = attn_mats.shape
    mod_lbl = info["mod_name"]

    cell_side = 280
    tiles: List[np.ndarray] = []
    vmin = float(attn_mats.min())
    vmax = float(np.percentile(attn_mats, 99.5))
    if vmax <= vmin + 1e-8:
        vmax = float(attn_mats.max())

    for hi in range(num_heads):
        u8 = normalize_map_to_uint8(attn_mats[hi], fixed_range=(vmin, vmax))
        big = cv2.resize(cv2.applyColorMap(u8, colormap_id), (cell_side, cell_side), interpolation=cv2.INTER_NEAREST)
        if dh >= 8:
            step = max(1, dh // 4)
            for k in range(step, dh, step):
                pos = int(round(k * cell_side / dh))
                cv2.line(big, (pos, 0), (pos, cell_side - 1), (55, 58, 68), 1)
                cv2.line(big, (0, pos), (cell_side - 1, pos), (55, 58, 68), 1)
        tiles.append(
            make_labeled_tile(
                big,
                f"Head {hi} ({dh}x{dh} Channel Attn)",
                f"temp={temps[hi]:.3f} | diag={diags[hi]:.3f} | entropy={ents[hi]:.3f}",
                STAGE_ACCENT_BGR["p2_sab"],
            )
        )

    return assemble_grid(
        tiles,
        ncols=min(4, num_heads),
        banner_title=f"P2-{mod_lbl} Transposed Channel-Channel Attention Maps (Q'_norm @ K'_norm^T) - {model_title}",
        banner_sub=(
            f"dim={info['dim']}, inner_dim={info['inner_dim']}, "
            f"heads={num_heads}, head_dim={dh} | Each cell (i,j) = attention weight from channel i to channel j"
        ),
        colormap_id=colormap_id,
    )


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
    t_scale = min(0.45, max(0.28, w / 860.0))
    cv2.putText(tile, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (240, 244, 252), 1, cv2.LINE_AA)
    if subtitle:
        s_scale = min(0.35, max(0.24, w / 1060.0))
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
        max_txt_w = max(120, canvas_w - 36)
        b_scale = 0.64
        b_thick = 2
        while b_scale > 0.28:
            b_thick = 2 if b_scale >= 0.48 else 1
            (title_w, _), _ = cv2.getTextSize(banner_title, cv2.FONT_HERSHEY_SIMPLEX, b_scale, b_thick)
            if title_w <= max_txt_w:
                break
            b_scale -= 0.03
        cv2.putText(canvas, banner_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, b_scale, (245, 248, 255), b_thick, cv2.LINE_AA)
        if banner_sub:
            s_scale = 0.41
            while s_scale > 0.24:
                (sub_w, _), _ = cv2.getTextSize(banner_sub, cv2.FONT_HERSHEY_SIMPLEX, s_scale, 1)
                if sub_w <= max_txt_w:
                    break
                s_scale -= 0.02
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
# 4. Composite Diagnostic Figures (Pyramid, Multinex Priors & CWA, IllumSAB Internals, Stats, Compare)
# ==================================================================================================

def build_pyramid_hero_figure(
    raw_panel: np.ndarray,
    ref_panel: np.ndarray,
    det_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    stage_slots: Dict[str, str],
    arch_tag: str,
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
        banner_title=f"YOLOv11 {arch_tag} Backbone Pyramid (160 -> 80 -> 40 -> 20) - {model_title}",
        banner_sub="Row 1: Aligned L2 Overlay | Row 2: Discrete Native-Grid Resolution | Row 3: Top-3 PCA Semantic RGB",
        colormap_id=colormap_id,
    )


def build_multinex_priors_and_cwa_figure(
    raw_panel: np.ndarray,
    ref_panel: np.ndarray,
    info: Dict[str, Any],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    attn_colormap_id: int,
    clip_pct: float,
    model_title: str,
    img_name: str,
    illum_out_dir: Path,
) -> Optional[np.ndarray]:
    """
    Create `01_multinex_illumination_priors_and_cwa_160x160.png` (2 rows x 5 cols)
    and export standalone borderless assets under `illum/`:
      Row 1 (Multinex 4-Channel Analytic Luminance Stack S_L at 160x160):
        1. Raw Low-Light Input (+ Dark/Mid/Bright % stats)
        2. Y_Rec.709 (ITU-R BT.709 visual brightness)
        3. Y_vmax (Max-RGB / highlight energy)
        4. Y_lightness (HSL lightness / contrast reg.)
        5. Y_L2 (Normalized RGB vector energy)
      Row 2 (Prior Diversity + PA1 Pre-Attn CWA Q/K Weight + Post-Attn Retinex FiLM):
        6. Chromatic-Highlight Gap (Y_vmax - Y_lightness)
        7. Pre-Attn CWA Attention A_qk = mean_c(CWA(S_L))
        8. Pre-Attn Q/K Weight W_qk (Blue <1.0 suppress | Gold >1.0 boost)
        9. Post-Attn Retinex FiLM Scale s (Blue <1.0 damp | Gold >1.0 amplify)
       10. Post-Attn Retinex FiLM Shift ||b||_2 (additive compensation energy)
    """
    priors = info.get("illum_priors")
    if priors is None:
        return None

    illum_out_dir.mkdir(parents=True, exist_ok=True)
    acc_il = STAGE_ACCENT_BGR["illum"]
    acc_cwa = STAGE_ACCENT_BGR["cwa"]
    acc_film = STAGE_ACCENT_BGR["film"]
    luma_cmap = cv2.COLORMAP_INFERNO

    _, heat_y709, _ = render_views_for_map(ref_panel, priors["y_rec709"], lb_size, crop_box, luma_cmap, alpha=1.0, fixed_range=(0.0, 1.0))
    _, heat_vmax, _ = render_views_for_map(ref_panel, priors["y_vmax"], lb_size, crop_box, luma_cmap, alpha=1.0, fixed_range=(0.0, 1.0))
    _, heat_lght, _ = render_views_for_map(ref_panel, priors["y_lightness"], lb_size, crop_box, luma_cmap, alpha=1.0, fixed_range=(0.0, 1.0))
    _, heat_yl2, _ = render_views_for_map(ref_panel, priors["y_l2"], lb_size, crop_box, luma_cmap, alpha=1.0, fixed_range=(0.0, 1.0))
    ov_gap, heat_gap, _ = render_views_for_map(ref_panel, priors["prior_gap"], lb_size, crop_box, attn_colormap_id, alpha=0.65, clip_percentile=clip_pct)

    cv2.imwrite(str(illum_out_dir / "s_l_y_rec709.png"), heat_y709)
    cv2.imwrite(str(illum_out_dir / "s_l_y_vmax.png"), heat_vmax)
    cv2.imwrite(str(illum_out_dir / "s_l_y_lightness.png"), heat_lght)
    cv2.imwrite(str(illum_out_dir / "s_l_y_l2.png"), heat_yl2)
    cv2.imwrite(str(illum_out_dir / "s_l_vmax_minus_lightness.png"), heat_gap)
    if info.get("linear_y709") is not None:
        _, heat_lin, _ = render_views_for_map(ref_panel, info["linear_y709"], lb_size, crop_box, luma_cmap, alpha=1.0, fixed_range=(0.0, 1.0))
        cv2.imwrite(str(illum_out_dir / "s_l_linear_y_rec709.png"), heat_lin)

    dark_mask = info.get("dark_mask_map")
    ov_dark, heat_dark = None, None
    if dark_mask is not None:
        ov_dark, heat_dark, _ = render_views_for_map(ref_panel, dark_mask, lb_size, crop_box, cv2.COLORMAP_VIRIDIS, alpha=0.62, fixed_range=(0.0, 1.0))
        cv2.imwrite(str(illum_out_dir / "darkness_mask_overlay.png"), ov_dark)
        cv2.imwrite(str(illum_out_dir / "darkness_mask_heatmap.png"), heat_dark)

    mu_val = float(info.get("log_mu", 0.0))
    mode_tag = f"log mu={mu_val:.0f}" if mu_val > 0.0 else "linear"
    ls = info.get("scene_luma_stats", {})
    raw_sub = (
        f"Dark:{ls.get('dark_pct', 0):.0f}% Mid:{ls.get('mid_pct', 0):.0f}% Bright:{ls.get('bright_pct', 0):.0f}% | {img_name}"
        if ls else img_name
    )
    row1 = [
        make_labeled_tile(raw_panel, "1. Raw Input (Linear Luma)", raw_sub, (140, 150, 168)),
        make_labeled_tile(heat_y709, f"2. S_L[0]: Y_Rec.709 ({mode_tag})", f"mean={priors['y_rec709'].mean():.3f} | p90={np.percentile(priors['y_rec709'], 90):.3f}", acc_il),
        make_labeled_tile(heat_vmax, f"3. S_L[1]: Y_vmax ({mode_tag})", f"mean={priors['y_vmax'].mean():.3f} | max={priors['y_vmax'].max():.3f}", acc_il),
        make_labeled_tile(heat_lght, f"4. S_L[2]: Y_light ({mode_tag})", f"mean={priors['y_lightness'].mean():.3f} | contrast reg.", acc_il),
        make_labeled_tile(heat_yl2, f"5. S_L[3]: Y_L2 ({mode_tag})", f"mean={priors['y_l2'].mean():.3f} | norm [0,1]", acc_il),
    ]

    if info.get("darkness_weighted_shift", False) and ov_dark is not None:
        tile6 = make_labeled_tile(
            ov_dark,
            "6. Darkness Mask (1 - Y_log)",
            f"Weights FiLM shift | mean={dark_mask.mean():.3f} [{dark_mask.min():.2f}..{dark_mask.max():.2f}]",
            acc_il,
        )
    else:
        tile6 = make_labeled_tile(
            ov_gap,
            "6. Prior Gap (Y_vmax - Y_light)",
            f"Colored lights/edges | mean={priors['prior_gap'].mean():.3f}",
            acc_il,
        )
    row2 = [tile6]

    if info["is_illumsab"] and info["qk_cwa_mean"] is not None:
        ov_cwa, heat_cwa, _ = render_views_for_map(
            ref_panel, info["qk_cwa_mean"], lb_size, crop_box, attn_colormap_id, alpha=0.60, fixed_range=(0.0, 1.0)
        )
        wq_blend, wq_pure = render_signed_diverging_map(
            ref_panel, info["qk_weight_mean"] - 1.0, lb_size, crop_box, clip_percentile=99.0, blend_alpha=0.78, min_limit=0.02
        )
        fs_blend, fs_pure = render_signed_diverging_map(
            ref_panel, info["film_scale_mean"] - 1.0, lb_size, crop_box, clip_percentile=99.0, blend_alpha=0.78, min_limit=0.02
        )
        ov_shift, heat_shift, _ = render_views_for_map(
            ref_panel, info["film_shift_l2"], lb_size, crop_box, attn_colormap_id, alpha=0.60, clip_percentile=clip_pct
        )

        cv2.imwrite(str(illum_out_dir / "cwa_qk_attn_overlay.png"), ov_cwa)
        cv2.imwrite(str(illum_out_dir / "cwa_qk_attn_heatmap.png"), heat_cwa)
        cv2.imwrite(str(illum_out_dir / "qk_weight_diverging.png"), wq_pure)
        cv2.imwrite(str(illum_out_dir / "qk_weight_overlay.png"), wq_blend)
        cv2.imwrite(str(illum_out_dir / "film_scale_diverging.png"), fs_pure)
        cv2.imwrite(str(illum_out_dir / "film_scale_overlay.png"), fs_blend)
        cv2.imwrite(str(illum_out_dir / "film_shift_l2_overlay.png"), ov_shift)
        cv2.imwrite(str(illum_out_dir / "film_shift_l2_heatmap.png"), heat_shift)

        wq_m = info["qk_weight_mean"]
        wq_s = float(info["qk_weight_std"].mean()) if info.get("qk_weight_std") is not None else 0.0
        fs_m = info["film_scale_mean"]
        fs_s = float(info["film_scale_std"].mean()) if info.get("film_scale_std") is not None else 0.0
        sh_m = info["film_shift_l2"]
        cwa_lbl = "7. Pre-Attn CWA A_qk (DW7+BN)" if info.get("use_bn", False) else "7. Pre-Attn CWA A_qk (7x7 DW)"
        sh_lbl = "10. Dark-Weighted Shift ||d*b||" if info.get("darkness_weighted_shift", False) else "10. Post-Attn FiLM Shift ||b||"

        row2.extend([
            make_labeled_tile(
                ov_cwa,
                cwa_lbl,
                f"mean={info['qk_cwa_mean'].mean():.3f} [{info['qk_cwa_mean'].min():.2f}..{info['qk_cwa_mean'].max():.2f}]",
                acc_cwa,
            ),
            make_labeled_tile(
                wq_blend,
                "8. Pre-Attn Q/K Weight W_qk",
                f"mean={wq_m.mean():.3f} [{wq_m.min():.2f}..{wq_m.max():.2f}] ch-std={wq_s:.3f}",
                acc_cwa,
            ),
            make_labeled_tile(
                fs_blend,
                "9. Post-Attn Retinex FiLM Scale",
                f"mean={fs_m.mean():.3f} [{fs_m.min():.2f}..{fs_m.max():.2f}] ch-std={fs_s:.3f}",
                acc_film,
            ),
            make_labeled_tile(
                ov_shift,
                sh_lbl,
                f"mean={sh_m.mean():.4f} | max={sh_m.max():.3f}",
                acc_film,
            ),
        ])
    else:
        blank = np.full_like(ref_panel, (18, 20, 24))
        for label in ("7. CWA (IllumSAB only)", "8. W_qk (IllumSAB only)", "9. FiLM Scale (IllumSAB only)", "10. FiLM Shift (IllumSAB only)"):
            row2.append(make_labeled_tile(blank, label, "Not active on this checkpoint", (90, 95, 110)))

    v_tag = f"{info['mod_name']} ({mode_tag}{', BN' if info.get('use_bn') else ''}{', DarkShift' if info.get('darkness_weighted_shift') else ''})"
    return assemble_grid(
        row1 + row2,
        ncols=5,
        banner_title=f"Multinex 4-Channel Luminance Priors (S_L) & {v_tag} (160x160) - {model_title}",
        banner_sub=(
            f"Row 1: Raw Input & 4 Analytic Luminance Maps S_L ({mode_tag}, stride-4 anti-noise avg_pool) | "
            "Row 2: Darkness Mask -> CWA A_qk -> W_qk (Blue=suppress, Gold=boost) -> Retinex FiLM Scale & Dark-Weighted Shift"
        ),
        colormap_id=luma_cmap,
    )


def build_p2_illumsab_internals_figure(
    ref_panel: np.ndarray,
    rec_pre: Dict[str, Any],
    proj_pre: Dict[str, Any],
    rec_post: Dict[str, Any],
    proj_post: Dict[str, Any],
    info: Dict[str, Any],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    attn_colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
    sab_out_dir: Path,
) -> np.ndarray:
    """
    Create `02_p2_illumsab_internals_160x160.png` (2 rows x 5 cols) and export standalone assets in `illumsab/`:
      Row 1 (Pre-Attn Illumination Re-weighting & Channel Attention):
        1. Pre-IllumSAB Input (L02 C3k2)
        2. Raw Query (Q before W_qk)
        3. Modulated Query (Q' = Q * W_qk)
        4. Value (V) Projection
        5. 4-Head Transposed Channel Attention Matrices
      Row 2 (Pre-FiLM -> Post-FiLM -> Residual -> Shift -> Post-IllumSAB):
        6. Pre-FiLM Projected Output (project_out(A@V * ffn_gate))
        7. Post-FiLM Output (scale * out + shift)
        8. Residual |gamma * out| (|L03 - L02|)
        9. Signed Energy Shift L2(L03) - L2(L02)
       10. Post-IllumSAB Output (L03 IllumSAB)
    """
    sab_out_dir.mkdir(parents=True, exist_ok=True)
    ph, pw = ref_panel.shape[:2]
    acc_p2 = STAGE_ACCENT_BGR["p2"]
    acc_sab = STAGE_ACCENT_BGR["p2_sab"]
    acc_cwa = STAGE_ACCENT_BGR["cwa"]
    acc_film = STAGE_ACCENT_BGR["film"]
    acc_res = STAGE_ACCENT_BGR["sab"]
    mod_lbl = info["mod_name"]

    ov_pre, _, nat_pre = render_views_for_map(
        ref_panel, proj_pre["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )

    # Share Q scale between raw Q and modulated Q' so modulation effect is directly comparable
    q_vmax = max(
        float(np.percentile(info["q_raw_l2"], clip_pct)),
        float(np.percentile(info["q_mod_l2"], clip_pct)),
        1e-6,
    )
    q_vmin = min(float(info["q_raw_l2"].min()), float(info["q_mod_l2"].min()))
    ov_q_raw, _, _ = render_views_for_map(
        ref_panel, info["q_raw_l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(q_vmin, q_vmax)
    )
    ov_q_mod, _, _ = render_views_for_map(
        ref_panel, info["q_mod_l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(q_vmin, q_vmax)
    )
    ov_v, _, _ = render_views_for_map(
        ref_panel, info["v_l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )
    attn_panel = render_channel_attn_panel(
        info["attn_matrices"], info["temperatures"], info["head_entropies"],
        info["head_diag_mass"], (ph, pw), attn_colormap_id,
    )

    # Share scale between Pre-FiLM and Post-FiLM so FiLM boost/damping is directly visible
    p_vmax = max(
        float(np.percentile(info["proj_pre_gate_l2"], clip_pct)),
        float(np.percentile(info["proj_post_gate_l2"], clip_pct)),
        1e-6,
    )
    p_vmin = min(float(info["proj_pre_gate_l2"].min()), float(info["proj_post_gate_l2"].min()))
    ov_pre_film, _, _ = render_views_for_map(
        ref_panel, info["proj_pre_gate_l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(p_vmin, p_vmax)
    )
    ov_post_film, _, _ = render_views_for_map(
        ref_panel, info["proj_post_gate_l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(p_vmin, p_vmax)
    )
    ov_res, heat_res, _ = render_views_for_map(
        ref_panel, info["residual_l2"], lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct
    )
    signed_blend, signed_pure = render_signed_diverging_map(
        ref_panel, info["signed_l2_shift"], lb_size, crop_box, clip_percentile=99.0, blend_alpha=0.78
    )
    ov_post, _, nat_post = render_views_for_map(
        ref_panel, proj_post["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct
    )

    # Export standalone assets
    cv2.imwrite(str(sab_out_dir / "q_raw_overlay.png"), ov_q_raw)
    cv2.imwrite(str(sab_out_dir / "q_illum_mod_overlay.png"), ov_q_mod)
    cv2.imwrite(str(sab_out_dir / "v_overlay.png"), ov_v)
    cv2.imwrite(str(sab_out_dir / "pre_film_overlay.png"), ov_pre_film)
    cv2.imwrite(str(sab_out_dir / "post_film_overlay.png"), ov_post_film)
    cv2.imwrite(str(sab_out_dir / "residual_delta_overlay.png"), ov_res)
    cv2.imwrite(str(sab_out_dir / "residual_delta_heatmap.png"), heat_res)
    cv2.imwrite(str(sab_out_dir / "signed_energy_shift.png"), signed_pure)
    cv2.imwrite(str(sab_out_dir / "pre_native_160.png"), nat_pre)
    cv2.imwrite(str(sab_out_dir / "post_native_160.png"), nat_post)
    cv2.imwrite(
        str(sab_out_dir / "channel_attn_heads.png"),
        render_standalone_channel_attn_figure(info, attn_colormap_id, model_title),
    )

    mean_temp = float(np.mean(info["temperatures"]))
    mean_ent = float(np.mean(info["head_entropies"]))
    shift_mean = float(info["signed_l2_shift"].mean())
    boost_pct = float((info["signed_l2_shift"] > 0).mean() * 100.0)

    row1 = [
        make_labeled_tile(
            ov_pre,
            f"1. Pre-{mod_lbl} (L{rec_pre['layer_idx']:02d} C3k2)",
            f"160x160 | C={info['dim']} | L2={proj_pre['l2_mean']:.4f}",
            acc_p2,
        ),
        make_labeled_tile(
            ov_q_raw,
            "2. Raw Query Q (Before W_qk)",
            f"Shared Q scale | L2={float(info['q_raw_l2'].mean()):.4f}",
            acc_res,
        ),
        make_labeled_tile(
            ov_q_mod,
            "3. Modulated Q' = Q * W_qk" if info["is_illumsab"] else "3. Query Q' (Post-Weight)",
            f"Shared Q scale | L2={float(info['q_mod_l2'].mean()):.4f} | dQ={info['q_illum_shift_pct']:.1f}%",
            acc_cwa,
        ),
        make_labeled_tile(
            ov_v,
            "4. Value (V) Projection",
            f"1x1+DW3x3 | C_in={info['inner_dim']} | L2={float(info['v_l2'].mean()):.4f}",
            acc_res,
        ),
        make_labeled_tile(
            attn_panel,
            f"5. Channel Attn ({info['num_heads']}x {info['head_dim']}x{info['head_dim']})",
            f"mean temp={mean_temp:.2f} | mean entropy={mean_ent:.2f}",
            acc_sab,
        ),
    ]

    row2 = [
        make_labeled_tile(
            ov_pre_film,
            "6. Pre-FiLM Out (Attn+FFN+Proj)",
            f"Shared scale | L2={float(info['proj_pre_gate_l2'].mean()):.4f}",
            acc_sab,
        ),
        make_labeled_tile(
            ov_post_film,
            "7. Post-FiLM (s*Out + b)" if info["is_illumsab"] else "7. Post-Gate Output",
            f"Shared scale | L2={float(info['proj_post_gate_l2'].mean()):.4f} | dFiLM={info['film_delta_pct']:.1f}%",
            acc_film,
        ),
        make_labeled_tile(
            ov_res,
            f"8. {mod_lbl} Residual |L03 - L02|",
            f"gamma={info['gamma_mean']:.3f} | dL2={float(info['residual_l2'].mean()):.4f} ({info['rel_residual_pct']:.1f}%)",
            acc_res,
        ),
        make_labeled_tile(
            signed_blend,
            "9. Signed Shift L2(L03)-L2(L02)",
            f"Blue(-)/Gold(+) | mean={shift_mean:+.4f} | +:{boost_pct:.0f}%",
            (110, 175, 255),
        ),
        make_labeled_tile(
            ov_post,
            f"10. Post-{mod_lbl} (L{rec_post['layer_idx']:02d} Out)",
            f"160x160 | L2={proj_post['l2_mean']:.4f} | Max={proj_post['max_val']:.2f}",
            acc_sab,
        ),
    ]

    return assemble_grid(
        row1 + row2,
        ncols=5,
        banner_title=f"Backbone P2 {mod_lbl} (160x160) Internal Pipeline Diagnostics - {model_title}",
        banner_sub=(
            "Row 1: Pre-L02 -> Raw Q vs Illum-Modulated Q' (shared scale) -> V -> 4-Head Channel Attn | "
            "Row 2: Pre-FiLM vs Post-FiLM (shared scale) -> Residual -> Signed Shift -> Post-L03"
        ),
        colormap_id=colormap_id,
    )


def build_all_layers_progression_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    arch_tag: str,
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

    first_idx = next(iter(records.values()))["layer_idx"]
    last_idx = list(records.values())[-1]["layer_idx"]
    return assemble_grid(
        overlay_tiles + native_tiles,
        ncols=ncols,
        banner_title=f"Full Backbone ({arch_tag}) Layer Progression (L{first_idx:02d} -> L{last_idx:02d}) - {model_title}",
        banner_sub="Top block: Spatial L2 Overlays | Bottom block: Discrete Native-Grid Heatmaps (160 -> 80 -> 40 -> 20)",
        colormap_id=colormap_id,
    )


def build_dual_attention_stages_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    stage_slots: Dict[str, str],
    info: Optional[Dict[str, Any]],
    skip_p3_sab_mod: Optional[nn.Module],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
) -> Optional[np.ndarray]:
    rows: List[List[np.ndarray]] = []
    row_descs: List[str] = []

    pre_k, post_k = stage_slots.get("p2_pre", ""), stage_slots.get("p2", "")
    if pre_k in records and post_k in records and info is not None:
        rec_pre, proj_pre = records[pre_k], projections[pre_k]
        rec_post, proj_post = records[post_k], projections[post_k]
        ov_pre, _, _ = render_views_for_map(ref_panel, proj_pre["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        ov_post, _, nat_post = render_views_for_map(ref_panel, proj_post["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        ov_res, _, _ = render_views_for_map(ref_panel, info["residual_l2"], lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct)
        signed_blend, _ = render_signed_diverging_map(ref_panel, info["signed_l2_shift"], lb_size, crop_box, clip_percentile=99.0, blend_alpha=0.78)
        mod_lbl = info["mod_name"]
        rows.append([
            make_labeled_tile(ov_pre, f"P2 Pre-{mod_lbl} L{rec_pre['layer_idx']:02d} (160)", f"L2={proj_pre['l2_mean']:.4f} | Max={proj_pre['max_val']:.2f}", STAGE_ACCENT_BGR["p2"]),
            make_labeled_tile(ov_post, f"P2 Post-{mod_lbl} L{rec_post['layer_idx']:02d} (160)", f"L2={proj_post['l2_mean']:.4f} | gamma={info['gamma_mean']:.3f}", STAGE_ACCENT_BGR["p2_sab"]),
            make_labeled_tile(ov_res, f"P2-{mod_lbl} Residual |Post-Pre|", f"dL2={float(info['residual_l2'].mean()):.4f} ({info['rel_residual_pct']:.1f}%)", STAGE_ACCENT_BGR["sab"]),
            make_labeled_tile(signed_blend, "P2 Signed Shift (Post - Pre)", f"Blue(-)/Gold(+) | mean={float(info['signed_l2_shift'].mean()):+.4f}", (110, 175, 255)),
            make_labeled_tile(nat_post, f"P2-{mod_lbl} Native 160x160", f"Sparsity={proj_post['sparsity']:.1f}%", STAGE_ACCENT_BGR["p2_sab"]),
        ])
        row_descs.append(f"P2/4 {mod_lbl} (160x160)")

    s4_keys = [k for k, r in records.items() if r["stride"] == 32]
    s4_tiles: List[np.ndarray] = []
    for key in s4_keys[:4]:
        rec, proj = records[key], projections[key]
        c, h_f, w_f = proj["shape"]
        ov, _, _ = render_views_for_map(ref_panel, proj["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        s4_tiles.append(
            make_labeled_tile(
                ov,
                f"L{rec['layer_idx']:02d} {rec['mod_type']} ({h_f}x{w_f})",
                f"C={c} | L2={proj['l2_mean']:.3f} | Max={proj['max_val']:.2f}",
                STAGE_ACCENT_BGR["p5"],
            )
        )
    sppf_k, c2psa_k = stage_slots.get("s4_sppf", ""), stage_slots.get("p5", "")
    if sppf_k in records and c2psa_k in records:
        t_sppf, t_psa = records[sppf_k]["tensor"][0], records[c2psa_k]["tensor"][0]
        if t_sppf.shape == t_psa.shape:
            d20 = torch.sqrt(((t_psa - t_sppf) ** 2).mean(dim=0) + 1e-12).numpy()
            ov_d20, _, _ = render_views_for_map(ref_panel, d20, lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct)
            s4_tiles.append(
                make_labeled_tile(
                    ov_d20,
                    "C2PSA Delta |C2PSA - SPPF|",
                    f"20x20 | Mean Delta L2={float(d20.mean()):.4f}",
                    STAGE_ACCENT_BGR["sab"],
                )
            )
    if s4_tiles:
        rows.append(s4_tiles)
        row_descs.append("P5/32 C2PSA (20x20)")

    p3_k, p3_sab_k = stage_slots.get("p3", ""), stage_slots.get("p3_sab", "")
    if p3_k in records and p3_sab_k in records:
        rec_p3, proj_p3 = records[p3_k], projections[p3_k]
        rec_sab, proj_sab = records[p3_sab_k], projections[p3_sab_k]
        ov_p3, _, nat_p3 = render_views_for_map(ref_panel, proj_p3["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        ov_sab, _, nat_sab = render_views_for_map(ref_panel, proj_sab["l2"], lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        d80 = torch.sqrt(((rec_sab["tensor"][0] - rec_p3["tensor"][0]) ** 2).mean(dim=0) + 1e-12).numpy()
        ov_d80, _, _ = render_views_for_map(ref_panel, d80, lb_size, crop_box, colormap_id, alpha=0.58, clip_percentile=clip_pct)
        g_sab = float(skip_p3_sab_mod.gamma.detach().float().mean().item()) if (skip_p3_sab_mod is not None and hasattr(skip_p3_sab_mod, "gamma")) else 0.0
        rows.append([
            make_labeled_tile(ov_p3, f"Pre-SAB L{rec_p3['layer_idx']:02d} (80x80)", f"L2={proj_p3['l2_mean']:.4f}", STAGE_ACCENT_BGR["p3"]),
            make_labeled_tile(ov_sab, f"Post-SAB L{rec_sab['layer_idx']:02d} (80x80)", f"L2={proj_sab['l2_mean']:.4f} | gamma={g_sab:.3f}", STAGE_ACCENT_BGR["p3_sab"]),
            make_labeled_tile(ov_d80, "Skip-P3 SAB |Post - Pre|", f"80x80 | dL2={float(d80.mean()):.4f}", STAGE_ACCENT_BGR["sab"]),
            make_labeled_tile(nat_p3, "Pre-SAB Native 80x80", f"Sparsity={proj_p3['sparsity']:.1f}%", STAGE_ACCENT_BGR["p3"]),
            make_labeled_tile(nat_sab, "Post-SAB Native 80x80", f"Sparsity={proj_sab['sparsity']:.1f}%", STAGE_ACCENT_BGR["p3_sab"]),
        ])
        row_descs.append("Neck Skip-P3 SAB (80x80)")

    if not rows:
        return None

    ncols = max(len(r) for r in rows)
    flat_tiles: List[np.ndarray] = []
    for r in rows:
        flat_tiles.extend(r)
        if len(r) < ncols:
            flat_tiles.extend([np.full_like(r[0], (18, 20, 24))] * (ncols - len(r)))

    return assemble_grid(
        flat_tiles,
        ncols=ncols,
        banner_title=f"Backbone Attention Stages ({' & '.join(row_descs)}) - {model_title}",
        banner_sub="Compares early illumination-guided channel attention (P2 160x160) with deep global spatial attention (P5 20x20)",
        colormap_id=colormap_id,
    )


def render_gamma_profile_panel(
    info: Dict[str, Any],
    width: int = 1220,
    height: int = 264,
) -> np.ndarray:
    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    gamma_vec = info["gamma_vec"]
    n_ch = len(gamma_vec)
    mod_lbl = f"P2-{info['mod_name']}"

    cv2.rectangle(panel, (10, 6), (width - 10, height - 10), (24, 27, 33), -1)
    cv2.rectangle(panel, (10, 6), (width - 10, height - 10), (48, 52, 64), 1)
    cv2.putText(
        panel,
        f"Learned {mod_lbl} Per-Channel Residual Gate (gamma, C={n_ch}) & Head Diagnostics",
        (22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 242, 252), 1, cv2.LINE_AA,
    )

    legend_items = [
        ((80, 225, 255), ">=mean"),
        ((245, 125, 225), "0.10..mean"),
        ((220, 160, 80), "<0.10"),
        ((110, 235, 140), "-- init=0.10"),
    ]
    lx = 620
    for col, txt in legend_items:
        cv2.rectangle(panel, (lx, 18), (lx + 10, 27), col, -1)
        cv2.putText(panel, txt, (lx + 14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (190, 200, 215), 1, cv2.LINE_AA)
        lx += 78 if len(txt) <= 6 else 98

    x0, x1 = 68, width - 355
    y0, y1 = 44, height - 36
    chart_w = max(100, x1 - x0)
    chart_h = max(60, y1 - y0)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (18, 20, 25), -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (56, 62, 74), 1)

    g_min_plot = min(0.0, float(gamma_vec.min()) * 1.1)
    g_max_plot = max(0.15, float(gamma_vec.max()) * 1.15)
    g_span = max(g_max_plot - g_min_plot, 1e-6)

    def _val_to_y(val: float) -> int:
        return int(round(y1 - np.clip((val - g_min_plot) / g_span, 0.0, 1.0) * chart_h))

    y_zero = _val_to_y(0.0)
    y_init = _val_to_y(0.10)

    for tick_v in (g_min_plot, 0.10, (g_max_plot + 0.10) * 0.5, g_max_plot):
        yt = _val_to_y(tick_v)
        cv2.line(panel, (x0, yt), (x1, yt), (38, 42, 50), 1)
        cv2.putText(panel, f"{tick_v:.2f}", (18, yt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (165, 175, 192), 1, cv2.LINE_AA)

    for ci, gv in enumerate(gamma_vec):
        bx1 = x0 + int(math.floor(ci * chart_w / n_ch))
        bx2 = max(bx1, min(x1 - 1, x0 + int(math.ceil((ci + 1) * chart_w / n_ch)) - 1))
        yb = _val_to_y(float(gv))
        top_y, bot_y = min(yb, y_zero), max(yb, y_zero)
        if gv >= 0.10:
            col = (245, 125, 225) if gv < info["gamma_mean"] else (80, 225, 255)
        elif gv >= 0.0:
            col = (220, 160, 80)
        else:
            col = (95, 110, 240)
        cv2.rectangle(panel, (bx1, top_y), (bx2, bot_y), col, -1)

    for dx in range(x0, x1, 10):
        cv2.line(panel, (dx, y_init), (min(x1, dx + 5), y_init), (110, 235, 140), 1)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ci = min(n_ch - 1, int(round(frac * (n_ch - 1))))
        tx = x0 + int(round(frac * (chart_w - 26)))
        cv2.putText(panel, f"ch{ci}", (tx, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (165, 175, 192), 1, cv2.LINE_AA)

    rx0, rx1 = width - 338, width - 22
    cv2.rectangle(panel, (rx0, y0), (rx1, y1), (18, 20, 25), -1)
    cv2.rectangle(panel, (rx0, y0), (rx1, y1), (56, 62, 74), 1)

    above_init_pct = float((gamma_vec > 0.10).mean() * 100.0)
    lines = [
        (f"gamma mean : {info['gamma_mean']:.4f} +/- {info['gamma_std']:.4f}", (240, 244, 252)),
        (f"gamma range: [{info['gamma_min']:.4f}, {info['gamma_max']:.4f}]", (195, 208, 226)),
        (f"channels > 0.10 : {above_init_pct:.1f}% ({int((gamma_vec > 0.10).sum())}/{n_ch})", (120, 238, 150)),
        (f"residual ||d||/||x|| : {info['rel_residual_pct']:.2f}%", (60, 220, 255)),
        ("Head   Temp    Diag    Entropy", (165, 178, 198)),
    ]
    for hi in range(info["num_heads"]):
        lines.append((
            f" H{hi}    {info['temperatures'][hi]:.3f}   {info['head_diag_mass'][hi]:.3f}   {info['head_entropies'][hi]:.3f}",
            (235, 195, 245),
        ))

    line_step = max(14, min(18, (y1 - y0 - 12) // max(1, len(lines))))
    ty = y0 + 18
    for text, color in lines:
        if ty <= y1 - 4:
            cv2.putText(panel, text, (rx0 + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            ty += line_step

    return panel


def render_illum_scale_and_regime_panel(
    info: Dict[str, Any],
    width: int = 1220,
    height: int = 276,
) -> Optional[np.ndarray]:
    """
    PA1-specific quantitative panel:
      - Left: Learned `0.9 * tanh(illum_scale)` per-channel bar chart (C_inner channels)
      - Right: Dark (<0.15) / Mid (0.15-0.50) / Bright (>0.50) active-pixel regime table
               comparing W_qk, FiLM scale, ||FiLM shift||, Pre-L2, Post-L2, and Signed ΔL2.
    """
    scale_vec = info.get("illum_scale_eff")
    regimes = info.get("regime_stats", [])
    if scale_vec is None and not regimes:
        return None

    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    cv2.rectangle(panel, (10, 6), (width - 10, height - 10), (24, 27, 33), -1)
    cv2.rectangle(panel, (10, 6), (width - 10, height - 10), (48, 52, 64), 1)
    cv2.putText(
        panel,
        "PA1 IllumSAB Diagnostics: Learned Q/K Modulation (0.9*tanh(beta)) & Dark/Mid/Bright Regime Behavior",
        (22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 242, 252), 1, cv2.LINE_AA,
    )

    y0, y1 = 44, height - 34
    # Left half: per-channel effective illum_scale bar chart
    x0, x1 = 64, 575
    cv2.rectangle(panel, (x0, y0), (x1, y1), (18, 20, 25), -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (56, 62, 74), 1)

    if scale_vec is not None and len(scale_vec) > 0:
        n_c = len(scale_vec)
        chart_w = x1 - x0
        chart_h = y1 - y0
        s_min, s_max = -0.92, 0.92
        init_eff = float(info.get("illum_scale_init_eff", 0.9 * math.tanh(0.3)))

        def _sy(val: float) -> int:
            return int(round(y1 - np.clip((val - s_min) / (s_max - s_min), 0.0, 1.0) * chart_h))

        y_zero = _sy(0.0)
        y_init = _sy(init_eff)
        for tv in (-0.9, -0.45, 0.0, 0.45, 0.9):
            yt = _sy(tv)
            cv2.line(panel, (x0, yt), (x1, yt), (38, 42, 50) if abs(tv) > 1e-4 else (82, 88, 102), 1)
            cv2.putText(panel, f"{tv:+.2f}", (14, yt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (165, 175, 192), 1, cv2.LINE_AA)

        for ci, sv in enumerate(scale_vec):
            bx1 = x0 + int(math.floor(ci * chart_w / n_c))
            bx2 = max(bx1, min(x1 - 1, x0 + int(math.ceil((ci + 1) * chart_w / n_c)) - 1))
            yb = _sy(float(sv))
            col = (70, 225, 255) if sv >= 0.0 else (240, 140, 80)
            cv2.rectangle(panel, (bx1, min(yb, y_zero)), (bx2, max(yb, y_zero)), col, -1)

        for dx in range(x0, x1, 10):
            cv2.line(panel, (dx, y_init), (min(x1, dx + 5), y_init), (110, 235, 140), 1)

        pos_pct = float((scale_vec > 0).mean() * 100.0)
        cv2.putText(
            panel,
            f"C_in={n_c} | mean={scale_vec.mean():+.3f} | std={scale_vec.std():.3f} | >0:{pos_pct:.0f}% | --init={init_eff:+.2f}",
            (x0 + 4, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (175, 192, 212), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(panel, "illum_scale N/A (not IllumSAB)", (x0 + 110, (y0 + y1) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 140, 155), 1, cv2.LINE_AA)

    # Right half: Dark / Mid / Bright regime table
    rx0, rx1 = 595, width - 20
    cv2.rectangle(panel, (rx0, y0), (rx1, y1), (18, 20, 25), -1)
    cv2.rectangle(panel, (rx0, y0), (rx1, y1), (56, 62, 74), 1)

    headers = [
        ("Regime (Y_709)", rx0 + 10),
        ("Area%", rx0 + 132),
        ("W_qk", rx0 + 190),
        ("FiLM s", rx0 + 252),
        ("||b||", rx0 + 318),
        ("PreL2", rx0 + 382),
        ("PostL2", rx0 + 444),
        ("dL2", rx0 + 510),
        ("+%", rx0 + 568),
    ]
    cv2.rectangle(panel, (rx0 + 1, y0 + 1), (rx1 - 1, y0 + 26), (30, 34, 42), -1)
    for h_txt, hx in headers:
        cv2.putText(panel, h_txt, (hx, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 205, 225), 1, cv2.LINE_AA)

    def _fmt(v: float, fmt_s: str) -> str:
        return "  -  " if (v is None or math.isnan(v)) else format(v, fmt_s)

    row_colors = [
        (140, 195, 255),  # Dark
        (215, 222, 232),  # Mid
        (95, 225, 255),   # Bright
        (135, 235, 165),  # All Active
    ]
    row_step = 38
    for idx, reg in enumerate(regimes[:4]):
        ry_top = y0 + 30 + idx * row_step
        ry_txt = ry_top + 22
        if idx % 2 == 0:
            cv2.rectangle(panel, (rx0 + 2, ry_top), (rx1 - 2, ry_top + row_step - 4), (24, 27, 34), -1)
        rc = row_colors[idx % len(row_colors)]
        cv2.putText(panel, reg["name"], (rx0 + 10, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, rc, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{reg['pct']:4.1f}%", (rx0 + 132, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 218, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["w_qk"], ".3f"), (rx0 + 190, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 240, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["film_scale"], ".3f"), (rx0 + 252, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 185, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["film_shift"], ".3f"), (rx0 + 318, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 195, 240), 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["pre_l2"], ".3f"), (rx0 + 382, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (185, 195, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["post_l2"], ".3f"), (rx0 + 444, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 236, 245), 1, cv2.LINE_AA)
        dl2 = reg["delta_l2"]
        d_col = (90, 235, 145) if (not math.isnan(dl2) and dl2 >= 0) else (105, 130, 250)
        cv2.putText(panel, _fmt(dl2, "+.3f"), (rx0 + 505, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, d_col, 1, cv2.LINE_AA)
        cv2.putText(panel, _fmt(reg["boost_pct"], ".0f") + ("%" if not math.isnan(reg["boost_pct"]) else ""), (rx0 + 568, ry_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 226, 238), 1, cv2.LINE_AA)

    cv2.putText(
        panel,
        "Active non-letterbox pixels only | dL2 = L2(Post-IllumSAB) - L2(Pre-IllumSAB) | +% = pixels boosted",
        (rx0 + 4, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (165, 178, 196), 1, cv2.LINE_AA,
    )
    return panel


def build_stats_chart(
    records: "OrderedDict[str, Dict[str, Any]]",
    projections: Dict[str, Dict[str, Any]],
    info: Optional[Dict[str, Any]],
    skip_p3_sab_mod: Optional[nn.Module],
    arch_tag: str,
    model_title: str,
    sab_out_dir: Optional[Path] = None,
) -> np.ndarray:
    rows_data: List[Tuple[str, str, float, float, float, Tuple[int, int, int]]] = []
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
    bot_h = 20
    width = 1220
    table_h = top_h + n * row_h + bot_h
    table = np.full((table_h, width, 3), (18, 20, 24), dtype=np.uint8)

    cv2.rectangle(table, (0, 0), (width - 1, 62), (28, 30, 36), -1)
    cv2.line(table, (0, 62), (width - 1, 62), (80, 155, 255), 2)
    title_txt = f"YOLOv11 Backbone ({arch_tag}) Quantitative Diagnostics - {model_title}"
    t_scale = min(0.58, max(0.40, (width - 40) / max(1, len(title_txt) * 13.0)))
    cv2.putText(table, title_txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, t_scale, (245, 248, 255), 2, cv2.LINE_AA)

    param_parts: List[str] = []
    if info is not None:
        t_str = ",".join(f"{t:.2f}" for t in info["temperatures"])
        extra = ""
        if info["is_illumsab"] and info["illum_scale_eff"] is not None:
            extra = f", dQ={info['q_illum_shift_pct']:.1f}%, dFiLM={info['film_delta_pct']:.1f}%"
        elif info["edge_scale"] is not None:
            extra = f", edge_scale={info['edge_scale']:.2f}"
        param_parts.append(
            f"P2-{info['mod_name']}(gamma={info['gamma_mean']:.3f} [{info['gamma_min']:.3f}..{info['gamma_max']:.3f}], "
            f"temp=[{t_str}], rel_delta={info['rel_residual_pct']:.1f}%{extra})"
        )
    if skip_p3_sab_mod is not None and hasattr(skip_p3_sab_mod, "gamma"):
        g_sp3 = float(skip_p3_sab_mod.gamma.detach().float().mean().item())
        param_parts.append(f"SkipP3-{type(skip_p3_sab_mod).__name__}(gamma={g_sp3:.3f})")
    if param_parts:
        sub_txt = " | ".join(param_parts)
        s_scale = min(0.38, max(0.28, (width - 40) / max(1, len(sub_txt) * 8.4)))
        cv2.putText(table, sub_txt, (16, 51), cv2.FONT_HERSHEY_SIMPLEX, s_scale, (175, 210, 240), 1, cv2.LINE_AA)

    cv2.putText(table, "Layer / Module", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(table, "Shape (C,H,W)", (295, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(table, "L2 Energy (RMS)", (450, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(table, "Sparsity (|x|<0.01)", (850, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)
    cv2.putText(table, "Max", (1115, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 205), 1, cv2.LINE_AA)

    max_l2 = max(max((r[2] for r in rows_data), default=1.0), 1e-6)
    for r_i, (name_txt, shape_txt, l2_val, sp_val, max_val, accent) in enumerate(rows_data):
        y_top = top_h + r_i * row_h
        y_mid = y_top + 24
        if r_i % 2 == 0:
            cv2.rectangle(table, (10, y_top), (width - 10, y_top + row_h - 2), (24, 27, 33), -1)
        cv2.rectangle(table, (10, y_top), (15, y_top + row_h - 2), accent, -1)

        cv2.putText(table, name_txt, (24, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 240, 248), 1, cv2.LINE_AA)
        cv2.putText(table, shape_txt, (295, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (180, 190, 208), 1, cv2.LINE_AA)

        bar_x0, bar_max_w = 450, 300
        l2_w = int(round((l2_val / max_l2) * bar_max_w))
        cv2.rectangle(table, (bar_x0, y_top + 9), (bar_x0 + bar_max_w, y_top + 27), (40, 44, 52), -1)
        if l2_w > 0:
            cv2.rectangle(table, (bar_x0, y_top + 9), (bar_x0 + l2_w, y_top + 27), accent, -1)
        cv2.putText(table, f"{l2_val:.4f}", (bar_x0 + bar_max_w + 10, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (230, 235, 245), 1, cv2.LINE_AA)

        sp_x0, sp_max_w = 850, 165
        sp_w = int(round((sp_val / 100.0) * sp_max_w))
        cv2.rectangle(table, (sp_x0, y_top + 9), (sp_x0 + sp_max_w, y_top + 27), (40, 44, 52), -1)
        if sp_w > 0:
            cv2.rectangle(table, (sp_x0, y_top + 9), (sp_x0 + sp_w, y_top + 27), (110, 195, 130), -1)
        cv2.putText(table, f"{sp_val:.1f}%", (sp_x0 + sp_max_w + 10, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (205, 215, 228), 1, cv2.LINE_AA)
        cv2.putText(table, f"{max_val:.2f}", (1115, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (235, 215, 150), 1, cv2.LINE_AA)

    blocks = [table]
    if info is not None:
        gamma_panel = render_gamma_profile_panel(info, width=width, height=264)
        blocks.append(gamma_panel)
        if sab_out_dir is not None:
            sab_out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sab_out_dir / "gamma_profile.png"), gamma_panel)

        regime_panel = render_illum_scale_and_regime_panel(info, width=width, height=276)
        if regime_panel is not None:
            blocks.append(regime_panel)
            if sab_out_dir is not None:
                cv2.imwrite(str(sab_out_dir / "illum_scale_and_regimes.png"), regime_panel)

    return np.vstack(blocks)


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
        banner_title=f"{rec['key']} - Resolution {h_f}x{w_f} (Stride {rec['stride']}, Channels {c})",
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
    common_slots = [s for s in ("p2", "p3", "p4", "p5") if s in slots_a and s in slots_b]
    row_a, row_b, row_abs, row_signed = [], [], [], []

    for slot in common_slots:
        ka, kb = slots_a[slot], slots_b[slot]
        ra, rb = records_a[ka], records_b[kb]
        pa, pb = proj_a[ka], proj_b[kb]
        _, h_f, w_f = pa["shape"]
        accent = STAGE_ACCENT_BGR.get(ra["accent_key"], (80, 160, 255))

        vmax = max(float(np.percentile(pa["l2"], clip_pct)), float(np.percentile(pb["l2"], clip_pct)), 1e-6)
        vmin = min(float(pa["l2"].min()), float(pb["l2"].min()))

        ov_a, _, _ = render_views_for_map(ref_panel, pa["l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(vmin, vmax))
        ov_b, _, _ = render_views_for_map(ref_panel, pb["l2"], lb_size, crop_box, colormap_id, alpha=alpha, fixed_range=(vmin, vmax))
        signed_2d = pb["l2"] - pa["l2"]
        diff_2d = np.abs(signed_2d)
        _, _, diff_nat = render_views_for_map(ref_panel, diff_2d, lb_size, crop_box, colormap_id, alpha=alpha, clip_percentile=clip_pct)
        signed_blend, _ = render_signed_diverging_map(ref_panel, signed_2d, lb_size, crop_box, clip_percentile=99.0, blend_alpha=0.78)

        row_a.append(make_labeled_tile(ov_a, f"[A] {slot.upper()} L{ra['layer_idx']:02d} {ra['mod_type']} ({h_f}x{w_f})", f"{label_a} | L2={pa['l2_mean']:.4f}", accent))
        row_b.append(make_labeled_tile(ov_b, f"[B] {slot.upper()} L{rb['layer_idx']:02d} {rb['mod_type']} ({h_f}x{w_f})", f"{label_b} | L2={pb['l2_mean']:.4f}", (90, 230, 130)))
        row_abs.append(make_labeled_tile(diff_nat, f"|B - A| {slot.upper()} ({h_f}x{w_f})", f"Mean |dL2|={float(diff_2d.mean()):.4f}", STAGE_ACCENT_BGR["sab"]))
        row_signed.append(make_labeled_tile(signed_blend, f"Signed (B - A) {slot.upper()}", f"Blue(A>B)/Gold(B>A) | mean={float(signed_2d.mean()):+.4f}", (110, 175, 255)))

    return assemble_grid(
        row_a + row_b + row_abs + row_signed,
        ncols=max(1, len(common_slots)),
        banner_title=f"Semantic Backbone Comparison (160 -> 80 -> 40 -> 20): [A] {label_a} vs [B] {label_b}",
        banner_sub="Row 1: Model A | Row 2: Model B (shared scale) | Row 3: |B - A| Native Grid | Row 4: Signed L2(B) - L2(A)",
        colormap_id=colormap_id,
    )


# ==================================================================================================
# 5. Main Pipeline
# ==================================================================================================

def run_inference_and_record(
    model: nn.Module,
    recorder: YOLOv11BackboneIllumSABRecorder,
    inp_tensor: torch.Tensor,
    orig_hw: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
    crop_box: Tuple[int, int, int, int],
    lb_size: int,
    conf_thres: float,
    iou_thres: float,
    fallback_illum_extractor: MultiPriorIllumExtractor,
) -> Tuple[
    "OrderedDict[str, Dict[str, Any]]",
    Dict[str, Dict[str, Any]],
    Optional[Dict[str, Any]],
    np.ndarray,
]:
    recorder.clear()
    with torch.no_grad():
        out = model(inp_tensor)
        pred_raw = out[0] if isinstance(out, tuple) else out

    p2_info: Optional[Dict[str, Any]] = None
    if recorder.p2_attn_mod is not None and recorder.p2_attn_input is not None:
        il_mod = recorder.illum_extractor_mod or fallback_illum_extractor
        illum_tensor = il_mod(inp_tensor)
        linear_illum_tensor = il_mod.extract_linear(inp_tensor)
        log_mu = float(getattr(il_mod, "log_mu", 0.0))
        edge_tensor = (
            recorder.edge_extractor_mod(inp_tensor)
            if (isinstance(recorder.p2_attn_mod, EdgeSAB) and recorder.edge_extractor_mod is not None)
            else None
        )
        p2_info = inspect_p2_attn_internals(
            recorder.p2_attn_mod,
            recorder.p2_attn_input,
            illum_tensor=illum_tensor,
            linear_illum_tensor=linear_illum_tensor,
            log_mu=log_mu,
            edge_tensor=edge_tensor,
            lb_size=lb_size,
            crop_box=crop_box,
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
    return records_copy, projections, p2_info, dets_np


def _print_regime_summary(stem: str, info: Dict[str, Any]) -> None:
    regimes = info.get("regime_stats", [])
    if not regimes:
        return
    print(f"    [PA1 Regime Analysis - {stem}] gamma={info['gamma_mean']:.4f} | rel_delta={info['rel_residual_pct']:.1f}% | dQ={info['q_illum_shift_pct']:.1f}% | dFiLM={info['film_delta_pct']:.1f}%")
    print("      Regime           Area%    W_qk   FiLM_s   ||b||   PreL2   PostL2     dL2   Boost%")
    for reg in regimes:
        if reg["count"] == 0:
            continue
        w_s = f"{reg['w_qk']:.3f}" if not math.isnan(reg["w_qk"]) else "  -  "
        fs_s = f"{reg['film_scale']:.3f}" if not math.isnan(reg["film_scale"]) else "  -  "
        sh_s = f"{reg['film_shift']:.3f}" if not math.isnan(reg["film_shift"]) else "  -  "
        print(
            f"      {reg['name']:<15} {reg['pct']:5.1f}%  {w_s:>6}  {fs_s:>7}  {sh_s:>6}  "
            f"{reg['pre_l2']:6.3f}  {reg['post_l2']:7.3f}  {reg['delta_l2']:+6.3f}   {reg['boost_pct']:5.1f}%"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    colormap_id = COLORMAP_TABLE[args.colormap]
    attn_colormap_id = COLORMAP_TABLE[args.attn_colormap]

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
    recorder_a = YOLOv11BackboneIllumSABRecorder(model_a, include_stem=args.include_stem)
    fallback_illum_ext = MultiPriorIllumExtractor(in_channels=3, downsample_stride=4).to(device).eval()

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
        recorder_b = YOLOv11BackboneIllumSABRecorder(model_b, include_stem=args.include_stem)

    image_paths = collect_input_images(args.source, args.num, args.shuffle)
    run_tag = (
        f"{model_name_a}_{meta_a['weights_label']}"
        if not args.compare_weights.strip()
        else f"compare_{meta_a['weights_label']}_vs_{meta_b['weights_label']}"
    )
    out_root = Path(args.output) / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"===> Visualizing YOLOv11 Backbone ({recorder_a.arch_display_tag}, "
        f"{'320->20' if args.include_stem else '160->20'}) on {len(image_paths)} image(s) -> '{out_root}'"
    )

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

            records_a, proj_a, info_a, dets_a = run_inference_and_record(
                model_a, recorder_a, inp_tensor, orig_hw, ratio_pad, crop_box,
                args.imgsz, args.conf_thres, args.iou_thres, fallback_illum_ext,
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

            # 0. Hero Backbone Pyramid (Pre-IllumSAB 160 -> Post-IllumSAB 160 -> 80 -> 40 -> 20)
            fig0 = build_pyramid_hero_figure(
                raw_panel, ref_panel, det_panel, records_a, proj_a, recorder_a.stage_slots,
                recorder_a.arch_display_tag, args.imgsz, crop_box, colormap_id,
                args.alpha, args.clip_percentile, model_title, Path(img_path).name,
            )
            cv2.imwrite(str(img_out_dir / "00_backbone_pyramid_160_to_20.png"), fig0)

            # 1. Multinex 4-Channel Analytic Luminance Stack S_L + CWA / W_qk / Retinex FiLM
            illum_dir = img_out_dir / "illum"
            sab_dir = img_out_dir / "illumsab"
            if info_a is not None:
                fig1 = build_multinex_priors_and_cwa_figure(
                    raw_panel, ref_panel, info_a, args.imgsz, crop_box,
                    attn_colormap_id, args.clip_percentile, model_title, Path(img_path).name, illum_dir,
                )
                if fig1 is not None:
                    cv2.imwrite(str(img_out_dir / "01_multinex_illumination_priors_and_cwa_160x160.png"), fig1)

            # 2. Dedicated P2 IllumSAB Internal Pipeline Diagnostics + standalone illumsab/ exports
            pre_k = recorder_a.stage_slots.get("p2_pre", "")
            post_k = recorder_a.stage_slots.get("p2", "")
            if info_a is not None and pre_k in records_a and post_k in records_a:
                fig2 = build_p2_illumsab_internals_figure(
                    ref_panel, records_a[pre_k], proj_a[pre_k], records_a[post_k], proj_a[post_k],
                    info_a, args.imgsz, crop_box, colormap_id, attn_colormap_id,
                    args.alpha, args.clip_percentile, model_title, sab_dir,
                )
                cv2.imwrite(str(img_out_dir / "02_p2_illumsab_internals_160x160.png"), fig2)

            # 3. Full Chronological Backbone Progression (L01..L11)
            fig3 = build_all_layers_progression_figure(
                ref_panel, records_a, proj_a, recorder_a.arch_display_tag,
                args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            cv2.imwrite(str(img_out_dir / "03_all_backbone_layers_160_to_20.png"), fig3)

            # 4. Dual Attention Stages: P2 IllumSAB (160x160) & P5 C2PSA (20x20)
            fig4 = build_dual_attention_stages_figure(
                ref_panel, records_a, proj_a, recorder_a.stage_slots,
                info_a, recorder_a.skip_p3_sab_mod,
                args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            if fig4 is not None:
                fig4_name = (
                    "04_attention_stages_p2_and_p5.png"
                    if info_a is not None
                    else "04_stage4_c2psa_20x20.png"
                )
                cv2.imwrite(str(img_out_dir / fig4_name), fig4)

            # 5. Quantitative Activation Stats + Gamma Profile + illum_scale + Dark/Mid/Bright Regimes
            fig5 = build_stats_chart(
                records_a, proj_a, info_a, recorder_a.skip_p3_sab_mod,
                recorder_a.arch_display_tag, model_title,
                sab_out_dir=sab_dir if info_a is not None else None,
            )
            cv2.imwrite(str(img_out_dir / "05_backbone_activation_stats.png"), fig5)

            # 6. Per-Layer Standalone Maps & Top-K Channel Sheets
            layers_root = img_out_dir / "layers"
            for key, rec in records_a.items():
                _, h_f, w_f = proj_a[key]["shape"]
                save_per_layer_assets(
                    layers_root / f"{key}_{h_f}x{w_f}", ref_panel, rec, proj_a[key],
                    args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile, args.channels,
                )

            # 7. Optional Side-by-Side Semantic Comparison
            if model_b is not None and recorder_b is not None:
                records_b, proj_b, _, _ = run_inference_and_record(
                    model_b, recorder_b, inp_tensor, orig_hw, ratio_pad, crop_box,
                    args.imgsz, args.conf_thres, args.iou_thres, fallback_illum_ext,
                )
                fig6 = build_semantic_comparison_figure(
                    ref_panel,
                    records_a, proj_a, recorder_a.stage_slots, meta_a["weights_label"],
                    records_b, proj_b, recorder_b.stage_slots, meta_b["weights_label"],
                    args.imgsz, crop_box, colormap_id, args.alpha, args.clip_percentile,
                )
                cv2.imwrite(str(img_out_dir / "06_backbone_comparison_160_to_20.png"), fig6)

            scales_str = " -> ".join(
                f"L{records_a[recorder_a.stage_slots[s]]['layer_idx']:02d}"
                f"({proj_a[recorder_a.stage_slots[s]]['shape'][1]}x{proj_a[recorder_a.stage_slots[s]]['shape'][2]})"
                for s in ("p2_pre", "p2", "p3", "p4", "p5")
                if s in recorder_a.stage_slots and recorder_a.stage_slots[s] in proj_a
            )
            print(f"  [{img_idx}/{len(image_paths)}] Saved {stem}: {scales_str} -> {img_out_dir}")
            if info_a is not None:
                _print_regime_summary(stem, info_a)
    finally:
        recorder_a.close()
        if recorder_b is not None:
            recorder_b.close()

    print(f"===> Done! Visualizations saved to: {out_root}")


if __name__ == "__main__":
    main()
