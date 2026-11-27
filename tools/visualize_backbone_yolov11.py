"""
tools/visualize_backbone_yolov11.py

Visualize multi-scale YOLOv11 Backbone feature maps across spatial resolutions
from 160x160 down to 20x20 (for 640x640 input), supporting both:
  - 25-layer P2-IllumSAB / P2-SAB / P2-EdgeSAB Backbone (`develop-yolov11-p2-illumsab`, L01..L11):
      - Stage 1 (P2/4,  160x160): L01_Conv3x3_P2_4  -> L02_C3k2_P2_4 -> L03_IllumSAB_P2_4
      - Stage 2 (P3/8,   80x80 ): L04_Conv3x3_P3_8  -> L05_C3k2_P3_8 (Skip-P3)
      - Stage 3 (P4/16,  40x40 ): L06_Conv3x3_P4_16 -> L07_C3k2_P4_16 (Skip-P4)
      - Stage 4 (P5/32,  20x20 ): L08_Conv3x3_P5_32 -> L09_C3k2_P5_32 -> L10_SPPF_P5_32 -> L11_C2PSA_P5_32 (Skip-P5)
  - 24-layer Standard / Skip-P3 Backbone (`main`, L01..L10)
  - Optional Stem (P1/2, 320x320): L00_Stem_Conv3x3_P1_2 via `--include_stem`

For deep P2-IllumSAB internal diagnostics (Multinex 4-channel S_L priors, CWA A_qk/W_qk,
Retinex FiLM scale/shift, 4-head channel attention, Dark/Mid/Bright regime table),
use `tools/visualize_backbone_illumsab_yolov11.py`.

Outputs saved under `<output>/<run_name>/<image_stem>/`:
  - `00_backbone_pyramid_160_to_20.png`     - Key stage outputs (160 -> 80 -> 40 -> 20): Overlay, Native Grid, PCA-RGB
  - `01_all_backbone_layers_160_to_20.png`  - Layer-by-layer progression (L01 -> L10)
  - `02_stage4_20x20_refinement.png`        - Deep 20x20 refinement: Conv -> C3k2 -> SPPF -> C2PSA & Attention Delta
  - `03_backbone_activation_stats.png`      - Quantitative L2 energy, max, mean & sparsity chart across layers
  - `04_backbone_comparison_160_to_20.png`  - Side-by-side comparison & |Delta| (when `--compare_weights` is set)
  - `layers/<layer_name>/`                  - Standalone borderless maps (`l2_overlay.png`, `l2_heatmap.png`,
                                              `native_grid.png`, `pca_rgb.png`) + `channel_sheet.png` (Top-K channels)

Usage Examples:
  # 1. Visualize trained checkpoint on an ExDark image
  python tools/visualize_backbone_yolov11.py --weights weights/training/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65

  # 2. Visualize official pretrained YOLOv11s backbone on 3 test images
  python tools/visualize_backbone_yolov11.py --weights auto --model yolov11s --num 3 --vis_gamma 0.65

  # 3. Compare two checkpoints side-by-side across 160 -> 80 -> 40 -> 20
  python tools/visualize_backbone_yolov11.py ^
      --weights model_results/yolov11s/best_mAP50.pth ^
      --compare_weights model_results/yolov11s-skip-p3-edge-sab/best_mAP50.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.65
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

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
    build_yolo_model,
    get_model_family,
    infer_model_from_checkpoint,
    load_yolo_checkpoint,
    non_max_suppression,
)
from net.Yolov11.common import EdgeSAB, IllumSAB, SAB  # noqa: E402
from net.Yolov11.utils import (  # noqa: E402
    _register_ultralytics_unpickle_aliases,
    de_parallel,
    scale_boxes,
)
from net.Yolov11.yolo import DetectionModel  # noqa: E402
from tools.visualize_backbone_illumsab_yolov11 import (  # noqa: E402
    _build_arch_cfg_dict,
    _extract_raw_sd,
    _infer_nc_from_sd,
    _infer_variant_from_sd,
    detect_ckpt_arch,
    detect_is_modular_ckpt as _detect_is_modular_sd,
)


# ==================================================================================================
# 1. YOLOv11 Backbone Metadata (Base L00..L10 & P2-IllumSAB/SAB/EdgeSAB L00..L11)
# ==================================================================================================

# (stage_group, canonical_label, stride, is_stage_output, semantic_slot)
YOLOV11_BACKBONE_BASE_META: Dict[int, Tuple[str, str, int, bool, str]] = {
    0: ("P1/2 (Stem)", "L00_Stem_Conv3x3_P1_2", 2, True, "p1"),
    1: ("P2/4 (Stage1)", "L01_Conv3x3_P2_4", 4, False, ""),
    2: ("P2/4 (Stage1)", "L02_C3k2_P2_4", 4, True, "p2"),
    3: ("P3/8 (Stage2)", "L03_Conv3x3_P3_8", 8, False, ""),
    4: ("P3/8 (Stage2)", "L04_C3k2_P3_8_SkipP3", 8, True, "p3"),
    5: ("P4/16 (Stage3)", "L05_Conv3x3_P4_16", 16, False, ""),
    6: ("P4/16 (Stage3)", "L06_C3k2_P4_16_SkipP4", 16, True, "p4"),
    7: ("P5/32 (Stage4)", "L07_Conv3x3_P5_32", 32, False, ""),
    8: ("P5/32 (Stage4)", "L08_C3k2_P5_32", 32, False, "s4_c3k2"),
    9: ("P5/32 (Stage4)", "L09_SPPF_P5_32", 32, False, "s4_sppf"),
    10: ("P5/32 (Stage4)", "L10_C2PSA_P5_32_SkipP5", 32, True, "p5"),
}

YOLOV11_BACKBONE_P2_ATTN_META: Dict[int, Tuple[str, str, int, bool, str]] = {
    0: ("P1/2 (Stem)", "L00_Stem_Conv3x3_P1_2", 2, True, "p1"),
    1: ("P2/4 (Stage1)", "L01_Conv3x3_P2_4", 4, False, ""),
    2: ("P2/4 (Stage1)", "L02_C3k2_P2_4_PreAttn", 4, True, "p2_pre"),
    3: ("P2/4 (P2-Attn)", "L03_P2Attn_P2_4_160", 4, True, "p2"),
    4: ("P3/8 (Stage2)", "L04_Conv3x3_P3_8", 8, False, ""),
    5: ("P3/8 (Stage2)", "L05_C3k2_P3_8_SkipP3", 8, True, "p3"),
    6: ("P4/16 (Stage3)", "L06_Conv3x3_P4_16", 16, False, ""),
    7: ("P4/16 (Stage3)", "L07_C3k2_P4_16_SkipP4", 16, True, "p4"),
    8: ("P5/32 (Stage4)", "L08_Conv3x3_P5_32", 32, False, ""),
    9: ("P5/32 (Stage4)", "L09_C3k2_P5_32", 32, False, "s4_c3k2"),
    10: ("P5/32 (Stage4)", "L10_SPPF_P5_32", 32, False, "s4_sppf"),
    11: ("P5/32 (Stage4)", "L11_C2PSA_P5_32_SkipP5", 32, True, "p5"),
}

KEY_PYRAMID_SLOTS: Tuple[str, ...] = ("p2", "p3", "p4", "p5")

STAGE_ACCENT_BGR: Dict[int, Tuple[int, int, int]] = {
    2: (200, 200, 90),    # P1/2  (320) - Cyan-Teal
    4: (255, 195, 55),    # P2/4  (160) - Sky Blue
    8: (95, 225, 110),    # P3/8  (80)  - Emerald Green
    16: (55, 175, 255),   # P4/16 (40)  - Amber Orange
    32: (120, 95, 255),   # P5/32 (20)  - Coral / Magenta
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
        description="Visualize YOLOv11 Backbone multi-scale features from 160x160 (P2/4) down to 20x20 (P5/32)."
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
        help="Optional second YOLOv11 checkpoint (.pth/.pt) for side-by-side 160->20 backbone comparison.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="YOLOv11 variant override (yolov11n/s/m/l/x). Auto-inferred from checkpoint if omitted (defaults to yolov11s).",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional YAML config path for custom YOLOv11 architectures.",
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
        "--source",
        type=str,
        default="",
        help="Input image file or directory. Auto-selects from dataset/ExDark or dataset/DarkFace if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization/backbone_yolov11",
        help="Root directory to save backbone feature visualizations (default: visualization/backbone_yolov11).",
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
        help="Also include L00 Stem (P1/2, 320x320) before L01..L10 (160->20).",
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
        help="Number of top active individual channels per layer sheet (default: 16, 0 to skip sheets).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=list(COLORMAP_TABLE.keys()),
        help="OpenCV colormap for activation heatmaps (default: turbo).",
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
        default=380,
        help="Display panel width in pixels for summary tiles (default: 380).",
    )
    return parser.parse_args()


# ==================================================================================================
# 2. Model Loading & Backbone Hook Recorder
# ==================================================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_class_names(classes_arg: str, nc: int, ckpt_names: Optional[List[str]] = None) -> List[str]:
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


def detect_is_modular_ckpt(ckpt_meta: Any) -> bool:
    if not isinstance(ckpt_meta, dict):
        return False
    sd = ckpt_meta.get("state_dict") or ckpt_meta.get("model_state_dict")
    if isinstance(sd, dict) and sd:
        sample_keys = list(sd.keys())[:15]
        return any(k.startswith(("backbone.", "module.backbone.")) for k in sample_keys)
    return False


def load_yolov11_for_backbone(
    weights_path: str,
    model_override: str = "",
    cfg_override: str = "",
    nc_override: Optional[int] = None,
    modular_override: Optional[bool] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, str, int, Dict[str, Any]]:
    """Build YOLOv11 architecture and load weights (`auto`, `.pt`, `.pth`, or `'none'`)."""
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
        raise ValueError(
            f"Expected a YOLOv11 variant (yolov11n/s/m/l/x), but resolved '{model_name}'."
        )

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
        _detect_is_modular_sd(raw_sd) if modular_override is None else bool(modular_override)
    )
    cfg_path = cfg_override if (cfg_override and os.path.exists(cfg_override)) else None
    detected_arch = detect_ckpt_arch(raw_sd) if raw_sd else None

    if cfg_path is not None or is_modular or detected_arch in {None, "modular", "p2_illumsab"}:
        model = build_yolo_model(
            model_name=model_name,
            num_classes=nc,
            in_channels=3,
            cfg=cfg_path,
            modular=is_modular,
        )
    else:
        cfg_dict = _build_arch_cfg_dict(model_name, nc, detected_arch)
        model = DetectionModel(cfg=cfg_dict, ch=3, nc=nc)
        if detected_arch == "p2_illumsab_v1" and getattr(model, "illum_extractor", None) is not None:
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
        "ckpt_names": ckpt_names,
        "weights_label": w_label,
    }
    model = model.to(device).eval()
    return model, model_name, nc, meta_info


class YOLOv11BackboneRecorder:
    """
    Captures YOLOv11 Backbone layers (L01..L11 for P2-IllumSAB/SAB/EdgeSAB, or L01..L10 for Base),
    plus optional L00 Stem (320x320), supporting both `DetectionModel` and modular `YOLOv11`.
    """

    def __init__(self, model: nn.Module, include_stem: bool = False):
        self.model = de_parallel(model)
        self.include_stem = include_stem
        self.handles: List[Any] = []
        self.records: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.stage_slots: Dict[str, int] = {}
        self.has_p2_attn: bool = False
        self._register()

    def _make_hook(self, layer_idx: int, mod: nn.Module, meta_row: Tuple[str, str, int, bool, str]):
        stage_group, label, stride, is_stage_out, slot = meta_row
        mod_type = type(mod).__name__
        if layer_idx == 3 and self.has_p2_attn:
            stage_group = f"P2/4 ({mod_type})"
            label = f"L03_{mod_type}_P2_4_160"

        def hook_fn(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.records[layer_idx] = {
                    "layer_idx": layer_idx,
                    "stage_group": stage_group,
                    "label": label,
                    "stride": stride,
                    "is_stage_out": is_stage_out,
                    "slot": slot,
                    "mod_type": mod_type,
                    "tensor": output.detach().float().cpu(),
                }

        return hook_fn

    def _register(self) -> None:
        start_idx = 0 if self.include_stem else 1
        p2_types = (IllumSAB, SAB, EdgeSAB)

        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            layers = list(self.model.model)
            self.has_p2_attn = len(layers) > 3 and isinstance(layers[3], p2_types)
            meta_tbl = YOLOV11_BACKBONE_P2_ATTN_META if self.has_p2_attn else YOLOV11_BACKBONE_BASE_META
            for idx in range(start_idx, len(meta_tbl)):
                if idx < len(layers):
                    mod = layers[idx]
                    row = meta_tbl[idx]
                    self.handles.append(mod.register_forward_hook(self._make_hook(idx, mod, row)))
                    if row[4]:
                        self.stage_slots[row[4]] = idx
        elif hasattr(self.model, "backbone"):
            bb = self.model.backbone
            self.has_p2_attn = hasattr(bb, "attn_p2") and isinstance(bb.attn_p2, p2_types)
            meta_tbl = YOLOV11_BACKBONE_P2_ATTN_META if self.has_p2_attn else YOLOV11_BACKBONE_BASE_META
            off = 1 if self.has_p2_attn else 0
            mod_map: Dict[int, nn.Module] = {
                0: bb.stem,
                1: bb.stage1[0],
                2: bb.stage1[1],
                3 + off: bb.stage2[0],
                4 + off: bb.stage2[1],
                5 + off: bb.stage3[0],
                6 + off: bb.stage3[1],
                7 + off: bb.stage4[0],
                8 + off: bb.stage4[1],
                9 + off: bb.stage4[2],
                10 + off: bb.stage4[3],
            }
            if self.has_p2_attn:
                mod_map[3] = bb.attn_p2
            for idx in range(start_idx, len(meta_tbl)):
                if idx in mod_map:
                    mod = mod_map[idx]
                    row = meta_tbl[idx]
                    self.handles.append(mod.register_forward_hook(self._make_hook(idx, mod, row)))
                    if row[4]:
                        self.stage_slots[row[4]] = idx
        else:
            raise AttributeError("Unsupported YOLOv11 model structure: missing `.model` and `.backbone`.")

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==================================================================================================
# 3. Image Preprocessing & Feature Map Mathematical Projections (L2, Mean, Max, PCA-RGB)
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
            PROJECT_ROOT / "dataset" / "ExDark" / "images" / "train",
            PROJECT_ROOT / "dataset" / "DarkFace" / "images" / "test",
            PROJECT_ROOT / "dataset" / "DarkFace" / "images" / "val",
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
                "No --source specified and no default images found under dataset/ExDark or dataset/DarkFace."
            )

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
    """Letterbox image to `(img_size, img_size)` and return tensor + unpad crop box."""
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
    return torch.from_numpy(tensor).unsqueeze(0), crop_box, (ratio, (dw, dh))


def load_optional_gt(
    img_path: str, orig_hw: Tuple[int, int], class_names: List[str]
) -> np.ndarray:
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
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def compute_spatial_projections(feat: torch.Tensor) -> Dict[str, Any]:
    """
    Compute spatial summary maps from a `(1, C, H, W)` feature tensor:
      - `l2`: RMS / L2 channel energy map `(H, W)`
      - `mean`: Mean activation map `(H, W)`
      - `max`: Max channel response map `(H, W)`
      - `pca_bgr`: Top-3 Principal Components projected to `[0, 255]` BGR `(H, W, 3)`
      - Quantitative statistics (`l2_mean`, `abs_mean`, `max_val`, `sparsity`)
    """
    f = feat[0].float()  # (C, H, W)
    c, h, w = f.shape
    l2_map = torch.sqrt((f ** 2).mean(dim=0) + 1e-12).numpy()
    mean_map = f.mean(dim=0).numpy()
    max_map = f.max(dim=0).values.numpy()

    # PCA 3-component RGB semantic projection across C channels
    flat = f.reshape(c, h * w).T  # (N_hw, C)
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    if min(h * w, c) >= 3 and float(flat_centered.std()) > 1e-8:
        try:
            u, s, _ = torch.pca_lowrank(flat_centered, q=min(6, c, h * w), center=False, niter=2)
            proj = (u[:, :3] * s[:3]).numpy()  # (N_hw, 3)
        except Exception:
            proj = flat_centered[:, :3].numpy()
        # Deterministic sign convention so dominant foreground stays visually consistent
        for k in range(3):
            if float(np.percentile(proj[:, k], 95) + np.percentile(proj[:, k], 5)) < 0.0:
                proj[:, k] = -proj[:, k]
            p_lo, p_hi = np.percentile(proj[:, k], 1.0), np.percentile(proj[:, k], 99.0)
            if p_hi > p_lo + 1e-8:
                proj[:, k] = np.clip((proj[:, k] - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            else:
                proj[:, k] = 0.0
        pca_rgb = (proj.reshape(h, w, 3) * 255.0).astype(np.uint8)
        pca_bgr = pca_rgb[:, :, ::-1].copy()
    else:
        u8 = normalize_map_to_uint8(l2_map)
        pca_bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    sparsity = float((f.abs() < 1e-2).float().mean().item()) * 100.0
    return {
        "shape": (c, h, w),
        "l2": l2_map,
        "mean": mean_map,
        "max": max_map,
        "pca_bgr": pca_bgr,
        "l2_mean": float(l2_map.mean()),
        "abs_mean": float(f.abs().mean().item()),
        "max_val": float(f.max().item()),
        "sparsity": sparsity,
    }


def unpad_and_resize_2d(
    map2d: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    nearest: bool = False,
) -> np.ndarray:
    """Map `(H_f, W_f)` feature map back to original unpadded image aspect ratio."""
    interp_up = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    interp_out = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    up = cv2.resize(map2d.astype(np.float32), (lb_size, lb_size), interpolation=interp_up)
    top, left, bottom_y, right_x = crop_box
    cropped = up[top:bottom_y, left:right_x]
    if cropped.size == 0:
        cropped = up
    return cv2.resize(cropped, target_wh, interpolation=interp_out)


def unpad_and_resize_bgr(
    img_grid_bgr: np.ndarray,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    target_wh: Tuple[int, int],
    nearest: bool = True,
) -> np.ndarray:
    """Resize a 3-channel feature visualization (e.g. PCA-RGB) preserving crisp grid cells."""
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
    """
    Return `(smooth_overlay_bgr, smooth_heatmap_bgr, native_grid_heatmap_bgr)`
    all sized identically to `base_bgr`.
    """
    h, w = base_bgr.shape[:2]
    smooth_aligned = unpad_and_resize_2d(map2d, lb_size, crop_box, (w, h), nearest=False)
    u8_smooth = normalize_map_to_uint8(
        smooth_aligned, clip_percentile=clip_percentile, fixed_range=fixed_range
    )
    heat_smooth = cv2.applyColorMap(u8_smooth, colormap_id)
    overlay_smooth = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_smooth, alpha, 0.0)

    # Native grid preserves true discrete spatial cells (160x160 vs 80x80 vs 40x40 vs 20x20)
    u8_raw = normalize_map_to_uint8(
        map2d, clip_percentile=clip_percentile, fixed_range=fixed_range
    )
    heat_raw_cell = cv2.applyColorMap(u8_raw, colormap_id)
    heat_native_grid = unpad_and_resize_bgr(heat_raw_cell, lb_size, crop_box, (w, h), nearest=True)
    return overlay_smooth, heat_smooth, heat_native_grid


# ==================================================================================================
# 4. Layout & Composite Drawing Helpers
# ==================================================================================================

def resize_to_panel(img_bgr: np.ndarray, panel_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel_h = max(64, int(round(h * (panel_w / float(w)))))
    interp = cv2.INTER_AREA if panel_w < w else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (panel_w, panel_h), interpolation=interp)


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
        txt_col = (0, 0, 0) if bright > 140 else (255, 255, 255)
        cv2.putText(
            canvas, label, (x1 + 3, by2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA
        )
    return canvas


def make_labeled_tile(
    img_bgr: np.ndarray,
    title: str,
    subtitle: str = "",
    accent_bgr: Tuple[int, int, int] = (70, 150, 255),
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    header_h = 44 if subtitle else 28
    tile = np.full((h + header_h, w, 3), 24, dtype=np.uint8)
    tile[header_h:, :] = img_bgr
    cv2.rectangle(tile, (0, 0), (w - 1, header_h - 1), (30, 32, 38), -1)
    cv2.line(tile, (0, 0), (w - 1, 0), accent_bgr, 3)
    cv2.line(tile, (0, header_h - 1), (w - 1, header_h - 1), (62, 66, 76), 1)

    t_scale = min(0.48, max(0.32, w / 820.0))
    cv2.putText(
        tile, title, (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX, t_scale, (240, 244, 252), 1, cv2.LINE_AA
    )
    if subtitle:
        s_scale = min(0.38, max(0.27, w / 1000.0))
        cv2.putText(
            tile, subtitle, (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX, s_scale, (168, 178, 195), 1, cv2.LINE_AA
        )
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
        cv2.putText(
            canvas, banner_title, (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 248, 255), 2, cv2.LINE_AA
        )
        if banner_sub:
            cv2.putText(
                canvas, banner_sub, (14, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 185, 202), 1, cv2.LINE_AA
            )

    for idx, tile in enumerate(tiles):
        r, c = idx // ncols, idx % ncols
        y0 = banner_h + pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        h_t, w_t = tile.shape[:2]
        canvas[y0 : y0 + h_t, x0 : x0 + w_t] = tile

    if colormap_id is not None:
        bar_h = max(120, canvas_h - banner_h - 2 * pad - 44)
        bar_x = canvas_w - cbar_w + 12
        bar_y = banner_h + pad + 22
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        grad_bar = cv2.applyColorMap(np.repeat(grad, 18, axis=1), colormap_id)
        canvas[bar_y : bar_y + bar_h, bar_x : bar_x + 18] = grad_bar
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 18, bar_y + bar_h), (125, 130, 140), 1)
        cv2.putText(
            canvas, "High", (bar_x - 4, bar_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (232, 236, 244), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, "Low", (bar_x - 2, bar_y + bar_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA
        )

    return canvas


# ==================================================================================================
# 5. Backbone Summary Figures & Per-Layer Exporters
# ==================================================================================================

def build_pyramid_hero_figure(
    raw_panel: np.ndarray,
    ref_panel: np.ndarray,
    det_panel: np.ndarray,
    gt_panel: np.ndarray,
    records: "OrderedDict[int, Dict[str, Any]]",
    projections: Dict[int, Dict[str, Any]],
    stage_slots: Dict[str, int],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
    img_name: str,
) -> np.ndarray:
    """
    Create `00_backbone_pyramid_160_to_20.png`:
      - Row 1: Raw Input + Stage Outputs (160 -> 80 -> 40 -> 20) Smooth L2 Overlay
      - Row 2: Brightened Ref + Stage Outputs Native-Grid L2 Heatmap
      - Row 3: Detections (or GT) + Stage Outputs PCA 3-Component RGB Semantic Map
    """
    ph, pw = ref_panel.shape[:2]
    key_indices = [stage_slots[s] for s in ("p2_pre", "p2", "p3", "p4", "p5") if s in stage_slots and stage_slots[s] in records]

    row1: List[np.ndarray] = [
        make_labeled_tile(raw_panel, "Input (Raw Low-Light)", f"{img_name}", (130, 140, 155))
    ]
    row2: List[np.ndarray] = [
        make_labeled_tile(ref_panel, "Display Underlay", "Gamma-adjusted reference", (170, 185, 205))
    ]
    row3: List[np.ndarray] = [
        make_labeled_tile(det_panel, "Model Predictions / GT", "Reference detections", (80, 220, 120))
    ]

    for idx in key_indices:
        rec = records[idx]
        proj = projections[idx]
        c, h_f, w_f = proj["shape"]
        stride = rec["stride"]
        accent = STAGE_ACCENT_BGR.get(stride, (80, 160, 255))

        ov_bgr, _, native_grid_bgr = render_views_for_map(
            ref_panel, proj["l2"], lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct
        )
        pca_aligned = unpad_and_resize_bgr(
            proj["pca_bgr"], lb_size, crop_box, (pw, ph), nearest=True
        )

        scale_tag = f"L{idx:02d} {rec['mod_type']} | {h_f}x{w_f} (P{int(np.log2(stride))}/{stride})"
        stat_sub = f"C={c} | L2={proj['l2_mean']:.3f} | Max={proj['max_val']:.2f}"

        row1.append(make_labeled_tile(ov_bgr, f"{scale_tag} [Overlay]", stat_sub, accent))
        row2.append(
            make_labeled_tile(
                native_grid_bgr,
                f"{h_f}x{w_f} Native Grid [L2]",
                f"Cell size {stride}x{stride}px | Sparsity {proj['sparsity']:.1f}%",
                accent,
            )
        )
        row3.append(
            make_labeled_tile(
                pca_aligned,
                f"{h_f}x{w_f} PCA-RGB [Top-3 PC]",
                f"Channel subspace ({c} -> 3 RGB)",
                accent,
            )
        )

    ncols = 1 + len(key_indices)
    return assemble_grid(
        row1 + row2 + row3,
        ncols=ncols,
        banner_title=f"YOLOv11 Backbone Pyramid (160 -> 80 -> 40 -> 20) — {model_title}",
        banner_sub="Row 1: Aligned L2 Energy Overlay | Row 2: Discrete Native-Resolution Grid | Row 3: Top-3 PCA Semantic RGB",
        colormap_id=colormap_id,
    )


def build_all_layers_progression_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[int, Dict[str, Any]]",
    projections: Dict[int, Dict[str, Any]],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
) -> np.ndarray:
    """Create `01_all_backbone_layers_160_to_20.png` covering L01..L10 (and L00 if enabled)."""
    ph, pw = ref_panel.shape[:2]
    overlay_tiles: List[np.ndarray] = []
    native_tiles: List[np.ndarray] = []

    for idx, rec in records.items():
        proj = projections[idx]
        c, h_f, w_f = proj["shape"]
        stride = rec["stride"]
        accent = STAGE_ACCENT_BGR.get(stride, (80, 160, 255))
        star = " [Stage Out]" if rec["is_stage_out"] else ""

        ov_bgr, _, native_bgr = render_views_for_map(
            ref_panel, proj["l2"], lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct
        )
        title = f"L{idx:02d} {rec['mod_type']} ({h_f}x{w_f}){star}"
        sub = f"{rec['stage_group']} | C={c} | L2={proj['l2_mean']:.3f}"
        overlay_tiles.append(make_labeled_tile(ov_bgr, title, sub, accent))
        native_tiles.append(make_labeled_tile(native_bgr, f"L{idx:02d} Native {h_f}x{w_f}", sub, accent))

    ncols = min(5, max(1, len(overlay_tiles)))
    return assemble_grid(
        overlay_tiles + native_tiles,
        ncols=ncols,
        banner_title=f"YOLOv11 Backbone Layer-by-Layer Progression (L{min(records):02d} -> L{max(records):02d}) — {model_title}",
        banner_sub="Top block: Spatial L2 Overlay on Image | Bottom block: Pure Native-Grid Resolution Heatmaps (160 -> 80 -> 40 -> 20)",
        colormap_id=colormap_id,
    )


def build_stage4_20x20_refinement_figure(
    ref_panel: np.ndarray,
    records: "OrderedDict[int, Dict[str, Any]]",
    projections: Dict[int, Dict[str, Any]],
    stage_slots: Dict[str, int],
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
    model_title: str,
) -> Optional[np.ndarray]:
    """Create `02_stage4_20x20_refinement.png`: Stage 4 Conv -> C3k2 -> SPPF -> C2PSA + Delta."""
    s4_indices = [idx for idx, rec in records.items() if rec["stride"] == 32]
    if len(s4_indices) < 2:
        return None

    accent = STAGE_ACCENT_BGR[32]
    tiles: List[np.ndarray] = []
    for idx in s4_indices:
        rec = records[idx]
        proj = projections[idx]
        c, h_f, w_f = proj["shape"]
        ov_bgr, _, native_bgr = render_views_for_map(
            ref_panel, proj["l2"], lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct
        )
        tiles.append(
            make_labeled_tile(
                ov_bgr,
                f"L{idx:02d} {rec['mod_type']} ({h_f}x{w_f}) Overlay",
                f"C={c} | L2={proj['l2_mean']:.3f} | Max={proj['max_val']:.2f}",
                accent,
            )
        )
        tiles.append(
            make_labeled_tile(
                native_bgr,
                f"L{idx:02d} {rec['mod_type']} Native {h_f}x{w_f}",
                f"Sparsity={proj['sparsity']:.1f}%",
                accent,
            )
        )

    sppf_idx = stage_slots.get("s4_sppf", -1)
    c2psa_idx = stage_slots.get("p5", -1)
    if sppf_idx in records and c2psa_idx in records:
        t9 = records[sppf_idx]["tensor"][0]
        t10 = records[c2psa_idx]["tensor"][0]
        if t9.shape == t10.shape:
            delta_l2 = torch.sqrt(((t10 - t9) ** 2).mean(dim=0) + 1e-12).numpy()
            ov_d, _, nat_d = render_views_for_map(
                ref_panel, delta_l2, lb_size, crop_box, colormap_id,
                alpha=alpha, clip_percentile=clip_pct
            )
            tiles.append(
                make_labeled_tile(
                    ov_d,
                    f"C2PSA Effect |L{c2psa_idx:02d} - L{sppf_idx:02d}|",
                    f"Mean Delta L2={float(delta_l2.mean()):.4f}",
                    (60, 220, 255),
                )
            )
            tiles.append(
                make_labeled_tile(
                    nat_d,
                    f"C2PSA Native 20x20 |L{c2psa_idx:02d} - L{sppf_idx:02d}|",
                    "Spatial regions re-weighted by PSA",
                    (60, 220, 255),
                )
            )

    # Reorder so Row 1 = Overlays, Row 2 = Native Grids
    overlays = tiles[0::2]
    natives = tiles[1::2]
    return assemble_grid(
        overlays + natives,
        ncols=len(overlays),
        banner_title=f"Stage 4 Deep 20x20 Refinement (Conv -> C3k2 -> SPPF -> C2PSA) — {model_title}",
        banner_sub="Shows multi-scale context aggregation (SPPF) and global spatial attention focusing (C2PSA) at 20x20",
        colormap_id=colormap_id,
    )


def build_backbone_stats_chart(
    records: "OrderedDict[int, Dict[str, Any]]",
    projections: Dict[int, Dict[str, Any]],
    model_title: str,
) -> np.ndarray:
    """Render `03_backbone_activation_stats.png`: L2 energy, Max response, and Sparsity across L01..L10."""
    items = list(records.keys())
    n = len(items)
    row_h = 42
    top_h = 84
    bot_h = 34
    width = 1080
    height = top_h + n * row_h + bot_h
    canvas = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width - 1, 58), (28, 30, 36), -1)
    cv2.line(canvas, (0, 58), (width - 1, 58), (80, 155, 255), 2)
    cv2.putText(
        canvas,
        f"YOLOv11 Backbone Quantitative Feature Diagnostics (160 -> 20) — {model_title}",
        (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 248, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Layer / Resolution         Shape (C,H,W)      L2 Energy (RMS)                     Sparsity (|x|<0.01)      Max",
        (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (175, 185, 205), 1, cv2.LINE_AA
    )

    max_l2 = max((projections[i]["l2_mean"] for i in items), default=1.0)
    max_l2 = max(max_l2, 1e-6)

    for r_i, idx in enumerate(items):
        rec = records[idx]
        proj = projections[idx]
        c, h_f, w_f = proj["shape"]
        y_top = top_h + r_i * row_h
        y_mid = y_top + 25
        if r_i % 2 == 0:
            cv2.rectangle(canvas, (10, y_top), (width - 10, y_top + row_h - 2), (24, 27, 33), -1)

        accent = STAGE_ACCENT_BGR.get(rec["stride"], (90, 170, 255))
        cv2.rectangle(canvas, (10, y_top), (15, y_top + row_h - 2), accent, -1)

        name_txt = f"L{idx:02d} {rec['mod_type']} ({h_f}x{w_f})"
        shape_txt = f"({c}, {h_f}, {w_f})"
        cv2.putText(canvas, name_txt, (24, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 240, 248), 1, cv2.LINE_AA)
        cv2.putText(canvas, shape_txt, (255, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (180, 190, 208), 1, cv2.LINE_AA)

        # L2 Energy bar
        bar_x0, bar_max_w = 400, 270
        l2_w = int(round((proj["l2_mean"] / max_l2) * bar_max_w))
        cv2.rectangle(canvas, (bar_x0, y_top + 10), (bar_x0 + bar_max_w, y_top + 28), (40, 44, 52), -1)
        if l2_w > 0:
            cv2.rectangle(canvas, (bar_x0, y_top + 10), (bar_x0 + l2_w, y_top + 28), accent, -1)
        cv2.putText(
            canvas, f"{proj['l2_mean']:.4f}", (bar_x0 + bar_max_w + 10, y_mid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, (230, 235, 245), 1, cv2.LINE_AA
        )

        # Sparsity bar
        sp_x0, sp_max_w = 760, 140
        sp_w = int(round((proj["sparsity"] / 100.0) * sp_max_w))
        cv2.rectangle(canvas, (sp_x0, y_top + 10), (sp_x0 + sp_max_w, y_top + 28), (40, 44, 52), -1)
        if sp_w > 0:
            cv2.rectangle(canvas, (sp_x0, y_top + 10), (sp_x0 + sp_w, y_top + 28), (110, 195, 130), -1)
        cv2.putText(
            canvas, f"{proj['sparsity']:.1f}%", (sp_x0 + sp_max_w + 8, y_mid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, (205, 215, 228), 1, cv2.LINE_AA
        )

        # Max value
        cv2.putText(
            canvas, f"{proj['max_val']:.2f}", (985, y_mid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, (235, 215, 150), 1, cv2.LINE_AA
        )

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
    """Save standalone borderless maps + Top-K individual channel sheet for a single Backbone layer."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    ph, pw = ref_panel.shape[:2]
    c, h_f, w_f = proj["shape"]
    stride = rec["stride"]
    accent = STAGE_ACCENT_BGR.get(stride, (80, 160, 255))

    ov_l2, heat_l2, native_l2 = render_views_for_map(
        ref_panel, proj["l2"], lb_size, crop_box, colormap_id,
        alpha=alpha, clip_percentile=clip_pct
    )
    pca_aligned = unpad_and_resize_bgr(proj["pca_bgr"], lb_size, crop_box, (pw, ph), nearest=True)

    # 1. Standalone clean borderless images for direct paper / slide insertion
    cv2.imwrite(str(layer_dir / "l2_overlay.png"), ov_l2)
    cv2.imwrite(str(layer_dir / "l2_heatmap.png"), heat_l2)
    cv2.imwrite(str(layer_dir / "native_grid.png"), native_l2)
    cv2.imwrite(str(layer_dir / "pca_rgb.png"), pca_aligned)

    if top_k_channels <= 0:
        return

    # 2. Detailed Top-K active channels sheet
    _, _, native_mean = render_views_for_map(
        ref_panel, proj["mean"], lb_size, crop_box, colormap_id,
        alpha=alpha, clip_percentile=clip_pct
    )
    _, _, native_max = render_views_for_map(
        ref_panel, proj["max"], lb_size, crop_box, colormap_id,
        alpha=alpha, clip_percentile=clip_pct
    )

    summary_tiles: List[np.ndarray] = [
        make_labeled_tile(ov_l2, "L2 Energy Overlay", f"Shape ({c}, {h_f}, {w_f})", accent),
        make_labeled_tile(native_l2, f"L2 Native Grid ({h_f}x{w_f})", f"Mean L2={proj['l2_mean']:.4f}", accent),
        make_labeled_tile(native_max, f"Max Response ({h_f}x{w_f})", f"Peak={proj['max_val']:.3f}", accent),
        make_labeled_tile(pca_aligned, f"PCA-RGB ({h_f}x{w_f})", "Top-3 Principal Components", accent),
    ]

    feat = rec["tensor"][0]  # (C, H, W)
    ch_energy = torch.sqrt((feat ** 2).mean(dim=(1, 2)) + 1e-12)
    k = min(int(top_k_channels), c)
    top_vals, top_idxs = torch.topk(ch_energy, k=k, largest=True)

    ch_tiles: List[np.ndarray] = []
    for rank, (ch_i, ch_e) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), start=1):
        ch_map = feat[ch_i].numpy()
        _, _, ch_native = render_views_for_map(
            ref_panel, ch_map, lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct
        )
        ch_tiles.append(
            make_labeled_tile(
                ch_native,
                f"Top#{rank} Ch {ch_i}/{c} ({h_f}x{w_f})",
                f"RMS={ch_e:.4f} | min={ch_map.min():.2f} max={ch_map.max():.2f}",
                (95, 110, 135),
            )
        )

    ncols = 4
    sheet = assemble_grid(
        summary_tiles + ch_tiles,
        ncols=ncols,
        banner_title=f"{rec['label']} — Resolution {h_f}x{w_f} (Stride {stride}, Channels {c})",
        banner_sub=f"Row 1: Aggregated Summary Maps | Remaining rows: Top-{k} Highest-Energy Individual Feature Channels",
        colormap_id=colormap_id,
    )
    cv2.imwrite(str(layer_dir / "channel_sheet.png"), sheet)


def build_backbone_comparison_figure(
    ref_panel: np.ndarray,
    records_a: "OrderedDict[int, Dict[str, Any]]",
    proj_a: Dict[int, Dict[str, Any]],
    slots_a: Dict[str, int],
    label_a: str,
    records_b: "OrderedDict[int, Dict[str, Any]]",
    proj_b: Dict[int, Dict[str, Any]],
    slots_b: Dict[str, int],
    label_b: str,
    lb_size: int,
    crop_box: Tuple[int, int, int, int],
    colormap_id: int,
    alpha: float,
    clip_pct: float,
) -> np.ndarray:
    """Compare two YOLOv11 checkpoints across semantic stages P2(160) -> P3(80) -> P4(40) -> P5(20)."""
    common_slots = [s for s in KEY_PYRAMID_SLOTS if s in slots_a and s in slots_b]
    row_a: List[np.ndarray] = []
    row_b: List[np.ndarray] = []
    row_diff: List[np.ndarray] = []

    for slot in common_slots:
        ia, ib = slots_a[slot], slots_b[slot]
        pa, pb = proj_a[ia], proj_b[ib]
        stride = records_a[ia]["stride"]
        _, h_f, w_f = pa["shape"]
        accent = STAGE_ACCENT_BGR.get(stride, (80, 160, 255))

        vmax = max(
            float(np.percentile(pa["l2"], clip_pct)),
            float(np.percentile(pb["l2"], clip_pct)),
            1e-6,
        )
        vmin = min(float(pa["l2"].min()), float(pb["l2"].min()))

        ov_a, _, _ = render_views_for_map(
            ref_panel, pa["l2"], lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct, fixed_range=(vmin, vmax)
        )
        ov_b, _, _ = render_views_for_map(
            ref_panel, pb["l2"], lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct, fixed_range=(vmin, vmax)
        )

        diff_2d = np.abs(pb["l2"] - pa["l2"])
        _, _, diff_nat = render_views_for_map(
            ref_panel, diff_2d, lb_size, crop_box, colormap_id,
            alpha=alpha, clip_percentile=clip_pct
        )

        row_a.append(
            make_labeled_tile(
                ov_a, f"[A] {slot.upper()} L{ia:02d} ({h_f}x{w_f}) {label_a}",
                f"L2={pa['l2_mean']:.4f} | Sparsity={pa['sparsity']:.1f}%", accent
            )
        )
        row_b.append(
            make_labeled_tile(
                ov_b, f"[B] {slot.upper()} L{ib:02d} ({h_f}x{w_f}) {label_b}",
                f"L2={pb['l2_mean']:.4f} | Sparsity={pb['sparsity']:.1f}%", (90, 230, 130)
            )
        )
        row_diff.append(
            make_labeled_tile(
                diff_nat, f"|B - A| {slot.upper()} ({h_f}x{w_f})",
                f"Mean |Delta L2|={float(diff_2d.mean()):.4f}", (80, 210, 255)
            )
        )

    return assemble_grid(
        row_a + row_b + row_diff,
        ncols=max(1, len(common_slots)),
        banner_title=f"YOLOv11 Backbone Comparison (160 -> 80 -> 40 -> 20): [A] {label_a} vs [B] {label_b}",
        banner_sub="Row 1: Model A (shared scale) | Row 2: Model B (shared scale) | Row 3: Absolute L2 Difference |B - A|",
        colormap_id=colormap_id,
    )


# ==================================================================================================
# 6. Main Execution Pipeline
# ==================================================================================================

def run_inference_and_record(
    model: nn.Module,
    recorder: YOLOv11BackboneRecorder,
    inp_tensor: torch.Tensor,
    orig_hw: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]],
    conf_thres: float,
    iou_thres: float,
) -> Tuple["OrderedDict[int, Dict[str, Any]]", Dict[int, Dict[str, Any]], np.ndarray]:
    recorder.clear()
    with torch.no_grad():
        out = model(inp_tensor)
        pred_raw = out[0] if isinstance(out, tuple) else out

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
            dets[:, :4] = scale_boxes(
                inp_tensor.shape[2:], dets[:, :4], orig_hw, ratio_pad=ratio_pad
            )
            dets_np = dets.cpu().numpy()

    records_copy: "OrderedDict[int, Dict[str, Any]]" = OrderedDict(
        (k, dict(v)) for k, v in recorder.records.items()
    )
    projections = {
        idx: compute_spatial_projections(rec["tensor"]) for idx, rec in records_copy.items()
    }
    return records_copy, projections, dets_np


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    colormap_id = COLORMAP_TABLE[args.colormap]

    model_a, model_name_a, nc_a, meta_a = load_yolov11_for_backbone(
        weights_path=args.weights,
        model_override=args.model,
        cfg_override=args.cfg,
        nc_override=args.nc,
        modular_override=args.modular,
        device=device,
    )
    class_names = resolve_class_names(args.classes, nc_a, meta_a.get("ckpt_names"))
    recorder_a = YOLOv11BackboneRecorder(model_a, include_stem=args.include_stem)

    model_b = None
    recorder_b = None
    meta_b: Dict[str, Any] = {}
    if args.compare_weights.strip():
        model_b, _, _, meta_b = load_yolov11_for_backbone(
            weights_path=args.compare_weights,
            model_override=args.model,
            cfg_override=args.cfg,
            nc_override=args.nc,
            modular_override=args.modular,
            device=device,
        )
        recorder_b = YOLOv11BackboneRecorder(model_b, include_stem=args.include_stem)

    image_paths = collect_input_images(args.source, args.num, args.shuffle)
    run_tag = (
        f"{model_name_a}_{meta_a['weights_label']}"
        if not args.compare_weights.strip()
        else f"compare_{meta_a['weights_label']}_vs_{meta_b['weights_label']}"
    )
    out_root = Path(args.output) / run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"===> Visualizing YOLOv11 Backbone ({'320->20' if args.include_stem else '160->20'}) "
        f"on {len(image_paths)} image(s) -> '{out_root}'"
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

            records_a, proj_a, dets_a = run_inference_and_record(
                model_a, recorder_a, inp_tensor, orig_hw, ratio_pad,
                args.conf_thres, args.iou_thres
            )

            gt_boxes = load_optional_gt(img_path, orig_hw, class_names)
            ref_full = apply_display_gamma(img_bgr, args.vis_gamma)
            det_full = (
                draw_detections_on_img(ref_full, dets_a, class_names, show_conf=True)
                if len(dets_a) > 0
                else draw_detections_on_img(ref_full, gt_boxes, class_names, show_conf=False)
            )
            gt_full = draw_detections_on_img(ref_full, gt_boxes, class_names, show_conf=False)

            raw_panel = resize_to_panel(img_bgr, args.panel_width)
            ref_panel = resize_to_panel(ref_full, args.panel_width)
            det_panel = resize_to_panel(det_full, args.panel_width)
            gt_panel = resize_to_panel(gt_full, args.panel_width)

            model_title = f"{model_name_a} ({meta_a['weights_label']}, epoch={meta_a['epoch']})"

            # 1. Hero 160 -> 80 -> 40 -> 20 Pyramid Summary
            fig_hero = build_pyramid_hero_figure(
                raw_panel, ref_panel, det_panel, gt_panel,
                records_a, proj_a, recorder_a.stage_slots, args.imgsz, crop_box,
                colormap_id, args.alpha, args.clip_percentile,
                model_title, Path(img_path).name,
            )
            cv2.imwrite(str(img_out_dir / "00_backbone_pyramid_160_to_20.png"), fig_hero)

            # 2. All Backbone Layers Progression (L01..L11 or L01..L10)
            fig_all = build_all_layers_progression_figure(
                ref_panel, records_a, proj_a, args.imgsz, crop_box,
                colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            cv2.imwrite(str(img_out_dir / "01_all_backbone_layers_160_to_20.png"), fig_all)

            # 3. Stage 4 (20x20) Deep Refinement (Conv -> C3k2 -> SPPF -> C2PSA)
            fig_s4 = build_stage4_20x20_refinement_figure(
                ref_panel, records_a, proj_a, recorder_a.stage_slots, args.imgsz, crop_box,
                colormap_id, args.alpha, args.clip_percentile, model_title,
            )
            if fig_s4 is not None:
                cv2.imwrite(str(img_out_dir / "02_stage4_20x20_refinement.png"), fig_s4)

            # 4. Quantitative Backbone Diagnostics Chart
            fig_stats = build_backbone_stats_chart(records_a, proj_a, model_title)
            cv2.imwrite(str(img_out_dir / "03_backbone_activation_stats.png"), fig_stats)

            # 5. Individual Per-Layer Standalone Maps + Top-K Channel Sheets
            layers_root = img_out_dir / "layers"
            for l_idx, rec in records_a.items():
                _, h_f, w_f = proj_a[l_idx]["shape"]
                sub_dir = layers_root / f"{rec['label']}_{h_f}x{w_f}"
                save_per_layer_assets(
                    sub_dir, ref_panel, rec, proj_a[l_idx],
                    args.imgsz, crop_box, colormap_id,
                    args.alpha, args.clip_percentile, args.channels,
                )

            # 6. Optional Side-by-Side Comparison across 160 -> 80 -> 40 -> 20
            if model_b is not None and recorder_b is not None:
                records_b, proj_b, _ = run_inference_and_record(
                    model_b, recorder_b, inp_tensor, orig_hw, ratio_pad,
                    args.conf_thres, args.iou_thres
                )
                fig_cmp = build_backbone_comparison_figure(
                    ref_panel,
                    records_a, proj_a, recorder_a.stage_slots, meta_a["weights_label"],
                    records_b, proj_b, recorder_b.stage_slots, meta_b["weights_label"],
                    args.imgsz, crop_box, colormap_id,
                    args.alpha, args.clip_percentile,
                )
                cv2.imwrite(str(img_out_dir / "04_backbone_comparison_160_to_20.png"), fig_cmp)

            scales_str = " -> ".join(
                f"L{recorder_a.stage_slots[s]:02d}({proj_a[recorder_a.stage_slots[s]]['shape'][1]}x{proj_a[recorder_a.stage_slots[s]]['shape'][2]})"
                for s in ("p2_pre", "p2", "p3", "p4", "p5")
                if s in recorder_a.stage_slots and recorder_a.stage_slots[s] in proj_a
            )
            print(f"  [{img_idx}/{len(image_paths)}] Saved {stem}: {scales_str} -> {img_out_dir}")
    finally:
        recorder_a.close()
        if recorder_b is not None:
            recorder_b.close()

    print(f"===> Done! Visualizations saved to: {out_root}")


if __name__ == "__main__":
    main()
