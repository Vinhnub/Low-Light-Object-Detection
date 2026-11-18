"""
tools/visualize_features.py

Visualize intermediate feature maps, spatial attention heatmaps, multi-scale
Backbone/Neck/Head representations, and Detect-head objectness/class maps for a
trained YOLOv5 model, saving detailed diagnostic images into `visualization/`.

Features:
  - Auto-detects YOLOv5 variant (yolov5n/s/m/l/x) and class count from checkpoint metadata
    (supports both `DetectionModel` (`model.0..24`) and modular `YOLOv5` checkpoints).
  - Captures every stage in the network via forward hooks:
      * Backbone: Stem (P1/2), Stage1 (P2/4), Stage2 (P3/8), Stage3 (P4/16), Stage4 + SPPF (P5/32)
      * Neck (FPN + PAN): Top-down upsample/concat/C3 and bottom-up downsample/concat/C3
      * Head (Detect): Per-scale (P3/8, P4/16, P5/32) objectness & class-confidence heatmaps
      * Optional internal sub-layers inside C3 and SPPF blocks (`--sublayers`)
  - Aligns upsampled feature heatmaps pixel-accurately back to the original image
    (stripping letterbox padding before overlay).
  - Saves structured visual outputs into `visualization/<run_name>/`:
      * `00_network_overview.png`       - End-to-end architecture flow & detection overlay
      * `01_backbone_vs_neck_scales.png`- P3/P4/P5 comparison across Backbone -> Neck -> Detect
      * `02_all_layers_progression.png` - Chronological L00..L23 spatial activation progression
      * `03_detect_head_details.png`    - Anchor-level & scale-level Detect head heatmaps
      * `04_layer_activation_stats.png` - Quantitative activation / sparsity chart across layers
      * `layers/<layer_name>.png`       - Detailed per-layer Top-K channels + Mean/Max/L2 maps

Usage Examples:
  # Auto-pick a sample ExDark test image using trained epoch_220 weights
  python tools/visualize_features.py --weights weights/training/epoch_220.pth

  # Visualize a specific low-light image with brighter display-only reference underlay
  python tools/visualize_features.py --weights weights/training/epoch_220.pth ^
      --source dataset/ExDark/images/test/2015_00401.jpg --vis_gamma 0.55

  # Include internal C3 / SPPF sub-layers and 20 channels per layer
  python tools/visualize_features.py --weights weights/training/epoch_220.pth ^
      --source dataset/ExDark/images/test --num 3 --channels 20 --sublayers
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
from net.Yolov5.utils import scale_boxes  # noqa: E402
from net.Yolov11.utils import _register_ultralytics_unpickle_aliases  # noqa: E402


# Semantic annotations for standard 25-layer YOLOv5 (v6.0 / v7.0)
YOLOV5_LAYER_META: Dict[int, Tuple[str, str]] = {
    0: ("Backbone", "L00_Stem_Conv6x6_P1_2"),
    1: ("Backbone", "L01_Conv3x3_P2_4"),
    2: ("Backbone", "L02_C3_P2_4"),
    3: ("Backbone", "L03_Conv3x3_P3_8"),
    4: ("Backbone", "L04_C3_P3_8_SkipP3"),
    5: ("Backbone", "L05_Conv3x3_P4_16"),
    6: ("Backbone", "L06_C3_P4_16_SkipP4"),
    7: ("Backbone", "L07_Conv3x3_P5_32"),
    8: ("Backbone", "L08_C3_P5_32"),
    9: ("Backbone", "L09_SPPF_P5_32_SkipP5"),
    10: ("FPN_Neck", "L10_FPN_Conv1x1_P5"),
    11: ("FPN_Neck", "L11_FPN_Upsample_2x"),
    12: ("FPN_Neck", "L12_FPN_Concat_P4"),
    13: ("FPN_Neck", "L13_FPN_C3_P4"),
    14: ("FPN_Neck", "L14_FPN_Conv1x1_P4"),
    15: ("FPN_Neck", "L15_FPN_Upsample_2x"),
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

# Semantic annotations for 28-layer EdgeSAB YOLOv5
YOLOV5_EDGESAB_LAYER_META: Dict[int, Tuple[str, str]] = {
    **YOLOV5_LAYER_META,
    17: ("PAN_Neck", "L17_PAN_C3_N3_P3_8"),
    20: ("PAN_Neck", "L20_PAN_C3_N4_P4_16"),
    23: ("PAN_Neck", "L23_PAN_C3_N5_P5_32"),
    24: ("PAN_HeadIn", "L24_EdgeSAB_N3_P3_8"),
    25: ("PAN_HeadIn", "L25_EdgeSAB_N4_P4_16"),
    26: ("PAN_HeadIn", "L26_EdgeSAB_N5_P5_32"),
    27: ("Head", "L27_Detect_P3_P4_P5"),
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
        description="Visualize YOLOv5 intermediate feature maps, spatial attention, and detect head heatmaps."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/training/epoch_220.pth",
        help="Path to trained YOLOv5 checkpoint (.pth or .pt).",
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
        "--model",
        type=str,
        default="",
        help="Optional YOLOv5 variant override (yolov5n/s/m/l/x). Auto-inferred from checkpoint if empty.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional YAML config path for custom YOLOv5 architectures.",
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
        help="Use modular YOLOv5 (backbone/neck/head). Auto-detected from checkpoint keys if omitted.",
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
        help="Comma-separated layer indices/names for detailed per-layer channel sheets (e.g. '0,4,6,9,17,20,23' or 'all' or 'none').",
    )
    parser.add_argument(
        "--sublayers",
        action="store_true",
        help="Also capture internal sub-modules inside C3 and SPPF blocks.",
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
        help="Display-only gamma (<1.0 brightens the background underlay without altering model input; default: 1.0 = raw).",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=420,
        help="Display panel width in pixels for summary tiles (default: 420).",
    )
    return parser.parse_args()


# ==================================================================================================
# 1. Model Loading & Class Resolution
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


def detect_is_modular_ckpt(ckpt_meta: Any) -> bool:
    if not isinstance(ckpt_meta, dict):
        return False
    sd = ckpt_meta.get("state_dict") or ckpt_meta.get("model_state_dict")
    if isinstance(sd, dict) and sd:
        sample_keys = list(sd.keys())[:10]
        return any(k.startswith(("backbone.", "module.backbone.")) for k in sample_keys)
    return False


def load_trained_yolov5(
    weights_path: str,
    model_override: str = "",
    cfg_override: str = "",
    nc_override: Optional[int] = None,
    modular_override: Optional[bool] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, str, int, Dict[str, Any]]:
    """Build YOLOv5 architecture matching the checkpoint and load trained weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    _register_ultralytics_unpickle_aliases()
    ckpt_meta = torch.load(weights_path, map_location="cpu", weights_only=False)

    model_name = model_override.strip().lower() if model_override.strip() else (
        infer_model_from_checkpoint(ckpt_meta, fallback_path=weights_path) or "yolov5s"
    )
    if get_model_family(model_name) != "yolov5":
        raise ValueError(
            f"This tool visualizes YOLOv5 architectures, but checkpoint/model resolved to '{model_name}'."
        )

    nc = nc_override
    if nc is None and isinstance(ckpt_meta, dict):
        if ckpt_meta.get("num_classes") is not None:
            nc = int(ckpt_meta["num_classes"])
        elif hasattr(ckpt_meta.get("model"), "nc"):
            nc = int(ckpt_meta["model"].nc)
        elif isinstance(getattr(ckpt_meta.get("model"), "yaml", None), dict):
            nc = ckpt_meta["model"].yaml.get("nc")
    nc = int(nc or 12)

    is_modular = detect_is_modular_ckpt(ckpt_meta) if modular_override is None else bool(modular_override)
    cfg_path = cfg_override if (cfg_override and os.path.exists(cfg_override)) else None

    model = build_yolo_model(
        model_name=model_name,
        num_classes=nc,
        in_channels=3,
        cfg=cfg_path,
        modular=is_modular,
    )
    load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)

    # Extract class names embedded inside official checkpoints if available
    ckpt_names: List[str] = []
    if isinstance(ckpt_meta, dict) and hasattr(ckpt_meta.get("model"), "names"):
        raw_names = ckpt_meta["model"].names
        if isinstance(raw_names, dict):
            ckpt_names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
        elif isinstance(raw_names, (list, tuple)):
            ckpt_names = [str(x) for x in raw_names]

    meta_info = {
        "epoch": ckpt_meta.get("epoch", "?") if isinstance(ckpt_meta, dict) else "?",
        "is_modular": is_modular,
        "ckpt_names": ckpt_names,
    }
    model = model.to(device).eval()
    return model, model_name, nc, meta_info


# ==================================================================================================
# 2. Input Resolution & Preprocessing
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
    """
    Apply official YOLOv5 letterbox preprocessing.

    Returns:
        tensor: (1, 3, img_size, img_size) float32 RGB in [0, 1]
        unpad_wh: (new_unpad_w, new_unpad_h)
        crop_box: (top, left, bottom_y, right_x) inside the letterboxed tensor
        ratio_pad: (ratio, pad) for `scale_boxes`
    """
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
# 3. Forward Hook Feature Extractor
# ==================================================================================================

class YOLOv5FeatureRecorder:
    """Registers PyTorch forward hooks across YOLOv5 layers and optional internal sub-modules."""

    def __init__(self, model: nn.Module, capture_sublayers: bool = False):
        self.model = model
        self.capture_sublayers = capture_sublayers
        self.handles: List[Any] = []
        self.records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._register()

    def _make_hook(self, key: str, stage: str, layer_idx: int, mod_type: str, is_sub: bool):
        def hook_fn(_module, _inputs, output):
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
                # Detect in eval mode returns (cat_preds, raw_scale_list)
                raw_scales = output[1] if (isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], list)) else None
                if raw_scales is not None:
                    self.records[key] = {
                        "key": key,
                        "stage": stage,
                        "layer_idx": layer_idx,
                        "mod_type": mod_type,
                        "is_sublayer": False,
                        "detect_scales": [t.detach().float().cpu() for t in raw_scales],
                    }
        return hook_fn

    def _register(self) -> None:
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            has_edgesab = any(
                type(m).__name__ in {"EdgeSAB", "SAB"} for m in self.model.model
            )
            meta_table = YOLOV5_EDGESAB_LAYER_META if has_edgesab else YOLOV5_LAYER_META
            for idx, mod in enumerate(self.model.model):
                stage, label = meta_table.get(idx, ("Layer", f"L{idx:02d}_{type(mod).__name__}"))
                mod_type = type(mod).__name__
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(label, stage, idx, mod_type, False))
                )
                if self.capture_sublayers and mod_type in {"C3", "SPPF", "C3TR", "C3SPP", "C3Ghost", "SPP", "EdgeSAB", "SAB"}:
                    for sub_name, sub_mod in mod.named_children():
                        sub_key = f"{label}.{sub_name}"
                        self.handles.append(
                            sub_mod.register_forward_hook(
                                self._make_hook(sub_key, f"{stage}_Sub", idx, type(sub_mod).__name__, True)
                            )
                        )
        else:
            # Modular YOLOv5 (`backbone`, `neck`, `head`)
            modular_points: List[Tuple[str, str, int, nn.Module]] = []
            if hasattr(self.model, "backbone"):
                bb = self.model.backbone
                modular_points.extend([
                    ("Backbone", "L00_Stem_P1_2", 0, bb.stem),
                    ("Backbone", "L02_Stage1_P2_4", 2, bb.stage1),
                    ("Backbone", "L04_Stage2_P3_8", 4, bb.stage2),
                    ("Backbone", "L06_Stage3_P4_16", 6, bb.stage3),
                    ("Backbone", "L09_Stage4_SPPF_P5_32", 9, bb.stage4),
                ])
            has_edgesab = False
            if hasattr(self.model, "neck"):
                nk = self.model.neck
                has_edgesab = hasattr(nk, "attn_p3")
                modular_points.extend([
                    ("FPN_Neck", "L10_FPN_ReduceP5", 10, nk.reduce_p5),
                    ("FPN_Neck", "L13_FPN_C3_P4", 13, nk.c3_p4),
                    ("FPN_Neck", "L14_FPN_ReduceP4", 14, nk.reduce_p4),
                    ("PAN_Neck" if has_edgesab else "PAN_HeadIn", "L17_NeckOut_N3_P3_8", 17, nk.c3_p3),
                    ("PAN_Neck", "L18_PAN_DownP3", 18, nk.down_p3),
                    ("PAN_Neck" if has_edgesab else "PAN_HeadIn", "L20_NeckOut_N4_P4_16", 20, nk.c3_n4),
                    ("PAN_Neck", "L21_PAN_DownP4", 21, nk.down_p4),
                    ("PAN_Neck" if has_edgesab else "PAN_HeadIn", "L23_NeckOut_N5_P5_32", 23, nk.c3_n5),
                ])
                if has_edgesab:
                    modular_points.extend([
                        ("PAN_HeadIn", "L24_EdgeSAB_N3_P3_8", 24, nk.attn_p3),
                        ("PAN_HeadIn", "L25_EdgeSAB_N4_P4_16", 25, nk.attn_p4),
                        ("PAN_HeadIn", "L26_EdgeSAB_N5_P5_32", 26, nk.attn_p5),
                    ])
            if hasattr(self.model, "head"):
                det_idx = 27 if has_edgesab else 24
                modular_points.append(("Head", f"L{det_idx:02d}_Detect_P3_P4_P5", det_idx, self.model.head))

            for stage, label, idx, mod in modular_points:
                self.handles.append(
                    mod.register_forward_hook(self._make_hook(label, stage, idx, type(mod).__name__, False))
                )

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ==================================================================================================
# 4. Heatmap & Rendering Helpers
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
    """Normalize a 2D float map to uint8 [0, 255] with optional upper percentile clipping."""
    arr = np.nan_to_num(arr2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        vmin = float(arr.min())
        if clip_percentile < 100.0:
            vmax = float(np.percentile(arr, clip_percentile))
        else:
            vmax = float(arr.max())
        if vmax <= vmin + 1e-8:
            vmax = float(arr.max())
    if vmax <= vmin + 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def feature_to_spatial_maps(feat: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute aggregated spatial maps from a `(1, C, H, W)` activation tensor."""
    f = feat[0]  # (C, H, W)
    l2_energy = torch.sqrt((f ** 2).mean(dim=0) + 1e-12).numpy()
    mean_map = f.mean(dim=0).numpy()
    max_map = f.max(dim=0).values.numpy()
    return {
        "l2": l2_energy,
        "mean": mean_map,
        "max": max_map,
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
        cv2.putText(canvas, label, (x1 + 3, by2 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA)
    return canvas


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
        s_scale = min(0.40, max(0.28, w / 1050.0))
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
) -> np.ndarray:
    """Arrange equal-sized (or padded) image tiles into a crisp grid with optional top banner & colorbar."""
    if not tiles:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    nrows = (len(tiles) + ncols - 1) // ncols

    banner_h = 56 if banner_title else 0
    cbar_w = 54 if colormap_id is not None else 0
    canvas_h = banner_h + pad + nrows * (th + pad)
    canvas_w = pad + ncols * (tw + pad) + cbar_w
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    if banner_title:
        cv2.rectangle(canvas, (0, 0), (canvas_w - 1, banner_h - 1), (26, 28, 34), -1)
        cv2.line(canvas, (0, banner_h - 1), (canvas_w - 1, banner_h - 1), (80, 150, 255), 2)
        cv2.putText(canvas, banner_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 248, 255), 2, cv2.LINE_AA)
        if banner_sub:
            cv2.putText(canvas, banner_sub, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (175, 185, 200), 1, cv2.LINE_AA)

    for idx, tile in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        y0 = banner_h + pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        h_t, w_t = tile.shape[:2]
        canvas[y0 : y0 + h_t, x0 : x0 + w_t] = tile

    if colormap_id is not None:
        bar_h = max(120, canvas_h - banner_h - 2 * pad - 44)
        bar_x = canvas_w - cbar_w + 10
        bar_y = banner_h + pad + 22
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        grad_bar = cv2.applyColorMap(np.repeat(grad, 18, axis=1), colormap_id)
        canvas[bar_y : bar_y + bar_h, bar_x : bar_x + 18] = grad_bar
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 18, bar_y + bar_h), (120, 125, 135), 1)
        cv2.putText(canvas, "High", (bar_x - 4, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Low", (bar_x - 2, bar_y + bar_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA)

    return canvas


def compute_detect_heatmaps(detect_scales: List[torch.Tensor]) -> List[Dict[str, Any]]:
    """
    Decode raw Detect outputs `[P3, P4, P5]` of shape `(1, na, H, W, 5+nc)` into
    per-scale objectness & class-confidence spatial heatmaps.
    """
    results: List[Dict[str, Any]] = []
    scale_names = ["P3/8 (Small)", "P4/16 (Medium)", "P5/32 (Large)"]
    for idx, raw in enumerate(detect_scales):
        probs = raw[0].sigmoid()  # (na, H, W, 5+nc)
        obj_per_anchor = probs[..., 4].numpy()  # (na, H, W)
        cls_per_anchor = probs[..., 5:].max(dim=-1).values.numpy()  # (na, H, W)
        joint_per_anchor = obj_per_anchor * cls_per_anchor

        results.append({
            "name": scale_names[idx] if idx < len(scale_names) else f"Scale_{idx}",
            "shape": tuple(raw.shape[1:]),
            "obj_max": obj_per_anchor.max(axis=0),
            "joint_max": joint_per_anchor.max(axis=0),
            "obj_anchors": obj_per_anchor,
        })
    return results


# ==================================================================================================
# 5. Composite Visualizers
# ==================================================================================================

def resize_to_panel(img_bgr: np.ndarray, panel_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    panel_h = max(64, int(round(h * (panel_w / float(w)))))
    return cv2.resize(img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA if panel_w < w else cv2.INTER_LINEAR)


def compute_tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    arr = t.numpy()
    l2 = np.sqrt(np.mean(arr ** 2))
    sparsity = float(np.mean(np.abs(arr) < 1e-2) * 100.0)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "l2": float(l2),
        "sparsity": sparsity,
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
    epoch: Any,
    img_name: str,
    n_preds: int,
) -> None:
    """Create `00_network_overview.png` summarizing all key stages from input to Detect head."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = args.panel_width
    base_tile_img = resize_to_panel(display_bgr, pw)

    tiles: List[np.ndarray] = [
        make_labeled_tile(
            resize_to_panel(display_bgr, pw),
            "Input Image (Display)",
            f"Original {display_bgr.shape[1]}x{display_bgr.shape[0]} | vis_gamma={args.vis_gamma:.2f}",
            accent_bgr=(180, 180, 180),
        ),
        make_labeled_tile(
            resize_to_panel(pred_bgr, pw),
            f"YOLOv5 Predictions ({n_preds})",
            f"{model_name} (epoch={epoch}) | conf>={args.conf_thres:.2f}",
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

    # Pick representative milestones across Backbone -> FPN -> PAN -> Head so grid forms a full 4x4 layout
    has_edgesab = any(
        r["layer_idx"] in {24, 25, 26} and r.get("mod_type") in {"EdgeSAB", "SAB"}
        for r in records.values()
        if not r["is_sublayer"]
    )
    if gt_bgr is not None:
        key_indices = {0, 2, 4, 6, 8, 9, 13, 24, 25, 26} if has_edgesab else {0, 2, 4, 6, 8, 9, 13, 17, 20, 23}
    else:
        key_indices = {0, 2, 4, 6, 8, 9, 10, 13, 24, 25, 26} if has_edgesab else {0, 2, 4, 6, 8, 9, 10, 13, 17, 20, 23}
    stage_colors = {
        "Backbone": (255, 170, 50),
        "FPN_Neck": (255, 110, 210),
        "PAN_HeadIn": (80, 230, 140),
        "PAN_Neck": (80, 210, 230),
    }

    for rec in records.values():
        if rec["is_sublayer"] or "tensor" not in rec:
            continue
        if rec["layer_idx"] not in key_indices:
            continue
        t = rec["tensor"]
        maps = feature_to_spatial_maps(t)
        stats = compute_tensor_stats(t)
        overlay, _ = render_heatmap_overlay(
            base_tile_img,
            maps["l2"],
            imgsz,
            crop_box,
            cmap_id,
            alpha=args.alpha,
            clip_percentile=args.clip_percentile,
        )
        c, h_f, w_f = t.shape[1], t.shape[2], t.shape[3]
        stride = max(1, imgsz // max(1, h_f))
        tiles.append(
            make_labeled_tile(
                overlay,
                f"{rec['key']} ({rec['mod_type']})",
                f"[{c}x{h_f}x{w_f}] s={stride} | L2={stats['l2']:.3f} max={stats['max']:.2f}",
                accent_bgr=stage_colors.get(rec["stage"], (120, 170, 255)),
            )
        )

    for d in detect_info:
        overlay, _ = render_heatmap_overlay(
            base_tile_img,
            d["obj_max"],
            imgsz,
            crop_box,
            cmap_id,
            alpha=args.alpha,
            clip_percentile=100.0,
            fixed_range=(0.0, 1.0),
        )
        tiles.append(
            make_labeled_tile(
                overlay,
                f"Detect Obj Map: {d['name']}",
                f"max_p={d['obj_max'].max():.3f} | mean_p={d['obj_max'].mean():.3f}",
                accent_bgr=(60, 90, 255),
            )
        )

    grid = assemble_grid(
        tiles,
        ncols=4,
        banner_title=f"YOLOv5 Feature Flow Overview - {img_name}",
        banner_sub=f"Weights: {Path(args.weights).name} ({model_name}, ep={epoch}) | Colormap: {args.colormap} | L2 Energy Maps",
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
    img_name: str,
) -> None:
    """Compare Backbone vs Neck vs Detect Objectness vs Detect Class*Obj at P3/8, P4/16, P5/32."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = args.panel_width
    base_tile_img = resize_to_panel(display_bgr, pw)

    by_idx = {r["layer_idx"]: r for r in records.values() if not r["is_sublayer"] and "tensor" in r}
    has_edgesab = any(
        r["layer_idx"] in {24, 25, 26} and r.get("mod_type") in {"EdgeSAB", "SAB"}
        for r in by_idx.values()
    )
    scale_pairs = [
        ("P3/8 (Small)", 4, 24 if has_edgesab else 17, 0),
        ("P4/16 (Medium)", 6, 25 if has_edgesab else 20, 1),
        ("P5/32 (Large)", 9, 26 if has_edgesab else 23, 2),
    ]

    tiles: List[np.ndarray] = []
    for scale_label, bb_idx, nk_idx, det_idx in scale_pairs:
        for role_title, rec_idx, accent in [
            ("Backbone", bb_idx, (255, 170, 50)),
            ("Neck Fused", nk_idx, (80, 230, 140)),
        ]:
            rec = by_idx.get(rec_idx)
            if rec is not None:
                t = rec["tensor"]
                maps = feature_to_spatial_maps(t)
                stats = compute_tensor_stats(t)
                overlay, _ = render_heatmap_overlay(
                    base_tile_img, maps["l2"], imgsz, crop_box, cmap_id,
                    alpha=args.alpha, clip_percentile=args.clip_percentile,
                )
                tiles.append(
                    make_labeled_tile(
                        overlay,
                        f"{scale_label} - {role_title} (L{rec_idx:02d})",
                        f"[{t.shape[1]}x{t.shape[2]}x{t.shape[3]}] L2={stats['l2']:.3f} max={stats['max']:.2f}",
                        accent_bgr=accent,
                    )
                )
            else:
                tiles.append(make_labeled_tile(base_tile_img.copy(), f"{scale_label} - {role_title}", "N/A"))

        if det_idx < len(detect_info):
            d = detect_info[det_idx]
            obj_ov, _ = render_heatmap_overlay(
                base_tile_img, d["obj_max"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            jnt_ov, _ = render_heatmap_overlay(
                base_tile_img, d["joint_max"], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            tiles.append(
                make_labeled_tile(
                    obj_ov,
                    f"{scale_label} - Detect Objectness",
                    f"max_obj={d['obj_max'].max():.3f} | mean={d['obj_max'].mean():.3f}",
                    accent_bgr=(60, 110, 255),
                )
            )
            tiles.append(
                make_labeled_tile(
                    jnt_ov,
                    f"{scale_label} - Detect Obj*Class",
                    f"max_score={d['joint_max'].max():.3f} | mean={d['joint_max'].mean():.3f}",
                    accent_bgr=(255, 80, 180),
                )
            )

    grid = assemble_grid(
        tiles,
        ncols=4,
        banner_title=f"Multi-Scale Progression: Backbone -> Neck -> Detect Head ({img_name})",
        banner_sub="Rows: P3/8 (Small), P4/16 (Medium), P5/32 (Large) | Compare noise suppression from Backbone to Neck/Head",
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
    img_name: str,
) -> None:
    """Render every single layer L00..L23 in order so the user sees the full network progression."""
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(320, args.panel_width))
    base_tile_img = resize_to_panel(display_bgr, pw)

    stage_colors = {
        "Backbone": (255, 170, 50),
        "FPN_Neck": (255, 110, 210),
        "PAN_Neck": (80, 210, 230),
        "PAN_HeadIn": (80, 230, 140),
    }

    tiles: List[np.ndarray] = []
    for rec in records.values():
        if rec["is_sublayer"] or "tensor" not in rec:
            continue
        t = rec["tensor"]
        maps = feature_to_spatial_maps(t)
        stats = compute_tensor_stats(t)
        overlay, _ = render_heatmap_overlay(
            base_tile_img, maps["l2"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=args.clip_percentile,
        )
        c, h_f, w_f = t.shape[1], t.shape[2], t.shape[3]
        tiles.append(
            make_labeled_tile(
                overlay,
                f"{rec['key']}",
                f"{rec['mod_type']} [{c}x{h_f}x{w_f}] L2={stats['l2']:.3f} sp={stats['sparsity']:.0f}%",
                accent_bgr=stage_colors.get(rec["stage"], (150, 170, 255)),
            )
        )

    grid = assemble_grid(
        tiles,
        ncols=6,
        banner_title=f"Full YOLOv5 Layer-by-Layer Progression (L00 -> L23) - {img_name}",
        banner_sub="Orange: Backbone | Pink: FPN Top-Down | Cyan: PAN Bottom-Up | Green: PAN Outputs to Detect",
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
    """Visualize anchor-level and combined heatmaps inside the Detect head across P3/P4/P5."""
    if not detect_info:
        return
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(280, min(360, args.panel_width))
    base_tile_img = resize_to_panel(display_bgr, pw)

    tiles: List[np.ndarray] = []
    for d in detect_info:
        na = d["obj_anchors"].shape[0]
        for a in range(min(3, na)):
            ov, _ = render_heatmap_overlay(
                base_tile_img, d["obj_anchors"][a], imgsz, crop_box, cmap_id,
                alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
            )
            tiles.append(
                make_labeled_tile(
                    ov,
                    f"{d['name']} - Anchor {a}",
                    f"max_obj={d['obj_anchors'][a].max():.3f}",
                    accent_bgr=(70, 150, 255),
                )
            )
        ov_max, _ = render_heatmap_overlay(
            base_tile_img, d["obj_max"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
        )
        tiles.append(
            make_labeled_tile(
                ov_max,
                f"{d['name']} - Max Objectness",
                f"max={d['obj_max'].max():.3f} | mean={d['obj_max'].mean():.3f}",
                accent_bgr=(60, 230, 120),
            )
        )
        ov_jnt, _ = render_heatmap_overlay(
            base_tile_img, d["joint_max"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=100.0, fixed_range=(0.0, 1.0),
        )
        tiles.append(
            make_labeled_tile(
                ov_jnt,
                f"{d['name']} - Max Obj*Class",
                f"max_score={d['joint_max'].max():.3f} | mean={d['joint_max'].mean():.3f}",
                accent_bgr=(255, 80, 180),
            )
        )

    grid = assemble_grid(
        tiles,
        ncols=5,
        banner_title=f"Detect Head Anchor & Scale Breakdown - {img_name}",
        banner_sub="Rows 1-3: P3/8, P4/16, P5/32 | Cols: Anchor 0-2, Max Objectness, Max Obj*Class (Range [0, 1])",
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
        selected_indices = set()
        for token in layer_filter.split(","):
            tok = token.strip()
            if tok.isdigit():
                selected_indices.add(int(tok))

    os.makedirs(layers_dir, exist_ok=True)
    cmap_id = COLORMAP_TABLE[args.colormap]
    pw = max(240, min(320, args.panel_width))
    base_tile_img = resize_to_panel(display_bgr, pw)

    saved = 0
    for rec in records.values():
        if "tensor" not in rec:
            continue
        if selected_indices is not None and rec["layer_idx"] not in selected_indices:
            continue

        t = rec["tensor"]  # (1, C, H, W)
        f = t[0]
        c, h_f, w_f = f.shape
        maps = feature_to_spatial_maps(t)
        stats = compute_tensor_stats(t)

        # Summary spatial maps
        l2_ov, l2_heat = render_heatmap_overlay(
            base_tile_img, maps["l2"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=args.clip_percentile,
        )
        mean_ov, _ = render_heatmap_overlay(
            base_tile_img, maps["mean"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=args.clip_percentile,
        )
        max_ov, _ = render_heatmap_overlay(
            base_tile_img, maps["max"], imgsz, crop_box, cmap_id,
            alpha=args.alpha, clip_percentile=args.clip_percentile,
        )

        tiles: List[np.ndarray] = [
            make_labeled_tile(base_tile_img, "Input Display", f"Layer: {rec['key']}", accent_bgr=(180, 180, 180)),
            make_labeled_tile(l2_ov, "L2 Energy Overlay", f"L2={stats['l2']:.3f} std={stats['std']:.3f}", accent_bgr=(80, 230, 140)),
            make_labeled_tile(l2_heat, "L2 Energy Map (Raw)", f"shape=[{c}, {h_f}, {w_f}]", accent_bgr=(80, 210, 230)),
            make_labeled_tile(max_ov, "Channel Max Overlay", f"max={stats['max']:.3f} min={stats['min']:.3f}", accent_bgr=(255, 160, 60)),
            make_labeled_tile(mean_ov, "Channel Mean Overlay", f"mean={stats['mean']:.3f} sp={stats['sparsity']:.1f}%", accent_bgr=(240, 110, 210)),
        ]

        # Select top-K channels ranked by spatial energy (L2 norm per channel)
        k_ch = min(max(0, args.channels), c)
        if k_ch > 0:
            ch_energy = torch.sqrt((f.view(c, -1) ** 2).mean(dim=1))
            top_indices = torch.topk(ch_energy, k=k_ch).indices.tolist()
            for ch_idx in top_indices:
                ch_map = f[ch_idx].numpy()
                _, ch_heat = render_heatmap_overlay(
                    base_tile_img, ch_map, imgsz, crop_box, cmap_id,
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
            banner_sub=f"Stage: {rec['stage']} | Output Shape: (1, {c}, {h_f}, {w_f}) | Top {k_ch} channels by L2 activation energy",
            colormap_id=cmap_id,
        )
        cv2.imwrite(os.path.join(layers_dir, f"{safe_name}.png"), grid)
        saved += 1

    return saved


def save_activation_stats_chart(
    save_path: str,
    records: "OrderedDict[str, Dict[str, Any]]",
    img_name: str,
    model_name: str,
) -> None:
    """Plot per-layer activation statistics across L00..L23 using Matplotlib (Agg backend)."""
    layer_recs = [r for r in records.values() if not r["is_sublayer"] and "tensor" in r]
    if not layer_recs:
        return

    labels = [f"L{r['layer_idx']:02d}\n{r['mod_type']}" for r in layer_recs]
    stats_list = [compute_tensor_stats(r["tensor"]) for r in layer_recs]
    l2_vals = [s["l2"] for s in stats_list]
    mean_vals = [s["mean"] for s in stats_list]
    max_vals = [s["max"] for s in stats_list]
    sparsity_vals = [s["sparsity"] for s in stats_list]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(labels) * 0.65), 8), sharex=True)
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
        ax1.set_title(f"Layer-wise Activation Magnitude & Sparsity ({model_name} - {img_name})", color="#f0f4ff", fontsize=11, pad=10)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", facecolor="#222530", edgecolor="#404556", labelcolor="#e6eaf2", fontsize=8)

        bar_colors = []
        for r in layer_recs:
            if r["stage"] == "Backbone":
                bar_colors.append("#f5a623")
            elif r["stage"] == "FPN_Neck":
                bar_colors.append("#d96bff")
            elif r["stage"] == "PAN_HeadIn":
                bar_colors.append("#4cd98c")
            else:
                bar_colors.append("#4fc3f7")

        ax2.bar(x, sparsity_vals, color=bar_colors, alpha=0.85, width=0.65)
        ax2.set_ylabel("Near-Zero Ratio (%)", color="#d8dce6", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as exc:
        print(f"[Warn] Could not render matplotlib stats plot ({exc}); skipping stats chart.")


# ==================================================================================================
# 6. Main Entrypoint
# ==================================================================================================

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("=" * 72)
    print("YOLOv5 FEATURE VISUALIZATION TOOL")
    print("=" * 72)

    model, model_name, nc, meta_info = load_trained_yolov5(
        weights_path=args.weights,
        model_override=args.model,
        cfg_override=args.cfg,
        nc_override=args.nc,
        modular_override=args.modular,
        device=device,
    )

    if not args.classes and meta_info["ckpt_names"] and len(meta_info["ckpt_names"]) == nc:
        # Prefer dataset names over numeric placeholders '0', '1', ...
        if not all(n.isdigit() for n in meta_info["ckpt_names"]):
            class_names = meta_info["ckpt_names"]
        else:
            class_names = resolve_class_names("", nc)
    else:
        class_names = resolve_class_names(args.classes, nc)

    img_paths = collect_input_images(args.source, max_num=args.num, shuffle=args.shuffle)
    ckpt_stem = Path(args.weights).stem
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Weights    : {args.weights} ({model_name}, epoch={meta_info['epoch']}, modular={meta_info['is_modular']})")
    print(f"Device     : {device}")
    print(f"Classes    : {len(class_names)} {class_names[:6]}{'...' if len(class_names) > 6 else ''}")
    print(f"Images     : {len(img_paths)} file(s)")
    print(f"Output Dir : {out_root.resolve()}")
    print("-" * 72)

    recorder = YOLOv5FeatureRecorder(model, capture_sublayers=args.sublayers)
    stride = int(max(model.stride).item()) if hasattr(model, "stride") else 32

    try:
        for idx, img_path in enumerate(img_paths, 1):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[Warn] Could not read image: {img_path}")
                continue

            h0, w0 = img_bgr.shape[:2]
            tensor, _unpad_wh, crop_box, ratio_pad = preprocess_image(img_bgr, img_size=args.imgsz, stride=stride)
            tensor = tensor.to(device)

            recorder.clear()
            with torch.no_grad():
                out = model(tensor)
                pred_out = out[0] if isinstance(out, tuple) else out

            dets = non_max_suppression(
                pred_out.float(),
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=300,
            )[0]

            pred_np = np.zeros((0, 6), dtype=np.float32)
            if len(dets):
                det_c = dets.clone()
                det_c[:, :4] = scale_boxes(tensor.shape[2:], det_c[:, :4], (h0, w0), ratio_pad=ratio_pad)
                pred_np = det_c.cpu().numpy()

            display_bgr = apply_display_gamma(img_bgr, args.vis_gamma)
            pred_bgr = draw_detections_on_img(display_bgr, pred_np, class_names, show_conf=True)

            gt_np = load_optional_gt(img_path, (h0, w0), class_names)
            gt_bgr = draw_detections_on_img(display_bgr, gt_np, class_names, show_conf=False) if len(gt_np) else None

            detect_rec = next((r for r in recorder.records.values() if "detect_scales" in r), None)
            detect_info = compute_detect_heatmaps(detect_rec["detect_scales"]) if detect_rec else []

            img_stem = Path(img_path).stem
            run_dir = out_root / f"{model_name}_{ckpt_stem}_{img_stem}"
            run_dir.mkdir(parents=True, exist_ok=True)

            save_network_overview(
                str(run_dir / "00_network_overview.png"),
                display_bgr, pred_bgr, gt_bgr, recorder.records, detect_info,
                args.imgsz, crop_box, args, model_name, meta_info["epoch"], Path(img_path).name, len(pred_np),
            )
            save_backbone_vs_neck_scales(
                str(run_dir / "01_backbone_vs_neck_scales.png"),
                display_bgr, recorder.records, detect_info,
                args.imgsz, crop_box, args, Path(img_path).name,
            )
            save_all_layers_progression(
                str(run_dir / "02_all_layers_progression.png"),
                display_bgr, recorder.records,
                args.imgsz, crop_box, args, Path(img_path).name,
            )
            save_detect_head_details(
                str(run_dir / "03_detect_head_details.png"),
                display_bgr, detect_info,
                args.imgsz, crop_box, args, Path(img_path).name,
            )
            save_activation_stats_chart(
                str(run_dir / "04_layer_activation_stats.png"),
                recorder.records, Path(img_path).name, model_name,
            )
            n_layers_saved = save_per_layer_channel_sheets(
                str(run_dir / "layers"),
                display_bgr, recorder.records, args.imgsz, crop_box, args,
            )

            print(
                f"[{idx}/{len(img_paths)}] {Path(img_path).name} -> {len(pred_np)} dets | "
                f"5 summary sheets + {n_layers_saved} layer detail sheets saved to '{run_dir}'"
            )
    finally:
        recorder.close()

    print("=" * 72)
    print(f"[SUCCESS] All feature visualizations saved under: {out_root.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
