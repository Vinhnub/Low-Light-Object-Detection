"""
YOLOv26 Utilities (`net.Yolov26.utils`):
- Official Ultralytics `yolo26*.pt` checkpoint loader (supports both `DetectionModel` and modular `YOLOv26`)
- ModelEMA, smart_optimizer, box utilities, NMS (delegated to shared Ultralytics-compatible utilities)
"""

import os
import sys
import urllib.request
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from net.Yolov11.utils import (
    ModelEMA,
    _register_ultralytics_unpickle_aliases as _base_register_unpickle,
    bbox2dist,
    box_iou,
    clip_boxes,
    de_parallel,
    dist2bbox,
    fuse_conv_and_bn,
    initialize_weights,
    is_parallel,
    make_anchors,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    scale_img,
    smart_optimizer,
    xywh2xyxy,
    xyxy2xywh,
)

PRETRAINED_URLS: Dict[str, str] = {
    "yolo26n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "yolo26m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "yolo26l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "yolo26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    "yolov26n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolov26s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "yolov26m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "yolov26l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "yolov26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
}


def _register_ultralytics_unpickle_aliases() -> None:
    """Register minimal `ultralytics` module stubs so official `yolo26*.pt` checkpoints unpickle."""
    _base_register_unpickle()
    from net.Yolov26 import common as y26_common

    if hasattr(sys.modules.get("ultralytics.nn.modules.block"), "SPPF"):
        # Keep SPPF attribute present (no-op if already set by Yolov11)
        pass
    else:
        setattr(sys.modules["ultralytics.nn.modules.block"], "SPPF", y26_common.SPPF)


def load_yolov26_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    model_name: str = "yolov26s",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load either an official Ultralytics YOLOv26 COCO pretrained `.pt` file
    (e.g., `yolo26n.pt`, `yolo26s.pt`) or a custom training `.pth` checkpoint
    into `model` (supporting both `DetectionModel` and modular `YOLOv26`).
    Automatically downloads official `.pt` weights into `./weights/pretrained/` if `"auto"` is passed.
    """
    if not ckpt_path or str(ckpt_path).lower() in {"none", "false", "no"}:
        return None

    m_key = model_name.lower().replace(".pt", "").replace(".pth", "")
    asset_name = m_key.replace("yolov26", "yolo26")
    if str(ckpt_path).lower() == "auto":
        os.makedirs("./weights/pretrained", exist_ok=True)
        ckpt_path = f"./weights/pretrained/{asset_name}.pt"

    if not os.path.exists(ckpt_path) and m_key in PRETRAINED_URLS:
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        url = PRETRAINED_URLS[m_key]
        if verbose:
            print(
                f"===> Downloading official COCO pretrained weights {asset_name}.pt from {url} ..."
            )
        urllib.request.urlretrieve(url, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    _register_ultralytics_unpickle_aliases()

    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state_dict = checkpoint_data
    if isinstance(checkpoint_data, dict):
        if "state_dict" in checkpoint_data:
            raw_state_dict = checkpoint_data["state_dict"]
        elif "ema" in checkpoint_data and checkpoint_data["ema"] is not None:
            ema_obj = checkpoint_data["ema"]
            raw_state_dict = (
                ema_obj.float().state_dict()
                if hasattr(ema_obj, "state_dict")
                else ema_obj
            )
        elif "model" in checkpoint_data and checkpoint_data["model"] is not None:
            m_obj = checkpoint_data["model"]
            raw_state_dict = (
                m_obj.float().state_dict()
                if hasattr(m_obj, "state_dict")
                else m_obj
            )

    seq_to_mod_prefix = {
        "model.0.": "backbone.stem.",
        "model.1.": "backbone.stage1.0.",
        "model.2.": "backbone.stage1.1.",
        "model.3.": "backbone.stage2.0.",
        "model.4.": "backbone.stage2.1.",
        "model.5.": "backbone.stage3.0.",
        "model.6.": "backbone.stage3.1.",
        "model.7.": "backbone.stage4.0.",
        "model.8.": "backbone.stage4.1.",
        "model.9.": "backbone.stage4.2.",
        "model.10.": "backbone.stage4.3.",
        "model.13.": "neck.c3k2_p4.",
        "model.16.": "neck.c3k2_p3.",
        "model.17.": "neck.down_p3.",
        "model.19.": "neck.c3k2_n4.",
        "model.20.": "neck.down_p4.",
        "model.22.": "neck.c3k2_n5.",
        "model.23.": "head.",
    }
    mod_to_seq_prefix = {v: k for k, v in seq_to_mod_prefix.items()}

    target_model = de_parallel(model)
    model_dict = target_model.state_dict()
    filtered_state_dict = {}
    for k, v in raw_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            v = v.float()
        if name not in model_dict:
            mapped = False
            for seq_p, mod_p in seq_to_mod_prefix.items():
                if name.startswith(seq_p):
                    cand = mod_p + name[len(seq_p) :]
                    if cand in model_dict:
                        name = cand
                        mapped = True
                        break
            if not mapped:
                for mod_p, seq_p in mod_to_seq_prefix.items():
                    if name.startswith(mod_p):
                        cand = seq_p + name[len(mod_p) :]
                        if cand in model_dict:
                            name = cand
                            break
        if name in model_dict and model_dict[name].shape == v.shape:
            filtered_state_dict[name] = v

    missing_keys, unexpected_keys = target_model.load_state_dict(
        filtered_state_dict, strict=False
    )
    if verbose:
        print(
            f"===> Loaded checkpoint '{ckpt_path}' | "
            f"Transferred {len(filtered_state_dict)}/{len(model_dict)} tensors "
            f"(missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})."
        )
    return checkpoint_data if isinstance(checkpoint_data, dict) else {"state_dict": raw_state_dict}


__all__ = [
    "PRETRAINED_URLS",
    "is_parallel",
    "de_parallel",
    "ModelEMA",
    "smart_optimizer",
    "load_yolov26_checkpoint",
    "make_divisible",
    "initialize_weights",
    "fuse_conv_and_bn",
    "scale_img",
    "make_anchors",
    "dist2bbox",
    "bbox2dist",
    "xywh2xyxy",
    "xyxy2xywh",
    "box_iou",
    "clip_boxes",
    "scale_boxes",
    "non_max_suppression",
]
