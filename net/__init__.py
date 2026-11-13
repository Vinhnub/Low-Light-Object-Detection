"""
Unified YOLO Model Registry and Factory (`net`).
Supports:
  - YOLOv5  (`yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`)
  - YOLOv11 (`yolov11n`/`yolo11n`, `yolov11s`/`yolo11s`, `m`, `l`, `x`)
  - YOLOv26 (`yolov26n`/`yolo26n`, `yolov26s`/`yolo26s`, `m`, `l`, `x`)
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch.nn as nn

from net import Yolov5, Yolov11, Yolov26
from net.Yolov11.utils import (
    ModelEMA,
    de_parallel,
    non_max_suppression,
    smart_optimizer,
)

MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    # YOLOv5
    "yolov5n": Yolov5.yolov5n,
    "yolov5s": Yolov5.yolov5s,
    "yolov5m": Yolov5.yolov5m,
    "yolov5l": Yolov5.yolov5l,
    "yolov5x": Yolov5.yolov5x,
    # YOLOv11 (accept both yolov11* and official yolo11*)
    "yolov11n": Yolov11.yolov11n,
    "yolov11s": Yolov11.yolov11s,
    "yolov11m": Yolov11.yolov11m,
    "yolov11l": Yolov11.yolov11l,
    "yolov11x": Yolov11.yolov11x,
    "yolo11n": Yolov11.yolo11n,
    "yolo11s": Yolov11.yolo11s,
    "yolo11m": Yolov11.yolo11m,
    "yolo11l": Yolov11.yolo11l,
    "yolo11x": Yolov11.yolo11x,
    # YOLOv26 (accept both yolov26* and official yolo26*)
    "yolov26n": Yolov26.yolov26n,
    "yolov26s": Yolov26.yolov26s,
    "yolov26m": Yolov26.yolov26m,
    "yolov26l": Yolov26.yolov26l,
    "yolov26x": Yolov26.yolov26x,
    "yolo26n": Yolov26.yolo26n,
    "yolo26s": Yolov26.yolo26s,
    "yolo26m": Yolov26.yolo26m,
    "yolo26l": Yolov26.yolo26l,
    "yolo26x": Yolov26.yolo26x,
}

SUPPORTED_MODELS = (
    "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
    "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov26n", "yolov26s", "yolov26m", "yolov26l", "yolov26x",
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
)


def normalize_model_name(model_name: str) -> str:
    """Normalize a model identifier (strip `.pt`/`.yaml`/`.yml` suffixes, lowercase)."""
    key = Path(str(model_name)).stem.lower()
    if key in MODEL_BUILDERS:
        return key
    for prefix in ("yolov26", "yolo26", "yolov11", "yolo11", "yolov5"):
        for scale in ("n", "s", "m", "l", "x"):
            cand = f"{prefix}{scale}"
            if cand in key:
                return cand
    return key


def get_model_family(model_name: str) -> str:
    """Return `'yolov5'`, `'yolov11'`, or `'yolov26'` for a model name or cfg path."""
    key = normalize_model_name(model_name)
    if "26" in key:
        return "yolov26"
    if "11" in key:
        return "yolov11"
    return "yolov5"


def infer_model_from_checkpoint(ckpt_meta: Any, fallback_path: Optional[str] = None) -> Optional[str]:
    """Infer model variant name from a loaded checkpoint dictionary or filename."""
    if isinstance(ckpt_meta, dict):
        if ckpt_meta.get("model_name"):
            return normalize_model_name(str(ckpt_meta["model_name"]))
        m_obj = ckpt_meta.get("model")
        yaml_cfg = getattr(m_obj, "yaml", None) if m_obj is not None else ckpt_meta.get("yaml")
        if isinstance(yaml_cfg, dict):
            yaml_file = str(yaml_cfg.get("yaml_file", ""))
            scale = str(yaml_cfg.get("scale", ""))
            if "26" in yaml_file or yaml_cfg.get("end2end") or yaml_cfg.get("reg_max") == 1:
                if scale in {"n", "s", "m", "l", "x"}:
                    return f"yolov26{scale}"
            if "11" in yaml_file and scale in {"n", "s", "m", "l", "x"}:
                return f"yolov11{scale}"
            if yaml_file:
                norm = normalize_model_name(yaml_file)
                if norm in MODEL_BUILDERS:
                    return norm
    if fallback_path:
        norm = normalize_model_name(fallback_path)
        if norm in MODEL_BUILDERS:
            return norm
    return None


def build_yolo_model(
    model_name: str = "yolov5s",
    num_classes: int = 80,
    in_channels: int = 3,
    cfg: Optional[str] = None,
    modular: bool = False,
    end2end: Optional[bool] = None,
) -> nn.Module:
    """
    Build any supported YOLOv5 / YOLOv11 / YOLOv26 model from name or YAML config.
    """
    key = normalize_model_name(model_name)
    family = get_model_family(cfg if cfg else key)

    if cfg is not None and str(cfg).lower() not in {"none", ""}:
        if family == "yolov26":
            kwargs: Dict[str, Any] = {"cfg": cfg, "ch": in_channels, "nc": num_classes}
            if end2end is not None:
                kwargs["end2end"] = end2end
            return Yolov26.DetectionModel(**kwargs)
        if family == "yolov11":
            return Yolov11.DetectionModel(cfg=cfg, ch=in_channels, nc=num_classes)
        return Yolov5.DetectionModel(cfg=cfg, ch=in_channels, nc=num_classes)

    if key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported choices: {list(SUPPORTED_MODELS)}"
        )

    builder = MODEL_BUILDERS[key]
    kwargs = {"num_classes": num_classes, "in_channels": in_channels, "modular": modular}
    if family == "yolov26" and end2end is not None:
        kwargs["end2end"] = end2end
    return builder(**kwargs)


def load_yolo_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    model_name: str = "yolov5s",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load official `.pt` or custom `.pth` checkpoint using the matching family loader."""
    family = get_model_family(model_name)
    if family == "yolov26":
        return Yolov26.load_yolov26_checkpoint(
            model, ckpt_path, model_name=model_name, verbose=verbose
        )
    if family == "yolov11":
        return Yolov11.load_yolov11_checkpoint(
            model, ckpt_path, model_name=model_name, verbose=verbose
        )
    return Yolov5.load_yolov5_checkpoint(
        model, ckpt_path, model_name=model_name, verbose=verbose
    )


__all__ = [
    "Yolov5",
    "Yolov11",
    "Yolov26",
    "MODEL_BUILDERS",
    "SUPPORTED_MODELS",
    "normalize_model_name",
    "get_model_family",
    "infer_model_from_checkpoint",
    "build_yolo_model",
    "load_yolo_checkpoint",
    "ModelEMA",
    "de_parallel",
    "smart_optimizer",
    "non_max_suppression",
]
