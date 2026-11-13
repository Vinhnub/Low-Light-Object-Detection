"""
YOLOv26 Core Architecture and Model Definitions (`net.Yolov26.yolo`).
Based on Ultralytics YOLOv26 (`ultralytics/cfg/models/26/yolo26.yaml`,
`ultralytics/nn/tasks.py`, and `ultralytics/nn/modules/head.py`).

Key differences from YOLOv11:
  - End-to-end NMS-free dual-head inference (`end2end=True`) with one-to-many & one-to-one heads
  - DFL-free box regression (`reg_max=1`, `dfl = nn.Identity()`)
  - Layer 9: `SPPF [1024, 5, 3, True]` (residual shortcut, no conv1 activation)
  - Head layers 13, 16, 19: `C3k2` always uses `c3k=True` across all scales
  - Head layer 22: `[-1, 1, C3k2, [1024, True, 0.5, True]]` (PSABlock attention inside C3k2)
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from net.Yolov11.yolo import (
    BaseModel,
    Classify,
    ClassificationModel,
    Detect,
    OBB,
    OBBModel,
    Pose,
    PoseModel,
    Segment,
    SegmentationModel,
    parse_model as _y11_parse_model,
)

from .common import C2PSA, C3k2, Conv, Conv2, DWConv, SPPF
from .utils import fuse_conv_and_bn, initialize_weights, make_divisible


# ==================================================================================================
# 1. Scaling Configurations (Ultralytics yolo26.yaml)
# ==================================================================================================

YOLOV26_SCALES: Dict[str, Tuple[float, float, int]] = {
    # variant: (depth_multiple, width_multiple, max_channels)
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def get_yolov26_cfg_dict(
    variant: str = "s",
    nc: int = 80,
    end2end: bool = True,
    reg_max: int = 1,
) -> Dict[str, Any]:
    """Return official Ultralytics YOLOv26 architecture configuration dictionary."""
    variant = variant.lower().replace("yolov26", "").replace("yolo26", "")
    if variant not in YOLOV26_SCALES:
        raise ValueError(
            f"Unsupported YOLOv26 variant '{variant}'. Choose from {list(YOLOV26_SCALES.keys())}."
        )
    return {
        "nc": nc,
        "end2end": bool(end2end),
        "reg_max": int(reg_max),
        "scale": variant,
        "scales": {k: list(v) for k, v in YOLOV26_SCALES.items()},
        "backbone": [
            # [from, repeats, module, args]
            [-1, 1, "Conv", [64, 3, 2]],               # 0-P1/2
            [-1, 1, "Conv", [128, 3, 2]],              # 1-P2/4
            [-1, 2, "C3k2", [256, False, 0.25]],       # 2
            [-1, 1, "Conv", [256, 3, 2]],              # 3-P3/8
            [-1, 2, "C3k2", [512, False, 0.25]],       # 4
            [-1, 1, "Conv", [512, 3, 2]],              # 5-P4/16
            [-1, 2, "C3k2", [512, True]],              # 6
            [-1, 1, "Conv", [1024, 3, 2]],             # 7-P5/32
            [-1, 2, "C3k2", [1024, True]],             # 8
            [-1, 1, "SPPF", [1024, 5, 3, True]],       # 9
            [-1, 2, "C2PSA", [1024]],                  # 10
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 11
            [[-1, 6], 1, "Concat", [1]],                    # 12 cat backbone P4
            [-1, 2, "C3k2", [512, True]],                   # 13
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # 14
            [[-1, 4], 1, "Concat", [1]],                    # 15 cat backbone P3
            [-1, 2, "C3k2", [256, True]],                   # 16 (P3/8-small)
            [-1, 1, "Conv", [256, 3, 2]],                   # 17
            [[-1, 13], 1, "Concat", [1]],                   # 18 cat head P4
            [-1, 2, "C3k2", [512, True]],                   # 19 (P4/16-medium)
            [-1, 1, "Conv", [512, 3, 2]],                   # 20
            [[-1, 10], 1, "Concat", [1]],                   # 21 cat head P5
            [-1, 1, "C3k2", [1024, True, 0.5, True]],       # 22 (P5/32-large, attention)
            [[16, 19, 22], 1, "Detect", ["nc"]],            # 23 Detect(P3, P4, P5)
        ],
    }


def parse_model(
    d: Dict[str, Any], ch: int = 3
) -> Tuple[nn.Sequential, List[int]]:
    """
    Parse an Ultralytics YOLOv26 model dictionary into `(nn.Sequential, save_indices)`.
    """
    d_copy = copy.deepcopy(d)
    d_copy.setdefault("end2end", True)
    d_copy.setdefault("reg_max", 1)
    return _y11_parse_model(d_copy, ch=ch)


# ==================================================================================================
# 2. Config-driven YOLOv26 DetectionModel
# ==================================================================================================

class DetectionModel(BaseModel):
    """
    YOLOv26 Detection Model (end-to-end NMS-free, DFL-free `reg_max=1` by default).
    Can be initialized from:
      - A variant name: 'yolov26n'/'yolo26n', 's', 'm', 'l', 'x'
      - A YAML file path: e.g. 'net/Yolov26/configs/yolov26s.yaml'
      - A configuration dictionary.
    """

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolov26s",
        ch: int = 3,
        nc: Optional[int] = None,
        end2end: Optional[bool] = None,
        reg_max: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = copy.deepcopy(cfg)
        elif (
            isinstance(cfg, (str, Path))
            and Path(cfg).suffix in {".yaml", ".yml"}
            and Path(cfg).exists()
        ):
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.safe_load(f)
            if "scale" not in self.yaml:
                for s in ("n", "s", "m", "l", "x"):
                    if f"yolo26{s}" in self.yaml_file.lower() or f"yolov26{s}" in self.yaml_file.lower():
                        self.yaml["scale"] = s
                        break
        else:
            cfg_str = str(cfg).lower().replace(".yaml", "").replace(".yml", "")
            variant = cfg_str.replace("yolov26", "").replace("yolo26", "")
            config_file = Path(__file__).parent / "configs" / f"yolov26{variant}.yaml"
            if config_file.exists():
                try:
                    import yaml

                    with open(config_file, encoding="utf-8") as f:
                        self.yaml = yaml.safe_load(f)
                    if "scale" not in self.yaml:
                        self.yaml["scale"] = variant
                except ImportError:
                    self.yaml = get_yolov26_cfg_dict(cfg_str, nc=nc or 80)
            else:
                self.yaml = get_yolov26_cfg_dict(cfg_str, nc=nc or 80)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml.get("nc"):
            self.yaml["nc"] = nc
        if end2end is not None:
            self.yaml["end2end"] = bool(end2end)
        else:
            self.yaml.setdefault("end2end", True)
        if reg_max is not None:
            self.yaml["reg_max"] = int(reg_max)
        else:
            self.yaml.setdefault("reg_max", 1)

        self.model, self.save = parse_model(copy.deepcopy(self.yaml), ch=ch)
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x: torch.Tensor):
                out = self.forward(x)
                if isinstance(out, tuple):
                    out = out[1]
                if "one2many" in out:
                    out = out["one2many"]
                return out["feats"]

            self.model.eval()
            m.training = True
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))]
            )
            self.stride = m.stride
            self.model.train()
            m.bias_init()
            m.end2end = bool(self.yaml.get("end2end", True))
        else:
            self.stride = torch.tensor([32.0])

        initialize_weights(self)

    @property
    def end2end(self) -> bool:
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value: bool) -> None:
        if isinstance(self.model[-1], Detect):
            self.model[-1].end2end = value


# ==================================================================================================
# 3. Explicit Modular Backbone / Neck / Head Architecture for Research
# ==================================================================================================

class YOLOv26Backbone(nn.Module):
    """
    Explicit YOLOv26 Backbone (Conv + C3k2 + shortcut SPPF + C2PSA).
    Returns multi-scale feature maps (P3/8, P4/16, P5/32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth_multiple: float = 0.50,
        width_multiple: float = 0.50,
        max_channels: int = 1024,
        c3k_override: bool = False,
    ):
        super().__init__()
        c = lambda x: make_divisible(min(x, max_channels) * width_multiple, 8)
        d = lambda n: max(round(n * depth_multiple), 1)

        self.out_channels = (c(512), c(512), c(1024))

        # Layer 0: P1/2
        self.stem = Conv(in_channels, c(64), k=3, s=2)

        # Layer 1-2: P2/4
        self.stage1 = nn.Sequential(
            Conv(c(64), c(128), k=3, s=2),
            C3k2(c(128), c(256), n=d(2), c3k=c3k_override, e=0.25),
        )

        # Layer 3-4: P3/8
        self.stage2 = nn.Sequential(
            Conv(c(256), c(256), k=3, s=2),
            C3k2(c(256), c(512), n=d(2), c3k=c3k_override, e=0.25),
        )

        # Layer 5-6: P4/16
        self.stage3 = nn.Sequential(
            Conv(c(512), c(512), k=3, s=2),
            C3k2(c(512), c(512), n=d(2), c3k=True),
        )

        # Layer 7-10: P5/32 + shortcut SPPF (act=False) + C2PSA
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), k=3, s=2),
            C3k2(c(1024), c(1024), n=d(2), c3k=True),
            SPPF(c(1024), c(1024), k=5, n=3, shortcut=True),
            C2PSA(c(1024), c(1024), n=d(2)),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)      # P1/2
        x = self.stage1(x)    # P2/4
        p3 = self.stage2(x)   # P3/8
        p4 = self.stage3(p3)  # P4/16
        p5 = self.stage4(p4)  # P5/32
        return p3, p4, p5


class YOLOv26Neck(nn.Module):
    """
    Explicit FPN + PAN Neck of YOLOv26.
    Uses `C3k2(..., c3k=True)` at all stages and adds PSABlock attention in layer 22 (`c3k2_n5`).
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (256, 256, 512),
        depth_multiple: float = 0.50,
        width_multiple: float = 0.50,
        max_channels: int = 1024,
    ):
        super().__init__()
        c3, c4, c5 = in_channels
        c = lambda x: make_divisible(min(x, max_channels) * width_multiple, 8)
        d = lambda n: max(round(n * depth_multiple), 1)

        c4_fpn = c(512)
        c3_out = c(256)
        c4_out = c(512)
        c5_out = c(1024)
        self.out_channels = (c3_out, c4_out, c5_out)

        # Top-down FPN pathway
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")        # Layer 11
        self.c3k2_p4 = C3k2(c5 + c4, c4_fpn, n=d(2), c3k=True)              # Layer 13

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")        # Layer 14
        self.c3k2_p3 = C3k2(c4_fpn + c3, c3_out, n=d(2), c3k=True)          # Layer 16 (N3/8)

        # Bottom-up PAN pathway
        self.down_p3 = Conv(c3_out, c3_out, k=3, s=2)                       # Layer 17
        self.c3k2_n4 = C3k2(c3_out + c4_fpn, c4_out, n=d(2), c3k=True)      # Layer 19 (N4/16)

        self.down_p4 = Conv(c4_out, c4_out, k=3, s=2)                       # Layer 20
        self.c3k2_n5 = C3k2(
            c4_out + c5, c5_out, n=1, c3k=True, e=0.5, attn=True
        )                                                                    # Layer 22 (N5/32, attn)

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.Tensor]:
        p3, p4, p5 = features

        f_p4 = self.c3k2_p4(torch.cat([self.upsample1(p5), p4], dim=1))
        n3 = self.c3k2_p3(torch.cat([self.upsample2(f_p4), p3], dim=1))

        n4 = self.c3k2_n4(torch.cat([self.down_p3(n3), f_p4], dim=1))
        n5 = self.c3k2_n5(torch.cat([self.down_p4(n4), p5], dim=1))

        return [n3, n4, n5]


class YOLOv26(nn.Module):
    """
    Modular Backbone + Neck + Head YOLOv26 Model (DFL-free `reg_max=1`, NMS-free `end2end=True`).
    Designed for easy customization in Low-Light Object Detection research.
    """

    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "s",
        in_channels: int = 3,
        reg_max: int = 1,
        end2end: bool = True,
        inplace: bool = True,
    ):
        super().__init__()
        variant = variant.lower().replace("yolov26", "").replace("yolo26", "")
        if variant not in YOLOV26_SCALES:
            raise ValueError(
                f"Unsupported YOLOv26 variant '{variant}'. Choose from {list(YOLOV26_SCALES.keys())}."
            )
        depth_multiple, width_multiple, max_channels = YOLOV26_SCALES[variant]
        c3k_override = variant in {"m", "l", "x"}

        self.num_classes = num_classes
        self.variant = variant
        self.names = [str(i) for i in range(num_classes)]
        self.inplace = inplace

        self.backbone = YOLOv26Backbone(
            in_channels=in_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            max_channels=max_channels,
            c3k_override=c3k_override,
        )
        self.neck = YOLOv26Neck(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            max_channels=max_channels,
        )
        self.head = Detect(
            nc=num_classes,
            reg_max=reg_max,
            end2end=end2end,
            ch=self.neck.out_channels,
        )

        # Build strides & initialize head biases
        s = 256
        self.head.inplace = inplace
        self.eval()
        self.head.training = True
        with torch.no_grad():
            dummy_out = self.forward(torch.zeros(1, in_channels, s, s))
            feats = dummy_out["one2many"]["feats"] if "one2many" in dummy_out else dummy_out["feats"]
            self.stride = torch.tensor([s / x.shape[-2] for x in feats])
        self.head.stride = self.stride
        self.train()
        self.head.bias_init()
        self.head.end2end = bool(end2end)
        initialize_weights(self)

    @property
    def end2end(self) -> bool:
        return getattr(self.head, "end2end", False)

    @end2end.setter
    def end2end(self, value: bool) -> None:
        self.head.end2end = value

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        return backbone_feats, neck_feats

    def forward(self, x: torch.Tensor):
        p3, p4, p5 = self.backbone(x)
        neck_feats = self.neck((p3, p4, p5))
        return self.head(neck_feats)

    def fuse(self):
        for m in self.modules():
            if isinstance(m, Conv2) and hasattr(m, "cv2"):
                m.fuse_convs()
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            if isinstance(m, Detect):
                m.fuse()
        return self


# ==================================================================================================
# 4. Model Factory Functions
# ==================================================================================================

def yolov26n(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv26]:
    """Build YOLOv26-Nano (~2.5M params)."""
    if modular:
        return YOLOv26(num_classes=num_classes, variant="n", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov26n", ch=in_channels, nc=num_classes, **kwargs)


def yolov26s(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv26]:
    """Build YOLOv26-Small (~9.5M params)."""
    if modular:
        return YOLOv26(num_classes=num_classes, variant="s", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov26s", ch=in_channels, nc=num_classes, **kwargs)


def yolov26m(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv26]:
    """Build YOLOv26-Medium."""
    if modular:
        return YOLOv26(num_classes=num_classes, variant="m", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov26m", ch=in_channels, nc=num_classes, **kwargs)


def yolov26l(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv26]:
    """Build YOLOv26-Large."""
    if modular:
        return YOLOv26(num_classes=num_classes, variant="l", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov26l", ch=in_channels, nc=num_classes, **kwargs)


def yolov26x(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv26]:
    """Build YOLOv26-XLarge."""
    if modular:
        return YOLOv26(num_classes=num_classes, variant="x", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov26x", ch=in_channels, nc=num_classes, **kwargs)


# Official Ultralytics naming aliases (yolo26n/s/m/l/x)
yolo26n = yolov26n
yolo26s = yolov26s
yolo26m = yolov26m
yolo26l = yolov26l
yolo26x = yolov26x

Model = DetectionModel

__all__ = [
    "YOLOV26_SCALES",
    "get_yolov26_cfg_dict",
    "Detect",
    "Segment",
    "Pose",
    "OBB",
    "Classify",
    "parse_model",
    "BaseModel",
    "DetectionModel",
    "SegmentationModel",
    "PoseModel",
    "OBBModel",
    "ClassificationModel",
    "Model",
    "YOLOv26Backbone",
    "YOLOv26Neck",
    "YOLOv26",
    "yolov26n",
    "yolov26s",
    "yolov26m",
    "yolov26l",
    "yolov26x",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
]
