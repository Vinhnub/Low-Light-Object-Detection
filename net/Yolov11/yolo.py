"""
YOLOv11 Core Architecture and Model Definitions (`net.Yolov11.yolo`).
Based on Ultralytics YOLOv11 (`ultralytics/nn/tasks.py` and `ultralytics/nn/modules/head.py`).

Provides both:
  1. Official YAML/Dict-driven YOLOv11 architecture (`DetectionModel`, `Model`, `parse_model`)
     compatible with official Ultralytics `yolo11*.pt` checkpoints and layer indexing.
  2. Explicit modular Backbone-Neck-Head YOLOv11 architecture (`YOLOv11Backbone`, `YOLOv11Neck`, `YOLOv11`)
     ideal for Low-Light Object Detection research (feature-level enhancement, multi-scale fusion, etc.).
  3. Factory builders: `yolov11n`, `yolov11s`, `yolov11m`, `yolov11l`, `yolov11x`
     (plus aliases `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`).
"""

import ast
import contextlib
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .common import (
    C1,
    C2,
    C2PSA,
    C3,
    CBAM,
    DFL,
    PSA,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3SPP,
    C3TR,
    C3k,
    C3k2,
    C3x,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    LightConv,
    Proto,
    PSABlock,
    SCDown,
)
from .utils import (
    dist2bbox,
    fuse_conv_and_bn,
    initialize_weights,
    make_anchors,
    make_divisible,
    scale_img,
)


# ==================================================================================================
# 1. Scaling Configurations (Ultralytics yolo11.yaml)
# ==================================================================================================

YOLOV11_SCALES: Dict[str, Tuple[float, float, int]] = {
    # variant: (depth_multiple, width_multiple, max_channels)
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def get_yolov11_cfg_dict(
    variant: str = "s",
    nc: int = 80,
) -> Dict[str, Any]:
    """Return official Ultralytics YOLOv11 architecture configuration dictionary."""
    variant = variant.lower().replace("yolov11", "").replace("yolo11", "")
    if variant not in YOLOV11_SCALES:
        raise ValueError(
            f"Unsupported YOLOv11 variant '{variant}'. Choose from {list(YOLOV11_SCALES.keys())}."
        )
    return {
        "nc": nc,
        "scale": variant,
        "scales": {k: list(v) for k, v in YOLOV11_SCALES.items()},
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
            [-1, 1, "SPPF", [1024, 5]],                # 9
            [-1, 2, "C2PSA", [1024]],                  # 10
        ],
        "head": [
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
            [[16, 19, 22], 1, "Detect", ["nc"]],           # 23 Detect(P3, P4, P5)
        ],
    }


# ==================================================================================================
# 2. YOLOv11 Output Heads (Detect, Segment, Pose, OBB, Classify)
# ==================================================================================================

class Detect(nn.Module):
    """
    Ultralytics YOLO Detect head (anchor-free decoupled box/class heads with DFL).
    Supports both standard one-to-many detection (YOLOv11) and optional end-to-end dual-head mode.
    """

    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    @staticmethod
    def _grouped_topk(x: torch.Tensor, k: int, groups: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[1]
        while groups > 1 and (n % groups or n // groups < k):
            groups //= 2
        if groups == 1:
            return x.topk(k, dim=1)
        size = n // groups
        values, index = x.reshape(x.shape[0], groups, size).topk(k, dim=-1)
        values, winners = values.flatten(1).topk(k, dim=1)
        return values, winners // k * size + index.flatten(1).gather(1, winners)

    def _gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return x.gather(1, index if x.ndim == 2 else index[..., None].expand(-1, -1, x.shape[-1]))

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Union[List[int], Tuple[int, ...]] = (),
    ):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
                for x in ch
            )
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self._end2end = bool(end2end)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self) -> Dict[str, nn.ModuleList]:
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self) -> Dict[str, nn.ModuleList]:
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    @property
    def end2end(self) -> bool:
        return getattr(self, "one2one_cv2", None) is not None and (
            getattr(self, "_end2end", False) or self.cv2 is None
        )

    @end2end.setter
    def end2end(self, value: bool) -> None:
        self._end2end = value

    def forward_head(
        self,
        x: List[torch.Tensor],
        box_head: Optional[nn.ModuleList] = None,
        cls_head: Optional[nn.ModuleList] = None,
    ) -> Dict[str, Any]:
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[Dict[str, Any], torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        preds = self.forward_head(x, **self.one2many)
        if getattr(self, "one2one_cv2", None) is not None:
            x_detach = [xi.detach() for xi in x] if self.training else x
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(
            preds["one2one" if self.end2end else "one2many"]
            if "one2one" in preds
            else preds
        )
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: Dict[str, Any]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: Dict[str, Any]) -> torch.Tensor:
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
            )
            self.shape = shape
        return (
            self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), x.get("angle"))
            * self.strides
        )

    def bias_init(self) -> None:
        """Initialize Detect() biases (requires stride availability)."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if getattr(self, "one2one_cv2", None) is not None:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def decode_bboxes(
        self,
        bboxes: torch.Tensor,
        anchors: torch.Tensor,
        angle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=not self.end2end and not self.xyxy, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores, *extra = preds.split(
            [s for s in (4, self.nc, preds.shape[-1] - 4 - self.nc) if s], dim=-1
        )
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        return torch.cat(
            [self._gather(boxes, idx), scores, conf, *(self._gather(e, idx) for e in extra)],
            dim=-1,
        )

    def get_topk_index(
        self, scores: torch.Tensor, max_det: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, nc = scores.shape[1:]
        k = min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1)
            scores, index = self._grouped_topk(scores, k, 1)
            return scores[..., None], self._gather(labels[..., None].float(), index), index
        groups = 8 if self.export and self.format == "engine" and not self.dynamic else 1
        ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1]
        scores = self._gather(scores, ori_index)
        scores, index = self._grouped_topk(scores.flatten(1), k, groups)
        return (
            scores[..., None],
            (index % nc)[..., None].float(),
            self._gather(ori_index, index // nc),
        )

    def fuse(self) -> None:
        end2end = self.end2end
        for name in tuple(self._modules):
            if name.startswith("one2one_"):
                setattr(self, name[8:] if end2end else name, None)


class Segment(Detect):
    """YOLOv11 Segment head for instance segmentation models."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Union[List[int], Tuple[int, ...]] = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))
            for x in ch
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return {"box_head": self.cv2, "cls_head": self.cv3, "mask_head": self.cv4}

    @property
    def one2one(self):
        return {
            "box_head": self.one2one_cv2,
            "cls_head": self.one2one_cv3,
            "mask_head": self.one2one_cv4,
        }

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        if box_head is None or cls_head is None or mask_head is None:
            return {}
        preds = super().forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        preds["mask_coeff"] = torch.cat(
            [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], dim=-1
        )
        return preds

    def forward(self, x: List[torch.Tensor]):
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach() if self.training else proto
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: Dict[str, Any]) -> torch.Tensor:
        return torch.cat([super()._inference(x), x["mask_coeff"]], dim=1)


class Pose(Detect):
    """YOLOv11 Pose head for keypoint estimation."""

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: Tuple[int, int] = (17, 3),
        reg_max: int = 16,
        end2end: bool = False,
        ch: Union[List[int], Tuple[int, ...]] = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return {"box_head": self.cv2, "cls_head": self.cv3, "pose_head": self.cv4}

    @property
    def one2one(self):
        return {
            "box_head": self.one2one_cv2,
            "cls_head": self.one2one_cv3,
            "pose_head": self.one2one_cv4,
        }

    def forward_head(self, x, box_head=None, cls_head=None, pose_head=None):
        if box_head is None or cls_head is None or pose_head is None:
            return {}
        preds = super().forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        preds["kpts"] = torch.cat(
            [pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], dim=-1
        )
        return preds

    def _inference(self, x: Dict[str, Any]) -> torch.Tensor:
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB(Detect):
    """YOLOv11 Oriented Bounding Box (OBB) head."""

    def __init__(
        self,
        nc: int = 80,
        ne: int = 1,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Union[List[int], Tuple[int, ...]] = (),
    ):
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1))
            for x in ch
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return {"box_head": self.cv2, "cls_head": self.cv3, "angle_head": self.cv4}

    @property
    def one2one(self):
        return {
            "box_head": self.one2one_cv2,
            "cls_head": self.one2one_cv3,
            "angle_head": self.one2one_cv4,
        }

    def forward_head(self, x, box_head=None, cls_head=None, angle_head=None):
        if box_head is None or cls_head is None or angle_head is None:
            return {}
        preds = super().forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        angle = torch.cat(
            [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], dim=-1
        )
        preds["angle"] = (angle.sigmoid() - 0.25) * math.pi
        return preds

    def _inference(self, x: Dict[str, Any]) -> torch.Tensor:
        return torch.cat([super()._inference(x), x["angle"]], dim=1)


class Classify(nn.Module):
    """YOLOv11 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
    ):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


# ==================================================================================================
# 3. YAML / Dict Model Parser & Official YOLOv11 Model Classes
# ==================================================================================================

def parse_model(
    d: Dict[str, Any], ch: int = 3
) -> Tuple[nn.Sequential, List[int]]:
    """
    Parse an Ultralytics YOLOv11 model.yaml dictionary into a PyTorch nn.Sequential module
    and a list of saved layer indices for skip connections.
    """
    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (
        d.get(x) for x in ("nc", "activation", "scales", "end2end")
    )
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    default_act = Conv.default_act
    if act:
        Conv.default_act = eval(act)

    ch_list = [ch] if isinstance(ch, int) else list(ch)
    layers, save, c2 = [], [], ch_list[-1]

    base_modules = frozenset(
        {
            Classify,
            Conv,
            Conv2,
            LightConv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2PSA,
            PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            SCDown,
            CBAM,
        }
    )
    repeat_modules = frozenset(
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            C2PSA,
        }
    )

    module_scope = {
        "Conv": Conv,
        "Conv2": Conv2,
        "LightConv": LightConv,
        "DWConv": DWConv,
        "DWConvTranspose2d": DWConvTranspose2d,
        "ConvTranspose": ConvTranspose,
        "Bottleneck": Bottleneck,
        "BottleneckCSP": BottleneckCSP,
        "C1": C1,
        "C2": C2,
        "C2f": C2f,
        "C3": C3,
        "C3x": C3x,
        "C3TR": C3TR,
        "C3SPP": C3SPP,
        "C3Ghost": C3Ghost,
        "C3k": C3k,
        "C3k2": C3k2,
        "SPP": SPP,
        "SPPF": SPPF,
        "GhostConv": GhostConv,
        "GhostBottleneck": GhostBottleneck,
        "Focus": Focus,
        "PSA": PSA,
        "PSABlock": PSABlock,
        "C2PSA": C2PSA,
        "SCDown": SCDown,
        "CBAM": CBAM,
        "Concat": Concat,
        "Proto": Proto,
        "Detect": Detect,
        "Segment": Segment,
        "Pose": Pose,
        "OBB": OBB,
        "Classify": Classify,
        "nn": nn,
        "nc": nc,
        "reg_max": reg_max,
        "end2end": bool(end2end),
        "kpt_shape": kpt_shape,
    }

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        args = list(args)
        m = eval(m, module_scope) if isinstance(m, str) else m
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError, SyntaxError, NameError, TypeError):
                    args[j] = module_scope[a] if a in module_scope else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch_list[f], args[0]
            if m is not Classify:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in {"m", "l", "x"}:
                    args[3:4] = [True]
        elif m is nn.BatchNorm2d:
            args = [ch_list[f]]
        elif m is Concat:
            c2 = sum(ch_list[x] for x in f)
        elif m in {Detect, Segment, Pose, OBB}:
            args.extend([reg_max, bool(end2end), [ch_list[x] for x in f]])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            m.legacy = legacy
        else:
            c2 = ch_list[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n)))
            if n > 1
            else m(*args)
        )
        if m is SPPF and len(args) <= 3:
            for block in m_ if n > 1 else [m_]:
                block.cv1.act = Conv.default_act
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch_list = []
        ch_list.append(c2)

    Conv.default_act = default_act
    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    """YOLOv11 base model."""

    def forward(
        self, x: torch.Tensor, profile: bool = False, visualize: bool = False
    ):
        return self._forward_once(x, profile, visualize)

    def _forward_once(
        self, x: torch.Tensor, profile: bool = False, visualize: bool = False
    ):
        y = []
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def fuse(self):
        """Fuse Conv2d() and BatchNorm2d() layers across the model for faster inference."""
        for m in self.model.modules():
            if isinstance(m, Conv2) and hasattr(m, "cv2"):
                m.fuse_convs()
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            if isinstance(m, Detect):
                m.fuse()
        return self

    def info(self, verbose: bool = False, img_size: int = 640) -> Tuple[int, int, int, float]:
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        n_l = len(list(self.modules()))
        if verbose:
            print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20}")
            for i, (name, p) in enumerate(self.named_parameters()):
                print(
                    f"{i:>5g} {name:>40} {str(p.requires_grad):>9} {p.numel():>12g} {str(list(p.shape)):>20}"
                )
        return n_l, n_p, n_g, 0.0

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1] if hasattr(self, "model") else None
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self


class DetectionModel(BaseModel):
    """
    YOLOv11 Detection Model.
    Can be initialized from:
      - A variant name: 'yolov11n'/'yolo11n', 'yolov11s'/'yolo11s', 'm', 'l', 'x'
      - A YAML file path: e.g. 'net/Yolov11/configs/yolov11s.yaml'
      - A configuration dictionary.
    """

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolov11s",
        ch: int = 3,
        nc: Optional[int] = None,
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
                    if f"yolo11{s}" in self.yaml_file.lower() or f"yolov11{s}" in self.yaml_file.lower():
                        self.yaml["scale"] = s
                        break
        else:
            cfg_str = str(cfg).lower().replace(".yaml", "").replace(".yml", "")
            variant = cfg_str.replace("yolov11", "").replace("yolo11", "")
            config_file = Path(__file__).parent / "configs" / f"yolov11{variant}.yaml"
            if config_file.exists():
                try:
                    import yaml

                    with open(config_file, encoding="utf-8") as f:
                        self.yaml = yaml.safe_load(f)
                    if "scale" not in self.yaml:
                        self.yaml["scale"] = variant
                except ImportError:
                    self.yaml = get_yolov11_cfg_dict(cfg_str, nc=nc or 80)
            else:
                self.yaml = get_yolov11_cfg_dict(cfg_str, nc=nc or 80)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc

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

    def forward(
        self,
        x: torch.Tensor,
        augment: bool = False,
        profile: bool = False,
        visualize: bool = False,
    ):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x: torch.Tensor):
        if getattr(self, "end2end", False) or type(self.model[-1]) is not Detect:
            return self._forward_once(x)
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size, dim=1)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None

    @staticmethod
    def _descale_pred(
        p: torch.Tensor,
        flips: Optional[int],
        scale: float,
        img_size: Tuple[int, int],
        dim: int = 1,
    ) -> torch.Tensor:
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y: List[torch.Tensor]) -> List[torch.Tensor]:
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y


Model = DetectionModel


class SegmentationModel(DetectionModel):
    """YOLOv11 Segmentation Model."""

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolo11s-seg.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
    ):
        super().__init__(cfg, ch, nc)


class PoseModel(DetectionModel):
    """YOLOv11 Pose Model."""

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolo11s-pose.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
    ):
        super().__init__(cfg, ch, nc)


class OBBModel(DetectionModel):
    """YOLOv11 Oriented Bounding Box Model."""

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolo11s-obb.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
    ):
        super().__init__(cfg, ch, nc)


class ClassificationModel(BaseModel):
    """YOLOv11 Classification Model."""

    def __init__(
        self,
        cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model: Optional[DetectionModel] = None,
        nc: int = 1000,
        cutoff: int = 10,
    ):
        super().__init__()
        if model is not None:
            self._from_detection_model(model, nc, cutoff)
        else:
            self.model = None
            self.nc = nc

    def _from_detection_model(self, model: DetectionModel, nc: int = 1000, cutoff: int = 10):
        model = copy.deepcopy(model)
        model.model = model.model[:cutoff]
        m = model.model[-1]
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels
        c = Classify(ch, nc)
        c.i, c.f, c.type = m.i, m.f, "Classify"
        model.model[-1] = c
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc


# ==================================================================================================
# 4. Explicit Modular Backbone / Neck / Head Architecture for Research
# ==================================================================================================

class YOLOv11Backbone(nn.Module):
    """
    Explicit YOLOv11 Backbone (Conv + C3k2 + SPPF + C2PSA).
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

        # Layer 7-10: P5/32 + SPPF + C2PSA
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), k=3, s=2),
            C3k2(c(1024), c(1024), n=d(2), c3k=True),
            SPPF(c(1024), c(1024), k=5, n=3, shortcut=False, act=True),
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


class YOLOv11Neck(nn.Module):
    """
    Explicit FPN + PAN Neck of YOLOv11 using C3k2 blocks.
    Takes (P3, P4, P5) from YOLOv11Backbone and outputs fused [N3, N4, N5].
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (256, 256, 512),
        depth_multiple: float = 0.50,
        width_multiple: float = 0.50,
        max_channels: int = 1024,
        c3k_override: bool = False,
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
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")  # Layer 11
        self.c3k2_p4 = C3k2(c5 + c4, c4_fpn, n=d(2), c3k=c3k_override)  # Layer 13

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")  # Layer 14
        self.c3k2_p3 = C3k2(c4_fpn + c3, c3_out, n=d(2), c3k=c3k_override)  # Layer 16 (N3/8)

        # Bottom-up PAN pathway
        self.down_p3 = Conv(c3_out, c3_out, k=3, s=2)  # Layer 17
        self.c3k2_n4 = C3k2(c3_out + c4_fpn, c4_out, n=d(2), c3k=c3k_override)  # Layer 19 (N4/16)

        self.down_p4 = Conv(c4_out, c4_out, k=3, s=2)  # Layer 20
        self.c3k2_n5 = C3k2(c4_out + c5, c5_out, n=d(2), c3k=True)  # Layer 22 (N5/32)

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-down FPN
        f_p4 = self.c3k2_p4(torch.cat([self.upsample1(p5), p4], dim=1))
        n3 = self.c3k2_p3(torch.cat([self.upsample2(f_p4), p3], dim=1))

        # Bottom-up PAN
        n4 = self.c3k2_n4(torch.cat([self.down_p3(n3), f_p4], dim=1))
        n5 = self.c3k2_n5(torch.cat([self.down_p4(n4), p5], dim=1))

        return [n3, n4, n5]


class YOLOv11(nn.Module):
    """
    Modular Backbone + Neck + Head YOLOv11 Model.
    Designed for easy customization in Low-Light Object Detection (e.g., inserting
    low-light enhancement or feature alignment blocks before/inside the backbone or neck).
    """

    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "s",
        in_channels: int = 3,
        reg_max: int = 16,
        end2end: bool = False,
        inplace: bool = True,
    ):
        super().__init__()
        variant = variant.lower().replace("yolov11", "").replace("yolo11", "")
        if variant not in YOLOV11_SCALES:
            raise ValueError(
                f"Unsupported YOLOv11 variant '{variant}'. Choose from {list(YOLOV11_SCALES.keys())}."
            )
        depth_multiple, width_multiple, max_channels = YOLOV11_SCALES[variant]
        c3k_override = variant in {"m", "l", "x"}

        self.num_classes = num_classes
        self.variant = variant
        self.names = [str(i) for i in range(num_classes)]
        self.inplace = inplace

        self.backbone = YOLOv11Backbone(
            in_channels=in_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            max_channels=max_channels,
            c3k_override=c3k_override,
        )
        self.neck = YOLOv11Neck(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            max_channels=max_channels,
            c3k_override=c3k_override,
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

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.head
        m.stride = fn(m.stride)
        self.stride = m.stride
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
        return self


# ==================================================================================================
# 5. Model Factory Functions
# ==================================================================================================

def yolov11n(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv11]:
    """Build YOLOv11-Nano (~2.6M params)."""
    if modular:
        return YOLOv11(num_classes=num_classes, variant="n", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov11n", ch=in_channels, nc=num_classes, **kwargs)


def yolov11s(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv11]:
    """Build YOLOv11-Small (~9.5M params)."""
    if modular:
        return YOLOv11(num_classes=num_classes, variant="s", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov11s", ch=in_channels, nc=num_classes, **kwargs)


def yolov11m(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv11]:
    """Build YOLOv11-Medium (~20.1M params)."""
    if modular:
        return YOLOv11(num_classes=num_classes, variant="m", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov11m", ch=in_channels, nc=num_classes, **kwargs)


def yolov11l(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv11]:
    """Build YOLOv11-Large (~25.4M params)."""
    if modular:
        return YOLOv11(num_classes=num_classes, variant="l", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov11l", ch=in_channels, nc=num_classes, **kwargs)


def yolov11x(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv11]:
    """Build YOLOv11-XLarge (~57.0M params)."""
    if modular:
        return YOLOv11(num_classes=num_classes, variant="x", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov11x", ch=in_channels, nc=num_classes, **kwargs)


# Official Ultralytics naming aliases (yolo11n/s/m/l/x)
yolo11n = yolov11n
yolo11s = yolov11s
yolo11m = yolov11m
yolo11l = yolov11l
yolo11x = yolov11x
