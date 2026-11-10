"""
YOLOv5 Core Architecture and Model Definitions.
Based on Ultralytics YOLOv5 (models/yolo.py).

Provides both:
  1. Official YAML/Dict-driven YOLOv5 architecture (`DetectionModel`, `Model`, `parse_model`)
     compatible with official Ultralytics YOLOv5 checkpoints and layer indexing.
  2. Explicit modular Backbone-Neck-Head YOLOv5 architecture (`YOLOv5Backbone`, `YOLOv5Neck`, `YOLOv5`)
     ideal for Low-Light Object Detection research (feature-level enhancement, multi-scale fusion, etc.).
  3. Factory builders: `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`.
"""

import contextlib
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .common import (
    C3,
    C3Ghost,
    C3SPP,
    C3TR,
    C3x,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    Concat,
    Contract,
    Conv,
    ConvTranspose,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
    TransformerBlock,
)
from .utils import (
    check_anchor_order,
    fuse_conv_and_bn,
    initialize_weights,
    make_divisible,
    scale_img,
)


# ==================================================================================================
# 1. Default Anchors & Scaling Configurations (YOLOv5 v6.0 / v7.0 P5 640)
# ==================================================================================================

DEFAULT_ANCHORS = [
    [10, 13, 16, 30, 33, 23],      # P3/8  (small objects)
    [30, 61, 62, 45, 59, 119],     # P4/16 (medium objects)
    [116, 90, 156, 198, 373, 326], # P5/32 (large objects)
]

YOLOV5_SCALES: Dict[str, Tuple[float, float]] = {
    # variant: (depth_multiple, width_multiple)
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}


def get_yolov5_cfg_dict(
    variant: str = "s",
    nc: int = 80,
    anchors: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Return official YOLOv5 v6.0/v7.0 architecture configuration dictionary."""
    variant = variant.lower().replace("yolov5", "")
    if variant not in YOLOV5_SCALES:
        raise ValueError(
            f"Unsupported YOLOv5 variant '{variant}'. Choose from {list(YOLOV5_SCALES.keys())}."
        )
    depth_multiple, width_multiple = YOLOV5_SCALES[variant]
    return {
        "nc": nc,
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "anchors": copy.deepcopy(anchors if anchors is not None else DEFAULT_ANCHORS),
        "backbone": [
            # [from, number, module, args]
            [-1, 1, "Conv", [64, 6, 2, 2]],          # 0-P1/2
            [-1, 1, "Conv", [128, 3, 2]],            # 1-P2/4
            [-1, 3, "C3", [128]],                    # 2
            [-1, 1, "Conv", [256, 3, 2]],            # 3-P3/8
            [-1, 6, "C3", [256]],                    # 4
            [-1, 1, "Conv", [512, 3, 2]],            # 5-P4/16
            [-1, 9, "C3", [512]],                    # 6
            [-1, 1, "Conv", [1024, 3, 2]],           # 7-P5/32
            [-1, 3, "C3", [1024]],                   # 8
            [-1, 1, "SPPF", [1024, 5]],              # 9
        ],
        "head": [
            [-1, 1, "Conv", [512, 1, 1]],            # 10
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 11
            [[-1, 6], 1, "Concat", [1]],             # 12 cat backbone P4
            [-1, 3, "C3", [512, False]],             # 13
            [-1, 1, "Conv", [256, 1, 1]],            # 14
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 15
            [[-1, 4], 1, "Concat", [1]],             # 16 cat backbone P3
            [-1, 3, "C3", [256, False]],             # 17 (P3/8-small)
            [-1, 1, "Conv", [256, 3, 2]],            # 18
            [[-1, 14], 1, "Concat", [1]],            # 19 cat head P4
            [-1, 3, "C3", [512, False]],             # 20 (P4/16-medium)
            [-1, 1, "Conv", [512, 3, 2]],            # 21
            [[-1, 10], 1, "Concat", [1]],            # 22 cat head P5
            [-1, 3, "C3", [1024, False]],            # 23 (P5/32-large)
            [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],  # 24 Detect(P3, P4, P5)
        ],
    }


# ==================================================================================================
# 2. YOLOv5 Output Heads (Detect, Segment, Classify)
# ==================================================================================================

class Detect(nn.Module):
    """YOLOv5 Detect head for object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(
        self,
        nc: int = 80,
        anchors: Tuple[List[int], ...] = (),
        ch: Tuple[int, ...] = (),
        inplace: bool = True,
    ):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (tx, ty, tw, th, obj, cls...)
        self.nl = len(anchors)  # number of detection layers (3 for P3, P4, P5)
        self.na = len(anchors[0]) // 2  # number of anchors per scale (3)
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )  # shape(nl,na,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv: [B, na * no, H, W]
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return (
            x
            if self.training
            else (torch.cat(z, 1),)
            if self.export
            else (torch.cat(z, 1), x)
        )

    def _make_grid(
        self, nx: int = 20, ny: int = 20, i: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a mesh grid and anchor grid for coordinate decoding at scale i."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = (
            torch.stack((xv, yv), 2).expand(shape) - 0.5
        )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand(shape)
        )
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for instance segmentation models."""

    def __init__(
        self,
        nc: int = 80,
        anchors: Tuple[List[int], ...] = (),
        nm: int = 32,
        npr: int = 256,
        ch: Tuple[int, ...] = (),
        inplace: bool = True,
    ):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x: List[torch.Tensor]):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (
            (x, p)
            if self.training
            else (x[0], p)
            if self.export
            else (x[0], p, x[1])
        )


class Classify(nn.Module):
    """YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


# ==================================================================================================
# 3. YAML / Dict Model Parser & Official YOLOv5 Model Classes
# ==================================================================================================

def parse_model(
    d: Dict[str, Any], ch: List[int]
) -> Tuple[nn.Sequential, List[int]]:
    """
    Parse a YOLOv5 model.yaml dictionary into a PyTorch nn.Sequential module
    and a list of saved layer indices for skip connections.
    """
    anchors = d["anchors"]
    nc = d["nc"]
    gd = d["depth_multiple"]
    gw = d["width_multiple"]
    act = d.get("activation")
    if act:
        Conv.default_act = eval(act)  # redefine default activation
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    module_scope = {
        "Conv": Conv,
        "DWConv": DWConv,
        "DWConvTranspose2d": DWConvTranspose2d,
        "ConvTranspose": ConvTranspose,
        "TransformerBlock": TransformerBlock,
        "Bottleneck": Bottleneck,
        "BottleneckCSP": BottleneckCSP,
        "CrossConv": CrossConv,
        "C3": C3,
        "C3x": C3x,
        "C3TR": C3TR,
        "C3SPP": C3SPP,
        "C3Ghost": C3Ghost,
        "SPP": SPP,
        "SPPF": SPPF,
        "GhostConv": GhostConv,
        "GhostBottleneck": GhostBottleneck,
        "Focus": Focus,
        "Contract": Contract,
        "Expand": Expand,
        "Concat": Concat,
        "Proto": Proto,
        "Detect": Detect,
        "Segment": Segment,
        "Classify": Classify,
        "nn": nn,
        "nc": nc,
        "anchors": anchors,
    }

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m, module_scope) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError, TypeError):
                args[j] = eval(a, module_scope) if isinstance(a, str) else a

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            ConvTranspose,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n)))
            if n > 1
            else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(
        self, x: torch.Tensor, profile: bool = False, visualize: bool = False
    ):
        return self._forward_once(x, profile, visualize)

    def _forward_once(
        self, x: torch.Tensor, profile: bool = False, visualize: bool = False
    ):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def fuse(self):
        """Fuse Conv2d() and BatchNorm2d() layers across the model for faster inference."""
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def info(self, verbose: bool = False, img_size: int = 640) -> Tuple[int, int, int, float]:
        """Return model summary statistics: (num_layers, num_params, num_gradients, GFLOPs)."""
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        n_l = len(list(self.modules()))
        if verbose:
            print(
                f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20}"
            )
            for i, (name, p) in enumerate(self.named_parameters()):
                print(
                    f"{i:>5g} {name:>40} {str(p.requires_grad):>9} {p.numel():>12g} {str(list(p.shape)):>20}"
                )
        return n_l, n_p, n_g, 0.0

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers."""
        self = super()._apply(fn)
        m = self.model[-1] if hasattr(self, "model") else None
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """
    YOLOv5 Detection Model.
    Can be initialized from:
      - A variant name: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x' (or 'n', 's', 'm', 'l', 'x')
      - A YAML file path: e.g. 'net/Yolov5/configs/yolov5s.yaml'
      - A configuration dictionary.
    """

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolov5s",
        ch: int = 3,
        nc: Optional[int] = None,
        anchors: Optional[List[List[int]]] = None,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = copy.deepcopy(cfg)
        elif isinstance(cfg, (str, Path)) and Path(cfg).suffix in {".yaml", ".yml"} and Path(cfg).exists():
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.safe_load(f)
        else:
            # Resolve variant name or look up in configs/ directory
            cfg_str = str(cfg).lower().replace(".yaml", "").replace(".yml", "")
            config_file = Path(__file__).parent / "configs" / f"{cfg_str}.yaml"
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, encoding="utf-8") as f:
                        self.yaml = yaml.safe_load(f)
                except ImportError:
                    self.yaml = get_yolov5_cfg_dict(cfg_str, nc=nc or 80, anchors=anchors)
            else:
                self.yaml = get_yolov5_cfg_dict(cfg_str, nc=nc or 80, anchors=anchors)

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            self.yaml["anchors"] = anchors  # override yaml value

        self.model, self.save = parse_model(copy.deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = (
                lambda x: self.forward(x)[0]
                if isinstance(m, Segment)
                else self.forward(x)
            )
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
            )  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        augment: bool = False,
        profile: bool = False,
        visualize: bool = False,
    ):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x: torch.Tensor):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(
        self,
        p: torch.Tensor,
        flips: Optional[int],
        scale: float,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """De-scale predictions following augmented inference (inverse operation)."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y: List[torch.Tensor]) -> List[torch.Tensor]:
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf: Optional[torch.Tensor] = None) -> None:
        """
        Initialize biases into Detect(), cf is class frequency.
        Reference: https://arxiv.org/abs/1708.02002 section 3.3
        """
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' alias for backward compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 Segmentation Model."""

    def __init__(
        self,
        cfg: Union[str, Path, Dict[str, Any]] = "yolov5s-seg.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
        anchors: Optional[List[List[int]]] = None,
    ):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 Classification Model."""

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
        elif cfg is not None:
            self._from_yaml(cfg)
        else:
            raise ValueError("Provide either 'cfg' or 'model' to initialize ClassificationModel.")

    def _from_detection_model(self, model: DetectionModel, nc: int = 1000, cutoff: int = 10):
        model = copy.deepcopy(model)
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels
        c = Classify(ch, nc)
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg: Union[str, Path, Dict[str, Any]]):
        self.model = None


# ==================================================================================================
# 4. Explicit Modular Backbone / Neck / Head Architecture for Research
# ==================================================================================================

class YOLOv5Backbone(nn.Module):
    """
    Explicit CSPDarknet53 + SPPF Backbone of YOLOv5 (v6.0 / v7.0).
    Returns multi-scale feature maps (P3/8, P4/16, P5/32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50,
    ):
        super().__init__()
        c = lambda x: make_divisible(x * width_multiple, 8)
        d = lambda n: max(round(n * depth_multiple), 1)

        self.out_channels = (c(256), c(512), c(1024))

        # Layer 0: P1/2 (6x6 Conv, stride 2, padding 2)
        self.stem = Conv(in_channels, c(64), k=6, s=2, p=2)

        # Layer 1-2: P2/4
        self.stage1 = nn.Sequential(
            Conv(c(64), c(128), k=3, s=2),
            C3(c(128), c(128), n=d(3), shortcut=True),
        )

        # Layer 3-4: P3/8
        self.stage2 = nn.Sequential(
            Conv(c(128), c(256), k=3, s=2),
            C3(c(256), c(256), n=d(6), shortcut=True),
        )

        # Layer 5-6: P4/16
        self.stage3 = nn.Sequential(
            Conv(c(256), c(512), k=3, s=2),
            C3(c(512), c(512), n=d(9), shortcut=True),
        )

        # Layer 7-9: P5/32 + SPPF
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), k=3, s=2),
            C3(c(1024), c(1024), n=d(3), shortcut=True),
            SPPF(c(1024), c(1024), k=5),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)     # P1/2
        x = self.stage1(x)   # P2/4
        p3 = self.stage2(x)  # P3/8
        p4 = self.stage3(p3) # P4/16
        p5 = self.stage4(p4) # P5/32
        return p3, p4, p5


class YOLOv5Neck(nn.Module):
    """
    Explicit CSP-PANet (FPN + PAN) Neck of YOLOv5 (v6.0 / v7.0).
    Takes (P3, P4, P5) from YOLOv5Backbone and outputs fused (N3, N4, N5).
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (128, 256, 512),
        depth_multiple: float = 0.33,
    ):
        super().__init__()
        c3, c4, c5 = in_channels
        d = lambda n: max(round(n * depth_multiple), 1)
        self.out_channels = (c3, c4, c5)

        # Top-down FPN pathway
        self.reduce_p5 = Conv(c5, c4, k=1, s=1)             # Layer 10
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")  # Layer 11
        self.c3_p4 = C3(c4 + c4, c4, n=d(3), shortcut=False)  # Layer 13

        self.reduce_p4 = Conv(c4, c3, k=1, s=1)             # Layer 14
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")  # Layer 15
        self.c3_p3 = C3(c3 + c3, c3, n=d(3), shortcut=False)  # Layer 17 (N3/8 out)

        # Bottom-up PAN pathway
        self.down_p3 = Conv(c3, c3, k=3, s=2)               # Layer 18
        self.c3_n4 = C3(c3 + c3, c4, n=d(3), shortcut=False)  # Layer 20 (N4/16 out)

        self.down_p4 = Conv(c4, c4, k=3, s=2)               # Layer 21
        self.c3_n5 = C3(c4 + c4, c5, n=d(3), shortcut=False)  # Layer 23 (N5/32 out)

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-down FPN
        f_p5 = self.reduce_p5(p5)
        f_p4 = self.c3_p4(torch.cat([self.upsample1(f_p5), p4], dim=1))

        f_p4_reduced = self.reduce_p4(f_p4)
        n3 = self.c3_p3(torch.cat([self.upsample2(f_p4_reduced), p3], dim=1))

        # Bottom-up PAN
        n4 = self.c3_n4(torch.cat([self.down_p3(n3), f_p4_reduced], dim=1))
        n5 = self.c3_n5(torch.cat([self.down_p4(n4), f_p5], dim=1))

        return [n3, n4, n5]


class YOLOv5(nn.Module):
    """
    Modular Backbone + Neck + Head YOLOv5 Model.
    Designed for easy customization in Low-Light Object Detection (e.g., inserting
    low-light enhancement or feature alignment blocks before/inside the backbone or neck).
    """

    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "s",
        in_channels: int = 3,
        anchors: Optional[List[List[int]]] = None,
        inplace: bool = True,
    ):
        super().__init__()
        variant = variant.lower().replace("yolov5", "")
        if variant not in YOLOV5_SCALES:
            raise ValueError(
                f"Unsupported YOLOv5 variant '{variant}'. Choose from {list(YOLOV5_SCALES.keys())}."
            )
        depth_multiple, width_multiple = YOLOV5_SCALES[variant]
        anchors = copy.deepcopy(anchors if anchors is not None else DEFAULT_ANCHORS)

        self.num_classes = num_classes
        self.variant = variant
        self.names = [str(i) for i in range(num_classes)]

        self.backbone = YOLOv5Backbone(
            in_channels=in_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
        )
        self.neck = YOLOv5Neck(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth_multiple,
        )
        self.head = Detect(
            nc=num_classes,
            anchors=tuple(anchors),
            ch=self.neck.out_channels,
            inplace=inplace,
        )

        # Setup strides and normalize anchors to grid units
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.head.stride = self.stride
        check_anchor_order(self.head)
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        self._initialize_biases()
        initialize_weights(self)

    def _initialize_biases(self, cf: Optional[torch.Tensor] = None) -> None:
        m = self.head
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999))
                if cf is None
                else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor]]:
        """Extract both backbone features (P3, P4, P5) and neck features [N3, N4, N5]."""
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        return backbone_feats, neck_feats

    def forward(self, x: torch.Tensor):
        p3, p4, p5 = self.backbone(x)
        neck_feats = self.neck((p3, p4, p5))
        return self.head(neck_feats)

    def fuse(self):
        """Fuse Conv2d and BatchNorm2d layers across backbone and neck."""
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.head
        m.stride = fn(m.stride)
        self.stride = m.stride
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


# ==================================================================================================
# 5. Model Factory Functions
# ==================================================================================================

def yolov5n(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv5]:
    """Build YOLOv5-Nano (depth=0.33, width=0.25, ~1.9M params)."""
    if modular:
        return YOLOv5(num_classes=num_classes, variant="n", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov5n", ch=in_channels, nc=num_classes, **kwargs)


def yolov5s(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv5]:
    """Build YOLOv5-Small (depth=0.33, width=0.50, ~7.2M params)."""
    if modular:
        return YOLOv5(num_classes=num_classes, variant="s", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov5s", ch=in_channels, nc=num_classes, **kwargs)


def yolov5m(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv5]:
    """Build YOLOv5-Medium (depth=0.67, width=0.75, ~21.2M params)."""
    if modular:
        return YOLOv5(num_classes=num_classes, variant="m", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov5m", ch=in_channels, nc=num_classes, **kwargs)


def yolov5l(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv5]:
    """Build YOLOv5-Large (depth=1.00, width=1.00, ~46.5M params)."""
    if modular:
        return YOLOv5(num_classes=num_classes, variant="l", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov5l", ch=in_channels, nc=num_classes, **kwargs)


def yolov5x(
    num_classes: int = 80,
    in_channels: int = 3,
    modular: bool = False,
    **kwargs,
) -> Union[DetectionModel, YOLOv5]:
    """Build YOLOv5-XLarge (depth=1.33, width=1.25, ~86.7M params)."""
    if modular:
        return YOLOv5(num_classes=num_classes, variant="x", in_channels=in_channels, **kwargs)
    return DetectionModel(cfg="yolov5x", ch=in_channels, nc=num_classes, **kwargs)
