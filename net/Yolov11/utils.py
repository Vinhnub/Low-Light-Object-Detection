"""
Utility functions for YOLOv11 architecture (`net.Yolov11.utils`), weight initialization,
structural re-parameterization (Conv+BN fusion), anchor point generation,
bounding box conversion, and Non-Maximum Suppression (NMS).
"""

import copy
import math
import os
import sys
import time
import types
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def is_parallel(model: nn.Module) -> bool:
    """Returns True if model is of type DP or DDP."""
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def de_parallel(model: nn.Module) -> nn.Module:
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from Ultralytics.
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ):
        self.ema = copy.deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1.0 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._cached_model_id: Optional[int] = None
        self._ema_fp: List[torch.Tensor] = []
        self._mod_fp: List[torch.Tensor] = []

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters and buffers using fused multi-tensor foreach ops."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            raw_m = de_parallel(model)
            if self._cached_model_id != id(raw_m) or not self._ema_fp:
                msd = raw_m.state_dict()
                esd = self.ema.state_dict()
                self._ema_fp = [
                    v for k, v in esd.items() if v.dtype.is_floating_point and k in msd
                ]
                self._mod_fp = [
                    msd[k] for k, v in esd.items() if v.dtype.is_floating_point and k in msd
                ]
                self._cached_model_id = id(raw_m)
            if self._ema_fp:
                torch._foreach_mul_(self._ema_fp, d)
                torch._foreach_add_(self._ema_fp, self._mod_fp, alpha=1.0 - d)


def smart_optimizer(
    model: nn.Module,
    name: str = "Adam",
    lr: float = 1e-4,
    momentum: float = 0.937,
    decay: float = 5e-5,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """
    Ultralytics 3-group smart optimizer:
      - g0: Normalization weights (no weight_decay)
      - g1: Conv / Linear weights (with weight_decay)
      - g2: Biases (no weight_decay)
    """
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in de_parallel(model).modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, bn):
            g[0].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            g[1].append(v.weight)

    opt_name = name.upper()
    if opt_name == "ADAM":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))
    elif opt_name == "ADAMW":
        optimizer = torch.optim.AdamW(
            g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[1], "weight_decay": decay})
    optimizer.add_param_group({"params": g[0], "weight_decay": 0.0})
    if verbose:
        print(
            f"===> Optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups: "
            f"{len(g[0])} weight(decay=0.0), {len(g[1])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
    return optimizer


PRETRAINED_URLS: Dict[str, str] = {
    "yolo11n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
    "yolo11s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
    "yolo11m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt",
    "yolo11l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt",
    "yolo11x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt",
    "yolov11n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
    "yolov11s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
    "yolov11m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt",
    "yolov11l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt",
    "yolov11x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt",
}


def _register_ultralytics_unpickle_aliases() -> None:
    """Register minimal `ultralytics` module stubs so official `.pt` checkpoints unpickle cleanly."""
    mod_names = [
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.nn.modules",
        "ultralytics.nn.modules.block",
        "ultralytics.nn.modules.conv",
        "ultralytics.nn.modules.head",
        "ultralytics.nn.tasks",
        "ultralytics.utils",
    ]
    for mod_name in mod_names:
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = []
            sys.modules[mod_name] = mod

    from net.Yolov11 import common as y11_common
    from net.Yolov11 import yolo as y11_yolo

    block_cls_names = [
        "Attention",
        "Bottleneck",
        "C1",
        "C2",
        "C2PSA",
        "C2f",
        "C3",
        "C3Ghost",
        "C3TR",
        "C3k",
        "C3k2",
        "C3x",
        "DFL",
        "GhostBottleneck",
        "PSA",
        "PSABlock",
        "Proto",
        "SCDown",
        "SPP",
        "SPPF",
    ]
    conv_cls_names = [
        "CBAM",
        "ChannelAttention",
        "Concat",
        "Conv",
        "Conv2",
        "ConvTranspose",
        "DWConv",
        "DWConvTranspose2d",
        "Focus",
        "GhostConv",
        "LightConv",
        "SpatialAttention",
    ]
    head_cls_names = ["Classify", "Detect", "OBB", "Pose", "Segment", "v10Detect"]
    task_cls_names = [
        "BaseModel",
        "ClassificationModel",
        "DetectionModel",
        "OBBModel",
        "PoseModel",
        "SegmentationModel",
    ]

    for cls_name in block_cls_names:
        if not hasattr(sys.modules["ultralytics.nn.modules.block"], cls_name):
            obj = getattr(y11_common, cls_name, type(cls_name, (nn.Module,), {}))
            setattr(sys.modules["ultralytics.nn.modules.block"], cls_name, obj)
    for cls_name in conv_cls_names:
        if not hasattr(sys.modules["ultralytics.nn.modules.conv"], cls_name):
            obj = getattr(y11_common, cls_name, type(cls_name, (nn.Module,), {}))
            setattr(sys.modules["ultralytics.nn.modules.conv"], cls_name, obj)
    for cls_name in head_cls_names:
        if not hasattr(sys.modules["ultralytics.nn.modules.head"], cls_name):
            obj = getattr(y11_yolo, cls_name, type(cls_name, (nn.Module,), {}))
            setattr(sys.modules["ultralytics.nn.modules.head"], cls_name, obj)
    for cls_name in task_cls_names:
        if not hasattr(sys.modules["ultralytics.nn.tasks"], cls_name):
            obj = getattr(y11_yolo, cls_name, type(cls_name, (nn.Module,), {}))
            setattr(sys.modules["ultralytics.nn.tasks"], cls_name, obj)
    if not hasattr(sys.modules["ultralytics.utils"], "IterableSimpleNamespace"):
        sys.modules["ultralytics.utils"].IterableSimpleNamespace = type(
            "IterableSimpleNamespace", (types.SimpleNamespace,), {}
        )


def load_yolov11_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    model_name: str = "yolov11s",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load either an official Ultralytics YOLOv11 COCO pretrained `.pt` file
    (e.g., `yolo11n.pt`, `yolo11s.pt`) or a custom training `.pth` checkpoint
    into `model` (supporting both `DetectionModel` and modular `YOLOv11`).
    Automatically downloads official `.pt` weights into `./weights/pretrained/` if `"auto"` is passed.
    """
    if not ckpt_path or str(ckpt_path).lower() in {"none", "false", "no"}:
        return None

    m_key = model_name.lower().replace(".pt", "").replace(".pth", "")
    # Canonical filename for Ultralytics assets is yolo11*.pt
    asset_name = m_key.replace("yolov11", "yolo11")
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

    # Bidirectional mapping between sequential `model.0..23.*` and modular `backbone/neck/head.*`
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

    target_model.load_state_dict(filtered_state_dict, strict=False)
    if verbose:
        print(
            f"===> Loaded {len(filtered_state_dict)}/{len(model_dict)} weight tensors from '{ckpt_path}'"
        )
    return checkpoint_data if isinstance(checkpoint_data, dict) else None


def make_divisible(x: float, divisor: int = 8) -> int:
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights to Ultralytics YOLO defaults."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d() and BatchNorm2d() layers for faster inference."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def scale_img(
    img: torch.Tensor,
    ratio: float = 1.0,
    same_shape: bool = False,
    gs: int = 32,
) -> torch.Tensor:
    """Scales img(bs,3,y,x) by ratio constrained to gs-multiple."""
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def make_anchors(
    feats: Union[List[torch.Tensor], torch.Tensor],
    strides: Union[torch.Tensor, List[float]],
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensors from feature maps."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i in range(len(feats)):
        stride = strides[i]
        h, w = (
            feats[i].shape[2:]
            if isinstance(feats, (list, tuple))
            else (int(feats[i][0]), int(feats[i][1]))
        )
        sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset
        sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Transform distance (ltrb) to box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    reg_max: Optional[int] = None,
) -> torch.Tensor:
    """Transform bbox (xyxy) to distance (ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)
    return dist


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes in (x1, y1, x2, y2) format."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Clip boxes (xyxy) to image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> torch.Tensor:
    """Rescale boxes (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(
    prediction: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: Tuple = (),
    max_det: int = 300,
    nm: int = 0,
    nc: int = 0,
    end2end: bool = False,
) -> List[torch.Tensor]:
    """
    Unified Non-Maximum Suppression supporting:
      1. End-to-End predictions `(B, max_det, 6 + nm)` [x1, y1, x2, y2, score, cls, ...] (YOLOv26)
      2. Anchor-Free predictions `(B, 4 + nc + nm, N)` (YOLOv8 / YOLOv11)
      3. Anchor-Based predictions `(B, N, 5 + nc + nm)` (YOLOv5)

    Returns:
        list of detections, on (n, 6 + nm) tensor per image [xyxy, conf, cls, ...]
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if classes is not None:
        classes_t = torch.tensor(classes, device=prediction.device)
    else:
        classes_t = None

    # 1. End-to-end NMS-free predictions: (B, K, 6 + nm)
    is_e2e = end2end or (
        prediction.ndim == 3 and prediction.shape[-1] == 6 + nm and prediction.shape[1] <= 1000
    )
    if is_e2e:
        output = []
        for pred in prediction:
            mask = pred[:, 4] > conf_thres
            if classes_t is not None:
                mask &= (pred[:, 5:6] == classes_t).any(1)
            idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
            output.append(pred[idx].float())
        return output

    # Detect anchor-free (B, 4 + nc + nm, N) vs anchor-based (B, N, 5 + nc + nm)
    is_anchor_free = prediction.ndim == 3 and prediction.shape[1] < prediction.shape[2]

    bs = prediction.shape[0]
    max_wh = 7680
    max_nms = 30000
    time_limit = 2.0 + 0.05 * bs
    t = time.time()

    if is_anchor_free:
        nc = nc or (prediction.shape[1] - 4 - nm)
        mi = 4 + nc
        xc = prediction[:, 4:mi].amax(1) > conf_thres
        multi_label &= nc > 1
        prediction = prediction.transpose(-1, -2)  # (B, N, 4 + nc + nm)
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            box, cls_scores, mask = x.split((4, nc, nm), 1)
            box = xywh2xyxy(box)

            if multi_label:
                i, j = torch.where(cls_scores > conf_thres)
                x = torch.cat((box[i], cls_scores[i, j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = cls_scores.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes_t is not None:
                x = x[(x[:, 5:6] == classes_t).any(1)]

            n = x.shape[0]
            if not n:
                continue
            if n > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    # Anchor-based YOLOv5 format: (B, N, 5 + nc + nm)
    nc = nc or (prediction.shape[2] - nm - 5)
    xc = prediction[..., 4] > conf_thres
    multi_label &= nc > 1
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes_t is not None:
            x = x[(x[:, 5:6] == classes_t).any(1)]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break
    return output
